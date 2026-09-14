/**
 * Worker B — JIT tool execution (SAD §2.3).
 *
 * Receives validated tool actions from Worker A over a MessageChannel, lints
 * them, resolves their ESM dependencies, and runs them through a scoped
 * AsyncFunction constructor with a deliberately tiny capability object.
 *
 * On the isolation boundary, stated plainly: the real boundary is this worker
 * itself. It is a dedicated worker, so there is no `document`, no DOM, and no
 * handle to the page — code that escapes the shadowing below still cannot touch
 * the UI, the user's session, or another origin. The identifier shadowing and
 * the linter are defence in depth that make accidental misuse loud and
 * deliberate misuse awkward; they are not a claim of in-realm escape-proofing,
 * which is not achievable in JavaScript. Everything that genuinely matters —
 * network reach, storage reach, runtime — is mediated by the capability object
 * and cannot be obtained by reaching for a global, because those capabilities
 * simply are not present in this worker's scope.
 */

import { MESSAGE, isFetchAllowed, ALLOWED_ESM_HOSTS, ALLOWED_FETCH_HOSTS } from '../utils/schema.js';

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

/** Wall-clock budget for a single tool invocation. */
const DEFAULT_TIMEOUT_MS = 20_000;
/** Cap on a single fetched body, so a tool cannot exhaust device memory. */
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

/**
 * Globals shadowed inside every tool body. `eval` and `arguments` are absent
 * deliberately: strict mode forbids them as parameter names, so shadowing them
 * is a SyntaxError rather than a protection.
 */
const SHADOWED_GLOBALS = Object.freeze([
  'self',
  'globalThis',
  'window',
  'document',
  'parent',
  'top',
  'location',
  'navigator',
  'indexedDB',
  'localStorage',
  'sessionStorage',
  'caches',
  'fetch',
  'XMLHttpRequest',
  'WebSocket',
  'EventSource',
  'Worker',
  'SharedWorker',
  'importScripts',
  'postMessage',
  'addEventListener',
  'removeEventListener',
  'close',
  'Function',
  'require',
  'process',
  'module',
  'exports',
]);

/**
 * Static checks run before a tool is ever constructed. Each rule carries the
 * remediation text that Worker A feeds back into the self-improvement loop, so
 * a rejected tool produces an actionable correction rather than "denied".
 */
const LINT_RULES = Object.freeze([
  {
    id: 'no-global-escape',
    pattern: /\b(?:globalThis|window|document|self\s*\.|top\s*\.|parent\s*\.)/,
    message: 'Reaching for a global scope object. Use only the `ctx` capabilities.',
  },
  {
    id: 'no-dynamic-eval',
    pattern: /\b(?:eval\s*\(|new\s+Function\s*\(|Function\s*\(\s*['"`])/,
    message: 'Dynamic code evaluation is not permitted inside a tool body.',
  },
  {
    id: 'no-raw-network',
    pattern: /\b(?:XMLHttpRequest|WebSocket|EventSource|navigator\s*\.\s*sendBeacon)\b/,
    message: 'Use ctx.fetch — it is the only network path, and it is throttled and allow-listed.',
  },
  {
    id: 'no-raw-fetch',
    pattern: /(?<!ctx\s*\.\s*)\bfetch\s*\(/,
    message: 'Call ctx.fetch(...), not the bare fetch global.',
  },
  {
    id: 'no-raw-storage',
    pattern: /\b(?:indexedDB|localStorage|sessionStorage|caches)\b/,
    message: 'Use ctx.memory.search / ctx.memory.insert for persistence.',
  },
  {
    id: 'no-worker-control',
    pattern: /\b(?:importScripts|postMessage|new\s+Worker|new\s+SharedWorker|\bclose\s*\(\s*\))/,
    message: 'A tool may not spawn workers or drive the message channel.',
  },
  {
    id: 'no-prototype-pollution',
    pattern: /(?:__proto__|constructor\s*\.\s*prototype|Object\s*\.\s*setPrototypeOf)/,
    message: 'Mutating prototypes is not permitted; build and return a plain value.',
  },
  {
    id: 'no-static-import',
    pattern: /^\s*import\s+[^(]/m,
    message: 'Declare packages in "dependencies" instead; they arrive on ctx.deps.',
  },
  {
    id: 'no-busy-loop',
    pattern: /\bwhile\s*\(\s*(?:true|1)\s*\)/,
    message: 'Unbounded loops block the sandbox. Bound the loop or await ctx.signal.',
  },
]);

/** Cache of resolved ESM namespaces, keyed by absolute URL. */
const moduleCache = new Map();

/** Pending memory round-trips out to Worker A, keyed by request id. */
const pendingDbCalls = new Map();
let dbCallSeq = 0;

/** Set once a tool has overrun its budget; the worker is then disposable. */
let poisoned = false;

let inferencePort = null;

self.addEventListener('message', (event) => {
  const data = event.data;
  if (data?.type === MESSAGE.BIND_INFERENCE && data.port) {
    bindInferencePort(data.port);
  }
});

function bindInferencePort(port) {
  inferencePort = port;
  port.onmessage = (event) => {
    void handlePortMessage(event.data ?? {});
  };
  port.start?.();
  port.postMessage({ type: MESSAGE.SANDBOX_READY, allowedHosts: ALLOWED_FETCH_HOSTS });
}

async function handlePortMessage(message) {
  switch (message.type) {
    case MESSAGE.LINT_TOOL: {
      inferencePort.postMessage({
        type: MESSAGE.LINT_RESULT,
        id: message.id,
        ...lintToolSource(message.code),
      });
      return;
    }
    case MESSAGE.EXEC_TOOL: {
      const result = await executeTool(message);
      inferencePort.postMessage({ type: MESSAGE.EXEC_RESULT, id: message.id, ...result });
      return;
    }
    case MESSAGE.DB_RESULT: {
      const pending = pendingDbCalls.get(message.id);
      if (!pending) return;
      pendingDbCalls.delete(message.id);
      if (message.error) pending.reject(new Error(message.error));
      else pending.resolve(message.result);
      return;
    }
    default:
  }
}

/* ------------------------------------------------------------------ linting */

/**
 * Lint and compile-check a tool body without running it.
 *
 * Compilation through the AsyncFunction constructor is itself a check: it
 * surfaces syntax errors with a real message, which is by far the most common
 * thing a 2B model gets wrong.
 *
 * @returns {{ok: boolean, errors: {rule: string, message: string, line: number, excerpt: string}[]}}
 */
export function lintToolSource(code) {
  const errors = [];
  const source = String(code ?? '');

  if (!source.trim()) {
    return { ok: false, errors: [{ rule: 'empty', message: 'Tool body is empty.', line: 1, excerpt: '' }] };
  }

  const lines = source.split('\n');
  for (const rule of LINT_RULES) {
    lines.forEach((line, index) => {
      // Ignore matches that only occur inside a comment.
      const stripped = line.replace(/\/\/.*$/, '').replace(/\/\*[\s\S]*?\*\//g, '');
      if (rule.pattern.test(stripped)) {
        errors.push({
          rule: rule.id,
          message: rule.message,
          line: index + 1,
          excerpt: line.trim().slice(0, 160),
        });
      }
    });
  }

  try {
    // Compile only. The constructor never invokes the body.
    // eslint-disable-next-line no-new
    new AsyncFunction(...SHADOWED_GLOBALS, 'args', 'ctx', `"use strict";\n${source}`);
  } catch (error) {
    errors.push({
      rule: 'syntax',
      message: `${error.name}: ${error.message}`,
      line: 1,
      excerpt: '',
    });
  }

  return { ok: errors.length === 0, errors };
}

/* --------------------------------------------------- throttled network layer */

/**
 * Token-bucket rate limiter plus a concurrency gate.
 *
 * A self-improving agent that writes its own fetch loops is exactly the thing
 * that hammers an endpoint by accident, so the budget is enforced here rather
 * than trusted to the generated code.
 */
class FetchController {
  #timestamps = [];
  #inFlight = 0;
  #queue = [];

  constructor({ maxPerWindow = 8, windowMs = 10_000, maxConcurrent = 2, timeoutMs = 15_000 } = {}) {
    this.maxPerWindow = maxPerWindow;
    this.windowMs = windowMs;
    this.maxConcurrent = maxConcurrent;
    this.timeoutMs = timeoutMs;
  }

  async request(input, init = {}) {
    const url = typeof input === 'string' ? input : String(input?.url ?? '');
    if (!isFetchAllowed(url)) {
      throw new Error(
        `Blocked request to "${url}". Allowed hosts: ${ALLOWED_FETCH_HOSTS.join(', ')}.`,
      );
    }

    const method = String(init.method ?? 'GET').toUpperCase();
    if (!['GET', 'POST', 'HEAD'].includes(method)) {
      throw new Error(`HTTP ${method} is not permitted from the sandbox.`);
    }

    await this.#acquire();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(url, {
        method,
        headers: sanitiseHeaders(init.headers),
        body: method === 'GET' || method === 'HEAD' ? undefined : init.body,
        signal: controller.signal,
        // A tool never acts on the user's behalf against a credentialed origin.
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
        mode: 'cors',
        redirect: 'follow',
      });

      // Re-validate after redirects — a permitted host must not be usable as an
      // open redirector into an arbitrary one.
      if (response.url && !isFetchAllowed(response.url)) {
        throw new Error(`Request redirected to a disallowed host: ${response.url}`);
      }

      return await materialise(response);
    } finally {
      clearTimeout(timer);
      this.#release();
    }
  }

  async #acquire() {
    for (;;) {
      const now = Date.now();
      this.#timestamps = this.#timestamps.filter((t) => now - t < this.windowMs);
      if (this.#inFlight < this.maxConcurrent && this.#timestamps.length < this.maxPerWindow) {
        this.#timestamps.push(now);
        this.#inFlight += 1;
        return;
      }
      const waitFor =
        this.#timestamps.length >= this.maxPerWindow
          ? this.windowMs - (now - this.#timestamps[0]) + 5
          : 25;
      await new Promise((resolve) => {
        this.#queue.push(resolve);
        setTimeout(resolve, Math.max(waitFor, 10));
      });
    }
  }

  #release() {
    this.#inFlight = Math.max(0, this.#inFlight - 1);
    const next = this.#queue.shift();
    next?.();
  }
}

function sanitiseHeaders(headers) {
  const safe = new Headers();
  if (!headers) return safe;
  const entries = headers instanceof Headers ? headers.entries() : Object.entries(headers);
  for (const [key, value] of entries) {
    const name = String(key).toLowerCase();
    if (['cookie', 'authorization', 'set-cookie', 'host', 'origin', 'referer'].includes(name)) {
      continue;
    }
    safe.set(name, String(value));
  }
  return safe;
}

/**
 * Read a response fully under the byte cap and hand back an inert facade, so a
 * tool holds data rather than a live stream it could keep open.
 */
async function materialise(response) {
  const reader = response.body?.getReader();
  const chunks = [];
  let total = 0;

  if (reader) {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_RESPONSE_BYTES) {
        await reader.cancel();
        throw new Error(`Response exceeded the ${MAX_RESPONSE_BYTES} byte sandbox limit.`);
      }
      chunks.push(value);
    }
  }

  const buffer = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const text = new TextDecoder().decode(buffer);

  return Object.freeze({
    ok: response.ok,
    status: response.status,
    statusText: response.statusText,
    url: response.url,
    headers: Object.fromEntries(response.headers.entries()),
    text: () => text,
    json: () => JSON.parse(text),
    bytes: () => buffer.slice(),
  });
}

/* ------------------------------------------------- dynamic ESM dependencies */

/** Derive a usable binding name from an ESM CDN URL: esm.sh/ky@1.2.0 -> "ky". */
function dependencyKey(url) {
  const { pathname } = new URL(url);
  const segments = pathname.split('/').filter(Boolean);
  let name = segments.at(-1) ?? 'module';
  if (segments[0]?.startsWith('@') && segments.length > 1) name = `${segments[0]}/${segments[1]}`;
  return name.replace(/@[^@/]*$/, '').replace(/\.(m?js|ts)$/, '') || 'module';
}

async function loadDependencies(urls, log) {
  const deps = Object.create(null);
  for (const url of urls ?? []) {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' || !ALLOWED_ESM_HOSTS.includes(parsed.hostname)) {
      throw new Error(`Dependency "${url}" is not on an allow-listed ESM CDN.`);
    }
    const href = parsed.toString();
    if (!moduleCache.has(href)) {
      log?.(`resolving ${href}`);
      // @vite-ignore keeps this a genuine runtime import rather than something
      // the bundler tries to resolve at build time.
      moduleCache.set(href, import(/* @vite-ignore */ href));
    }
    try {
      deps[dependencyKey(href)] = await moduleCache.get(href);
    } catch (error) {
      moduleCache.delete(href);
      throw new Error(`Failed to import "${href}": ${error.message}`);
    }
  }
  return deps;
}

/* ------------------------------------------------------ capability context */

function makeContext({ toolName, deps, controller, signal }) {
  const log = (...parts) => {
    inferencePort?.postMessage({
      type: MESSAGE.SANDBOX_LOG,
      toolName,
      message: parts.map(stringify).join(' ').slice(0, 500),
    });
  };

  const memory = Object.freeze({
    search: (term, limit = 5) => callInference('search', { term, limit }),
    insert: (text, meta = {}) => callInference('insert', { text, meta }),
  });

  return Object.freeze({
    toolName,
    deps: Object.freeze(deps),
    fetch: (input, init) => controller.request(input, init),
    memory,
    log,
    signal,
    env: Object.freeze({
      now: () => Date.now(),
      random: () => crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32,
      locale: 'en',
    }),
  });
}

function callInference(op, payload) {
  if (!inferencePort) return Promise.reject(new Error('Sandbox is not bound to the memory index.'));
  const id = `db-${dbCallSeq++}`;
  return new Promise((resolve, reject) => {
    pendingDbCalls.set(id, { resolve, reject });
    inferencePort.postMessage({ type: MESSAGE.DB_QUERY, id, op, payload });
    setTimeout(() => {
      if (!pendingDbCalls.has(id)) return;
      pendingDbCalls.delete(id);
      reject(new Error(`Memory ${op} timed out.`));
    }, 8000);
  });
}

function stringify(value) {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/* -------------------------------------------------------------- execution */

async function executeTool(message) {
  const { toolName, code, dependencies = [], args = {}, specs = [], timeoutMs = DEFAULT_TIMEOUT_MS } =
    message;

  if (poisoned) {
    return {
      ok: false,
      phase: 'sandbox',
      error: 'Sandbox is poisoned by a previous overrun and must be respawned.',
      needsRespawn: true,
    };
  }

  const lint = lintToolSource(code);
  if (!lint.ok) {
    return { ok: false, phase: 'lint', error: formatLintErrors(lint.errors), errors: lint.errors };
  }

  const controller = new FetchController();
  const abort = new AbortController();
  const started = performance.now();

  let deps;
  try {
    deps = await loadDependencies(dependencies, (msg) =>
      inferencePort?.postMessage({ type: MESSAGE.SANDBOX_LOG, toolName, message: msg }),
    );
  } catch (error) {
    return { ok: false, phase: 'dependencies', error: error.message };
  }

  const ctx = makeContext({ toolName, deps, controller, signal: abort.signal });
  const shadowValues = SHADOWED_GLOBALS.map(() => undefined);

  let run;
  try {
    run = new AsyncFunction(...SHADOWED_GLOBALS, 'args', 'ctx', `"use strict";\n${code}`);
  } catch (error) {
    return { ok: false, phase: 'compile', error: `${error.name}: ${error.message}` };
  }

  const invoke = (callArgs) =>
    withTimeout(
      Promise.resolve().then(() => run(...shadowValues, callArgs, ctx)),
      timeoutMs,
      abort,
    );

  let result;
  try {
    result = await invoke(args);
  } catch (error) {
    if (error?.name === 'SandboxTimeout') {
      poisoned = true;
      return {
        ok: false,
        phase: 'timeout',
        error: `Tool "${toolName}" exceeded its ${timeoutMs}ms budget.`,
        needsRespawn: true,
      };
    }
    return {
      ok: false,
      phase: 'runtime',
      error: `${error?.name ?? 'Error'}: ${error?.message ?? String(error)}`,
      stack: truncate(error?.stack, 800),
    };
  }

  const specReport = await runSpecs(specs, invoke);
  return {
    ok: specReport.ok,
    phase: specReport.ok ? 'done' : 'specs',
    result: serialisable(result),
    specs: specReport.cases,
    error: specReport.ok ? undefined : formatSpecFailures(specReport.cases),
    durationMs: Math.round(performance.now() - started),
  };
}

/**
 * Run the declared behaviour specs. A tool that produces a value but fails its
 * own spec is a failure, and the report feeds the self-improvement loop.
 */
async function runSpecs(specs, invoke) {
  const cases = [];
  for (const spec of specs ?? []) {
    const name = String(spec?.name ?? 'unnamed');
    try {
      const actual = await invoke(spec?.args ?? {});
      const passed =
        spec?.expect === undefined
          ? true
          : JSON.stringify(serialisable(actual)) === JSON.stringify(spec.expect);
      cases.push({ name, passed, actual: serialisable(actual), expected: spec?.expect });
    } catch (error) {
      cases.push({ name, passed: false, error: error?.message ?? String(error) });
    }
  }
  return { ok: cases.every((c) => c.passed), cases };
}

function withTimeout(promise, ms, abort) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      abort?.abort();
      const error = new Error(`Timed out after ${ms}ms.`);
      error.name = 'SandboxTimeout';
      reject(error);
    }, ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/** Strip anything that cannot survive structured clone. */
function serialisable(value, depth = 0) {
  if (value == null || depth > 6) return value ?? null;
  const type = typeof value;
  if (type === 'string' || type === 'number' || type === 'boolean') return value;
  if (type === 'bigint') return value.toString();
  if (type === 'function' || type === 'symbol') return `[${type}]`;
  if (value instanceof Date) return value.toISOString();
  if (value instanceof Error) return { name: value.name, message: value.message };
  if (Array.isArray(value)) return value.slice(0, 200).map((v) => serialisable(v, depth + 1));
  if (ArrayBuffer.isView(value)) return Array.from(value.slice(0, 200));
  const out = {};
  for (const [key, entry] of Object.entries(value).slice(0, 100)) {
    out[key] = serialisable(entry, depth + 1);
  }
  return out;
}

function formatLintErrors(errors) {
  return errors.map((e) => `line ${e.line}: ${e.message}`).join('\n');
}

function formatSpecFailures(cases) {
  return cases
    .filter((c) => !c.passed)
    .map((c) => `spec "${c.name}" failed: ${c.error ?? `expected ${JSON.stringify(c.expected)}, got ${JSON.stringify(c.actual)}`}`)
    .join('\n');
}

function truncate(text, max) {
  if (!text) return undefined;
  return text.length > max ? `${text.slice(0, max)}…` : text;
}
