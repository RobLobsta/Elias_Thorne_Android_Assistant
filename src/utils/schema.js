/**
 * Agentic creation schema (SAD §2.3) — parsing, validation and the shared
 * message contracts between the UI thread, Worker A and Worker B.
 *
 * Model output is untrusted text. Everything that crosses into the sandbox is
 * validated here first so that Worker B only ever sees a well-formed action.
 */

export const ACTIONS = Object.freeze(['create_tool', 'invoke_tool', 'self_improve']);

/** ESM CDNs Worker B is permitted to dynamically import from. */
export const ALLOWED_ESM_HOSTS = Object.freeze([
  'esm.sh',
  'cdn.jsdelivr.net',
  'cdn.skypack.dev',
  'unpkg.com',
]);

/** Hosts the throttled fetch controller will talk to. */
export const ALLOWED_FETCH_HOSTS = Object.freeze([
  ...ALLOWED_ESM_HOSTS,
  'api.open-meteo.com',
  'api.duckduckgo.com',
  'en.wikipedia.org',
  'r.jina.ai',
]);

export const MESSAGE = Object.freeze({
  // UI thread -> Worker A
  INIT: 'INIT',
  PROMPT: 'PROMPT',
  CANCEL: 'CANCEL',
  BIND_SANDBOX: 'BIND_SANDBOX',
  MEMORY_SEARCH: 'MEMORY_SEARCH',
  MEMORY_ADD: 'MEMORY_ADD',
  TOOL_LIST: 'TOOL_LIST',
  TOOL_FORGET: 'TOOL_FORGET',
  SANDBOX_READY: 'SANDBOX_READY',
  // Worker A -> UI thread
  STATUS: 'STATUS',
  TOKEN: 'TOKEN',
  REPLY: 'REPLY',
  TELEMETRY_LOG: 'TELEMETRY_LOG',
  ERROR: 'ERROR',
  TOOLS: 'TOOLS',
  MEMORY_RESULT: 'MEMORY_RESULT',
  SANDBOX_RESET: 'SANDBOX_RESET',
  // Worker A <-> Worker B (over the MessageChannel)
  BIND_INFERENCE: 'BIND_INFERENCE',
  EXEC_TOOL: 'EXEC_TOOL',
  EXEC_RESULT: 'EXEC_RESULT',
  LINT_TOOL: 'LINT_TOOL',
  LINT_RESULT: 'LINT_RESULT',
  SANDBOX_LOG: 'SANDBOX_LOG',
  DB_QUERY: 'DB_QUERY',
  DB_RESULT: 'DB_RESULT',
});

const TOOL_NAME_PATTERN = /^[a-z][a-z0-9_]{1,47}$/i;
const MAX_CODE_BYTES = 64 * 1024;
const MAX_DEPENDENCIES = 8;

class SchemaError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'SchemaError';
    this.field = field;
  }
}

/**
 * Pull the first JSON action object out of a model completion.
 *
 * Handles ```json fences, bare fences, and a naked object, and tolerates the
 * trailing prose small models like to append.
 *
 * @returns {{ action: object, prose: string } | null}
 */
export function extractAction(completion) {
  if (!completion) return null;

  const fenced = /```(?:json)?\s*([\s\S]*?)```/gi;
  for (const match of completion.matchAll(fenced)) {
    const parsed = tryParseObject(match[1]);
    if (parsed && ACTIONS.includes(parsed.action)) {
      return {
        action: parsed,
        prose: (completion.slice(0, match.index) + completion.slice(match.index + match[0].length)).trim(),
      };
    }
  }

  const start = completion.indexOf('{');
  if (start === -1) return null;
  const end = findMatchingBrace(completion, start);
  if (end === -1) return null;
  const parsed = tryParseObject(completion.slice(start, end + 1));
  if (!parsed || !ACTIONS.includes(parsed.action)) return null;
  return {
    action: parsed,
    prose: (completion.slice(0, start) + completion.slice(end + 1)).trim(),
  };
}

function tryParseObject(raw) {
  try {
    const value = JSON.parse(raw.trim());
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

/** Brace matcher that respects string literals and escapes. */
function findMatchingBrace(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * Validate and normalise a parsed action. Throws {@link SchemaError} with the
 * offending field so the self-improvement loop can feed a precise correction
 * back to the model rather than a generic failure.
 */
export function validateAction(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new SchemaError('Action must be a JSON object.', 'action');
  }
  const { action } = raw;
  if (!ACTIONS.includes(action)) {
    throw new SchemaError(`"action" must be one of ${ACTIONS.join(', ')}.`, 'action');
  }

  const toolName = String(raw.toolName ?? '').trim();
  if (!TOOL_NAME_PATTERN.test(toolName)) {
    throw new SchemaError(
      '"toolName" must be 2-48 characters, starting with a letter and containing only letters, digits or underscores.',
      'toolName',
    );
  }

  const telemetryMessage = String(raw.telemetryMessage ?? '').trim();
  if (telemetryMessage.length < 3) {
    throw new SchemaError(
      '"telemetryMessage" is required: one plain sentence telling the user what you are doing.',
      'telemetryMessage',
    );
  }

  const dependencies = normaliseDependencies(raw.dependencies);

  if (action === 'invoke_tool') {
    return {
      action,
      toolName,
      dependencies,
      telemetryMessage,
      args: normaliseArgs(raw.code ?? raw.args),
    };
  }

  const code = String(raw.code ?? '');
  if (!code.trim()) {
    throw new SchemaError('"code" is required for create_tool and self_improve.', 'code');
  }
  if (byteLength(code) > MAX_CODE_BYTES) {
    throw new SchemaError(`"code" exceeds the ${MAX_CODE_BYTES} byte sandbox limit.`, 'code');
  }

  return { action, toolName, dependencies, telemetryMessage, code };
}

function normaliseDependencies(value) {
  if (value == null) return [];
  const list = Array.isArray(value) ? value : [value];
  if (list.length > MAX_DEPENDENCIES) {
    throw new SchemaError(`At most ${MAX_DEPENDENCIES} dependencies may be requested.`, 'dependencies');
  }
  return list.map((entry) => {
    const url = String(entry ?? '').trim();
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      throw new SchemaError(`"${url}" is not an absolute URL.`, 'dependencies');
    }
    if (parsed.protocol !== 'https:') {
      throw new SchemaError('Dependencies must be served over https.', 'dependencies');
    }
    if (!ALLOWED_ESM_HOSTS.includes(parsed.hostname)) {
      throw new SchemaError(
        `"${parsed.hostname}" is not an allow-listed ESM CDN (${ALLOWED_ESM_HOSTS.join(', ')}).`,
        'dependencies',
      );
    }
    return parsed.toString();
  });
}

function normaliseArgs(value) {
  if (value == null || value === '') return {};
  if (typeof value === 'object') return value;
  const parsed = tryParseObject(String(value));
  if (parsed) return parsed;
  return { input: String(value) };
}

function byteLength(text) {
  return new TextEncoder().encode(text).length;
}

/** Host allow-list check used by the sandbox fetch controller. */
export function isFetchAllowed(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' && ALLOWED_FETCH_HOSTS.includes(parsed.hostname);
  } catch {
    return false;
  }
}

export { SchemaError };
