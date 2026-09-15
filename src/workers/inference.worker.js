/**
 * Worker A — BitNet 2B4T inference, Orama memory, and the agentic loop.
 * (SAD §1, §2.2, §2.3.)
 *
 * This worker owns everything slow and everything stateful: the ONNX Runtime
 * Web session on the WebGPU backend, the Orama index behind Elias's memory, the
 * persistent tool registry, and the self-improvement loop that repairs tools
 * when they fail. It never touches the DOM; it talks to the UI thread over
 * postMessage and to the sandbox (Worker B) over a MessageChannel.
 *
 * The one invariant worth stating loudly: no tool registration, no tool
 * configuration change and no self-correction pass commits anything before a
 * TELEMETRY_LOG has been posted to the UI thread. The user sees what Elias is
 * about to do to himself before he does it.
 */

import { create, insert, search, save, load, count, remove, getByID } from '@orama/orama';

import {
  buildMemoryBlock,
  buildPersonaBlock,
  AGENT_NAME,
} from '../utils/persona.js';
import { MESSAGE, extractAction, validateAction, SchemaError } from '../utils/schema.js';
import {
  STORES,
  idbGet,
  idbSet,
  idbDelete,
  idbAll,
  idbKeys,
  requestPersistentLock,
  IdleAutosaver,
} from '../utils/storage.js';
import { BpeTokenizer, applyChatTemplate, DEFAULT_CHAT_TEMPLATE } from '../utils/tokenizer.js';

/** Dimensionality of the local hashing embedding used for vector recall. */
const EMBED_DIMS = 256;
/** How many memories are retrieved and injected ahead of a turn. */
const RECALL_LIMIT = 4;
/** Ceiling on self-correction passes for a single user turn (SAD §2.3). */
const MAX_CORRECTION_PASSES = 3;
/** Conversation turns kept in the rolling context window. */
const HISTORY_TURNS = 6;

/**
 * Deployment base, injected by Vite. Both the model directory and the staged
 * ONNX runtime hang off it, so the worker addresses them correctly whether the
 * app is served from a domain root or a GitHub Pages subpath.
 */
const APP_BASE = new URL(import.meta.env.BASE_URL, self.location.origin);
const MODELS_BASE = new URL('models/', APP_BASE);

/** Defaults matching the published BitNet b1.58-2B-4T configuration. Override
 * them in public/models/model-config.json to match your own ONNX export. */
const DEFAULT_MODEL_CONFIG = Object.freeze({
  modelFile: 'bitnet-2b4t.onnx',
  numHiddenLayers: 30,
  numKeyValueHeads: 5,
  headDim: 128,
  maxContextTokens: 2048,
  maxNewTokens: 320,
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1,
  template: DEFAULT_CHAT_TEMPLATE,
});

const state = {
  engine: null,
  memory: null,
  tools: null,
  sandbox: null,
  config: DEFAULT_MODEL_CONFIG,
  history: [],
  generating: false,
  abort: false,
  backend: 'unavailable',
};

/* ------------------------------------------------------------- UI messaging */

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function status(stage, detail) {
  post(MESSAGE.STATUS, { stage, detail });
}

/**
 * The telemetry hook from SAD §2.3. Every autonomous change to Elias's own
 * configuration routes through here *first*.
 */
function telemetry(message, meta = {}) {
  post(MESSAGE.TELEMETRY_LOG, { message, at: Date.now(), ...meta });
}

function fail(error, context) {
  post(MESSAGE.ERROR, {
    message: error?.message ?? String(error),
    context,
    stack: error?.stack?.slice(0, 600),
  });
}

/* ------------------------------------------------------------- local memory */

/**
 * Deterministic hashing embedding.
 *
 * A second neural model purely for embeddings would double the memory budget on
 * a phone, so recall uses Orama's BM25 for lexical precision and this hashed
 * bag-of-words vector for fuzzy neighbourhood, combined in hybrid mode. The
 * vector is L2-normalised so cosine similarity behaves. Swap in a real encoder
 * here if you have the headroom — nothing else needs to change.
 */
export function embed(text, dims = EMBED_DIMS) {
  const vector = new Float32Array(dims);
  const tokens = String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  for (const token of tokens) {
    vector[hash32(token) % dims] += 1;
    // A character trigram channel keeps morphological variants close together.
    for (let i = 0; i + 3 <= token.length; i += 1) {
      vector[hash32(token.slice(i, i + 3)) % dims] += 0.5;
    }
  }
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm) || 1;
  return Array.from(vector, (value) => value / norm);
}

function hash32(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** Orama-backed persistent memory with idle-time snapshots to IndexedDB. */
class MemoryIndex {
  static SCHEMA = {
    text: 'string',
    kind: 'string',
    source: 'string',
    ts: 'number',
    embedding: `vector[${EMBED_DIMS}]`,
  };

  #db;
  #autosaver;

  constructor(db) {
    this.#db = db;
    this.#autosaver = new IdleAutosaver(() => this.#snapshot());
  }

  static async open() {
    const db = create({ schema: MemoryIndex.SCHEMA, id: 'elias-memory' });
    try {
      const snapshot = await idbGet(STORES.MEMORY, 'orama-snapshot');
      if (snapshot) {
        await load(db, snapshot);
        status('memory', `restored ${await count(db)} memories`);
      }
    } catch (error) {
      // A corrupt or version-shifted snapshot must not brick the boot; start
      // clean rather than refusing to run.
      console.warn('[memory] snapshot restore failed, starting empty:', error);
      await idbDelete(STORES.MEMORY, 'orama-snapshot').catch(() => {});
    }
    return new MemoryIndex(db);
  }

  async add(text, { kind = 'fact', source = 'conversation' } = {}) {
    const trimmed = String(text ?? '').trim();
    if (!trimmed) return null;
    const id = await insert(this.#db, {
      text: trimmed,
      kind,
      source,
      ts: Date.now(),
      embedding: embed(trimmed),
    });
    this.#autosaver.schedule();
    return id;
  }

  async recall(term, limit = RECALL_LIMIT) {
    const query = String(term ?? '').trim();
    if (!query) return [];
    try {
      const results = await search(this.#db, {
        mode: 'hybrid',
        term: query,
        properties: ['text'],
        vector: { value: embed(query), property: 'embedding' },
        similarity: 0.2,
        limit,
      });
      return results.hits.map((hit) => ({
        id: hit.id,
        score: hit.score,
        text: hit.document.text,
        kind: hit.document.kind,
        ts: hit.document.ts,
      }));
    } catch (error) {
      console.warn('[memory] hybrid recall failed, falling back to full text:', error);
      const results = await search(this.#db, { term: query, properties: ['text'], limit });
      return results.hits.map((hit) => ({ id: hit.id, score: hit.score, text: hit.document.text }));
    }
  }

  async forget(id) {
    const existing = await getByID(this.#db, id);
    if (!existing) return false;
    await remove(this.#db, id);
    this.#autosaver.schedule();
    return true;
  }

  size() {
    return count(this.#db);
  }

  flush() {
    return this.#autosaver.flushNow();
  }

  async #snapshot() {
    await idbSet(STORES.MEMORY, 'orama-snapshot', await save(this.#db));
  }
}

/* ------------------------------------------------------------ tool registry */

/** Persistent registry of JIT tools Elias has written for himself. */
class ToolRegistry {
  #tools = new Map();

  static async open() {
    const registry = new ToolRegistry();
    const keys = await idbKeys(STORES.TOOLS).catch(() => []);
    const records = await idbAll(STORES.TOOLS).catch(() => []);
    keys.forEach((key, index) => {
      if (records[index]) registry.#tools.set(String(key), records[index]);
    });
    return registry;
  }

  names() {
    return [...this.#tools.keys()];
  }

  list() {
    return [...this.#tools.values()].map(({ code, ...rest }) => ({
      ...rest,
      codeBytes: new TextEncoder().encode(code ?? '').length,
    }));
  }

  get(name) {
    return this.#tools.get(name) ?? null;
  }

  /** Commit a tool. Callers must have posted telemetry before reaching here. */
  async commit(record) {
    const previous = this.#tools.get(record.toolName);
    const merged = {
      ...record,
      revision: (previous?.revision ?? 0) + 1,
      createdAt: previous?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
    };
    this.#tools.set(record.toolName, merged);
    await idbSet(STORES.TOOLS, record.toolName, merged);
    return merged;
  }

  async forget(name) {
    if (!this.#tools.delete(name)) return false;
    await idbDelete(STORES.TOOLS, name);
    return true;
  }
}

/* --------------------------------------------------------- sandbox bridge */

/** Request/response bridge over the MessageChannel to Worker B. */
class SandboxBridge {
  #port;
  #pending = new Map();
  #seq = 0;
  ready = false;

  constructor(port) {
    this.#port = port;
    port.onmessage = (event) => this.#onMessage(event.data ?? {});
    port.start?.();
  }

  close() {
    this.#port.onmessage = null;
    for (const { reject } of this.#pending.values()) {
      reject(new Error('Sandbox connection replaced.'));
    }
    this.#pending.clear();
    this.#port.close?.();
  }

  lint(code) {
    return this.#request(MESSAGE.LINT_TOOL, { code }, 10_000);
  }

  exec(payload) {
    return this.#request(MESSAGE.EXEC_TOOL, payload, (payload.timeoutMs ?? 20_000) + 10_000);
  }

  #request(type, payload, timeoutMs) {
    const id = `sb-${this.#seq++}`;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#port.postMessage({ type, id, ...payload });
      setTimeout(() => {
        if (!this.#pending.has(id)) return;
        this.#pending.delete(id);
        reject(new Error(`Sandbox did not answer ${type} within ${timeoutMs}ms.`));
      }, timeoutMs);
    });
  }

  async #onMessage(message) {
    switch (message.type) {
      case MESSAGE.SANDBOX_READY:
        this.ready = true;
        status('sandbox', 'JIT sandbox online');
        return;

      case MESSAGE.SANDBOX_LOG:
        post(MESSAGE.TELEMETRY_LOG, {
          message: `[${message.toolName}] ${message.message}`,
          at: Date.now(),
          channel: 'sandbox',
        });
        return;

      // Worker B has no database handle of its own; memory access is proxied.
      case MESSAGE.DB_QUERY: {
        try {
          const result =
            message.op === 'search'
              ? await state.memory.recall(message.payload?.term, message.payload?.limit ?? 5)
              : await state.memory.add(message.payload?.text, {
                  kind: 'tool-output',
                  source: `tool:${message.payload?.meta?.toolName ?? 'unknown'}`,
                });
          this.#port.postMessage({ type: MESSAGE.DB_RESULT, id: message.id, result });
        } catch (error) {
          this.#port.postMessage({ type: MESSAGE.DB_RESULT, id: message.id, error: error.message });
        }
        return;
      }

      case MESSAGE.LINT_RESULT:
      case MESSAGE.EXEC_RESULT: {
        const pending = this.#pending.get(message.id);
        if (!pending) return;
        this.#pending.delete(message.id);
        pending.resolve(message);
        return;
      }

      default:
    }
  }
}

/* ----------------------------------------------------------- ONNX / BitNet */

/**
 * ONNX Runtime is served from /ort/ rather than imported as a bundled module.
 *
 * The runtime resolves its 28 MB WASM binary at execution time from
 * `env.wasm.wasmPaths` no matter how the JS half arrives, so bundling the JS
 * only pulls the binary through the build for nothing. Staged assets also mean
 * the first inference works with the radio off, which matters for an agent
 * whose whole premise is that it runs on the handset.
 */
const ORT_BASE = new URL('ort/', APP_BASE).toString();
let ortModule = null;

function loadOrt() {
  // @vite-ignore: this is a runtime URL, not a module the bundler should chase.
  ortModule ??= import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`);
  return ortModule;
}

/**
 * Generic decoder-only ONNX runner.
 *
 * Rather than hard-coding one export's tensor names, this introspects the
 * session's IO and adapts to the `past_key_values.N.key` / `present.N.key`
 * conventions the common exporters emit. That matters because BitNet ONNX
 * exports are still a moving target.
 */
class BitNetEngine {
  #session;
  #tokenizer;
  #config;
  #kvLayers = [];
  #wantsAttentionMask = false;
  #wantsPositionIds = false;
  #stopIds = new Set();

  constructor({ session, tokenizer, config, backend }) {
    this.#session = session;
    this.#tokenizer = tokenizer;
    this.#config = config;
    this.backend = backend;
    this.#introspect();
  }

  get tokenizer() {
    return this.#tokenizer;
  }

  static async create(config, { onProgress } = {}) {
    const ort = await loadOrt();
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 4, 4);
    ort.env.logLevel = 'warning';

    const modelUrl = new URL(config.modelFile, MODELS_BASE).toString();
    const head = await fetch(modelUrl, { method: 'HEAD' }).catch(() => null);
    if (!head?.ok) {
      throw new Error(`No BitNet weights at ${modelUrl} (HTTP ${head?.status ?? 'unreachable'}).`);
    }
    // A dev server's SPA fallback answers 200 with index.html for a missing
    // file, which otherwise surfaces much later as an opaque JSON parse error.
    const contentType = head.headers.get('content-type') ?? '';
    if (contentType.includes('text/html')) {
      throw new Error(`No BitNet weights at ${modelUrl} — the server returned HTML.`);
    }

    onProgress?.('loading tokenizer');
    const tokenizer = await BpeTokenizer.load(MODELS_BASE);

    onProgress?.('compiling graph for WebGPU');
    const providers = [];
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) providers.push('webgpu');
    providers.push('wasm');

    let session = null;
    let backend = null;
    let lastError = null;
    for (const provider of providers) {
      try {
        session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
          graphOptimizationLevel: 'all',
          enableMemPattern: provider === 'wasm',
        });
        backend = provider;
        break;
      } catch (error) {
        lastError = error;
        onProgress?.(`${provider} backend unavailable, trying next`);
      }
    }
    if (!session) throw lastError ?? new Error('No execution provider could load the model.');

    return new BitNetEngine({ session, tokenizer, config, backend });
  }

  #introspect() {
    const inputs = this.#session.inputNames;
    const outputs = this.#session.outputNames;
    this.#wantsAttentionMask = inputs.includes('attention_mask');
    this.#wantsPositionIds = inputs.includes('position_ids');

    const pastPattern = /^past_key_values\.(\d+)\.(key|value)$/;
    const layers = new Map();
    for (const name of inputs) {
      const match = pastPattern.exec(name);
      if (!match) continue;
      const index = Number(match[1]);
      const entry = layers.get(index) ?? {};
      entry[match[2]] = name;
      layers.set(index, entry);
    }

    for (const [index, entry] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
      const presentKey =
        outputs.find((n) => n === `present.${index}.key`) ??
        outputs.find((n) => n === `present_key_values.${index}.key`);
      const presentValue =
        outputs.find((n) => n === `present.${index}.value`) ??
        outputs.find((n) => n === `present_key_values.${index}.value`);
      if (!presentKey || !presentValue) continue;
      this.#kvLayers.push({
        pastKey: entry.key,
        pastValue: entry.value,
        presentKey,
        presentValue,
      });
    }

    for (const stop of this.#config.template?.stopTokens ?? []) {
      const id = this.#tokenizer.specialId(stop);
      if (id != null) this.#stopIds.add(id);
    }
  }

  get usesCache() {
    return this.#kvLayers.length > 0;
  }

  /**
   * Stream a completion.
   * @param {string} prompt fully templated prompt text
   * @param {{onToken?: (text: string) => void, shouldStop?: () => boolean}} hooks
   */
  async generate(prompt, { onToken, shouldStop } = {}) {
    const ort = await loadOrt();
    const { maxContextTokens, maxNewTokens } = this.#config;

    let ids = this.#tokenizer.encode(prompt);
    if (ids.length > maxContextTokens - maxNewTokens) {
      // Keep the tail: the persona is re-prepended every turn anyway.
      ids = ids.slice(-(maxContextTokens - maxNewTokens));
    }

    let cache = this.usesCache ? this.#emptyCache(ort) : null;
    let position = 0;
    const generated = [];
    let emitted = '';

    for (let step = 0; step < maxNewTokens; step += 1) {
      if (shouldStop?.()) break;

      const inputIds = step === 0 || !this.usesCache ? ids.concat(generated) : [generated.at(-1)];
      const feeds = this.#buildFeeds(ort, inputIds, position, cache);
      const outputs = await this.#session.run(feeds);

      const logits = this.#lastTokenLogits(outputs.logits);
      const next = sampleToken(logits, generated, this.#config);

      if (this.#stopIds.has(next)) break;
      generated.push(next);
      position += inputIds.length;

      if (this.usesCache) {
        cache = this.#collectCache(outputs, cache);
      }

      // Decode the whole tail each step so multi-byte characters are never
      // split across two emitted chunks.
      const decoded = this.#tokenizer.decode(generated);
      if (decoded.length > emitted.length) {
        onToken?.(decoded.slice(emitted.length));
        emitted = decoded;
      }
    }

    return emitted;
  }

  #emptyCache(ort) {
    const { numKeyValueHeads, headDim } = this.#config;
    const cache = {};
    for (const layer of this.#kvLayers) {
      const empty = () =>
        new ort.Tensor('float32', new Float32Array(0), [1, numKeyValueHeads, 0, headDim]);
      cache[layer.pastKey] = empty();
      cache[layer.pastValue] = empty();
    }
    return cache;
  }

  #buildFeeds(ort, inputIds, position, cache) {
    const length = inputIds.length;
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds, BigInt), [1, length]),
    };

    if (this.#wantsAttentionMask) {
      const total = position + length;
      feeds.attention_mask = new ort.Tensor(
        'int64',
        BigInt64Array.from({ length: total }, () => 1n),
        [1, total],
      );
    }
    if (this.#wantsPositionIds) {
      feeds.position_ids = new ort.Tensor(
        'int64',
        BigInt64Array.from({ length }, (_, i) => BigInt(position + i)),
        [1, length],
      );
    }
    if (cache) Object.assign(feeds, cache);
    return feeds;
  }

  #collectCache(outputs, previous) {
    const next = {};
    for (const layer of this.#kvLayers) {
      next[layer.pastKey] = outputs[layer.presentKey] ?? previous[layer.pastKey];
      next[layer.pastValue] = outputs[layer.presentValue] ?? previous[layer.pastValue];
    }
    // Free the WebGPU/wasm buffers we are no longer feeding back.
    for (const [name, tensor] of Object.entries(previous)) {
      if (next[name] !== tensor) tensor?.dispose?.();
    }
    return next;
  }

  #lastTokenLogits(tensor) {
    const [, sequence, vocab] = tensor.dims;
    const data = tensor.data;
    const offset = (sequence - 1) * vocab;
    const slice = new Float32Array(vocab);
    for (let i = 0; i < vocab; i += 1) slice[i] = Number(data[offset + i]);
    return slice;
  }

  async dispose() {
    await this.#session.release?.();
  }
}

/** Temperature / top-k / top-p sampling with a repetition penalty. */
function sampleToken(logits, generated, config) {
  const { temperature = 0.7, topK = 40, topP = 0.9, repetitionPenalty = 1.1 } = config;

  if (repetitionPenalty !== 1) {
    // Penalise only the recent window; penalising the whole history makes a
    // small model incoherent rather than merely less repetitive.
    for (const id of new Set(generated.slice(-64))) {
      logits[id] = logits[id] > 0 ? logits[id] / repetitionPenalty : logits[id] * repetitionPenalty;
    }
  }

  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i] > logits[best]) best = i;
    return best;
  }

  let candidates = Array.from(logits, (logit, id) => ({ id, logit: logit / temperature }));
  candidates.sort((a, b) => b.logit - a.logit);
  if (topK > 0) candidates = candidates.slice(0, topK);

  const max = candidates[0].logit;
  let total = 0;
  for (const candidate of candidates) {
    candidate.probability = Math.exp(candidate.logit - max);
    total += candidate.probability;
  }
  for (const candidate of candidates) candidate.probability /= total;

  if (topP > 0 && topP < 1) {
    let cumulative = 0;
    const nucleus = [];
    for (const candidate of candidates) {
      nucleus.push(candidate);
      cumulative += candidate.probability;
      if (cumulative >= topP) break;
    }
    candidates = nucleus;
    const renorm = candidates.reduce((sum, c) => sum + c.probability, 0);
    for (const candidate of candidates) candidate.probability /= renorm;
  }

  let roll = crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32;
  for (const candidate of candidates) {
    roll -= candidate.probability;
    if (roll <= 0) return candidate.id;
  }
  return candidates.at(-1).id;
}

/* ------------------------------------------------------- fallback engine */

/**
 * Deterministic stand-in used when no weights are present or WebGPU/WASM cannot
 * host them.
 *
 * This is not a language model and does not pretend to be one. It exists so the
 * rest of the architecture — wake word, state machine, telemetry, linting,
 * sandbox execution, memory — is exercisable end to end on a device that has
 * never downloaded a 2B parameter file. It emits the same action schema the
 * real model does, so the JIT path is genuinely tested rather than stubbed.
 */
class HeuristicEngine {
  backend = 'heuristic';

  constructor(getTools) {
    this.getTools = getTools;
  }

  /**
   * @param {string} prompt the fully templated prompt (unused except as a
   *   fallback source for the user turn)
   * @param {{onToken?: Function, userText?: string}} hooks `userText` is the
   *   raw turn; relying on it avoids re-parsing a prompt whose turn markers are
   *   configurable.
   */
  async generate(prompt, { onToken, userText } = {}) {
    const reply = this.#respond(userText ?? lastUserTurn(prompt, state.config.template));
    // Stream it, so the UI path is identical to the real engine's.
    for (const chunk of reply.match(/.{1,6}/gs) ?? []) {
      onToken?.(chunk);
      await new Promise((resolve) => setTimeout(resolve, 12));
    }
    return reply;
  }

  #respond(input) {
    const text = input.toLowerCase();

    // The agent loop feeds tool output back for a spoken summary; close that
    // loop so invoke_tool produces a real answer rather than boilerplate.
    const toolResult = /^tool result:\s*([\s\S]+?)\n/i.exec(input);
    if (toolResult) return summariseToolResult(toolResult[1]);

    const buildMatch = /(?:build|write|create|make)\s+(?:me\s+)?a\s+tool\s+(?:that\s+|to\s+|which\s+)?(.+)/.exec(
      text,
    );
    if (buildMatch) return this.#createToolAction(buildMatch[1]);

    const tool = this.#bestTool(text);
    if (tool) {
      const numbers = (text.match(/-?\d+(?:\.\d+)?/g) ?? []).map(Number);
      return fence({
        action: 'invoke_tool',
        toolName: tool,
        dependencies: [],
        code: JSON.stringify({ input, numbers }),
        telemetryMessage: `Invoking my ${tool} tool.`,
      });
    }

    if (/\b(who are you|your name|what are you)\b/.test(text)) {
      return `I am ${AGENT_NAME}. I run entirely on this device — my weights, my memory and my tools all live here, and nothing I hear leaves the handset.`;
    }
    if (/\b(time|clock)\b/.test(text)) {
      return `It is ${new Date().toLocaleTimeString()}.`;
    }
    if (/\b(date|today)\b/.test(text)) {
      return `Today is ${new Date().toLocaleDateString(undefined, { dateStyle: 'full' })}.`;
    }
    if (/\bremember\b/.test(text)) {
      return 'Noted. I have written that to my local index.';
    }
    if (/\b(what tools|which tools|capabilities)\b/.test(text)) {
      const tools = this.getTools();
      return tools.length
        ? `I currently hold ${tools.length} tool${tools.length === 1 ? '' : 's'}: ${tools.join(', ')}.`
        : 'I have no tools yet. Ask me to build one and I will write it.';
    }

    return `My weights are not loaded, so I am running on my fallback reasoner. Drop a BitNet export into public/models and I will think properly. You said: "${input}".`;
  }

  /**
   * Pick the registered tool whose name best overlaps the utterance.
   *
   * Words are compared on a four-character stem so "add 12 and 30" reaches a
   * tool named `adds_numbers_together`. Crude, but this only has to be good
   * enough to demonstrate routing without a language model.
   */
  #bestTool(text) {
    const spoken = new Set((text.match(/[a-z]{3,}/g) ?? []).map(stem));
    let best = null;
    let bestScore = 0;

    for (const name of this.getTools()) {
      const parts = name.split('_').filter((part) => part.length >= 3 && !STOP_WORDS.has(part));
      const score = parts.reduce((sum, part) => sum + (spoken.has(stem(part)) ? 1 : 0), 0);
      if (score > bestScore) {
        bestScore = score;
        best = name;
      }
    }
    return bestScore > 0 ? best : null;
  }

  #createToolAction(description) {
    const trimmed = description.trim();
    const toolName = slug(trimmed) || 'generated_tool';
    const code = [
      '// Generated by the fallback reasoner to exercise the JIT pipeline.',
      `const description = ${JSON.stringify(trimmed)};`,
      'const numbers = Array.isArray(args?.numbers) ? args.numbers.map(Number).filter(Number.isFinite) : [];',
      'const total = numbers.reduce((sum, n) => sum + n, 0);',
      'ctx.log(`ran over ${numbers.length} value(s)`);',
      'return { description, count: numbers.length, total };',
    ].join('\n');

    return fence({
      action: 'create_tool',
      toolName,
      dependencies: [],
      code,
      telemetryMessage: `Writing a new tool called ${toolName} that ${trimmed}.`,
    });
  }
}

const STOP_WORDS = new Set(['tool', 'the', 'and', 'for', 'that', 'with']);

/** Crude suffix-stripping stem so "add" and "adds" compare equal. */
function stem(word) {
  return word.replace(/(?:ies|ing|ed|es|s)$/, '').slice(0, 4);
}

/** Turn a tool's JSON result into something speakable. */
function summariseToolResult(raw) {
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    return `The tool returned: ${raw.slice(0, 200)}`;
  }
  if (value && typeof value === 'object' && typeof value.total === 'number' && value.count) {
    return `That comes to ${value.total} across ${value.count} value${value.count === 1 ? '' : 's'}.`;
  }
  if (value == null) return 'The tool ran and returned nothing.';
  if (typeof value === 'object') {
    const pairs = Object.entries(value)
      .slice(0, 3)
      .map(([key, entry]) => `${key} is ${typeof entry === 'object' ? JSON.stringify(entry) : entry}`);
    return `The tool reports ${pairs.join(', ')}.`;
  }
  return `The tool returned ${value}.`;
}

function fence(object) {
  return ['```json', JSON.stringify(object, null, 2), '```'].join('\n');
}

function slug(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .split('_')
    .slice(0, 3)
    .join('_');
}

/**
 * Recover the most recent user turn from a rendered prompt.
 *
 * The turn markers come from the configured template rather than being
 * hard-coded, and the last match wins — matching the first one silently
 * swallowed the whole conversation, markers included.
 */
function lastUserTurn(prompt, template = DEFAULT_CHAT_TEMPLATE) {
  const slot = template.user ?? DEFAULT_CHAT_TEMPLATE.user;
  const [open, close] = slot.split('{content}');
  if (!open || !close) return prompt.trim();

  const pattern = new RegExp(`${escapeRegExp(open)}([\\s\\S]*?)${escapeRegExp(close)}`, 'g');
  const matches = [...prompt.matchAll(pattern)];
  return (matches.at(-1)?.[1] ?? prompt).trim();
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/* ------------------------------------------------------------- agent loop */

function buildPrompt(userText, memories) {
  const persona = buildPersonaBlock({
    toolNames: state.tools.names(),
    capabilities: { backend: state.backend },
  });
  const memoryBlock = buildMemoryBlock(memories);

  const messages = [{ role: 'system', content: memoryBlock ? `${persona}\n\n${memoryBlock}` : persona }];
  for (const turn of state.history.slice(-HISTORY_TURNS * 2)) messages.push(turn);
  messages.push({ role: 'user', content: userText });

  return applyChatTemplate(messages, state.config.template);
}

async function handleTurn(userText) {
  if (state.generating) {
    post(MESSAGE.ERROR, { message: 'Already thinking — say "stop" to interrupt.' });
    return;
  }
  state.generating = true;
  state.abort = false;

  try {
    status('thinking', 'recalling');
    const memories = await state.memory.recall(userText);
    if (memories.length) {
      post(MESSAGE.MEMORY_RESULT, { hits: memories, reason: 'recall' });
    }

    let prompt = buildPrompt(userText, memories);
    let followUp = null;
    let spoken = '';

    for (let pass = 0; pass <= MAX_CORRECTION_PASSES; pass += 1) {
      status('thinking', pass === 0 ? 'generating' : `self-correction pass ${pass}`);

      const completion = await state.engine.generate(prompt, {
        onToken: (text) => post(MESSAGE.TOKEN, { text }),
        shouldStop: () => state.abort,
        userText: followUp ?? userText,
      });

      if (state.abort) {
        spoken = '';
        break;
      }

      const parsed = extractAction(completion);
      if (!parsed) {
        spoken = completion.trim();
        break;
      }

      let action;
      try {
        action = validateAction(parsed.action);
      } catch (error) {
        if (!(error instanceof SchemaError) || pass === MAX_CORRECTION_PASSES) {
          spoken = parsed.prose || completion.trim();
          break;
        }
        // SAD §2.3: a self-correction pass is announced before it runs.
        telemetry(`Correcting my own output: ${error.message}`, { channel: 'self-improve' });
        followUp = `Your JSON action was rejected (${error.field}): ${error.message} Re-emit the corrected object only.`;
        prompt = appendCorrection(prompt, completion, followUp);
        continue;
      }

      // SAD §2.3: telemetry is posted before anything commits.
      telemetry(action.telemetryMessage, { channel: 'self-improve', action: action.action });

      const outcome = await applyAction(action);

      if (outcome.ok) {
        if (outcome.speak) {
          spoken = outcome.speak;
          break;
        }
        followUp = `Tool result: ${JSON.stringify(outcome.result).slice(0, 1200)}\nAnswer the user in one or two spoken sentences. Do not emit JSON.`;
        prompt = appendCorrection(prompt, completion, followUp);
        continue;
      }

      if (pass === MAX_CORRECTION_PASSES) {
        spoken = `I could not get ${action.toolName} working: ${outcome.error}`;
        break;
      }

      telemetry(`Repairing ${action.toolName} — ${firstLine(outcome.error)}`, {
        channel: 'self-improve',
        action: 'self_improve',
      });
      followUp = `Your tool "${action.toolName}" failed during ${outcome.phase}:\n${outcome.error}\nEmit a corrected "self_improve" action for the same toolName.`;
      prompt = appendCorrection(prompt, completion, followUp);
    }

    const reply = spoken || (state.abort ? '' : 'I do not have an answer for that yet.');
    state.history.push({ role: 'user', content: userText });
    if (reply) state.history.push({ role: 'assistant', content: reply });

    await rememberTurn(userText, reply);
    post(MESSAGE.REPLY, { text: reply, aborted: state.abort });
    status('idle');
  } catch (error) {
    fail(error, 'turn');
    status('idle');
  } finally {
    state.generating = false;
  }
}

function appendCorrection(prompt, completion, instruction) {
  const template = state.config.template ?? DEFAULT_CHAT_TEMPLATE;
  return (
    prompt +
    completion +
    (template.assistant?.includes('<|eot_id|>') ? '<|eot_id|>' : '\n') +
    (template.user ?? '{content}').replace('{content}', instruction) +
    (template.generationPrefix ?? '')
  );
}

function firstLine(text) {
  return String(text ?? '').split('\n')[0].slice(0, 160);
}

/**
 * Persist anything durable from the turn. Explicit "remember ..." statements
 * are stored verbatim; everything else is stored as a lower-weight episode so
 * recall has context without the index filling with small talk.
 */
async function rememberTurn(userText, reply) {
  const explicit = /\b(?:remember|note that|don'?t forget)\b[:,]?\s*(.+)/i.exec(userText);
  if (explicit) {
    await state.memory.add(explicit[1].trim(), { kind: 'fact', source: 'user' });
    return;
  }
  if (reply && userText.length > 12) {
    await state.memory.add(`Q: ${userText}\nA: ${reply}`, { kind: 'episode', source: 'conversation' });
  }
}

/** Execute a validated action. Telemetry has already been posted by the caller. */
async function applyAction(action) {
  if (!state.sandbox?.ready) {
    return { ok: false, phase: 'sandbox', error: 'The JIT sandbox is not connected.' };
  }

  switch (action.action) {
    case 'create_tool':
    case 'self_improve': {
      const lint = await state.sandbox.lint(action.code);
      if (!lint.ok) {
        return {
          ok: false,
          phase: 'lint',
          error: lint.errors.map((e) => `line ${e.line}: ${e.message}`).join('\n'),
        };
      }

      const committed = await state.tools.commit({
        toolName: action.toolName,
        code: action.code,
        dependencies: action.dependencies,
        telemetryMessage: action.telemetryMessage,
      });
      post(MESSAGE.TOOLS, { tools: state.tools.list() });
      await state.memory.add(
        `Tool "${action.toolName}" revision ${committed.revision}: ${action.telemetryMessage}`,
        { kind: 'tool', source: 'self-improve' },
      );

      return {
        ok: true,
        speak:
          action.action === 'create_tool'
            ? `I have written a tool called ${action.toolName}.`
            : `I have updated ${action.toolName} to revision ${committed.revision}.`,
      };
    }

    case 'invoke_tool': {
      const tool = state.tools.get(action.toolName);
      if (!tool) {
        return {
          ok: false,
          phase: 'registry',
          error: `No tool named "${action.toolName}" is registered. Create it first.`,
        };
      }

      const result = await state.sandbox.exec({
        toolName: tool.toolName,
        code: tool.code,
        dependencies: tool.dependencies,
        args: action.args,
        specs: tool.specs ?? [],
      });

      if (result.needsRespawn) {
        // Only the thread that created Worker B can terminate it.
        post(MESSAGE.SANDBOX_RESET, { reason: result.error });
      }
      if (!result.ok) return { ok: false, phase: result.phase, error: result.error };
      return { ok: true, result: result.result };
    }

    default:
      return { ok: false, phase: 'schema', error: `Unsupported action "${action.action}".` };
  }
}

/* --------------------------------------------------------------- lifecycle */

async function loadModelConfig() {
  try {
    const response = await fetch(new URL('model-config.json', MODELS_BASE));
    if (!response.ok) return DEFAULT_MODEL_CONFIG;
    const override = await response.json();
    return {
      ...DEFAULT_MODEL_CONFIG,
      ...override,
      template: { ...DEFAULT_CHAT_TEMPLATE, ...(override.template ?? {}) },
    };
  } catch {
    return DEFAULT_MODEL_CONFIG;
  }
}

async function boot() {
  status('boot', 'claiming persistent storage');
  const lock = await requestPersistentLock();
  post(MESSAGE.STATUS, { stage: 'storage', detail: lock });

  state.config = await loadModelConfig();
  state.memory = await MemoryIndex.open();
  state.tools = await ToolRegistry.open();
  post(MESSAGE.TOOLS, { tools: state.tools.list() });

  status('boot', 'starting inference engine');
  try {
    state.engine = await BitNetEngine.create(state.config, {
      onProgress: (detail) => status('boot', detail),
    });
    state.backend = state.engine.backend;
    status('ready', `BitNet 2B4T on ${state.engine.backend}`);
  } catch (error) {
    state.engine = new HeuristicEngine(() => state.tools.names());
    state.backend = 'heuristic';
    post(MESSAGE.STATUS, {
      stage: 'ready',
      detail: 'fallback reasoner',
      degraded: true,
      reason: error.message,
    });
  }

  post(MESSAGE.STATUS, {
    stage: 'online',
    detail: {
      backend: state.backend,
      memories: await state.memory.size(),
      tools: state.tools.names(),
      persisted: lock.persisted,
    },
  });
}

self.addEventListener('message', async (event) => {
  const message = event.data ?? {};
  try {
    switch (message.type) {
      case MESSAGE.INIT:
        await boot();
        return;

      case MESSAGE.BIND_SANDBOX:
        state.sandbox?.close();
        state.sandbox = new SandboxBridge(message.port);
        return;

      case MESSAGE.PROMPT:
        await handleTurn(String(message.text ?? '').trim());
        return;

      case MESSAGE.CANCEL:
        state.abort = true;
        return;

      case MESSAGE.MEMORY_SEARCH:
        post(MESSAGE.MEMORY_RESULT, {
          hits: await state.memory.recall(message.term, message.limit ?? 10),
          reason: 'query',
        });
        return;

      case MESSAGE.MEMORY_ADD:
        await state.memory.add(message.text, { kind: message.kind ?? 'fact', source: 'user' });
        await state.memory.flush();
        post(MESSAGE.STATUS, { stage: 'memory', detail: `${await state.memory.size()} memories` });
        return;

      case MESSAGE.TOOL_LIST:
        post(MESSAGE.TOOLS, { tools: state.tools.list() });
        return;

      case MESSAGE.TOOL_FORGET:
        await state.tools.forget(message.toolName);
        post(MESSAGE.TOOLS, { tools: state.tools.list() });
        return;

      default:
    }
  } catch (error) {
    fail(error, message.type);
  }
});
