/**
 * Weight provisioning (SAD §2.2).
 *
 * A 2B parameter export is a gigabyte-class file. It cannot be committed, it
 * cannot be published to a GitHub Pages site (1 GB per site, 100 MB per file),
 * and it must not be pulled down unannounced on a handset's mobile data. So the
 * weights are not part of the build at all: this module resolves where they may
 * be found, reports what it would cost to fetch them, and downloads them only
 * when the user says so.
 *
 * Three places are consulted, in order:
 *
 *   1. `models/<modelFile>` next to the app. Present when someone dropped an
 *      export into `public/models` before building, and the only case where the
 *      service worker's cache-first rule applies on its own.
 *   2. The IndexedDB weights store, keyed by the remote URL. This is what makes
 *      the download a one-time cost — it survives app updates, and with the
 *      persistent storage lock held it survives Android's cache cleanup too.
 *   3. `weightsUrl` from `model-config.json`, over the network.
 *
 * Nothing here touches the DOM or ONNX Runtime; it hands back bytes and lets
 * the inference worker decide what to do with them.
 */

import { STORES, idbGet, idbSet, idbDelete } from './storage.js';

/**
 * Where a downloaded file is kept, keyed by its URL.
 *
 * IndexedDB rather than Cache Storage — see the note on STORES in
 * utils/storage.js. In short: Chromium caps a single Cache entry at roughly
 * 200 MB and fails past it with an opaque error, which is smaller than any
 * model worth running. The store is not versioned, so a download survives app
 * updates; the URL is the key, so changing `weightsUrl` fetches afresh.
 */
const KEY_PREFIX = 'weights:';

/** Progress callbacks fire per chunk; this throttles them to something a UI can use. */
const PROGRESS_INTERVAL_MS = 200;

/** Loopback is a secure context as far as browsers are concerned, so it is here too. */
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

/**
 * Normalise a configured remote URL.
 *
 * Absolute, and https unless it is loopback — weights are executable in every
 * sense that matters, and a plaintext fetch of a gigabyte of model is worth
 * refusing. Relative values are rejected outright: one would silently resolve
 * against the app's own origin and turn into a 404 at download time, which is a
 * far more confusing failure than being told the config is wrong.
 */
function remoteUrl(value, field) {
  if (!value) return null;
  let url;
  try {
    url = new URL(String(value));
  } catch {
    throw new Error(`model-config.json: ${field} must be an absolute URL, got "${value}".`);
  }
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && LOOPBACK.has(url.hostname))) {
    throw new Error(
      `model-config.json: ${field} must be https (or http on loopback), got "${url.protocol}//${url.hostname}".`,
    );
  }
  return url.toString();
}

/**
 * Work out where each required file could come from.
 *
 * Pure — no network, no caches — so the boot path can decide what to tell the
 * user before committing to anything expensive, and so it is testable outside a
 * browser.
 *
 * @param {object} config Merged model config.
 * @param {URL|string} modelsBase Directory the app serves `models/` from.
 * @returns {{ model: Source, externalData: Source|null, tokenizer: Source }}
 */
export function resolveWeightSources(config, modelsBase) {
  const base = String(modelsBase);
  const source = (file, remote, field) => ({
    file,
    local: file ? new URL(file, base).toString() : null,
    remote: remoteUrl(remote, field),
  });

  const model = source(config.modelFile, config.weightsUrl, 'weightsUrl');
  if (!model.file) throw new Error('model-config.json: modelFile is required.');

  // An export with external weight data is two files that must arrive
  // together; ONNX Runtime Web will not go and find the second one itself.
  const dataFile = config.weightsDataFile || null;
  const externalData = dataFile ? source(dataFile, config.weightsDataUrl, 'weightsDataUrl') : null;
  if (externalData && !externalData.remote && model.remote) {
    externalData.remote = new URL(dataFile, model.remote).toString();
  }

  // The tokenizer belongs to the export, so unless it is named explicitly it is
  // looked for beside the weights. Both of the layouts this is pointed at — a
  // Hugging Face `resolve/<rev>/` path and a release-asset directory — put the
  // two files side by side.
  const tokenizer = source('tokenizer.json', config.tokenizerUrl, 'tokenizerUrl');
  if (!tokenizer.remote && model.remote) {
    tokenizer.remote = new URL('tokenizer.json', model.remote).toString();
  }

  return { model, externalData, tokenizer };
}

/**
 * Is this file served by the app itself?
 *
 * A dev server's SPA fallback answers 200 with index.html for anything missing,
 * so the content type is checked too — otherwise a missing export surfaces much
 * later as an opaque parse error inside ONNX Runtime.
 *
 * @returns {Promise<{ ok: boolean, bytes: number }>}
 */
export async function probeLocal(url) {
  if (!url) return { ok: false, bytes: 0 };
  const response = await fetch(url, { method: 'HEAD' }).catch(() => null);
  if (!response?.ok) return { ok: false, bytes: 0 };
  if ((response.headers.get('content-type') ?? '').includes('text/html')) {
    return { ok: false, bytes: 0 };
  }
  return { ok: true, bytes: Number(response.headers.get('content-length')) || 0 };
}

/** Has this URL already been downloaded and kept? */
export async function probeStored(url) {
  const record = await readStored(url);
  return record ? { ok: true, bytes: record.bytes.byteLength } : { ok: false, bytes: 0 };
}

async function readStored(url) {
  if (!url) return null;
  try {
    const record = await idbGet(STORES.WEIGHTS, KEY_PREFIX + url);
    // A record written by a half-finished upgrade, or truncated on disk, is
    // worse than no record: it would load and fail much further downstream.
    if (!record?.bytes?.byteLength) return null;
    return record;
  } catch {
    return null;
  }
}

/**
 * What a download would cost, as far as the server will admit.
 *
 * Best-effort: a HEAD may be refused, or answered without a length, in which
 * case the caller falls back to the size declared in `model-config.json`.
 */
export async function probeRemoteSize(url) {
  if (!url) return 0;
  const response = await fetch(url, { method: 'HEAD', mode: 'cors' }).catch(() => null);
  if (!response?.ok) return 0;
  return Number(response.headers.get('content-length')) || 0;
}

/**
 * Fetch a file, reporting progress, and keep it for next time.
 *
 * A store that has it already short-circuits the network entirely — that is what
 * makes the app work offline. A write that fails must not fail the download:
 * the bytes in hand are usable for this session either way, so the caller is
 * told whether they stuck rather than being thrown at.
 *
 * @returns {Promise<{ bytes: Uint8Array, cached: boolean }>}
 */
export async function download(url, { onProgress, signal, expectedBytes = 0 } = {}) {
  const stored = await readStored(url);
  if (stored) {
    onProgress?.(stored.bytes.byteLength, stored.bytes.byteLength);
    return { bytes: stored.bytes, cached: true };
  }

  const response = await fetch(url, { signal, mode: 'cors', credentials: 'omit' });
  if (!response.ok) {
    throw new Error(`Weight download failed: ${response.status} ${response.statusText} for ${url}`);
  }
  const bytes = await readWithProgress(response, onProgress, expectedBytes);

  let cached = false;
  try {
    await idbSet(STORES.WEIGHTS, KEY_PREFIX + url, { bytes, storedAt: Date.now() });
    cached = true;
  } catch {
    // Out of quota, or the platform refused the write. The bytes in hand still
    // work for this session; the caller is told they did not persist, so the
    // user can be warned rather than silently paying the download every boot.
  }
  return { bytes, cached };
}

/**
 * Drain a response body into one contiguous buffer, reporting progress.
 *
 * ONNX Runtime wants a single `Uint8Array`, so the chunks have to be joined
 * eventually; doing it once at the end costs one copy rather than the quadratic
 * re-allocation of growing a buffer per chunk.
 */
async function readWithProgress(response, onProgress, expectedBytes) {
  const total = Number(response.headers.get('content-length')) || expectedBytes || 0;

  if (!response.body?.getReader) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    onProgress?.(buffer.byteLength, total || buffer.byteLength);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  let lastReport = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    const now = Date.now();
    if (now - lastReport >= PROGRESS_INTERVAL_MS) {
      lastReport = now;
      onProgress?.(received, total);
    }
  }
  onProgress?.(received, total || received);

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

/** Drop everything this module has stored, so a bad download can be retried. */
export async function evict(urls) {
  for (const url of urls) {
    if (url) await idbDelete(STORES.WEIGHTS, KEY_PREFIX + url).catch(() => {});
  }
}
