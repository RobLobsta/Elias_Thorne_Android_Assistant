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
 *   2. The Cache Storage bucket, keyed by the remote URL. This is what makes
 *      the download a one-time cost — it survives app updates, and with the
 *      persistent storage lock held it survives Android's cache cleanup too.
 *   3. `weightsUrl` from `model-config.json`, over the network.
 *
 * Nothing here touches the DOM or ONNX Runtime; it hands back bytes and lets
 * the inference worker decide what to do with them.
 */

/**
 * Shared with the service worker's cache-first rule. Deliberately unversioned:
 * a download this large must survive app updates.
 */
export const MODEL_CACHE = 'elias-models';

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
export async function probeCache(url) {
  if (!url || typeof caches === 'undefined') return { ok: false, bytes: 0 };
  try {
    const cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url, { ignoreVary: true });
    if (!hit) return { ok: false, bytes: 0 };
    return { ok: true, bytes: Number(hit.headers.get('content-length')) || 0 };
  } catch {
    return { ok: false, bytes: 0 };
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
 * The response is cloned into Cache Storage while the original is read, so the
 * bytes are written to disk as they arrive rather than being buffered a second
 * time in JavaScript. A cache write that fails — quota, a cross-origin response
 * without CORS headers — must not fail the download, because the bytes in hand
 * are still perfectly usable for this session.
 *
 * @returns {Promise<{ bytes: Uint8Array, cached: boolean }>}
 */
export async function download(url, { onProgress, signal, expectedBytes = 0 } = {}) {
  const cache = typeof caches !== 'undefined' ? await caches.open(MODEL_CACHE).catch(() => null) : null;

  const hit = await cache?.match(url, { ignoreVary: true }).catch(() => null);
  if (hit) {
    return { bytes: await readWithProgress(hit, onProgress, expectedBytes), cached: true };
  }

  const response = await fetch(url, { signal, mode: 'cors', credentials: 'omit' });
  if (!response.ok) {
    throw new Error(`Weight download failed: ${response.status} ${response.statusText} for ${url}`);
  }

  // Started before the body is read so both branches drain together; a lagging
  // branch would otherwise buffer the whole file in memory.
  let stored = false;
  const stash = cache
    ? cache
        .put(url, response.clone())
        .then(() => {
          stored = true;
        })
        .catch(() => {})
    : Promise.resolve();

  let bytes;
  try {
    bytes = await readWithProgress(response, onProgress, expectedBytes);
  } catch (error) {
    // Let the cache write settle before the error propagates. Aborting the
    // fetch errors both branches of the tee, so the put rejects and stores
    // nothing — but a caller that evicts on failure would otherwise be racing
    // it, and could delete the entry just before a partial one landed.
    await stash;
    throw error;
  }
  await stash;
  return { bytes, cached: stored };
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

/** Drop everything this module has cached, so a bad download can be retried. */
export async function evict(urls) {
  if (typeof caches === 'undefined') return;
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  if (!cache) return;
  for (const url of urls) {
    if (url) await cache.delete(url, { ignoreVary: true }).catch(() => {});
  }
}
