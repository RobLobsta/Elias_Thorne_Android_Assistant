/*
 * Service worker — offline shell and weight caching (SAD §2.2).
 *
 * Elias is an on-device agent, so "works with the radio off" is a requirement,
 * not a nicety. Two caching policies:
 *
 *   - App shell: stale-while-revalidate. The UI comes back instantly and
 *     updates itself in the background.
 *   - Weights, tokenizer and the ONNX runtime binaries: cache-first and never
 *     revalidated. These are hundreds of megabytes and immutable for a given
 *     filename; re-checking them on every boot would burn the user's data.
 *
 * Cross-origin requests (the sandbox's ESM imports, tool fetches) are passed
 * straight through. They are throttled and allow-listed in Worker B, and
 * caching third-party code here would quietly extend its lifetime.
 */

const VERSION = 'v1';
const SHELL_CACHE = `elias-shell-${VERSION}`;
// Deliberately unversioned: weights survive app updates. Bump only on a format
// change, and expect a full re-download when you do.
const MODEL_CACHE = 'elias-models';

const SHELL_ASSETS = ['/', '/index.html', '/manifest.json', '/icons/icon-192.png'];

/** Immutable, large, cache-first. */
const IMMUTABLE_PREFIXES = ['/models/', '/ort/'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .catch(() => {})
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('elias-shell-') && key !== SHELL_CACHE)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (IMMUTABLE_PREFIXES.some((prefix) => url.pathname.startsWith(prefix))) {
    event.respondWith(cacheFirst(request, MODEL_CACHE));
    return;
  }

  event.respondWith(staleWhileRevalidate(request, SHELL_CACHE));
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request, { ignoreVary: true });
  if (cached) return cached;

  const response = await fetch(request);
  // A partial response cannot be replayed as a whole one; hand it back unstored.
  if (response.ok && response.status === 200) {
    cache.put(request, response.clone()).catch(() => {
      // Quota exhausted mid-download: serving the response still succeeds.
    });
  }
  return response;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request, { ignoreVary: true });

  const network = fetch(request)
    .then((response) => {
      if (response.ok && response.status === 200) {
        cache.put(request, response.clone()).catch(() => {});
      }
      return response;
    })
    .catch(() => null);

  if (cached) return cached;

  const response = await network;
  if (response) return response;

  // Offline with nothing cached: fall back to the shell for navigations.
  if (request.mode === 'navigate') {
    const shell = await cache.match('/index.html', { ignoreVary: true });
    if (shell) return shell;
  }
  return new Response('Offline and not cached.', { status: 503, statusText: 'Offline' });
}

self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') self.skipWaiting();
});
