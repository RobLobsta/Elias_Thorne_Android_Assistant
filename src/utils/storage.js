/**
 * IndexedDB state resiliency service (SAD §2.2).
 *
 * Two jobs:
 *   1. Take out a persistent storage lock via navigator.storage.persist() so
 *      that Android's cache cleanup cannot evict Elias's memory or his cached
 *      BitNet weights.
 *   2. Provide a dependency-free IndexedDB key/value store plus an idle-time
 *      autosaver, so Orama's in-memory arrays are flushed to disk whenever the
 *      device is quiet rather than on a timer that fights the inference loop.
 *
 * Runs unchanged on the UI thread and inside workers — nothing here touches the
 * DOM.
 */

const DB_NAME = 'elias-thorne';
const DB_VERSION = 1;

/** Object stores: conversational state, Orama snapshots, the tool registry. */
export const STORES = Object.freeze({
  STATE: 'state',
  MEMORY: 'memory',
  TOOLS: 'tools',
});

let dbPromise = null;

function openDatabase() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is unavailable in this context.'));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      for (const store of Object.values(STORES)) {
        if (!db.objectStoreNames.contains(store)) db.createObjectStore(store);
      }
    };
    request.onsuccess = () => {
      const db = request.result;
      // A version change from another tab must not leave us holding a stale
      // handle; drop the cache and let the next call reopen.
      db.onversionchange = () => {
        db.close();
        dbPromise = null;
      };
      resolve(db);
    };
    request.onerror = () => reject(request.error ?? new Error('IndexedDB open failed.'));
    request.onblocked = () => reject(new Error('IndexedDB upgrade blocked by another tab.'));
  }).catch((error) => {
    dbPromise = null;
    throw error;
  });
  return dbPromise;
}

function transact(storeName, mode, operation) {
  return openDatabase().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(storeName, mode);
        const store = tx.objectStore(storeName);
        let result;
        try {
          result = operation(store);
        } catch (error) {
          tx.abort();
          reject(error);
          return;
        }
        tx.oncomplete = () => resolve(result instanceof IDBRequest ? result.result : result);
        tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed.'));
        tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted.'));
      }),
  );
}

export function idbGet(store, key) {
  return transact(store, 'readonly', (s) => s.get(key));
}

export function idbSet(store, key, value) {
  return transact(store, 'readwrite', (s) => s.put(value, key));
}

export function idbDelete(store, key) {
  return transact(store, 'readwrite', (s) => s.delete(key));
}

export function idbKeys(store) {
  return transact(store, 'readonly', (s) => s.getAllKeys());
}

export function idbAll(store) {
  return transact(store, 'readonly', (s) => s.getAll());
}

export function idbClear(store) {
  return transact(store, 'readwrite', (s) => s.clear());
}

/**
 * Request the persistent storage lock and report what the platform granted.
 *
 * Chrome on Android grants persistence silently to installed PWAs and to
 * origins with sufficient engagement, so this is called once at boot and again
 * after install; a denial is not fatal, it just means eviction is possible.
 *
 * @returns {Promise<{persisted: boolean, requested: boolean, quota: number, usage: number}>}
 */
export async function requestPersistentLock() {
  const report = { persisted: false, requested: false, quota: 0, usage: 0 };
  if (typeof navigator === 'undefined' || !navigator.storage) return report;

  try {
    if (navigator.storage.persisted) {
      report.persisted = await navigator.storage.persisted();
    }
    if (!report.persisted && navigator.storage.persist) {
      report.requested = true;
      report.persisted = await navigator.storage.persist();
    }
    if (navigator.storage.estimate) {
      const { quota = 0, usage = 0 } = await navigator.storage.estimate();
      report.quota = quota;
      report.usage = usage;
    }
  } catch {
    // Permissions policy or a private-browsing context: report what we have.
  }
  return report;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

const requestIdle =
  typeof requestIdleCallback === 'function'
    ? requestIdleCallback
    : (cb) => setTimeout(() => cb({ didTimeout: true, timeRemaining: () => 0 }), 1);
const cancelIdle = typeof cancelIdleCallback === 'function' ? cancelIdleCallback : clearTimeout;

/**
 * Coalesces writes and flushes them while the device is idle.
 *
 * `schedule()` is cheap and may be called on every mutation; the flush itself
 * runs at most once per idle window, never concurrently with itself, and is
 * forced immediately when the page is backgrounded or torn down.
 */
export class IdleAutosaver {
  #flush;
  #timeout;
  #handle = null;
  #running = false;
  #dirty = false;
  #listenersBound = false;

  /**
   * @param {() => Promise<void>} flush the persistence routine
   * @param {{timeout?: number}} [options] max wait before forcing a flush
   */
  constructor(flush, { timeout = 4000 } = {}) {
    this.#flush = flush;
    this.#timeout = timeout;
    this.#bindLifecycle();
  }

  schedule() {
    this.#dirty = true;
    if (this.#handle != null) return;
    this.#handle = requestIdle(() => {
      this.#handle = null;
      void this.flushNow();
    }, { timeout: this.#timeout });
  }

  async flushNow() {
    if (this.#running) {
      this.#dirty = true;
      return;
    }
    if (!this.#dirty) return;
    this.#running = true;
    this.#dirty = false;
    try {
      await this.#flush();
    } catch (error) {
      // Keep the dirty flag so the next idle window retries rather than
      // silently dropping the write.
      this.#dirty = true;
      console.warn('[storage] idle flush failed:', error);
    } finally {
      this.#running = false;
      if (this.#dirty) this.schedule();
    }
  }

  dispose() {
    if (this.#handle != null) cancelIdle(this.#handle);
    this.#handle = null;
  }

  #bindLifecycle() {
    if (this.#listenersBound || typeof addEventListener !== 'function') return;
    this.#listenersBound = true;
    const force = () => {
      if (this.#handle != null) {
        cancelIdle(this.#handle);
        this.#handle = null;
      }
      void this.flushNow();
    };
    addEventListener('pagehide', force);
    addEventListener('freeze', force);
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') force();
      });
    }
  }
}
