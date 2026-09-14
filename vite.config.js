import { copyFile, cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(root, 'public');
const outDir = resolve(root, 'dist');

// The SAD blueprint places index.html inside public/, so public/ is the Vite
// root rather than its static directory. These assets therefore need copying
// by hand instead of relying on Vite's implicit publicDir behaviour.
const STATIC_ASSETS = ['manifest.json', 'sw.js', 'icons', 'models', 'ort'];

/**
 * Paths the service worker, the web app manifest and Bubblewrap all reference
 * by their literal URL. Vite fingerprints anything it finds referenced from
 * index.html, which would break those references, so the stable names are
 * restored after the bundle is written and the fingerprinted duplicates are
 * removed.
 */
const STABLE_URLS = [
  { url: '/manifest.json', pattern: /\/assets\/manifest-[A-Za-z0-9_-]+\.json/g },
  { url: '/icons/icon-192.png', pattern: /\/assets\/icon-192-[A-Za-z0-9_-]+\.png/g },
];

/**
 * ONNX Runtime is staged into public/ort and loaded from there at runtime
 * rather than bundled.
 *
 * Bundling it drags a 28 MB WASM binary through Rollup on every build for no
 * benefit: the runtime resolves its binaries from `ort.env.wasm.wasmPaths` at
 * execution time regardless. Staging keeps the agent genuinely offline-capable
 * (no CDN on first inference) and the build fast.
 */
const ORT_ENTRY = 'ort.webgpu.bundle.min.mjs';

function stageOnnxRuntime() {
  return {
    name: 'elias-stage-ort',
    async buildStart() {
      const from = resolve(root, 'node_modules', 'onnxruntime-web', 'dist');
      const to = resolve(appRoot, 'ort');
      if (!existsSync(from)) {
        this.warn('onnxruntime-web is not installed; run npm install first.');
        return;
      }

      const entry = await readFile(resolve(from, ORT_ENTRY), 'utf8');
      // onnxruntime-web ships eight WASM variants totalling ~83 MB and this
      // build needs exactly one of them. Which one is read off the entry
      // bundle rather than hard-coded, so an ORT upgrade that switches variant
      // does not silently stage the wrong binary.
      const referenced = new Set(entry.match(/ort-wasm-[a-z0-9.\-]*\.(?:wasm|mjs)/g) ?? []);
      if (referenced.size === 0) {
        this.warn(`Could not determine which WASM binary ${ORT_ENTRY} needs.`);
      }

      await mkdir(to, { recursive: true });
      for (const file of [ORT_ENTRY, ...referenced]) {
        const source = resolve(from, file);
        if (existsSync(source)) await copyFile(source, resolve(to, file));
      }
    },
  };
}

function copyStaticAssets() {
  return {
    name: 'elias-copy-static',
    apply: 'build',
    async closeBundle() {
      await mkdir(outDir, { recursive: true });
      for (const asset of STATIC_ASSETS) {
        const from = resolve(appRoot, asset);
        if (!existsSync(from)) continue;
        await cp(from, resolve(outDir, asset), { recursive: true, force: true });
      }

      await restoreStableUrls();

      const models = resolve(outDir, 'models');
      if (existsSync(models)) {
        const weights = (await readdir(models)).filter((f) => f.endsWith('.onnx'));
        if (weights.length === 0) {
          this.warn(
            'No .onnx weights in public/models — Elias will boot on the fallback ' +
              'reasoner. See public/models/README.md.',
          );
        }
      }
    },
  };
}

async function restoreStableUrls() {
  const indexPath = resolve(outDir, 'index.html');
  if (!existsSync(indexPath)) return;

  let html = await readFile(indexPath, 'utf8');
  const orphans = new Set();

  for (const { url, pattern } of STABLE_URLS) {
    html = html.replace(pattern, (match) => {
      orphans.add(match.replace(/^\//, ''));
      return url;
    });
  }

  await writeFile(indexPath, html);
  for (const orphan of orphans) {
    await rm(resolve(outDir, orphan), { force: true });
  }
}

// WebGPU does not require cross-origin isolation, but the ONNX Runtime WASM
// fallback needs SharedArrayBuffer for threading. `credentialless` keeps the
// esm.sh dynamic imports in Worker B working while still isolating the page.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  root: appRoot,
  publicDir: false,
  base: '/',
  plugins: [stageOnnxRuntime(), copyStaticAssets()],
  server: {
    host: true,
    headers: isolationHeaders,
    // src/ lives beside the Vite root, not inside it.
    fs: { allow: [root] },
  },
  preview: {
    host: true,
    headers: isolationHeaders,
  },
  worker: {
    format: 'es',
  },
  build: {
    outDir,
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: resolve(appRoot, 'index.html'),
    },
  },
});
