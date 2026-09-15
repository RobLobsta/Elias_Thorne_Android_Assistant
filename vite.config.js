import { copyFile, cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(root, 'public');
const outDir = resolve(root, 'dist');

/**
 * Deployment base path, e.g. "/Elias_Thorne_Android_Assistant/" for a GitHub
 * Pages project site. Everything the app addresses by absolute URL — the
 * manifest, the service worker and its scope, the staged ONNX runtime, the
 * model directory — is derived from this, so the same build works at a
 * subpath or at a domain root.
 */
const BASE = normaliseBase(process.env.BASE_PATH ?? '/');

function normaliseBase(value) {
  const trimmed = String(value).trim();
  if (!trimmed || trimmed === '/') return '/';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}/`;
}

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
  { path: 'manifest.json', pattern: /\/?(?:[\w.-]+\/)*assets\/manifest-[A-Za-z0-9_-]+\.json/g },
  { path: 'icons/icon-192.png', pattern: /\/?(?:[\w.-]+\/)*assets\/icon-192-[A-Za-z0-9_-]+\.png/g },
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
      await rebaseManifest();
      await injectPrecacheList();
      // GitHub Pages runs Jekyll on branch-based publishes, which drops files
      // and directories beginning with an underscore.
      await writeFile(resolve(outDir, '.nojekyll'), '');

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

  for (const { path, pattern } of STABLE_URLS) {
    html = html.replace(pattern, (match) => {
      const relative = match.replace(/^\//, '').replace(BASE.replace(/^\//, ''), '');
      orphans.add(relative);
      return BASE + path;
    });
  }

  await writeFile(indexPath, html);
  for (const orphan of orphans) {
    await rm(resolve(outDir, orphan), { force: true });
  }
}

/**
 * The web app manifest is authored with root-absolute URLs, which is correct
 * for dev and for a domain-root deploy. Rebase them when building for a
 * subpath — a manifest whose scope does not cover start_url makes the app
 * uninstallable, and Bubblewrap reads these same values.
 */
async function rebaseManifest() {
  if (BASE === '/') return;
  const manifestPath = resolve(outDir, 'manifest.json');
  if (!existsSync(manifestPath)) return;

  const rebase = (value) => {
    if (typeof value === 'string') {
      return value.startsWith('/') && !value.startsWith('//')
        ? BASE.replace(/\/$/, '') + value
        : value;
    }
    if (Array.isArray(value)) return value.map(rebase);
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, rebase(v)]));
    }
    return value;
  };

  const manifest = rebase(JSON.parse(await readFile(manifestPath, 'utf8')));
  manifest.id = BASE;
  manifest.start_url = BASE;
  manifest.scope = BASE;
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

/**
 * Write the emitted asset filenames into the service worker's precache list.
 *
 * The service worker is a plain static file, so it cannot know the hashed
 * names Rollup produced. Without this the first offline load renders unstyled.
 */
async function injectPrecacheList() {
  const swPath = resolve(outDir, 'sw.js');
  const assetsDir = resolve(outDir, 'assets');
  if (!existsSync(swPath) || !existsSync(assetsDir)) return;

  const assets = (await readdir(assetsDir))
    .filter((file) => !file.endsWith('.map'))
    .map((file) => `${BASE}assets/${file}`)
    .sort();

  const sw = await readFile(swPath, 'utf8');
  const injected = sw.replace(
    /const BUILD_ASSETS = \[\];/,
    `const BUILD_ASSETS = ${JSON.stringify(assets)};`,
  );
  if (injected === sw) {
    this?.warn?.('Could not find the BUILD_ASSETS placeholder in sw.js.');
  }
  await writeFile(swPath, injected);
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
  base: BASE,
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
