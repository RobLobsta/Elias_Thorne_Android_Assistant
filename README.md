# Elias Thorne

An autonomous, on-device agent. A BitNet 2B4T ternary model runs through ONNX
Runtime Web on WebGPU, memory lives in a local Orama index, and the tools Elias
writes for himself execute in a sandboxed worker. Nothing leaves the handset.

Built to the [Software Architecture Document](Software_Architecture_Document.md);
section references below point back at it.

## Status

| Capability | State |
| --- | --- |
| Wake word, interruption, audio state machine (§2.1) | working |
| Acoustic echo cancellation constraints (§2.1) | working |
| Orama memory with persistent storage lock (§2.2) | working |
| JIT tool creation, linting, sandboxed execution (§2.3) | working |
| Self-improvement loop with pre-commit telemetry (§2.3) | working |
| PWA shell, offline caching, TWA config (§3) | working |
| Installable APK from GitHub Actions | working |
| BitNet inference on WebGPU (§2.2) | working — needs weights, see below |

A 2B parameter export is a gigabyte-class file. It cannot be committed, it
cannot be served from a project Pages site (1 GB per site, 100 MB per file), and
the APK is a Trusted Web Activity that carries no web assets at all — so the
weights are never part of the build. There are two ways to supply them and one
fallback:

1. **Build them in.** Drop the export into `public/models` before
   `npm run build`. Needs a deployment that can serve a file that size.
2. **Download them on demand.** Set `weightsUrl` in
   `public/models/model-config.json` to an export on a CORS-enabled host, and
   Elias offers the download from a banner in the UI. Nothing is fetched until
   the user presses the button — a phone on mobile data must not lose a gigabyte
   to an app starting up — and the bytes are kept in Cache Storage afterwards,
   so it is a one-time cost that survives app updates.
3. **Neither.** Elias boots on a **fallback reasoner**: a deterministic,
   non-neural responder that emits the same action schema the real model does,
   so the wake word, telemetry, linter, sandbox and memory are all genuinely
   exercised end to end. The backend pill reads `fallback reasoner` and the
   banner says why.

Local files win over the remote source, and the build prints which of the three
you are shipping.

See [`public/models/README.md`](public/models/README.md) for the config keys and
how to make `model-config.json` match your export.

### It does not have to be BitNet

There is no official BitNet 2B4T ONNX export — Microsoft publishes safetensors
and GGUF only — so `weightsUrl` ships empty. But the graph runner is not
BitNet-specific: it introspects whatever decoder-only ONNX you give it. The
exports under Hugging Face's [`onnx-community`](https://huggingface.co/onnx-community)
org already match what it expects, and they come in browser-sized
quantisations.

This was verified end to end with SmolLM2-360M-Instruct (`q4f16`, 261 MB):

```json
{
  "modelFile": "model.onnx",
  "weightsUrl": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/onnx/model_q4f16.onnx",
  "tokenizerUrl": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json",
  "numHiddenLayers": 32,
  "numKeyValueHeads": 5,
  "headDim": 64,
  "template": {
    "bos": "",
    "system": "<|im_start|>system\n{content}<|im_end|>\n",
    "user": "<|im_start|>user\n{content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
    "generationPrefix": "<|im_start|>assistant\n",
    "stopTokens": ["<|im_end|>", "<|endoftext|>"]
  }
}
```

Note the `template` block: SmolLM2 is ChatML, not Llama-3. Swapping those four
strings is the whole of what a differently-templated model needs.

**Speed.** That run took ~50 s per reply on four desktop cores, because ONNX
Runtime fell back to single-threaded WASM. WASM threads need
`crossOriginIsolated`, which needs COOP/COEP response headers, which GitHub
Pages cannot set. On a handset the path that matters is WebGPU — which Chrome
on Android has, and which the engine prefers when present — but that was not
measurable in this environment, since it has no GPU. Treat the CPU number as a
floor, not as what a phone will do.

## Quick start

```bash
npm install
npm run icons     # regenerate the PWA icon set (already committed)
npm run dev       # http://localhost:5173
npm test          # dependency-free logic checks
npm run build     # -> dist/
```

`npm run dev` and `npm run build` both stage the ONNX Runtime WASM binary into
`public/ort/` from `node_modules`, so inference works with the radio off.

Speech recognition needs a secure context. `localhost` counts; a LAN IP does
not, so test voice over HTTPS or through port forwarding.

## Layout

```
android/twa-manifest.json      Bubblewrap TWA configuration
public/
  index.html                   app shell (also the Vite root)
  manifest.json                PWA descriptors
  sw.js                        offline shell + weight caching
  icons/                       generated PWA/launcher icons
  models/                      model-config.json; weights + tokenizer if bundled (gitignored)
  ort/                         staged ONNX Runtime binaries (gitignored)
src/
  main.js                      UI orchestrator, audio state machine, TTS
  styles.css                   visual display
  workers/
    inference.worker.js        Worker A: BitNet, Orama, the agentic loop
    sandbox.worker.js          Worker B: linting and JIT tool execution
  utils/
    audio.js                   mic constraints, AEC, VAD
    storage.js                 persistent storage lock, IndexedDB, idle saves
    persona.js                 identity lock, wake anchors, system persona
    schema.js                  action schema, message contracts, allow-lists
    tokenizer.js               byte-level BPE + chat templating
    weights.js                 where the weights live, and fetching them
scripts/
  check.mjs                    logic checks (npm test)
  generate-icons.mjs           icon generation (npm run icons)
```

`persona.js`, `schema.js` and `tokenizer.js` are additions to the SAD's
blueprint. Each exists because two threads need the same definition and
duplicating it would let them drift: the UI thread and Worker A must agree on
the wake anchors, both workers must agree on the action schema, and the
tokenizer is large enough that inlining it would bury the inference logic.

## How a turn works

1. **Wake.** One continuous `SpeechRecognition` feeds a router. In `[Listening]`
   it matches the anchors `Hey Elias`, `Elias`, `Thorne!`; anything the speaker
   says after the anchor becomes the command, so a single breath works.
2. **Capture.** Interim results update the caption; a final result or 6 s of
   silence commits the utterance.
3. **Recall.** Worker A queries Orama in hybrid mode (BM25 + vector) and injects
   the hits ahead of the user turn.
4. **Generate.** BitNet streams tokens to the UI as they decode.
5. **Act.** If the completion contains a JSON action, it is validated. Whatever
   happens next — registering a tool, patching one, running a self-correction
   pass — a `TELEMETRY_LOG` is posted to the UI **before** anything commits.
6. **Execute.** `invoke_tool` goes over the MessageChannel to Worker B; the
   result comes back and is fed to the model for a spoken summary.
7. **Repair.** A failing tool re-enters the loop with the lint output or runtime
   error attached, up to three correction passes, each one announced first.
8. **Speak.** The reply is spoken one sentence per utterance, which bounds how
   long "stop" takes to land.

### Interruption

While `[Speaking]`, the recognition router becomes the interruption gate from
§2.1: it looks at **interim** results only, matches nothing but `stop` and
`wait`, and never consults the model. On a match it calls
`speechSynthesis.cancel()`, clears the utterance queue, cancels generation, and
rolls back to `[Listening]`.

The SAD describes a separate listener for this. Android backs every
`SpeechRecognition` with one shared speech service, and a second concurrent
instance either fails to start or steals the first one's audio — so the gate is
implemented as a state-dependent router over the single recogniser instead. The
behaviour is what §2.1 specifies; the instance count is not.

Hardware AEC is what makes any of this possible: without the platform
subtracting speaker output from mic input, the microphone hears Elias's own
voice. `src/utils/audio.js` requests it explicitly and reports in the telemetry
box whether the device actually applied it. Auto gain control is deliberately
off — it ramps mic gain up during the pauses between sentences, which both
floods the VAD and undermines the echo canceller's gain model.

## The sandbox

Worker B runs tool bodies through a scoped `AsyncFunction` constructor with
globals shadowed and a small capability object (`ctx.fetch`, `ctx.memory`,
`ctx.deps`, `ctx.log`, `ctx.signal`, `ctx.env`) as the only way out.

**What the boundary actually is.** The real boundary is the dedicated worker:
no DOM, no page handle, no same-origin session, no storage handle. Code that
defeats the identifier shadowing still has none of those things, because they
are not present in the worker's scope. The shadowing and the linter are defence
in depth that make accidental misuse loud; they are not a claim of in-realm
escape-proofing, which JavaScript cannot provide. What genuinely matters —
network reach, storage reach, and runtime — is mediated:

- **Network.** HTTPS only, host allow-list, `GET`/`POST`/`HEAD` only, credentials
  omitted, redirects re-validated against the allow-list, 2 MB response cap,
  15 s timeout, and a token bucket of 8 requests per 10 s with a concurrency of 2.
- **Dependencies.** Dynamic `import()` from `esm.sh`, jsDelivr, Skypack or unpkg
  only, resolved once and cached, surfaced on `ctx.deps` keyed by package name.
- **Storage.** No database handle. `ctx.memory.search` / `.insert` are proxied to
  Worker A over the MessageChannel.
- **Runtime.** A tool that overruns its budget poisons the sandbox, which then
  refuses further work and asks the UI thread to terminate and respawn it —
  only the creating thread can terminate a worker, which is why `main.js` owns
  both worker lifecycles rather than Worker A owning Worker B.

Lint failures are returned with a line number and a remediation sentence, which
is what makes the self-improvement loop produce a targeted fix rather than a
reroll.

## Memory

Orama holds the index; snapshots are written to IndexedDB during idle windows
(`requestIdleCallback`, forced on `pagehide`/`visibilitychange`), so persistence
never contends with the inference loop. `navigator.storage.persist()` is
requested at boot to stop Android's cache cleanup evicting the index or the
cached weights — the storage pill shows whether the platform granted it.

Recall is hybrid: BM25 for lexical precision, plus a 256-dimension hashed
bag-of-words vector for fuzzy neighbourhood. A second neural model purely for
embeddings would double the memory budget on a phone; the hashing embedding is
deterministic, free, and swappable for a real encoder in one function
(`embed()` in `inference.worker.js`) if you have the headroom.

## Getting it onto a phone

A Trusted Web Activity is a Chrome tab in an app shell — the APK carries no web
assets, only the URL to open. So the site has to be published before an APK can
exist, and everything below assumes that order.

Three routes, cheapest first.

### 1. No APK at all

Open the published site in Chrome on the phone and use **Add to home screen**.
You get a standalone window, the microphone, the wake word and offline caching.
This is the fastest way to check a change on real hardware, and for day-to-day
testing it is usually enough.

### 2. GitHub Actions (this repo)

`.github/workflows/pages-and-apk.yml` publishes the site to GitHub Pages and
builds a signed APK from it on every push to `main`, or on demand from the
Actions tab.

One-time setup: **Settings → Pages → Source → GitHub Actions**. Nothing else —
no accounts, no secrets.

Each run leaves an **elias-thorne-apk** artifact on the run summary. Download,
unzip and install:

```bash
adb install -r app-release-signed.apk
```

Or open the artifact link on the phone and install from Chrome, once
"install unknown apps" is allowed for the browser.

By default the workflow generates a throwaway signing key per run, so the
signature changes every build and Android will refuse to upgrade in place —
`adb uninstall ai.eliasthorne.assistant` first. To keep one stable key, add
these repository secrets and the workflow uses them instead:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | `base64 -w0 android.keystore` |
| `ANDROID_KEYSTORE_PASSWORD` | the store password |
| `ANDROID_KEY_PASSWORD` | the key password (alias `elias`) |

### 3. Locally

Needs a JDK; Bubblewrap downloads its own JDK 17 and Android SDK (~1.2 GB) on
first run.

```bash
BASE_PATH=/Elias_Thorne_Android_Assistant/ npm run build
npx vite preview --port 4173          # serves the bundle Bubblewrap reads

mkdir -p build/twa && cp android/twa-manifest.json build/twa/
cd build/twa
npx @bubblewrap/cli update --skipVersionUpgrade
npx @bubblewrap/cli build
```

`update --skipVersionUpgrade` generates the whole Android project from
`twa-manifest.json` without prompting. Plain `update` stops to ask for a version
name, and `init` is fully interactive — neither works unattended.

### Three things that will bite

**Microphone permission.** Bubblewrap does not add `RECORD_AUDIO`, and without
it `getUserMedia` fails inside the TWA with no visible error — the wake word
just never fires. The workflow patches the generated manifest; if you build by
hand, add this to `build/twa/app/src/main/AndroidManifest.xml` before building:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-feature android:name="android.hardware.microphone" android:required="true" />
```

**Android SDK licences.** Gradle fails at `:app:minifyReleaseWithR8` with
"licences have not been accepted" unless they are accepted first. The workflow
runs `sdkmanager --licenses`; locally, `yes | sdkmanager --licenses` once.

**The URL bar.** Removing it needs `/.well-known/assetlinks.json` served from
the *domain root* with the signing key's fingerprint. On a project Pages site
the root is `roblobsta.github.io`, which this repository cannot publish to — so
the TWA shows a thin URL bar. It is cosmetic and does not affect the
microphone, the wake word or anything else. To get rid of it, serve the app
from a root domain (or a `roblobsta.github.io` user-site repo) and publish the
`assetlinks.json` the workflow emits alongside the APK.

### Deploying elsewhere

The build is base-path aware. `BASE_PATH` sets where it will be served:

```bash
BASE_PATH=/ npm run build                   # domain root
BASE_PATH=/some/prefix/ npm run build       # subpath
```

It rewrites the Vite base, the manifest's `id`/`start_url`/`scope`/icons, and
the service worker's precache list; the worker reads its own registration scope
at runtime, and the model and ONNX runtime directories hang off `BASE_URL`.

If the host can set headers, send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

`credentialless` rather than `require-corp` so the sandbox's ESM imports keep
working. The dev and preview servers already send both. GitHub Pages cannot set
headers, so the page is not cross-origin isolated there: WebGPU — the primary
inference path — is unaffected, but the ONNX WASM fallback loses
`SharedArrayBuffer` and runs single-threaded.

## Verifying

`npm test` runs 30 dependency-free checks over the wake anchors, the
interruption gate, the persona identity lock, the action schema and the
resolution of weight sources.

Worker-level behaviour was verified by driving the built app in Chromium. If you
want to reproduce it, install `playwright` and script these against
`npm run preview`:

- **Agent loop.** Ask "build a tool that adds numbers together", then "add 12 and
  30 and 5". Expect a tool in the registry, telemetry posted before the commit,
  a sandbox log from inside the tool, and the answer "That comes to 47 across 3
  values."
- **Persistence.** Reload. The tool and the memories survive.
- **Weight provisioning.** Verified in Chromium against a purpose-built tiny
  decoder ONNX graph (`input_ids` + `attention_mask` + `position_ids`, two KV
  layers, a `logits` head) served from a second origin. All four cases: weights
  bundled in `public/models`; a remote `weightsUrl` offered, downloaded with
  progress, compiled, and still cached after a reload; the same with external
  weight data in a sidecar file; and neither, falling back with the banner
  explaining why. Plus the failure paths — a corrupt graph and a missing sidecar
  both surface a readable message, keep Elias on the fallback reasoner, and
  evict the bad cache entry so Retry starts clean.
- **Sandbox.** Bind a `MessageChannel` to `sandbox.worker.js` directly and check
  that lint rejects `globalThis` and bare `fetch` and reports syntax errors;
  that a failing spec fails the run; that `ctx.fetch` to a non-allow-listed host
  and a non-allow-listed dependency are both refused; and that a hanging tool
  times out, marks the sandbox poisoned and demands a respawn.

The APK toolchain was run end to end and the resulting APK inspected with
`aapt2` and `apksigner`: package `ai.eliasthorne.assistant`, `RECORD_AUDIO` and
`android.hardware.microphone` present, the launch URL baked in as
`https://roblobsta.github.io/Elias_Thorne_Android_Assistant/`, and a valid
signature. The subpath build was driven in Chromium too — full agent loop,
service worker registered at the right scope, and a styled render after an
offline reload.

Not verified here: a successful live import from `esm.sh`, the fetch throttle's
timing, and the workflow running on GitHub's own runners. The sandbox this was
built in proxies egress through a TLS interceptor that its Chromium build does
not trust, so every browser request to a CDN fails at the TLS layer. The
allow-list rejection path and the import failure path are both verified; the
success path is not. Every step of the workflow was rehearsed locally against
the same commands, but the first Actions run is still the first Actions run.

## Known limitations

- **Weights.** See Status. The graph runner has now been run against a real
  instruct model (SmolLM2-360M, `q4f16`) as well as synthetic float32 and
  float16 graphs. No *BitNet* export has been through it, because none exists in
  ONNX — so `weightsUrl` ships empty and the headline claim of a ternary model
  on WebGPU is still unproven. Note also that ONNX Runtime Web has no ternary
  kernel: an ONNX BitNet would be stored and executed densely, so it would carry
  the size and the cost of any other 2B model.
- **Wake word cost.** Continuous `SpeechRecognition` on Android is
  network-backed and battery-hungry, and the platform ends sessions on its own
  schedule (there is a restart supervisor with backoff). A real always-on wake
  word wants a small on-device keyword spotter.
- **Fallback tool routing** matches tool names against the utterance by
  four-character word stems. It is crude by design — it only has to demonstrate
  routing without a language model.
- **Interruption latency** is bounded by how quickly the platform delivers an
  interim result, which varies by device.
