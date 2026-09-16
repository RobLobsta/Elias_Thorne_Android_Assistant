# Elias Thorne — native Android

A BitNet b1.58-2B-4T assistant that runs entirely on the handset. No server, no
API key, and after the model is installed, no network.

This is the native counterpart to the PWA in the repository root. That version
cannot run BitNet: ONNX Runtime Web has no ternary kernel, and there is no
BitNet ONNX export to give it. This one links bitnet.cpp directly.

## Why native, in one number

```
PEAK_RSS_MB=1363   RssAnon_MB=223   RssFile_MB=1139
```

llama.cpp `mmap`s the GGUF, so ~1.14 GB of the model stays **file-backed and
evictable** and only ~223 MB is dirty anonymous memory. On a 4 GB phone that is
the difference between running and being killed. A browser cannot do this —
WebAssembly has only its linear heap, so the same model would be 1.2 GB of
un-evictable RAM inside the renderer.

## Build

```bash
git submodule update --init --recursive     # pulls bitnet.cpp + its llama.cpp
export ANDROID_HOME=/path/to/android-sdk    # needs NDK 27.2.12479018
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The first build compiles the whole of llama.cpp for `arm64-v8a` and takes
several minutes. Only arm64 is built — every phone worth running a 2B model on
is arm64, and each extra ABI is another full native build.

## Getting the model onto the phone

The GGUF is ~1.2 GB, so it is not in the APK. Download it once, **patch it**,
then push it:

```bash
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \
    ggml-model-i2_s.gguf --local-dir .

python3 tools/patch_gguf_pretokenizer.py ggml-model-i2_s.gguf model.gguf

adb push model.gguf /sdcard/Download/
```

Then open the app and pick the file. It is copied into the app's own storage
(so it can be `mmap`ed) and loaded on every launch after that.

### The patch is not optional

Microsoft's GGUF omits `tokenizer.ggml.pre`. llama.cpp says so and carries on
with the wrong pre-tokenizer:

```
load: missing pre-tokenizer type, using: 'default'
load: GENERATION QUALITY WILL BE DEGRADED!
```

BitNet uses the Llama-3 tokenizer. With the default one every prompt is
mis-split, so the model sees sequences unlike anything it was trained on. The
weights are fine; one metadata string is missing. `tools/patch_gguf_pretokenizer.py`
adds it. The `gguf` Python package cannot open the file at all — it rejects
tensor type 36, bitnet.cpp's `I2_S` sharing an id with a quant mainline removed
— so the script walks the container itself.

## Two things the GGUF gets wrong, handled in code

- **The chat template.** The one embedded in the file renders
  `Human: … BITNETAssistant: …`, which is not what the model was trained on and
  produces fluent nonsense that loops. The real template, from the model repo's
  `tokenizer_config.json`, is `Role: content<|eot_id|>` with `Assistant: ` as
  the generation prompt. `elias_core.cpp` renders that itself and ignores the
  file's version. Any *other* GGUF's template is used as declared, so a standard
  instruct model can be dropped in unchanged.
- **The end-of-turn token.** `<|eot_id|>` (128009) is not nominated as an EOS
  token, so `llama_vocab_is_eog` never fires and the model runs past its answer.
  `resolve_stop_tokens` looks the literal up by name instead.

## Layout

```
app/src/main/cpp/
  elias_core.{h,cpp}   model, sampler, generation loop — no JNI, no Android
  elias_jni.cpp        the JNI surface, marshalling only
  test_cli.cpp         host harness: same core, runnable on a desktop
  CMakeLists.txt       links bitnet.cpp statically into libelias.so
  bitnet/              submodule: microsoft/BitNet
app/src/main/java/ai/eliasthorne/assistant/
  MainActivity.kt      the turn loop and UI
  LlamaBridge.kt       Kotlin side of the JNI boundary
  Voice.kt             SpeechRecognizer in, TextToSpeech out
  ModelStore.kt        where the GGUF lives, and importing one
tools/
  patch_gguf_pretokenizer.py
```

`elias_core` is deliberately free of both JNI and Android so the same generation
path can be run on a desktop:

```bash
clang++ -O2 -std=c++17 -o elias-test \
  app/src/main/cpp/elias_core.cpp app/src/main/cpp/test_cli.cpp \
  -Iapp/src/main/cpp -I<bitnet>/3rdparty/llama.cpp/include \
  -I<bitnet>/3rdparty/llama.cpp/ggml/include \
  -L<bitnet>/build/bin -lllama -lggml -lggml-base -lggml-cpu

./elias-test model.gguf "What is the capital of France?"
```

## If it is slow

**Start here: is the model small enough to stay in memory?** This dominates
everything else, and it is the opposite of what the "Why native" section above
implies. llama.cpp maps the weights from the file, which keeps resident memory
low precisely *because* the kernel is free to drop those pages. Generation reads
the whole model once per token, so on a device that is always short of memory
every token waits on flash. Measured with identical code, model and machine,
back to back:

| Page cache | Prompt eval | Generation |
| ---------- | ----------- | ---------- |
| cold       | 2.7 tok/s   | **2.0 tok/s**  |
| warm       | 106.3 tok/s | **21.9 tok/s** |

A phone with 4 GB total never gets to stay warm with a 1.2 GB model, so it runs
permanently in the top row. The same harness on a 469 MB model — small enough to
stay resident — gives **46 tok/s**. That is the fix: pick a model that fits in
free RAM, not one that merely loads.

The app now measures free memory at load and says so plainly when the model is
too big for the device. Long-press the status line to copy the full diagnostics.
The number to look at is the split: if prompt evaluation is fast and only
generation is slow, the build is fine and the device cannot hold the weights.

Two build settings also matter, and both defaulted the wrong way:

- **The Android Gradle plugin compiles a debug variant's native code at `-O0`.**
  `cppFlags += "-O3"` in `build.gradle.kts` applies only to the app's own two
  source files, not to the ggml subproject where all the arithmetic is. This
  measured **0.3 tok/s on device**. The CMakeLists now appends `-O3 -DNDEBUG` to
  the debug flags, which has to be done by appending to the *normal* variable:
  the NDK toolchain sets `CMAKE_C_FLAGS_DEBUG` at `project()` time, and a normal
  variable shadows the cache, so forcing it into the cache is silently ignored.
- **No `-march`.** Cross-compiling leaves `GGML_NATIVE` off, and with no
  `GGML_CPU_ARM_ARCH` ggml adds no architecture flag at all — so the build
  targets baseline `armv8-a` and the quantised dot products take the scalar
  path. Now `armv8.2-a+dotprod+fp16`.

Check what a build actually used rather than assuming:

```bash
python3 - app/.cxx/Debug/*/arm64-v8a/compile_commands.json <<'EOF'
import json,sys
for c in json.load(open(sys.argv[1])):
    if c['file'].endswith('ggml-cpu.c'):
        print([t for t in c['command'].split() if t.startswith(('-O','-march'))])
EOF
# expect: ['-O3', '-DNDEBUG', '-march=armv8.2-a+dotprod+fp16']
```

`+dotprod` needs ARMv8.2, which means roughly a 2018-or-later chip. On anything
older the app dies immediately with `SIGILL`. Rebuild for the baseline if so:

```bash
./gradlew :app:assembleDebug -PeliasArmArch=armv8-a
```

Threads are set to `cores - 1`, capped at 6. On a big.LITTLE phone that includes
the little cores, which can be slower than using the big ones alone — worth
trying 4 if throughput disappoints.

## If it crashes

Get the native stack, which is the only thing that identifies the cause:

```bash
adb logcat -c && adb logcat | tee crash.log      # then reproduce
grep -E "elias-native|DEBUG|FATAL|JNI DETECTED" crash.log
```

Two crash causes have already been found and fixed — by code inspection and by
reproducing the conditions on a desktop, not from a device log, since no device
was available. Both abort the process rather than raising anything catchable,
which is why they read as "the app closes" with no error:

- **Invalid Modified UTF-8 into `NewStringUTF`.** Byte-level BPE splits
  multi-byte characters across tokens — `😀` arrives as a 3-byte piece then a
  1-byte piece — and Android's CheckJNI aborts on malformed input rather than
  returning an error. `elias_core` now holds a partial character back until the
  token that completes it, and `test_cli` asserts the invariant on every piece.
- **A dangling token pointer.** `llama_batch_get_one` stores the pointer it is
  given rather than copying it, so a `llama_token` declared inside the decode
  loop was read back by the next `llama_decode` after going out of scope. It
  survived on x86 by landing in the same stack slot each iteration; different
  code generation is free to do otherwise.

## What has been verified, and what has not

Verified on an x86 host and by inspecting the built APK:

- bitnet.cpp loads Microsoft's GGUF and generates: *"The capital of France is
  Paris. The capital of Germany is Berlin."* at **18.4 tok/s** on four cores.
- Memory is `mmap`-backed: 223 MB dirty, 1.14 GB file-backed.
- The arm64-v8a cross-compile links cleanly, and `libelias.so` in the APK is
  `elf64-littleaarch64` carrying `ggml_gemm_i2_i8_s` and the rest of the ternary
  kernels, with all four JNI entry points exported.
- Mainline llama.cpp **cannot** read this GGUF (`type 36 … IQ4_NL_4_4 REMOVED`),
  which is why the fork is vendored rather than the upstream project.

Not verified, and the honest gaps:

- **Nothing has run on a phone.** There was no device or emulator available, and
  no qemu to execute arm64 here. The APK builds and contains the right code; it
  has never been launched.
- **Conversational quality is poor, and this is the honest headline.** A single
  factual question is answered correctly — *"The capital of France is Paris."* —
  but multi-turn conversation degrades badly, with the model referring to
  earlier turns incoherently, and arithmetic loops. Two things were measured and
  applied: a repetition penalty of 1.25 over a 256-token window (at 1.1 it
  answered and then looped for the entire token budget; at 1.25 it answers and
  stops), and no system turn (the model paraphrases instructions back instead of
  following them). Both help. Neither makes it good. This is BitNet 2B4T at
  `i2_s`, not the plumbing — the same prompts through the same core behave the
  same on a desktop. If it disappoints, any instruct GGUF can be swapped in
  through the same picker and its own chat template will be used;
  Qwen2.5-1.5B-Instruct at Q4_K_M is a similar size and much stronger at chat.
- **Speech recognition may need a network.** The model does not, but Android's
  `SpeechRecognizer` is usually server-backed. The app asks for on-device
  recognition (`EXTRA_PREFER_OFFLINE`) and falls back to typing, which always
  works offline.
