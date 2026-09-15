# Model directory

Elias needs a BitNet 2B4T ONNX export. It is a gigabyte-class file, so it is
never committed and never part of the build — GitHub Pages caps a site at 1 GB
and a single file at 100 MB, and the APK is a Trusted Web Activity that carries
no web assets at all. There are two supported ways to get the weights onto the
device, and one fallback for when neither has been arranged.

## 1. Build them in

Drop the export into this directory before running `npm run build`. Everything
here except `model-config.json` and this README is gitignored.

| File                    | Required | Notes                                                  |
| ----------------------- | -------- | ------------------------------------------------------ |
| `bitnet-2b4t.onnx`      | yes      | Filename must match `modelFile` in `model-config.json`. |
| `bitnet-2b4t.onnx_data` | maybe    | Present when the export uses external weight data. Name it in `weightsDataFile`. |
| `tokenizer.json`        | yes      | HuggingFace fast-tokenizer format (byte-level BPE).     |
| `model-config.json`     | yes      | Committed. Edit to match your export.                   |

This only works for a deployment that can actually serve a file that size. A
project Pages site cannot; a private host or a local `npm run preview` can.

## 2. Download them on demand

Point `model-config.json` at a host that serves the export, and Elias offers to
fetch it from a banner in the UI. Nothing is downloaded until the user presses
the button — a phone on mobile data must not lose a gigabyte to an app starting
up — and the bytes are kept in Cache Storage afterwards, so it is a one-time
cost that survives app updates.

```json
{
  "modelFile": "bitnet-2b4t.onnx",
  "weightsUrl": "https://huggingface.co/<user>/<repo>/resolve/main/bitnet-2b4t.onnx",
  "weightsDataFile": "bitnet-2b4t.onnx_data",
  "weightsSizeBytes": 1150000000
}
```

| Key                | Meaning                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| `weightsUrl`       | Absolute **https** URL to the `.onnx`. Empty disables the offer entirely.       |
| `weightsDataFile`  | External-data filename as the graph refers to it. Required if the export has one. |
| `weightsDataUrl`   | Where to get that file. Defaults to the same directory as `weightsUrl`.        |
| `tokenizerUrl`     | Where to get `tokenizer.json`. Defaults to the same directory as `weightsUrl`. |
| `weightsSizeBytes` | Download size to show the user, used only when the host refuses a HEAD request. |

The host has to allow cross-origin reads — the download is a `cors` fetch and an
opaque response cannot be cached or handed to ONNX Runtime. Hugging Face
`resolve/` URLs and GitHub release assets both do. A URL that is not absolute
https is rejected at boot with the field named, rather than turning into a 404
much later.

Local files still win: if an export is present in this directory, it is used and
nothing is downloaded.

## 3. Neither — the fallback reasoner

With no weights and no `weightsUrl`, Elias boots on a deterministic non-neural
responder that emits the same action schema the real model does. The wake word,
state machine, telemetry, linter, sandbox and memory are all fully exercised, so
the pipeline can be tested on a device that has never downloaded a 2B parameter
file. The backend pill reads "fallback reasoner" and the banner says why.

The build says which of the three you are shipping, and warns only for this one.

## Making `model-config.json` match your export

`numHiddenLayers`, `numKeyValueHeads` and `headDim` size the empty KV cache fed
to the graph on the first forward pass. They must match the exported model or
the first `session.run` fails with a shape mismatch. The committed values follow
the published BitNet b1.58-2B-4T configuration (30 layers, 5 KV heads, 128 head
dim); read them off your own `config.json` if you exported something else.

The `template` block renders the chat prompt. Replace the four slot strings if
your export was trained with different turn markers — the tokenizer resolves
whatever special tokens you name, so no code change is needed.

## Graph requirements

`inference.worker.js` introspects the session rather than assuming one export's
naming, and supports:

- `input_ids` (int64, required), plus optional `attention_mask` and `position_ids`
- KV cache inputs named `past_key_values.<n>.key` / `.value`
- matching outputs named `present.<n>.key` / `.value`, or `present_key_values.<n>.*`
- a `logits` output shaped `[batch, sequence, vocab]`

A graph without a KV cache still runs — the worker falls back to re-feeding the
whole sequence each step — but expect it to be far slower.
