# Model directory

Drop the BitNet 2B4T ONNX export here. Nothing in this directory is committed —
weights are cache-managed at runtime by `public/sw.js` and evicted only if the
persistent storage lock (`navigator.storage.persist()`) was refused.

## Expected contents

| File                    | Required | Notes                                                  |
| ----------------------- | -------- | ------------------------------------------------------ |
| `bitnet-2b4t.onnx`      | yes      | Filename must match `modelFile` in `model-config.json`. |
| `bitnet-2b4t.onnx_data` | maybe    | Present when the export uses external weight data.      |
| `tokenizer.json`        | yes      | HuggingFace fast-tokenizer format (byte-level BPE).     |
| `model-config.json`     | yes      | Committed. Edit to match your export.                   |

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

## Without weights

If no weights are present, Elias boots on the fallback reasoner: a deterministic
non-neural responder that emits the same action schema the real model does. The
wake word, state machine, telemetry, linter, sandbox and memory are all fully
exercised, so the pipeline can be tested on a device that has never downloaded a
2B parameter file. The backend pill reads "fallback reasoner" when this is the
case.
