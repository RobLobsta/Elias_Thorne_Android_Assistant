#!/usr/bin/env python3
"""Add the missing `tokenizer.ggml.pre` key to Microsoft's BitNet GGUF.

Microsoft's `ggml-model-i2_s.gguf` ships without `tokenizer.ggml.pre`. llama.cpp
therefore falls back to its 'default' pre-tokenizer and says so:

    load: missing pre-tokenizer type, using: 'default'
    load: GENERATION QUALITY WILL BE DEGRADED!

BitNet b1.58-2B-4T uses the Llama-3 tokenizer, whose pre-tokenizer splits text
on a specific regex. Getting that wrong mis-splits every prompt, so the model
sees token sequences unlike anything it was trained on and answers badly. The
weights are fine; only this one string is missing.

The obvious fix — the `gguf` Python package — cannot open the file at all: it
rejects tensor type 36, which is bitnet.cpp's I2_S sharing an id with a quant
mainline removed. So this walks the container itself.

Safe to do because a GGUF tensor's `offset` is relative to the start of the data
section, not to the file. Growing the header moves the data section but not the
offsets within it; only the alignment padding has to be recomputed.

    python3 patch_gguf_pretokenizer.py in.gguf out.gguf [--pre llama-bpe]
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# Byte width of each fixed-size GGUF metadata value type.
FIXED_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_STRING = 8
TYPE_ARRAY = 9


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        if len(out) != n:
            raise ValueError("unexpected end of file")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def skip_value(self, vtype: int) -> None:
        """Advance past one value without decoding it."""
        if vtype == TYPE_STRING:
            self.take(self.u64())
        elif vtype == TYPE_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if elem_type in FIXED_WIDTHS:
                self.take(FIXED_WIDTHS[elem_type] * count)
            else:
                for _ in range(count):
                    self.skip_value(elem_type)
        elif vtype in FIXED_WIDTHS:
            self.take(FIXED_WIDTHS[vtype])
        else:
            raise ValueError(f"unknown metadata value type {vtype}")


def encode_string_kv(key: str, value: str) -> bytes:
    kb = key.encode("utf-8")
    vb = value.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", TYPE_STRING) + struct.pack("<Q", len(vb)) + vb


def patch(src: Path, dst: Path, pre: str) -> None:
    raw = src.read_bytes()
    r = Reader(raw)

    if r.take(4) != GGUF_MAGIC:
        sys.exit(f"{src} is not a GGUF file")
    version = r.u32()
    if version != 3:
        print(f"warning: GGUF version {version}, expected 3 — continuing", file=sys.stderr)
    n_tensors = r.u64()
    n_kv = r.u64()

    kv_start = r.pos
    alignment = 32
    keys = []
    for _ in range(n_kv):
        key = r.string()
        keys.append(key)
        vtype = r.u32()
        if key == "general.alignment":
            before = r.pos
            alignment = struct.unpack("<I", raw[before + 0 : before + 4])[0] if vtype == 4 else 32
        r.skip_value(vtype)
    kv_end = r.pos

    if pre_key := "tokenizer.ggml.pre" in keys:
        sys.exit("tokenizer.ggml.pre is already present; nothing to do")
    del pre_key

    # Tensor info: name, n_dims, dims[], type, offset.
    for _ in range(n_tensors):
        r.string()
        n_dims = r.u32()
        r.take(8 * n_dims)
        r.u32()
        r.u64()
    info_end = r.pos

    def align_up(value: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    data_start = align_up(info_end)
    tensor_data = raw[data_start:]

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", version)
    header += struct.pack("<Q", n_tensors)
    header += struct.pack("<Q", n_kv + 1)
    header += raw[kv_start:kv_end]
    header += encode_string_kv("tokenizer.ggml.pre", pre)
    header += raw[kv_end:info_end]
    header += b"\x00" * (align_up(len(header)) - len(header))

    with dst.open("wb") as out:
        out.write(header)
        out.write(tensor_data)

    print(f"tensors        : {n_tensors}")
    print(f"metadata keys  : {n_kv} -> {n_kv + 1}")
    print(f"alignment      : {alignment}")
    print(f"added          : tokenizer.ggml.pre = {pre!r}")
    print(f"wrote          : {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--pre", default="llama-bpe", help="pre-tokenizer name (default: llama-bpe)")
    args = ap.parse_args()

    if args.dst.exists():
        sys.exit(f"{args.dst} already exists")
    if shutil.disk_usage(args.dst.parent).free < args.src.stat().st_size * 1.05:
        sys.exit("not enough free space to write the patched copy")
    patch(args.src, args.dst, args.pre)


if __name__ == "__main__":
    main()
