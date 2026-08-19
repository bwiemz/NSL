#!/usr/bin/env python3
"""Materialize the production train/val split (roadmap item 9).

`data/tokens/train_new.bin` is the shipped tokenization of the real corpus
(8,925,916 u16 tokens; see models/benchmarks/TOKENIZER_AND_MFU_2026_07_29.md).
The production recipes train on a PREFIX of it and report cross-entropy on a
held-out TAIL they never read:

    [ 0 .............. TRAIN_TOKENS ) [ gap ) [ .... VAL_TOKENS .... ]
      prod_train_slice.bin                     prod_val_slice.bin

The gap is deliberate. `DataLoader(..., drop_last=true)` consumes whole
`seq_len` windows, so without it a training window could end mid-way into the
first validation window and leak a suffix the val loss then scores itself on.
Sized so the split lands on whole 1024-token windows for both batch=1 (50M)
and batch=2 (500M).

The corpus is local data, not committed. This script is idempotent: it
rewrites a slice only when it is missing or the wrong size.

Usage (from the repo root):
    python models/benchmarks/make_prod_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data/tokens/train_new.bin"
TRAIN_SLICE = REPO / "data/tokens/prod_train_slice.bin"
VAL_SLICE = REPO / "data/tokens/prod_val_slice.bin"

# Must match models/coder{50m,500m}/config.nsl — pinned by
# crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs, which reads the
# consts out of config.nsl and this file's numbers out of here.
TRAIN_TOKENS = 8_388_608  # 8192 windows of 1024
VAL_TOKENS = 524_288  # 512 windows of 1024


def main() -> int:
    if not CORPUS.exists():
        print(
            f"missing local corpus: {CORPUS}\n"
            "It is not committed — see models/coder50m/pretrain_prod.nsl for "
            "its provenance (this repo's train split + 15 unrelated local "
            "projects, tokenized by the shipped two-stage transition=40960 "
            "tokenizer at vocab 49152).",
            file=sys.stderr,
        )
        return 1

    data = CORPUS.read_bytes()
    n_tok = len(data) // 2
    val_start = n_tok - VAL_TOKENS
    if val_start <= TRAIN_TOKENS:
        print(
            f"corpus has {n_tok} tokens; a {TRAIN_TOKENS}-token train slice and "
            f"a {VAL_TOKENS}-token held-out tail would OVERLAP. Refusing to "
            "write a split whose validation set is part of its training set.",
            file=sys.stderr,
        )
        return 1

    gap = val_start - TRAIN_TOKENS
    for path, blob in (
        (TRAIN_SLICE, data[: TRAIN_TOKENS * 2]),
        (VAL_SLICE, data[val_start * 2 :]),
    ):
        if path.exists() and path.stat().st_size == len(blob):
            print(f"[split] {path.name} already correct ({len(blob)} bytes)")
            continue
        path.write_bytes(blob)
        print(f"[split] wrote {path.name} ({len(blob)} bytes)")

    print(
        f"[split] {n_tok} corpus tokens -> {TRAIN_TOKENS} train / {gap} unused "
        f"gap / {VAL_TOKENS} held-out"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
