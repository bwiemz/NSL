#!/usr/bin/env python3
"""Materialize the production train/val split (roadmap item 9).

`data/tokens/train_new.bin` is the shipped tokenization of the real corpus
(8,925,916 u16 tokens; see models/benchmarks/TOKENIZER_AND_MFU_2026_07_29.md).
The production recipes train on a PREFIX of it and report cross-entropy on a
held-out TAIL they never read:

    [ 0 .............. TRAIN_TOKENS ) [ gap ) [ .... VAL_TOKENS .... ]
      prod_train_slice.bin                     prod_val_slice.bin

The gap is deliberate, but NOT because a window could straddle it — the two
slices are separate files, so the loaders cannot cross between them at all,
and any prefix/tail cut is index-disjoint even with a zero gap. What the gap
buys is SEPARATION: this corpus is a concatenation of source trees, so tokens
immediately after the training prefix are usually the continuation of the same
file or project. Starting the held-out set right at the boundary would score
the model on text whose preceding context it just trained on, flattering the
val loss. Both slices are also whole numbers of 1024-token windows, so
`drop_last=true` discards nothing and the derived step counts are exact.

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

# Must match models/coder{50m,500m,1b}/config.nsl — pinned by
# crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs, which reads the
# consts out of config.nsl and this file's numbers out of here, for EVERY size
# that reads the split. 500M and 1B deliberately share these two files so the
# two production runs are scored on identical held-out text; both counts are
# whole numbers of delivery slots at both geometries (a slot is
# batch_size * seq_len contiguous tokens: 2 * 1024 = 2048 at 500M,
# 2 * 2048 = 4096 at 1B), so `drop_last=true` discards nothing either way.
TRAIN_TOKENS = 8_388_608  # 8192 windows of 1024, or 4096 of 2048
VAL_TOKENS = 524_288  # 512 windows of 1024, or 256 of 2048


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
        # Size alone is NOT enough to call a slice current: TRAIN_TOKENS and
        # VAL_TOKENS are fixed, so a REGENERATED corpus (retokenized, or a
        # different mix) produces slices of exactly the same length with
        # different contents. A size-only guard would leave the old slices in
        # place and every downstream gate would stay green while the recipe
        # trained on stale data. Compare content — cheaply, over the ends and
        # a strided interior sample rather than the whole 16 MB.
        if path.exists() and path.stat().st_size == len(blob):
            existing = path.read_bytes()
            probe = slice(None, 1 << 16)
            tail = slice(-(1 << 16), None)
            stride = max(1, len(blob) // (1 << 16))
            if (
                existing[probe] == blob[probe]
                and existing[tail] == blob[tail]
                and existing[::stride] == blob[::stride]
            ):
                print(f"[split] {path.name} already correct ({len(blob)} bytes)")
                continue
            print(f"[split] {path.name} is stale — corpus changed under it; rewriting")
        path.write_bytes(blob)
        print(f"[split] wrote {path.name} ({len(blob)} bytes)")

    print(
        f"[split] {n_tok} corpus tokens -> {TRAIN_TOKENS} train / {gap} unused "
        f"gap / {VAL_TOKENS} held-out"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
