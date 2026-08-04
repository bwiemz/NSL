#!/usr/bin/env python3
"""Build a REAL (non-synthetic) u16 token corpus by byte-level tokenizing
the repository's own source text.

Item 7 needs a real-corpus trajectory arm: the banked SR-BF16 curves all ran
on `gen_tokens.py`'s stream, which is a fixed 16384-token block tiled ~244x —
a memorization drill whose loss collapses to ~0.01 and whose update-magnitude
distribution narrows accordingly. This corpus has none of that structure: it
is the concatenated bytes of the repo's NSL/Rust/Markdown sources, each byte
one token id (0..255, a strict subset of the models' 49152 vocab), files
separated by a 0 (text files carry no NUL bytes). Deterministic: fixed
directory list, sorted paths, so every run and every seed sees the same
stream.

Byte-level rather than BPE because the repo's BPE tokenizer is a training
SCRIPT (models/coder-rl/data/build_tokenizer.py), not an artifact, and this
environment has no `tokenizers` package to run it with. The trajectory
comparison needs real token statistics (no artificial periodicity), which
bytes of real text provide; it does not need a production vocabulary. The
label "real-corpus (byte-level)" travels with the results.

Usage:
    python3 tools/build_byte_corpus.py [--out data/tokcorpus/byte_tokens.bin]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Source of the text, in a fixed order. `crates` is the bulk; the NSL
#: dirs keep the stream on-domain for the coder models.
DIRS = ["stdlib", "examples", "spec", "docs/wiki", "crates"]
SUFFIXES = {".nsl", ".rs", ".md"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "tokcorpus" / "byte_tokens.bin",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for d in DIRS:
        root = REPO / d
        if root.is_dir():
            files.extend(
                p for p in sorted(root.rglob("*")) if p.suffix in SUFFIXES
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "wb") as f:
        for p in files:
            data = p.read_bytes()
            for byte in data:
                f.write(struct.pack("<H", byte))
            f.write(struct.pack("<H", 0))
            n += len(data) + 1
    print(f"wrote {n:,} byte-level tokens from {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
