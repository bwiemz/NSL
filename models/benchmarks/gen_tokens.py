"""Generate synthetic u16 token files for the P0 certification campaign.

A FIXED pseudo-random block (numpy PCG64, seed 1234) tiled to the requested
length: varied enough that the loss curve reflects real optimization
dynamics, repetitive enough to be learnable (memorization drill), and —
because the block is fixed — IDENTICAL across model-init seeds, so
`--seed` varies only the initialization.

Format: raw little-endian u16 token IDs, read by `load_mmap(path, 3)`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate(path: Path, n_tokens: int, vocab: int, block: int) -> None:
    rng = np.random.Generator(np.random.PCG64(1234))
    base = rng.integers(0, vocab, size=block, dtype=np.uint16)
    reps = (n_tokens + block - 1) // block
    tokens = np.tile(base, reps)[:n_tokens]
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)
    print(f"wrote {path}: {n_tokens} u16 tokens (vocab {vocab}, block {block})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--vocab", type=int, default=49152)
    parser.add_argument("--block", type=int, default=16384)
    args = parser.parse_args()
    generate(args.path, args.tokens, args.vocab, args.block)


if __name__ == "__main__":
    main()
