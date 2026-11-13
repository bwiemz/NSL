#!/usr/bin/env python3
"""Encode the repository's own tracked text into a u16 token stream.

The checks here need tokens with real structure, and the corpora the recipes train
on are local data that a fresh checkout does not have. This builds a stand-in from
what every checkout does have — the tracked source and documentation — with the
shipped v1 tokenizer (`models/tokenizers/nsl_code_v1_t40960_v49152.json`).

    tools/refcheck/make_corpus.py refcheck-work/data
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
EXTENSIONS = {".rs", ".nsl", ".md", ".py", ".toml", ".sh", ".json", ".yml"}
VAL_TOKENS = 262_144
MAX_FILE_BYTES = 2_000_000


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    )
    files = [
        f
        for f in listed.stdout.splitlines()
        if Path(f).suffix in EXTENSIONS and "CHANGELOG" not in f
    ]
    random.Random(1234).shuffle(files)  # a fixed order, so the stream is reproducible
    texts: list[str] = []
    for name in files:
        path = REPO / name
        try:
            if path.stat().st_size <= MAX_FILE_BYTES:
                texts.append(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    tokenizer = Tokenizer.from_file(
        str(REPO / "models/tokenizers/nsl_code_v1_t40960_v49152.json")
    )
    stream = np.concatenate(
        [np.asarray(e.ids, dtype=np.uint16) for e in tokenizer.encode_batch(texts)]
    )
    train, val = stream[:-VAL_TOKENS], stream[-VAL_TOKENS:]
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    counts = np.bincount(train, minlength=49152).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    print(f"{len(texts)} files -> {len(train)} train + {len(val)} held-out tokens")
    print(
        f"unigram entropy of the train stream: {-(p * np.log(p)).sum():.3f} nats "
        f"(a model that has learned nothing but token frequencies scores this)"
    )


if __name__ == "__main__":
    main()
