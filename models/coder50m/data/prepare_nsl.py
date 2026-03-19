#!/usr/bin/env python3
"""Tokenize NSL source files for Stage 2 finetuning."""

import struct
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("pip install tokenizers")
    sys.exit(1)


def collect_nsl_files(repo_root: Path) -> list[Path]:
    dirs = ["stdlib", "examples", "tests"]
    files: list[Path] = []
    for d in dirs:
        files.extend(sorted((repo_root / d).rglob("*.nsl")))
    return files


def tokenize_files(files: list[Path], tokenizer: Tokenizer) -> list[int]:
    all_ids: list[int] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(0)  # separator
    return all_ids


def write_u16_bin(ids: list[int], path: Path) -> None:
    with open(path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<H", tid % 65536))
    print(f"  Wrote {len(ids):,} tokens to {path} ({path.stat().st_size:,} bytes)")


def sample_general_tokens(path: Path, n: int) -> list[int]:
    ids: list[int] = []
    with open(path, "rb") as f:
        for _ in range(n):
            data = f.read(2)
            if not data:
                break
            ids.append(struct.unpack("<H", data)[0])
    return ids


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    tokenizer_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    general_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = Path(__file__).resolve().parent

    if not tokenizer_path:
        print("Usage: python prepare_nsl.py <tokenizer.json> [general_tokens.bin]")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    nsl_files = collect_nsl_files(repo_root)
    print(f"Found {len(nsl_files)} .nsl files")

    nsl_ids = tokenize_files(nsl_files, tokenizer)
    print(f"Total NSL tokens: {len(nsl_ids):,}")
    write_u16_bin(nsl_ids, out_dir / "nsl_tokens.bin")

    if general_path and general_path.exists():
        nsl_1m = (nsl_ids * (1_000_000 // len(nsl_ids) + 1))[:1_000_000]
        general_9m = sample_general_tokens(general_path, 9_000_000)
        mixed = nsl_1m + general_9m
        print(f"Mixed: {len(nsl_1m):,} NSL + {len(general_9m):,} general = {len(mixed):,}")
        write_u16_bin(mixed, out_dir / "mixed_tokens.bin")

    print("Done.")


if __name__ == "__main__":
    main()
