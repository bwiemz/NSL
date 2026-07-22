#!/usr/bin/env python3
"""P1 Muon item 5: build the byte-level code corpus for the Muon-vs-AdamW
comparison.

Real code (this repo's Rust + NSL sources, padded with cargo-registry crate
sources for volume), byte-level tokenization (vocab 256 — no external
tokenizer dependency), deterministic file order, held-out validation split
BY FILE (no window leakage):

  - every 20th repo .rs file  -> val_rust  (16 x 1025 windows)
  - every 4th  .nsl file      -> val_nsl   (16 x 1025 windows)
  - everything else, sorted   -> train stream (u16 .bin, mode-3 mmap)

Outputs (in --out):
  train_tokens.bin          u16 token stream (byte values), truncated to
                            --train-tokens+1
  val_rust_in.bin/.lab.bin  f32 flat [16*1024] inputs / labels (mode-1 mmap)
  val_nsl_in.bin/.lab.bin   f32 flat [16*1024] inputs / labels
  corpus_manifest.json      file lists + sizes (reproducibility record)

Every training arm reads the SAME train_tokens.bin -> identical data order
across optimizers/lrs (the item-5 equal-data requirement).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

VAL_WINDOWS = 16
SEQ = 1024


def collect(repo: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    rs = sorted(
        p
        for p in (repo / "crates").rglob("*.rs")
        if "target" not in p.parts
    )
    nsl: list[Path] = []
    for d in ["stdlib", "examples", "tests", "spec", "models"]:
        root = repo / d
        if root.exists():
            nsl.extend(sorted(root.rglob("*.nsl")))
    val_rust = rs[::20]
    train_rs = [p for p in rs if p not in set(val_rust)]
    val_nsl = nsl[::4]
    train_nsl = [p for p in nsl if p not in set(val_nsl)]
    return train_rs, train_nsl, val_rust, val_nsl


def registry_files(limit_bytes: int) -> list[Path]:
    reg = Path.home() / ".cargo" / "registry" / "src"
    if not reg.exists():
        return []
    out: list[Path] = []
    total = 0
    for p in sorted(reg.rglob("*.rs")):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz == 0 or sz > 512 * 1024:
            continue
        out.append(p)
        total += sz
        if total >= limit_bytes:
            break
    return out


def stream_bytes(files: list[Path]) -> bytes:
    chunks: list[bytes] = []
    for f in files:
        try:
            chunks.append(f.read_bytes())
            chunks.append(b"\x00")  # file separator
        except OSError:
            continue
    return b"".join(chunks)


def write_u16(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(data)}H", *data))


def write_val(path_prefix: Path, data: bytes) -> None:
    need = VAL_WINDOWS * (SEQ + 1)
    assert len(data) >= need, f"val corpus too small: {len(data)} < {need}"
    ins: list[float] = []
    labs: list[float] = []
    for w in range(VAL_WINDOWS):
        window = data[w * (SEQ + 1) : (w + 1) * (SEQ + 1)]
        ins.extend(float(b) for b in window[:SEQ])
        labs.extend(float(b) for b in window[1:])
    with open(f"{path_prefix}_in.bin", "wb") as f:
        f.write(struct.pack(f"<{len(ins)}f", *ins))
    with open(f"{path_prefix}_lab.bin", "wb") as f:
        f.write(struct.pack(f"<{len(labs)}f", *labs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "muon_data")
    ap.add_argument("--train-tokens", type=int, default=8_000_000)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)

    train_rs, train_nsl, val_rust, val_nsl = collect(repo)
    train_files = train_nsl + train_rs  # NSL first: every run sees NSL code
    train_data = stream_bytes(train_files)
    if len(train_data) < args.train_tokens + 1:
        reg = registry_files((args.train_tokens + 1 - len(train_data)) * 2)
        train_files += reg
        train_data += stream_bytes(reg)
    assert len(train_data) >= args.train_tokens + 1, (
        f"corpus too small: {len(train_data)}"
    )
    train_data = train_data[: args.train_tokens + 1]
    write_u16(args.out / "train_tokens.bin", train_data)

    write_val(args.out / "val_rust", stream_bytes(val_rust))
    write_val(args.out / "val_nsl", stream_bytes(val_nsl))

    manifest = {
        "train_tokens": len(train_data),
        "train_files": len(train_files),
        "val_rust_files": len(val_rust),
        "val_nsl_files": len(val_nsl),
        "val_windows": VAL_WINDOWS,
        "seq": SEQ,
        "vocab": 256,
    }
    (args.out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
