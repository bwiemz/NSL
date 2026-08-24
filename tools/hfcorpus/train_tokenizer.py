#!/usr/bin/env python3
"""Train the v2 tokenizer for the mixed web + code + chat corpus, and score it.

WHY A SECOND TOKENIZER. The shipped v1 tokenizer (`transition=40960`, vocab
49152) learned its vocabulary from 48 MB of source code. The pretraining corpus
is now ~10B tokens of English web text, of which the code corpus is under 0.1%.
A vocabulary shaped by code spends its merges on indentation runs and bracket
sequences that web prose does not contain, so it would encode the bulk of the
corpus at a materially worse bytes/token -- a direct multiplier on the number of
sequence positions every pretraining run has to pay for.

v2 changes exactly two things: the training mixture, and six reserved special
tokens (v1 reserved none, so it had no representable document boundary at all).
Everything else -- two-stage cl100k->line training, vocab 49152, min_freq 2 --
is held fixed so the two are comparable, and 49152 keeps the id space inside the
u16 stream format that `load_mmap(path, 3)` requires.

v2 is written to a NEW path and the existing corpora are not touched. v1 remains
the tokenizer of record for `data/tokens/train_new.bin` and everything pinned
against it (`pretrain_prod_agreement_gate`, the 5.403 B/tok fingerprint).

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/train_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKBENCH = REPO / "tools/tokbench/target/release/tokbench"
OUT_DIR = REPO / "data/tokenizer"

# Reserved surfaces, loaded from the ONE committed record (see extract.py for
# why an inline copy is refused by the gate). Training order matters — it
# decides which surfaces BPE has already merged when ids are assigned — so the
# record keeps them as an ORDERED array.
SPECIALS: list[str] = json.loads(
    (REPO / "models/tokenizers/special_tokens.json").read_text()
)["surfaces"]

VOCAB = 49152
TRANSITION = 40960
MIN_FREQ = 2

# Mixture for TOKENIZER TRAINING. It tracks the PRETRAINING mixture on purpose:
# a vocabulary learned from the wrong proportions is exactly what made v1
# expensive, costing 33.5% more sequence positions once the corpus turned out to
# be mostly web. The corpus is now ~60% general code, so the sample is too.
# `nsl` stays over-represented relative to its 0.25% share -- it is the language
# the models exist to write, and at 48 MB it costs the others almost nothing.
# Every part must be a directory of shards produced by extract.py, NOT a raw
# corpus file. Feeding `data/tokcorpus/combined_train.txt` directly would bypass
# the sanitizer that exists precisely for it: that corpus carries 35 raw
# `<|...|>` surfaces, so the trainer would learn merges for the very surfaces the
# tokenizer reserves. Build it with `extract.py code --out data/text/code-train`.
SAMPLE_PARTS = [
    ("stack", REPO / "data/text/toksample-stack"),
    ("web", REPO / "data/text/toksample-web"),
    ("sft", REPO / "data/text/toksample-sft"),
    ("nsl", REPO / "data/text/code-train"),
]


def run(cmd: list[str]) -> str:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out.rstrip(), flush=True)
    if proc.returncode:
        raise SystemExit(f"command failed ({proc.returncode})")
    return out


def assemble_sample(dest: Path) -> dict[str, int]:
    """Concatenate the parts into one NUL-separated corpus file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    with dest.open("wb") as out:
        first = True
        for name, src in SAMPLE_PARTS:
            if not src.is_dir():
                raise SystemExit(
                    f"part '{name}' must be an extract.py output directory, not "
                    f"{src} — a raw corpus file has not been through the "
                    "special-surface sanitizer"
                )
            paths = sorted(src.glob("*.txt"))
            if not paths:
                raise SystemExit(f"no text for part '{name}' at {src}")
            total = 0
            for p in paths:
                blob = p.read_bytes()
                if not first:
                    out.write(b"\x00")
                    total += 1
                out.write(blob)
                total += len(blob)
                first = False
            sizes[name] = total
            print(f"[sample] {name}: {total / 1e9:.3f} GB from {len(paths)} file(s)", flush=True)
    sizes["total"] = dest.stat().st_size
    return sizes


def to_jsonl(shard_dir: Path, dest: Path, limit: int) -> int:
    """Convert NUL-separated shards to the {path,bytes,text} JSONL eval format."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as out:
        for shard in sorted(shard_dir.glob("*.txt")):
            for i, doc in enumerate(shard.read_text(encoding="utf-8").split("\x00")):
                if not doc:
                    continue
                out.write(
                    json.dumps(
                        {
                            "path": f"{shard.name}#{i}",
                            "bytes": len(doc.encode("utf-8")),
                            "text": doc,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
                if n >= limit:
                    return n
    return n


def bytes_per_token(tokenizer: Path, heldout: Path, label: str) -> float | None:
    if not heldout.exists():
        return None
    out = run([str(TOKBENCH), "eval", "--tokenizer", str(tokenizer),
               "--heldout", str(heldout), "--name", label])
    for line in out.splitlines():
        if "BYTES/TOKEN" in line:
            return float(line.split()[1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, default=None,
                    help="path to the reproduced v1 tokenizer, for the comparison table")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not TOKBENCH.exists():
        raise SystemExit(f"tokbench not built: {TOKBENCH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2 = OUT_DIR / f"nsl_mix_v2_t{TRANSITION}_v{VOCAB}.json"
    sample = REPO / "data/text/toksample/mixed.txt"

    if not args.skip_train:
        sizes = assemble_sample(sample)
        print(f"[sample] total {sizes['total'] / 1e9:.3f} GB", flush=True)
        cmd = [str(TOKBENCH), "train2", "--corpus", str(sample), "--out", str(v2),
               "--stage1", "cl100k", "--relaxed", "line",
               "--vocab-size", str(VOCAB), "--transition", str(TRANSITION),
               "--min-freq", str(MIN_FREQ), "--max-token-bytes", "0"]
        for s in SPECIALS:
            cmd += ["--special-token", s]
        train_out = run(cmd)
        ids = {}
        for line in train_out.splitlines():
            if line.startswith("special "):
                _, surface, _, tid = line.split()
                ids[surface] = int(tid)
        record = json.dumps({"tokenizer": v2.name, "ids": ids}, indent=2)
        (OUT_DIR / "special_tokens.json").write_text(record)
        # Also beside the committed tokenizer: OUT_DIR is under gitignored
        # `data/`, so a fresh clone would get the tokenizer without the ids that
        # make it usable.
        tracked = REPO / "models/tokenizers"
        if tracked.is_dir():
            (tracked / "special_tokens.json").write_text(record)
        print(f"[train] reserved ids: {ids}", flush=True)

    # Held-out sets: two existing code sets, plus web and chat cut from splits
    # the tokenizer sample never saw.
    heldouts = {
        "nsl": REPO / "data/tokcorpus/heldout.jsonl",
        "ood": REPO / "data/tokcorpus/ood.jsonl",
        "web": REPO / "data/text/heldout/web.jsonl",
        "sft": REPO / "data/text/heldout/sft.jsonl",
    }
    for key, shard_dir in (("web", REPO / "data/text/val-web"),
                           ("sft", REPO / "data/text/val-sft")):
        if shard_dir.is_dir() and not heldouts[key].exists():
            n = to_jsonl(shard_dir, heldouts[key], limit=20_000)
            print(f"[heldout] {key}: {n:,} documents", flush=True)

    rows = []
    for key, path in heldouts.items():
        row = {"set": key}
        if args.v1:
            row["v1"] = bytes_per_token(args.v1, path, f"v1-{key}")
        row["v2"] = bytes_per_token(v2, path, f"v2-{key}")
        rows.append(row)

    print("\n=== bytes/token (higher is better) ===")
    header = f"{'held-out':<8}" + ("  {:>8}".format("v1 code") if args.v1 else "") + f"  {'v2 mix':>8}"
    print(header)
    for r in rows:
        if r.get("v2") is None:
            continue
        line = f"{r['set']:<8}"
        if args.v1:
            line += f"  {r['v1']:>8.3f}" if r.get("v1") is not None else f"  {'-':>8}"
        line += f"  {r['v2']:>8.3f}"
        if args.v1 and r.get("v1"):
            line += f"   ({(r['v2'] / r['v1'] - 1) * 100:+.1f}%)"
        print(line)

    (OUT_DIR / "comparison.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
