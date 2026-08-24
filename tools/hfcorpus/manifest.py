#!/usr/bin/env python3
"""Assemble and check the committed corpus manifest.

The bins under `data/tokens/mix/` are local artifacts — 45 GB that will never
be committed, on a machine where /tmp has already eaten one tokenizer. What IS
committed must therefore be enough to (a) verify a local copy byte-for-byte,
(b) rebuild the corpus from sources, and (c) refuse a silent tokenizer swap.
That is exactly what this manifest records:

  * sha256 + token count + B/tok of every bin, from the per-bin manifests the
    pipeline already writes at encode time;
  * sha256 of the committed tokenizer JSONs — the per-bin manifests name the
    tokenizer by PATH, and a path is not an identity: retraining to the same
    filename would silently invalidate every recorded hash while every check
    still passed;
  * the Hugging Face dataset REVISIONS actually downloaded, recovered from
    huggingface_hub's own download metadata rather than trusted to memory;
  * the reserved-surface list and the ids the build assigned them;
  * per-set dedup status — stated, not implied. FineWeb ships MinHash-deduped;
    the Stack sample is 20 disjoint shards of a Spark partitioning; the
    distillation set REQUIRED de-duplication (its repo ships two overlapping
    shard series whose naive concatenation duplicates 41,682 traces).

`build` writes models/datasets/CORPUS_MANIFEST_v2.json from local state.
`check` verifies committed state (tokenizer hashes, id agreement) and, with
--hash-bins, re-hashes the local bins against the record (~2 min for 45 GB).

Usage, from the repo root:
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/manifest.py build
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/manifest.py check [--hash-bins]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "models/datasets/CORPUS_MANIFEST_v2.json"
TOKENIZER_DIR = REPO / "models/tokenizers"
MIX = REPO / "data/tokens/mix"
HF = REPO / "data/hf"

# Bins the corpus consists of. An explicit list, not a glob: a stray .bin in
# the mix directory must show up as "unexpected", not get silently manifested.
BINS = [
    "stack_train.bin", "web_train.bin", "sft_train.bin", "nsl_train.bin",
    "web_val.bin", "sft_val.bin", "pretrain_train.bin",
]

DATASETS = {
    "stack": {
        "repo": "HuggingFaceCode/stack-v3-train",
        "sample": "20 of 8,192 shards, strided (indices k*409 for k in 0..20)",
        "dedup": "no additional dedup; the 20 shards are disjoint partitions "
                 "of one Spark job, sampled with a stride so the subset is "
                 "not correlated with partition order",
    },
    "fineweb": {
        "repo": "HuggingFaceFW/fineweb",
        "sample": "config sample-10BT, all 15 parquet files",
        "dedup": "upstream MinHash-deduplicated by the FineWeb pipeline; "
                 "train/val split cut at parquet-file granularity so the "
                 "held-out file shares no crawl segment with train",
    },
    "distill": {
        "repo": "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
        "sample": "config canonical, the train-*-of-00006 series ONLY",
        "dedup": "REQUIRED: the repo ships two overlapping shard series in "
                 "one directory (train-*-of-00005 AND -of-00006); the "
                 "default glob concatenates both and duplicates 41,682 "
                 "traces. Only -of-00006 is read (52,205 + 2,872 + 2,860 "
                 "= 57,937 rows, matching the dataset card)",
    },
    "nsl": {
        "repo": "local: data/tokcorpus/combined_train.txt",
        "sample": "this repository + 15 unrelated local projects",
        "dedup": "none (curated local corpus)",
    },
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def hf_revision(dataset_dir: Path) -> str | None:
    """The HF commit actually downloaded, from huggingface_hub's metadata.

    Every `.metadata` file under `.cache/huggingface/download/` records the
    repo commit hash on its first line. One download = one revision, but that
    is an assumption worth checking, so read them ALL and refuse to pick if
    they disagree.
    """
    metas = sorted((dataset_dir / ".cache/huggingface/download").rglob("*.metadata"))
    revs = set()
    for m in metas:
        with m.open() as f:
            revs.add(f.readline().strip())
    if not revs:
        return None
    if len(revs) > 1:
        raise SystemExit(
            f"{dataset_dir.name}: {len(revs)} distinct revisions in download "
            f"metadata ({sorted(revs)}) — the local copy is a MIX of commits "
            "and cannot be manifested as one revision. Re-fetch."
        )
    return revs.pop()


def build() -> int:
    tokenizers = {}
    for tj in sorted(TOKENIZER_DIR.glob("*.json")):
        tokenizers[f"models/tokenizers/{tj.name}"] = sha256_file(tj)

    specials = json.loads((TOKENIZER_DIR / "special_tokens.json").read_text())

    sources = {}
    for name, info in DATASETS.items():
        entry = dict(info)
        d = HF / name
        if d.is_dir():
            rev = hf_revision(d)
            if rev:
                entry["revision"] = rev
            # The files actually on disk, not the pattern that requested them.
            # (revision, file list) is what makes a re-fetch byte-comparable;
            # a stride formula quoted from the fetch script is one refactor
            # away from being a lie about historical data.
            parquet = sorted(q.relative_to(d).as_posix()
                             for q in d.rglob("*.parquet"))
            if parquet:
                entry["files"] = parquet
        sources[name] = entry

    bins = {}
    for b in BINS:
        mf = MIX / f"{Path(b).stem}.manifest.json"
        if not mf.exists():
            print(f"build: missing per-bin manifest {mf}", file=sys.stderr)
            return 1
        if not (MIX / b).exists():
            # A sidecar whose bin was deleted describes nothing; manifesting
            # it would pin a hash no artifact carries.
            print(f"build: {mf.name} exists but {b} does not — stale sidecar",
                  file=sys.stderr)
            return 1
        m = json.loads(mf.read_text())
        entry = {"sha256": m["sha256"], "tokens": m["tokens"]}
        for k in ("bytes_per_token", "max_token_id", "composition", "repetition", "seed"):
            if k in m:
                entry[k] = m[k]
        bins[b] = entry

    unexpected = sorted(p.name for p in MIX.glob("*.bin") if p.name not in BINS)
    if unexpected:
        print(f"build: unexpected bins in {MIX}: {unexpected}", file=sys.stderr)
        return 1

    manifest = {
        "_comment": "Committed identity of the v2 pretraining corpus. The bins "
                    "are local-only; this is what makes them checkable and "
                    "rebuildable. Regenerate with tools/hfcorpus/manifest.py "
                    "build; verify with check.",
        "format": "little-endian u16 token ids, no header (load_mmap(path, 3))",
        "tokenizer_of_record": "models/tokenizers/nsl_mix_v2_t40960_v49152.json",
        "tokenizers_sha256": tokenizers,
        "reserved": {"surfaces": specials["surfaces"], "ids": specials["ids"]},
        "sources": sources,
        "bins": bins,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)}: {len(bins)} bins, "
          f"{len(tokenizers)} tokenizer hashes, {len(sources)} sources")
    return 0


def fail(msgs: list[str], msg: str) -> None:
    print(f"  FAIL  {msg}")
    msgs.append(msg)


def check(hash_bins: bool) -> int:
    if not MANIFEST.exists():
        print(f"check: {MANIFEST} missing — run build first", file=sys.stderr)
        return 1
    man = json.loads(MANIFEST.read_text())
    bad: list[str] = []

    for rel, want in man["tokenizers_sha256"].items():
        p = REPO / rel
        if not p.exists():
            fail(bad, f"{rel}: file missing")
        elif (got := sha256_file(p)) != want:
            fail(bad, f"{rel}: sha256 {got[:12]}… != manifest {want[:12]}…")
        else:
            print(f"  PASS  {rel} hash")

    tok = json.loads((REPO / man["tokenizer_of_record"]).read_text())
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
    for surface, tid in man["reserved"]["ids"].items():
        if added.get(surface) != tid:
            fail(bad, f"{surface}: tokenizer id {added.get(surface)} != manifest {tid}")
    print(f"  PASS  {len(man['reserved']['ids'])} reserved ids match the tokenizer"
          if not bad else "")

    if hash_bins:
        for b, entry in man["bins"].items():
            p = MIX / b
            if not p.exists():
                fail(bad, f"{b}: bin missing locally")
                continue
            got = sha256_file(p)
            if got != entry["sha256"]:
                fail(bad, f"{b}: sha256 {got[:12]}… != manifest {entry['sha256'][:12]}…")
            else:
                print(f"  PASS  {b} ({entry['tokens']:,} tokens)")

    print(f"check: {'FAIL — ' + str(len(bad)) + ' problem(s)' if bad else 'OK'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check"])
    ap.add_argument("--hash-bins", action="store_true",
                    help="also re-hash the local 45 GB of bins (check only)")
    args = ap.parse_args()
    return build() if args.mode == "build" else check(args.hash_bins)


if __name__ == "__main__":
    sys.exit(main())
