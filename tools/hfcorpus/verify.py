#!/usr/bin/env python3
"""Check the properties the corpus has to have before a run is worth starting.

Every check here is one that a broken corpus would otherwise pass silently:
a tokenizer whose reserved surfaces fragment into several tokens, a stream whose
ids do not fit the u16 format the loader mmaps, a document separator that was
never actually emitted, or a train/val pair that overlaps.

Usage (from the repo root):
    tools/hfcorpus/.venv/bin/python tools/hfcorpus/verify.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOKBENCH = REPO / "tools/tokbench/target/release/tokbench"
# Both halves of the special-id check must describe the SAME build. Reading the
# tokenizer from the committed copy and the ids from the training scratch
# directory would compare one build's surfaces against another's ids and call
# it agreement.
TOKENIZER_DIR = REPO / "models/tokenizers"
TOKENIZER = TOKENIZER_DIR / "nsl_mix_v2_t40960_v49152.json"
SPECIALS_FILE = TOKENIZER_DIR / "special_tokens.json"

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    print("== tokenizer ==")
    if not TOKENIZER.exists():
        print(f"  tokenizer missing: {TOKENIZER}", file=sys.stderr)
        return 1
    tok = json.loads(TOKENIZER.read_text())
    vocab = tok["model"]["vocab"]
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
    total = len(vocab) + sum(1 for c in added if c not in vocab)
    check(total <= 65536, "vocabulary fits the u16 stream", f"{total} ids")

    specials = json.loads(SPECIALS_FILE.read_text())["ids"] if SPECIALS_FILE.exists() else {}
    check(bool(specials), "reserved ids recorded", str(specials))
    for surface, tid in specials.items():
        check(added.get(surface) == tid,
              f"{surface} is one added token at the recorded id",
              f"tokenizer says {added.get(surface)}, record says {tid}")

    print("\n== corpora ==")
    mix = REPO / "data/tokens/mix"
    manifests = sorted(mix.glob("*.manifest.json")) if mix.is_dir() else []
    check(bool(manifests), "at least one tokenized corpus exists")

    eos = specials.get("<|endoftext|>")
    for mf in manifests:
        m = json.loads(mf.read_text())
        binp = REPO / m["output"]
        if not binp.exists():
            check(False, f"{binp.name} present")
            continue
        arr = np.memmap(binp, dtype="<u2", mode="r")
        check(arr.size == m["tokens"],
              f"{binp.name}: token count matches its manifest",
              f"{arr.size:,} on disk vs {m['tokens']:,} recorded")
        # `max_token_id` is written by pretokenize.py; make_mix.py manifests
        # describe a composition instead. Both land here, so neither key may be
        # assumed -- a KeyError would skip the mixed corpus entirely, which is
        # the one that actually gets trained on.
        if "max_token_id" in m:
            check(m["max_token_id"] <= 65535, f"{binp.name}: ids fit u16",
                  f"max {m['max_token_id']}")
        if "composition" in m:
            share = sum(c["share"] for c in m["composition"].values())
            check(abs(share - 1.0) < 1e-3,
                  f"{binp.name}: composition shares sum to 1", f"{share:.5f}")
        if (m.get("doc_sep_token") or "composition" in m) and eos is not None:
            # Sample rather than scan: presence is what is being established.
            head = np.asarray(arr[: 1 << 24])
            check(int((head == eos).sum()) > 0,
                  f"{binp.name}: end-of-document id is actually present")

    print("\n== train/val disjointness ==")
    # Every (train, val) pair, not a hardcoded web-only check (item 5): the
    # sft pair silently had NO disjointness coverage before.
    for stem in ("web", "sft", "stack"):
        train = mix / f"{stem}_train.bin"
        val = mix / f"{stem}_val.bin"
        if not (train.exists() and val.exists()):
            # Say so. A section that prints nothing when its inputs are
            # absent reads exactly like a section whose checks all passed.
            absent = [p.name for p in (train, val) if not p.exists()]
            check(False, f"{stem} train/val disjointness could not be checked",
                  f"missing {', '.join(absent)}")
            continue
        a = np.memmap(train, dtype="<u2", mode="r")
        b = np.memmap(val, dtype="<u2", mode="r")
        n = min(1 << 20, a.size, b.size)
        # The split is cut at file/shard granularity, so no window of the val
        # stream should appear at the head or tail of the train stream.
        check(not np.array_equal(np.asarray(a[:n]), np.asarray(b[:n])),
              f"{stem} train and val heads differ")
        # Compare TAIL to TAIL. The failure this guards against is val having
        # been cut as a suffix of the same stream, i.e. a[-val.size:] == b; that
        # leaves a[-n:] equal to b[-n:], while a[-n:] vs b[:n] differs and the
        # check would pass on real contamination.
        check(not np.array_equal(np.asarray(a[-n:]), np.asarray(b[-n:])),
              f"{stem} val is not a suffix of the train stream")

    print("\n== stack repo-id disjointness ==")
    # The val shard sits off the training stride, but partitioning is an
    # ARGUMENT — the repo_id columns are EVIDENCE. Empty intersection or fail.
    train_dir = REPO / "data/hf/stack/data"
    val_dir = REPO / "data/hf/stack-val/data"
    if not (train_dir.is_dir() and any(val_dir.glob("*.parquet")) if val_dir.is_dir() else False):
        check(False, "stack repo-id disjointness could not be checked",
              "stack or stack-val parquet missing")
    else:
        import pyarrow.parquet as pq

        def ids(d):
            out = set()
            for f in sorted(d.glob("*.parquet")):
                out.update(pq.read_table(f, columns=["repo_id"])["repo_id"].to_pylist())
            return out

        train_ids = ids(train_dir)
        val_ids = ids(val_dir)
        overlap = train_ids & val_ids
        # BOTH sides must be non-empty or the intersection proves nothing —
        # a cleared train dir would otherwise pass '0 repos ∩ val = 0'.
        check(len(train_ids) > 0, "stack train carries repositories",
              f"{len(train_ids)} repos")
        check(len(val_ids) > 0, "stack-val carries repositories",
              f"{len(val_ids)} repos")
        check(not overlap,
              f"stack train ({len(train_ids)} repos) and val "
              f"({len(val_ids)} repos) share no repo_id",
              f"{len(overlap)} shared")

    print("\n== benchmark decontamination record ==")
    decon = REPO / "data/text/decontamination.json"
    if not decon.exists():
        check(False, "decontamination report exists",
              "run tools/hfcorpus/decontaminate.py")
    else:
        rep = json.loads(decon.read_text())
        check(bool(rep.get("method")), "decontamination method is stated")
        for name in ("val-stack", "val-web", "val-sft"):
            entry = rep.get("sets", {}).get(name)
            if entry is None:
                check(False, f"decontamination covers {name}", "not scanned")
            else:
                check(entry.get("contaminated_documents") == 0,
                      f"{name} is clean of benchmark text",
                      f"{entry.get('contaminated_documents')} contaminated")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
