# Item 5 — the v2 corpus finalized for evaluation (2026-08-25)

Roadmap row: "Finalize the v2 corpus for model evaluation. Add a
repository-disjoint Stack/code validation set and benchmark
decontamination." The 22.6B `pretrain_train.bin` is untouched — this adds
the evaluation half of its identity.

## stack_val — repository-disjoint, and VERIFIED so

- One fresh Stack v3 shard (index **204**, same pinned revision as train)
  fetched into `data/hf/stack-val/` — a separate directory, because
  `manifest.py` globs `data/hf/stack/**` into the *train* source's file
  list, and a val shard dropped there would silently join the training
  provenance. `fetch.py --only stack-val` refuses indices on the training
  stride (multiples of 409).
- Extracted with `--max-bytes 60000000` (the `web_val` convention):
  **1,774 documents from 1,773 repositories → 15,377,827 tokens** at
  3.904 B/tok (`stack_val.bin`, 30.8 MB).
- Disjointness is EVIDENCE, not an argument: Stack v3 rows carry a
  `repo_id` column the pipeline previously read and discarded. The 20
  train shards hold 421,948 repos with zero cross-shard duplicates, and
  `verify.py` now intersects the train and val `repo_id` columns and
  requires an empty overlap (currently 421,948 ∩ 1,773 = 0).

## Benchmark decontamination — method stated, results committed

`tools/hfcorpus/decontaminate.py`: exact normalized-line matching against
**HumanEval** (prompt + canonical_solution + test) and **MBPP**
(full + sanitized; text/prompt + code + test_list) — 7,247 reference lines
≥ 30 chars with import/from boilerplate excluded; a document is
contaminated at ≥ 2 distinct matched lines or one matched line ≥ 60 chars.
Catches verbatim inclusion only, not paraphrase; the method string is
recorded verbatim in the manifest so the claim's limits travel with it.
Both benchmarks are pinned as revisioned manifest sources like every other
input; they never enter any corpus bin.

| set | documents | contaminated | disposition |
|---|---:|---:|---|
| **val-stack** | 1,774 | **0** | must be clean (gated) |
| **val-web** | 20,145 | **0** | must be clean (gated) |
| **val-sft** | 2,858 | **0** | re-cut: the first scan found **14 genuine MBPP problems** (verbatim test asserts) in the distillation validation split; `extract.py --drop-contaminated` removed them |
| stack-train | 424,728 | 17 (0.004%) | reported |
| web-train | 14,587,198 | 0 | reported |
| sft-train | 52,205 | **273 (0.52%)** | reported — the distillation traces contain benchmark problems at a real rate; any HumanEval/MBPP evaluation of an SFT'd model must be read against this committed number |
| code-train | 4,152 | 0 | reported |

Scan cost: 73 GB in 13.4 min. The dropping filter and the verifying scan
share ONE rule (`decontaminate.doc_is_contaminated`, imported by
`extract.py`), so they cannot disagree — the single-copy doctrine the
reserved-surface machinery established.

First-contact lessons: the method needed one refinement on real data —
`from collections import counter` (exactly 30 chars) flagged an innocent
repository, so import lines are excluded from the reference set as
boilerplate that carries no benchmark identity. And `sft_val` shrank from
2,258,310 to **2,251,516 tokens**; its sha changed accordingly.

## What now holds the line

- `verify.py`: train/val head+tail disjointness generalized to ALL pairs
  (sft_val previously had NO disjointness coverage), the repo_id
  intersection, and the decontamination record — 46/46 checks.
- `corpus_manifest_gate.rs`: 8 bins pinned; a new test requires the
  decontamination record complete (method stated, benchmarks pinned as
  sources, every val set present and clean, every train set reported).
- `manifest.py build` REFUSES to run without a decontamination report — a
  corpus manifested for evaluation without one is a claim with no evidence.
