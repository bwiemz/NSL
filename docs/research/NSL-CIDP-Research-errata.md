# Compile-Integrated Data Pipeline (CIDP) — Errata

**Companion to** `NSL-CIDP-Research.md.pdf`. **Date:** 2026-11-13.

This file lists the statements in the April 2026 CIDP proposal that a reader
should not rely on, and says what is true instead. The proposal was checked
in November 2026 against its own arithmetic, the papers it cites, later
literature, and the NSL repository. Each entry says how it was checked; where
a point rests on a secondary summary or on memory of a paper rather than a
fresh reading, the entry says so. Section numbers are those printed in the
PDF.

---

## Part 1 — Errors in the proposal

### E1 — The deduplication design does not do what its parameters say

**Proposal.** §2.2 configures `n_hashes = 128`, `bands = 16`, `threshold =
0.8`, `scope = batch`, and on any band hit executes "weight[s] *= 0.1". It
sizes the filter as "~100MB for 10B token dataset" (64 MB in the §4 report)
and says "The Bloom filter accumulates over the training epoch."

**Correction.** MinHash with locality-sensitive hashing (LSH) splits a
signature into bands; two sequences that agree on a whole band become a
candidate pair.

- There is no verification step, so `threshold = 0.8` is never used. In
  standard LSH a band hit only nominates a candidate, which is then compared
  properly. Here a hit multiplies the weight at once. With 16 bands of 8
  rows a sequence pair of Jaccard similarity `s` is hit with probability
  `1 − (1 − s⁸)¹⁶`:

  | Jaccard similarity | 0.5  | 0.6   | 0.674 | 0.8   |
  |--------------------|-----:|------:|------:|------:|
  | Probability of hit | 6.1% | 23.7% | 50%   | 94.7% |

  Half of all pairs at similarity 0.674 are penalised, and a quarter at 0.6.
- The penalty is not soft. The weight is multiplied once per band hit, and
  an exact duplicate hits all 16 bands, so its weight is 0.1¹⁶ = 10⁻¹⁶.
- Bloom-filter false positives compound. 10B tokens in 2048-token windows is
  4.88M windows, and 16 band keys each makes 78M keys. A 64 MB filter gives
  6.6 bits per key and a 4.3% false-positive rate per lookup, so by the end
  of an epoch 1 − (1 − 0.043)¹⁶ = 51% of clean sequences take at least one
  false ×0.1. At 100 MB the figures are 10.2 bits per key, 0.73% and 11%.
- The unit is wrong. Signatures are computed on fixed 2048-token windows of
  the packed stream. The same document at a different offset, with different
  neighbours, has low window-level similarity and is missed. Deduplication
  has to work per document, before packing.
- Neither scope works. `scope = batch` compares 32 windows with each other,
  which is useless. A filter that accumulates over the epoch marks every
  sequence as a duplicate in the second epoch unless it is reset.
- Relevant prior work is not cited. LSHBloom
  (https://arxiv.org/abs/2411.04257) is this index done carefully. FineWeb
  uses 5-grams and 112 hashes in 14 bands of 8, which gives a 77% hit
  probability at similarity 0.75, and found that global MinHash
  deduplication made results worse than deduplicating within each crawl
  dump; that is a caution against `scope = global`. DCLM deduplicates with a
  Bloom filter, not MinHash (from secondary summaries of
  https://arxiv.org/abs/2406.11794).

**Source.** Numeric check using the standard LSH candidate-probability and
Bloom false-positive formulas; https://arxiv.org/abs/2406.17557 (FineWeb)
and https://arxiv.org/abs/2411.04257.

### E2 — Downweighting duplicates saves no compute, so the claimed speedup does not follow

**Proposal.** §2.2: "This matches SoftDedup's finding that soft
deduplication achieves the same perplexity with 26% fewer steps while
preserving accuracy." §4 report: "Effective training speedup from dedup:
~12%" and "Low-quality downweighted: ~22% of web data". §5 table: "Online
model-loss (GPU, zero extra cost)".

**Correction.** SoftDedup lowers the probability of sampling a sequence,
using an n-gram "commonness" score. That is why it reaches baseline
perplexity in at least 26% fewer steps. CIDP instead multiplies the loss of
a sample that it runs forward and backward anyway. No compute is saved, so
neither the 12% speedup nor "zero extra cost" filtering follows.

The proposal's own sample output in §4 shows the cost. If 22% of web data is
downweighted to 0.1, it still takes 22% of the compute and contributes
0.22 × 0.1 / (0.78 + 0.22 × 0.1) = 2.7% of the gradient weight.

**Source.** https://aclanthology.org/2024.acl-long.220; arithmetic recomputed
from the proposal's figures.

### E3 — Several mechanisms cannot work as written

Each item gives the proposal's statement, then the correction.

- §2.1, "grad_weighted = weight[s] × grad[s]", described as "a single
  scalar multiply per sample" in the fused optimizer step. A batched
  backward pass never materializes per-sample gradients. Only the loss can
  be weighted.
- §2.4, "Score each sample: difficulty = forward_pass_loss(sample)". That is
  an extra forward pass per sample, about a third of a training step, which
  contradicts the §4 figure "Data pipeline overhead: 0.8% of total training
  step time". The scores are also stale by up to 10K samples.
- §2.4, drawing from the easy end of a streaming priority queue. The queue
  converges to holding everything except the 10K hardest samples seen so
  far, and releases those in a burst when the phase changes.
- §3, "No CPU involvement in the data path", with the mmap read drawn on the
  GPU, and §2.2, "using idle SMs during data loading". A memory-mapped file
  is host memory. A token batch is 32 × 2048 × 4 bytes, about 262 KB. A
  training stream that saturates the GPU leaves no idle streaming
  multiprocessors (SMs).
- §2.5, "special_count = count(tokens where token < special_token_boundary)".
  This assumes special token ids are the lowest ids. In this repository they
  are 41190, 41256 and 49145–49149
  (`models/tokenizers/special_tokens.json`).
- §2.5, `max_consecutive_repeat = 10` with weight 0. One local run of a
  repeated token discards a whole 2048-token window, and such runs are
  common in code.
- §2.3, "When --weights is provided, the compiler computes per-source
  perplexity". Compile-time mixing therefore needs a trained model before
  pretraining starts.
- §1.2, "NSL's dataset block is a compiled construct" and "The compiler
  generates native code for this data pipeline." Neither is true of the
  repository; see Part 3.

**Source.** Logic, the proposal's own figures, and a read of the named
repository file.

### E4 — Online loss-threshold weighting biases training in ways the proposal does not discuss

**Proposal.** §2.1: "samples with extremely high loss are likely noise or
out-of-distribution", with per-phase windows such as "threshold = [0.1,
10.0]" and "threshold = [1.0, 3.0]", the last commented "focus on hardest
useful samples". §4 report: "model perplexity on code is 2.3× higher than
web → needs more code".

**Correction.** The first five points are judgements about the design; the
threshold point is arithmetic.

- It changes the objective. The weights are a function of the loss that is
  not differentiated through, so the model optimizes a truncated loss that
  emphasizes mid-loss sequences.
- High loss is mostly irreducible entropy, not noise. Rare domains, names,
  numbers and other languages are systematically suppressed. RHO-LOSS and
  DoReMi avoid this by using reducible or excess loss against a reference
  model (from memory; not re-checked against the sources).
- It is self-reinforcing. A downweighted high-loss region is never learned,
  stays high-loss and stays excluded.
- Downweighting low-loss data invites forgetting.
- The source mixer has the same flaw. Reading higher perplexity on code as
  "needs more code" conflates reducible and irreducible loss. Code is
  normally the lower-perplexity source, and the rule drifts toward the
  noisiest source.
- The absolute thresholds are miscalibrated. With this repository's
  49,152-token vocabulary every sequence starts near ln(49152) = 10.8, above
  the phase-1 upper bound of 10, so phase 1 downweights everything, which
  normalizes to uniform weights. The mean loss over 2048 tokens is tightly
  concentrated. With a typical late-training loss above 3.0, the phase-3
  window [1.0, 3.0] upweights the easier-than-average minority, the opposite
  of its comment.

Published evidence points the other way on two counts. In DCLM's own
comparison, perplexity filtering scored far below the fastText classifier,
CORE 29.0 against 41.0 (from a secondary summary of
https://arxiv.org/abs/2406.11794, not the paper's table). The closest
published method, Sow et al. (ICLR 2025, https://arxiv.org/abs/2502.06733),
reports that deprioritizing redundant or uninformative data works best. That
means low-loss samples (from memory; the abstract does not spell it out),
whereas CIDP downweights the high-loss tail too.

**Source.** As given per item; `ln(49152)` recomputed.

### E5 — The cited quality-filtering paper says roughly the opposite

**Proposal.** §1.1: "a category of "low-quality" data actually helps (as
shown by the Data-Quality Illusion paper)". §2.1: the paper "showed that CQF
filtering can remove data that actually helps training."

**Correction.** CQF is classifier-based quality filtering. The paper is
arXiv 2510.00866 (Nait Saada et al.), now titled "Removing Noise, not
Finding Gold: Quality Filtering for Large-Scale Pretraining" (ICML 2026). It
says that classifier filtering improves downstream performance, and can even
beat training on the high-quality set itself because it implicitly filters
that set too. Its conclusion is that the benefit comes from removing noise.
It does not claim that low-quality data helps.

**Source.** https://arxiv.org/abs/2510.00866

### E6 — The curriculum paper's remedy is not the one the proposal builds

**Proposal.** §2.4 cites the "Curriculum-LR paper, Nov 2025" for the
conflict with learning-rate (LR) decay, then specifies "effective_lr[sample]
= lr × (difficulty / mean_difficulty)".

**Correction.** Luo et al. (arXiv 2511.18903, ICLR 2026) order data by
ascending quality, and fix the clash with LR decay by using a milder decay
or by averaging the last checkpoints, for +1.64% over shuffling. The
proposal applies the finding to an easy-to-hard ordering by model loss, which
is a different axis, and the per-sample LR formula is its own invention.

Under AdamW that formula is a loss reweight, not a change of learning rate.
It upweights high-loss samples, while the quality scoring of §2.1 downweights
them, and both multiply into the same `batch.weights`.

**Source.** https://arxiv.org/abs/2511.18903

### E7 — Citation slips and unverified figures

- §2.3 and reference 4, "the perplexity-correlation data selection approach
  (Mizrahi et al., ICLR 2025)". Two papers are fused. Mizrahi et al. is BETR
  (https://arxiv.org/abs/2507.12466, July 2025), which ranks documents by
  similarity to benchmark examples; it is not an ICLR 2025 paper. Perplexity
  correlations is Thrush, Potts and Hashimoto (ICLR 2025,
  https://arxiv.org/abs/2409.05816). It selects domains whose losses, across
  many public models, correlate with benchmark scores. It does not upweight
  whatever the current model finds high-perplexity, which is what §2.3 does.
- Reference 6, "Wettig, A., et al." for Data Mixing Laws. The authors are
  Ye, Liu, Sun, Zhan, Zhou and Qiu
  (https://iclr.cc/virtual/2025/poster/28624).
- Reference 5. The FineWeb paper's title is "The FineWeb Datasets: Decanting
  the Web for the Finest Text Data at Scale".
- §1.1 and §2.2, "~1.4TB RAM" for MinHash over 100T tokens. The proposal
  gives no source and the figure could not be verified.

The §2.3 figures for LLaMA's mix, 67% web and 4.5% code, are right (from
memory; not re-checked against the source).

---

## Part 2 — Superseded by later work

- **ADAPT** (Zhao et al., ICLR 2026, https://arxiv.org/abs/2605.05227; only
  the abstract was read). It reweights samples online without changing the
  dataset size, and reports better generalization than offline selection at
  equal compute. That supports the proposal's general direction, but its
  quality signal is similarity-based, not a raw loss threshold. Together
  with Sow et al. (E4) it is the baseline a CIDP-style feature must beat.
- **The quality-filtering paper's final version** (version 4, June 2026,
  https://arxiv.org/abs/2510.00866). It adds an optimization-driven
  definition of quality estimated from small proxy runs. That is a cheap,
  offline alternative to scoring with the model in the loop.
- **Luo et al.** (https://arxiv.org/abs/2511.18903). The paper implies the
  inexpensive gain is checkpoint averaging with a milder LR floor, not a
  curriculum scheduler.
- **D³** (https://arxiv.org/abs/2605.31164; found by search, not read).
  Other loss-based scheduling work from 2026 exists; a revision should at
  least look at it.

---

## Part 3 — What the code implements

No part of the proposal is implemented: there are no `cidp_*.rs` files, no
`quality:`, `dedup:`, `curriculum:`, `validate:` or `mixing:` sections (a
`quality` field in a dataset block is rejected as an unknown field), no
`--cidp-report` flag, no weighted cross-entropy in the standard library, and
no per-sample loss path.

Older, adjacent code that a reader might mistake for it:

- **The `dataset` block.** NSL source can write one, and the checker accepts
  `source`, `format`, `transform`, `filter`, `shuffle`, `packing`,
  `max_samples`, `shuffle_buffer`, `sequence_length`, `resume_from` and
  others (`crates/nsl-semantic/src/checker/block.rs`). Code generation reads
  only its packing statistics, for attention planning
  (`crates/nsl-codegen/src/pca_activation.rs`). The block loads, filters and
  shuffles nothing, so it is not the compiled pipeline §1.2 describes. No
  shipped model uses one; all use `load_mmap` and `DataLoader(...)`.
- **The DataLoader.** `crates/nsl-runtime/src/dataloader.rs`, reached from
  NSL source through `DataLoader(...)`. It is a multi-threaded Rust loader
  over a flat token tensor with a seeded shuffle, resume, rank sharding and
  optional packing at end-of-sequence tokens
  (`crates/nsl-runtime/src/packing.rs`). It has no weights, filters or
  deduplication. It is already free of Python, so the "zero Python overhead"
  gain the proposal claims is already in hand.
- **Offline corpus tools.** `tools/hfcorpus/` is Python, run before training
  and not reachable from NSL source. It assembles FineWeb sample-10BT
  (already filtered and deduplicated per dump upstream), Stack v3 and the
  repository corpus; removes benchmark contamination by exact line match
  (`decontaminate.py`); mixes at document granularity and reports repetition
  (`make_mix.py`); and runs checks on the output (`verify.py`). The project
  adds no deduplication or quality filter of its own.

The offline tools are where the code and the proposal differ by design, and
the code has the better of it: for a 50M to 1B parameter model on about 20B
tokens this is a reasonable design and is closer to what the literature
supports than CIDP.

`docs/wiki/Glossary.md` used to describe CIDP as existing compiler
behaviour. It now says the proposal is not implemented.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Do not build the GPU MinHash and Bloom path, the loss-threshold schedule,
   the priority-queue curriculum or the perplexity-driven mixer as specified.
2. Keep data quality offline. Add exact-hash document deduplication across
   all sources, then MinHash (FineWeb's 14 × 8 with candidate verification)
   inside the code corpus only, per source, and report the rates.
3. If online weighting is still wanted, build only the sound core as an
   experiment: a per-sequence source id, weighted cross-entropy, per-source
   loss logging, then fixed per-source weights and a Sow-et-al-style in-batch
   reweight against the unweighted baseline at equal compute.
4. Test checkpoint averaging (Luo et al.) as a separate, cheap arm.

---

## Part 5 — Implementation defects found in the same review

The review that checked the proposal also ran the code. CIDP itself is not
implemented, so this part lists what was found wrong in the adjacent code
that Part 3 describes, as distinct from the proposal's own errors. Fixed
items are described in `CHANGELOG.md` under "Unreleased". The open list is
as of 2026-11-13.

### Fixed

- **A packed row with more than 256 documents hung training.** The row
  panicked in a `DataLoader` worker thread and the run then hung, or, with
  one worker, ended the epoch early with exit status 0. A worker panic now
  aborts the process (`crates/nsl-runtime/src/dataloader.rs`; test
  `crates/nsl-cli/tests/packing_separator_gate.rs`).
- **`DataLoader(..., packing = true)` without `pack_separator` defaulted
  the separator to token id 0.** Leaving `pack_separator` out is now a
  compile error.
- **The Glossary described CIDP as existing compiler behaviour.** The entry
  in `docs/wiki/Glossary.md` now says the proposal is not implemented.

### Open, in the adjacent code

- **The train/val disjointness check passes on real overlap.** It compares
  only the first and the last 2²⁰ tokens of the two files, so a validation
  set copied whole from the middle of the training set passes. Only the
  Stack split has real evidence, from repository ids. Where: `main` in
  `tools/hfcorpus/verify.py`. Reproduced by running. It should hash every
  validation document, test it against the training set, and fail on a hit.
- **A `dataset` block accepts fields and honours none of them.** A block
  with a nonexistent `source` file, a `filter` that rejects everything,
  `shuffle`, `max_samples` and `format = parquet` compiles and runs with no
  effect and no error; `transform`, `shuffle_buffer` and `resume_from` are
  the same. Where: `check_dataset_def` in
  `crates/nsl-semantic/src/checker/block.rs`. Reproduced by running.
- **`DataLoader(..., shuffle = true)` permutes batches, not sequences.**
  Each batch is a fixed contiguous slice of `batch_size × seq_len` tokens,
  so the same sequences meet in the same order every epoch, and each batch
  is one contiguous region of the corpus. Where: `start_workers_at` in
  `crates/nsl-runtime/src/dataloader.rs`. By reading the code only. Right
  behaviour: permute window indices and gather `batch_size` per batch.
- **The decontamination rate has blind spots.** Only benchmark lines of at
  least 30 normalized characters are matched, so a benchmark solution made
  of short lines is invisible, and contamination in training sets is
  reported, not removed. This is documented behaviour. Where:
  `benchmark_lines` and `doc_is_contaminated` in
  `tools/hfcorpus/decontaminate.py`. By reading the code only.
- **Mixed-in documents are placed by document index, not token position.**
  Injected documents are spread uniformly over the base corpus's document
  indices, so the local mixture share varies with base document length.
  The effect is small. Where: `main` in `tools/hfcorpus/make_mix.py`. By
  reading the code only.
