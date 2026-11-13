# CPKD (Compiler-Planned Knowledge Distillation) — Errata

**Companion to** `CPKD.pdf`. **Date:** 2026-11-13.

This file lists the statements in the CPKD proposal that should not be relied
on, and what is true instead. The proposal was written in April 2026 and
checked in November 2026 against its own arithmetic, the literature, and the
code that implements it. The arithmetic below was recomputed; external facts
were checked against the linked source unless the entry says otherwise; and
statements about the code describe the repository on the date above, where
the fused loss kernel was also compared numerically with PyTorch. The PDF
itself is unchanged.

Abbreviations: KL is Kullback-Leibler divergence, JSD is Jensen-Shannon
divergence, CE is cross-entropy, LM is language model, AD is automatic
differentiation, GEMM is a general matrix multiply, HBM is the proposal's
name for GPU main memory, and SM is a GPU streaming multiprocessor.

---

## Part 1 — Errors in the proposal

### E1 — The loss kernel needs vocabulary-wide normalisers, and the gradient needs a second pass

**Proposal.** §2.2, kernel pseudo-code. Inside the loop over vocabulary tiles
it computes "t_probs_tile = softmax(t_logits_tile / T)" and
"s_log_probs_tile = log_softmax(s_logits_tile / T)", and accumulates
`grad_hidden` in the same loop.

**Correction.** A softmax over one tile is not the softmax over the
vocabulary; the normaliser is known only after the last tile. The loss can
still be computed in one pass by carrying an online log-sum-exp per row, but
the gradient needs the finished normalisers and therefore a second pass over
the tiles. The CE term also needs its own normaliser at T = 1, which the
pseudo-code never computes. The gradient formula itself,
"alpha * d_CE + (1-alpha) * T * (s_probs - t_probs)", is right once the
probabilities are the vocabulary-wide ones. The implemented kernel is built
this way: three log-sum-exp accumulations in the forward (teacher at T,
student at T, student at 1), then a backward that recomputes the logits.

**Source.** Logic; the implemented kernel was compared numerically with
PyTorch (Part 3).

### E2 — A frozen teacher saves no activations; the 8 GB line does not exist

**Proposal.** §1.1 lists "Teacher activations (for logits)" at 8.0 GB,
annotated "saved for KL loss backward", and concludes "The teacher model
(14 GB) plus its activations (8 GB) consume 22 GB". §6 repeats it:
"~8 GB (activations saved)" for PyTorch against "0 (dead gradient
elimination)" for CPKD.

**Correction.** A frozen teacher run under `torch.no_grad()` saves nothing
for a backward pass. Only one layer's transient activation is live at a time.
The 8 GB line should not be in the PyTorch baseline, which leaves
41 - 8 = 33 GB before logits. The §6 row repeats the same nonexistent cost.

### E3 — The 41 GB to 22.9 GB comparison is between different models, and none of the 18.1 GB comes from the fused loss

**Proposal.** §5 build report: "Memory saved by CPKD: 18.1 GB
(44% reduction)", measured against the §1.1 baseline of "~41+ GB".

**Correction.** Each sum is right, but the two tables describe different
models and neither includes the logit tensors.

| Line (GB) | §1.1 baseline | §5 report | Difference | What the difference is |
|---|---|---|---|---|
| Teacher weights | 14.0 | 11.6 | 2.4 | a smaller teacher (5.8B, not 7B) |
| Student weights + FP32 master | 6.0 | 6.6 | -0.6 | a larger student (1.1B, not 1B) |
| Student optimizer | 8.0 | 3.3 | 4.7 | CPDT, a different subsystem |
| Teacher activations | 8.0 | 0.5 | 7.5 | the line that should not exist (E2) |
| Student activations | 3.0 | 0.8 | 2.2 | FASE and CSHA, other subsystems |
| Gradient buffer | 2.0 | 0.1 | 1.9 | FASE |
| Total | 41.0 | 22.9 | 18.1 | 18.1 / 41.0 = 44% |

The logits are the "+" in "41+" and appear in neither sum. CPKD's own
contribution is the roughly 2.1 GB of logits at batch 4 (see E9), which is
separate from the 18.1 GB. The rest is a smaller teacher, the fictitious
activation line, and savings credited to other proposals.

The baseline has three further problems. It counts a BF16 student copy and
FP32 master weights as separate allocations, which PyTorch automatic mixed
precision does not hold separately. Its 2 GB gradient buffer for 1B FP32
parameters would be 4 GB. And it has no CUDA-context line, although the CFA
proposal's build report budgets 800 MB for one.

**Source.** Recomputed from the proposal's own figures.

### E4 — The spectral rank of the LM head does not give a top-k

**Proposal.** §2.4 takes the 90% energy rank of the LM head's singular
values and concludes, for a rank of 800, "top-k=800 captures 90% of the
signal". It adds "For LLaMA-7B, the effective rank is ~500-2000".

**Correction.** The singular values of the LM head (a d_model × vocab matrix,
so of rank at most d_model) describe the dimension of the subspace the logits
live in. Top-k truncation is sparsity in the vocabulary basis, chosen per
token. The second does not follow from the first. The 500-2000 figure for
LLaMA-7B has no source and could not be verified. Top-k truncation of teacher
probabilities is also a biased estimator that miscalibrates the student.

**Source.** Logic for the first point. For the bias of top-k truncation:
"Sparse Logit Sampling", ACL 2025, https://aclanthology.org/2025.acl-long.885

### E5 — The shared-computation fusion of §2.1 saves nothing

**Proposal.** §2.1: "Read x_t from HBM (once)" and "Read x_s from HBM
(once)", concluding "this saves 16 kernel launches and 16 HBM reads".

**Correction.** The fused listing reads two different tensors once each,
which is the same two reads as the unfused listing above it. No mechanism is
given for the 16 saved launches or reads. The shared embedding of the same
section, one lookup projected to both widths, changes the teacher's inputs,
which makes it a different model.

### E6 — The two-stream schedule contradicts itself

**Proposal.** §2.3. The pipeline diagram runs "[Stream 0]
fused_kl_ce(t_hidden_final, s_hidden_final) → loss + grad" alongside
"[Stream 1] student.backward() begins". The text says the teacher forward
"still occupies the GPU during student idle time" and that two CUDA streams
run concurrently with "the GPU's SMs split between teacher and student".

**Correction.** The student backward depends on the gradient the loss kernel
produces, so the two cannot overlap as drawn. A single GPU doing sequential
work has no student idle time to fill. Two streams in one CUDA context do not
partition the SMs: large matrix multiplications each fill the device, so
overlapping them does not reduce total work and raises peak memory (this
description of CUDA behaviour was not re-checked against a source). The
premise that one of two transformers is compute-bound while the other is
memory-bound is unsupported.

### E7 — The cited Liger-Kernel already ships fused distillation losses

**Proposal.** The abstract presents it as CPKD's contribution to "eliminate
the full teacher logit tensor from HBM", and §6 gives PyTorch's teacher
logits as "Full [B,S,V] tensor in HBM" against "Never materialized (fused
KL-CE)". Liger-Kernel is reference 5.

**Correction.** Never materialising the logits is prior art in the work the
proposal cites. Liger-Kernel's README lists Fused Linear JSD and Fused Linear
KL Divergence distillation kernels, "reducing memory usage by up to 80%".
When the KL variant was added was not established, so it is not known whether
it predates the proposal.

**Source.** https://github.com/linkedin/Liger-Kernel

### E8 — The design uses none of the loss variants in the work it cites

**Proposal.** §2.2 gives one loss, "L = α × CE(student_logits, labels) +
(1-α) × T² × KL(softmax(teacher_logits/T), softmax(student_logits/T))". The
references include MiniLLM and "On-Policy Distillation of Language Models".

**Correction.** The design is the 2015 forward-KL recipe of the first
reference and nothing else. MiniLLM argues for reverse KL, and the on-policy
paper for on-policy data and generalised divergences; neither appears in the
design (the content of these two papers is stated from memory and was not
re-checked against them).

**Source.** That the Minitron reference exists was confirmed at
https://arxiv.org/abs/2407.14679; the other four citations were not
re-checked.

### E9 — Smaller slips

- §1.1, "batch × 2048 × 32000 × 4 = 256 MB per sample". The product is
  262,144,000 bytes, which is 262 MB or 250 MiB, and 256 in neither unit. The
  §2.2 figure of 2 GB for two tensors at batch 4 is right
  (2 × 4 × 2048 × 32000 × 4 = 2.10 GB).
- §5, "k=800 (spectral rank), 32× reduction". 32000 / 800 = 40×. The 32×
  belongs to the k = 1000 of §2.4 (32000 / 1000).
- Three incompatible layer mappings appear: "mapping to every other teacher
  layer" (§2.1), "pairs teacher layers 0+1 with student layer 0" (§2.3), and
  "selects the k most important teacher layers" (§2.5).
- §4, "Σ_l feature_match[l] × d_teacher × seq × 2 ≤ feature_memory_budget"
  omits the batch factor that the attention-transfer constraint beside it
  includes.

**Source.** Recomputed from the proposal's own figures.

---

## Part 2 — Superseded by later work

- **Liger-Kernel fused-linear distillation losses**
  (https://github.com/linkedin/Liger-Kernel). The baseline to compare against
  is a chunked fused-linear distillation loss, not materialised logits.
- **Sparse Logit Sampling, ACL 2025**
  (https://aclanthology.org/2025.acl-long.885). Top-k logit compression
  should be dropped in favour of random-sampling distillation.
- **Minitron-style prune-then-distill** (https://arxiv.org/abs/2407.14679).
  It uses logit-only forward KL with no CE term. This was confirmed for the
  Minitron-SSM follow-up; the original paper's loss section was not read.
  MiniLLM and DistiLLM use reverse or skew KL (not re-checked against the
  sources). The proposal's defaults of alpha = 0.5 and T = 2.0 are a
  classification-era recipe.
- **This repository's own fused linear-CE loss.** It abandoned the scalar
  per-row kernel design for large vocabularies. A comment in
  `crates/nsl-codegen/src/wengert_lower.rs` records that design as 335×
  slower in the forward and 504× in the backward than the GEMM-chunked path
  at rows = 1024, V = 49152, H = 512. The KL-CE kernel is still the per-row
  design, with two dot products.

---

## Part 3 — What the code implements

- **`distill(teacher=, student=, epochs=)` block — implemented.**
  `crates/nsl-codegen/src/stmt_distill.rs`; `distill_as_train_block` in
  `crates/nsl-codegen/src/cpkd.rs` lowers it to a synthetic train block. It
  requires `--source-ad`; there is no tape fallback, and a step body that
  source AD cannot extract is a hard error.
- **Frozen teacher — implemented structurally.** Teacher fields enter the
  differentiated graph as inputs, not parameters, so no teacher gradient
  exists. Teacher weights were checked unchanged after SGD at lr = 4.
- **Fused KL-CE kernel — implemented, narrower than proposed.**
  `build_forward` and `build_backward` in
  `crates/nsl-codegen/src/cpkd_fused_loss.rs`; `nsl_fused_kl_ce_forward` and
  `nsl_fused_kl_ce_backward` in `crates/nsl-runtime/src/fused_kl_ce.rs`.
  Turned on by `@fused_kl_ce(enabled=true, vocab_size=, hidden_size=,
  teacher_hidden=, batch_size=, seq_len=)` on the distill block plus a
  `fused_kl_ce(...)` call in the step. It is f32 and GPU-only, needs hidden
  sizes that are multiples of 32 and literal alpha and T, and is capped at
  `vocab_size <= 8192` (`KL_CE_MAX_VOCAB_V1`). The code's mathematics is
  right where the proposal's pseudo-code is not (E1): loss and all three
  student gradients agree with a PyTorch reference
  (`F.cross_entropy` plus `F.kl_div` scaled by T², mean over valid rows),
  the loss to about 1e-7 and the gradients to about 2e-8. The proposal's
  scope is unmet: its vocabulary of 32000, and this repository's
  49152-vocabulary models, are above the cap. A CPU composite of the same
  name in `stdlib/nsl/nn/losses.nsl` also matches PyTorch.
- **`loss:` section — report-only.** It is parsed, range-checked and echoed
  in the build report (`render_report` in `cpkd.rs`). None of `alpha`,
  `temperature`, `feature_weight` or `feature_layers` changes the loss, which
  is whatever the step body computes. Neither side is right: a declared loss
  should drive the loss or be reported as unused.
- **Two-stream teacher scheduling — absent.** The build report prints
  "sequential". The code is right (E6).
- **Spectral k-selector — dead code.** `effective_vocab_rank` and
  `teacher_lm_head_rank` in `crates/nsl-codegen/src/cpkd_spectral.rs` have no
  callers outside that file's own tests. The idea is unsound (E4), so neither
  side is right.
- **Feature matching and attention transfer — absent.**
  `attn_transfer = true` is a semantic error, which is the right behaviour
  for an unimplemented feature.
- **CEP student design — report-only.**
  `nsl check --cpkd-design-student <budget> [--cpkd-target <gpu>]`
  (`run_cpkd_design` in `crates/nsl-cli/src/commands/cpkd_design.rs`,
  `design_student` and `select_layer_mapping` in
  `crates/nsl-codegen/src/cpkd_student.rs`) prints an advisory architecture
  and layer map. No student source or weights are generated or initialised.
- **WGGO variables — present, never enabled.**
  `crates/nsl-codegen/src/wggo_cpkd.rs` is a sidecar whose configuration
  (`constraints.cpkd`) is set only in test modules. Its costs are
  placeholders (25 µs, 60 µs, 0.30), and with no quality term the solver
  always chooses all-off.
- **Absent altogether:** `distill_step`, `forward_with_intermediates`,
  `cosine_distance`, the `--distill` and `--teacher-weights` flags,
  `cpkd_schedule.rs`, `cpkd_feature.rs`, and any teacher evaluation mode.

Teacher and student may have different hidden sizes (checked at 32/64 and
64/32). Different vocabularies are refused: the kernel takes one
`vocab_size`. No file under `models/` or `examples/` contains a `distill`
block, and the fused path is exercised end to end only by GPU tests that are
ignored by default (`fused_kl_ce_step` in
`crates/nsl-cli/tests/fused_loss_gradient_cert_gpu.rs`).

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Rebuild the fused loss on the GEMM-chunked linear-CE path. That removes
   the 8192 cap for 2 · chunk · V · 4 bytes of scratch, independent of batch.
2. Until then, let `fused_kl_ce` expand to its composite inside a distill
   block when the kernel cannot fire, so it runs on every model and on CPU.
3. Drop spectral top-k and two-stream overlap as goals.
4. Restate the memory case like-for-like, against Liger's fused loss (E3, E7).
5. Add reverse-KL and JSD options, a logit-only preset (alpha = 0, T = 1), a
   teacher evaluation mode, and a uniform, last-to-last default layer map.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code. This part lists what it found wrong in the
implementation, as distinct from the proposal's own errors in Part 1. Fixed
items are described in `CHANGELOG.md` under "Unreleased"; the open list is as
of 2026-11-13. Each item ends with how it was established: reproduced by
running, numeric check, or by reading the code only. File names given without
a directory are under `crates/nsl-codegen/src/`.

### Fixed

None of the defects below has been fixed.

### Open

- **The teacher has no evaluation mode.** With `dropout(..., 0.5, true)` in
  the teacher's forward and SGD at lr = 0, the loss changes every step (84.2,
  85.6, 91.2, 93.7, 90.1); with `false` it is constant. Dropout is emitted
  unless its training argument is a literal false or p = 0, and a teacher
  using the stdlib `Dropout` module fails extraction. `extract_expr_inner`
  (`source_ad.rs`). Reproduced by running.
- **A bad target gives a NaN or infinite fused loss and the step still
  updates.** The kernel skips only target −100; the host-side mask treats
  every negative as ignored. Target −1 prints `NaN` and target = V prints
  `inf`; the composite gives the −100 loss for −1 and aborts for V.
  `build_forward`, `build_backward` (`cpkd_fused_loss.rs`);
  `lower_fused_kl_ce_forward` (`wengert_lower.rs`). Reproduced by running.
- **A failed kernel launch would train on a zero loss.** The return codes of
  `nsl_fused_kl_ce_forward` and `nsl_fused_kl_ce_backward` are discarded; on
  failure the zero-filled buffers give loss 0 and zero gradients and
  training continues. It should abort. `lower_fused_kl_ce_forward`,
  `lower_fused_kl_ce_backward_extract` (`wengert_lower.rs`). By reading the
  code only; not triggered.
- **The `loss:` section is accepted, reported and not honoured.** With
  `alpha`, `temperature`, `feature_weight` and `feature_layers` set and a
  step body of `mse_loss(s_logits, t_logits)`, the report prints the values
  and none affects anything. Call-site `fused_kl_ce(..., 0.6, 0.0)` and
  `(..., 1.7, 2.0)` pass `nsl check` when `loss:` omits those keys.
  `render_report` (`cpkd.rs`). Reproduced by running.
- **`--target cpu` does not keep the fused path off the GPU.** With
  host-placed tensors and `--target cpu`, the log shows a device allocation
  and the kernel's result. `lower_fused_kl_ce_forward` (`wengert_lower.rs`)
  allocates device tensors unconditionally. Reproduced by running.
- **`fused_kl_ce` in a distill block has no fallback.** When the kernel
  cannot fire (any vocabulary above 8192) extraction fails and the build
  stops, though every op of the composite is supported under source AD.
  `stdlib/nsl/nn/losses.nsl` says the composites "run anywhere".
  `extract_expr_inner` (`source_ad.rs`). Reproduced by running.
- **`--training-reference` does not run the composite baseline in a distill
  block.** A comment in `compile_train_block_inner`
  (`stmt_train/driver.rs`) says it does; the build ends in "distill step
  body could not be extracted for source AD". Reproduced by running.
- **The build report points at an analysis that never runs.** It says
  spectral compression is advisory, "see `nsl check --cpkd-design-student` /
  cpkd_spectral"; that command never calls `cpkd_spectral.rs`, and nothing
  else does. Its `MAX_SKETCH_CAP` of 192 could not represent the proposal's
  own 500 to 2000. `render_report` (`cpkd.rs`). Reproduced by running; the
  absence of callers by reading the code only.
- **A wrong `batch_size` or `seq_len` on `@fused_kl_ce` is blamed on another
  decorator.** The abort message says "the @fused_lm_ce decorator pins
  batch_size * seq_len", which the program does not contain.
  `nsl_fused_lce_targets_i64_alloc` in
  `crates/nsl-runtime/src/fused_linear_ce.rs`. Reproduced by running.
- **The advisory layer map can cluster.** It takes the top-k teacher layers
  by importance and pairs them in order; nothing prevents, say, teacher
  layers 20 to 31 for a 12-layer student, or guarantees last-to-last. Layers
  the importance pass does not match score 0.0 silently. The CLI tests never
  pass `--weights`. `select_layer_mapping` (`cpkd_student.rs`). By reading
  the code only.
- **The on-device numerical test cannot fail for dx_s, dW_s or the KL
  term.** Inputs are U(−0.3, 0.3); the reference gradient maxima, 2.3e-3 and
  4.2e-3, are below the 5e-3 absolute tolerance, so all-zero gradients pass.
  Swapping the KL direction moves the loss 5.4e-5 against a 1e-3 tolerance.
  `crates/nsl-codegen/tests/cpkd_fused_kl_ce_numerical.rs`. Numeric check.
- **A module-level tensor alias aborts at exit.** This is a general
  ownership defect found while testing, not part of CPKD: a module-level
  `let t = t0` of a tensor that is then used inside a train or distill step
  ends the run with `free(): double free detected in tcache 2`; the same
  program without the alias exits cleanly. Reproduced by running.
