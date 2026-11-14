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
  source AD cannot extract is a hard error. A loss written as an ordinary
  function is extracted: source AD inlines it.
- **Frozen teacher — implemented structurally.** Teacher fields enter the
  differentiated graph as inputs, not parameters, so no teacher gradient
  exists. Teacher weights were checked unchanged after SGD at lr = 4. The
  teacher is also evaluated rather than trained: inside a method called on
  it, `dropout` is the identity and `is_training()` is false.
- **Fused KL-CE kernel — implemented, narrower than proposed.**
  `build_forward` and `build_backward` in
  `crates/nsl-codegen/src/cpkd_fused_loss.rs`; `nsl_fused_kl_ce_forward` and
  `nsl_fused_kl_ce_backward` in `crates/nsl-runtime/src/fused_kl_ce.rs`.
  Turned on by `@fused_kl_ce(enabled=true, vocab_size=, hidden_size=,
  teacher_hidden=, batch_size=, seq_len=)` on the distill block plus a
  `fused_kl_ce(...)` call in the step. It is f32 and GPU-only, needs hidden
  sizes that are multiples of 32 and literal alpha and T, and is capped at
  `vocab_size <= 8192` (`KL_CE_MAX_VOCAB_V1`). Where it does not apply (no
  enabled decorator, `--target cpu`, `--training-reference`) the same call
  is the composite below, so a distill block can use `fused_kl_ce` on any
  model and on the CPU. The code's mathematics is
  right where the proposal's pseudo-code is not (E1): loss and all three
  student gradients agree with a PyTorch reference
  (`F.cross_entropy` plus `F.kl_div` scaled by T², mean over valid rows),
  the loss to about 1e-7 and the gradients to about 2e-8. The proposal's
  scope is unmet: its vocabulary of 32000, and this repository's
  49152-vocabulary models, are above the cap. The composite of the same
  name in `stdlib/nsl/nn/losses.nsl` also matches PyTorch, under tape AD
  and, inlined into a distill step, under source AD.
- **`loss:` section — a checked statement, not a definition.** The loss is
  whatever the step body computes. The section may state `alpha` and
  `temperature`; they must then equal the literals of the block's
  `fused_kl_ce(...)` call, and a section no such call consumes is a compile
  error. `feature_weight` and `feature_layers` are refused. The build
  report (`render_report` in `cpkd.rs`) prints the loss the step computes.
- **Two-stream teacher scheduling — absent.** The build report prints
  "sequential". The code is right (E6).
- **Spectral k-selector — absent.** `cpkd_spectral.rs` was removed: it had
  no callers, and the idea is unsound (E4).
- **Feature matching and attention transfer — absent.**
  `attn_transfer = true`, `feature_weight` and `feature_layers` are semantic
  errors, which is the right behaviour for an unimplemented feature.
- **CEP student design — report-only.**
  `nsl check --cpkd-design-student <budget> [--cpkd-target <gpu>]`
  (`run_cpkd_design` in `crates/nsl-cli/src/commands/cpkd_design.rs`,
  `design_student` and `select_layer_mapping` in
  `crates/nsl-codegen/src/cpkd_student.rs`) prints an advisory architecture
  and layer map (one teacher layer per student layer, spread over the
  teacher's depth, last to last). No student source or weights are
  generated or initialised.
- **WGGO variables — present, never enabled.**
  `crates/nsl-codegen/src/wggo_cpkd.rs` is a sidecar whose configuration
  (`constraints.cpkd`) is set only in test modules. Its costs are
  placeholders (25 µs, 60 µs, 0.30), and with no quality term the solver
  always chooses all-off.
- **Absent altogether:** `distill_step`, `forward_with_intermediates`,
  `cosine_distance`, the `--distill` and `--teacher-weights` flags,
  `cpkd_schedule.rs` and `cpkd_feature.rs`.

Teacher and student may have different hidden sizes (checked at 32/64 and
64/32). Different vocabularies are refused: the kernel takes one
`vocab_size`. No file under `models/` or `examples/` contains a `distill`
block. The composite path is exercised end to end against PyTorch by
default (`crates/nsl-cli/tests/cpkd_distill_loss_gate.rs`); the fused
kernel only by GPU tests that are ignored by default (`fused_kl_ce_step` in
`crates/nsl-cli/tests/fused_loss_gradient_cert_gpu.rs`, and
`crates/nsl-cli/tests/fused_kl_ce_gpu_gate.rs`, which holds it to the same
PyTorch numbers).

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Rebuild the fused loss on the GEMM-chunked linear-CE path. That removes
   the 8192 cap for 2 · chunk · V · 4 bytes of scratch, independent of batch.
2. Until then `fused_kl_ce` expands to its composite inside a distill block
   wherever the kernel does not apply (done; Part 5).
3. Drop spectral top-k and two-stream overlap as goals.
4. Restate the memory case like-for-like, against Liger's fused loss (E3, E7).
5. Add reverse-KL and JSD options and a logit-only preset (alpha = 0,
   T = 1). (The teacher evaluation mode and the uniform, last-to-last layer
   map are done; Part 5.)

---

## Part 5 — Implementation defects found in the same review

The review also ran the code. This part lists what it found wrong in the
implementation, as distinct from the proposal's own errors in Part 1. Fixed
items are described in `CHANGELOG.md` under "Unreleased"; the open list is as
of 2026-11-13. File names given without a directory are under
`crates/nsl-codegen/src/`.

### Fixed

- A `let t = t0` of a tensor copied the handle without taking a reference,
  so two owning names shared one: at module level, with the tensor used in
  a train or distill step, the run ended with `free(): double free detected
  in tcache 2`. The binding now retains, as the assignment `t = t0` did
  (`compile_stmt_dispatch` in `stmt.rs`); test
  `crates/nsl-cli/tests/module_tensor_alias_gate.rs`.
- The teacher had no evaluation mode: `dropout(..., 0.5, true)` in its
  forward fired on every distillation step, so the target the student was
  fitted to changed from step to step, and a teacher built on the stdlib
  `Dropout` module did not compile. Inside a method called on the teacher
  (and anything it calls), `dropout` is now the identity whatever its
  training argument and `is_training()` is false (`frozen_inline_depth` in
  `source_ad.rs`). With `dropout(..., 0.5, true)` in the teacher the losses
  match a PyTorch run with the teacher in `eval()`; test
  `crates/nsl-cli/tests/cpkd_distill_loss_gate.rs`.
- The `loss:` section was parsed, range-checked, echoed in the build report
  and otherwise ignored, and call-site `fused_kl_ce(..., 0.6, 0.0)` or
  `(..., 1.7, 2.0)` passed `nsl check` when the section omitted those keys.
  Now `alpha` and `temperature` in the section must equal the literals of
  the block's `fused_kl_ce(...)` call, and a section whose values no such
  call consumes is a compile error; `feature_weight` and `feature_layers`
  are refused (there is no feature matching); the call-site literals are
  range-checked by `nsl check` whether or not the section restates them;
  and the report prints the loss the step computes and what computes it.
  Fixtures whose step body is not `fused_kl_ce` lost their `loss:` section.
  Test `cpkd_distill_loss_gate.rs`.
- `--target cpu` did not keep the fused path off the GPU: the lowering
  allocated device tensors and launched the kernel. On a host-only target
  the decorator is now set aside with a warning and `fused_kl_ce` is the
  standard library's composite (`compile_train_block_inner` in
  `stmt_train/driver.rs`). The test runs on a build without CUDA, where a
  device allocation is fatal.
- `fused_kl_ce` in a distill block had no fallback: where the kernel could
  not fire, extraction failed and the build stopped. Without an enabled
  `@fused_kl_ce` decorator the call now inlines the standard library's
  composite, differentiated at compile time; its loss and the student's
  updates over four SGD steps match PyTorch. An enabled decorator that
  cannot build the kernel (a missing shape hint, a non-literal alpha) is a
  compile error naming what is missing.
- `--training-reference` did not run the composite baseline in a distill
  block; the build ended in "distill step body could not be extracted for
  source AD". It now runs the composite, as the comment in
  `compile_train_block_inner` said, with the same PyTorch-checked numbers.
- The build report said spectral logit compression was advisory, "see `nsl
  check --cpkd-design-student` / cpkd_spectral"; that command never called
  `cpkd_spectral.rs` and nothing else did. The report line is gone, and so
  is `cpkd_spectral.rs` with its tests: it had no callers, its
  `MAX_SKETCH_CAP` of 192 could not represent the proposal's own 500 to
  2000, and the idea it served is unsound (E4).
- The advisory layer map took the top-k teacher layers by importance and
  paired them in order, so nothing stopped teacher layers 20 to 31 serving
  a 12-layer student; with no `--weights` every score tied and the student
  was mapped onto the teacher's first k layers; and a layer the importance
  pass did not match was shown with an importance of 0.0.
  `select_layer_mapping` (`cpkd_student.rs`) now cuts the teacher's depth
  into one segment per student layer, takes the most important layer of
  each segment (the deepest on a tie or with no scores), and always pairs
  the last student layer with the last teacher layer. A layer without a
  score is reported as having none. Unit tests in `cpkd_student.rs`.
- A target the fused kernels could not use gave a NaN or infinite loss and
  the step still updated: the kernels skip only target -100 while the
  lowering's mask ignores every negative target, so -1 printed `NaN`, and a
  target equal to the vocabulary size printed `inf`. The host-side copy of
  the targets that both fused losses make before a launch now rewrites
  every negative target to -100 and aborts, naming the row, on a target at
  or above `vocab_size` (`normalize_fused_targets` in
  `crates/nsl-runtime/src/fused_linear_ce.rs`), for `@fused_lm_ce` as well
  as `@fused_kl_ce`. With -1 in place of -100 the fused loss and updates
  match PyTorch; tests `crates/nsl-cli/tests/fused_kl_ce_gpu_gate.rs` (GPU)
  and the unit tests beside `normalize_fused_targets`.
- A fused KL-CE kernel launch that failed returned a code the compiled step
  discarded, leaving a zero-filled loss buffer: loss 0, zero gradients, and
  training continued. `nsl_fused_kl_ce_forward` and
  `nsl_fused_kl_ce_backward` now report the CUDA error and exit with the
  runtime's CUDA-driver code (13); a runtime built without CUDA exits the
  same way instead of returning -1. Triggered on the GPU by asking a valid
  module for an entry it does not have
  (`a_failed_launch_exits_instead_of_returning` in
  `crates/nsl-codegen/tests/cpkd_fused_kl_ce_numerical.rs`).
- A wrong `batch_size` or `seq_len` on `@fused_kl_ce` aborted with a
  message about "the @fused_lm_ce decorator", which the program does not
  contain. `nsl_fused_lce_targets_i64_alloc` is now told which decorator
  it is checking and names it. Test `fused_kl_ce_gpu_gate.rs` (GPU).
- The on-device numerical test could not fail for dx_s, dW_s or the KL
  term: its inputs were U(-0.3, 0.3), the reference gradient maxima were
  below its absolute 5e-3 tolerance, and swapping the KL direction moved
  the loss by less than its forward tolerance.
  `crates/nsl-codegen/tests/cpkd_fused_kl_ce_numerical.rs` now draws from
  U(-1, 1), compares against an f64 reference written out in the test,
  holds the loss to 2e-5 and each gradient tensor to 1e-4 of its own
  largest entry. Two further tests show it can fail: on the device, a
  backward that never ran, another temperature and a missing KL term are
  each reported; and reversing or dropping the KL term moves the reference
  by more than a hundred times those tolerances.

### Open

Nothing from this review is open. Two limits of the fixes above are worth
knowing. The fused kernel is still capped at `vocab_size <= 8192` (Part 3);
above it a distill block runs the composite, which materialises both logit
tensors. And a student, unlike the teacher, is extracted in training mode
only where its dropout is written with a literal training argument: a
student built on the stdlib `Dropout` module, whose forward asks
`is_training()`, still fails extraction inside a distill block.
