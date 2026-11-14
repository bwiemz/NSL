# CCR — Errata

**Companion to** `CCR.pdf`. **Date:** 2026-11-13.

The proposal (April 2026) was audited in November 2026 against its own
arithmetic, the literature it cites, and the code that implements it. This
file lists the statements in the PDF that should not be relied on and what
is true instead. Arithmetic was recomputed. Each literature correction names
the source it was checked against; where a point rests on a search summary,
an abstract, or was not re-checked against the source, the entry says so.
Sections of the PDF are named by topic rather than by number.
Abbreviations: CCR is compiler-chosen recomputation; MILP is
mixed-integer linear program; SDPA is scaled dot-product attention.

---

## Part 1 — Errors in the proposal

### E1 — Peak memory is a maximum over time, not a sum over tensors

**Proposal.** Decision framework and MILP formulation: the memory constraint
is "∑ memory(dt) ≤ B", RECOMPUTE has "near-zero memory cost (only input
deps)", and its time cost is "recompute cost × downstream use count". The
repetitive-block section counts "3 ⋅ ∣F∣ options per tensor".

**Correction.** The formulation is wrong in five places.

- Peak memory is a maximum over the step, not a sum over tensors. Recomputed
  tensors are live one block at a time, so they do not add up across blocks.
  The proposal's own memory-planner section says "tensors with disjoint live
  ranges share the same HBM region", which the summed constraint ignores.
- RECOMPUTE is not near-zero in memory for the same reason: during the
  backward of a block, that block's interior tensors are all materialised.
  The transient cost is one block's interiors.
- A recomputed tensor is materialised once and then read by all of its
  consumers. Its cost is not multiplied by the downstream use count.
- The per-tensor cost `C(t, d)` is not separable when one recomputed tensor
  is an input to another, because the cost of recomputing the second depends
  on the decision for the first. Checkmate, the MILP precedent for this
  problem, handles this with per-timestep schedule variables; the proposal's
  formulation has none (Checkmate's formulation was not re-checked against
  the source).
- The option count is 2 + |F| per tensor (save at full precision, recompute,
  or save in one of |F| formats), not 3·|F|. With the three formats named in
  the decision framework that is 5, not 9. The MILP formulation section
  lists 7 decisions, which is 2 + 5 because it adds two FP8 formats.

**Source.** Pure logic, and recomputed from the proposal's own figures.

### E2 — Activation memory grows linearly with sequence length here, not quadratically

**Proposal.** Motivation: "Activations grow quadratically with sequence
length (through attention intermediates) and linearly with batch size. At
seq=8192, batch=4, activations balloon to approximately 180 GB".

**Correction.** The proposal's own figures show linear growth: 42 GB in the
breakdown table rising to about 180 GB is a factor of 180 / 42 = 4.3 for a
fourfold longer sequence (taking the table to describe the 7B model at
seq=2048, batch=4, as in the evaluation configurations).

Megatron's Equation 1 gives per-layer activation memory as
`sbh(34 + 5as/h)`, with sequence length `s`, batch `b`, hidden size `h` and
`a` attention heads. The `5as/h` term is the stored attention maps. Reading
the result as bytes at 16-bit precision (this unit reading was not re-checked
against the source), a 7B-like model with 32 layers, h = 4096, a = 32 at
b = 4 gives, in units of 10^9 bytes:

| | s = 2048 | s = 8192 |
|---|---|---|
| without attention maps (`34·sbh` per layer) | 36.5 GB | 146 GB |
| with attention maps | 122 GB | 1520 GB |

The proposal's 42 GB and 180 GB match the first row, the linear case with
no stored attention maps. That is also NSL's case: it uses fused (flash)
SDPA, which stores no s×s maps. The quadratic term the sentence appeals to
is absent from both the proposal's numbers and NSL's design.

**Source.** Recomputed from the proposal's own figures; Megatron Equation 1
from https://ar5iv.labs.arxiv.org/html/2205.05198.

### E3 — Activations do not dominate in the proposal's own table

**Proposal.** Motivation: "activation memory dominates GPU memory at typical
batch sizes", directly above the memory breakdown table.

**Correction.** In that table optimizer state is 56 GB of 126 GB (44%) and
activations are 42 GB (33%). Activations are the second-largest component.
The table's own arithmetic is correct (14 + 14 + 56 + 42 = 126 GB; 11%, 11%,
44%, 33%).

**Source.** Recomputed from the proposal's own figures.

### E4 — Saving 11% on parameters would mean eliminating them

**Proposal.** Motivation, in the list of existing techniques: the
weight-compression techniques "reduce parameter memory (~11% saved, varies
with aggressiveness)".

**Correction.** Parameters are 11% of the total in the table above, so a
saving of 11% of the total is all of the parameter memory. A technique that
compresses parameters saves some fraction of that 11%, not the whole of it.

### E5 — The "circular recompute dependency" failure mode cannot occur

**Proposal.** Failure modes: "MILP decided RECOMPUTE for a and b, where a is
a recompute input for b and vice versa."

**Correction.** Recompute dependencies follow the forward computation graph,
which is acyclic. If `a` is an input to `b`, then `b` is not an input to
`a`. No DAG constraint in the solver is needed, and no such error can be
raised. The implementation's own comments say the same.

### E6 — "Solved once at compile time" contradicts where its inputs come from

**Proposal.** Abstract: "The MILP is solved once at compile time". Decision
framework: "CCR's profile pass measures them per format on the target GPU
during the first training step". Outlier handling: "The outlier analysis
runs at compile time using a calibration trace from the first 100 training
steps".

**Correction.** These cannot all hold. A solve whose cost inputs are measured
during the first training step, and whose outlier inputs come from the first
100 steps, is not complete at compile time.

### E7 — Adacc's 1.01–1.37× is over full recomputation, not Megatron selective

**Proposal.** Prior work: "Achieves 1.01-1.37× speedup over SOTA Megatron".
Evaluation baselines: "closest academic peer; 1.01-1.37× speedup over
Megatron selective."

**Correction.** Adacc §4.2 reads "improves the throughput by 1.01×-1.37× and
1.09×-1.28× compared to Full Recomputation and Quantization". The word
"selective" does not appear in the paper. The hardware was 8× V100 32 GB and
the models went up to GPT-4.7B. Full recomputation is the expensive baseline
(Megatron measures 36–39% overhead for it, against 7% or less for selective;
see E9), so the evaluation goal "expected to match or exceed Adacc" is a
much lower bar than the proposal implies.

**Source.** https://arxiv.org/html/2508.00806v2.

### E8 — Adacc does not show the one-block solve is within 0.5% of a full-graph solver

**Proposal.** Repetitive-block reduction: "Adacc's paper established this is
a safe approximation; loss is within 0.5% of a full-graph solver across
tested GPT configurations."

**Correction.** The Adacc paper contains no comparison with a full-graph
solver. Its "0.46%-0.5%" is the validation-loss difference against the
uncompressed baseline (§4.3), and "0.5 second" is its solve time. On block
reuse it says only that a "locally optimal policy for one block" is applied
to the others. The safety of solving one representative block is therefore
not established by the citation.

**Source.** https://arxiv.org/html/2508.00806v2.

### E9 — The Megatron figures combine two techniques and misstate the overheads

**Proposal.** Prior work, on selective recomputation: "Achieves ~5×
activation memory reduction at ~5% compute overhead." Evaluation baselines:
full recomputation is "coarse; ~30% compute overhead at ~5× memory
reduction" and selective is "~5% compute overhead at ~5× memory reduction."

**Correction.**

- The 5× is sequence parallelism and selective recomputation combined, under
  tensor parallelism: "Individually, both techniques cut the memory
  requirement nearly in half, and combined provide a 5x reduction".
- Selective alone had 7% overhead on the paper's 22B test. The 4% and 2%
  figures were obtained only together with sequence parallelism.
- Full recomputation was measured at 36–39% overhead, not about 30%, and it
  keeps only layer inputs, a reduction far larger than 5×.
- Because it depends on tensor parallelism, the 5× is not available on a
  single GPU.

**Source.** https://ar5iv.labs.arxiv.org/html/2205.05198.

### E10 — PyTorch's memory budget solves a knapsack; it is not a heuristic

**Proposal.** Abstract: "one-dimensional automatic solvers (PyTorch's
activation_memory_budget API)". Motivation: "by runtime heuristics
(PyTorch's memory budget)". Prior work: "the Inductor backend runs an
automatic checkpointing solver."

**Correction.** The configuration source describes the option this way: "It
solves a 0-1 knapsack to find the minimum recompute necessary to stay below
the activation memory budget". It offers two estimators of recompute run
time, `flops` and `profile`, and three solvers, `dp`, `greedy` and `ilp`.
The budget is a fraction from 0 to 1 between full checkpointing and the
default min-cut strategy. This is the same design NSL ended up implementing
(see Part 3), and PyTorch's has a real cost estimator.

**Source.**
https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_functorch/config.py.

### E11 — "Outliers emerge early and persist" is unsupported, and Adacc assumes the opposite

**Proposal.** Outlier handling: "the decisions are stable thereafter
(outliers emerge early in pre-training and persist)."

**Correction.** The proposal gives no evidence for this. It is at odds with
Adacc, whose third contribution is a policy re-evaluated on an
exponential-backoff schedule because activation distributions change during
training.

**Source.** https://arxiv.org/html/2508.00806v2.

### E12 — Stochastic operations are never addressed

**Proposal.** Correctness tests, item 1: "MILP decisions with all SAVE_FP16
must produce bitwise-identical results to baseline."

**Correction.** The proposal does not mention dropout or any other
random-number-drawing operation. Recomputing such an operation in the
backward pass gives a wrong gradient unless the same random values are
reproduced. The test quoted above passes trivially, because with every
tensor saved nothing is recomputed, and no listed test exercises RECOMPUTE
through a stochastic operation. Adacc handles the case by saving dropout
masks, compressed to one bit per element.

**Source.** https://arxiv.org/html/2508.00806v2 for the Adacc statement.

### E13 — Smaller slips

- References: TorchTitan's first author is Wanchao Liang, not "Liang, C."
  (https://arxiv.org/abs/2410.06511).
- Outlier handling: the 39% accuracy loss from naive INT4 quantisation is
  cited as "(Adacc §4)"; the paper states it in §3.2
  (https://arxiv.org/html/2508.00806v2).
- Checkmate, the actual MILP precedent, appears only in the reference list.
  The prior-work section does not discuss it and calls Adacc's formulation
  "CCR's direct precedent".
- Two statements could not be verified, which is not the same as being
  false: that "A small number of channels (often <1%) carry values 10-100×
  larger than the median", and the description of the IM3 format as "3-bit
  mantissa, 2-bit exponent equivalent".

---

## Part 2 — Superseded by later work

- PyTorch selective activation checkpointing. `torch.utils.checkpoint` now
  takes a `policy_fn`; the PyTorch blog's "Policy 2" (always save matmul,
  convolution and attention outputs, recompute the rest) is exactly NSL's
  `--checkpoint-selective`
  (https://pytorch.org/blog/activation-checkpointing-techniques/). Together
  with the knapsack in E10, this leaves compression as CCR's only remaining
  differentiator from stock PyTorch.
- PyTorch's knapsack solver has kept moving. A memory-efficient exact
  knapsack, `dp_knapsack_sliding_hirschberg`, shipped in PyTorch 2.10, using
  O(W) rather than O(nW) memory (https://arxiv.org/abs/2608.08740). The
  current configuration file also has an
  `activation_memory_budget_require_full_coverage` option whose comment
  mentions a "regional activation memory budget" (config URL in E10).
- PyTorch's baseline saves less than NSL's. Eager PyTorch and the min-cut
  partitioner save only what the backward formulas read; the blog above
  reports "a 50% memory reduction by recomputing only pointwise ops". NSL's
  baseline keeps every forward intermediate until the end of the backward,
  so part of CCR's measured saving recovers a baseline PyTorch has by
  default.
- Flash attention removed the O(s²) term that motivated Megatron's selective
  recomputation. NVIDIA's Megatron Bridge documentation rates recomputing
  the attention core as a moderate saving at low cost "because Flash
  Attention already recomputes internally"
  (https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/nemo-mbridge-perf-activation-recompute/SKILL.html;
  taken from a search summary, the page itself was not read). NSL uses flash
  attention, so the O(s²) saving is not there to be had (see E2).
- Activation-compression theory (https://arxiv.org/abs/2605.01255, May 2026;
  abstract only was read): compression is safe for linear operators when it
  is unbiased and problematic for nonlinear ones. NSL's
  `--checkpoint-compress` rounds to nearest, which is biased, and applies it
  to matmul outputs that feed the SiLU backward and the SDPA backward, both
  nonlinear consumers.
- Adacc is still at arXiv v2 (August 2025). No venue publication or
  follow-up was found.

---

## Part 3 — What the code implements

CCR is implemented as block-granular recomputation on the source-AD
(source-level automatic differentiation) path. It is switched on by
`--checkpoint-blocks`, which requires `--source-ad`; without `--source-ad`
the build fails rather than ignoring the request. The modifiers are
`--checkpoint-selective`, `--checkpoint-budget-mib`,
`--checkpoint-stride N|auto|dp` and `--checkpoint-compress bf16`. There is
no decorator form: `@checkpoint(policy="selective")` is refused, as is
`@checkpoint` on a model or a model method.
`@checkpoint(policy="full")` belongs to CSHA's attention-prologue recompute,
a different mechanism not covered here. The planner is
`crates/nsl-codegen/src/ccr.rs`, driven from
`crates/nsl-codegen/src/stmt_train/plan_ccr.rs` and
`crates/nsl-codegen/src/stmt_train/driver.rs`.

- MILP over per-tensor decisions, solved with `lp_solve` or `good_lp`:
  absent. No LP or MILP dependency exists in any `Cargo.toml`. Instead,
  `plan_impl` segments the tape by `blocks.N` parameter names; tensors
  interior to a block are recomputed and tensors escaping it are saved. The
  implementation is right: the MILP was mis-specified (E1) and is
  unnecessary.
- Memory budget: partial. At a fixed stride `--checkpoint-budget-mib` runs
  a 0/1 knapsack (`apply_budget`) that flips the recompute tensors that
  cost most to recompute back to save. The budget is an absolute number of
  MiB of saved interior bytes, not the proposal's fraction of default.
  Under `--checkpoint-stride auto|dp` the same number is instead the cap on
  the projected peak, and the knapsack does not run. The budget needs
  statically known tensor sizes and the build fails without them.
- Choosing how much to recompute: implemented, and not in the proposal.
  `--checkpoint-stride N` saves every N-th block boundary; `auto` searches
  uniform strides (`select_stride`); `dp` runs a Pareto dynamic program over
  block boundaries (`select_partition_dp`). `auto` and `dp` need a static
  size for every tensor computed inside the blocks and fail the build
  otherwise. Static sizes come from literal shapes and from a DataLoader
  with literal batch size and sequence length, carried through the forward
  (`ccr_static_sizes` in `stmt_train/plan_ccr.rs`).
- Decisions SAVE_FP, SAVE_CFIP (IF4, IL2, IM3, FP8) and RECOMPUTE: partial.
  Only save and recompute exist per tensor. The flag
  `--checkpoint-compress bf16`, which requires `--checkpoint-selective`,
  casts all of the selective policy's saved matmul outputs to bf16, each at
  the end of its block segment (`insert_compressed_saves`); it is
  all-or-nothing, accepts f32 activations only and, as documented, is not
  bit-exact. CFIP formats, outlier separation and per-tensor format choice
  are absent.
- Selective policy: implemented, and not in the proposal.
  `--checkpoint-selective` (`CcrPolicy::Selective`, `selective_saves`) keeps
  matmul-, convolution- and attention-class outputs and recomputes the cheap
  remainder. This is the PyTorch blog's "Policy 2". It is not Megatron's
  selective recomputation, which recomputes the attention core and saves the
  linear layers.
- Profile pass measuring costs on the target GPU: absent. For the knapsack
  the cost of recomputing a tensor is an operation count from its static
  shapes (`recompute_ops`), the same kind of estimate as PyTorch's `flops`
  estimator; the `dp` mode uses a launch count times the launch overhead
  the GPU database records for the local device (H100 constants, with a
  warning, when the device is not identified). Nothing is measured.
- `@checkpoint(budget=0.5 | "auto", strategy=…)` on the model: absent.
- Compile error when the budget is infeasible: absent. The budget is soft;
  when no stride fits, the planner says so and uses the minimum-peak plan.
  The proposal is right for a flag named "budget".
- CSHA tiers as CCR decision points: the opposite is implemented. Chains
  claimed by CSHA are force-saved and exempt from recomputation (`plan`,
  argument `csha_claimed_ops`).
- FASE Deferred credit to the budget: absent. The proposal adds the freed
  gradient-buffer memory to a planner-derived budget. The code used to add
  the parameter bytes to the absolute budget typed by the user; that was
  removed, and the budget is the number given.
- CPFT pruning, CTQS re-solve, M36 planner integration and outlier analysis:
  absent.
- Stochastic operations (missing from the proposal, E12): dropout masks and
  outputs, and `randn` draws, are force-saved rather than recomputed.
- Adjoint last-use freeing and per-segment early free: implemented, and not
  in the proposal. `insert_adjoint_last_use_frees` frees tensors at their
  last use in the backward; the project's notes call this the decisive
  memory win. These, with the stride and partition search, are the parts
  that deliver.
- Recomputing only what the backward reads: implemented for block
  interiors, and not in the proposal. `backward_reads` classifies each
  freed interior by how the final adjoint uses it; one read only for its
  shape (the operand of a residual add) or not at all is not recomputed.

Recomputation was checked to be bit-identical to the baseline on CPU for
stride 1, 2 and 3, the selective policy, the budget, `dp` and dropout. GPU
behaviour was not checked. The CPU end-to-end parity tests are
`crates/nsl-cli/tests/ccr_checkpoint_parity.rs` and
`crates/nsl-cli/tests/ccr_checkpoint_activation_parity.rs`. These compare
NSL with NSL; `crates/nsl-cli/tests/ccr_reference_gradient_gate.rs` compares
the losses and trained weights of a small model, under each checkpointing
configuration, with values computed by PyTorch.

In the shipped recipes, `models/coder500m/pretrain_prod.nsl` uses
`--checkpoint-blocks` and `models/coder1b/pretrain_prod.nsl` adds
`--checkpoint-selective`. No recipe uses the budget or compression.

The proposal expects "2-5× activation memory reduction at 2-10% compute
overhead". The measurements recorded in
`models/benchmarks/MEMORY_REDUCTION_RESULTS.md` are of whole-device VRAM,
not activations alone: on a 16M model at sequence length 1024, 7103 → 4008
MiB (−44%) at 840 → 972 ms per micro-batch (+16%); −54% and −53% at
sequence lengths 2048 and 4096; on a 503M model at sequence length 512,
15 366 → 11 934 MiB (−22%) with the backward going from 2.64 s to 2.73 s.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Replace the MILP with what was built, and model peak memory as a maximum
   over time, not a sum over tensors (E1).
2. Restate the bar: Adacc's gain is over full recomputation, selective alone
   costs 7%, and PyTorch already ships a knapsack with cost estimators.
3. Specify stochastic operations (save them, or replay the generator state)
   and add a test that recomputes through one (E12).
4. Keep or recompute only tensors whose values the backward reads; this is
   PyTorch's default and should be NSL's baseline before any recomputation.
5. Compression is the one remaining differentiator: use unbiased rounding on
   tensors with linear consumers and cast per segment, or drop it.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code, on CPU only. This part lists what it found
wrong in the implementation, as distinct from the proposal's own errors in
Part 1. Fixed items are described in `CHANGELOG.md` under "Unreleased". The
open list is as of 2026-11-14.

### Fixed

- A `randn` inside a checkpointed block was drawn again when the block was
  recomputed in the backward, so the gradient was taken through activations
  the forward never produced: the first loss was identical with and without
  `--checkpoint-blocks`, and the weights after one step of stochastic
  gradient descent were not. Random draws are now always saved
  (`draws_random_numbers` in `crates/nsl-codegen/src/ccr.rs`); the test is
  `crates/nsl-cli/tests/ccr_random_draw_gate.rs`.
- The wiki's Architecture Overview expanded CCR as "common-kernel
  combination rewriting". It now reads "compiler-chosen recomputation".
- `--checkpoint-compress fp16` cast the saved activations with no range
  check: with pre-activations above 65504 the loss was NaN from the second
  step, while the baseline, the selective policy and `bf16` stayed finite,
  and `fp16` saved no more bytes than `bf16`. `fp16` was withdrawn; the flag
  accepts `bf16` only, and bf16 has the exponent range of f32.
- A compressed save was restored by a cast to f32 whatever dtype the tensor
  had before it was compressed, so an f64 or half-precision activation would
  have come back with a different dtype. The compressing cast
  (`nsl_ccr_compress_bf16`) now accepts f32 only and exits with a message
  naming the flag for anything else; the test is
  `crates/nsl-runtime/tests/ccr_compress_dtype_refusal.rs`. No model that
  reaches the cast with another dtype was found: an f64 model does not take
  the source-AD path at all.
- `--checkpoint-budget-mib` and `--checkpoint-stride auto|dp` did nothing
  unless every tensor shape was literal: the build noted that it had no
  static sizes, kept stride 1 and full recompute, and exited 0. Even on a
  fully annotated model a quarter of the block tensors had no size (the
  result of `rmsnorm` or `silu` has an open shape whatever the `let`
  binding it says) and counted as zero bytes. Sizes are now carried through
  the forward from the literal shapes, the model fields and a DataLoader
  with literal `batch_size` and `seq_len` and `drop_last=true`
  (`ccr_static_sizes` in `crates/nsl-codegen/src/stmt_train/plan_ccr.rs`),
  so an unannotated model with such a loader is sized. Where a block tensor
  still has no size, `auto` and `dp` fail the build, and so does the budget
  when no recomputed tensor has one. The 1B recipe is such a model: 96 of
  its 800 block tensors, the rotary-embedding tables, have no static size,
  so `--checkpoint-stride auto` is refused there, where it used to resolve
  to stride 1 without saying so.
- `@checkpoint` on a model or a model method was accepted with any
  arguments and had no effect. Both are now errors.
- `@checkpoint(policy="selective")` on a top-level function turned on
  selective recomputation for the whole train block, whether or not the
  step called the function, and a step that did call it fell back to tape
  AD. The policy is refused on a decorator, with a pointer to
  `--checkpoint-blocks --checkpoint-selective`; the planner no longer reads
  decorators. The invalid-policy diagnostic, which already listed
  `"selective"` as refused, is thereby true.
- Under `--checkpoint-stride auto|dp` the budget was spent twice: it capped
  the projected peak, then the knapsack flipped up to the same number of
  bytes back to save. The knapsack now runs at a fixed stride only.
- Under FASE Deferred all parameter bytes were added to the budget the user
  typed. The budget is now the number given.
- `--checkpoint-compress` could not lower the peak at the end of the
  forward, because the casts were appended after the whole forward and
  every full-precision save was still live there. Each save is now cast,
  and its full-precision original freed, at the end of its own block
  segment (`insert_compressed_saves`). The memory effect was not
  re-measured.
- The `dp` cost model used H100 constants on every machine. It now uses the
  GPU-database entry of the local device when the driver names one; when it
  does not (no device, a build without CUDA, an unlisted card) it says so
  and names the constants it falls back to.
- The `dp` search counted every result of a block as a launch, including
  saved results that are never recomputed, so where launches dominated
  every partition cost the same and ties fell to search order. It now
  counts the results a partition actually recomputes; with a budget every
  partition fits, it keeps every anchor.
- `--checkpoint-blocks` recomputed the matrix product feeding each residual
  add, which the backward reads only for its shape, and recomputed
  interiors the backward does not read at all. Only an interior whose value
  the backward reads is recomputed now (`backward_reads` in
  `crates/nsl-codegen/src/ccr.rs`); one read for shape only is replaced by
  a one-element stub captured before it is freed. Under
  `--layerwise-accum` the shape-only case is still recomputed.
- The knapsack's costs were placeholders: 25, 4 or 1 per output byte, so a
  score matmul and a d→4d projection weighed the same, and every tensor was
  charged at least one 256 KiB quantum. The value of keeping a tensor is
  now an operation count from its static shapes (`recompute_ops`), a tensor
  the backward would not recompute is never kept, and the budget axis is
  divided finely enough that a small tensor is charged its own size. It
  remains an estimate, not a measurement.
- Under the selective policy the outputs that feed each residual add stayed
  resident although the backward never reads their values. A plain matrix
  product the backward reads for shape only is now freed at the end of its
  segment. This does not apply with `--checkpoint-compress` or
  `--layerwise-accum`.
- The `dp` decision line printed "kept 2 of 2 block anchors" when it kept 2
  of 4. The denominator is now the number of anchors there were to choose
  from, and the sizes on the `auto` and `dp` lines are printed to two
  decimals instead of whole MiB.
- The `dp` and `auto` stride tests could not fail: their fixture had
  symbolic shapes, so the search never ran. The fixture now has literal
  shapes and the tests assert the stride and the anchors worked out by hand
  from them (`crates/nsl-cli/tests/ccr_periodic_stride_gate.rs`).
- When the baseline was not deterministic from run to run, the GPU parity
  cases compared only the first loss, a pure forward that recomputation
  cannot affect. They now compare the whole loss curve within the
  baseline's own noise band, and a nondeterministic CPU baseline is a
  failure (`parity_case` in `crates/nsl-cli/tests/ccr_checkpoint_parity.rs`).

The gradients under every checkpointing configuration are now also checked
against values computed by PyTorch for the same model, in
`crates/nsl-cli/tests/ccr_reference_gradient_gate.rs`.

### Open

- The peak that `auto` and `dp` test against the budget counts the tensors
  computed inside the block segments and nothing else: it omits adjoint
  temporaries, the logits and everything outside the segments. The decision
  lines and the help text now say so, and a tensor of unknown size is an
  error instead of zero bytes, but "fits the budget" is still a verdict
  against that figure and not against the step's peak memory. Where:
  `project_activation_peak` in `crates/nsl-codegen/src/ccr.rs`. Right:
  project the backward's temporaries as well, or rename the budget.
- The selective policy saves the s×s score matrix of a decomposed attention
  and recomputes only its softmax. Where: `selective_saves` in
  `crates/nsl-codegen/src/ccr.rs`. By reading the code only. Which
  products the policy keeps is a design choice; the budget's knapsack does
  now weigh a score product by its own cost.
- The `dp` time estimate is a launch count times a database constant, with
  a bandwidth floor. It still counts a tensor that is read for shape only
  as a launch. Where: `select_partition_dp` in
  `crates/nsl-codegen/src/ccr.rs`.
