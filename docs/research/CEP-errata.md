# CEP — Errata

**Companion to** `CEP.pdf`. **Date:** 2026-11-13.

`CEP.pdf` (April 2026) proposes using the NSL compiler as the evaluation
oracle for structured pruning and hardware-aware architecture search; this
file lists the statements in it that should not be relied on and says what
is true instead. The entries come from a November 2026 review of the
proposal against its own arithmetic, the literature, and the CEP code in
this repository. Every number below was recomputed; external facts carry
the URL they were checked against, and are marked where they were not
re-checked. Section numbers are the PDF's.

---

## Part 1 — Errors in the proposal

### E1 — The greedy algorithm's constraint check cannot steer the search

**Proposal.** §3.4: "if satisfies_constraints(profile, constraints):" accept
the trial, else "# This component is too important or pushes past
constraints" and continue.

**Correction.** The constraints are upper bounds on memory and latency, and
removing a component only lowers both. A trial can therefore violate them
only if the current model already does. The check is either always true
(the baseline is feasible) or a deadlock (the baseline is infeasible and no
single removal crosses the line, so every step is rejected). "Too
important" has no mechanism behind it: the constraints contain no accuracy
term. The rule that works is to keep pruning until the constraints and the
sparsity target are both met, and to report failure if either is missed.

The same algorithm returns one configuration, the end of a single greedy
path, where §2.2 promises "Output: Pareto-optimal pruning configurations".

### E2 — The importance score is weight magnitude, which reference [1] reports as worse than random

**Proposal.** The abstract says "CEP replaces both" runtime importance
estimation and the evaluation step of neural architecture search (NAS)
with compilation. §3.3 calls its importance score novel and says it
"doesn't require any forward or backward passes".

**Correction.** The score is weight magnitude multiplied by heuristics.
The proposal's own reference [1], LLM-Pruner, compares that criterion with
others in its Table 1 (LLaMA-7B, 20% of parameters pruned, no tuning,
WikiText2 perplexity, lower is better):

| Criterion                          | Perplexity |
|------------------------------------|-----------:|
| L2 magnitude                       |     582.41 |
| Random                             |      27.51 |
| Taylor (element-wise, first order) |      19.09 |

At 50% pruning the L2 criterion gives 39,266. The claim that CEP replaces
runtime importance estimation is therefore unsupported. What compilation
analysis can replace is the hardware-cost half of the evaluation, not the
question of which components the model needs.

**Source.** LLM-Pruner Table 1 as rendered at
https://ar5iv.labs.arxiv.org/html/2305.11627.

### E3 — The importance formula is defined twice, omits the output projection, and its position factor is not symmetric

**Proposal.** §2.2: "Per-head importance: ‖Wq_h‖_F × ‖Wk_h‖_F". §3.3:
"weight_magnitude(Wq_h, Wk_h, Wv_h)" times `spectral_energy`,
`roofline_slack` and `position_factor`, the last given as
"1.0 + 0.3 × |2l/L - 1| (U-shaped curve)".

**Correction.**
- The two definitions differ, and neither includes the output projection
  `Wo_h`. A head whose `Wo_h` is zero contributes nothing to the model
  whatever its query, key and value (Q, K, V) norms are.
- §3.3 says "Heads with low spectral energy encode diffuse,
  low-information patterns", so concentration counts in a head's favour.
  §7.4 says "a head with low-rank weights may benefit from LoRA-style
  replacement rather than pruning", so the same concentration marks a head
  for replacement. The two statements conflict.
- The position factor is not U-shaped over layers 0 to L−1. For L = 8 it
  gives 1.3 at layer 0, 1.0 at layer 4 and 1.225 at layer 7. Dividing by
  L−1 instead of L, `1.0 + 0.3 × |2l/(L−1) − 1|`, gives 1.3 at both ends.

**Source.** Numeric check of the formula for L = 8.

### E4 — The flagship pruned configuration is not a legal grouped-query attention configuration

**Proposal.** §3.1: "attn: GroupedQueryAttention(512, 7, 4, 0.1)" with the
comment "7 Q heads, 4 KV heads". §3.2 lists layers at "7/8 heads" and
"6/8 heads" in a model with 4 KV heads. §2.2: "the Q/K/V projection weight
matrices shrink from [D, D] to [D, D×6/8]".

**Correction.** In grouped-query attention (GQA) several Q heads share one
key/value (KV) head, and the proposal's own search-space table (§4.3)
requires that `n_kv_heads` "Must divide n_heads". Neither 7 nor 6 is
divisible by 4, so these configurations are illegal. Under GQA, heads are
pruned by whole KV group: going from 8 to 6 Q heads means going from 4 to
3 KV heads. The shape statement is also wrong for GQA: K and V are
`[D, D·KV/H]`, which is `[512, 256]` for the 8-head, 4-KV model, not
`[D, D]`.

### E5 — The attention FLOP table is wrong for the stated model

**Proposal.** §3.1 table: "Total attention FLOPs 2.14 GFLOP" before and
"1.87 GFLOP (12.5% less)" after removing one head.

**Correction.** The proposal's 537M floating-point operations (FLOPs) for
the `Wq` matmul is 2 × 1024 × 512 × 512 = 536.9M: 1024 tokens through a
square 512×512 projection. 2.14 G is four such projections
(4 × 536.9M = 2.147 G).
The model has 8 Q heads and 4 KV heads, so `Wk` and `Wv` are 512×256 and
the projections cost 2 × 536.9M + 2 × 268.4M = 1.611 G. Removing one Q
head on the proposal's own terms (K and V unchanged) shrinks `Wq` and `Wo`
to 469.8M each, for 1.476 G: a drop of 8.3%, not 12.5%.

The total also leaves out the attention core, the QKᵀ and attention-times-V
products. At 1024 tokens, 8 heads and head dimension 64 these are
2 × (2 × 1024² × 64 × 8) = 2.147 G, as large as the projections the table
does count.

**Source.** Recomputed from the proposal's own figures.

### E6 — The parameter counts do not reconcile

**Proposal.** §3.2: "Pruned: 31.2M params, 3.7 GB peak memory (36%
smaller)". §6.3: "Layer 4: 8→6 heads, FFN 1408→768 (-3.8M params)" and
"Layer 2: 7→6 heads". §6.3: "Search space: 324 candidates" and
"Binary size: 142MB → 103MB". §8.1: a 50% row at 24.4M parameters.

**Correction.** The model (D = 512, 8 layers, 8 Q heads, 4 KV heads, a
feed-forward network (FFN) of width 1408 with three matrices, embedding
tied to the output head) has 48,767,488 parameters. One layer is
786,432 (attention) + 2,162,688 (FFN) + 1,024 (two norms) = 2,950,144;
eight layers are 23,601,152; the tied embedding is 25,165,824 and the
final norm 512.

- The per-layer configuration listed in §3.2 gives 44,507,648 parameters
  (44.5M, counting K and V as unchanged), which is 8.7% smaller than the
  original, not 31.2M and 36%.
- Taking a layer from 8 to 6 heads (K and V unchanged) and FFN 1408 to
  768 removes 131,072 + 983,040 = 1,114,112 parameters (1.11M), not 3.8M.
  A whole layer is only 2.95M.
- "Layer 2: 7→6 heads" starts from 7 heads in a model whose layers all
  have 8.
- The 50% row of §8.1 cannot be reached by pruning heads and FFN width.
  Half the model is 24.38M parameters, and all eight blocks together hold
  only 23.60M; the remaining 25.17M is the embedding.
- The search in §6.2 has five axes of three values each, which is
  3⁵ = 243 candidates, not 324.
- §8.1 gives the unpruned binary as 195 MB. By that table's rule a 27%
  reduction of 195 MB is 142 MB, so the "142MB" printed in §6.3 as the
  starting size is the pruned size.

**Source.** Recomputed from the proposal's own figures and the layer shapes
of `models/coder50m/model.nsl`.

### E7 — The §8.1 "projected results" are linear scaling, not a model

**Proposal.** §8.1 table: parameters 48.8M / 39.0M / 34.2M / 24.4M, peak
memory 5.2 / 4.2 / 3.7 / 2.6 GB, latency 6.9 / 5.5 / 4.8 / 3.5 μs/tok and
binary size 195 / 156 / 137 / 98 MB at 0 / 20 / 30 / 50% sparsity. §6.3:
"Kernel launches per forward: 127 → 98".

**Correction.** Every column is the baseline multiplied by (1 − sparsity):

| Sparsity | Params (M) | Memory (GB) | Latency (μs) | Binary (MB) |
|---------:|-----------:|------------:|-------------:|------------:|
| 20%      |      39.04 |        4.16 |         5.52 |       156.0 |
| 30%      |      34.16 |        3.64 |         4.83 |       136.5 |
| 50%      |      24.40 |        2.60 |         3.45 |        97.5 |

These match the proposal's cells to rounding (the 30% memory cell is
printed as 3.7). Memory and latency do not scale with parameter count. In
the implemented cost estimate for this model the output head alone
accounts for 51.5 of 117.6 GFLOP per forward pass, and head and FFN
pruning do not touch it.

The kernel-launch reduction does not follow either. Removing heads or FFN
width changes kernel sizes, not the number of kernels launched; the
implemented estimate reports 66 launches before and after. §3.1's row
"FA total kernel launches" gives "batch×8 CTAs", which confuses launches
with CTAs (cooperative thread arrays, the blocks within one launch).

**Source.** Recomputed from the proposal's own figures; the launch and
FLOP counts are from running the implemented estimate.

### E8 — The latency figures are incoherent with the constraints and with "exact"

**Proposal.** §6.3: "6.9μs/token (H100)". The constraints in §2 and §6
are "latency_per_token < 5ms", "< 3ms" and "< 2ms". The abstract promises
"exact hardware performance metrics"; §1 describes "roofline-bounded
latency estimates".

**Correction.** The constraints are 290 to 725 times looser than the
stated baseline (2 ms / 6.9 μs = 290; 5 ms / 6.9 μs = 725), so they could
never bind. The proposal's own 127 kernel launches, at the 5 μs launch
overhead recorded in the repository's GPU database, already cost 635 μs
per forward pass. A roofline figure is a bound, so the metrics cannot be
both roofline-derived and exact.

**Source.** Recomputed from the proposal's own figures.

### E9 — The architecture-search objective is degenerate

**Proposal.** §2.3: "Score: param_count × roofline_utilization / latency".

**Correction.** Latency is roughly proportional to FLOPs, and FLOPs are
roughly proportional to parameters, so the score is nearly constant across
candidates. What variation remains slightly favours the smallest block
stack, because embedding parameters add to the numerator at no FLOP cost.
The objective contains no task-quality term at all.

**Source.** Confirmed by running the implemented search with this
objective over 216 candidates: the top result was the minimum of every
searched axis except `d_model`, and the top ten scores spanned 4%.

### E10 — Prior work is misdescribed

**Proposal.** Abstract: "the 'true pruning' that PyTorch's zeroed-weight
approach only approximates". §7.2: "When PyTorch prunes a head, the weight
tensor is zeroed or masked." §7.1: "PyTorch has no equivalent of nsl check
--perf."

**Correction.** Zeroing or masking describes only `torch.nn.utils.prune`.
Torch-Pruning's DepGraph removes parameters physically, and the proposal's
own references [1] (LLM-Pruner) and [2] (Minitron) also produce physically
smaller models. PyTorch does have an analytic FLOP counter that needs no
GPU: `torch.utils.flop_counter.FlopCounterMode`.

**Source.** https://github.com/VainF/Torch-Pruning (README);
https://docs.pytorch.org/docs/main/generated/torch.utils.flop_counter.FlopCounterMode.html.

### E11 — Two of the listed dependencies are not in the repository

**Proposal.** §1: "The WCET analyzer (M53) provides worst-case latency
bounds." §5 lists WCET analysis (M53) and shape algebra (M49) among the
nine subsystems CEP composes, and §9 names `shape_algebra.rs` and
`wcet.rs` as dependencies.

**Correction.** The worst-case execution time (WCET) analyzer was removed
from the tree; `STATUS.md` records this. Neither `shape_algebra.rs` nor
`wcet.rs` exists under `crates/nsl-codegen/src/`.

### E12 — Citation slips

- [3] Wanda was published at ICLR 2024, not "ICML 2024"
  (https://arxiv.org/abs/2306.11695).
- [5] LLM-NAS exists (https://arxiv.org/abs/2510.01472; Zhu, Zhang,
  Huang) and is evaluated on HW-NAS-Bench. Its abstract says "days to
  minutes". The "~200" candidates attributed to it in the §8.2 table
  could not be verified.
- [1] was checked for title and venue (https://arxiv.org/abs/2305.11627)
  and [2] for title and authors (https://arxiv.org/abs/2407.14679); both
  are correct. The NeurIPS 2024 venue of [2] and references [4], [6] and
  [7] were not re-checked against their sources.

---

## Part 2 — Superseded by later work

- "Rethinking the Value of Training-Free Structured Pruning of LLMs"
  (TMLR 2025, https://mlanthology.org/tmlr/2025/lele2025tmlr-rethinking)
  finds significant degradation on some tasks even at low pruning rates,
  and that depth pruning usually beats width pruning. CEP prunes width by
  default and considers dropping a layer only under `--cep-joint`.
- SNIPER (https://arxiv.org/abs/2608.12953) reports greedy pruning methods
  missing their target budgets by up to 33% and replaces them with a
  knapsack allocation followed by a fine-grained stage. CEP's search is
  greedy too. It stops at the first step that meets the target, so it
  overshoots by less than one step: on the 50M-parameter model shape a 30%
  request gives 30.2%.
- Minitron, the proposal's reference [2], recovers accuracy with knowledge
  distillation on under 3% of the training data (abstract,
  https://arxiv.org/abs/2407.14679). The proposal's recovery recipe (§6.1)
  is two epochs of plain cross-entropy at learning rate 1e-5, although NSL
  already has a distillation feature, CPKD.
- FLAP (https://ar5iv.labs.arxiv.org/html/2312.11983) uses one forward
  pass to obtain both a pruning metric and a bias compensation for the
  removed components. CEP has no compensation step. 2SSP (TMLR 09/2025,
  https://arxiv.org/abs/2501.17771) combines a width stage with a depth
  stage.
- SliceGPT and Minitron prune the residual width, and angular-distance or
  block-influence criteria choose which layers to drop (not re-checked
  against the sources). CEP does not prune `d_model` and uses no
  activation signal.
- The LLMCompass paper (ISCA 2024, https://arxiv.org/pdf/2312.03134)
  states that roofline analysis is "fast but not accurate" and reports
  about 4% end-to-end error for its own operator-level simulator. A
  roofline-based oracle needs a stated error, not the word "exact". CEP's
  estimate has no measured error. It does add the GPU database's
  `kernel_launch_overhead_ns` per kernel launch; for small models at batch
  1 that term dominates.

---

## Part 3 — What the code implements

CEP is listed as Experimental in `STATUS.md`. The code is in
`crates/nsl-codegen/src/` (`cep.rs`, `cep_oracle.rs`, `cep_importance.rs`,
`cep_search.rs`, `cep_rewrite.rs`, `cep_slice.rs`, `cep_weights.rs`,
`cep_extract.rs`, `cep_emit_source.rs`), with decorator validation in
`crates/nsl-semantic/src/cep.rs` and the command-line entry points in
`crates/nsl-cli/src/commands/cep.rs`.

How it is turned on:

- `nsl build` takes `--cep-prune` or `--cep-joint` (both require
  `--weights`, an `.nslm` or `.safetensors` file), with `--cep-target`,
  `--cep-sparsity`, `--cep-out`, `--cep-emit-weights` and
  `--cep-emit-source`. The proposal's `--sparsity` and `--target`
  spellings are not the CEP flags. A plan that misses its sparsity target
  or a constraint exits with status 1 and writes nothing.
- `nsl check` takes `--cep-search` and `--cep-profile`.
- Decorators: `@cep_prune(target, sparsity, granularity, preserve,
  constraints={peak_memory_bytes, latency_us})`, `@cep_search(target,
  objective, constraints)` and `@search(axis, [values])`. The `objective`
  of `@cep_search` is required; there is no default.
- The proposal's target names do not resolve. `--cep-target rtx5070ti` is
  an error (the database name is `RTX-5070-Ti`), and there is no RTX 4060
  entry.
- No shipped model carries a CEP decorator. One integration test prunes a
  model with non-zero weights and runs the result; the others use all-zero
  weights and check shapes and reports.

Component by component:

- Compilation oracle (§2.1) — partial; an analytic estimate, not a
  compilation. `evaluate` in `cep_oracle.rs` is closed-form arithmetic on
  a 13-field `ModelSpec` and calls only the matmul, softmax, norm and
  embedding cost functions of `cost_model.rs`. Fusion events and kernel
  launch counts come from a fixed template (`detect_fusion_events`,
  `block_kernel_launches`). No code generation, memory planner or fusion
  pass runs. The estimate is for one forward pass of batch 1 over the
  model's maximum sequence length; its reports say so. Neither side is
  right: the proposal's "exact" is unachievable, and the implementation
  should not be called a compilation oracle.
- "~50ms" per candidate (§2.1) — the implementation takes microseconds
  (216 candidates in 0.00 s), because it does not compile anything.
- Pruned computation "physically absent from the binary" — report-only.
  `--cep-prune` prints a report and exits; no binary is produced. The user
  rebuilds from the files written by `--cep-emit-source` and
  `--cep-emit-weights`.
- Weight slicing — implemented, in `apply_prune_delta_to_weights`
  (`cep_slice.rs`). It slices `wq`/`wk`/`wv` columns, `wo` rows and the
  three FFN matrices by the GQA group mapping, rewrites the head counts
  the attention stores as tensors, and moves every tensor of a renumbered
  layer. The sliced weights load into the rewritten source and give the
  forward output of the unpruned model when the pruned components were
  dead. Residual width is untouched, so the norms need no change for head
  and FFN pruning.
- A different block variant per layer (§3.2) — absent. NSL declares all
  layers with one block constructor, so `derive_uniform_pruned_spec` in
  `cep_emit_source.rs` refuses any non-uniform result. When
  `--cep-emit-source` is given the search therefore prunes all layers in
  step (each layer still loses its own least important heads and
  neurons), and `preserve` is refused with that flag. Without it the plan
  is per layer and only the delta and the weights can be written.
- The greedy rule of §3.4 — differs. `prune_greedy` and
  `joint_prune_search` (`cep_search.rs`) share one loop that removes, at
  each step, the head group, FFN step or (joint) layer with the least
  importance per parameter saved, and stops when the sparsity target and
  both limits are met or nothing more may be removed. Head scores and
  neuron magnitudes are made comparable by dividing each by the mean of
  its per-layer total. The implementation is right about the stopping
  rule; see E1. The cross-kind normalisation is a heuristic of the
  implementation's own.
- Per-neuron FFN importance `‖w_in‖₁ × ‖w_out‖₁` — differs. The planner
  and the slicer both rank a neuron by the product of the L2 norms of its
  `w_gate` column, `w_up` column and `w_down` row
  (`ffn_neuron_magnitudes`). The FFN here has three matrices and the
  score uses all of them.
- Spectral energy of the head's own slice — differs. `analyse_weight_map`
  (`cep_importance.rs`) uses σ₁²/‖W‖_F² of the whole `Wq`, which is the
  same for every head in a layer. Neither side is right: as coded it is a
  per-layer multiplier.
- Roofline slack from the cost model — placeholder. `slack_from_profile`
  (`cep.rs`) assigns the constant 0.8 or 1.5 by a compute-bound flag. It
  is uniform across the layers of a uniform model, so it never changes
  the ranking.
- Position factor — differs. `position_factor` (`cep_importance.rs`) uses
  `l/(L−1)` and is symmetric. The implementation is right; see E3.
- `head_dim = d_model / n_heads` in search (§4.3) — differs.
  `SearchAxes::enumerate` (`cep_rewrite.rs`) holds `head_dim` at the base
  model's value for every candidate and does not require `n_heads` to
  divide `d_model`. NSL's attention takes the head dimension as an
  explicit argument, so the implementation is right.
- Constraints `peak_memory < 8GB`, `latency_per_token < 5ms`,
  `params < 20M` — partial. Only `peak_memory_bytes` (an integer) and
  `latency_us` exist; there is no parameter constraint, and the latency
  is for a whole forward pass over the maximum sequence length, not per
  token.
- Inline `@search(d_model, [..])` inside expressions (§6.2) — absent; only
  the top-level decorator form exists.
- Joint search over layer subsets (§2.4) — partial. `joint_prune_search`
  is the greedy loop above with dropping a layer as a third kind of step.
  It follows one path and never revisits a choice, which is much weaker
  than the search described.
- Pareto set of configurations — absent; one configuration is returned.
- Activation and norm type as search axes (§4.3) — partial. LayerNorm
  costs more parameters and FLOPs than RMSNorm. The activation changes
  only the FFN matmul count, so ReLU, GELU and SiLU are the same candidate
  to the estimate; the report lists them once.

`--cpkd-design-student` reuses `architecture_search` and
`analyse_weight_map` (`cpkd_student.rs`), so what is said here about the
search and the importance analysis applies to it too.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Describe the oracle as an analytic estimate with a measured error bar
   (5–10 measured configurations per GPU), including launch overhead.
2. Replace the §3.4 rule: prune until the constraints and the sparsity
   target are both met, and report failure when either is missed.
3. Add a calibration-based importance mode (one forward pass, FLAP-style
   bias compensation); keep magnitude only as a fallback labelled weak.
4. Replace the greedy phases with a budget allocation over (layer,
   component) items, and require an explicit budget for the search.
5. Add depth and residual-width pruning, per-layer block variants, and
   distillation (CPKD) as the default recovery.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code, on CPU and through the analysis-only CEP
flags. This part lists what it found wrong in the implementation, as
distinct from the proposal's own errors in Part 1. Fixes are described in
`CHANGELOG.md` under "Unreleased". The lists are as of 2026-11-13.

### Fixed

- Head importance was read along the wrong axis. `wq` is stored as
  `[d_model, n_heads·head_dim]`, but rows were sliced whenever `n_heads`
  divided the row count, `wk`/`wv` were cut by Q head instead of KV group,
  and `wo` was never read; with two heads of a layer zeroed, CEP pruned the
  two live ones. A head's magnitude is now
  `‖Wq_h‖·‖Wk_g‖·‖Wv_g‖·‖Wo_h‖`, taken over the head's columns of `wq`,
  its KV group's columns of `wk` and `wv`, and its rows of `wo`
  (`AttentionWeights::head_magnitude` in `cep_importance.rs`). A projection
  whose shape is not NSL's `[in, out]` is an error.
- Surviving FFN neurons were chosen by the column norm of `w_gate` alone:
  with 64 neurons given a large gate column and a zero `w_down` row, all 64
  survivors were dead. Neurons are now ranked by
  `‖gate_j‖·‖up_j‖·‖down_j‖` (`ffn_neuron_magnitudes`), and the planner
  and the slicer use the same ranking.
- FFN importance was never computed for NSL models: the lookup wanted names
  such as `ffn.w_in`, and NSL's are `ffn.w_gate`, `w_up` and `w_down`.
  `analyse_weight_map` now reads NSL's names, records a magnitude per
  neuron, and fails when any of a layer's seven weight tensors is missing
  instead of scoring it zero.
- `--cep-joint` with a dropped layer left the norm weights on the wrong
  layer: only `attn` and `ffn` tensors were dropped and renumbered. Every
  tensor a block owns is now dropped or renumbered with it
  (`apply_prune_delta_to_weights` in `cep_slice.rs`).
- The parameter count added an LM head even when it is tied to the
  embedding: `nsl check --cep-profile models/coder50m/model.nsl` printed
  "Params: 73.9M" for a model of 48,767,488, and sparsity was measured
  against the inflated count. A head is now counted only when the model
  declares a second `[d_model, vocab]` matrix; the command prints
  "Params: 48.8M (48767488)".
- The estimate's FLOPs, bytes and latency did not form one model.
  Attention was costed as a softmax, without the QKᵀ and A·V matmuls, and
  its bytes assumed a stored score matrix while a FlashAttention fusion was
  credited; arithmetic intensity divided all FLOPs by some of the bytes;
  launch overhead was omitted; values were always priced as 4 bytes. The
  attention core is now one fused kernel that is charged both matmuls and
  the softmax and stores no score matrix, intensity uses the layer's own
  bytes, and each kernel launch adds the target's launch overhead. The
  value width is read from the weights file when one is given. The batch
  is still 1 and the sequence length the model's maximum; every report
  prints those assumptions. The fusion events, the launch counts and the
  code-size allowance are still a fixed template, and are labelled as such.
- Memory and latency constraints could not steer the search: a step was
  rejected when the result would still exceed the limit, so a baseline over
  the limit was returned unchanged. A limit is now a target: the search
  keeps removing components until the sparsity target and both limits hold
  (`prune_search` in `cep_search.rs`).
- A missed sparsity target was reported as success: requests of 0.3 and 0.9
  gave 11.5% and 34.5%, both printing "All constraints satisfied." with
  exit status 0. A plan that misses its target or a limit now prints
  "NOT MET" with the figures, exits with status 1, and writes no file.
- `--cep-joint --cep-emit-source` wrote no source file and said nothing,
  and the joint delta file recorded `"mode": "prune"`. The source is now
  written, with the block array shortened for dropped layers, and the file
  records `"mode": "joint"`.
- Each one-head step of `--cep-prune` removed a whole GQA group, and once a
  layer was down to one group further steps changed nothing yet were logged
  "[OK] ... pruned". A step is now one KV group, the log lists only steps
  that were applied, and a model with nothing that may be removed is
  reported as such.
- The search pruned every layer's heads to the floor before it touched any
  FFN, then took one layer's FFN to its minimum before the next; a `d_ff`
  of 128 had one candidate width; the retained-capacity score steered
  nothing. Each step now takes whichever head group or FFN step (an eighth
  of the layer's width) loses the least importance per parameter saved,
  across all layers, and the retained capacity is what that ranking
  minimises the loss of.
- The sliced weights file could not be loaded into the emitted source: the
  stored scalars `_n_heads` and `_n_kv_heads` were copied through
  unchanged, so the forward failed on a reshape, and tensor order was not
  preserved. The slicer rewrites the stored head counts and keeps the input
  order, and CEP reads and writes `.nslm` directly, so
  `--weights base.nslm --cep-emit-weights pruned.nslm` gives a file that
  `model_load` accepts in the emitted source. A test runs that chain and
  compares the pruned model's output with a NumPy reference.
- CEP rejected checkpoints written by `model_save` and `nsl export`, which
  name tensors `blocks[0].attn.wq`. Both spellings are now read
  (`BlockIndex` in `cep_weights.rs`) and names are written back in the
  spelling they were read in.
- With `--cep-emit-weights` and `--cep-emit-source` together, a non-uniform
  result exited 1 on the source after the weights file had been written.
  All outputs are now produced in memory and written only if every one
  succeeded, and with `--cep-emit-source` the search keeps all layers at
  one shape so that the source can express the result.
- `nsl check --cep-search` ranked the smallest stack first by default,
  listed `rmsnorm` and `layernorm` duplicates as ranks 1 and 2, and with
  "Feasible: 0 candidates" still printed "Top 3 architectures", exit
  status 0. `@cep_search` now requires an `objective`; `param_efficiency`
  is printed with a note on what it cannot tell; candidates whose metrics
  are all equal are listed once (LayerNorm now costs more than RMSNorm, so
  those two no longer are); and a search with no feasible candidate marks
  its rows "(infeasible)", exits with status 1 and writes no file.
- `--cep-profile` labelled the time of a whole forward over the maximum
  sequence length as "us/token". It now prints the figure "per forward
  pass" with the per-token average beside it, and the search and prune
  reports say "us/forward".
- The pass record counted rejected steps as applied rewrites. It counts the
  steps applied; there are no rejected steps any more.
- Two tests asserted the constraint defect as expected behaviour. They are
  replaced by tests that a limit is pruned towards and that an unreachable
  one is reported.
- Five tests could not fail or compared the estimate with itself. Four are
  rewritten against hand-worked values (the reduction achieved, a known
  spectrum, the bytes a fusion saves); `weight_l1` and its test are gone
  with the code that used it; `cep_g18_e2e_binary_claim` is removed, and a
  test that builds and runs the pruned model takes its place.
- The search fixture used by the command-line test collapsed to one
  candidate. It now spans nine, and the test checks their order.

### Open

None of the defects found in this review remains open. Limits that are
part of the design, not defects, are described in Part 3.
