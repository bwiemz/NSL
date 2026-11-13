# NSL-WGGO Research — Errata & Resolved Inconsistencies

**Companion to** `NSL-WGGO-Research.md.pdf`. **Date:** 2026-06-30.

The 2026-06 paper-vs-implementation audit surfaced three internal
inconsistencies between the paper's **formal model** (§2.2 decision variables,
§2.4 constraints) and its **algorithm** (§3.2 Level-1 DP). This errata records a
recommended resolution for each, chosen to match the shipped implementation
where the code already embodies a coherent design choice. Each entry states the
chosen interpretation, the rationale, what the code does today, and which
downstream audit gap the resolution unblocks. Where a resolution *narrows* the
paper's model, the broader form is preserved as a future extension.

> **Author sign-off required.** These are recommendations, not unilateral
> changes to the research. Each resolution is reversible by editing this file
> and the next paper revision. Items marked **Open sub-decision** genuinely
> need the author's call before the corresponding code gap is closed.

---

## E1 — ZeRO sharding: one factor, not three independent factors

**Inconsistency.** §2.2 declares three *independent* decision variables —
`s_p[l]` (parameter shard), `s_g[l]` (gradient shard), and an optimizer-state
shard — each over `{1, …, N}`. But §3.2's Level-1 DP transition reads "Choose
ZeRO shard factor for this stage" (**singular**), and the DP state carries a
single accumulated term, not three.

**Resolution (recommended).** Collapse to a single per-layer ZeRO factor
`z[l] ∈ {1, N}`. Align §2.2 to §3.2.

**Rationale.**
- ZeRO in practice is *nested*, not independent: stage-1 shards optimizer
  state, stage-2 adds gradients, stage-3 adds parameters. Valid configurations
  form a chain (monotone in sharding aggressiveness), not a free product of
  three factors. A single factor captures the real design space; three
  independent factors admit incoherent points (e.g. sharded parameters but
  replicated gradients) that no runtime implements.
- Three independent factors triple the Level-1 DP state width for marginal
  benefit, in tension with the paper's own "solved in milliseconds" claim
  (§3.2, §5.1).

**Shipped code.** `wggo_dp.rs` already models a single factor: `LayerPlan`
exposes `shard_params` / `shard_grads` / `shard_optim`, but the solver sets all
three equal (one `Cand.shard`) and the candidate domain is `{1, N}`. The three
named output fields are a reporting convenience, not independent variables.

**Future extension.** A ZeRO-*stage* enum `{0,1,2,3}` (rather than a binary
factor) would expose the nested chain faithfully without the incoherent
cross-product — the recommended path if partial sharding is ever needed.
*(Unblocks audit gap #6.)*

---

## E2 — PCA is a first-class joint decision variable

**Inconsistency.** §2.1 lists PCA as the 6th technique WGGO co-optimizes
("Annotate attention vertices with segment-ID config"), and §1.1's conflict set
treats all six jointly. But §2.2's decision-variable block and §2.3's objective
omit any PCA variable, and §5's worked example never mentions packing.

**Resolution (recommended).** Add PCA as an explicit per-layer decision
variable `pck[l] ∈ {none, segment_id, tile_skip, multi_seq}` to §2.2, and add
its work-reduction term to the §2.3 forward-time. The §2.2 omission is an
editorial gap, not an intended exclusion.

**Rationale.**
- The shipped ILP *already* decides it: `LayerDecision.packing_mode ∈ {0,1,2,3}`
  with a `packing_modes_mask` feasibility constraint and per-mode
  `packing_savings` in the cost model. PCA is already a joint variable in code;
  the paper should match.
- Leaving it out of §2.2 is precisely *why* the implementation computes
  `packing_mode` but never lowers it (audit gap #4): with no formal variable
  there was no specified consumer. Naming `pck[l]` makes gap #4 well-defined —
  the variable must drive the attention kernel's segment-ID config at Level-3.

**Shipped code.** `wggo_ilp.rs` (`LayerDecision.packing_mode`,
`packing_modes_mask`, `packing_savings`). The chosen value *is* plumbed onward —
into `AppliedLayer.packing_mode` (`wggo_apply.rs`) and then
`WggoOverrides.packing_mode` (`wggo_overrides.rs`) — but no attention kernel
ever reads it (`csha.rs` and `pca_*.rs` have zero references to it); the only
reader is the report renderer (`wggo.rs`). So the decision is computed, carried
through three structs, and dropped before codegen. Lowering it to the kernel's
segment-ID config is the wiring this resolution authorizes. *(Unblocks audit
gap #4.)*

---

## E3 — Per-weight adapter rank: a two-level split (WGGO budget → WRGA allocation)

**Inconsistency.** §2.2 declares `r[l,w] ∈ {0,2,4,8,16}` — an adapter rank *per
weight matrix w, per layer l* — but does not say whether it is a joint-ILP
variable or delegated. The implementation does neither literally: the ILP
carries a *per-layer* `adapter_rank` (not per-matrix), and WRGA's spectral SVD
heuristic allocates an *integer-valued but off-grid* rank per projection
(`RankAllocation.rank: usize`, computed by flooring a spectral share — it can be
3, 5, 7, …, not snapped to the paper's `{0,2,4,8,16}`).

**Resolution (recommended).** Formalize the shipped two-level split. WGGO's
Level-2 ILP chooses a per-layer rank *budget* `r[l] ∈ {0,2,4,8,16}` (discrete,
joint with the other layer decisions); WRGA's roofline/SVD heuristic then
distributes that budget across the layer's weight matrices `w`, producing the
effective `r[l,w]`. The paper's `r[l,w]` is the **product of the two levels**,
not a flat ILP variable.

**Rationale.**
- A flat `r[l,w]` over every weight matrix × every layer explodes the ILP (a
  32-layer model with ~7 matrices/layer → ~224 rank variables), defeating the
  "~20–50 variables per layer" tractability claim (§3.3).
- Per-matrix rank allocation is exactly what spectral analysis does well
  (allocate rank where the singular-value spectrum is rich); the per-layer
  budget is what trades off against the *other* techniques. The split matches
  each decision to the level that can cost it.
- WGGO already constrains *which projections* WRGA may touch via
  `wggo_overrides` placement filtering (PR #272) — a partial realization of
  per-matrix control.

**Shipped code.** `wggo_ilp.rs` (`LayerDecision.adapter_rank`, per-layer);
`wrga.rs::run_spectral` + `apply_wggo_placement_filter` (per-projection
allocation, consumes `wggo_overrides`). The realized rank is
`RankAllocation.rank: usize` (`wrga_spectral.rs`, a floored spectral share);
only the intermediate `effective_rank: f64` is continuous.

**Open sub-decision for the author.** Snap WRGA's off-grid integer SVD rank to
the discrete set `{0,2,4,8,16}` so the realized `r[l,w]` lives in the paper's
declared domain? *(Unblocks audit gap #5 — recommended, small change.)*

---

## Cross-reference: how these unblock the remaining roadmap

| Errata | Resolves the ambiguity behind | Audit gap unblocked |
|--------|-------------------------------|---------------------|
| E1     | ZeRO shard split `s_p`/`s_g`/`s_os` | #6 (medium) |
| E2     | PCA `packing_mode` orphaned from codegen | #4 (high) |
| E3     | per-weight rank `r[l,w]` domain & ownership | #5 (medium) |

The §2.4 **shape-compatibility** constraint (audit gap #1) is *not* an
inconsistency — it was simply unimplemented — and is addressed directly in code
(`wggo_shape.rs`), not here.

---

## Additional errata (2026-11-13)

The entries above reconcile the proposal's formal model with its algorithm.
In November 2026 the proposal was checked again, this time against its own
arithmetic, the papers it cites, and the code that implements it. The entries
below record what that check found. Every quotation was compared with the
PDF, every figure was recomputed, and each external fact says how it was
verified. Section numbers are the PDF's. Entries are ordered by how far a
reader relying on the proposal would be misled. The six techniques keep the
proposal's names: WRGA (adapters), CSHA (attention fusion), FASE (gradient
accumulation), PCA (sequence packing), CEP (pruning) and CPDT (sharding and
optimizer precision).

### E4 — The decomposed solve is not globally optimal

**Proposal.** §3.1 calls the per-layer integer linear programs (ILPs)
"independent once inter-layer decisions fix layer shapes", §4.1 repeats that
they are independent and can be solved in parallel, and §3.4 says WGGO feeds
the kernel generators "the globally-optimal configuration".

**Correction.** Two of the §2.4 constraints are sums over all layers: the
per-GPU memory constraint (`Σ_l [...] ≤ GPU_memory`) and the pruning-quality
constraint (`Σ_l,k h[l,k] × importance(l,k) ≥ min_retained_importance`). Both
couple the layers, so the per-layer problems are not independent and solving
them separately does not give a global optimum of the §2 model. §5 restates
the importance constraint per layer, as
`Σ_k h[k] × importance[k] ≥ min_importance`, without saying that the model
has changed. Alpa, whose decomposition the proposal borrows, makes no
global-optimality claim.

**Source.** For Alpa: checked against the paper,
https://arxiv.org/abs/2201.12023.

### E5 — The objective has no quality term, so most variables are trivial

**Proposal.** §2.3: "minimize: total_step_time(decisions)".

**Correction.** Step time is the only quantity in the objective. With nothing
that rewards model quality, an adapter only costs time, so its best rank is
0; a lower optimizer precision is always cheaper; a higher fusion level is
always cheaper. Only the constraints create trade-offs, and those are
feasibility interactions, not a joint optimisation. The implementation ran
into this: a comment in `crates/nsl-codegen/src/wggo.rs` records that "the
cost-only objective always chose rank 0".

### E6 — §1.1 says no pass ordering works; §6.2 then uses one

**Proposal.** §1.1: "No ordering of independent passes resolves all four
conflicts. The decisions must be made jointly." §6.2 resolves the same
conflicts with the priority order "CEP > CPDT > CSHA > WRGA > FASE > PCA" and
says this "handles 90% of cases in ~500ms"; §7.2 adds "within 5% of optimal".

**Correction.** The two sections contradict each other: §6.2 resolves the
conflicts with exactly the kind of fixed ordering §1.1 rules out. Neither
"90% of cases" nor "within 5% of optimal" has a source. The proposal's own
example disagrees with the second figure. §7.2 reports a step time of 3.8 ms
against 4.2 ms for greedy, and 4.2 / 3.8 = 1.105, so greedy is 10.5% slower;
it reports 1.6 GB against 2.0 GB, so greedy uses 25% more memory.

**Source.** Recomputed from the proposal's own figures.

### E7 — Alpa is misdescribed, including the solve time WGGO is compared to

**Proposal.** §4.1: "Alpa uses ILP for intra-operator parallelism only (one
concern) and takes 30-60 seconds for compilation." The §8.1 table gives
Alpa's solve time as "30-60 seconds", its cost model as "External profiler"
and its computation graph as "Traced at runtime".

**Correction.**

- Solve time. Table 5 of the Alpa paper (its §8.4) gives 2393 s in total,
  about 40 minutes, for GPT-39B on 64 GPUs, and the paper says the search
  takes at most several hours. Nothing in it supports 30–60 seconds, so
  §4.1's "comparable time" compares WGGO with a figure Alpa does not report.
- Concerns. Alpa handles two, not one: an ILP for intra-operator parallelism
  and a dynamic program (DP) for inter-operator, pipeline slicing.
- Cost model. The cost in Alpa's ILP is analytic: bytes communicated divided
  by bandwidth, with compute cost set to zero (its §4.2). Profiling is used
  only for the latency of inter-operator stages.
- Tracing. JAX traces a program ahead of execution, not at runtime.

**Source.** Checked against the paper, https://arxiv.org/abs/2201.12023.

### E8 — The compile-time figures disagree with one another

**Proposal.** §4: "8-layer model: ~1.5 seconds". §7.2: "Optimization time:
2.3 seconds (Level 1: 0.05s, Level 2: 1.8s, Level 3: 0.45s)". §10: "a single
2-5 second compilation pass".

**Correction.** These describe the same pass on an 8-layer model and do not
agree.

- The rows of the §4 table sum, for 8 layers, to 10 + 20 + 200 + 50 +
  8 × 100 + 8 × 50 + 100 + 50 = 1630 ms, or 1.63 s, not ~1.5 s.
- §4.2 says repetition cuts Level 2 "from 8 × 100ms = 800ms to ~150ms".
  With that, the same sum is 980 ms, about 1.0 s.
- §7.2 reports 2.3 s for an 8-layer model, with Level 2 alone at 1.8 s.
- §10 and the §8.1 table say 2–5 seconds; §4 gives ~15 seconds for 96
  layers.
- Evaluating one configuration takes "~1ms" in §2.3 and "microseconds" in
  §10, and §1 speaks of a "50ms compilation oracle".
- The Level 1 DP is "O(L² × S)" in §3.1 and "O(L × S × M)" in §3.2, and it
  explores "~320 states" in §4.1 and "~10K-100K states" in §3.2.

**Source.** Recomputed from the proposal's own figures.

### E9 — The example report in §7.2 is internally inconsistent

**Proposal.** §7.2, the sample output headed "WGGO Global Optimization
Report".

**Correction.** The lines of the sample contradict one another, so none of
its figures should be reused.

- Level 1 says "layers 2-5 thinned, layers 0,1,6,7 full width", but Level 2
  lists layer 6 with 7/8 heads and a feed-forward network (FFN) width of
  1024, the width of thinned layers 2 and 3.
- Conflict [3] defers the fused FASE step "on layers 2-5", although the
  report's `s_g=8` applies to every layer.
- Conflict [1] reports the CSHA downgrade for layer 4 only. Layers 2, 3 and
  5 have the same pruning, and layer 6 keeps CSHA-L2 with a pruned head.
- "0.3M adapters". A rank-r low-rank adapter (LoRA) on a 512 × 512 matrix
  has r × 1024 parameters. The listed adapters (rank 4 on three matrices,
  rank 8 on four) come to 3 × 4 × 1024 + 2 × 2 × 8 × 1024 = 45,056, about
  45K. The same example sets `budget=100K` in §7.1.
- "31.2M pruned model". The listed prunes remove about 4.9M parameters of
  roughly 50M: nine heads at 4 × 64 × 512 each, plus the FFN reductions from
  width 1408 counted over three 512-wide FFN matrices per layer. About half
  of the model is embedding, which this pruning does not touch.
- "98 MB (vs 195 MB unpruned)" is a ratio of 0.50, against a parameter ratio
  of 31.2 / 50 = 0.62 in the line above it.
- The "Trainable params" line adds the adapters to the whole pruned model:
  the example trains the full model and the LoRA adapters at once.

**Source.** Recomputed from the proposal's own figures.

### E10 — The per-layer variable count is too low; §6.1 contradicts §5

**Proposal.** §4.1: Level 2 "solves per-layer with ~30 variables each"; the
§8.1 table: "Decomposed: ~30 vars per layer". §6.1: "the relaxed LP lower
bound is tight because the cost model is nearly linear in most regions".

**Correction.** The count covers only the decision variables of §3.3. The
lookup-table (LUT) linearisation that §5 relies on needs one binary selector
variable per table entry, which is 8 × 5 × 4 = 160 by the proposal's own
count, plus the rows that link the selectors to the decisions. The §6.1
claim about the linear-programming (LP) relaxation contradicts §5, which
says the objective "is nonlinear in the decision variables"; a roofline
model is also a max() of two terms.

**Source.** Recomputed from the proposal's own figures.

### E11 — A roofline estimate is a bound, so the cost model is a proxy

**Proposal.** §3.3: "This is not a surrogate or proxy — it's the actual
roofline-bounded latency estimate from M37."

**Correction.** A roofline figure is a bound computed from peak FLOPs and
memory bandwidth. The implemented cost model shows how far such a bound can
sit from the real time: it predicts roughly 0.5 ms per micro-batch for the
benchmark model whose measured micro-batch time is 632 ms. After allowing
for batch size and GPU, the gap is estimated at about two orders of
magnitude.

**Source.** The 632 ms figure is in
`models/benchmarks/WGGO_SCALING_RESULTS.md`. The prediction and the adjusted
gap are estimates made from the implemented cost table, not measurements.

### E12 — 8-bit optimizer state is offered without a quantisation scheme

**Proposal.** §2.2: "p_m[l] ∈ {8, 16, 32}" and "p_v[l] ∈ {8, 16, 32}"; the
§7.2 example shows "INT8 m/INT8 v".

**Correction.** The proposal treats the two Adam moments as free variables
over three bit widths and specifies no quantisation scheme. 8-bit Adam state
needs block-wise dynamic quantisation, as in bitsandbytes (Dettmers et al.,
2022) (not re-checked against the source). Plain FP16 storage of the second
moment `v` underflows; that was established when the CPDT proposal, which
owns optimizer precision, was checked alongside this one. The implementation
rejects 8-bit moments outright.

### E13 — Smaller slips in the formal model

- `s_os[l]` is used in the §2.4 memory constraint, and again in §5 and
  §7.2, but §2.2 never declares it. §2.2 declares only `s_p` and `s_g`; E1
  above counts the optimizer-state shard as a third declared variable.
- §2.4: "a[l] = 1 AND s_g[l] > 1 → fused_step_after_reduce(l) = 1"
  constrains `fused_step_after_reduce`, which is not a declared variable.
- The §2.4 shape-compatibility constraint is vacuous for a transformer's
  residual stream.
- §1 counts optimizer precision as "4^8 (precision per layer)". §2.2 gives
  each of the two moments three values, which is 9 choices per layer and
  9^8 = 43,046,721 for eight layers, not 4^8 = 65,536.

### E14 — Reference list

- [3] merges two papers. "Combinatorial Register Allocation and Instruction
  Scheduling" is by Castañeda Lozano et al., ACM TOPLAS 2019
  (https://arxiv.org/abs/1804.02452). Shobaki et al., ACM TACO 2013, is a
  different paper
  (https://dblp1.uni-trier.de/rec/journals/taco/ShobakiSR13.html).
  Source: search-result summaries and the DBLP record, not the papers.
- [4] is real but truncated. The authors are Chang, Chen and King, the
  journal is *Computers & Mathematics with Applications* 34(9), and the
  title ends "… in multi-issue processors". A search-result summary of
  Wilken et al. says the only experiment in [4] took 20 minutes for 10
  instructions, so [4] is not evidence that ILP in a compiler is cheap.
  Source: search-result summaries, not the papers.
- [6] is "DeepCompile: A Compiler-Driven Approach to Optimizing Distributed
  Deep Learning Training", Tanaka et al.,
  https://arxiv.org/abs/2504.09983. It schedules ZeRO stage 3 operations
  from profiles (prefetching, selective unsharding, offloading). It does not
  unify pruning, LoRA and fusion, so §8.2's "DeepCompile is moving toward
  unification" overstates it. Source: search-result summaries.
- [5], Unity (OSDI 2022), is listed but never discussed in the text,
  although its title as cited names joint optimisation, the proposal's own
  subject.

---

## Part 2 — Superseded by later work

- **Mist** (EuroSys 2025, https://arxiv.org/abs/2503.19050; read directly)
  is the closest current work. It tunes data, tensor and pipeline
  parallelism together with activation checkpointing, ZeRO stages 0–3 and
  offloading ratios. It predicts runtime and memory in one symbolic pass
  (reported errors 1.8% and 2.1%), then solves a mixed-integer linear
  program (MILP) between stages and a constrained problem within each. It
  reports 1.28× over Megatron-LM and 1.27× over Aceso. The proposal's
  framing that nobody co-optimises several training optimisations is no
  longer true. The idea worth reusing is the predictor driven by symbolic
  shapes; WGGO has no shape-derived predictor at all.
- **DeepCompile** (https://arxiv.org/abs/2504.09983, revised February 2026;
  from search-result summaries) is the right baseline for the ZeRO and
  communication part of WGGO. See E14 for what it does.
- **Importance scoring.** Weight magnitude, which WGGO uses, is the weakest
  criterion in use. Alternatives for heads are gradient-based head
  importance (Michel et al., 2019), Taylor scores (LLM-Pruner) and
  activation-aware scores (Wanda, |W|·‖x‖) (these three not re-checked
  against the sources). For removing whole blocks, scores based on
  representation similarity are standard: angular distance
  (https://arxiv.org/abs/2403.17887) and Block Influence
  (https://arxiv.org/abs/2403.03853), both from search-result summaries.
  `--wggo-layer-prune-fraction` ranks blocks by weight RMS instead.
- **Hardware.** The repository's GPU table
  (`crates/nsl-codegen/src/gpu_specs.rs`) includes B100, B200 and RTX 5070
  Ti, but WGGO does not select a GPU from it on the command line. WGGO's
  default interconnect bandwidth is 300 GB/s
  (`crates/nsl-codegen/src/wggo_dp.rs`) while CPDT's `--cpdt-intra-bw`
  defaults to 900 GB/s for the same link, and WGGO does not read that flag.
- **Solvers.** If a genuinely coupled MILP is ever wanted, Rust has the
  `good_lp` crate with HiGHS or microlp backends (not re-checked against the
  sources). At the current problem size no solver is needed.

---

## Part 3 — What the code implements

WGGO exists and runs, but no ILP is solved. `--wggo [full|greedy|off|auto]`
is the only switch, and `--pretrain-optimized` implies `greedy`. The
`@wggo(...)` decorator is accepted and has no effect. Without `--wggo-weights`
or `--wggo-memory-budget` the plan is the same for every model, and by the
project's own benchmark it does not change step time (see the list at the end
of this part). Files named below without a directory are in
`crates/nsl-codegen/src/`.

- **Level 1 DP (§3.2): implemented, mostly idle.** `solve` in `wggo_dp.rs`
  is a knapsack over 256 memory buckets. In production each layer has one
  candidate on one GPU and two on N GPUs, because
  `ImportanceScores::default()` is always passed. The thin and prune options
  are reachable only through the force flags `--wggo-prune-layers` and
  `--wggo-layer-prune-fraction`. `max_stages > 1` and `zero_stage_search`
  appear to be set only in tests (not fully traced).
- **Level 2 "ILP" (§3.3, §5, §6.1): partial; an exhaustive scan, not an
  ILP.** `solve_layer` in `wggo_ilp.rs` runs nested loops over a
  precomputed cost table of 800 entries. There is no LP relaxation and no
  subtree is pruned; the one bound is a comparison with the best cost so
  far, made after a candidate is fully evaluated. The module comment that
  names `argmin_feasible` as the bounding function is not accurate. The scan
  is exact for its own formulation: checked against an independent
  brute-force search on ten budget and gate combinations, the maximum
  difference was 0. That formulation is monotone in every axis (E5), so the
  answer is the extreme of each axis. When importance scores are supplied
  the scan enumerates all 255 head subsets, although the cost depends only
  on the head count. `full` and `greedy` return identical plans whenever no
  weights file is supplied.
- **Cost model (§2.3, §3.3): partial; constants, not derived from the
  model.** `wggo_cost.rs` builds one table from a layer shape written into
  `run_on_wengert_with_weights` in `wggo.rs`: batch 1, sequence 1024,
  d_model 512, head_dim 64, 8 heads, 4 KV heads, a 2-byte dtype and FFN
  widths from 384 to 2048, costed for an A100. The table is cloned for every
  layer, including the bucket that holds embedding and norm parameters. The
  coefficients are constants: a CSHA bonus of 0.20/0.45/0.65 applied to the
  whole layer including the FFN, backward = 2 × forward, FASE speedup 0.10,
  packing 0.15, thin layer 0.6×, non-layer overhead 15%. The QKᵀ and AV
  matmuls of attention are not modelled; softmax is the only term quadratic
  in sequence length. Forward and backward are per micro-batch at batch 1,
  the optimizer term is per optimizer step, and `grad_accumulation` is not
  read.
- **Conflict handling (§1.1, §6.2): implemented, after the solve.** The four
  conflicts are detected and resolved after solving, in both modes, by
  `greedy_resolve` in `wggo_conflicts.rs`. The code is right to do something
  simple here; the proposal's claim that a joint solve is required is the
  side that is wrong (E6).
- **Importance scoring (§4, step 3): partial.** `--wggo-weights` turns on
  weight-magnitude scoring (`wggo_weight_analysis.rs`). Gradient scoring
  through `@wggo_target` (`wggo_gradient_scorer.rs`) cannot be reached from
  the command line: only `compile_and_calibrate` writes the file it reads,
  and nothing calls that function.
- **Optimizer precision (§2.2): partial.** The default is 32/32 bits. The
  second moment `v` is always 32 bits. The first moment `m` becomes 16 bits
  on every block layer once a `--wggo-weights` file with any nonzero value
  is supplied; under `--wggo-memory-budget` alone it becomes 16 bits only on
  a layer whose share of the budget cannot hold a 32-bit one. A 16-bit first
  moment is stored as BF16. 8-bit is never chosen (E12).

Which decisions reach generated code:

| Decision | Reaches generated code? |
|---|---|
| Head keep-mask `h` | Count only. The fused attention dispatch in `csha.rs` keeps the first k heads, not the heads chosen. |
| FFN width `f` | No. It is dropped, and the plan prints a warning. |
| CSHA level `c` | Yes, as a preference that CSHA re-negotiates. |
| Adapter rank and placement `r` | Yes, through the WRGA placement filter. Rank is 0 unless the model uses `@wrga`. |
| FASE fusion `a` | Yes, through the FASE mode table. |
| Moment precision `p_m`, `p_v` | Only with `--wggo-moment-precision` or `--wggo-memory-budget`, and only for parameters named `blocks.N`, `layers.N` or `h.N`. |
| Packing mode | Validated and logged; a module-level preference only. |
| Parameter shard factor `s_p` | Yes. The minimum across layers becomes a recommendation to CPDT. |
| Gradient and optimizer shard factors, pipeline stage, communication schedule, thin layers | Report only. |
| Whole-layer pruning | Yes, through `--wggo-prune-layers` or `--wggo-layer-prune-fraction`. |

No shipped model uses `@wggo`, `@wggo_target` or `--wggo-weights`. The tests
check direction (for example, that the solver picks the highest savings).
None checks optimality, none uses more than 10 head groups, and none uses a
shape other than d_model 512.

### Benchmark records that cite WGGO

`WGGO_ALL6_RESULTS.md`, `WGGO_MATRIX_RESULTS.md` and
`WGGO_SCALING_RESULTS.md` under `models/benchmarks/` carry WGGO's name but
measure fused attention, tile skipping ("Tier-B") and optimizer-state
offload, not WGGO.

- The one WGGO comparison is null: 2264 ms per micro-batch with the plan
  against 2256 ms without, a difference of 0.35%. The record's sentence that
  WGGO "chose the configuration the numbers validate" is not supported by
  that.
- The plan the record prints ("8/8 heads", "FFN=2048", "m=32b v=32b") is the
  constant default plan. The benchmark model happens to match the built-in
  layer shape.
- The same plan line reads "CSHA-L3 requested → Level2 applied", with the
  reason `smem_208kb_exceeds_180kb`. WGGO's estimate of shared memory (SMEM)
  is 128 KB where CSHA's is 208 KB, so WGGO's L3 request never lands on that
  model.
- "CEP evidence gate: random init ⇒ keep-all" misstates the gate. The gate
  is the absence of a weights file; a weights file of randomly initialised
  values opens it.
- The arithmetic in the tables is correct: tokens per second, the speedups
  of 4.9×, 6.6× and 8.8×, the count of 140 runs, and 16.6 GB.
- Small inconsistencies:
  - `WGGO_MATRIX_RESULTS.md` points at `WGGO_MATRIX_TABLE.md`, which now
    holds the later campaign's numbers (for 512 × 16, 3641 ms in the text
    and 1113 ms in the table).
  - "grows only 1.04×" compares the two ends of the sequence range; the
    4096 × 2 row is 1309 / 1113 = 1.18×.
  - GPU utilisation of "57–80 %" "across the whole matrix" leaves out the
    three fused-off rows, which are at 9–15%.
  - `WGGO_SCALING_RESULTS.md` says accumulation "is free per micro-batch"
    beside 693 ms at accumulation 1 and 632 ms at accumulation 4 and 8.
  - The "16.3M-param" model has 18.6M parameters by its definition, or 16.5M
    without the tied embedding.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Drop "globally-optimal", "ILP" and the Alpa timing comparison; describe
   the pass as a per-layer table scan followed by a priority resolver.
2. Derive each layer's shape, dtype, batch and sequence length from the
   model first. Every time and memory figure WGGO prints depends on it.
3. Replace the enumeration with its closed form: sort heads by importance,
   scan the head count, and apply the conflict rules inside the scan.
4. Calibrate the cost table against measured phase timings, and replace
   weight magnitude with gradient- or activation-based importance.
5. Build a coupled MILP in the style of Mist only if a measured case shows
   the monotone solution losing.

---

## Part 5 — Implementation defects found in the same review

The review that produced the entries above also ran the code. This part
lists what it found wrong in the implementation, as distinct from the
proposal's own errors. Fixed items are described in `CHANGELOG.md` under
"Unreleased". The open list is as of 2026-11-13. Paths that do not start
with `crates/` are relative to `crates/nsl-codegen/src/`.

### Fixed

- `--wggo-memory-budget` lowered both Adam moments to 16 bits on every block
  at any budget, including one a hundred times the size of the model,
  because the solver's objective is time only and fewer bytes is always
  faster. With no sensitivity evidence, both solvers now keep a 32-bit first
  moment wherever it fits the layer's share of the budget and lower it only
  where it does not (`budget_alone_decides_precision` in `wggo_ilp.rs`).
- The solver could choose a 16-bit second moment, which the runtime stored
  as f16 and which underflows to zero for gradients of realistic size, so
  the step becomes `m_hat / eps`. The second moment's domain is now 32 bits
  only (`second_moment_bits_allowed`), a 16-bit first moment is stored as
  BF16, and the solver's byte accounting matches what is allocated.
- The note "--wggo-importance=auto fell back to magnitude scoring" printed
  three to five times on every compile, including builds that never mention
  WGGO, and said the solver "runs against magnitude-based importance scores"
  when, without `--wggo-weights`, the scores are uniform. It now prints
  once, only when WGGO was requested, and says which case applies
  (`apply_auto_mode_fallback_note` in `compiler/entry_points.rs`).

### Open

- Every plan, time and memory figure is computed for one built-in layer
  (d_model 512, 8 heads, FFN up to 2048), not for the model. A model of
  33,344 parameters and a 12-head, d_model-768 model both got "8/8 heads,
  FFN=2048, CSHA-L3". Where: `wggo.rs`, `run_on_wengert_with_weights`.
  Reproduced by running. Right behaviour: derive each layer's shape, dtype,
  batch and sequence length from the model.
- `--wggo-memory-budget` is checked against that built-in layer. A 200 MiB
  budget on a model whose parameters take 133 KB was refused as infeasible,
  and a large model is undersized by the same arithmetic. The help text now
  says the figures are model units; the sizing is unchanged. Where:
  `wggo.rs`, `run`. Reproduced by running. Right behaviour: size from the
  model, or refuse the flag when the shape is unknown.
- `--wggo-weights` scoring slices every tensor as d_model 512 with 8 heads
  of 64 and never reads its shape. A 12-head, d_model-768 model got 8 "head"
  scores from the first 262,144 values of each matrix and a plan asking for
  6 active heads; a smaller tensor is skipped silently, so d_model below 512
  reads as "no weights". Where: `wggo_weight_analysis.rs`, `score_layer`.
  Reproduced by running. Right behaviour: check shapes, stop on a mismatch.
- A `--wggo-weights` file of randomly initialised weights counts as
  evidence: any nonzero value opens the gate, and the default prune fraction
  is 0.25 with no opt-in. In `full` mode, scores differing by ±0.3% gave 6–7
  heads, FFN 384 and CSHA level 1 on all 8 blocks. Since the fixes above the
  only precision it can cause is a BF16 first moment. Where:
  `wggo_weight_analysis.rs`, `analyze`, `apply_to`. Reproduced by running.
- Head scores are rounded through f32 while the retention threshold keeps
  its f64 value, so the intended top-k set misses the `>=` test by about
  1e-8 in roughly half the layers: eight layers meant to keep 6 heads kept
  6,7,6,6,7,7,6,7. Where: `wggo.rs`, `run_weight_analysis`. Reproduced by
  running. Right behaviour: compute the threshold from the final scores in
  one precision, with a tolerance.
- With `--wggo-prune-fraction 0.0` the keep-all set itself failed that test
  in 2 of 8 layers. Those layers silently took the fallback decision (CSHA
  level 0, FASE not fused, FFN 1024) and the report printed "Step time: inf
  μs" with no warning. Where: `wggo_ilp.rs`, `fallback_decision`;
  `wggo_apply.rs`, `apply`. Reproduced by running. Right behaviour: warn on
  any infeasible layer.
- The importance cache written beside the weights file
  (`<file>.wggo-importance.json`) is keyed by the file's hash alone and
  stores the threshold. After a run at 0.25, `--wggo-prune-fraction 0.5`
  still pruned 2 heads; with a cache from 0.5, a fraction of 0.0 still
  pruned 3–4. Where: `wggo_weight_analysis_cache.rs`, `try_load`. Reproduced
  by running. Right behaviour: key on fraction, layer names and shape too.
- In `full` mode with importance scores the solver picks pruned heads with
  CSHA level 3, the conflict resolver then forces level 1, and the report
  prints the cost from before that change: 251.72 μs for a plan that then
  cost 479.58 μs, about 60% worse per layer than all heads at level 3.
  Where: `wggo.rs`, `splice_resolved`. Reproduced by running. Right
  behaviour: recost, and put "CSHA ≥ 2 ⇒ no pruned heads" inside the scan.
- "Peak memory" is not the plan's memory. The Level 1 DP sizes each kept
  layer at the table's cheapest entry (1 head, FFN 384), 5.8× fewer
  parameter bytes than the 8-head, FFN-2048 plan printed beside it: 76.68 MB
  shown where the plan's own accounting is 377 MB. Where: `wggo_dp.rs`,
  `solve`; `wggo_apply.rs`, `apply`. Numeric check. Right behaviour: sum the
  bytes of the configuration the solver chose.
- The under-sized bytes of the item above (1.44 MB per layer) are also what
  CPDT's ZeRO fit test receives, so it tests a model that does not exist.
  Where: `cpdt_zero.rs`, `ModelSize::from_applied_plan`. By reading the code
  only.
- `--wggo-layer-prune-fraction` ranks blocks by the RMS of all their
  tensors, pooling normalisation gains (all 1.0) with the weight matrices.
  On freshly initialised weights the ranking is noise: scores of 1.0000 and
  0.9993 selected `blocks.1`. Where: `wggo_forced_prune.rs`,
  `select_by_magnitude`. Reproduced by running.
- The target GPU is always A100-SXM: the lookup is given the compile target
  ("cuda", "cpu", "sm_120"), never a GPU name. The per-layer shared-memory
  budget defaults to H100's 228 KB whatever the GPU. Where: `wggo.rs`,
  `run`; `wggo_ilp.rs`, the `Default` of `LayerIlpConstraints`. Reproduced
  by running on a CPU target; for CUDA targets, by reading the code only.
  Right behaviour: use CSHA's `resolve_gpu_spec`.
- The Level 1 DP rounds each layer up to 1/256 of the budget, so 96
  identical layers using 76% of the budget are refused (288 of 256 buckets),
  and 4 layers on 2 GPUs get 2 layers sharded where 1 suffices. Little
  effect today: one candidate per layer, and a refusal degrades to a
  warning. Where: `wggo_dp.rs`, `solve`. Numeric check.
- A fused FASE step subtracts a parameter-sized gradient buffer from the
  layer's activation bytes, which never contained one; at batch 1 this
  zeroes the activations. Where: `wggo_ilp.rs`, `apply_fase`. By reading the
  code only.
- `@wggo(mode=…, target=…)` is accepted and discarded. No plan is produced,
  yet `--activation-report` prints "@wggo: witnessed by construction". The
  validator's result is dropped, and a string `target` stores the argument's
  name, not its value. Where: `crates/nsl-semantic/src/checker/stmt.rs`,
  `check_stmt`; `crates/nsl-semantic/src/wggo.rs`,
  `validate_wggo_decorator`. Reproduced by running.
- A model using the standard library's grouped-query attention through its
  unpacked `forward` got `PCA=segment_id`, a 15% whole-layer discount and a
  `[pca] … wggo-override-consumed` line, beside `packing_detected=false`.
  Any masked attention call in an imported module triggers it. Where:
  `compiler/kernel.rs`, `compile_flash_attention_kernels`; `wggo.rs`,
  `default_constraints_for`. Reproduced by running.
- The 32-bit pin for embedding and LM-head parameters never applies: they
  fall into the catch-all `other` layer, whose role only forbids going below
  16 bits, and under a budget the plan showed `other` at 16 bits. Where:
  `wggo_graph.rs`, `layer_prefix`; `wggo.rs`, `role_sensitivity_floor`.
  Reproduced by running, before the budget fix above; it now needs a budget
  too small for a 32-bit first moment.
- WGGO estimates 128 KB of shared memory for CSHA level 3 on the built-in
  layer; CSHA's own figure is 208 KB against a 180 KB limit, so the level 3
  that WGGO plans and costs is applied as level 2 (Part 3, benchmark
  records). Where: `wggo_cost.rs`, `smem_required`. Numeric check.
- With more than 10 head groups the head enumeration builds all 2^n subsets
  (2³² at 32 heads) and then keeps the first 1024, which are subsets of the
  first ten groups, so keep-all is not a candidate. The comment says "top-k
  groups". Unreachable while the head count is pinned at 8; live once shapes
  come from the model. Where: `wggo_ilp.rs`, `enumerate_head_configs`.
  Numeric check.
- The report's per-layer line prints the kept head count twice ("6/6
  heads"), not kept over total. Where: `wggo.rs`, `render_report`.
  Reproduced by running.
- The decision trace labels optimizer precision "INT32 m, FP32 v"; neither
  moment is an integer. Where: `wggo_ilp.rs`, `build_decision_trace`.
  Reproduced by running.
- The gradient-scoring path compares raw calibration scores with a threshold
  of about 6.0: scores near 1e-3 make every layer infeasible, and scores
  like `[100, 1, …]` keep 1 of 8 heads. Its unit test checks only the report
  field. The path cannot be reached from the command line (Part 3). Where:
  `wggo.rs`, `run_weight_analysis`. Numeric check.
- The §2.4 shape gate cannot fire, because every layer is given the same
  `d_model` or no width at all, and its `reshape_possible` test (one width
  divides the other) is not a valid reshape test. Where: `wggo_shape.rs`,
  `validate`; `wggo.rs`, `inter_layer_dims`. By reading the code only.
- The scorer built as the "magnitude fallback" is handed a provider that
  returns no weights, not the weights file. Where: `wggo.rs`,
  `run_on_wengert_with_weights`, at its call to `build_scorer`. By reading
  the code only.
