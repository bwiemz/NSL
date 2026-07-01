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
