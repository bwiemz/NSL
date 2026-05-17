## V-B.2-predicate — Findings

**Spec reference:** §7.3 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
**Date:** 2026-05-13
**Budget used:** ~25 minutes
**Worktree HEAD:** `87384c75`
**Branch:** `worktree-feat-pca-tier-b15-and-b2`

---

### 1. Forward's iteration order (baseline)

Each forward CTA owns one q-tile (`grid_x = num_q_tiles`); within the CTA the
emitter unrolls the per-warp q sub-iterations (`q_iter ∈ 0..(block_q/4)`) at
Rust time, and each q_iter contains a PTX-runtime loop over kv-tiles. The
predicate is currently emitted *inside* the kv-tile loop body, just before
the QK^T inner work.

Evidence:

- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:90` — `for q_iter in 0..iters` (Rust-unrolled).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:179,189-191`
  ```
  V2_LOOP_KV_S_{q_iter}:
      ...
      add.u64 %k_start, %k_start, {block_kv};
      setp.lt.u64 %p0, %k_start, %k_max;
      @%p0 bra V2_LOOP_KV_S_{q_iter};
  ```
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs:113-121`
  — call site for `emit_skip_predicate` with `qt_reg = "%bid_x"` (CTA-level
  q-tile ordinal) and `kvt_reg = "%r_kvt_ord_TB"` (derived from `%k_start`).

Structural summary (forward): **Q-outer (Rust-unrolled per q_iter, with the
CTA itself fixing the global q-tile via `%bid_x`) / KV-inner (PTX runtime
loop)**. Predicate fires once per (`%bid_x`, `%k_start >> log2(block_kv)`)
pair inside the kv-tile loop body.

---

### 2. Backward's iteration order

**Case: (β) — KV-outer / Q-inner (within-CTA).**

Backward's CTA grid is identical to forward's at the outer level — each CTA
owns one q-block via `%bid_x` (`crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs:330-335`,
`cvt.u64.u32 %q_start, %bid_x; mul.lo.u64 %q_start, %q_start, block_q;`).
The CTA-level launch contract is therefore the same Q-outer mapping.

But **within** each CTA the loop nesting is flipped relative to forward:

- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:741` — `V2_BWD_LOOP_KV:`
  (PTX-runtime kv-tile loop, **outer**).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:742-743` — `kv_load::emit_k_suffixed` / `emit_v_suffixed`
  (K, V reloaded each kv iter).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:744-773` — dV / dK SMEM
  zero-init per kv-tile.
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:775` — `for q_iter in 0..iters`
  (Rust-unrolled q sub-iterations, **inner**).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:815-817` —
  `ds_compute::emit(q_iter); dv_accum::emit(q_iter); dqdk_accum::emit(q_iter);`
  the per-q phase calls live inside the kv-outer loop.
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs:854-857` —
  `finalize::emit_store_kv_only(... , 0)`; `add.u64 %k_start, %k_start, block_kv;`
  `setp.lt.u64 %p0, %k_start, %k_max;` `@%p0 bra V2_BWD_LOOP_KV;` — dK/dV
  flushed to f32 scratch at the bottom of each kv iter (Option A residency
  pattern; the classic FA-2 backward convention).

So within a backward CTA the iteration runs `for kvt { for qt { ... } }`.
This is the canonical **FA-2 backward dK/dV-resident convention** mentioned
in §7.3.1 as case (β).

---

### 3. Predicate-emission shape decision

Per spec §7.3.2 the verification answers "does `emit_skip_predicate` need an
`IterationOrder` parameter, or does it reuse wholesale?"

The spec's §7.3 case-(β) outcome is binding: **B.2 proceeds with the
parameterized form** (`IterationOrder { QOuter, KVOuter }`). The spec commits
to this in §7.3 line "Outcome if case (β): `emit_skip_predicate` gains an
`IterationOrder` parameter..." and §7.3.4 "If V-B.2-predicate surfaces case
(β), B.2 implementation **proceeds with the parameterized form** (not the
wholesale-reuse form). ... ~50 lines split between `pca_tilerange.rs`
(predicate emission) and `ds_compute.rs` (call site)."

#### Technical observation (informational)

The current `emit_skip_predicate` body in `crates/nsl-codegen/src/pca_tilerange.rs:368-441`
is structurally symmetric in `qt_reg` and `kvt_reg` — it loads
`(qmin[qt], qmax[qt])` and `(kvmin[kvt], kvmax[kvt])` symmetrically, then
tests `(qmax < kvmin) || (qmin > kvmax)`. Both `qt_reg` and `kvt_reg` arrive
as caller-supplied register names, and the four range tables are addressable
identically regardless of which axis is the outer loop. The predicate is
therefore **mathematically and PTX-structurally symmetric** under loop swap.

What changes between cases (α) and (β) is the **call site**, not the body:

- Forward (α-ish, see §1): qt_reg=`%bid_x` (CTA-uniform), kvt_reg derived
  from `%k_start` via shr by `log2(block_kv)`.
- Backward (β): qt_reg=`%bid_x` (CTA-uniform, unchanged), kvt_reg again
  derived from `%k_start` via shr by `log2(block_kv)` — but `%k_start` is
  now the OUTER loop's induction var rather than the inner.

Both `%bid_x` and the shifted-`%k_start` are warp-uniform CTA-uniform values
in both directions, so ptxas's uniformity inference should produce a uniform
`BRA.U` for the back-edge of `@%p_skip_TB bra <skip_label>` either way.

The `IterationOrder` parameter is therefore primarily a **call-site /
documentation contract**, not a body-level branch in the PTX template. It
will be most useful for: (a) selecting the appropriate `on_skip_label`
naming convention (so backward's skip lands at the correct point in the
KV-outer loop), and (b) making the contract between caller and helper
explicit at the type level so a future B.3 forward-only refactor cannot
accidentally drop the kv-outer call site.

Per the spec's binding decision, B.2 will still add the parameter.

#### B.2 scope estimate

**~200 LOC total** for B.2 (matches the case-(β) projection in §7.3.4
risk register at spec line 575):

- `pca_tilerange.rs::emit_skip_predicate` — add `IterationOrder` enum +
  parameter; conditional `on_skip_label`/`@%p_skip_TB bra` placement;
  ~30–50 LOC including the enum decl + doc comments.
- `flash_attention_v2/phases/backward/ds_compute.rs` — insert the
  Tier-B-gated predicate call at the head of the ds-compute body
  (one per (q_iter, kvt) pair), mirroring `s_compute.rs:92-124`'s shape;
  ~50–80 LOC including the `cfg.csha.is_some_and(...)` gating + the
  `%r_kvt_ord_TB` derivation + scope braces.
- `flash_attention_v2/phases/backward/prelude.rs` — add the range-table
  preamble call to `emit_range_table_preamble`, mirroring forward's
  prelude wiring (already imported via `phases::backward::prelude`);
  ~30 LOC.
- Backward snapshot refresh — `.snap.new` for at least one segment-masked
  causal backward variant (e.g. the existing `backward_kernel_segment_masked_causal_32_32_32.snap.new`
  staged in the worktree); ~20 LOC of `cargo insta accept`'d output.
- Tests / no-op guarantee — extend `synthesize_backward(config)` to take
  a `tier_b: Option<...>` parameter the way forward did in PR #168; one
  short test that asserts byte-identical output when `tier_b = None`;
  ~20 LOC.

These add up to ~150–200 LOC. Allowing for review-driven scope creep
(label-collision fixes, register-pool extensions in `prelude.rs`, possibly
a small `pca_tilerange::iteration_order` doctest), **estimate 200 LOC**.

---

### 4. Sub-questions answered

- **Q:** Does backward's iteration order match forward's at the
  predicate-evaluation point?
  **A:** **No.** Within a CTA, forward is Q-outer / KV-inner (Rust-unrolled
  q_iter wraps a PTX-runtime kv loop) while backward is KV-outer / Q-inner
  (a PTX-runtime kv loop wraps the Rust-unrolled q_iter). Both share the
  CTA-level Q-outer launch contract (`%bid_x = q-tile`), so the qt operand
  is unchanged; only the loop nesting and skip-label placement differs.

- **Q:** Does ptxas's uniformity tracking still produce `BRA.U` for the
  predicate's back-edge regardless of iteration order?
  **A:** **Yes.** Both `%bid_x` and `%k_start >> log2(block_kv)` are
  CTA-uniform in both directions (the latter because `%k_start` is the
  same induction variable for every thread in the block, and `shr` of
  a uniform input produces a uniform output). The `setp.lt.u16 %qmax,
  %kvmin` / `setp.gt.u16 %qmin, %kvmax` operands are therefore both
  uniform, and ptxas should emit `@%p_skip_TB bra ...` as `BRA.U` in
  both cases. Operand order in the `setp` does not affect uniformity
  inference (uniformity is per-operand, not order-dependent).

- **Q:** Are the four range tables (`qtile_min/max`, `kvtile_min/max`)
  accessible from backward's loop structure without rearrangement?
  **A:** **Yes — symmetric addressing.** The tables are SMEM-resident
  via `tier_b_range_table_offset(config, Direction::Backward)` (mirrors
  forward's `Direction::Forward` slot; `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs`
  defines both directions). Both `qt_reg` and `kvt_reg` index into their
  own tables independently, so the kv-outer / q-inner order doesn't
  require any table-layout transpose.

---

### 5. Decision

**B.2 proceeds with the parameterized `emit_skip_predicate`** form
(`IterationOrder { QOuter, KVOuter }` enum added), per spec §7.3.4's
binding case-(β) outcome.

Estimated total B.2 scope: **~200 LOC**.

---

### 6. Open questions for B2-2

- **Per-phase predicate placement.** Backward emits THREE matmul phases per
  (q_iter, kvt) pair: `ds_compute::emit`, `dv_accum::emit`, `dqdk_accum::emit`
  (`crates/nsl-codegen/src/flash_attention_v2/mod.rs:815-817`). Does the
  skip predicate fire ONCE per (q_iter, kvt) and branch over all three
  phases, or once per phase? Recommendation: **once per (q_iter, kvt)** —
  insert at the top of `ds_compute::emit` and choose `on_skip_label` so
  the branch target lands at the end of `dqdk_accum::emit` (skipping
  ds + dv + dqdk together). This matches forward's "skip the whole tile
  body" semantic and avoids three redundant range-table lookups per
  (q_iter, kvt) pair. **Verify in B2-2 implementation.**

- **dK/dV residency interaction.** Because backward zeros dK/dV at the
  *top* of each kv iter and flushes them at the *bottom* of each kv iter
  (the KV-outer loop body at `mod.rs:744-773` and `:854`), a skipped
  (q_iter, kvt) contributes zero to dK/dV for that kv-tile. The zero-init
  + skip + flush sequence is **correct** under the §7.1 symmetric-zero
  property: P=0 ⇒ dV=P^T·dO=0, dP=dO·V^T=0, dS=P⊙(dP - D)=0, so the dK/dV
  scratch sees a no-op RMW. **Confirmed; no B2-2 action needed.**

- **dQ residency.** dQ is zero-init'd ONCE at the top of the kernel
  (`mod.rs:701-725`) and accumulates across ALL kv iterations, with the
  final flush in `emit_store_dq_only` at `mod.rs:871` AFTER the KV loop
  exits. A skipped (q_iter, kvt) contributes zero to dQ for that kvt;
  the cross-kvt persistence of the dQ accumulator is what makes the skip
  safe. **Confirmed.**

- **Label-name namespace.** Forward uses `KV_TILE_SKIP_TB_{q_tile_iter}`;
  backward will need a distinct namespace (e.g. `BWD_KV_TILE_SKIP_TB_{q_iter}_{kvt}`
  or just `BWD_KV_TILE_SKIP_TB_{q_iter}` if the predicate fires once per
  q_iter inside the kv-outer loop). PTX labels are entry-scoped, so
  collision with forward's `KV_TILE_SKIP_TB_*` is technically impossible,
  but the explicit `BWD_` prefix is the project convention (see backward
  prelude's `BW_PCA_LOAD_*` labels, `prelude.rs:366-389`). **Adopt
  `BWD_KV_TILE_SKIP_TB_{q_iter}` for B2-2.**

- **`pca_tilerange::should_emit_tier_b` gating.** Forward gates the
  Tier-B predicate emission on `should_emit_tier_b(config, seq_len, residency)`
  (`s_compute.rs:93`). Backward should gate identically. The gating
  function reads `config.segment_masked` + `seq_len`/`block_kv` ratios
  and is direction-agnostic, so reuse is straightforward. **No new gate
  function needed.**

---

### 7. Gate verdict

V-B.2-predicate is **GREEN** — case (β) pinned with file:line evidence,
the §7.3.4 parameterized-form path is the documented case-(β) outcome,
and B.2 scope (~200 LOC) is sized per the §7.3.4 risk-register
projection (line 575: "~50 additional lines"). The open questions are
operational (label naming, per-phase vs per-(q_iter,kvt) placement) and
do not block B.2 start.
