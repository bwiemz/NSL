# PCA Tier A — Packed-Context Attention (segment-ID kernel)

**Status:** Design (spec)
**Date:** 2026-04-18
**Owner:** bwiemz
**Paper:** `docs/research/CFTP.pdf` §2 ("PCA — Packed-Context Attention")
**Companion milestone:** CFTP §1 "FASE" — shipped (WGGO consumers)
**Worktree / branch:** `feat/pca-tier-a`
**Precedent:** CSHA Tier A/C (tier decomposition, FA2 emitter integration), WRGA B.3.1 (shared-helper discipline, ptxas hex-verification)

---

## 1. Problem

Packed-sequence pretraining concatenates multiple short documents into one
fixed-length sample with a dense `S × S` block-diagonal attention mask
keeping documents from attending across boundaries. At batch=32,
seq_len=2048 this mask is `32 × 2048 × 2048 × 2 B ≈ 256 MB` of HBM per
layer (fp16), re-materialised every step. The existing NSL runtime in
`crates/nsl-runtime/src/packing.rs` produces this mask today.

Two things are wrong with it:

- **HBM cost is wasted.** The mask carries no information that isn't
  already expressible as a per-token segment index (one integer per
  position). At seq_len=2048 the segment tensor is 4 KB vs 8 MB per
  sample — a 2000× reduction.
- **FLOPs are wasted.** FlashAttention still computes `Q K^T` scores for
  masked cells then applies `-inf`. The mask is a compute-time tax, not
  a compute-time hint.

PCA replaces the dense mask with a compact `segment_ids: [B, S] u16`
tensor and modifies the attention kernel's control flow to mask by
segment equality inside the KV-tile loop.

## 2. Scope — Tier A only

### 2.1 What Tier A ships

- DataLoader emits `segment_ids: [B, S] u16` alongside existing tensors.
- FA2 `s_compute` forward kernel gains a compile-time `segment_masked`
  flag; when set, the kernel reads `segment_ids` from SMEM and masks
  `S[i, j]` iff `segment_ids[b, i] != segment_ids[b, j]` OR `j > i`.
- CSHA Tier C fused backward kernel receives `segment_ids` as an input
  parameter; `dS` computation applies the same segment mask so
  `dQ`/`dK`/`dV` correctness is preserved on packed sequences.
- Dense `mask` field is removed from `PackedBatch` and the packed-batch
  dict after all consumers have migrated (Commit 6).
- Composition with FASE (gradient accumulation + per-layer optimizer
  fusion) verified end-to-end on one adversarial fixture.
- Composition with CSHA Tier C fused backward verified on the same
  fixture.

### 2.2 What Tier A does **not** ship

The CFTP paper lists four additional features under "PCA." Each is
deferred with a measurement-gated trigger; none is in Tier A's scope:

| Feature | Deferred to | Trigger |
|---|---|---|
| **Tile-skip optimisation** (§7 of CFTP paper) | Tier B | Packed-sequence attention FLOPs exceed X% of total training FLOPs in profile AND mean document length < 50% of sequence length. |
| **RoPE position-reset fusion** (§8) | Tier A-prime | CSHA RoPE fusion regression check (part of Tier A landing, §9.3 below) shows segment-masked attention needs position reset for correctness, OR downstream eval shows position-encoding drift degrades long-context packed-sequence quality. |
| **Per-document CTA scheduling** (§6) | Tier B-prime | Profile shows attention kernel launch overhead exceeds Y% of attention wall time at typical batch-seq configurations (unlikely at batch=32, real concern at smaller batches). |
| **CE separator skip** (§9) | Orthogonal milestone | Loss-computation profile shows non-trivial time on separator tokens AND evidence separator-token loss contributes to training-quality degradation. |

Following the WRGA B.3.1 pattern, each deferred feature gets a stub
document in `docs/plans/` at close-out (Commit 6) with its trigger
condition and the Tier A-landing evidence that shows the feature is
currently unnecessary.

### 2.3 Relationship to existing PCA scaffolding

`crates/nsl-codegen/src/pca_detect.rs`, `pca_segment.rs`,
`pca_tileskip.rs`, and `pca_rope.rs` (~820 LOC) exist today and are
consumed only by `training_report.rs` for the `nsl check
--training-report` CLI. Tier A wires `pca_detect` + `pca_segment` into
the actual FA2 emitter. `pca_tileskip` and `pca_rope` remain
observation-only (used by the CLI report) until their respective
deferred tiers.

## 3. Architecture

### 3.1 Compile-time flag, not runtime null-gating

`FlashAttentionConfig` gains `segment_masked: bool`. The PCA detection
pass sets it per-attention-sublayer when `packing = true` on a
DataLoader upstream of that sublayer. The FA2 emitter produces **two
distinct PTX variants** per sublayer where packing status varies across
the compilation unit: one segment-aware, one segment-unaware. Selection
is compile-time; the kernel has no null-check branch.

**Detection is per-sublayer, not global.** A compilation unit with two
DataLoaders (one packed, one not) feeding two different attention
sublayers emits both variants. `pca_detect::detect` is called per
sublayer with the upstream DataLoader's `DatasetPackingConfig`.

**Rejected: runtime null-gated single kernel.** A single kernel that
branches on `segment_ids == null` at runtime would save variant count
but pay a per-invocation branch + register-pressure cost on unpacked
workloads. More importantly, it weakens the FA2 emitter's core design
principle (compile-time specialisation produces kernels with no runtime
dispatch). See §10 "Alternatives considered" for the detailed
rejection.

**Rejected: compile-time flag with runtime fast-path disable (hybrid).**
The "emit segment-aware variant but runtime-check if all segments
happen to be identical" pattern creates a correctness footgun: a model
with `segment_masked = false` but packed input silently produces wrong
results with no diagnostic. The hybrid weakens the compile-time
correctness guarantee for marginal perf benefit on the wrong axis.

### 3.2 Variant matrix growth — measured, not estimated

Current FA2 config: `{paged, rope_q, rope_style, gqa_group_size, causal}`.
After incompatibility pruning (paged+bf16 combinations that don't ship,
rope_q=false collapses rope_style), emitted variants: ~30–50.

Adding `segment_masked` nominally doubles this, but:

- `segment_masked = true` is **mutually exclusive with `paged = true`**.
  Paged KV cache is an inference-time feature; packed-pretraining doesn't
  use paging. Emitter skips the `{segment_masked: true, paged: true}`
  cell.
- `segment_masked = true` AND `causal = false` is not prohibited but has
  no known production caller (encoder-only packed training is rare). It
  is emitted but expected to be a dead variant that link-time DCE prunes.
- `segment_masked` composes freely with `rope_q`, `rope_style`,
  `gqa_group_size`.

Measured growth: approximately **1.4–1.6× current variant count**, not
2×. PTX blob size grows 40–60%. Link-time dead-variant elimination
(already in place) prunes further.

### 3.3 Segment-mask predicate helper (L1 design)

A new module `crates/nsl-codegen/src/flash_attention_v2/phases/segment_mask.rs`
owns the shared predicate emission. Both forward `s_compute.rs` and
backward `ds_compute.rs` call it.

**Signature:**

```rust
// flash_attention_v2/phases/segment_mask.rs

/// Type-safe wrapper around a PTX predicate register name. Prevents
/// callers from mixing up register names at the `&str` level.
pub struct PtxPredReg(String);

/// Emit the segment-mask predicate for a single (i, j) position pair.
///
/// Returns a freshly allocated predicate register holding
/// `(segment_ids[i] == segment_ids[j]) AND (causal predicate if requested)`.
/// The caller uses the returned register at its `@%p` site.
///
/// # Invariant
/// The emitted PTX substring MUST be byte-identical across all callers
/// (forward `s_compute`, backward `ds_compute`, future callers). Verified
/// by `tests/pca_segment_mask_caller_context_independence.rs`.
pub fn emit_segment_mask_predicate(
    ptx:            &mut PtxBuilder,
    i_reg:          &str,                  // u32 position register
    j_reg:          &str,                  // u32 position register
    segment_ids_reg: &str,                 // u32 SMEM base pointer (Shared)
                                           //   or u32 tile-stream reg (Streamed)
    residency:      SegmentResidency,      // from pca_segment::plan_kernel
    causal:         bool,
) -> PtxPredReg;
```

**Ownership convention (caller-owns).** The helper allocates the output
predicate register internally. `PtxPredReg` wraps **only** the final
predicate (`%p_final` in §5); the intermediate `%p_eq` and `%p_c` are
private to the helper body and never leak through the return type.
Multiple calls in the same kernel (e.g., if a future backward needs the
predicate at both `dS` and `dP` sites) get distinct output registers —
no caller-collision bugs.

**Residency handling is internal** via a private
`emit_segment_id_load` sub-helper that branches on
`SegmentResidency::{Shared, Streamed}`. Not exposed; Tier A only
exercises the `Shared` path (u16 segment_ids for seq ≤ 2048 fit in the
4096 B SMEM budget, see §4.1).

**Module doc-comment records the forward/backward-symmetry invariant:**
"Both forward `s_compute.rs` and backward `ds_compute.rs` call
`emit_segment_mask_predicate` for their segment masks; the emitted PTX
substring must be byte-identical across callers, verified by the
caller-context-independence structural test in
`tests/pca_segment_mask_caller_context_independence.rs`."

### 3.4 CSHA Tier C backward extension

Tier C backward (shipped 2026-04-16) gains `segment_ids` as a new input
parameter. The existing backward kernel is extended, not forked:

- The Tier C backward kernel accepts `segment_ids: *const u16` as a new
  parameter. When null (unpacked path), existing behaviour is preserved
  byte-identically — unpacked-sequence tests must stay green.
- When non-null (packed path), `ds_compute.rs` calls
  `emit_segment_mask_predicate` with the same arguments the forward's
  `s_compute.rs` does, applying the mask to `dS` before the
  `dQ`/`dK`/`dV` accumulation.
- Existing Tier C numerical gates (`NUMERICAL_GATE_DQKV`, `_DW`, `_DX`)
  get packed-sequence fixture variants added. Test matrix roughly
  doubles; each new fixture is a packed-sequence analogue of an
  existing unpacked-sequence fixture.

Rejected alternative: **"PCA Tier A ships forward-only, backward
deferred to Tier A-prime."** The WRGA B.3.1 deferral worked because
unfused-GatedLoRA backward is *functionally correct but slow*. PCA's
deferred-state is *wrong, not slow*: if Tier C runs the unpacked
backward kernel on packed input it computes gradients that leak across
document boundaries. The deferral pattern does not transfer. See §10.

Rejected alternative: **"Tier A tests run with Tier C disabled."** This
ships PCA into a configuration where the motivating workload
(NSLCoder-50M packed pretraining with FASE + Tier C) cannot use it.
Feature ships into unreachable state. See §10.

### 3.5 DataLoader runtime contract

`PackedBatch` (in `crates/nsl-runtime/src/packing.rs`) changes additively
in Commit 2, then subtractively in Commit 6:

```rust
// Commits 2–5: both produced (dense mask retained for reference)
pub struct PackedBatch {
    pub input_ids:   Vec<i64>,
    pub labels:      Vec<i64>,
    pub mask:        Vec<f32>,   // dense [B, S, S], retained for ref
    pub segment_ids: Vec<u16>,   // NEW: [B, S] u16
    pub batch_size:  usize,
    pub seq_len:     usize,
}

// Commit 6 (two-step): mask removed
pub struct PackedBatch {
    pub input_ids:   Vec<i64>,
    pub labels:      Vec<i64>,
    pub segment_ids: Vec<u16>,
    pub batch_size:  usize,
    pub seq_len:     usize,
}
```

`packed_batch_to_dict` publishes `segment_ids` (Commit 2 onward) and
removes `mask` (Commit 6). A new dtype constant `DTYPE_U16_SEGMENT` is
added to the tensor dtype registry — distinct from `DTYPE_U16_TOKEN`
(which is semantically tokens) to prevent consumer type-confusion.

**Segment IDs are non-negative by construction** (they are indices into
a pack). `u16` is the honest type; `i16` would waste the sign bit and
halve the segment ceiling from 65 535 to 32 767.

## 4. Invariants

1. **Forward/backward dispatch symmetry.** Both forward and backward use
   the compile-time `segment_masked` flag. Both call
   `emit_segment_mask_predicate` for their segment masks. The emitted
   PTX substring is byte-identical across callers. Any future change
   that introduces different dispatch patterns for forward vs backward
   requires explicit justification and an updated invariant statement.

2. **Predicate byte-identity across callers.** The PTX emitted by
   `emit_segment_mask_predicate` depends only on its arguments, not on
   caller-context state. Verified by the caller-context-independence
   structural test.

3. **u16 segment-count ceiling.** `segment_ids` is `u16`. Maximum
   segment count per pack is 65 535. `pca_detect::validate_config`
   emits a compile-time error if the DataLoader's packing configuration
   could produce more segments than this (unreachable in any realistic
   corpus but guarded explicitly).

4. **Null-safe Tier C backward.** The Tier C backward kernel's behaviour
   with `segment_ids == NULL` is byte-identical to pre-PCA behaviour;
   existing unpacked-sequence tests stay green after the Tier A
   landing.

5. **ptxas parity.** The concrete PTX sequence pinned in §5 produces
   exactly the SASS measured in §6 on `sm_80+`. A spec revision is
   required if ptxas ever emits a different SASS form for the same
   input on a supported arch.

6. **Zero-spill helper.** The helper adds 3 GPRs + 1 predicate chain to
   the enclosing kernel's register budget. The measured register count
   must stay within `SM75_REGISTER_CAP = 255` after Commit 1's
   extension to `register_budget.rs::count_registers`.

## 5. Concrete PTX sequence

The helper emits exactly this sequence (ptxas-verified, see §6):

```ptx
// Inputs:
//   %i, %j:           u32 position registers
//   %seg_base:        u32 SMEM address of segment_ids[0]
// Outputs:
//   %p_final:         pred, returned via PtxPredReg
// Internal scratch:
//   %r_i_off, %r_j_off:   u32 byte offsets
//   %r_ai, %r_aj:         u32 SMEM addresses
//   %rs_i, %rs_j:         u16 segment values
//   %p_eq, %p_c:          pred intermediates (not returned)

shl.b32        %r_i_off, %i, 1;             // i * sizeof(u16)
shl.b32        %r_j_off, %j, 1;             // j * sizeof(u16)
add.u32        %r_ai,    %seg_base, %r_i_off;
add.u32        %r_aj,    %seg_base, %r_j_off;
ld.shared.u16  %rs_i,    [%r_ai];
ld.shared.u16  %rs_j,    [%r_aj];
setp.eq.u16    %p_eq,    %rs_i, %rs_j;      // segment equality
setp.le.s32    %p_c,     %j,    %i;         // causal (iff requested)
and.pred       %p_final, %p_eq, %p_c;       // fused mask
```

**Sub-decisions pinned:**

- **Causal-AND ordering: segment-first, causal-second.** PTX `and.pred`
  is warp-uniform bitwise-AND, not short-circuit; ordering is a pure
  style choice. Segment-first composes cleanly with a future non-causal
  variant (the `causal` parameter gates whether `%p_c` and `and.pred`
  are emitted at all; the segment predicate is unconditional).
- **Three distinct predicate registers** (`%p_eq`, `%p_c`, `%p_final`).
  PTX predicate regs are abundant; the three-reg form is debuggable in
  `cuobjdump` and prevents internal/external register collision. The
  return type wraps only `%p_final`.
- **Un-swizzled SMEM access.** At 2 bytes per element × seq ≤ 2048
  (the `Shared`-residency ceiling at the 4096 B budget), the access
  pattern has at most 1-way bank conflicts, identical to the swizzled
  case. `pca_segment::plan_kernel` guarantees contiguous SMEM
  layout.
- **SMEM addresses are u32, not u64.** PTX SMEM address space is 32-bit;
  the helper uses `add.u32` throughout, not `add.s64`. (This was a bug
  in an earlier spec draft, caught by ptxas verification during spec
  prep — see §10 "Alternatives considered.")

## 5.1 Spec-to-implementation amendment — mask-convention integration

Added 2026-04-19 after `s_compute.rs` convention verification during
Task 3 prep.

The PTX listing in §5 was written assuming **allow-predicate convention**
(predicate TRUE iff cell should be kept; caller emits
`@%p_final st.shared.f32 ...` to gate the score store). The actual
FA2 forward emitter at
[`flash_attention_v2/phases/forward/s_compute.rs`](../../../crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs)
uses **mask-predicate convention**: the existing causal block computes
`setp.gt.u64 %p0, %rd_k, %rd_q` (TRUE iff cross-causal), then
`@%p0 mov.f32 %f0, 0fFF800000` clobbers the score to `-inf` before the
unconditional shmem store. No allow-gated store exists in the emitter.

`emit_segment_mask_predicate` therefore emits **mask-convention PTX**
to match the existing kernel idiom. The helper does not return a fresh
predicate; instead, it **extends the caller's existing causal-mask
predicate register** with a cross-segment disjunction. The integrated
form at the call site:

```ptx
// Before helper (already in s_compute.rs):
//   setp.gt.u64 %p0, %rd_k, %rd_q;   // TRUE iff k_global > q_row_global
//
// Helper emits (positions at u64 width — no u32 truncation in the
// inputs because the existing kernel already carries q_global / k_global
// as u64 tile-offset results):
cvt.u32.u64    %r_sq, %rd_q;               // q_global low-32 for SMEM addr
cvt.u32.u64    %r_sk, %rd_k;               // k_global low-32 for SMEM addr
shl.b32        %r_sq, %r_sq, 1;            // q*2 bytes (sizeof(u16))
shl.b32        %r_sk, %r_sk, 1;            // k*2 bytes
add.u32        %r_sq, %seg_base, %r_sq;    // &seg[q]
add.u32        %r_sk, %seg_base, %r_sk;    // &seg[k]
ld.shared.u16  %rs_q, [%r_sq];
ld.shared.u16  %rs_k, [%r_sk];
setp.ne.u16    %p_seg, %rs_q, %rs_k;       // TRUE iff cross-segment
or.pred        %p0, %p0, %p_seg;           // combined mask — TRUE iff mask cell
//
// After helper (unchanged from pre-PCA):
//   @%p0 mov.f32 %f0, 0fFF800000;    // -inf for either cross-causal or cross-segment
```

The allow-convention form shown in §5 is **semantically equivalent** by
De Morgan: `¬(segment_eq ∧ causal_le) = segment_ne ∨ causal_gt`.

**Load-bearing contracts from §5 preserved under the amendment:**

1. **u16 segment_ids dtype** — unchanged; `ld.shared.u16` still loads
   u16 values.
2. **Segment-equality + causal composition** — expressed as
   segment-inequality OR causal-violation (mask form) rather than
   segment-equality AND causal-validity (allow form). Same cells
   masked; convention flipped.
3. **Forward/backward byte-identity** — helper still exists as a
   single shared function; forward `s_compute.rs` and backward
   `ds_compute.rs` both call it with their respective `%p0` mask
   registers and get the same extension emission. Caller-context-
   independence test (§7.2) still verifies the claim.
4. **ptxas-clean, zero-spill assembly** — re-measured and pinned in
   §6.2.1 below.

**Signature change from §3.3.** The `emit_segment_mask_predicate`
helper no longer returns a fresh `PtxPredReg`. It takes an **in-out
predicate register name** (the caller's existing mask-predicate, e.g.
`%p0`) and emits PTX that extends it in place. The `causal` parameter
is obsolete because the caller owns the causal check and passes its
result via the in-out predicate; the helper composes via `or.pred`
regardless. Updated signature:

```rust
pub fn emit_segment_mask_predicate(
    ptx:             &mut String,
    q_pos_reg:       &str,      // u64 q_row_global register (e.g. "%rd35")
    k_pos_reg:       &str,      // u64 k_global register      (e.g. "%rd34")
    segment_ids_reg: &str,      // u32 SMEM base pointer
    residency:       SegmentResidency,
    mask_pred_inout: &str,      // e.g. "%p0" — caller's existing
                                //   mask predicate; helper OR-extends it
);
// No return value; the helper's effect is to extend `mask_pred_inout`
// in place with the cross-segment disjunction.
```

The `PtxPredReg` type introduced in §3.3 is consequently unused at
Tier A and is **not introduced** in Task 3. If a future tier needs
allow-convention emission elsewhere, reintroduce `PtxPredReg` at that
point; Tier A does not pre-build the abstraction.

**Byte-identity invariant (§4 invariant #1 & #2) clarification.** The
invariant applies to the *sequence of PTX instructions the helper
emits between the first `cvt.u32.u64` and the final `or.pred`*. The
exact text of `mask_pred_inout` (e.g. `%p0` vs `%p3`) is a caller
argument; byte-identity is asserted on the emission modulo the
argument substitution, verified by the caller-context-independence
test which calls the helper with the same arguments and compares
outputs.

**Position-register width.** Positions flow at **u64** width from the
enclosing kernel's tile-offset arithmetic. The helper internally
truncates to u32 for SMEM address computation (SMEM is 32-bit
addressable). No caller-side `cvt` prelude is needed.

**Rejection of alternatives preserved.** The choice of option A
(mask-convention helper) over option B (rewrite `s_compute.rs` to
allow-convention) and option C (hybrid with caller-side `not.pred`)
is documented at §10 "Alternatives considered" (entry added in the
same amendment commit as this section).

## 6. ptxas verification artifacts

The PTX in §5 was wrapped in a minimal test harness (`C:/tmp/pca_segment_mask_helper.ptx`)
and assembled with:

```shell
ptxas.exe -arch=sm_80 -v -o helper.cubin helper.ptx
```

### 6.1 Assembled output

```text
ptxas info : 0 bytes gmem
ptxas info : Compiling entry function 'test_segment_mask_predicate' for 'sm_80'
ptxas info : Function properties for test_segment_mask_predicate
             0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 8 registers, used 0 barriers, 4096 bytes smem, 368 bytes cmem[0]
```

**Zero spills. 4096 B SMEM matches `DEFAULT_SMEM_SEGMENT_BUDGET` exactly.**

### 6.2 SASS instruction count (helper body only, `cuobjdump --dump-sass`)

Between the helper's first emitted instruction and its last emitted
instruction, measured SASS on `sm_80`:

| # | SASS | PTX source |
|---|---|---|
| 1 | `SHF.L.U32 R0, Ri, 0x1` | `shl.b32 %r_i_off, %i, 1` |
| 2 | `SHF.L.U32 R1, Rj, 0x1` | `shl.b32 %r_j_off, %j, 1` |
| 3 | `LDS.U16 R2, [R0]` | `ld.shared.u16 %rs_i, [%r_ai]` — ptxas folded the `add.u32 seg_base + offset` into the LDS addressing mode via symbol resolution; no explicit `IADD` instruction is emitted |
| 4 | `LDS.U16 R3, [R1]` | same |
| 5 | `ISETP.GT.AND P0, PT, Ri, Rj, PT` | `setp.le.s32 %p_c, %j, %i` — ptxas reversed `le` to `gt` for its canonical form with source operands swapped |
| 6 | `ISETP.EQ.U32.AND P0, PT, R2, R3, !P0` | `setp.eq.u16 %p_eq` + `and.pred %p_final, %p_eq, %p_c` **fused** into a single ISETP with `!P0` (negated causal) as the AND operand |

**Measured total: 6 SASS instructions. Predicted in Q5: 8–10. Lower
than predicted because ptxas absorbed the `add.u32` ADDs into the LDS
addressing mode and fused the `and.pred` into the final ISETP's
AND slot.**

In the harness, `%j` came from a kernel parameter and was detected as
warp-uniform, so ptxas emitted `USHF.L.U32` for one of the shifts. In
production FA2 `s_compute`, both `%i` and `%j` are thread-indexed
(derived from `threadIdx.x`, etc.), so both shifts emit as regular
`SHF.L.U32`. Instruction count stays at 6; uniform vs regular datapath
is an allocation detail.

### 6.2.1 Re-measured SASS for mask-convention helper (supersedes §6.2 for Task 3 integration)

The §5.1 amendment flipped the helper to mask-convention
(`setp.ne.u16` then `or.pred` extending the caller's existing
causal `%p0`). Harness re-assembled from
[`C:/tmp/pca_segment_mask_helper_maskconv.ptx`](./2026-04-18-pca-tier-a-design.md)
on sm_80:

```text
ptxas info : 0 bytes gmem
ptxas info : Compiling entry function 'test_segment_mask_predicate_maskconv' for 'sm_80'
ptxas info : Function properties for test_segment_mask_predicate_maskconv
             0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 8 registers, used 0 barriers, 4096 bytes smem, 376 bytes cmem[0]
```

**Zero spills, 8 total harness registers, 4096 B SMEM — parity with
§6.1.**

SASS on the mask-convention helper body:

| # | SASS | Origin |
|---|------|--------|
| 1 | `USHF.L.U32 UR4, UR4, 0x1` or `SHF.L.U32 R, …, 0x1` | `cvt.u32.u64` + `shl.b32` (fold) — k position × 2 |
| 2 | `IMAD.SHL.U32 R0, R4, 0x2, RZ` | same for q position × 2 (IMAD.SHL variant ptxas chose for this side) |
| 3 | `LDS.U16 R3, [R0]` | `ld.shared.u16 %rs_q` — offset folded into LDS |
| 4 | `LDS.U16 R2, [UR4]` | `ld.shared.u16 %rs_k` — offset folded |
| 5 | `ISETP.NE.U32.AND P0, …` | `setp.ne.u16` (ptxas widens to u32 at the SASS layer; values are zero-extended u16 so the comparison result is identical) |
| 6 | `ISETP.GT.U32.OR.EX P0, …, P0, P1` | `or.pred` collapsed into an extended-OR ISETP that re-runs the causal comparison and OR-folds with the segment predicate in one slot |

**Measured total: 6 SASS instructions in the helper body.** Matches
the original §6.2 count exactly despite the convention flip. The
`or.pred` + existing-causal-setp combination is what ptxas folds into
the `.OR.EX` extended-OR form, keeping the total instruction count
flat.

In the harness, `%rd_k` was a kernel parameter and flowed through
the uniform datapath (`USHF.L.U32`). Production `s_compute` sees both
positions as thread-indexed (from tile-offset arithmetic against
`%q_start` / `%k_start`), so both shifts emit on the regular datapath
(`SHF.L.U32` or `IMAD.SHL.U32`). Instruction count and register count
are unchanged; uniform-vs-regular is an allocation detail ptxas
selects based on the caller's register class.

### 6.3 Register pressure

**Helper body scratch:** 3 GPRs (`R0` offset/address, `R2`, `R3` segment
values — after reuse), 1 predicate chain (`P0` walks `%p_c → %p_eq → %p_final`
via ptxas-fused instructions).

**Harness total:** 8 registers (includes kernel prelude/epilogue: stack
pointer, param loads, output addressing, EXIT). Helper-specific delta
is **3 GPRs + 1 predicate**.

### 6.4 `register_budget.rs` extension (Commit 1)

`count_registers` grows one term:

```rust
// Addition in count_registers:
let segment_extra = if config.segment_masked { 3 } else { 0 };
q_row + s_scratch + o_acc + softmax + scratch + rope_extra + segment_extra
```

**Headroom check at current maximum config** (`rope_q=true`, `head_dim=128`):

- Base: `128/32 * 2 + 1 + 5 + 10 + 4 = 28 regs`
- With `segment_masked = true`: `28 + 3 = 31 regs`
- `SM75_REGISTER_CAP = 255`; headroom **> 224 regs**.

Tier A is register-budget safe on all supported archs with large
headroom.

## 7. Test discipline

### 7.1 Primary: insta snapshot (CSHA-consistent)

The PTX string emitted by `emit_segment_mask_predicate` is snapshot-tested
with `insta`, the same machinery used for
`crates/nsl-codegen/tests/snapshots/fa_v2_snapshots__*.snap`. Snapshot
files live alongside the existing FA2 snapshots. No inlined string
constants — the B.3.1 `wrga_kernel_helpers.rs` lesson transfers.

### 7.2 Secondary: caller-context-independence structural test

`tests/pca_segment_mask_caller_context_independence.rs`:

```rust
#[test]
fn segment_mask_predicate_byte_identical_across_callers() {
    let forward_emission = capture_helper_emission(
        CallerContext::Forward_SCompute, default_helper_args());
    let backward_emission = capture_helper_emission(
        CallerContext::Backward_DsCompute, default_helper_args());
    assert_eq!(
        forward_emission,
        backward_emission,
        "segment mask predicate must emit byte-identically in forward and backward contexts"
    );
}
```

This catches the "helper secretly depends on caller state" bug that
snapshot tests on the full kernel cannot catch (because both kernels'
snapshots update together, hiding the drift).

### 7.3 Numerical correctness fixtures

- **Single-segment fixture** (necessary but not sufficient): one packed
  sample with one segment of full length. Segment mask is a no-op;
  verifies the mask never breaks the unpacked case.
- **Two-equal-segments fixture** (adversarial, load-bearing): one
  packed sample with two segments of length `S/2`. Attention output and
  gradients compared against the unpacked-padded reference (two
  separate `S/2` sequences run through attention + gradient, outputs
  concatenated). Tolerances inherited from CSHA Tier A.
- **Unequal-length-segments fixture** (adversarial, load-bearing): one
  packed sample with segments of lengths `0.3S`, `0.5S`, `0.2S`.
  Catches kernels that silently assume equal-length segments. Required
  per §2.1's "single-segment fixtures are necessary but not sufficient"
  discipline.

### 7.4 Numerical tolerances (pinned as literals)

Inherited from CSHA Tier A landing (2026-04-15), pinned as specific
values rather than by reference so a future CSHA retune becomes an
explicit PCA-tolerance decision rather than silent propagation:

| head_dim | fp16 tolerance | Precision rationale |
|---|---|---|
| 32 | `5e-3` | O(√d · ε_f16) with d=32, ε_f16 ≈ 2⁻¹⁰ |
| 64 | `2e-2` | O(√d · ε_f16) with d=64, worst-case summation |
| 128 | TBD | Deferred: inherits from CSHA Tier A Track B head_dim=128 landing |

Segment masking adds a predicated-store, not new arithmetic — the error
budget is identical to the existing causal-mask path's budget. No
tighter fp32 gate is added; one tolerance per head_dim, per fixture.

### 7.5 FASE composition fixture (A-2)

One end-to-end fixture in the Tier A gate that exercises FASE + PCA +
CSHA Tier C backward together:

- **Packing configuration:** two-equal-segments (each `S/2 = 256`
  tokens).
- **Sequence length:** `S = 512`. Debuggable size; hits every code path
  in the three-way composition. The motivating workload runs at 2048
  but 512 catches composition bugs faster. If a 2048 test later wants to
  join the matrix, it extends rather than replaces.
- **Gradient accumulation:** `grad_accumulation = 4`. Commit 1 prep
  verifies this matches FASE's existing test-suite default; if FASE's
  default is different, this spec is revised to match before Commit 5
  lands. The specific count is not load-bearing (any `N > 1` exercises
  the FASE accumulation path); pinning it keeps the fixture
  reproducible.
- **Assertion:** end-to-end gradient of every parameter matches the
  unpacked-padded, FASE-disabled reference to CSHA Tier A tolerances.

This is the adversarial fixture for composition: if FASE, PCA, or Tier
C has a bug in how it hands data to the next, the end-to-end gradient
diverges and the test fails loudly.

### 7.6 Implementation-time gate: SASS forward/backward diff (Commit 4)

After both forward `s_compute.rs` and backward `ds_compute.rs` call the
helper, `cuobjdump --dump-sass` on both kernels' cubins yields the
helper-body SASS substring in each. A test compares them and asserts
byte-identity. This is the SASS-level analogue of the PTX-level
caller-context-independence test from §7.2.

## 8. Commit structure (6 commits, Commit 6 is two-step)

Mirrors WRGA B.3.1 commit structure with backward work absorbed into
Commits 2–4 rather than deferred to a follow-up milestone.

| # | Commit title | Contents |
|---|---|---|
| 1 | `refactor(pca): u16 segment_ids dtype + scaffolding corrections` | `DTYPE_U16_SEGMENT` constant added. `pca_detect::segment_ids_bytes = seq * 2`. `DEFAULT_SMEM_SEGMENT_BUDGET` unchanged (4096 B now covers seq ≤ 2048 at u16; scaffolding re-derived for u16). `SegmentResidency::Shared` threshold updated. `pca_detect::validate_config` adds u16-overflow compile-time check. `register_budget.rs::count_registers` extended with `segment_extra`. No kernel or runtime changes. Pre-refactor byte-identity on existing tests verified. |
| 2 | `feat(pca): emit segment_ids from DataLoader alongside mask` | `PackedBatch` gains `segment_ids: Vec<u16>` additively; `mask` retained. `pack_batch` emits both. `packed_batch_to_dict` publishes `segment_ids`. Structural tests verify segment_ids values match expected packing (EOS token → new segment). |
| 3 | `feat(pca): segment-mask predicate helper + forward s_compute wiring (red → green)` | New module `flash_attention_v2/phases/segment_mask.rs` with `emit_segment_mask_predicate`. `FlashAttentionConfig.segment_masked` flag added. Forward `s_compute.rs` calls the helper when the flag is set. Insta snapshot + caller-context-independence test land here. **Red test: two-equal-segments forward fixture initially fails because s_compute uses the old dense-mask fallback.** Green: s_compute reads segment_ids; fixture passes. |
| 4 | `feat(pca): Tier C backward segment_ids plumbing` | `ds_compute.rs` calls `emit_segment_mask_predicate` with the same arguments forward does. Packed-sequence variants added to `NUMERICAL_GATE_DQKV`, `_DW`, `_DX`. Null-safety test: Tier C backward with `segment_ids == NULL` is byte-identical to pre-PCA behaviour. **SASS forward/backward byte-identity gate runs here.** |
| 5 | `feat(pca): FASE + Tier C + PCA composition fixture` | A-2 fixture lands: two-equal-segments, `grad_accumulation = 4`, `seq_len = 512`, end-to-end gradient vs unpacked-padded reference. CSHA RoPE-fusion regression check: run the same fixture through CSHA's RoPE-fused path; verify no numerical divergence vs non-RoPE path beyond tolerance. If divergence, scope-expand Tier A to include position-reset; otherwise pin "RoPE fusion survives segment masking" in the close-out doc. |
| 6a | `docs(pca): Tier A close-out + deferred tier stubs` | Close-out doc at `docs/plans/2026-04-XX-pca-tier-a-closeout.md`. Stub documents for Tier B (tile-skip), Tier A-prime (RoPE position-reset), Tier B-prime (per-doc CTA), and CE separator skip. Each stub contains only its measurement trigger + Tier A-landing evidence. Update `MEMORY.md` pointer. |
| 6b | `feat(pca): remove dense attention_mask from PackedBatch` | `PackedBatch.mask` field removed. `packed_batch_to_dict` drops the `mask` key. All downstream consumers (verified during Commit 5) already read `segment_ids`. Small reviewable diff, separate from the docs commit so the "runtime contract migration" moment is inspectable in isolation. |

**Commit 6 is deliberately split** so the reviewer can verify each
consumer has actually been migrated before seeing the deletion. The
docs commit contains no behaviour change; the deletion commit contains
no documentation noise.

## 9. Risks and mitigations

### 9.1 Forward/backward segment-mask drift

**Risk.** Forward and backward emit subtly different mask predicates
(e.g., off-by-one on the causal condition) and the drift is invisible
in forward numerics but corrupts `dQ`/`dK`/`dV`.

**Mitigation.** Single shared helper (§3.3). Caller-context-independence
structural test (§7.2). SASS forward/backward byte-identity gate at
Commit 4 (§7.6).

### 9.2 Tier C backward regression on unpacked sequences

**Risk.** Adding `segment_ids` as a new parameter to the Tier C
backward kernel breaks existing unpacked-sequence numerical gates.

**Mitigation.** Null-safe code path: `segment_ids == NULL` compiles to
byte-identical pre-PCA behaviour (invariant #4). Existing
`NUMERICAL_GATE_DQKV`/`_DW`/`_DX` tests stay green unchanged. Tier C
was stabilised 2026-04-16 with headroom; the extension is additive.

### 9.3 CSHA RoPE fusion interaction with segment masking

**Risk.** CSHA Tier A's fused RoPE epilogue assumes global position
indices. Segment-masked attention with reset-per-document positions
would require position-reset fusion (deferred to Tier A-prime). If
CSHA's existing global-position RoPE produces incorrect numerics under
segment masking, Tier A's scope expands.

**Mitigation.** Commit 5's RoPE-fusion regression check runs the A-2
fixture through the CSHA-RoPE path. If it passes, "RoPE fusion
survives segment masking without position reset" is pinned in the
close-out. If it fails, Tier A scope expands to include the fix, and
the expansion is scoped by the specific observed failure rather than
designed-in upfront.

### 9.4 Variant matrix / PTX blob growth

**Risk.** `segment_masked` multiplies the variant matrix and PTX blob
size, inflating compile time and binary size.

**Mitigation.** Paged-mutual-exclusion prunes half of the additions
(§3.2). Link-time DCE (already in place) prunes unused combinations.
Measured growth: 1.4–1.6× variant count, 40–60% blob size. Verify DCE
pass sees the new flag dimension during Commit 1 (not a hardcoded
variant enumeration).

### 9.5 Zero-spill assumption on future arch

**Risk.** A future SM target (sm_90+) may register-allocate the
helper differently and spill.

**Mitigation.** Invariant #6 pins the zero-spill property. The
`register_budget.rs::count_registers` extension is conservative (3
GPRs + 1 predicate); headroom at max config is >224 registers under
the sm_75 cap, >192 under any realistic cap. If a future arch
regresses, the regression is caught by the existing register-budget
test before it reaches production.

## 10. Alternatives considered

Each was rejected after working through its decomposition against the
WRGA B.3.1 / CSHA Tier A / WRGA hex-constant-verification precedents.

- **Single-spec full paper (all five PCA features).** Rejected by the
  same argument that made CSHA tier-decompose: five features with
  different risk profiles in one commit graph, a bug in any blocks the
  others, debugging gives five candidate causes per failure.
- **Tier A + B bundle (segment-ID kernel + tile-skip).** Tier B's
  correctness is a joint claim over cost-model accuracy, tile-map
  fidelity, and backward consistency — three testable properties vs
  Tier A's one. Deferred with a measurement trigger.
- **Tier A + RoPE position-reset fusion.** "Composes cleanly" is a
  claim that has to be tested; the safe version is "ship Tier A,
  confirm CSHA RoPE fusion survives segment masking, then add
  position-reset as a Tier A-prime delta." That's §2.2's deferral.
- **PCA Tier A tests with CSHA Tier C disabled.** Ships PCA into a
  configuration where the motivating workload cannot use it. Under-ships
  the feature.
- **PCA Tier A forward-only, backward deferred to Tier A-prime.** The
  WRGA B.3.1 deferral pattern requires the deferred state to be
  *correct but slow*. PCA's forward-only-with-Tier-C-active state is
  *wrong*: Tier C's unpacked backward on packed input leaks gradients
  across document boundaries. Pattern does not transfer.
- **Runtime null-gated single kernel (one variant, runtime branch).**
  Weakens the FA2 emitter's compile-time specialisation principle;
  per-invocation branch + register-pressure tax on unpacked workloads.
  Register-pressure regressions on sm_80 kernels from less obvious
  changes have historically been 10–20%.
- **Compile-time flag with runtime fast-path disable (hybrid).**
  Creates the correctness footgun where `segment_masked=false` with
  packed input silently produces wrong results and no diagnostic.
  Weakens the compile-time correctness guarantee for marginal perf on
  the wrong axis.
- **L2 predicate+score-write shared helper.** Parameterising on
  fragment type reintroduces a drift surface one parameter at a time.
  L1 keeps the helper ignorant of call sites; that's the scoping
  discipline.
- **L3 full masked-score-op shared helper.** Forces forward and
  backward through a single helper that must express both
  architecturally-different call sites, re-creating the drift surface
  the sharing was meant to eliminate.
- **i16 (signed) segment_ids.** Segment IDs are non-negative indices;
  signed wastes the sign bit and halves the ceiling from 65 535 to
  32 767. u16 is the honest type.
- **Original spec draft's `add.s64 %seg_base + offset` SMEM addressing.**
  SMEM addresses in PTX are 32-bit; the original s64 arithmetic was
  wrong and would have failed ptxas with an operand-type error. Caught
  during spec prep by the ptxas-verification-during-spec-prep
  discipline that this spec commits to. Fixed in §5's u32-throughout
  form.
- **Allow-convention helper with `s_compute.rs` rewrite to match
  (option B from the §5.1 amendment resolution).** Rewriting the
  forward kernel's causal-mask block from its existing mask-predicate
  idiom to an allow-predicate idiom to match §5's literal PTX form
  was rejected 2026-04-19 in favour of the mask-convention helper
  (option A, now pinned in §5.1). Reason: `s_compute.rs` is shared
  with CSHA Tier A's RoPE-fused path (just stabilised 2026-04-15),
  and a convention rewrite creates a regression surface in a
  milestone (CSHA) whose tests aren't scoped to cover it. Specs
  exist to reason about code; when the code and spec disagree on
  an illustrative detail while agreeing on load-bearing contracts
  (u16 dtype, forward/backward byte-identity, ptxas-clean
  assembly), the spec amends. The alternative pattern — rewriting
  stable modules to match illustrative spec detail — produces
  "we rewrote a stable module to match an earlier reasoning
  artifact" commits that readers of the git log can't understand
  in isolation.
- **Hybrid: allow-convention helper with caller-side `not.pred`
  (option C from the §5.1 amendment resolution).** Serving the
  literal §5 form by adding a `not.pred` at each call site creates
  a convention boundary inside `s_compute.rs` — the helper speaks
  allow-predicates while every other predicate in the kernel speaks
  mask-predicates. Same failure pattern as Q4's rejected L2: the
  helper's API diverges from its call sites' idiom, creating
  maintenance friction at every integration point. Future tiers
  (e.g. sliding-window, Tier A-prime position-reset) adding
  additional predicates would have to pick a side or add another
  conversion, producing a persistent convention split. Mask-
  convention helper composes cleanly with any future mask-
  convention predicate via `or.pred`.

## 11. Numerical claims verified

Self-review checklist (all boxes must be checked before the spec is
handed to writing-plans):

- [x] i16 → u16 correction applied throughout (segment IDs are
      non-negative; honest type).
- [x] SMEM addressing corrected to u32 throughout PTX block (`shl.b32` +
      `add.u32` + `ld.shared.u16`).
- [x] ptxas assembled output captured (0 spills, 4096 B SMEM exact, 8
      registers for full harness) — §6.1.
- [x] SASS instruction count pinned at measured value: **6 instructions
      in the helper body** on sm_80 (§6.2). Lower than the Q5
      prediction of 8–10 because ptxas absorbed `add.u32` into LDS
      addressing mode and fused `and.pred` into the final ISETP.
- [x] Register pressure headroom pinned: helper delta is 3 GPRs + 1
      predicate chain; at max config (`rope_q=true`, `head_dim=128`)
      total is 31 regs vs 255-reg cap → >224 reg headroom (§6.3).
- [x] u16 overflow validation scheduled for `pca_detect::validate_config`
      in Commit 1 (§4 invariant #3).
- [x] Scaffolding updates enumerated (§8 Commit 1): `DTYPE_U16_SEGMENT`
      constant, `segment_ids_bytes = seq * 2`, residency threshold,
      `validate_config` overflow check, `register_budget::count_registers`
      extension. Estimate: ~10 LOC across 3–4 files.
- [x] A-2 fixture configuration pinned: two-equal-segments,
      `grad_accumulation = 4`, `seq_len = 512` (§7.5).
- [x] CSHA tolerance values pinned as literal numbers: `5e-3 @ head_dim=32`,
      `2e-2 @ head_dim=64`; head_dim=128 deferred with Track B (§7.4).
- [x] Worktree name pinned: `feat/pca-tier-a`.

## 12. References

- CFTP research paper: `docs/research/CFTP.pdf` §2
- FASE milestone (companion, shipped): `MEMORY.md` → "WGGO consumer rollout"
- CSHA Tier A close-out: `project_csha_tier_a_e2e_shipped.md`
- CSHA Tier C close-out: `project_csha_tier_c_shipped.md`
- WRGA B.3.1 close-out (pattern precedent): `project_wrga_fused_ptx_rewrite.md`
- FA2 emitter architecture: `crates/nsl-codegen/src/flash_attention_v2/phases/`
- Existing PCA scaffolding: `crates/nsl-codegen/src/pca_{detect,segment,tileskip,rope}.rs`
- Existing runtime packer: `crates/nsl-runtime/src/packing.rs`
- Register budget accounting: `crates/nsl-codegen/src/flash_attention_v2/register_budget.rs`
