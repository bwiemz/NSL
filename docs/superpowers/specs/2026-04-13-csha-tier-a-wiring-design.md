# CSHA Tier A — Runtime Wiring + Level 1 PTX Emission Design

**Date:** 2026-04-13
**Status:** Design spec for tasks A.1, A.2, A.4 of CSHA Tier A (see
`memory/project_csha_tiers.md`). A.3 (patterns restoration) landed as
commit `dd92dc7`.
**Target branch:** `feat/csha` worktree.

---

## Problem

The CSHA compiler passes (`csha::run`) compute a `CshaPlan` with boundary
chains, pipeline feasibility, per-head specialization, patterns, and a
`csha_apply::bridge` output containing per-layer `KernelSpec`s and
`FusionMark`s. **None of this reaches `cuLaunchKernel` today.** When
`@csha` is set, `stmt.rs:3630` computes the plan, prints a summary to
stderr, and discards it — the `CshaPlan.kernels` / `CshaPlan.marks` /
`CshaPlan.patterns` fields are dead code from codegen's point of view.

Result: with or without `@csha`, the emitted PTX and the `cuLaunchKernel`
arg list are identical. No HBM traffic is saved. The CSHA feature is
observable only via the `--csha-report` text.

This spec covers the minimum wiring to make **Level 1 (boundary fusion)**
real end-to-end — i.e., emit a single fused kernel for
`RMSNorm → matmul(Wq|Wk|Wv) → RoPE` and launch it with the right args.
Level 2 (pipelining) and Level 3 (block fusion) are out of scope (Tier B).

---

## Tasks in scope

| ID | Description | Est. LOC |
|----|-------------|----------|
| A.1 | Runtime dispatch wiring — plumb the six CSHA extras (`x`, `norm_weight`, `Wq/Wk/Wv/Wo`, `rmsnorm_eps`, `active_heads`, `d_model`) through `nsl_kernel_launch` | ~400 |
| A.2 | Level 1 PTX emission — see sub-task split below | ~1330 |
| A.4 | Weight-informed emission — kernel body respects `active_heads` mask (skip pruned heads in per-head loop bounds, don't just encode it in the kernel name) | ~400 |

A.5 (end-to-end HBM-reduction proof) is downstream of all three and is
not in this spec; it goes in its own plan after A.1-A.4 land.

### A.2 sub-task breakdown (milestone split added 2026-04-13)

After hands-on exploration of the 5639-LOC `crates/nsl-codegen/src/flash_attention.rs` PTX emitter and the Cranelift-level FA call site, A.2 as originally written is too coarse for a single session. Each sub-step touches deep, load-bearing PTX machinery (MMA scheduling, SMEM swizzle, online softmax) and is more responsibly landed in its own commit with design review between them.

| ID | Scope | Est. LOC | Session cost |
|----|-------|----------|--------------|
| A.2.0 | Proper layer identification — replace `BridgeResult::first_extras()` stopgap with per-FA-call layer lookup. Use Wengert `Param` prefix or enclosing function/model name to select the right `CshaExtras` for each SDPA call. Add multi-layer test. | ~80 | ≤½ |
| A.2.1 | Thread real tensor DataIds (x, norm_weight, Wq/Wk/Wv/Wo) into the FA FFI call. Consume the bridge `marks` to identify upstream matmul/RMSNorm/RoPE nodes and extract their weight/input Cranelift `Value`s. Suppress the separate lowering of those claimed nodes. | ~350 | 1 |
| A.2.2 | Emit **RMSNorm prologue** PTX: load x tile from HBM → compute `rsqrt(sum(x²)/D + ε) * w` → store normalized x in SMEM in place of the pre-projected Q. Gated on `config.csha.fused_rmsnorm`. | ~250 | 1 |
| A.2.3 | Emit **matmul projection** PTX: `x_norm` × `Wq/Wk/Wv` tiles → Q/K/V in SMEM via m16n8k16 MMA. Reuses the existing MMA primitives in `matmul_mma.rs`. Gated on `config.csha.fused_projections`. | ~300 | 1 |
| A.2.4 | Emit **RoPE epilogue** PTX: apply rotation to Q/K in registers before they enter QK^T. Re-uses the existing `rope` cache-write helper but inlines it into the kernel body instead of a separate launch. | ~150 | ½ |
| A.2.5 | Light up `nsl_flash_attention_csha` runtime body — dispatch to the CSHA PTX variant based on kernel name suffix. Add snapshot tests for the new `_cshaL1_…` variants (byte-identical non-CSHA path still required). | ~200 | ½ |

Dependency chain: **A.2.0 → A.2.1 → (A.2.2, A.2.3, A.2.4 in any order) → A.2.5**. A.2.0 is a pure-plumbing prerequisite that makes A.1 correct for multi-layer models and unblocks every subsequent sub-task.

---

## Architecture (three coordinated cuts)

### Cut 1 — codegen consumes `bridge()` output (A.1 + A.4 prerequisite)

**Where:** new field on `CompileContext` (or equivalent per-function
compile-state struct) carrying a `HashMap<String, CshaExtras>` keyed by
layer name.

**When:** populated in `stmt.rs:3630` right after `csha::run_on_wengert`
returns a non-`Off` plan. The map is built by walking
`plan.kernels: Vec<KernelSpec>` (each entry already carries the CSHA
level, SMEM, active_heads, fused flags) — just convert to the
`CshaExtras` shape that `FlashAttentionConfig` already expects.

**Who consumes it:** `compile_flash_attention_call` in
`crates/nsl-codegen/src/expr/calls.rs:1112` — before building the
`FlashAttentionConfig`, look up the current layer name in the side-table
and, if present, set `config.csha = Some(extras)`. Layer name is derived
from the same metadata the planner used (matmul's `Param(...)` input
prefix, e.g. `blocks.0.attn.wq` → `blocks.0`).

**Marks:** `csha_apply::apply_marks_to_graph` is already correct but
never called on the real fusion graph used by `epilogue_fusion` /
`reduction_fusion`. Call it from the same stmt.rs hook before those
passes run, so CSHA-claimed matmul nodes aren't re-fused elsewhere.

### Cut 2 — `compile_kernel_call` appends CSHA args (A.1)

**Where:** `expr/calls.rs:2314–2424`. Today the arg array is built as
`[output_ptr, input1_ptr, input2_ptr, ...]` for the tensor inputs only.

**Change:** if the kernel being launched is a CSHA-tagged FA kernel
(recognized either by `is_csha_fused(kid)` on the kernel-id or by a new
`kernel_has_csha_extras: bool` flag stored alongside the
`kernel_bytes`), append these pointers/scalars in order:

```
  x_ptr, norm_weight_ptr, Wq_ptr, Wk_ptr, Wv_ptr, Wo_ptr,
  rmsnorm_eps (f32 as i32 bits), active_heads (u32), d_model (u32)
```

All six tensor pointers are already in scope at the FA call site —
they are the same `DataId`s used by the upstream matmul launches that
CSHA's `FusionMark` has claimed. The emitter must *not* lower those
marked matmul/RMSNorm/RoPE ops as separate launches — `apply_marks`
already tags them, but today the lowering doesn't skip on the tag.
Add a single check in the lowering loop: `if is_csha_fused(op.kid) {
continue; }`.

**SMEM:** `shared_mem_bytes` at `expr/calls.rs:2402` is hardcoded to 0.
Replace with `crate::flash_attention::shared_mem_bytes(&config)`, which
already includes the CSHA prologue/projection/epilogue contribution.

### Cut 3 — PTX body matches the new param list (A.2)

**Where:** `crates/nsl-codegen/src/flash_attention.rs` —
`emit_flash_attention_entry` already declares the six CSHA params
(commit `dda59d9`). What's missing is the *body* that uses them.

**Body sketch (Level 1 only):**

```
# Prologue — RMSNorm of x loaded into Q tile
ld.global.v4.f16    %x_tile,     [%x_ptr + tile_off];
ld.global.v4.f16    %w_tile,     [%norm_weight_ptr + ...];
<compute rsqrt(sum(x²)/D + eps) once per row>
<x * norm_weight * rsqrt_val → x_hat in registers>

# Main — existing FA kernel body, except Q/K/V come from x_hat @ W*
# (fused matmul) instead of being loaded from pre-projected HBM tensors.
# For Level 1 this is ONE tile per MMA step; pipelining across tiles is
# Level 2 (Tier B).
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 %Q_tile, %x_hat, %Wq_tile, %Q_acc;
...

# Epilogue — RoPE on Q/K in registers before they enter the FA
# QK^T step. RoPE cos/sin from compile-time LUT baked into .rodata.
```

Factor common pieces into helpers:
- `emit_rmsnorm_prologue(w, x_ptr, norm_weight_ptr, eps, d_model) -> PtxFragment`
- `emit_matmul_projection(w, x_hat_reg, W_ptr, out_reg, head_dim) -> PtxFragment`
- `emit_rope_epilogue(w, qk_reg, cos_lut, sin_lut, dim) -> PtxFragment`

The existing `flash_attention_kernel_name` already disambiguates these
variants via the `_cshaL1_n{rmsnorm}_p{proj}_o{output}_h{active_heads}`
suffix, so the PTX cache is safe.

### Cut 4 — honor `active_heads` (A.4)

**Where:** the per-head loop in the FA body. Today it iterates
`0..num_heads`. Change to iterate `0..active_heads` with a compile-time
check that the kernel's `active_heads` param matches what the caller
passed (an `assert` is cheap; a skip is cheaper).

**Side effect:** `grid.x` must be computed as `active_heads` at launch,
not `num_heads`. Update in `compile_kernel_call`.

---

## Test plan

**Unit (cargo test):**
- `csha_apply::bridge` produces a `CshaExtras` whose `active_heads`
  matches `LayerSpec.n_active_heads` — add if missing.
- `flash_attention::shared_mem_bytes` returns non-zero for a CSHA-enabled
  config — existing `snapshot_flash_attention_*` tests run with
  `csha: None`, add a `snapshot_flash_attention_cshaL1` variant.
- `compile_kernel_call`: fake a minimal Wengert list + CSHA plan, drive
  it through codegen, and assert that (a) a single CSHA kernel launch
  is emitted (not 3 matmul launches + 1 FA launch), (b) the arg list
  contains the six extras in the documented order.

**Integration:**
- End-to-end: `nsl build examples/tiny_transformer.nsl --csha=auto
  --csha-report`. Assert the report now lists a `Kernel:` line with
  `_cshaL1_` prefix for each layer, and that a PTX trace (parse the
  generated `.ptx` file) shows one fused kernel per layer, not three
  separate matmul kernels followed by FA.

**Regression:** all existing 64 CSHA tests + 12 FA snapshot tests must
continue to pass. Expect some `.snap` updates for the FA snapshots that
default to `csha: None` because the param-list order may shift.

---

## Risk and fallback

- **Biggest risk:** the arg-marshalling change is cross-cutting (codegen
  + runtime + PTX). If any of the six pointers is the wrong type or wrong
  dtype, the kernel silently corrupts memory. Mitigation: TDD the three
  cuts independently; never skip the arg-count assertion at the
  `nsl_kernel_launch` boundary.
- **Fallback:** all CSHA features live behind `CompileOptions.csha_mode`.
  If Tier A lands and something breaks in the wild, `--csha=off` must
  produce byte-identical PTX and byte-identical arg lists to the
  pre-CSHA build. Add a snapshot test that asserts this.

---

## Out of scope (Tier B/C/D follow-ups)

- Level 2 producer-consumer pipelined kernel body (~2000 LOC, paper §9.2)
- Level 3 full-block fusion (attention + FFN in one persistent CTA)
- Source-AD fused backward (dO → dQ/dK/dV/dWq/dWk/dWv in one kernel)
- Per-head mixed FP8/BF16 precision (`csha_precision.rs` restoration)

All four will reference this spec's cuts but introduce their own PTX
emission paths.
