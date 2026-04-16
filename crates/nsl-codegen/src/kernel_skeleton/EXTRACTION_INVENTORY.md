# FA v2 Prolog Variation Points (snapshot of 2026-04-16)

## .version

- `8.7` — emitted by both `phases/forward/prelude.rs:56` and `phases/backward/prelude.rs:56`
- No other version value appears anywhere in `flash_attention_v2/`

## .target

- `sm_75` — emitted by both `phases/forward/prelude.rs:28` and `phases/backward/prelude.rs:57`
- No other sm target appears anywhere in `flash_attention_v2/`

## .shared decl variants

Two mutually exclusive paths — controlled by `needs_dynamic_smem(config)` / `backward_needs_dynamic_smem(config)`:

- **Static** `.shared .align 16 .b8 shmem[N]` — emitted **inside** the function body (`{ ... }`),
  when `total_bytes(config) <= 48 * 1024`.
  - Forward: `phases/forward/prelude.rs:80-83`
  - Backward: `phases/backward/prelude.rs:103-107`

- **Dynamic** `.extern .shared .align 16 .b8 shmem[]` — emitted at **MODULE scope**,
  *before* the `.visible .entry` directive, when `total_bytes(config) > 48 * 1024`.
  - Forward: `phases/forward/prelude.rs:37`
  - Backward: `phases/backward/prelude.rs:62`
  - Constraint: `.extern .shared` cannot appear inside a PTX function body; it must precede `.visible .entry`.

`total_bytes` values seen across `fa_v2_snapshots.rs` test configs (no-csha, `wq_tile_bytes == 0`):
- `csha_canonical()` (block_q=32, block_kv=32, head_dim=32): **4 608 bytes** → static
  - Q=32×32×2=2048, KV=32×32×2=2048, SP=4×32×4=512
- `non_csha_canonical()` (block_q=64, block_kv=64, head_dim=128): **33 792 bytes** → static
  - Q=64×128×2=16384, KV=64×128×2=16384, SP=4×64×4=1024
- `csha_l2_rope_config()` (block_q=32, block_kv=32, head_dim=32, d_model=32, level2): **10 752 bytes** → static
  - Base 4608 + Wq+Wk+Wv=3×2×32×32=6144

All three snapshot configs fall under the 48 KB static cap; no snapshot currently exercises
the dynamic path. The dynamic path is exercised at runtime by large configs (e.g. block_q=128,
head_dim=256) that exceed 48 KB.

## .param types used

All `.param` declarations are at entry-function scope (inside the `(…)` param block).

| Type | Count (forward) | Count (backward) | Representative params |
|------|-----------------|------------------|-----------------------|
| `.param .u64` | 32 | 40 | `q_ptr`, `k_ptr`, `v_ptr`, `out_ptr`, `batch`, `heads`, `seq_len`, `head_dim`, all CSHA weight ptrs, projection save ptrs, `dO_ptr`, `dq_ptr`, `dk_ptr`, `dv_ptr`, `dwq_ptr`, `dwk_ptr`, `dwv_ptr`, `dx_ptr` |
| `.param .f32` | 2 | 2 | `scale`, `csha_eps` |
| `.param .u32` | 2 | 2 | `csha_active_heads`, `csha_d_model` |

Register naming in `ld.param` loads follows two conventions:
- Numbered pool: `%rd0`–`%rd7` for the 8 core tensor pointers (q/k/v/out/batch/heads/seq_len/head_dim)
- Named: `%scale`, `%logsumexp_base`, `%rd_bwd_*`, `%rd_c*`, `%f_xn`, `%x_norm_base`, etc.

`.u32` params (`csha_active_heads`) are loaded with `ld.param.u32 %r10, [csha_active_heads]`
(forward `csha_hooks.rs:22`) — the destination register is in the `.reg .u32 %r<16>` pool.

## tid/lane/warp pattern

Verbatim 5-line block, emitted identically by both forward (`prelude.rs:204-208`) and
backward (`prelude.rs:224-228`):

```ptx
    mov.u32 %tid_x, %tid.x;
    shr.u32 %warp_id, %tid_x, 5;       // warp_id = tid_x / 32
    and.b32 %lane, %tid_x, 31;          // lane = tid_x % 32
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %bid_y, %ctaid.y;
```

Note: the forward prelude includes the inline comments (`// warp_id = …`, `// lane = …`);
the backward prelude omits them. The instruction sequence is identical in both.

The registers are declared earlier in the function body by:
```ptx
    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;
```
(`forward/prelude.rs:94`, inside the function body before the tid block).

## FA v2 snapshot test configs (from fa_v2_snapshots.rs)

Two base configs drive all phase tests:

| Test group | Config helper | block_q | block_kv | head_dim | causal | csha | rope |
|------------|--------------|---------|----------|----------|--------|------|------|
| `*__32x32x32_*` | `csha_canonical()` | 32 | 32 | 32 | true | None | false |
| `*__64x64x128_*` | `non_csha_canonical()` | 64 | 64 | 128 | true | None | false |
| `*__csha_l2_rope*` | `csha_l2_rope_config()` | 32 | 32 | 32 | true | Level2(eps=1e-5, d_model=32) | Adjacent |

Individual snapshot tests:
- `phase_prelude__32x32x32_snapshot`: csha_canonical forward prelude
- `phase_prelude__64x64x128_snapshot`: non_csha_canonical forward prelude
- `phase_q_load__32x32x32_snapshot`: Q load, iter 0, csha_canonical
- `phase_q_load__64x64x128_snapshot`: Q load, iter 0, non_csha_canonical
- `phase_s_compute__32x32x32_causal_snapshot`: S compute, iter 0, causal, csha_canonical
- `phase_s_compute__64x64x128_causal_snapshot`: S compute, iter 0, causal, non_csha_canonical
- `phase_s_compute__label_uniqueness_across_iters`: label dedup regression, csha_canonical
- `phase_softmax__32x32x32_snapshot`: softmax, csha_canonical
- `phase_softmax__64x64x128_snapshot`: softmax, non_csha_canonical
- `phase_pv_accum__32x32x32_snapshot`: PV accumulation, iter 0, csha_canonical
- `phase_pv_accum__64x64x128_snapshot`: PV accumulation, iter 0, non_csha_canonical
- `phase_finalize__32x32x32_snapshot`: finalize, iter 0, csha_canonical
- `phase_finalize__64x64x128_snapshot`: finalize, iter 0, non_csha_canonical
- `phase_pv_accum__label_uniqueness_across_iters`: label dedup regression, csha_canonical
- `phase_csha_hooks__prologue_null_config`: CSHA prologue, csha=None
- `phase_csha_hooks__prologue_active_l2_rope`: CSHA prologue, csha_l2_rope_config
- `phase_csha_hooks__projection_skeleton_l2_rope`: CSHA matmul projection, csha_l2_rope_config
- `phase_csha_hooks__epilogue_l2_rope`: CSHA rope epilogue, csha_l2_rope_config
- `phase_csha_hooks__active_heads_guard`: active_heads guard, csha_l2_rope_config
- `phase_csha_hooks__label_uniqueness_across_iters`: label dedup regression across iters 0+1
- `kernel_full__32x32x32_nocsha`: full kernel, csha_canonical
- `kernel_full__64x64x128_nocsha`: full kernel, non_csha_canonical
- `kernel_full__32x32x32_csha_l2_rope`: full kernel, csha_l2_rope_config

## Skeleton API coverage (proposed)

```
emit_ptx_header(PtxVersion::{V7_0, V8_7}, TargetSm::{Sm75, Sm80})  ← covers both FA and WRGA
emit_static_smem_decl(bytes: usize)                                ← covers all FA static configs
emit_dynamic_smem_extern()                                         ← covers FA dynamic-SMEM path
emit_shmem_base_cvta()                                             ← fixed content; no params
emit_thread_lane_warp_registers()                                  ← fixed content; no params
emit_param_block(entry_name, &[Param])                             ← covers arbitrary param lists
emit_ld_param_u64(dest_reg, param_name)                            ← line-level helper
emit_ld_param_f32(dest_reg, param_name)                            ← line-level helper
emit_ld_param_u32(dest_reg, param_name)                            ← line-level helper (csha_active_heads, csha_d_model)
emit_smem_zero_pad_predicated(smem_base_reg, real, padded, dtype_bits)  ← NEW for WRGA; FA can adopt later
```

## API coverage cross-check

Every variation observed in steps 1-6 is covered by the proposed API:

| Observed variation | Covered by |
|-------------------|------------|
| `.version 8.7` (only value seen) | `emit_ptx_header(PtxVersion::V8_7, …)` |
| `.target sm_75` (only value seen) | `emit_ptx_header(…, TargetSm::Sm75)` |
| Static `.shared .align 16 .b8 shmem[N]` inside function body | `emit_static_smem_decl(bytes)` |
| Dynamic `.extern .shared .align 16 .b8 shmem[]` at module scope | `emit_dynamic_smem_extern()` |
| `.param .u64` pointer loads | `emit_ld_param_u64(dest, name)` |
| `.param .f32` scalar loads | `emit_ld_param_f32(dest, name)` |
| `.param .u32` integer loads | `emit_ld_param_u32(dest, name)` |
| `cvta.shared.u64 %shmem_base, shmem` | `emit_shmem_base_cvta()` |
| Arbitrary-length param block (32–44 params) | `emit_param_block(entry_name, &[Param])` |
| 5-line tid/warp/lane/bid_x/bid_y sequence | `emit_thread_lane_warp_registers()` |
| `.reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y` declaration | `emit_thread_lane_warp_registers()` |
| WRGA smem zero-pad | `emit_smem_zero_pad_predicated(…)` |

**Gap resolved:** `.param .u32` loads (`ld.param.u32 %r10, [csha_active_heads]` and
`ld.param.u32 %r11, [csha_d_model]`) are emitted in both forward (`csha_hooks.rs:22`) and
backward preludes. `emit_ld_param_u32(dest_reg, param_name)` is now included in the proposed
API above alongside `emit_ld_param_u64` and `emit_ld_param_f32`, completing full coverage of
FA's emission patterns. Implement alongside the other line-level helpers in Task A2/A3.
