# FlashAttention-2 Scalar-Path Emitter Rewrite — Design Spec

**Date:** 2026-04-14
**Branch:** `feat/csha-fa-scalar-rewrite` (target merge into `feat/csha`)
**Scope:** Scalar forward path only. MMA forward, backward pass, and quantized variant are out of scope.

## Motivation

The Part 1 numerical integration test landed at commit `034b73a` surfaced that the current scalar FA-2 emitter (`crates/nsl-codegen/src/flash_attention.rs`, function `emit_flash_attention_entry`, ~2000 LOC) is structurally incorrect. Five interdependent phases carry inconsistent thread-to-element mappings:

1. **Phase 2 S = Q·K^T** (`flash_attention.rs:1651-1764`) — thread `t` handles linear indices `t, t+128, t+256, …`, deriving `q_row = linear / block_kv`. For any realistic `(block_q, block_kv)` this spans multiple query rows per thread.
2. **Phase 3 softmax** (`flash_attention.rs:1782-1846`) — `%row_max` / `%row_sum` reduced warp-wide via `shfl.sync.bfly`. Produces one max/sum per warp, not per row. Attention softmax is per-row; this silently mixes rows.
3. **Phase 5 O_acc += P·V** (`flash_attention.rs:1905-1955`) — P indexed as `%f(3+k_iter)` with no `q_row` component; assumes each thread handles exactly one query row. Contradicts Phase 2's mapping.
4. **emit_output_store** (`flash_attention.rs:2026-2071`) — fixed stride 128 regardless of `head_dim`; broadcasts one accumulator across 4–8 rows for `head_dim ≠ 128`.
5. **Register allocation** — single `%row_max` / `%row_sum` / `%correction` per thread cannot track multiple rows; fixes to Phase 2/3 would exceed per-thread state.

No scalar-path config tested produces numerically correct output. Each defect in isolation cannot be patched without breaking the other phases' assumptions. This spec defines a parallel v2 implementation with a unified thread-mapping contract that all phases obey.

**Not motivation:** performance. Correctness is the sole driver; perf is a separate concern for a follow-up task.

## Section 1 — Kernel architecture (warp-per-row contract)

**Block shape:** 128 threads = 4 warps per block. `warp_id = tid/32`, `lane = tid%32`.

**Outer q_tile_iter loop:** iterates `ceil(block_q / 4)` times. Inside one iteration:
- `q_row_global = q_start + q_tile_iter * 4 + warp_id`
- Each warp owns one query row for the duration of this iteration.
- `row_max`, `row_sum`, `correction`, `O_acc[0..head_dim/32]` are warp-local per-lane registers — lanes within the warp jointly represent the row's state.
- State resets at the top of each iteration (row_max = -inf, row_sum = 0, O_acc = 0).

**Hard constraint:** `block_q % 4 == 0`. Supported `block_q ∈ {4, 8, 16, 32, 64, 128}`. All four warps are always active — no tail iteration with inactive warps (which would hang a `shfl.sync` with `0xFFFFFFFF` membermask).

**Lane partitioning within a warp:**

- **Phase 1 (Q load):** lane L loads `Q[q_row, d=L]`, `Q[q_row, d=L+32]`, … into `head_dim/32` registers per lane. The Q row is distributed over lanes along the d dimension.

- **Phase 2 (S = Q·K^T):** the dot product spans all d, distributed across 32 lanes. Sequential over k, warp-wide reduction per k:
  ```
  for k in 0..block_kv:
      // K[k, :] distributed over lanes identically to Q
      partial = Σ_{i} Q[q_row, L+32i] * K[k, L+32i]     // lane-local
      S_full  = shfl.sync add-butterfly over 32 lanes    // every lane holds full S[q_row, k]
      if lane == 0: shmem_S[warp_id, k] = S_full         // one lane writes to warp's row in shmem
  ```
  Full row of S (block_kv values) lives in shmem, not registers. One accumulator register per lane holds the current k's partial.

- **Phase 3 (online softmax):** reads the S row from shmem. Full-warp butterfly for `row_max` over 32 S-chunks — all lanes hold values from the SAME row's S (broadcast-restored via shfl or re-loaded from shmem), so reduction is semantically correct per row. Same for `row_sum`. **P writeback in-place:** `P[q_row, k] = exp(S[q_row, k] - new_max)` is written back to the same `shmem_S[warp_id, :]` region, overwriting the raw S values. Phase 5 reads P from there.

- **Phase 5 (O_acc += P·V):** each k iteration, `P[q_row, k]` is loaded once from shmem (shared across all 32 lanes of the warp — scalar broadcast). `V[k, d]` is distributed across lanes like K. `O_acc[L+32i] += P * V[k, L+32i]` — lane-local, no reduction.

- **Phase 6 (output store + LSE):** after `q_tile_iter` finalizes `O_acc / row_sum`, lane L writes `out[q_row_global, d=L+32i]` for `i ∈ 0..head_dim/32`. Coalesced f16 stores. LSE is written by lane 0 of each warp for `logsumexp[batch, head, q_row_global]`.

**Register budget per thread (canonical block_q=64, block_kv=64, head_dim=128):**

| Purpose | Registers |
|---|---|
| Q row slice | `head_dim/32` = 4 |
| S dot-product accumulator | 1 (per-k scratch) |
| O_acc | `head_dim/32` = 4 |
| Softmax state (row_max, row_sum, correction, old_max, new_max) | 5 |
| Loop counters, shfl tmp, addressing scratch | ~10 |
| **Total** | **~24** |

Comfortably under sm_75's 255-register cap.

## Section 2 — Config space and SMEM layout

### Supported config matrix

Configs outside this matrix → `CodegenError` from `validate_scalar_v2_config` with a field-specific message.

| Parameter | Allowed values | Rationale |
|---|---|---|
| `block_q` | 4, 8, 16, 32, 64, 128 | Multiples of 4 (warp-per-row); `block_q % 4 == 0` hard constraint |
| `block_kv` | 16, 32, 64, 128 | Multiples of 32 preferred (warp-aligned k-loop); smaller values work with predicated tail |
| `head_dim` | 32, 64, 128, 256 | Multiples of 32 (lane partitioning of d dimension) |
| `gqa_group_size` | 1, 2, 4, 8 | Unchanged from v1 — affects only K/V head index computation |
| `causal` | bool | Per-row mask: `k_global > q_row_global` → S = -inf. Single `setp.gt` inside Phase 2 k-loop |
| `paged` | bool | Unchanged — block table indirection lives in K/V base address computation, orthogonal to thread mapping |
| `rope_q` | bool + `rope_style` | Pre-rotate Q during Phase 1 load; each lane rotates its own d slice |
| `tree_mask` | bool | Unchanged — DFS timestamp check emitted inside the causal-mask branch |
| `csha` | Option<CshaExtras> | Unchanged — prologue/projection/epilogue scaffolds sit in front of / behind the core phases; they obey the same warp-per-row contract for their own Q-row state |

### Out of scope for v2

- `gpu_sm >= 80` routes through the MMA path via `use_mma_path` — selector keeps those on v1 until a separate spec covers MMA correctness.
- Backward pass (`synthesize_flash_attention_bwd_*`).
- Quantized variant (`nsl_flash_attention_quantized`).
- `head_dim < 32` or non-multiple-of-32.
- Configs whose SMEM exceeds 48 KB (see matrix below): caller gets a `CodegenError` pointing to the MMA path or to tile-size reduction.

### SMEM layout decision — all-f16 storage for Q/K/V, f32 for S/P

Canonical non-CSHA config (64/64/128) overflows the sm_75 48 KB static-SMEM default if Q/K/V are f32. All-f16 storage with `cvt.f32.f16` on load keeps the full config matrix under 48 KB and matches FA-2 community convention.

**The S rows shmem region stays f32** (declared as `.b8` sized for f32 semantics, NOT downgraded to f16 when P is written back in-place). Softmax precision requires f32 at the `exp` and `reduce` sites, and Phase 5 reads P as an f32 scalar broadcast against f32 `O_acc`. The "all-f16 storage" applies to Q/K/V only; implementors must not confuse the S/P region for an f16 one.

### SMEM budget by config (f16 Q/K/V, f32 S/P)

| Region | Formula | 32/32/32 | 64/64/128 | 128/128/256 |
|---|---|---|---|---|
| Q tile | `block_q × head_dim × 2` | 2 KB | 16 KB | 64 KB ⚠ |
| K/V tile (V reuses K) | `block_kv × head_dim × 2` | 2 KB | 16 KB | 64 KB ⚠ |
| S/P rows (f32) | `4 × block_kv × 4` | 0.5 KB | 1 KB | 2 KB |
| **Total** | | **4.5 KB** | **33 KB** | **130 KB ❌** |

128/128/256 is unsupported in v2.

### Validation

`validate_scalar_v2_config(config: &FlashAttentionConfig) -> Result<(), CodegenError>` called at the top of `synthesize_flash_attention_ptx_v2`. Rejects out-of-matrix configs with a message naming the offending field (e.g. `"block_q = 5: must be one of {4, 8, 16, 32, 64, 128}"`). Also enforces the 48 KB SMEM ceiling.

## Section 3 — Migration plumbing

### File layout

```
crates/nsl-codegen/src/
├── flash_attention.rs                   (v1, unchanged — deletion follows soak)
├── flash_attention_v2/
│   ├── mod.rs                           (public entry: synthesize_flash_attention_ptx_v2,
│   │                                     flash_attention_kernel_name_v2, shared_mem_bytes_v2)
│   ├── phases/
│   │   ├── prelude.rs                   (param decls, register decls, param loads, index comp)
│   │   ├── q_load.rs                    (Phase 1: warp-lane Q load into shmem + rope_q rotation)
│   │   ├── s_compute.rs                 (Phase 2: per-k lane-partial + shfl butterfly, S → shmem)
│   │   ├── softmax.rs                   (Phase 3: online max/sum, P writeback in-place over S)
│   │   ├── pv_accum.rs                  (Phase 5: O_acc += P * V with lane-d partitioning)
│   │   ├── finalize.rs                  (Phase 6: O_acc / row_sum + output store + LSE)
│   │   └── csha_hooks.rs                (Tier A.2.2 prologue / A.2.3 projection / A.2.4 epilogue
│   │                                     — obey warp-per-row contract)
│   ├── smem_layout.rs                   (offsets + size calc, validate_scalar_v2_config)
│   └── register_budget.rs               (per-config register count, compile-time assertion)
└── flash_attention_selector.rs          (chooses v1 vs v2 via env + config compatibility)
```

Per-phase file split caps each file at ~300 LOC and makes phase-boundary reviews tractable. The v1 monolith's 2000-line single file was the organisational condition that let the cross-phase inconsistency hide.

### Selector function

```rust
pub fn synthesize_flash_attention_ptx(config: &FlashAttentionConfig) -> Vec<u8> {
    match select_emitter(config) {
        Emitter::V1 => v1::synthesize_flash_attention_ptx(config),
        Emitter::V2 => v2::synthesize_flash_attention_ptx_v2(config),
    }
}

fn select_emitter(config: &FlashAttentionConfig) -> Emitter {
    // MMA path not this spec's concern — keep routing it to v1 until the MMA spec lands.
    if use_mma_path(config.gpu_sm) { return Emitter::V1; }
    match std::env::var("NSL_FA_EMITTER").as_deref() {
        Ok("v2") => Emitter::V2,
        Ok("v1") => Emitter::V1,
        _        => Emitter::V1,   // default = v1 until v2 soaks
    }
}
```

Existing callers (`expr/advanced.rs:1525`, `builtins.rs`, etc.) call the unchanged outer `synthesize_flash_attention_ptx` — no call-site touch-ups. Same wrapper pattern for `flash_attention_kernel_name` and `shared_mem_bytes`.

### Commit staging (~15 commits on `feat/csha-fa-scalar-rewrite`, merged into `feat/csha`)

1. Scaffold: new directory, module decls, empty phase files with `todo!()` bodies, selector defaulting to V1.
2. `validate_scalar_v2_config` + unit tests (every rejection path asserted).
3. `smem_layout` offsets + `register_budget` counter + per-config assertion tests.
4. `prelude` (param decls, register decls, param loads) + snapshot test.
5. `q_load` Phase 1 (warp-lane + optional rope_q) + snapshot test.
6. `s_compute` Phase 2 (dot product + shfl butterfly + causal mask) + snapshot test.
7. `softmax` Phase 3 (online max/sum + in-place P writeback) + snapshot test.
8. `pv_accum` Phase 5 + snapshot test.
9. `finalize` Phase 6 (output store + LSE) + snapshot test.
10. `csha_hooks` (Tier A.2.2/A.2.3/A.2.4 against v2 contract) + snapshot test.
11. Wire outer `emit_flash_attention_entry_v2` orchestrator + full-kernel snapshot test.
12. **Enable `NSL_FA_EMITTER=v2` + run Part 1 integration test → canonical-config numerical gate.**
    Commit message must state: "numerical gate on canonical config only; commit 13 extends to full matrix." This prevents a bisector or reviewer from treating commit 12 as sufficient on its own.
13. Extend Part 1 integration test with parametrized sweep across the supported matrix (see Section 4).
14. **Soak period.** `feat/csha` stays on v1 default. Document the flag in `README.md` + `SPECIFICATION.md`. Land CSHA Tier A dependents using `NSL_FA_EMITTER=v2` explicitly.
15. **Deletion commit.** Flip default to v2, remove `flash_attention.rs` v1 file + selector dispatch, update all imports. Separate PR for clean rollback optionality.

**Bisect posture:** commits 1–11 are green under default v1 selector (no behaviour change). Commit 12 is the pivot — any regression bisects to here. Commit 15 is the point of no return for v1.

### Snapshot strategy

v2 gets its own `insta` snapshot directory `crates/nsl-codegen/snapshots/flash_attention_v2/`. v1 snapshots untouched. Per-phase snapshot tests use representative configs (block_q=32/head_dim=32 CSHA canonical + block_q=64/head_dim=128 non-CSHA canonical). ~12 new snapshots across phase tests + 2 full-kernel snapshots + 1 CSHA-active full-kernel snapshot.

## Section 4 — Test strategy

### Layer 1: unit tests (pure Rust, <1 s)

- `validate_scalar_v2_config`: every rejection path has a test. Error messages asserted against stable strings.
- `smem_layout::offsets`: for each matrix entry, assert total SMEM ≤ 48 KB and Q/K/S regions don't overlap.
- `register_budget::count`: per config, assert ≤ 32 registers per thread.

### Layer 2: snapshot tests via `insta` (~2 s)

- One test per phase file: emit phase with fixed config, diff generated PTX against snapshot.
- Two full-kernel snapshots: `kernel_full__32x32x32_nocsha`, `kernel_full__64x64x128_nocsha`.
- CSHA-active variant: `kernel_full__32x32x32_cshaL2_rope`.

### Layer 3: ptxas assembly validation (~3 s)

Extend `csha_ptx_ptxas_validation.rs` with `v2_kernel_assembles_on_sm120`. Sets `NSL_FA_EMITTER=v2`, synthesizes PTX for each matrix entry, pipes through `ptxas --gpu-name sm_75`. Catches emission defects unit/snapshot tests can't see (undeclared registers, malformed operands, SMEM overflow). Gated on `ptxas` in PATH; skips otherwise.

### Layer 4: Part 1 GPU integration test (parametrized sweep, extended at commit 13)

`csha_ffi_classic_path_matches_cpu_reference` extended with a parametrized sweep:

| block_q | block_kv | head_dim | causal | variant |
|---|---|---|---|---|
| 4 | 32 | 32 | false | minimum block_q — exactly one q_tile_iter with all 4 warps active; catches bugs in loop boundary / state reset that higher block_q hides under "previous iteration" masking |
| 32 | 32 | 32 | false | CSHA canonical |
| 32 | 32 | 32 | true | CSHA canonical + causal |
| 64 | 64 | 128 | false | non-CSHA canonical |
| 64 | 64 | 128 | true | non-CSHA canonical + causal |
| 16 | 16 | 64 | true | small config |
| 128 | 128 | 128 | true | max config |

Each case: alloc device buffers, H2D, launch `nsl_flash_attention_csha` with classic-path (NULL extras) config, D2H, compare against CPU naive reference. `max_abs_diff < 5e-3` (f16 storage precision budget).

All cases run under `NSL_FA_EMITTER=v2`. An additional smoke case runs under `v1` and asserts divergence IS large — documents the pre-existing defect and proves the selector routes.

### Layer 5: Part 2 GPU integration test (unblocked by this spec)

`csha_ffi_fused_path_matches_cpu_reference` in a new file `crates/nsl-codegen/tests/csha_cuda_launch_fused.rs`. Same shape as Part 1, but with non-NULL CSHA extras (RMSNorm `x_ptr`, `norm_weight`, `Wq/Wk/Wv/Wo`, `eps`, `active_heads`, `d_model`). CPU reference reproduces full RMSNorm → Q/K/V projection → causal FA → RoPE composition. `#[ignore]`-gated like Part 1.

### Layer 6: regression — existing v1 tests stay green

- All 1377 existing `nsl-codegen` lib tests pass under default (v1) selector.
- Existing ptxas validators continue passing against v1 PTX.
- No snapshot churn in v1 snapshots.

### CI gate commits

- Commit 12: unit + snapshot + ptxas + Part 1 canonical sweep all green. GPU test on maintainer box.
- Commit 13: Part 1 full matrix sweep green. GPU test.
- Commit 15 (deletion): same as 12, plus v2 snapshots remain green, v1 snapshots deleted.

### Explicit non-goals for this test suite

- Performance/throughput validation.
- MMA path numerical coverage.
- Backward pass numerical coverage.

## Section 5 — Risks, open questions, deferred decisions

### Known risks

1. **Register pressure on sm_60/sm_61 (Pascal).** ~24 regs per thread × 128 threads × typical occupancy fits comfortably within Pascal's 65536 regs/SM. Compile-time assertion in `register_budget` catches any future config change that pushes over 255.
2. **SMEM bank conflicts at Phase 2 shfl boundary.** All 32 lanes may read the same k slot from `shmem_S[warp_id, :]` in Phases 3/5. Mitigation: pad S row stride to 33 f32s instead of 32 (or 17 if block_kv=16). Implementor confirms via `ptxas --verbose` after Phase 2 lands.
3. **`shfl.sync.bfly` membermask.** v2 uses `0xFFFFFFFF` (full warp). Safe because `block_q % 4 == 0` constraint guarantees all warps are always active; `validate_scalar_v2_config` enforces this.
4. **CSHA extras' warp-per-row compliance.** Tier A.2.2/A.2.3/A.2.4 scaffolds currently assume v1's thread-linear mapping. Commit 10 ports them to v2's contract. The existing `csha_l2_rope_ptx_assembles_on_sm120` ptxas test stays green under v2 — any emission regression is caught before commit 12's numerical gate. Commit 10 also adds a CSHA-active row to Part 1's sweep.

### Open questions (deferred to implementation)

1. **Q load strategy — cooperative vs warp-local.** Either satisfies the contract. Picked during commit 5 on occupancy-vs-code-size tradeoff.
2. **Epilogue fusion with RoPE Q rotation.** Possible to fuse rotation into the f16→f32 cvt on shmem load. Commit 5 implementor flags; if phase file grows beyond 300 LOC, split rope into its own file.
3. **LSE auxiliary output storage order.** v2 lane 0 of each warp writes `logsumexp[batch, head, q_row_global]`. Trivial, but must match backward-path expectations — noted for the backward-pass milestone, not blocking.

### Dependencies on external work

1. **MMA path investigation (parallel 1-day detour).** If MMA is structurally correct, v2 covers sm<80 only. If broken the same way, a follow-up spec B applies this spec's contract to the MMA path. Either way v2 completes on its own timeline.
2. **Emitter-hygiene follow-ups.** The four `TODO(csha-emitter)` patches in Part 1's test (`.target sm_120`, `.version 8.7`, em-dash strip, NUL/newline order) belong in `emit_ptx_header` and the string-building sites. Apply to both v1 and v2; can land independently.
3. **Backward pass alignment.** v2's output format (`out` f16, `logsumexp` f32) must match the backward path's expectations. Commit 9 decides: either v2 writes f32 out, or backward is updated to `cvt.f16.f32` on load. Current lean is f32 out to minimise backward-pass churn.

### Rollback plan

- Commits 1–11: no runtime effect (selector still defaults to v1). Revert is `git revert <commit>`.
- Commit 12+ introduces v2 under `NSL_FA_EMITTER=v2`. Rollback is flipping the env var; no revert needed.
- Commit 15 flips default to v2 and deletes v1. Rollback is revert 15 + flip env-var default. Worst case: new PR restoring v1 from git history.

### Success criteria for the whole rewrite

- All rows of Part 1's parametrized sweep pass under `NSL_FA_EMITTER=v2` with `max_abs_diff < 5e-3` vs CPU reference.
- v1 selector path remains byte-for-byte identical on the existing snapshot baseline (no regression).
- Part 2 (fused CSHA) integration test lands and passes against v2.
- Existing workspace tests (1377 nsl-codegen lib, 2155 workspace total on main) remain green throughout.
- CSHA Tier A.5 e2e numerical test (currently BLOCKED per `project_csha_tiers.md`) unblocks and passes.
