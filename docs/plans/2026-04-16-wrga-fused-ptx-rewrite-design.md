# WRGA Fused-LoRA/IA³ PTX Rewrite — Design Spec

**Date:** 2026-04-16
**Status:** Design approved; implementation plan pending
**Branch context:** builds on `feat/wrga-cpu-gpu-test-fix` (test-source fix + discovery)
**Supersedes (framing):** parts of `docs/superpowers/specs/2026-04-13-*-wrga-b3-*` close-out

---

## Preamble — Why This Milestone Exists

B.3 shipped `synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx` with PTX that looked plausible in string-pattern tests (scale-as-param, MMA-count-per-K-tile, interleaved-epilogue structure) and passed numerical tests through B.3 Task 4's **CPU-fallback** FFI bodies. The B.3 close-out memory (2026-04-13) characterised the remaining gap as "CPU→GPU tensor placement blocks the `#[ignore]`-d hardening test from firing the real cudarc launch" and documented three options to close that gap.

On 2026-04-16, the test-source fix (option 1 of those three, landed as `feat/wrga-cpu-gpu-test-fix`, commit `1b1525a`) placed all tensors on CUDA. The hardening test's fallback-reason changed from `"inputs not on GPU"` to `CUDA_ERROR_INVALID_PTX` — exposing a second, larger gap that was hidden behind the CPU-fallback path: **the emitted PTX is pseudocode, not a working kernel**. Specific defects include:

- `.param .u64 y_ptr` used directly as a memory operand (`st.global.f32 [%y_ptr + 0]`) without `ld.param.u64 %rd, [y_ptr]` first
- MMA operand names emitted without the `%` prefix (`{epi_final0, ...}` instead of `{%epi_final0, ...}`)
- `.shared` memory never declared, but `ld.shared.b32` instructions emitted
- `%smem_base_*`, `%mma_a_row`, `%mma_b_row`, `%mma_addr` declared but never assigned
- No `%tid.x` read — every thread runs identical instructions
- `u32`/`u64` type mismatch: `mad.lo.u32 ..., %smem_base_b` where `smem_base_b` is `.u64`
- `k_iters = k / 16 = 0` for sub-MMA `k=8` shapes — main loop empty, accumulator stays zero

Every one of these is a hard `ptxas` reject. None were caught because B.3's tests never fed the PTX to ptxas.

**This spec defines the milestone that closes the gap properly:**
1. a shared `kernel_skeleton/` module extracted from FA v2's proven per-phase files,
2. full rewrites of both `synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx` against that skeleton,
3. four layers of testing (skeleton snapshots, unit ptxas sweep, integration numerical at 1e-4, E2E launch-counter), all of which must be green for the milestone to close.

**Institutional lesson (to be inscribed at the top of `project_wrga_ptx_scaffolding_discovered.md` upon close-out):**

> B.3 shipped PTX scaffolding that looked correct in string-pattern tests but was never validated against ptxas or real launches. The 2026-04-16 discovery found this; this milestone closes the gap. **Future PTX-emitting milestones must include ptxas validation from the first commit.**

---

## Section 1 — Architecture and Commit Sequencing

### Two-layer emitter with shared kernel boilerplate

One new module (`kernel_skeleton/`), two rewritten emitters (`synthesize_fused_lora_ptx`, `synthesize_fused_ia3_ptx`), six commits in strict TDD order. No change to B.3's surrounding infrastructure (kernel handle routing, dedup registry, `try_cuda_launch_fused_*`, AST rewrite dispatch, WGGO→WRGA wiring) — that code all works; it simply never got to run real PTX.

### Module layout

```
crates/nsl-codegen/src/kernel_skeleton/
  mod.rs          pub re-exports
  header.rs       emit_ptx_header(ptx, PtxVersion, TargetSm)
  smem.rs         emit_static_smem_decl, emit_dynamic_smem_extern, emit_shmem_base_cvta
  indexing.rs     emit_thread_lane_warp_registers — the tid_x/warp_id/lane dance
  pad.rs          emit_smem_zero_pad_predicated — predicated st.shared for OOB tile regions
  params.rs       emit_param_block, emit_ld_param_u64 — param-name→rd-register loader
  tests/
    snapshots/    per-variant PTX snapshots (see §2)
```

The existing `matmul_mma.rs` is unchanged. Its primitives are correct at what they emit; the B.3 bug was never that `emit_mma_instruction` emitted wrong PTX — it was that callers never initialized the registers (`%mma_addr`, `%mma_a_row`, `%mma_b_row`, `%smem_base_*`) the helpers assumed existed.

### Six-commit sequence (strict TDD)

| # | Commit | What it contains | Gate tests at end of commit |
|---|---|---|---|
| 1 | `refactor(fa): extract kernel_skeleton from v2 phases` | Pure extraction from `flash_attention_v2/phases/forward/prelude.rs` and `smem_layout.rs` into new `kernel_skeleton/` module. FA v2 phases become callers of the extracted helpers. Per-variant skeleton snapshot files in `tests/snapshots/`. | FA v2 existing snapshot tests: **byte-identical**. New `kernel_skeleton/tests/snapshots/*.snap`: all green. |
| 2 | `test(wrga): ptxas validation unit test for fused LoRA (red)` | New test `lora_ptx_validates_against_cumoduleloaddata` with 6 configs. Calls `synthesize_fused_lora_ptx(cfg)` and feeds to `cudarc::driver::sys::cuModuleLoadData` (or `nvcc --ptx` when no CUDA device). Asserts success. | Unit ptxas test: **fails against current pseudocode**. Establishes the gate. |
| 3 | `feat(wrga): rewrite synthesize_fused_lora_ptx against kernel_skeleton` | Delete current `wrga_fused_ptx.rs` LoRA body. Rewrite using `kernel_skeleton` helpers + `matmul_mma` primitives per §3. | Unit ptxas test from (2): **green**. |
| 4 | `test(wrga): integration numerical test for fused LoRA at 1e-4` | New `#[cfg(feature="cuda")]`-gated test `build_4_fused_real_launch`. Forces `NSL_WRGA_FUSED_CUDA=1` on, asserts output matches CPU-reference at 1e-4. | Integration test: **green** on CUDA machines. |
| 5 | `feat(wrga): rewrite synthesize_fused_ia3_ptx + unit + integration tests` | Apply §4's kernel structure. 5-config ptxas sweep. Two integration fixtures (y=8 baseline + y=16 γ-scaling). | IA³ unit + integration: green. |
| 6 | `test(wrga): flip build_4_fused_cuda_actually_fires from #[ignore] to #[cfg(feature="cuda")]` | Remove `#[ignore]` attribute; add `#[cfg(feature="cuda")]` gate. | E2E launch-counter test: **green** on CUDA machines. **Milestone closes.** |

### What stays shipped (zero changes)

- Kernel handle routing in `compiler::Compiler::fused_ptx_kernels`
- Dedup registry via `LoraKernelKey`
- `try_cuda_launch_fused_lora` / `_ia3` in `nsl-runtime/src/fused_adapter.rs`
- Arg marshaling, grid/block derivation, error handling
- AST-level fusion decision and conditional rewrite in `wrga_adapter_rewrite.rs`
- WGGO→WRGA consumer wiring (`allocate_ranks` honoring overrides)
- B.2.1's runtime LoRA/IA³/GatedLoRA materialisation (side-table + forward rewrite + seed route-through)
- The test-source fix (`BUILD4_SRC_GPU`) from the 2026-04-16 discovery branch

### What's explicitly out of scope (separate milestones)

- **B.3.1**: GatedLoRA epilogue + PTX sigmoid (Taylor approximation or 256-entry LUT) — its own design problem with different test discipline around accuracy/register pressure
- **B.4 or later**: sm_90 WGMMA path, `ldmatrix.sync.aligned.m8n8.x4` for fragment loads, `cp.async.ca` for SMEM staging overlap, multi-warp-per-tile larger output tiles, multi-tile γ staging through SMEM
- **Deeper FA refactor** extracting the full "matmul kernel template" with a fusion-callback pattern — natural follow-up once FA and WRGA are both stable and provide two concrete instances to abstract over

---

## Section 2 — `kernel_skeleton` Contract

### Principle: fine-grained primitives, not whole-prolog helpers

The honest shared surface between FA's attention kernel and WRGA's matmul kernel is ~15 lines of actually-identical PTX. The skeleton reflects that. FA's `prelude.rs` does NOT get replaced wholesale with a single `emit_kernel_prolog` call — FA's prolog contains 30+ params, CSHA register blocks, rope_q register blocks, Tier C `save_activations` blocks, and FA-specific block-index routing. Trying to parameterize all that through one mega-helper would either force FA into a 30-field config struct WRGA would never populate, or special-case FA and defeat the extraction.

Instead, the skeleton offers small, fixed-shape primitives that both callers compose into their own prolog structure.

### Helpers

#### `header.rs`

```rust
pub enum PtxVersion { V7_0, V8_7 }  // extendable as needed
pub enum TargetSm   { Sm75, Sm80 }  // extendable as needed

pub fn emit_ptx_header(ptx: &mut String, version: PtxVersion, sm: TargetSm);
```

Emits exactly:
```
.version N.M
.target sm_K
.address_size 64
<blank line>
```

FA uses `(V8_7, Sm75)`; WRGA uses `(V7_0, Sm80)`. Bump versions as features require. No hidden branches in the helper.

#### `smem.rs`

```rust
pub fn emit_static_smem_decl(ptx: &mut String, bytes: usize);
pub fn emit_dynamic_smem_extern(ptx: &mut String);  // emits at module scope
pub fn emit_shmem_base_cvta(ptx: &mut String);      // emits in function body
```

Three separate helpers matching FA's current three-way split. Static path writes `    .shared .align 16 .b8 shmem[{bytes}];\n` inside the function body. Dynamic path writes `.extern .shared .align 16 .b8 shmem[];\n\n` at module scope BEFORE `.visible .entry`. CVTA helper writes `    cvta.shared.u64 %shmem_base, shmem;\n` inside the function body.

Callers pick which they need. FA's existing `needs_dynamic_smem(config)` and `total_bytes(config)` logic stays in FA — the skeleton doesn't care why the caller chose static vs. dynamic.

#### `indexing.rs`

```rust
pub fn emit_thread_lane_warp_registers(ptx: &mut String);
```

Emits exactly (matching [prelude.rs:94, 204-208](crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs#L94)):

```
    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;
    mov.u32 %tid_x, %tid.x;
    shr.u32 %warp_id, %tid_x, 5;
    and.b32 %lane, %tid_x, 31;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %bid_y, %ctaid.y;
```

Zero parameters — PTX convention fixes the register names. Callers needing different names alias locally with `mov`.

#### `pad.rs`

```rust
pub fn emit_smem_zero_pad_predicated(
    ptx: &mut String,
    smem_base_reg: &str,   // e.g. "%a_tile_base"
    real_extent: u32,      // e.g. 4 (rank)
    padded_extent: u32,    // e.g. 16 (MMA k)
    dtype_bits: u32,       // 16 for f16, 32 for f32
);
```

Emits the predicated-store loop that zeros `[real_extent, padded_extent)` region of SMEM for the given dtype. Used by WRGA for rank-padding in the (x@A)@B path and for m-padding when `m < 16`.

**When `real_extent == padded_extent`:** helper emits zero PTX instructions (pure no-op). The `(16, 8, 8, 16)` test config locks this behavior in via a snapshot.

Designed to be callable from FA's head_dim-edge handling in the future (not a commit-1 requirement; the signature is chosen to not lock FA out).

#### `params.rs`

```rust
pub struct Param {
    pub ty: ParamTy,       // U64, F32, U32
    pub name: &'static str,
}

pub fn emit_param_block(ptx: &mut String, entry_name: &str, params: &[Param]);
pub fn emit_ld_param_u64(ptx: &mut String, dest_reg: &str, param_name: &str);
pub fn emit_ld_param_f32(ptx: &mut String, dest_reg: &str, param_name: &str);
```

`emit_param_block` emits `.visible .entry {entry_name} (\n    .param .ty name,\n...\n)` block. Callers pass the full param list. FA has 30+ params; WRGA LoRA has 6; WRGA IA³ has 4.

`emit_ld_param_u64`/`emit_ld_param_f32` each emit exactly one line. Callers name their own destination registers — FA uses `%rd0..%rd9`, WRGA uses `%rd_x, %rd_w, ...`. The helper is a line-level utility; register numbering is caller-side.

### The diff-and-parameterize discipline for Commit 1

Before writing the extraction, the author performs this inventory:

1. Run `git log --all -- crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` to enumerate every variation FA's prolog has grown across its history.
2. Grep `flash_attention_v2/` for every `.target`, every `.shared`, every `.version` directive emitted to confirm the helper signatures above cover all current variants.
3. For each helper, add a docstring listing the FA variants it currently has to produce (e.g., "header.rs:emit_ptx_header produces: `(V8_7, Sm75)` for FA v2 forward; `(V7_0, Sm80)` for WRGA fused").

If the inventory reveals a variant the proposed API doesn't cover, the skeleton API is extended BEFORE any extraction happens. No speculative "clean" API that misses FA reality.

### Snapshot-test discipline (robust beyond docstrings)

Every variant the skeleton promises to produce has a pinned PTX snapshot in `kernel_skeleton/tests/snapshots/`. Example files:

```
kernel_skeleton/tests/snapshots/
  header__v87_sm75.snap                      # FA v2 forward/backward
  header__v70_sm80.snap                      # WRGA LoRA/IA³
  static_smem_decl__1536_bytes.snap          # WRGA LoRA SMEM
  static_smem_decl__768_bytes.snap           # WRGA IA³ SMEM
  dynamic_smem_extern.snap                   # FA large configs
  thread_lane_warp_registers.snap            # shared exact output
  smem_zero_pad__rank4_to_16_f16.snap        # WRGA rank-pad
  smem_zero_pad__rank16_to_16_f16.snap       # no-op case (empty snapshot)
  ld_param_u64.snap                          # single-line helper
  ...
```

When a future developer adds a new FA variant and the skeleton doesn't cover it, the FA snapshot fails *and* there's no pre-existing skeleton snapshot to match it against — the gap is obvious at the skeleton layer, not requiring trace-back from FA. The skeleton snapshots are 5-10 tiny files, cheap to maintain, and turn "skeleton covers these variants" from a docstring claim into a tested invariant.

### Commit 1 gate

- All existing FA v2 snapshot tests (`fa_v2_snapshots.rs`, ~40+ snapshot functions) pass byte-identically.
- All new `kernel_skeleton/tests/snapshots/*.snap` pass on first run.
- Commit body contains no logic change — only moves of existing lines from `prelude.rs`/`smem_layout.rs` into `kernel_skeleton/` and replaces them with helper calls. Any snapshot diff = extraction bug; fix inline before merging.

---

## Section 3 — Fused-LoRA Kernel Structure

### Kernel signature (unchanged from B.3)

```
.visible .entry nsl_wrga_fused_lora_m{M}n{N}k{K}r{R}(
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .f32 scale,
    .param .u64 y_ptr
)
```

### Grid/block (unchanged from B.3's launcher)

One warp (32 threads) per output tile of `(BM=16, BN=8)`.
Grid: `((m+15)/16, (n+7)/8, 1)`.
Block: `(32, 1, 1)`.

### SMEM layout (single CTA scope, static allocation)

| Region | Size | Lifetime |
|---|---|---|
| `x_tile` | 16×16 f16 = 512 B | per K-iteration (restaged each iter) |
| `w_tile` | 16×8 f16 = 256 B | per K-iteration |
| `a_tile` | 16×R' f16 = 512 B (R'=16 padded from rank≤16) | per K-iteration |
| `b_tile` | R'×8 f16 = 256 B | staged once, post-main-loop |
| **Total** | **1536 B** | well under the 48 KB static cap |

All four regions share one `shmem[1536]` byte array with fixed offsets. No dynamic SMEM.

### Phase sequence in emitter (Rust-level call order)

```rust
// 1. Header + SMEM + indexing — from kernel_skeleton
emit_ptx_header(ptx, V7_0, Sm80);
emit_param_block(ptx, name, &[x_ptr, w_ptr, a_ptr, b_ptr, scale, y_ptr]);
// (entry body begins)
emit_static_smem_decl(ptx, 1536);
emit_register_pool(ptx, &wrga_register_budget(&cfg));  // WRGA-specific
emit_shmem_base_cvta(ptx);
emit_thread_lane_warp_registers(ptx);

// 2. Param loads — WRGA-specific names, one per param
emit_ld_param_u64(ptx, "%rd_x", "x_ptr");
emit_ld_param_u64(ptx, "%rd_w", "w_ptr");
emit_ld_param_u64(ptx, "%rd_a", "a_ptr");
emit_ld_param_u64(ptx, "%rd_b", "b_ptr");
emit_ld_param_u64(ptx, "%rd_y", "y_ptr");
emit_ld_param_f32(ptx, "%scale_reg", "scale");

// 3. Output-tile coords: (row_block, col_block) = (bid_x, bid_y)
//    row_base = bid_x * 16;  col_base = bid_y * 8;
emit_output_tile_coords(ptx);

// 4. Init accumulators to zero
emit_zero_accumulators(ptx, &["main_accum"], 8);   // x@W accumulator, 8 f32 per tile
emit_zero_accumulators(ptx, &["epi_interm"], 4);   // x@A accumulator, 4 f32

// 5. Interleaved main K-loop
let k_iters = (k_in + MMA_K - 1) / MMA_K;
for k_tile in 0..k_iters {
    let k_remaining = (k_in - k_tile * MMA_K).min(MMA_K);
    // 5a. Stage x/w/a into SMEM, with k-tail and rank-pad predication
    emit_stage_x_tile(ptx, k_tile, k_remaining);
    emit_stage_w_tile(ptx, k_tile, k_remaining);
    emit_stage_a_tile(ptx, k_tile, k_remaining);
    emit_smem_zero_pad_predicated(
        ptx, "%a_tile_base + a_rank_col_offset",
        rank, 16, 16,
    );
    ptx.push_str("    bar.sync 0;\n");

    // 5b. Load fragments via matmul_mma helpers
    matmul_mma::emit_load_a_fragment_smem(ptx, &main_a_frag, "%x_tile_base", 32);
    matmul_mma::emit_load_b_fragment_smem(ptx, &main_b_frag, "%w_tile_base", 16);
    matmul_mma::emit_load_a_fragment_smem(ptx, &epi_a_frag,  "%a_tile_base", 32);

    // 5c. Main MMA: main_accum += x_tile @ w_tile
    matmul_mma::emit_mma_instruction(ptx, &main_d, &main_a, &main_b, &main_c);

    // 5d. Epilogue MMA: epi_interm += x_tile @ a_tile
    //     CORRECTNESS-CRITICAL: reuses %main_a_frag. See invariant (1) below.
    matmul_mma::emit_mma_instruction(ptx, &epi_interm_d, &main_a_frag, &epi_a_frag, &epi_interm_c);

    ptx.push_str("    bar.sync 0;\n");
}

// 6. Post-loop: stage b_tile once, compute (x@A) @ B
emit_stage_b_tile(ptx);
emit_smem_zero_pad_predicated(ptx, "%b_tile_base", rank, 16, 16);
ptx.push_str("    bar.sync 0;\n");
matmul_mma::emit_load_b_fragment_smem(ptx, &epi_b_frag, "%b_tile_base", 16);
matmul_mma::emit_mma_instruction(ptx, &epi_final_d, &epi_interm_frag, &epi_b_frag, &epi_final_c);

// 7. Scale epi_final by %scale_reg, fold into main_accum
for i in 0..8 {
    ptx.push_str(&format!("    mul.f32 %epi_final{i}, %epi_final{i}, %scale_reg;\n"));
    ptx.push_str(&format!("    add.f32 %main_accum{i}, %main_accum{i}, %epi_final{i};\n"));
}

// 8. Store with m-bounds predication
emit_store_output_tile_predicated(ptx, m_actual);
```

### Register budget (LoRA, single 16×8 output tile)

| Pool | Count | Purpose |
|---|---|---|
| `.reg .f32 %main_accum<8>` | 8 | x@W accumulator |
| `.reg .f32 %epi_interm<4>` | 4 | x@A accumulator |
| `.reg .f32 %epi_final<4>` | 4 | (x@A)@B accumulator |
| `.reg .b32 %main_a_frag<4>` | 4 | x fragment (reused in both MMAs per iter) |
| `.reg .b32 %main_b_frag<2>` | 2 | W fragment |
| `.reg .b32 %epi_a_frag<4>` | 4 | A fragment |
| `.reg .b32 %epi_b_frag<2>` | 2 | B fragment |
| `.reg .u64` rd scratch | ~16 | param pointers + SMEM bases + addr math |
| `.reg .u32` scratch | ~12 | tid derivatives, tile indices |
| `.reg .pred` | ~4 | predication for tail/pad stores |

Well inside a warp's register budget. No spill pressure.

### Correctness invariants (must survive any future edit)

**Invariant 1 — Fragment reuse in interleaved epilogue.**
`epi_interm += x_tile @ a_tile` (step 5d) MUST consume `%main_a_frag` directly — the same register-fragment just used in step 5c's `main_accum += x_tile @ w_tile`. This is correct because `m16n8k16` A-fragments encode only the `m×k` tile layout per lane and are independent of the B matrix's semantic meaning — the fragment distribution across 32 lanes is a function of MMA shape, not of what B contains. The source code at the step-5d call site MUST have a comment referencing this invariant:

```rust
// INVARIANT: m16n8k16 A-fragments encode only the m×k tile and are
// independent of the B matrix. %main_a_frag loaded from x_tile in step 5b
// is valid as the A-operand for BOTH x@W (B=w_tile) and x@A (B=a_tile).
// Reloading x would double the SMEM read cost and risk lane-alignment
// regressions. See spec §3 invariant (1).
```

A grep-based unit test asserts the emitter's output contains `%main_a_frag` as an operand in exactly two MMA instructions per K-iteration (one main, one epi_interm).

**Invariant 2 — Interleaved epilogue is load-bearing.**
`epi_interm += x_tile @ a_tile` MUST happen INSIDE the main K-loop, using the current iteration's `%main_a_frag` and `%epi_a_frag`. Moving it post-loop would see only the last `x_tile` since SMEM is overwritten each iter. B.3 got this right in prose; the rewritten kernel enforces it via the emitter's call order (step 5d is inside the loop body, not after it).

**Invariant 3 — `bar.sync 0` discipline.**
One barrier before fragment loads (step 5b) — SMEM writes from step 5a must complete before any thread reads. One barrier at end of iteration (after step 5d) — SMEM reads must complete before the next iteration overwrites SMEM in step 5a. Removing either is a silent-corruption bug.

**Invariant 4 — Padding happens in SMEM, not HBM.**
Global loads in steps 5a and 6 are predicated to skip out-of-bounds indices; SMEM fills those slots via `emit_smem_zero_pad_predicated`'s `st.shared.b16 [addr], 0`. HBM traffic stays minimal — the fusion win (avoiding HBM round-trips for the adapter matmul) is preserved. Any "optimization" that loads zeros from a pre-zeroed HBM buffer defeats the fusion purpose.

**Invariant 5 — `matmul_mma` register preconditions.**
The `matmul_mma::emit_load_a_fragment_smem` / `emit_load_b_fragment_smem` helpers assume `%mma_addr`, `%mma_a_row`, `%mma_b_row`, `%smem_base_*` are declared and initialized with correct per-lane values for `m16n8k16` lane layout. The rewrite MUST initialize these registers before the first fragment load call. A dedicated unit test asserts the emitter's init sequence produces the expected PTX for lane-derivation from `%tid_x` (`%mma_a_row = %tid_x >> 2`, `%mma_b_row = (%tid_x >> 2) & 7`, etc., per the m16n8k16 spec).

### Test matrix for LoRA unit ptxas sweep (6 configs)

| `(m, n, k, rank)` | Why this config |
|---|---|
| `(16, 8, 16, 16)` | Canonical Ampere shape; no padding on any axis |
| `(16, 8, 32, 4)` | Multi-K-iter (2 tiles); rank-padding on a_tile |
| `(1, 8, 8, 2)` | Sub-MMA m AND k; rank-padding — smallest real config |
| `(4, 8, 8, 2)` | Matches `BUILD4_SRC_GPU` hardening-test shape |
| `(32, 16, 64, 8)` | Multi-tile output grid (2×2); 4 K-iters; rank-pad |
| `(16, 8, 8, 16)` | K-padding required; rank == MMA k, NO rank-padding — exercises `emit_smem_zero_pad_predicated` no-op path |

Each config's PTX is fed to `cudarc::driver::sys::cuModuleLoadData` (or `nvcc --ptx` when no CUDA device is available). Test asserts success per config, with config-specific failure messages pointing at the offending shape.

---

## Section 4 — Fused-IA³ Kernel Structure

### Why simpler than LoRA

IA³'s epilogue is `y = (x @ W) * γ` — γ is a per-output-column vector of size `n`. No A/B intermediate matmuls, no rank dimension, no interleaved epilogue, no scale param (γ IS the scaling).

### Kernel signature

```
.visible .entry nsl_wrga_fused_ia3_m{M}n{N}k{K}(
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 gamma_ptr,
    .param .u64 y_ptr
)
```

### Grid/block

Identical to LoRA — one warp per `(16, 8)` output tile.

### SMEM layout

| Region | Size | Lifetime |
|---|---|---|
| `x_tile` | 16×16 f16 = 512 B | per K-iteration |
| `w_tile` | 16×8 f16 = 256 B | per K-iteration |
| **Total** | **768 B** | half LoRA's footprint |

No `a_tile`, no `b_tile`. Consequently no rank-axis padding.

### Register budget (IA³)

| Pool | Count | Purpose |
|---|---|---|
| `.reg .f32 %main_accum<8>` | 8 | x@W accumulator |
| `.reg .f32 %gamma<8>` | 8 | γ slice for this output tile's 8 cols |
| `.reg .b32 %main_a_frag<4>` | 4 | x fragment |
| `.reg .b32 %main_b_frag<2>` | 2 | W fragment |
| `.reg .u64` rd scratch | ~12 | param pointers + SMEM base |
| `.reg .u32` scratch | ~10 | tid derivatives |
| `.reg .pred` | ~4 | predication for tail/OOB stores |

Drops 14 registers vs. LoRA. No register pressure concerns.

### Phase sequence

```rust
// 1. Header + SMEM + indexing (same kernel_skeleton helpers as LoRA)
emit_ptx_header(ptx, V7_0, Sm80);
emit_param_block(ptx, name, &[x_ptr, w_ptr, gamma_ptr, y_ptr]);
emit_static_smem_decl(ptx, 768);
emit_register_pool(ptx, &wrga_ia3_register_budget(&cfg));
emit_shmem_base_cvta(ptx);
emit_thread_lane_warp_registers(ptx);

// 2. Param loads (no scale; γ is via HBM not param)
emit_ld_param_u64(ptx, "%rd_x", "x_ptr");
emit_ld_param_u64(ptx, "%rd_w", "w_ptr");
emit_ld_param_u64(ptx, "%rd_gamma", "gamma_ptr");
emit_ld_param_u64(ptx, "%rd_y", "y_ptr");

// 3. Output-tile coords (shared helper with LoRA)
emit_output_tile_coords(ptx);

// 4. Zero-init main_accum only (no epi pools)
emit_zero_accumulators(ptx, &["main_accum"], 8);

// 5. Main K-loop — stage x/w, load fragments, MMA (NO step 5d, NO a/b tiles)
for k_tile in 0..k_iters {
    emit_stage_x_tile(ptx, k_tile, k_remaining);
    emit_stage_w_tile(ptx, k_tile, k_remaining);
    ptx.push_str("    bar.sync 0;\n");
    matmul_mma::emit_load_a_fragment_smem(ptx, &main_a_frag, "%x_tile_base", 32);
    matmul_mma::emit_load_b_fragment_smem(ptx, &main_b_frag, "%w_tile_base", 16);
    matmul_mma::emit_mma_instruction(ptx, &main_d, &main_a, &main_b, &main_c);
    ptx.push_str("    bar.sync 0;\n");
}

// 6. Load γ from HBM — 8 f32 values for this tile's output cols
//    %rd_gamma_col = %rd_gamma + col_base * 4
for i in 0..8 {
    ptx.push_str(&format!(
        "    ld.global.f32 %gamma{i}, [%rd_gamma_col + {}];\n", i * 4,
    ));
}

// 7. Broadcast multiply: main_accum *= γ (per output column)
for i in 0..8 {
    ptx.push_str(&format!(
        "    mul.f32 %main_accum{i}, %main_accum{i}, %gamma{i};\n"
    ));
}

// 8. Store with m- and n-bounds predication (shared helper with LoRA)
emit_store_output_tile_predicated(ptx, m_actual);
```

### Correctness invariants

**Invariant A — γ loaded AFTER the main K-loop, not inside it.**
Unlike A/B in LoRA (where x@A must interleave to reuse fragments), γ is K-independent — it only multiplies the final accumulator. Inside-loop loading would waste cache bandwidth with no correctness benefit.

**Invariant B — γ loaded from HBM directly, not via SMEM.**
For `n=8` this is 32 bytes (one cache line) per CTA — not worth a SMEM hop and stage-sync cost. If `n` grows such that multi-tile output is emitted per CTA, γ would be staged through SMEM; that path is deferred.

**Invariant C — No γ padding required.**
γ is exactly size `n`. Sub-MMA `n < 8` cases are handled by step 8's predicated store, which skips OOB output cols — γ's OOB slots are never read because the corresponding `main_accum` values are never stored.

### Test matrix for IA³ unit ptxas sweep (5 configs)

| `(m, n, k)` | Why this config |
|---|---|
| `(16, 8, 16)` | Canonical Ampere shape |
| `(16, 8, 32)` | Multi-K-iter |
| `(1, 8, 8)` | Sub-MMA m and k — smallest |
| `(4, 8, 8)` | Hardening-test shape equivalent |
| `(32, 16, 64)` | Multi-tile output grid |

### Integration test fixtures for IA³

Two numerical fixtures in `wrga_adapter_runtime_equivalence.rs`:

**Fixture A — full compute path:**
```
x = ones([4, 8]), W = ones([8, 8]), γ = ones([8])
Expected y[i,j] = (x@W)[i,j] * γ[j] = 8 * 1 = 8 elementwise
```
Proves matmul pipeline runs end-to-end with γ as a passthrough.

**Fixture B — γ actually multiplies:**
```
x = ones([4, 8]), W = ones([8, 8]), γ = [2, 2, 2, 2, 2, 2, 2, 2]
Expected y[i,j] = 8 * 2 = 16 elementwise
```
Proves γ is read and applied, not silently ignored. Together the two fixtures catch:
- Matmul doesn't run → fixture A fails
- γ not read → fixture A passes, fixture B fails (still 8, not 16)
- γ applied but scale wrong → both fixtures fail with wrong values

Both fixtures asserted at 1e-4 under `NSL_WRGA_FUSED_CUDA=1`.

---

## Section 5 — Consolidated Test Matrix, Open Risks, Close-Out

### Consolidated test coverage across the 6 commits

| Commit | New test | Purpose | Gate state at commit end |
|---|---|---|---|
| 1 | — (existing FA snapshots act as gate) | Extraction fidelity | FA v2 snapshots byte-identical ✓ |
| 1 | `kernel_skeleton/tests/snapshots/*.snap` | Per-variant skeleton coverage | All 5-10 skeleton snapshots green ✓ |
| 2 | `lora_ptx_validates_against_cumoduleloaddata` | Reject invalid PTX | **Red** against pseudocode |
| 3 | — (test from commit 2 flips) | Rewrite is valid | Commit-2 test green ✓ |
| 4 | `build_4_fused_real_launch` (CUDA-gated) | Numerical correctness on real PTX | Green on CUDA ✓ |
| 5 | `ia3_ptx_validates_against_cumoduleloaddata`, `build_ia3_fused_fixture_a/b` | IA³ correctness | Green ✓ |
| 6 | `build_4_fused_cuda_actually_fires` (flipped) | Dispatch + non-fallback proof | Green on CUDA, `count ≥ 1` ✓ |

### Failure-mode coverage map

- **Skeleton variant snapshots**: drift in `kernel_skeleton/` during future FA refactors. Fails when a new FA variant isn't represented.
- **Unit ptxas sweep (per-emitter)**: invalid PTX syntax, missing `.shared` decls, uninitialized registers, operand-type mismatches, `.param`-as-operand bugs — exactly the class B.3 had.
- **Integration numerical**: fragment-layout bugs (correct PTX that silently computes the wrong thing), lane-distribution errors, wrong SMEM offsets, tile-iteration bugs. Only visible when the kernel actually runs.
- **E2E launch-counter**: dispatch-path regressions (AST rewrite stops emitting the fused Call, `try_cuda_launch_fused_*` silently falls back). Catches infrastructure drift without needing numerical precision.

All four layers are needed because each catches a distinct failure mode. B.3's mistake was having only the string-pattern variant of the first layer plus CPU-fallback integration — no real-PTX validation at all.

### Open risks and deferred items

**1. `ldmatrix.sync.aligned.m8n8.x4` not used.**
`matmul_mma::emit_load_a_fragment_smem` uses 4 × `ld.shared.b32` per fragment. Works; slower than `ldmatrix`. Deferred perf pass. Not a correctness risk.

**2. `cp.async.ca` not used.**
Sync `ld.global + st.shared` blocks the warp during staging. Ampere's `cp.async` overlaps staging with MMA. Same scope as ldmatrix — correctness first, perf pass later.

**3. `matmul_mma` register preconditions are now documented.**
Invariant 5 of §3. The rewrite MUST initialize `%mma_addr`, `%mma_a_row`, `%mma_b_row`, `%smem_base_*` per m16n8k16 lane layout before first call. Unit test for init sequence enforces it structurally.

**4. `(16, 8, 8, 16)` pad-helper skip path is a tested invariant.**
When `real_extent == padded_extent`, `emit_smem_zero_pad_predicated` emits zero instructions. Snapshot `smem_zero_pad__rank16_to_16_f16.snap` (empty content) locks this in.

**5. Integration tests gate on `#[cfg(feature="cuda")]`.**
Contributors without local GPUs can still run unit ptxas sweep (no GPU required for compile validation). Integration + E2E need a CUDA device. Matches existing FA-GPU test gating.

**6. Perf is NOT a milestone gate.**
If the kernel is numerically correct but 2× slower than B.2.1's unfused path, the milestone still closes. Perf work (ldmatrix, cp.async, multi-warp tiles, WGMMA on sm_90) is separately tracked as a follow-up milestone.

**7. Fragment-reuse invariant is load-bearing.**
`%main_a_frag` serves both MMAs in the LoRA K-iteration. Source code comment at the call site + grep-based unit test asserting "same fragment appears in exactly two MMA instructions per K-iteration".

### Close-out criteria (all must hold for milestone to close)

1. All 6 commits merged (main or a landing PR).
2. FA v2 snapshot tests: 100% byte-identical after commit 1.
3. `kernel_skeleton/tests/snapshots/*.snap`: all green; every promised variant has a pinned snapshot.
4. WRGA LoRA unit ptxas test: green across all 6 configs.
5. WRGA LoRA integration test: green at 1e-4 under `NSL_WRGA_FUSED_CUDA=1` on CUDA machines.
6. WRGA IA³ unit ptxas test: green across all 5 configs.
7. WRGA IA³ integration test: green on both fixtures at 1e-4.
8. `build_4_fused_cuda_actually_fires`: flipped from `#[ignore]` to `#[cfg(feature="cuda")]`; `count ≥ 1` on CUDA.
9. Memory updates:
   - `project_wrga_ptx_scaffolding_discovered.md`: prepend retrospective paragraph + "CLOSED 2026-XX-XX" marker.
     - Retrospective (verbatim): *"B.3 shipped PTX scaffolding that looked correct in string-pattern tests but was never validated against ptxas or real launches. The 2026-04-16 discovery found this; this milestone closes the gap. Future PTX-emitting milestones must include ptxas validation from the first commit."*
   - New `project_wrga_fused_ptx_rewrite.md`: records invariants (A-fragment reuse, interleaved epilogue, SMEM-side padding, `bar.sync` discipline, `matmul_mma` register preconditions, kernel_skeleton extraction boundaries) so future edits don't regress them.
   - `MEMORY.md`: update index pointers — mark the `project_wrga_ptx_scaffolding_discovered.md` entry as CLOSED; add the new `project_wrga_fused_ptx_rewrite.md` entry.

### Explicit non-goals (deferred to separate milestones)

- **B.3.1 — GatedLoRA**: PTX sigmoid (Taylor approximation vs. 256-entry LUT), interleaved-epilogue extension for gate multiplication. Its own numerics-accuracy and register-pressure design problem.
- **B.4 or later — perf passes**: `ldmatrix.sync.aligned.m8n8.x4`, `cp.async.ca` staging overlap, multi-warp-per-tile output, multi-tile γ staging through SMEM, sm_90 WGMMA path.
- **Deep kernel_skeleton refactor**: extracting the full "matmul kernel template" with fusion-callback pattern — natural follow-up once FA and WRGA are both stable and provide two concrete instances to abstract over.

---

## Dependencies and Prerequisites

This milestone builds on:

- **CSHA Tier A** (shipped 2026-04-15): FA v2 per-phase structure is the source for `kernel_skeleton/` extraction.
- **CSHA Tier C** (structural close-out 2026-04-15, numerical gate still blocked): independent of this milestone; Tier C's fused-backward work and this PTX rewrite don't share code paths.
- **WRGA A through B.3** (shipped 2026-04-13): all infrastructure around the two synthesizers remains unchanged. This milestone replaces only `synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx` bodies.
- **WRGA test-source fix** (shipped 2026-04-16 on `feat/wrga-cpu-gpu-test-fix`): `BUILD4_SRC_GPU` stays as-is; commit 6 flips the hardening test's `#[ignore]` to `#[cfg(feature="cuda")]`.
- **B.2.1 runtime adapter infrastructure**: side-table allocation, forward rewrite, init emission, seed route-through — all unchanged.

## Branching strategy

- Commits 1-6 land on a single feature branch `feat/wrga-fused-ptx-rewrite` (or similar).
- Commit 1 (FA extraction) could land independently as a pre-req PR if its snapshot discipline is proven clean before WRGA changes start — this reduces review surface.
- The existing `feat/wrga-cpu-gpu-test-fix` branch remains the landing point for the test-source fix and discovery memory; this milestone's branch may or may not merge that in depending on PR ordering.

## Spec self-review notes

- No placeholders (TBD/TODO) remain. All test configs, commit gates, memory-file updates, and invariant wordings are specified.
- No internal contradictions detected between sections (LoRA SMEM total 1536 B and IA³ 768 B consistent with region breakdowns; test matrix in §3/§4 consistent with consolidated table in §5).
- Scope is focused on one milestone: rewrite two PTX synthesizers + extract shared skeleton + four-layer test discipline. GatedLoRA, perf passes, and deep refactor are explicitly deferred.
- Ambiguity check: "what counts as a real CUDA launch" pinned as `FUSED_ADAPTER_GPU_LAUNCH_COUNT` increment in `try_cuda_launch_fused_*` success path (existing counter, no new definition). "ptxas validation" pinned as `cudarc::driver::sys::cuModuleLoadData` return code 0 (with `nvcc --ptx` as a no-GPU fallback). "Numerical correctness" pinned at 1e-4 absolute tolerance elementwise.
