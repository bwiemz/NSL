# WRGA Milestone B.3 — Epilogue-Fused LoRA / IA³ MMA PTX

**Status:** Design approved 2026-04-13.

**Goal:** Ship the epilogue-fused MMA kernel that matches the WRGA paper's "+0 memory ops" claim for LoRA and IA³ adapter sites on sm_80+. B.2.1's unfused 3-FFI path becomes the sm_75 fallback. GatedLoRA's sigmoid-gated epilogue defers to B.3.1 (needs a PTX sigmoid). Custom PTX emission; no Cutlass runtime-epilogue hand-off; paper-aligned.

## Background

B.2.1 landed runtime adapter application. `@adapter(type=lora, ...)` now allocates real A/B tensors via a side-table and rewrites `x @ self.W` to `x @ self.W + ((x @ self.A) @ self.B) * scale` at AST time. Every adapter site compiles to **three `nsl_tensor_matmul` calls** (plus scale+add) — three Cutlass invocations through the runtime FFI. Build 4 of the runtime equivalence test proves correctness at `16.0` per element.

The paper (Innovation 4, Fusion-Integrated Adapters) specifies epilogue fusion:
> "Instead of computing `y = W₀x + BAx` as two matmuls + an add, the adapter's contribution `BAx` is computed inside the epilogue of the `W₀x` matmul kernel. The epilogue (the code that writes the matmul result to memory) is extended to also accumulate the adapter's contribution before the write. This eliminates one memory round-trip."

Today NSL has **no standalone matmul-to-PTX path**. All non-FA matmul dispatches through `nsl_tensor_matmul` to Cutlass. B.3 introduces the first compile-time matmul PTX emitter, specifically scoped to adapter sites.

## Architectural choices (locked during brainstorm)

1. **Custom PTX emission, not Cutlass epilogue-fusion API.** The paper positions epilogue fusion as "Why this is impossible in PyTorch" — a compile-time moat. Using Cutlass's runtime epilogue API would replicate what PyTorch+TE already offers. Custom PTX honors the paper's framing and lets B.3 serve as scaffolding for future compile-time fusion work.

2. **Scope: LoRA + IA³ in B.3.** GatedLoRA's epilogue needs a PTX `sigmoid` (Taylor approximation or lookup table), which is a separable concern. LoRA + IA³ share 90% of the matmul-epilogue infrastructure; adding IA³ once LoRA works costs maybe 2 days.

3. **Dual-tolerance validation.** B.2.1's Build 4 at 1e-5 keeps covering the unfused path (sm_75 fallback or opt-out). A new `build_4_fused` test runs the same source through the fused kernel with 1e-4 tolerance. **1e-4, not 1e-3.** The fused path uses the same MMA instructions with the same f32 accumulator as the unfused path; the only numeric difference is associativity (`(x@W) + (x@A)@B*scale` accumulates into one register in the fused path vs. three independent stores in the unfused). That's well under 1e-4 for reasonable tensor magnitudes. **1e-3 tolerance is a bug signal, not an acceptable relaxation** — if 1e-3 is needed to pass, investigate the PTX emission (likely a missing f32→f16 conversion or an accumulator reset).

4. **`activation_live` handling: verification, not active removal.** See Task 3 below — B.2.1's rewrite becomes conditional on `FusionDecision`. For fused sites, the intermediate never enters the AST, so `activation_live` never gets it. Task 3 asserts this rather than removing an existing entry.

5. **sm_80+ gate.** Fused path fires only when `use_mma_path()` (from `flash_attention.rs:62`) returns true. sm_75 and older use B.2.1's unfused path unchanged.

6. **Tile size m16n8k16.** Matches the f16-input/f32-accumulator MMA helpers FA already uses. Most register-efficient Ampere configuration.

## Register budget analysis

For a single m16n8k16 tile on sm_80 with one CTA of 128 threads (4 warps):

| Component | Registers (per thread) |
|---|---|
| Main `x @ W` accumulator (8 × f32 fragments across 2 m16 halves) | 8 |
| Main A fragment (x tile) | 4 |
| Main B fragment (W tile) | 2 |
| Epilogue A fragment (x@A result, rank dim = K) | 4 |
| Epilogue B fragment (adapter B) | 2 |
| Epilogue `(x@A)@B` accumulator | 8 |
| SMEM base pointers (x, W, A, B) | 4 |
| K-tile loop counter + bounds | 2 |
| Predicate registers (load-boundary + tail handling) | 2 |
| Double-buffered load staging (K-pipeline, x + W) | 8 |
| **Subtotal fixed overhead** | **~44** |
| K-loop iteration registers (scratch for each tile) | ~4–8 |
| Scale constant + address arithmetic | 2 |

**Fixed overhead ~50 registers.** The remaining ~200 registers (at occupancy=1) must accommodate the epilogue's rank-dimension tile loop. The epilogue's A fragment loads a `[tile_m, rank]` slice of `x @ A`; its K dimension IS `rank`. For `rank ≤ 16`, the epilogue fits in one MMA iteration (k=16 matches the m16n8k16 tile's K). For `rank = 32`, two MMA iterations are needed; register pressure grows linearly with the number of epilogue tiles.

**Confirmed ceiling: rank ≤ 16 single-pass.** Higher ranks would require splitting the epilogue across multiple `mma.sync` calls, which adds loop-counter + pipeline-stage registers that push close to the 255-reg limit. **B.3 ships `rank ≤ 16`; `rank > 16` compiles with an explicit error pointing at a follow-up milestone for multi-pass epilogues.** The decorator's `rank=...` is a compile-time integer literal, so the error fires cleanly at the inject pass.

## Task breakdown

### Task 1 — Standalone MMA matmul infrastructure

**What it does:** Extract reusable MMA primitives from `flash_attention.rs:1323-1422` into a new `crates/nsl-codegen/src/matmul_mma.rs` module. FA's helpers are deeply inlined with attention-specific loops; B.3 needs parameterized versions.

**New module signature:**
```rust
pub fn emit_mma_matmul(
    ptx: &mut String,
    shape: MmaShape,                 // m16n8k16 for B.3
    lhs_layout: FragmentLayout,      // Row or Col
    rhs_layout: FragmentLayout,
    acc_dtype: AccDtype,             // F32 for B.3
    lhs_smem_reg: &str,
    rhs_smem_reg: &str,
    acc_regs: &[&str; 8],            // f32 accumulator fragments
);
```

Plus the fragment-load helpers (`emit_load_a_fragment_smem`, `emit_load_b_fragment_smem`) factored out as `pub(crate) fn` with layout parameters.

**FA stays untouched.** Extraction happens by copy-then-parameterize, not by refactoring FA to use the new module. FA's backward-MMA path has too many attention-specific optimizations (swizzle, online softmax interleaving) to unify in B.3's scope. After B.3 ships, a follow-up can migrate FA to the new module if beneficial.

**Unit tests** (inline in `matmul_mma.rs`): golden PTX outputs for m16n8k16 f16×f16→f32, both `Row×Col` and `Col×Row` layouts. Assert the generated `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` instruction and that register names are referenced correctly.

**Acceptance:** `cargo test -p nsl-codegen matmul_mma` passes. FA backward MMA tests remain green (untouched).

### Task 2 — Adapter-epilogue PTX generator

**Files:**
- Create: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

**What it does:** Generates the full PTX kernel for one LoRA or IA³ site. Entry points:

```rust
pub struct FusedLoraConfig {
    pub site_id: String,
    pub m: u32,               // batch dim
    pub n: u32,               // output dim = d_out
    pub k: u32,               // input dim = k_in (matches W's k)
    pub rank: u32,            // rank (≤ 16)
    pub target_sm: u32,       // 80 or 86 etc.
    // NOTE: scale (alpha/rank) is NOT part of the kernel dedup key or
    // config struct — it's passed at launch time as a kernel parameter
    // (.param .f32).  This enables dedup by (m, n, k, rank, target_sm)
    // across sites with different alpha values.  See Task 4 dedup notes.
}

pub fn synthesize_fused_lora_ptx(config: &FusedLoraConfig) -> String;

pub struct FusedIa3Config {
    pub site_id: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub target_sm: u32,
}

pub fn synthesize_fused_ia3_ptx(config: &FusedIa3Config) -> String;
```

**LoRA kernel structure (interleaved epilogue — CRITICAL):**

The naive "main loop then epilogue" structure is **wrong**: by the end of the main K-tile loop, SMEM has been overwritten each iteration, and only the last K-tile's x fragment lives in `main_A_frag`. A post-loop `x @ A` would compute `x_last_tile @ A`, not `x @ A`. Since `x @ A` has K dimension = k_in (same as the main matmul's K), the epilogue needs the FULL x row.

**Correct structure: interleave the `x @ A` accumulation into the main K-loop.** Each main iteration, after the main MMA, also multiplies the current x tile against the corresponding A tile slice, accumulating into `epilogue_intermediate` (`x @ A` in registers). This piggybacks on x tiles already in SMEM/registers (zero extra HBM reads for x) and only costs one additional A-tile load per K iteration.

1. Header + `.entry` declaration with params `x`, `W`, `A`, `B`, `scale`, `y`. (scale is a `.param .f32`, not a PTX literal — see Task 4 dedup notes.)
2. Register decl: main accum × 8, main A frag × 4, main B frag × 2, epilogue A frag × 4 (for a tile of the `A` matrix, NOT for `x @ A`), `epilogue_intermediate` accum × 4 (f32, stores `x @ A` tile result), epilogue B frag × 2, epilogue final accum × 8, SMEM pointers, loop counters, scale register.
3. Load `scale` from `.param` into a dedicated f32 register once (before the K-loop).
4. **Interleaved main K-loop** — each iteration:
    a. Load x tile and W tile from global → SMEM → `main_A_frag` / `main_B_frag`.
    b. `mma.sync main_A_frag @ main_B_frag → main_accum` (standard `x @ W` accumulation).
    c. Load the corresponding A tile slice (rows matching this K chunk) from global → SMEM → `epilogue_A_frag`.
    d. `mma.sync main_A_frag @ epilogue_A_frag → epilogue_intermediate` (accumulates the current x tile's contribution to `x @ A`).
5. **Epilogue (after K-loop completes):** `epilogue_intermediate` now holds the full `x @ A` result in registers.
    a. Load `B` tile from global → SMEM → `epilogue_B_frag`.
    b. `mma.sync epilogue_intermediate @ epilogue_B_frag → epilogue_final_accum` (this is `(x@A) @ B`).
    c. Multiply `epilogue_final_accum` by the scale register (8 × `mul.f32`).
    d. Add `epilogue_final_accum` into `main_accum` (8 × `add.f32`).
6. Store `main_accum` to global `y`.

This structure ensures `x @ A` is computed over the full K dimension and keeps everything register/SMEM-resident.

**IA³ kernel structure (no second MMA, no interleaving needed):**
Same header + main K-loop as above through step 4b (standard `x @ W` accumulation; no epilogue interleaving because IA³ has no `x @ A` term). Epilogue after the main K-loop:
   - Load `γ` (1-D vector of shape `[n]`) from global into registers.
   - Broadcast-multiply each element of `main_accum` by the corresponding `γ[j]` (8 × `mul.f32`).
   - Store.

**Register-count static assertion:** at the top of each synthesize fn, `assert!(config.rank <= 16, "B.3 rank ceiling; multi-pass epilogue is follow-up");`.

**Acceptance:** Golden-PTX tests (same format as FA's `snapshot_tests.rs`). Assert the generated PTX contains exactly 2 `mma.sync.aligned.m16n8k16` for LoRA, 1 for IA³; contains the scale literal as `.f32`; has the correct parameter count.

### Task 3 — Verify `activation_live` is clean for fused sites

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fusion.rs` (add assertion; no active removal)
- Modify: `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` (new test)

**What it does:** This is where the architecture decision matters. Since Task 4 makes B.2.1's AST rewrite conditional on `FusionDecision`, fused sites emit a single `nsl_adapter_fused_lora_matmul` FFI call instead of three matmul calls. The `x @ A` intermediate is **never in the AST** for fused sites, so it never enters the Wengert list, so it never gets a `VarId`, so `activation_live` never contains it.

Task 3 is therefore a **verification pass**, not an active code change. Add an assertion in `wrga_fusion.rs` after the fusion-plan pass completes: for each `FusionDecision::EpilogueFusedLora`, walk the Wengert list and assert that no `Matmul(x, A)` op exists with the site's VarIds. If present, that's a bug (rewrite failed to fuse or site was mis-classified).

**New test:** `fused_lora_site_leaves_no_intermediate_activation`. Compile a LoRA-decorated source with `EpilogueFusedLora` in the plan. Assert `plan.memory.assignments` contains no entry for an `x @ A` intermediate at that site. Also assert the total activation count is one less than the unfused path would produce.

**Acceptance:** Assertion fires correctly on a synthetic test where the rewrite incorrectly emitted the triple. Normal compiles pass without the assertion firing.

### Task 4 — Dispatch wiring: new FFI + conditional AST rewrite

**Files:**
- Create runtime FFI: `crates/nsl-runtime/src/fused_adapter.rs` (new file)
- Modify: `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` (conditional emission)
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs` (PTX kernel registration)

**New FFIs:**
```rust
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_lora_matmul(
    x: i64,                    // tensor ptr
    w: i64,
    a: i64,
    b: i64,
    scale: f32,
    ptx_kernel_ptr: i64,       // pre-compiled PTX module handle
) -> i64;                      // y tensor ptr

#[no_mangle]
pub extern "C" fn nsl_adapter_fused_ia3_matmul(
    x: i64,
    w: i64,
    gamma: i64,
    ptx_kernel_ptr: i64,
) -> i64;
```

Implementation: load PTX kernel from handle, set up CUDA grid/block dims from tensor shapes, launch. The PTX is synthesized at compile time and embedded as a static string resource in the output binary; the runtime just launches it.

**Existing `nsl_tensor_matmul` is untouched.** No flag parameter, no optional params. New FFIs have clean signatures tailored to fused adapters.

**B.2.1 AST rewrite modification (NOT a new rewrite pass):**

In `wrga_adapter_rewrite.rs`, modify `synthesize_lora_adapted` to check the site's `FusionDecision` before emitting:

```rust
fn synthesize_lora_adapted(original, lhs, site, ctx) -> Expr {
    match site.fusion_decision {
        Some(FusionDecision::EpilogueFusedLora { .. }) if ctx.sm_version >= 80 => {
            // Fused path: single FFI call.
            build_call(
                "nsl_adapter_fused_lora_matmul",
                [lhs, self_W_expr, self_loraA_expr, self_loraB_expr,
                 FloatLit(site.alpha as f32 / site.rank as f32),
                 ptx_kernel_handle_for(site)],
            )
        }
        _ => {
            // Unfused path: B.2.1's triple-matmul + scale+add expression (existing code).
            build_lora_triple(original, lhs, site, ctx)
        }
    }
}
```

Same pattern for IA³ (`synthesize_ia3_adapted`).

**PTX kernel registration:** During the fusion pass, for each `FusionDecision::EpilogueFusedLora`, call `synthesize_fused_lora_ptx(config)` and store the result in `compiler.fused_ptx_kernels: HashMap<SiteId, String>`. Emit the strings as static symbols in the output binary; runtime handles are tensor-pointer offsets into the embedded PTX.

**AdapterSite extension:** Add `fusion_decision: Option<FusionDecision>` (or similar discriminant) so the rewrite pass can see it without cross-looking-up the plan.

**Acceptance:** Single LoRA/IA³ compile produces exactly one fused FFI call per site on sm_80+, three unfused calls on sm_75. Verified via IR dump or an FFI-call-count side-channel test.

### Task 5 — Dual-tolerance equivalence test

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

**What it does:**

Keep B.2.1's existing Build 4 at 1e-5 — it exercises the unfused path, which remains the sm_75 fallback and the explicit opt-out.

Add `build_4_fused`: same source as Build 4, compiled with `--target-sm=80` (or equivalent), expected output element-wise equal to 16.0 within **1e-4 tolerance** (NOT 1e-3). Rationale: same MMA instructions, same f32 accumulator, same rounding — the only numeric difference is that `(x@W) + (x@A)@B*scale` accumulates into one register in the fused path vs. three independent stores in the unfused. Associativity in f32 affects the lowest-significance bits but stays well under 1e-4 for `x, W, A, B ~ O(1)` and batch/dim `~ 10^1`.

If 1e-4 fails, investigate before loosening — probable causes: missing f32→f16 conversion on an adapter operand, incorrect accumulator reset between main and epilogue MMA, or a register aliasing bug.

Add `build_5_kernel_count`: runs the fused compile, captures CUDA kernel launch count via an NVTX range or a runtime side-channel, asserts exactly 1 kernel launch per adapter site (vs. 3 for unfused).

**Acceptance:** both new tests pass. The 1e-5 sm_75 fallback test still passes. Total equivalence suite is 6 tests (Build 1-4 from B.2.1 + `build_4_fused` + `build_5_kernel_count`).

### Task 6 — Close-out

1. Memory file `project_wrga_milestone_b3.md` documenting the 5 task commits, register-budget analysis, tolerance decisions, known limitations.
2. Append to `MEMORY.md`.
3. Full regression (semantic, codegen lib + tests, flash_attention, nsl-cli e2e, wrga suites, cuda build, release build, clippy).
4. Do NOT merge — controlling session merges after subagent review.

## File structure

**Create:**
- `crates/nsl-codegen/src/matmul_mma.rs` — reusable MMA primitives (Task 1).
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — LoRA/IA³ epilogue PTX generator (Task 2).
- `crates/nsl-runtime/src/fused_adapter.rs` — two new FFIs (Task 4).

**Modify:**
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` — conditional emission in `synthesize_lora_adapted` + `synthesize_ia3_adapted` (Task 4).
- `crates/nsl-codegen/src/wrga_fusion.rs` — verification assertion (Task 3).
- `crates/nsl-codegen/src/compiler/entry_points.rs` — PTX kernel registration + embedding (Task 4).
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — extend `AdapterSite` with `fusion_decision` discriminant (Task 4).
- `crates/nsl-codegen/src/lib.rs` — `pub mod` declarations for new modules.

**Test:**
- `crates/nsl-codegen/src/matmul_mma.rs` — inline unit tests for MMA emission (Task 1).
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — inline golden-PTX tests (Task 2).
- `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` — activation-live verification (Task 3).
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — `build_4_fused` + `build_5_kernel_count` (Task 5).

## Out of scope

- **GatedLoRA epilogue fusion.** Needs a PTX sigmoid impl (Taylor approximation or lookup table). Becomes B.3.1 or later.
- **Multi-pass epilogue for `rank > 16`.** Hard error at compile time; follow-up milestone.
- **`nsl_tensor_matmul` (Cutlass path) refactor.** Untouched. B.3 adds new FFIs, doesn't modify existing.
- **FA backward MMA refactor.** `flash_attention.rs` MMA helpers are copied-then-parameterized into `matmul_mma.rs`; FA itself remains untouched. Unification is a separate follow-up.
- **Non-Ampere tensor cores (sm_90+ WGMMA).** sm_90 is reachable via the existing FA WGMMA path but B.3 stays on Ampere m16n8k16.

## Risk register

1. **MMA helper parameterization.** Extracting FA's helpers into a generic form may surface hidden FA-specific assumptions (swizzle patterns tied to attention SMEM layout, online-softmax-flavored accumulator init). Mitigation: Task 1 unit tests exercise the extracted helpers on a plain matmul input with no attention-adjacent state. If a hidden coupling surfaces, copy-paste-and-adapt rather than force unification.

2. **Register budget actually >16 rank.** The analysis above estimates; the PTX assembler (`ptxas`) produces the authoritative count. After Task 2 ships a first kernel, run `ptxas -v` and compare. If the actual count is lower than estimated (more headroom), tighten the error to the real max. If higher (less headroom), the ceiling drops and we document `rank ≤ 8` or whatever the reality is.

3. **Epilogue accumulator reset.** The main `x @ W` accumulator must NOT be cleared before the epilogue adds `(x@A)@B * scale`. Common bug: fresh-zero both accumulators. Mitigation: golden-PTX test explicitly checks there's no redundant accumulator zero between main matmul and epilogue; runtime equivalence test (Build 4 fused) catches any real output divergence.

4. **Associativity tolerance.** If `build_4_fused` fails at 1e-4, the diff amount is diagnostic: ~1e-3 → likely a register-aliasing or double-count bug; ~1.0+ → scale factor wrong or epilogue not firing; exactly identical to unfused → test didn't actually exercise the fused path (check kernel-count assertion).

5. **PTX kernel embedding + dedup.** Each adapter site would naively get its own synthesized PTX, but dedup by `(m, n, k, rank, target_sm)` lets sites share kernels. For this dedup to be effective when sites have **different alpha values**, `scale = alpha/rank` MUST be a kernel launch parameter (`.param .f32`), not a PTX literal. If scale were a literal, each distinct alpha value forks a new kernel even when `(m, n, k, rank)` matches — losing dedup for any model where users tune alpha per layer. The launch-parameter approach costs one extra register load at kernel entry and enables full dedup. Task 4's `synthesize_fused_lora_ptx` emits `.param .f32 scale` and `nsl_adapter_fused_lora_matmul` passes it at launch time. On NSLCoder with 48 LoRA sites and two distinct rank values, dedup collapses 48 kernels to 2.

6. **Windows stack budget.** B.2.1's 16 MB main-thread bootstrap holds today. B.3 adds PTX-generation recursion depth (synthesize fn has nested loop emission). If tests regress with stack overflow, bump to 32 MB.
