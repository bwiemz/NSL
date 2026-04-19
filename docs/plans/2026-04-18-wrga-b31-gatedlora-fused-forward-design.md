# WRGA B.3.1 — Fused GatedLoRA Forward PTX — Design Spec

**Date:** 2026-04-18
**Status:** Design approved; implementation plan pending
**Predecessor:** WRGA fused-LoRA/IA³ PTX rewrite (2026-04-16) — `docs/plans/2026-04-16-wrga-fused-ptx-rewrite-CLOSEOUT.md`
**Successor:** WRGA B.3.2 (fused GatedLoRA backward) — deferred, see §3.4 for trigger conditions

---

## Preamble — Why This Milestone Exists

The 2026-04-16 WRGA rewrite shipped real ptxas-validated m16n8k16 kernels for LoRA and IA³ adapters. GatedLoRA — the third adapter type that B.2.1 introduced and B.3 scaffolded — was deferred because its PTX sigmoid computation was a distinct design problem (Taylor approximation vs. 256-entry lookup table vs. hardware approx path).

B.3.1 closes that deferral **forward-only**. The kernel computes:

```
y[i, j] = (x @ W)[i, j] + sigmoid(gate[j]) * ((x @ A) @ B)[i, j] * scale
```

where `gate` is a per-output-column f32 vector of shape `[output_dim]` initialized to zeros (per B.2.1's inject pass). Sigmoid applies element-wise to the gate vector and broadcasts across the output's row dimension.

Backward (gradient flow through GatedLoRA during training) stays on B.2.1's unfused adapter-triple path, where source-AD materializes the sigmoid's forward value on the tape and the backward `σ·(1-σ)` multiply is a trivial elementwise op with no kernel-launch overhead to eliminate. B.3.2 is the deferred milestone that would fuse the backward if and only if a measurement trigger demands it; §3.4 specifies the exact trigger.

**Institutional lesson inherited from 2026-04-16:** `project_wrga_ptx_scaffolding_discovered.md` retrospective pinned "Future PTX-emitting milestones must include ptxas validation from the first commit." B.3.1 honors this rule from commit 2 forward, and B.3.2's stub document (written by commit 6 of this milestone) propagates the rule explicitly.

---

## Section 1 — Architecture and Emitter Layout

### Two-layer refactor-and-extend against the existing rewrite

One new public synthesizer, one shared kernel-body extraction, one new PTX-level helper (`emit_sigmoid_approx_fused`), two new WRGA-specific helpers (`emit_gate_load_per_thread`, `emit_gatedlora_fold`). No changes to B.2.1 runtime adapter infrastructure, no changes to IA³'s synthesizer, no changes to FA v2.

### New public entry point

`synthesize_fused_gatedlora_ptx(config: &FusedGatedLoraConfig) -> String` in `crates/nsl-codegen/src/wrga_fused_ptx.rs`. `FusedGatedLoraConfig` mirrors `FusedLoraConfig` (site_id, m, n, k, rank, target_sm). Separate type so dispatch and kernel-dedup don't confuse GatedLoRA and LoRA kernel instances.

The existing AST rewrite in `wrga_adapter_rewrite.rs` already distinguishes `AdapterKind::GatedLora` from LoRA and IA³ (`synthesize_gatedlora_adapted` branch); no AST rewrite changes needed. This milestone adds the FUSED-PATH dispatcher that routes `AdapterKind::GatedLora + target_sm >= 80` to `synthesize_fused_gatedlora_ptx`, in parallel with the existing LoRA and IA³ fused dispatchers.

### Shared kernel body extraction

New private function `emit_fused_adapter_kernel_body(ptx, dims, target_sm, fold: FoldKind)` extracted from the existing LoRA synthesizer body. Contains: header, param block, SMEM decl, register pool, indexing, param loads, output-tile coords, accumulator init, main K-loop with interleaved epilogue, post-loop `(x@A) @ B` MMA, store. Exposes exactly one variation point — the fold step — parameterized by `FoldKind`:

```rust
#[allow(dead_code)]
pub enum FoldKind {
    /// LoRA: `main_accum += epi_final * scale`  (scalar scale uniform across tile)
    Scalar {
        scale: f32,
    },
    /// GatedLoRA: `main_accum += epi_final * sigmoid(gate[col]) * scale`  (per-column)
    PerColumnSigmoid {
        scale: f32,
        gate_ptr_param_name: &'static str,
        gate_load_phase: GateLoadPhase,
        partial_tile_mask: PartialTileMask,
    },
}

#[allow(dead_code)]
pub enum GateLoadPhase {
    /// Load gate at end of final main-K-iteration, before its `bar.sync`.
    /// Overlaps HBM load latency with the final main MMA.  Default.
    LastKIter,
    /// Load gate after the main K-loop completes, before the post-loop
    /// (x@A)@B MMA.  Used when k_iters == 1 (no last-iteration work to
    /// hide behind).  Automatically selected by the emitter; preserved
    /// as a variant so B.3.1's default-LastKIter isn't confused for a
    /// universal rule.
    PostLoop,
}

#[allow(dead_code)]
pub enum PartialTileMask {
    /// Mask fold contribution with `@%p_colN`; OOB gate reads are harmless
    /// because their fold output is discarded.  Cost: 2 setp + 2 predicated
    /// adds per thread.  Default.
    FoldResultMask,
    /// Pre-populate OOB gate SMEM slots with a sentinel (-20.0) such that
    /// `sigmoid(-20) ≈ 0` naturally zeros the contribution.  Saves fold
    /// predicates at the cost of an SMEM staging pass.  B.3.1 doesn't stage
    /// gate through SMEM, so this variant is unused.
    SentinelGate,
}
```

LoRA's existing `synthesize_fused_lora_ptx` is **refactored** (not rewritten) to call `emit_fused_adapter_kernel_body(ptx, dims, Sm80, FoldKind::Scalar { scale })`. The refactor preserves byte-identical PTX; the 6 existing LoRA ptxas tests and the D1 integration test are the gates (both are ptxas-compile + real-launch, not snapshot — see §3.2 commit 1 acceptance).

### File map delta from the 2026-04-16 close-out

- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — 1 new public fn (`synthesize_fused_gatedlora_ptx`), 1 new struct (`FusedGatedLoraConfig`), 1 extracted private fn (`emit_fused_adapter_kernel_body`), 3 new enums (`FoldKind`, `GateLoadPhase`, `PartialTileMask`); existing `synthesize_fused_lora_ptx` refactored to call the shared body.
- `crates/nsl-codegen/src/wrga_kernel_helpers.rs` — 3 new helpers: `emit_sigmoid_approx_fused`, `emit_gate_load_per_thread`, `emit_gatedlora_fold`.
- `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs` — 10 new ptxas tests (§3.1).
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — 4 new integration fixtures (§3.1).
- `docs/plans/2026-04-18-wrga-b31-gatedlora-CLOSEOUT.md` — close-out note (commit 6).
- `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md` — deferred-trigger stub (commit 6), §3.4 verbatim.
- WRGA paper (existing) — Appendix B gets a new subsection for the approximation-helper testing rule (commit 6, §3.3).

### Commit sequence (6 commits, strict TDD)

See §3.2 for per-commit acceptance checklists.

1. **`refactor(wrga): extract emit_fused_adapter_kernel_body; LoRA calls it with FoldKind::Scalar`** — byte-identical LoRA PTX; D1 + all non-ignored LoRA tests green. Register-collision audit and `%n_real` promotion gates named in acceptance.
2. **`feat(wrga): emit_sigmoid_approx_fused + structural assertions`** — PTX sigmoid helper with 3 structural tests; narrowed mul-line constant check.
3. **`test(wrga): GatedLoRA ptxas validation (red, 10 configs)`** — stub `synthesize_fused_gatedlora_ptx`; 10 red configs established.
4. **`feat(wrga): synthesize_fused_gatedlora_ptx against kernel_body + PerColumnSigmoid`** — 10 red flip green.
5. **`test(wrga): GatedLoRA integration fixtures (3 saturation @ 1e-4 + 1 mid-range @ 1e-3)`** — 4 integration fixtures; mid-range inline ULP comment.
6. **`docs(wrga): B.3.1 close-out + B.3.2 stub + WRGA paper Appendix B update`** — memory files, close-out doc, stub doc, paper appendix rule B.2.

### What stays untouched

- B.2.1 runtime adapter side-table, init emission, forward rewrite, seed route-through
- Kernel handle routing, dedup registry, `try_cuda_launch_fused_*`, arg marshaling
- AST-level `synthesize_gatedlora_adapted` rewrite in `wrga_adapter_rewrite.rs`
- WGGO → WRGA consumer wiring
- FA v2 extraction helpers in `kernel_skeleton/`
- IA³ synthesizer
- All existing LoRA tests (pass-through regression gates)

---

## Section 2 — PTX-Level Specifics

### PTX sigmoid emission (`emit_sigmoid_approx_fused`)

Four PTX instructions, one scratch register. Caller passes input/output register names as `&str`; scratch `%sig_tmp` is declared in `emit_lora_register_pool` extension (see Register Budget below).

```
    mul.f32  %sig_tmp, <input>, 0fBFB8AA3B;   // * -log2(e) = -1.4426950408
    ex2.approx.f32 %sig_tmp, %sig_tmp;         // 2^(-x * log2e) = e^(-x)
    add.f32  %sig_tmp, %sig_tmp, 0f3F800000;   // + 1.0
    rcp.approx.f32 <output>, %sig_tmp;         // 1 / (1 + e^(-x))
```

**Hex constants (verified):**
- `0fBFB8AA3B` = `-log₂(e) = -1.4426950408889634` as f32. Sign bit 1, exponent `0x7F = 127`, mantissa `0x38AA3B`. DO NOT confuse with `0f3FB8AA3B` (positive log₂(e), would silently compute `sigmoid(-x)` as `sigmoid(x)` — a wrong-but-plausible result that can pass loose tolerances). Also DO NOT confuse with `0f3FBE2FB9` (a historical bad value drafted in an earlier spec revision; decodes to +1.486, not -1.443).
- `0f3F800000` = `1.0` as f32. Sign 0, exponent 127, mantissa 0.

**Structural test (commit 2) asserts positively on the correct mul-line content, not just negatively on known-bad:**

```rust
#[test]
fn emit_sigmoid_approx_fused_has_correct_log2e_constant() {
    let mut ptx = String::new();
    emit_sigmoid_approx_fused(&mut ptx, "%in", "%out");
    let mul_line = ptx.lines()
        .find(|l| l.contains("mul.f32") && l.contains("%sig_tmp"))
        .expect("expected mul.f32 %sig_tmp line");
    assert!(
        mul_line.contains("0fBFB8AA3B"),
        "mul line must use -log2(e) = 0fBFB8AA3B; got: {mul_line}"
    );
    // Belt-and-suspenders: reject the historical bad value.
    assert!(
        !ptx.contains("0f3FBE2FB9"),
        "regression to spec-draft bad positive constant detected"
    );
    // Sanity: exactly one ex2.approx and one rcp.approx in the emission.
    assert_eq!(ptx.matches("ex2.approx.f32").count(), 1);
    assert_eq!(ptx.matches("rcp.approx.f32").count(), 1);
    assert!(ptx.contains("0f3F800000"), "missing +1.0 constant");
}
```

### Gate parameter and prolog load

**Parameter block addition under `FoldKind::PerColumnSigmoid`:** the param block emitted by `emit_fused_adapter_kernel_body` includes `.param .u64 gate_ptr` in addition to the LoRA-shared `x_ptr, w_ptr, a_ptr, b_ptr, scale, y_ptr`. This param declaration is conditional on the `FoldKind` variant — `Scalar` (LoRA) keeps the 6-param list unchanged; `PerColumnSigmoid` (GatedLoRA) adds `gate_ptr` as the 7th param.

**Prolog load (REQUIRED before the main K-loop):** `emit_fused_adapter_kernel_body`'s prolog emits under the `PerColumnSigmoid` branch:

```rust
if matches!(fold, FoldKind::PerColumnSigmoid { .. }) {
    emit_ld_param_u64(ptx, "%rd_gate", "gate_ptr");
}
```

producing the single PTX line:

```
    ld.param.u64 %rd_gate, [gate_ptr];
```

This uses the existing `emit_ld_param_u64` helper from `kernel_skeleton::params` (shipped 2026-04-16). After this prolog load, `%rd_gate` holds the HBM base address of the gate vector; `emit_gate_load_per_thread` (below) consumes it.

Without this prolog load, commit 3's red state is ambiguous (undefined-reference ptxas failure could mask "stub returns empty" vs "prolog missing register init"). This section is the contract the prolog emission MUST satisfy.

### Per-thread gate load (`emit_gate_load_per_thread`)

Each thread owns exactly 2 output columns per the m16n8k16 D-fragment layout (`main_accum0/2` at `col_base_in_tile + 0`, `main_accum1/3` at `col_base_in_tile + 1`). Gate load scope matches: thread t loads EXACTLY 2 f32 values, not the full gate vector.

Emission (called after `emit_matmul_mma_lane_init` sets up `%r5 = col_base_in_tile = (tid%4)*2`, AND after the prolog load above populates `%rd_gate`):

```
    // Per-thread gate load — 2 cols per thread
    add.u32  %r_gate, %col_base, %r5;      // global col = col_base + col_base_in_tile
    shl.b32  %r_gate, %r_gate, 2;          // * 4 bytes (f32)    TODO(gate-dtype): shift=1 if f16
    cvt.u64.u32 %rd_gate_addr, %r_gate;
    add.u64  %rd_gate_addr, %rd_gate, %rd_gate_addr;
    ld.global.f32 %gate0, [%rd_gate_addr];
    ld.global.f32 %gate1, [%rd_gate_addr + 4];
```

After the load, caller immediately applies sigmoid in-place:

```rust
emit_sigmoid_approx_fused(ptx, "%gate0", "%gate0");
emit_sigmoid_approx_fused(ptx, "%gate1", "%gate1");
```

### Gate-load scheduling — `LastKIter` default

Schedule the gate load + sigmoid compute at the **end of the final main-K-iteration**, after that iteration's MMAs and before its `bar.sync`. The MMA latency hides the HBM load. Using nested `if let` for portability across Rust editions:

```rust
for k_tile in 0..k_iters {
    emit_stage_x_tile(...);
    emit_stage_w_tile(...);
    emit_stage_a_tile(...);
    ptx.push_str("    bar.sync 0;\n");
    emit_load_a_fragment_smem(&main_a_frag, "%x_tile_base", 32);
    emit_load_b_fragment_smem(&main_b_frag, "%w_tile_base", 16);
    emit_load_a_fragment_smem(&epi_a_frag, "%a_tile_base", 32);
    emit_mma_instruction(&main_accum, &main_a_frag, &main_b_frag, &main_accum);
    emit_mma_instruction(&epi_interm, &main_a_frag, &epi_a_frag, &epi_interm);

    if k_tile == k_iters - 1 {
        if let FoldKind::PerColumnSigmoid {
            gate_load_phase: GateLoadPhase::LastKIter, ..
        } = fold {
            emit_gate_load_per_thread(ptx);
            emit_sigmoid_approx_fused(ptx, "%gate0", "%gate0");
            emit_sigmoid_approx_fused(ptx, "%gate1", "%gate1");
        }
    }

    ptx.push_str("    bar.sync 0;\n");
}
```

`PostLoop` variant is defined but **NOT auto-selected by B.3.1**. All shipped ptxas configs have `k_iters ≥ 1` for the x@A epilogue path, and B.3.1's synthesizer unconditionally emits `GateLoadPhase::LastKIter`. The `PostLoop` variant is preserved in the `FoldKind::PerColumnSigmoid` enum for a potential future B.3.x milestone that wants to benchmark load-phase alternatives; it has NO emission site in B.3.1's code. A later session that implements `PostLoop` emission would need to add the branch in `emit_fused_adapter_kernel_body` (post-loop, between the `(x@A)@B` MMA and the fold step) and set `gate_load_phase: PostLoop` at the `synthesize_fused_gatedlora_ptx` call site for configs where `k_iters == 1`.

Keeping the variant reserved (marked `#[allow(dead_code)]`) rather than removing it preserves the enum shape described in the FoldKind contract without committing B.3.1 to an untested code path.

### Partial-tile masking — `FoldResultMask`

For sub-MMA output tiles where `n < 8`, OOB gate HBM reads are NOT predicated at load time (predicating adds 2 setp-branches per thread and still leaves the OOB-read undefined). Instead, predicate the FOLD step so OOB column contributions never reach `main_accum`:

```
    // Per-thread fold: main_accum += epi_final * sigmoid(gate) * scale
    // mask OOB output cols so partial tile padding doesn't contribute.
    setp.lt.u32  %p_col0, %r5, %n_real;                // col_base_in_tile+0 < n_real
    add.u32      %r_col1, %r5, 1;
    setp.lt.u32  %p_col1, %r_col1, %n_real;            // col_base_in_tile+1 < n_real

    // main_accum0 (col 0, row 0), main_accum2 (col 0, row 8) → gate0, p_col0
    mul.f32      %fold_tmp, %epi_final0, %gate0;
    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;
    @%p_col0 add.f32 %main_accum0, %main_accum0, %fold_tmp;
    mul.f32      %fold_tmp, %epi_final2, %gate0;
    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;
    @%p_col0 add.f32 %main_accum2, %main_accum2, %fold_tmp;

    // main_accum1 (col 1, row 0), main_accum3 (col 1, row 8) → gate1, p_col1
    mul.f32      %fold_tmp, %epi_final1, %gate1;
    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;
    @%p_col1 add.f32 %main_accum1, %main_accum1, %fold_tmp;
    mul.f32      %fold_tmp, %epi_final3, %gate1;
    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;
    @%p_col1 add.f32 %main_accum3, %main_accum3, %fold_tmp;
```

Scale flows through via `%scale_reg`, loaded from a `.param .f32 scale` at prolog (identical to LoRA's kernel dedup discipline — kernel cache key is `(m, n, k, rank, target_sm)`; scale is a launch-time value not baked into PTX).

Belt-and-suspenders: the store step's existing `%n_real` / `%m_real` predicate in `emit_lora_store_output` is a SECOND bound check. Both fold-mask and store-mask must remain. Removing either allows OOB values to leak into y.

### Register budget additions for `PerColumnSigmoid`

Beyond LoRA's existing `emit_lora_register_pool` declarations:

- `.reg .f32 %gate0, %gate1;` — per-thread sigmoid(gate) values (persistent through fold)
- `.reg .f32 %sig_tmp;` — sigmoid scratch (reused by each sigmoid call)
- `.reg .f32 %fold_tmp;` — fold scratch (reused per main_accum element)
- `.reg .u64 %rd_gate;` — HBM pointer for gate param (loaded from `.param .u64 gate_ptr` at prolog)
- `.reg .u64 %rd_gate_addr;` — computed per-thread gate address
- `.reg .u32 %r_gate;` — gate global-col index scratch
- `.reg .u32 %r_col1;` — col_base_in_tile + 1 scratch for second predicate
- `.reg .pred %p_col0, %p_col1;` — fold-mask predicates

Total: 10 new registers. Well under any budget limit.

**Reused from LoRA's existing pool (no new declaration needed):**

- `%scale_reg` — already declared in `emit_lora_register_pool` and loaded from `.param .f32 scale` at LoRA's prolog. GatedLoRA's fold step reads it via the same mechanism. Confirmed during commit 1 prep via `grep -n '%scale_reg' crates/nsl-codegen/src/wrga_kernel_helpers.rs`; if the grep comes back empty (prior-session implementer used a different name), add `%scale_reg` to the additions list above as the 11th register and update the pre-audit grep list in the collision-prevention section below.

**Name-collision prevention (commit 1 pre-audit):** all 10 names must be zero-match in `grep -n '%fold_tmp\|%r_col1\|%p_col0\|%p_col1\|%sig_tmp\|%rd_gate\|%rd_gate_addr\|%r_gate\|%gate0\|%gate1' crates/nsl-codegen/src/wrga_*.rs`. If any match, ALL GatedLoRA additions prefix with `%gl_` (e.g., `%gl_fold_tmp`, `%gl_p_col0`); prefix convention noted in the extraction commit message.

**`%n_real` sourcing:** commit 1 verifies `%n_real` is declared in `emit_lora_register_pool`'s shared scope (not local to `emit_lora_store_output`). If currently local, commit 1 promotes it into the shared pool as a single-line addition before the extracted body can reference it. This is a byte-identical behavior change for LoRA (declaration moves; emission order within function body changes but PTX semantics identical; ptxas accepts both orderings; D1 integration test is the behavior-level gate).

### Correctness invariants (appended at milestone close)

Audit `project_wrga_fused_ptx_rewrite.md` for the current last invariant number (claimed #12 per 2026-04-16 close-out). B.3.1 appends four new invariants starting one past the actual last entry:

> **#13 — Gate load is per-thread, not warp-broadcast.** The m16n8k16 D-fragment distributes output columns so thread t owns cols `(tid%4)*2` and `+1`. `emit_gate_load_per_thread` MUST load exactly these 2 values per thread, not all 8. Warp-broadcast would 4× HBM gate traffic for no benefit since each thread consumes only the 2 values its D-fragment owns.

> **#14 — Gate-load `LastKIter` scheduling hides HBM latency.** The fused-body emitter schedules `emit_gate_load_per_thread + emit_sigmoid_approx_fused` at the end of the final K-iteration (before its `bar.sync`) so the sigmoid computation overlaps the final main MMA's latency. B.3.1 ships with `GateLoadPhase::LastKIter` unconditionally; all shipped ptxas configs have `k_iters ≥ 1` for the epilogue, so the LastKIter emission path covers the full test matrix. The `PostLoop` variant is reserved in the enum for a potential future milestone but has no emission site in B.3.1 (see §2 "Gate-load scheduling"). Moving the gate load to a post-loop emission site without simultaneously implementing a `PostLoop` branch in `emit_fused_adapter_kernel_body` silently regresses performance.

> **#15 — `FoldResultMask` is load-bearing for partial tiles.** OOB gate reads are NOT predicated at load time. The OOB column's fold contribution to `main_accum` is predicated via `@%p_colN`. The existing `%n_real` store-mask in `emit_lora_store_output` is the second bound check. Both masks must remain.

> **#16 — PTX sigmoid immediate constants pinned.** `0fBFB8AA3B = -log₂(e)` (NOT `0f3FB8AA3B` which is +log₂(e) and silently wrong). `0f3F800000 = 1.0`. Structural test in commit 2 pins these constants explicitly and rejects the historical bad value `0f3FBE2FB9` as belt-and-suspenders.

---

## Section 3 — Test Matrix, Commits, Close-Out, B.3.2 Stub

### 3.1 Test matrix

#### Unit ptxas layer — 10 new tests

All 10 tests invoke the public `synthesize_fused_gatedlora_ptx(FusedGatedLoraConfig { ... })` entry point. `emit_fused_adapter_kernel_body` and other internal emitters are tested transitively through the public API — never directly. This prevents test coupling to internal refactor churn.

**Reused LoRA shapes (6 configs) — validate `FoldKind::PerColumnSigmoid` against the shape matrix that `FoldKind::Scalar` already covers:**

| `(m, n, k, rank)` | Purpose |
|---|---|
| `(16, 8, 16, 16)` | Canonical Ampere shape |
| `(16, 8, 32, 4)`  | Multi-K-iter + rank-pad |
| `(1, 8, 8, 2)`    | Sub-MMA m and k — smallest |
| `(4, 8, 8, 2)`    | Hardening-test shape equivalent |
| `(32, 16, 64, 8)` | Multi-tile output grid |
| `(16, 8, 8, 16)`  | K-padding + rank == MMA-k no-pad path |

These 6 configs each spawn a test invoking `synthesize_fused_gatedlora_ptx` with `gate = zeros` (the simplest gate configuration; shape coverage is the goal, not gate behavior).

**GatedLoRA-distinctive (4 configs) — validate properties only meaningful under `PerColumnSigmoid`:**

| `(m, n, k, rank)` | Gate pattern | Purpose |
|---|---|---|
| `(16, 8, 16, 16)` | `gate = zeros([8])`       | Broadcast `sigmoid(0) = 0.5` validity across all threads |
| `(16, 8, 16, 16)` | `gate = alternating ±30`   | Per-column saturation; validates `ex2.approx` handles saturation range |
| `(16, 8, 8, 16)`  | `gate = zeros([8])`       | Sub-MMA K path + rank == MMA-k no-pad interaction |
| `(16, 13, 16, 4)` | `gate = uniform [-2, 2] on first 13 cols` | **Partial-n output (second tile n=5) + multi-tile grid + non-trivial sigmoid mid-range values + rank/dim mismatch.** The ONLY config that actually evaluates `FoldResultMask`'s `%p_col0`/`%p_col1` predicates false at runtime. With n=13, tile 0 is full (cols 0-7) and tile 1 is partial (cols 8-12 valid, cols 13-15 OOB). Without this config the predicated-add machinery is emitted but never triggered — a broken `FoldResultMask` implementation would pass every other test and produce correct output. |

**Total: 10 new ptxas tests.** The 6 existing LoRA-Scalar tests are regression gates (must stay green across commits 1 and 4), not B.3.1's add-count.

#### Integration fixture layer — 4 new fixtures

Source model shape across all 4 fixtures:
- `W = ones([8, 8])` → `x@W = 8` per element (the adapter-independent base contribution)
- `x = ones([4, 8])`
- `A = ones([8, 2])`, `B = ones([2, 8])` → `x @ A @ B = 16` per element (raw adapter delta)
- `alpha = 2, rank = 2` → `scale = 1.0`
- `y[i, j] = 8 + sigmoid(gate[j]) * 16 * 1.0`

Fixtures vary only in gate values:

| Fixture | Gate | Sigmoid | Expected y | Tolerance |
|---|---|---|---|---|
| **A — baseline** | `zeros([8])` | `0.5` exactly | **16.0** | 1e-4 |
| **B — positive saturation** | `ones([8]) * 30` | `1.0` exactly | **24.0** | 1e-4 |
| **C — negative saturation** | `ones([8]) * (-30)` | `0.0` exactly | **8.0** | 1e-4 |
| **D — mid-range curve** | `ones([8]) * 1.0` (exact f32) | `≈ 0.7310586` | **≈ 19.697** | **1e-3** |

**Fixture C's 8.0 expected value is specifically diagnostic:** the base `x@W = 8` must pass through unchanged when the adapter contribution zeros out. If a broken kernel's fold predicate path leaks anything into main_accum when `sigmoid(gate) ≈ 0`, the test fails at 1e-4.

**Each fixture's docstring MUST state α, r, and scale explicitly** so a reader can re-derive the expected y without reconstructing from source:

```rust
// Fixture A — baseline. alpha=2, rank=2, scale=alpha/rank=1.0.
// x=ones([4,8]), W=ones([8,8]), A=ones([8,2]), B=ones([2,8]), gate=zeros([8]).
// y[i,j] = (x@W)[i,j] + sigmoid(gate[j]) * ((x@A)@B)[i,j] * scale
//        = 8         + sigmoid(0)       * 16              * 1.0
//        = 8 + 0.5 * 16
//        = 16.0 per element.
```

#### Mid-range Fixture D — REQUIRED inline ULP comment

Fixture D carries this comment verbatim:

```rust
// Tolerance 1e-3 (vs 1e-4 on A/B/C): gate=1.0f exercises the actual
// ex2.approx + rcp.approx sigmoid curve shape between saturation points,
// not exact values.
//
// Error propagation at rank=2 accumulator magnitudes:
//   ex2.approx worst-case ULP:      ~2 ULP (~5e-8 at input -1.4427)
//   rcp.approx worst-case ULP:      ~1 ULP (~6e-8 at input ≈1.368)
//   Combined via 1/(1+e) derivative: ~9e-8 on sigmoid output
//   Through fold at rank=2:          ~5e-6 per element
//   Main-path MMA noise floor:       ~2e-4 (dominates)
//   1e-3 tolerance:                  ~5x over main MMA, ~200x over sigmoid
//
// Bug class this fixture catches: a sign-based-stub kernel emitting
// `select(x > 0, 1.0, select(x < 0, 0.0, 0.5))` passes every saturation
// fixture (A/B/C) but returns 0.5 here — wrong by |0.5 - 0.7310586| ≈ 0.2311
// on the sigmoid value.
//
//   Correct y:   (x@W = 8) + 0.7310586 · 16 · 1.0 ≈ 19.697
//   Stub   y:   (x@W = 8) + 0.5       · 16 · 1.0 = 16.0
//                                                   ^^^^
//   The (x@W = 8) base contribution is identical in both paths and cancels.
//   y-level error = |19.697 − 16.0| = 3.697, driven entirely by the
//   adapter-path factor 16 · (0.7311 − 0.5) ≈ 3.697. This is ~3700× the
//   1e-3 tolerance — fixture fails loudly.
//
// This is the ONLY fixture in the B.3.1 matrix that discriminates
// "emitted rcp.approx(1 + ex2.approx(-x * log2e))" from "silently stubbed."
```

The rcp.approx input is `1 + e^(-1) ≈ 1.3679`, not 1.731 (which would be `1 + e^(+1)`).

### 3.2 Commit sequence — per-commit acceptance checklists

**Commit 1 — refactor.** `refactor(wrga): extract emit_fused_adapter_kernel_body; LoRA calls it with FoldKind::Scalar`

- [ ] `cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda lora_ptx_validates` — 6/6 LoRA tests green. **Gate type stated in commit message**: "Pinned as `ptxas-compile` semantic validation — not snapshot comparison. Byte-identical PTX not required; ptxas-accepted PTX is."
- [ ] `cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1` — all existing WRGA tests green including `build_4_fused_real_launch`, `build_4_fused_cuda_actually_fires`, IA³ fixtures A and B. **Gate type stated in commit message**: "The numerical + launch-counter tests are the behavior-level byte-identity gate; if y drifts or gpu-launch-count falls to 0, the refactor regressed."
- [ ] Register-collision grep pre-audit performed on the 10 GatedLoRA-addition names; zero matches, OR matches found and `%gl_` prefix applied uniformly. Result stated in commit message.
- [ ] `%n_real` audit: promoted to shared pool if currently local to `emit_lora_store_output`. Promotion stated in commit message.
- [ ] **Internal-caller stability check**: `grep -rn 'emit_lora_register_pool\|emit_lora_store_output\|emit_lora_output_tile_coords\|emit_lora_tile_bases' crates/ --include='*.rs'` before and after the refactor; result lists must be identical modulo the new internal caller inside `emit_fused_adapter_kernel_body`. Catches "I moved this function but something imported it from an unexpected place" bugs.
- [ ] No changes to `synthesize_fused_ia3_ptx` or any IA³-specific helper.

**Commit 2 — sigmoid helper.** `feat(wrga): emit_sigmoid_approx_fused + structural assertions`

- [ ] `emit_sigmoid_approx_fused` added to `wrga_kernel_helpers.rs`.
- [ ] Structural assertion test (§2 above) passes. Critical: positive assertion on `0fBFB8AA3B` in the mul-line, belt-and-suspenders negative on `0f3FBE2FB9`, count assertions on `ex2.approx.f32` and `rcp.approx.f32` exactly once each.
- [ ] `emit_gate_load_per_thread` added with structural test asserting 2 `ld.global.f32` emissions + correct per-thread address arithmetic (`add.u32 %r_gate, %col_base, %r5` + `shl.b32 %r_gate, %r_gate, 2`).
- [ ] `emit_gatedlora_fold` added with structural test asserting predicated adds on `main_accum0/1/2/3` with correct mask wiring (`main_accum0,2` gated by `%p_col0`, `main_accum1,3` gated by `%p_col1`).

**Commit 3 — red test.** `test(wrga): GatedLoRA ptxas validation (red, 10 configs)`

- [ ] `FusedGatedLoraConfig` struct added to `wrga_fused_ptx.rs` (public API).
- [ ] `synthesize_fused_gatedlora_ptx` stub added returning empty or trivially-invalid PTX.
- [ ] 10 new tests added to `wrga_fused_ptx_ptxas.rs`, invoking the public entry point.
- [ ] All 10 new tests FAIL against the stub. Commit message records the failure mode (e.g., "all 10 panic with 'nvcc rejected PTX: empty entry'").
- [ ] Existing 6 LoRA-Scalar tests still pass.

**Commit 4 — GatedLoRA emitter.** `feat(wrga): synthesize_fused_gatedlora_ptx against kernel_body + PerColumnSigmoid`

- [ ] All 10 new ptxas tests flip GREEN.
- [ ] Existing 6 LoRA-Scalar tests still pass (no regression).
- [ ] Existing `wrga_adapter_runtime_equivalence` tests still pass (LoRA and IA³ byte-identity-of-behavior preserved).

**Commit 5 — integration fixtures.** `test(wrga): GatedLoRA integration fixtures (3 saturation @ 1e-4 + 1 mid-range @ 1e-3)`

- [ ] 4 new `#[cfg(feature = "cuda")]`-gated tests added to `wrga_adapter_runtime_equivalence.rs`.
- [ ] Fixture A: `y = 16.0 ± 1e-4` under `NSL_WRGA_FUSED_CUDA=1`.
- [ ] Fixture B: `y = 24.0 ± 1e-4`.
- [ ] Fixture C: `y = 8.0 ± 1e-4`.
- [ ] Fixture D: `y ≈ 19.697 ± 1e-3`, expected value computed via CPU reference sigmoid (not hardcoded literal).
- [ ] Fixture D carries the full inline ULP-derivation comment from §3.1 verbatim.
- [ ] `[nsl-gpu-launch-count] ≥ 1` asserted on each fixture (proves fused PTX fired, not CPU fallback).
- [ ] Each fixture's docstring states α, r, resulting scale, and the y computation from first principles.

**Commit 6 — close-out.** `docs(wrga): B.3.1 close-out + B.3.2 stub + WRGA paper Appendix B update`

- [ ] `docs/plans/2026-04-18-wrga-b31-gatedlora-CLOSEOUT.md` created with all acceptance checks completed.
- [ ] `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md` created with §3.4 content verbatim.
- [ ] `project_wrga_fused_ptx_rewrite.md` appended with invariants #13–#16 (actual numbering confirmed during commit 6 prep against current last entry).
- [ ] `MEMORY.md` index entry added linking the close-out.
- [ ] **WRGA paper Appendix B updated** with new subsection "B.2 Approximation-based activation helper testing rule" containing the institutional rule from §3.3 below. Paper path located during commit 6 prep; acceptance check is "Appendix B now contains subsection B.2 with the approximation-helper rule."

### 3.3 Close-out criteria and institutional-memory promotion

**B.3.1 is shipped when:**

1. All 6 commits landed on a single feature branch, PR-merged to main (or equivalent per user's branching preference).
2. 10 new ptxas tests green on CUDA machines; 4 new integration fixtures pass at their prescribed tolerances.
3. All pre-existing LoRA, IA³, FA v2, and non-WRGA regression gates green (same baseline preserved as after the 2026-04-16 close-out, including the 6 pre-existing CSHA Tier C `.snap` drifts and 1 pre-existing lib failure on `flash_attention.rs`).
4. Invariants #13–#16 appended to `project_wrga_fused_ptx_rewrite.md`.
5. B.3.2 stub document written per §3.4.
6. WRGA paper Appendix B updated with subsection B.2 (institutional rule below).

**Institutional rule promoted to WRGA paper Appendix B.2 (commit 6):**

> **B.2 — Approximation-based activation helper testing rule.** Any approximation-based activation helper (sigmoid, tanh, GELU, softmax, and related) emitted as a dedicated PTX helper MUST have at least one mid-range integration fixture whose tolerance is set such that a sign-based-stub implementation (one that returns a small set of constants based on sign/zero of the input) would FAIL the fixture. Pure-saturation fixtures (testing only inputs that produce exact-constant outputs like 0.0, 0.5, 1.0) are necessary but not sufficient; they are unable to distinguish "the approximation is emitted correctly" from "the helper is silently stubbed." Discovered during WRGA B.3.1 tolerance design (2026-04-18); generalizes beyond sigmoid and beyond WRGA.

### 3.4 B.3.2 deferred-trigger stub — verbatim content

Create `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md` with this content verbatim (subject to dates-and-paths adjustment at commit 6 prep):

```markdown
# WRGA B.3.2 — Fused GatedLoRA Backward (DEFERRED)

**Status:** deferred pending measurement trigger. Do NOT schedule this
milestone without satisfying the trigger below.

## Measurement-gated promotion trigger

Promote B.3.2 from deferred to scheduled **only when** a GatedLoRA
training benchmark at seq=2048, rank=16 shows backward time materially
larger than forward. Specifically: if `backward_time > 2.5 × forward_time`
on a representative workload (Llama-3-8B-style, batch × seq = 65k tokens
per microbatch, sm_80 or sm_90), schedule B.3.2.

If backward stays within 1.5× to 2× forward and remains matmul-bound
(profiler confirms ≥ 80% of backward time in matmul kernels rather than
elementwise ops), the unfused adapter-triple backward path is adequate;
**B.3.2 remains deferred indefinitely**. Elementwise sigmoid backward
(`σ(x)·(1−σ(x))` via the retained forward-tape value) is already cheap
under source-AD — there is no kernel-launch overhead to eliminate.

## Inherited institutional-memory rules

B.3.2, when scheduled, inherits the institutional lesson from the
2026-04-16 WRGA PTX scaffolding retrospective (see
`project_wrga_ptx_scaffolding_discovered.md`): **ptxas validation MUST
appear in B.3.2's first real-implementation commit, not deferred to a
later test layer.** The commit that introduces the backward PTX emitter
is the same commit that introduces its ptxas unit sweep. This rule is
non-negotiable; it is the WRGA retrospective's load-bearing institutional
correction. B.3.1 further promoted this to the WRGA paper's Appendix B.2
as a generalized rule covering all approximation-based PTX helpers.

## Four-layer test discipline inheritance

B.3.2 MUST ship with all four test layers green at milestone close:

1. **Skeleton snapshots** — any new helpers gated by per-variant pinned
   snapshots (or structural assertions for simple helpers; see B.3.1's
   convention).
2. **Unit ptxas** — backward kernel PTX validated across the same shape
   matrix as forward (6 reused LoRA shapes + any backward-distinctive
   configs), feeding each emission to `cuModuleLoadData` or `ptxas`.
3. **Integration numerical** — gradient correctness verified against
   finite-difference reference at 1e-3 tolerance (looser than forward's
   1e-4 because finite-difference step-size trades off against
   condition-number noise; set the step size and justify the tolerance
   inline at the fixture).
4. **E2E launch-counter** — `backward_fused_cuda_actually_fires` parallel
   to the forward hardening test, asserting `[nsl-gpu-launch-count] ≥ 1`
   during a full forward+backward pass.

## Scope boundary

B.3.2 covers ONLY fused backward for GatedLoRA. Fused backward for LoRA
and IA³ is a separate milestone (B.3.3 or similar) if demanded by the
same measurement trigger applied to those adapter types. Do not conflate.
```

---

## Dependencies and Prerequisites

This milestone builds on:

- **WRGA fused-LoRA/IA³ PTX rewrite** (shipped 2026-04-16) — `synthesize_fused_lora_ptx`, `wrga_kernel_helpers.rs`, `kernel_skeleton/`, `ptxas_validation.rs`, and the four-layer test discipline.
- **B.2.1 runtime adapter infrastructure** — `@adapter(type=gated_lora, ...)` decorator, `gate_<site>` side-table allocation + zeros init, `synthesize_gatedlora_adapted` AST rewrite, seed route-through for `m.gate_<site> = full([...], value).to(cuda)` assignments.
- **CUDA 13.x / sm_80+** — `ex2.approx.f32` and `rcp.approx.f32` are Ampere baseline; PTX 7.0 sufficient.

## Branching strategy

6 commits land on a single feature branch `feat/wrga-b31-gatedlora-fused-forward` (or similar), branched from the current `main` (post-2026-04-16-merge). PR targets main.

## Spec self-review notes

- Hex constants audited: `0fBFB8AA3B` appears everywhere `-log₂(e)` is referenced (4 occurrences). `0f3FBE2FB9` appears 3 times, all as historical-bad-value negative references. `0f3F800000 = 1.0` appears in §2 sigmoid emission.
- Test count reconciled: 10 new ptxas tests (6 reused shapes × PerColumnSigmoid + 4 GatedLoRA-distinctive); 4 new integration fixtures; 6 commits; 4 new invariants (#13–#16 pending numbering audit).
- Fixture math verified: A=16.0, B=24.0, C=8.0, D≈19.697; each = `(x@W = 8) + sigmoid(gate) · 16 · 1.0`.
- `rcp.approx` input corrected: `1 + e^(-1) ≈ 1.3679` at gate=1.0f, NOT 1.731.
- B.3.2 stub: 3 required elements (trigger / retrospective reference / four-layer inheritance) present verbatim in §3.4.
- Institutional-memory promotion mechanism named explicitly (Option A: commit 6 appends to WRGA paper Appendix B.2).
- Commit 1's byte-identity gate type stated explicitly (ptxas-compile semantic validation + behavior-level numerical/launch-counter tests, NOT snapshot comparison).
- Register-collision pre-audit gate listed in commit 1 acceptance.
- `%n_real` promotion audit gate listed in commit 1 acceptance.
- All 10 ptxas tests invoke the public `synthesize_fused_gatedlora_ptx` entry point (internal emitter functions tested transitively).
- `let_chains` dependency avoided via nested `if let` for Rust-edition portability.
- All enum variants get `#[allow(dead_code)]` since `GateLoadPhase::PostLoop` and `PartialTileMask::SentinelGate` are not emitted by B.3.1.

### Fixes applied in 2026-04-18 review round

Six review findings from the approved design walkthrough addressed inline:

1. **Prolog `%rd_gate` load made explicit.** Section 2 now names the `.param .u64 gate_ptr` param addition (conditional on `FoldKind::PerColumnSigmoid`) and the prolog `ld.param.u64 %rd_gate, [gate_ptr]` emission via `emit_ld_param_u64`. Without this, commit 3's red state would be ambiguous (undefined-reference ptxas failure masking "stub returns empty").
2. **`%scale_reg` reuse documented.** Section 2's register budget explicitly notes `%scale_reg` is reused from LoRA's existing pool; commit 1 prep confirms via grep; if the grep fails, `%scale_reg` is added as the 11th register.
3. **Internal-caller stability check added to commit 1 acceptance.** `grep -rn` on internal emitter function names before/after refactor must yield identical results modulo the new internal caller inside `emit_fused_adapter_kernel_body`.
4. **`PostLoop` auto-selection soft-worded.** B.3.1 ships with `GateLoadPhase::LastKIter` unconditionally. The `PostLoop` variant is reserved in the enum (marked `#[allow(dead_code)]`) but has no emission site in B.3.1. Invariant #14 reworded to reflect this.
5. **Fixture D comment clarified with explicit base-cancellation.** Comment now shows correct-y and stub-y side by side, makes the `(x@W = 8)` cancellation visible, and documents the y-level error as `~3700× the 1e-3 tolerance` so the "fails loudly" framing is quantified.
6. **`(16, 13, 16, 4)` partial-n config swapped in.** Replaces the previous `(16, 16, 16, 4)` multi-tile-but-no-partial-n config. n=13 means first tile is n=8 full, second tile is n=5 partial. This is the ONLY config in the matrix that actually evaluates `FoldResultMask`'s `%p_col0`/`%p_col1` predicates false at runtime. Without it the predicated-add machinery is emitted but never triggered; a broken `FoldResultMask` would pass every other test.
