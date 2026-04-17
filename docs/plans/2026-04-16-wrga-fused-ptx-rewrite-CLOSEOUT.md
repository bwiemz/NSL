# WRGA Fused-LoRA/IA³ PTX Rewrite — Milestone Close-Out

**Closed:** 2026-04-16
**Branch:** `feat/wrga-fused-ptx-rewrite`
**Spec:** [2026-04-16-wrga-fused-ptx-rewrite-design.md](2026-04-16-wrga-fused-ptx-rewrite-design.md)
**Plan:** [2026-04-16-wrga-fused-ptx-rewrite-plan.md](2026-04-16-wrga-fused-ptx-rewrite-plan.md)

## Close-out criteria (§5 of spec)

| # | Criterion | State |
|---|---|---|
| 1 | All 6 logical commits merged | ✓ (bundled across 18 task-level commits for bisect granularity) |
| 2 | FA v2 snapshot tests byte-identical after commit 1 | ✓ (17 pass / 6 pre-existing baseline failures unchanged) |
| 3 | `kernel_skeleton/tests/snapshots/*.snap` all green | ✓ (14 snapshots across 9 helpers × variants) |
| 4 | WRGA LoRA unit ptxas test green across 6 configs | ✓ |
| 5 | WRGA LoRA integration test green at 1e-4 under NSL_WRGA_FUSED_CUDA=1 | ✓ (`build_4_fused_real_launch`) |
| 6 | WRGA IA³ unit ptxas test green across 5 configs | ✓ |
| 7 | WRGA IA³ integration test green on both fixtures at 1e-4 | ✓ (`ia3_fixture_a_baseline`, `ia3_fixture_b_gamma_scaling`) |
| 8 | `build_4_fused_cuda_actually_fires` flipped to `#[cfg(feature="cuda")]`, count ≥ 1 | ✓ |
| 9a | `project_wrga_ptx_scaffolding_discovered.md` prepended retrospective + CLOSED marker | ✓ |
| 9b | `project_wrga_fused_ptx_rewrite.md` created with invariants | ✓ (12 invariants: 6 from spec + 6 discovered in Task D1) |
| 9c | MEMORY.md index updated | ✓ |

## Institutional lesson (for future PTX milestones)

> B.3 shipped PTX scaffolding that looked correct in string-pattern tests but was never validated against ptxas or real launches. The 2026-04-16 discovery found this; this milestone closes the gap. **Future PTX-emitting milestones must include ptxas validation from the first commit.**

## Surprises discovered during execution (worth recording)

Task D1's integration test (real cudarc launch on BUILD4 shape) surfaced 5 correctness bugs that unit ptxas validation could not catch:

1. **`a_tile` is a B-operand** in the epilogue MMA (x @ A), not an A-operand. The original C4 design loaded it with `emit_load_a_fragment_smem`; fixed by switching to `emit_load_b_fragment_smem` with col-major staging + per-lane `%smem_base_a_lane_u32`.
2. **Compile-time m-bound used prescan placeholder** (`config.m=1`), zeroing 15 of 16 rows at PTX-emit time. Fixed with runtime `setp.lt.u32` predicate per row so all `m_param` rows stage.
3. **NSL GPU tensors are f32, not f16.** Staging helpers originally used f16 stride (×2). Fixed to f32 stride (×4) + `cvt.rn.f16.f32` + `st.shared.b16`.
4. **Per-lane SMEM col base** needed for B-fragment alignment. Added `%smem_base_{w,a,b}_lane_u32 = base_u32 + (tid/4) * 32`.
5. **`cuCtxSynchronize()` return value was silently discarded** — masked a kernel-level misaligned-address fault as a later `cuMemcpyDtoH_v2` failure with misleading error. Fixed in `crates/nsl-runtime/src/cuda/mod.rs`.

All five invariants are recorded in [project_wrga_fused_ptx_rewrite.md](../../../../.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_fused_ptx_rewrite.md) as load-bearing invariants #7-#11 (plus #12 for the `cvt.rn.f16x2.f32` pack between epilogue MMAs, noted in the spec but confirmed during D1).

## Deferred follow-ups (unchanged from spec §5)

- **B.3.1** — GatedLoRA epilogue + PTX sigmoid (Taylor approximation vs. 256-entry LUT)
- **B.4 or later** — sm_90 WGMMA path, `ldmatrix.sync.aligned.m8n8.x4`, `cp.async.ca` staging overlap, multi-warp-per-tile, multi-tile γ staging, perf benchmarking
- **Deep `kernel_skeleton` refactor** — fusion-callback template pattern, once FA and WRGA are both stable
- **FA param-block migration** — FA still uses `%rd0..%rd9` numbered-pool param loads vs. WRGA's named `%rd_x`/`%rd_w`/etc; migrating FA to `emit_param_block` is a separate cleanup deferred out of this milestone

## Branch state summary

Commits from branch-base (`feat/wrga-cpu-gpu-test-fix` HEAD) to milestone close:
- 3 from the baseline branch: test-source fix, spec, plan
- 18 from the implementation: inventory, scaffolding, 7 extractions, ptxas infra, LoRA red-gate, 4 LoRA helpers, LoRA synthesizer rewrite, C marker, D integration, 4 IA³ tasks, F1 hardening flip, this close-out

Ready to merge as a single PR to main.
