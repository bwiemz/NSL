# WRGA B.3.1 Fused GatedLoRA Forward PTX -- Milestone Close-Out

**Closed:** 2026-04-18
**Branch:** `feat/wrga-b31-gatedlora-fused-forward`
**Spec:** [2026-04-18-wrga-b31-gatedlora-fused-forward-design.md](2026-04-18-wrga-b31-gatedlora-fused-forward-design.md)
**Plan:** [2026-04-18-wrga-b31-gatedlora-fused-forward-plan.md](2026-04-18-wrga-b31-gatedlora-fused-forward-plan.md)

## Close-out criteria (spec §3.3) -- state at close

| # | Criterion | State |
|---|---|---|
| 1 | All 6 logical commits merged (expanded to 7 with infrastructure) | v |
| 2 | 10 new ptxas tests green on CUDA (6 LoRA shapes x PerColumnSigmoid + 4 GatedLoRA-distinctive) | v |
| 3 | 4 new integration fixtures pass at prescribed tolerances | v (A/B/C at 1e-4, D at 1e-3) |
| 4 | All pre-existing LoRA/IA3/FA regression gates green | v (11 LoRA+IA3 ptxas preserved, fa_v2 6 pre-existing failures unchanged, 1 pre-existing a23 lib failure unchanged) |
| 5 | Memory file invariants #13-#16 appended (plus #17-#18 bonus) | v |
| 6 | B.3.2 stub document written | v |
| 7 | WRGA paper Appendix B.2 addendum written | v (companion markdown; PDF regeneration separate) |

## Scope expansion surfaced by Task 5.1

Task 5.1's fixture run revealed that B.3.1's plan had missed an infrastructure gap: the AST rewrite dispatch / runtime FFI / kernel registration for GatedLoRA didn't exist yet. LoRA and IA3 had this from B.3; GatedLoRA was always on the deferred list and never got it.

Added Tasks **5.0.a-c** (3 infrastructure tasks) covering:
- `synthesize_gatedlora_adapted` fused-dispatch branch (5.0.a)
- `try_cuda_launch_fused_gatedlora` real cudarc launch + `nsl_adapter_fused_gatedlora_matmul` FFI (5.0.b)
- `wrga_prescan.rs` GatedLoRA synthesis loop + separate `fused_gatedlora_ptx_kernels` registry + codegen registration emission (5.0.c)

Pre-existing bug fixed along the way: `synthesize_gatedlora_adapted`'s `field_symbols["sigmoid"]` lookup had been returning an arbitrary field name (e.g., `w`) as the sigmoid callee, causing codegen errors. Fixed in `functions.rs`.

Final commit count: 21 commits on `feat/wrga-b31-gatedlora-fused-forward` (spec + plan + enums + extraction + LoRA refactor + 3 helpers + stub + red test + synthesizer + 3 infra + 4 fixtures + 6 close-out).

## Final test counts

- `nsl-codegen --lib`: 1631 passed / 1 pre-existing (`a23_projection_tile_sweep` in `flash_attention.rs`, out-of-scope)
- `fa_v2_snapshots`: 17 passed / 6 pre-existing CSHA Tier C drifts
- `wrga_fused_ptx_ptxas`: 21 passed / 0 failed (6 LoRA + 5 IA3 + 10 GatedLoRA)
- `wrga_adapter_runtime_equivalence`: 15 passed / 0 failed / 0 ignored (11 LoRA+IA3 + 4 GatedLoRA fixtures)

## Institutional lesson promoted to WRGA paper Appendix B.2

> Any approximation-based activation helper (sigmoid, tanh, GELU, softmax, and related) emitted as a dedicated PTX helper MUST have at least one mid-range integration fixture whose tolerance is set such that a sign-based-stub implementation would FAIL the fixture. Pure-saturation fixtures are necessary but not sufficient.

## Surprise bugs caught by the test discipline

1. **Em-dash in PTX comment** (Task 4.1): `emit_gate_load_per_thread` used a Unicode em-dash (U+2014). ptxas 13.2 rejects non-ASCII when PTX exceeds ~3500 lines (multi-K-iter configs). Caught by the `(16, 8, 32, 4)` ptxas config.

2. **Sigmoid symbol lookup bug** (Task 5.1): `synthesize_gatedlora_adapted` looked up `field_symbols["sigmoid"]` but the sigmoid symbol was never inserted into the map, causing codegen to fall back on an arbitrary field symbol. A pre-existing B.2.1-era bug caught only when Fixture A's unfused path fired.

3. **LoRA/GatedLoRA registry key collision** (Task 5.0.c): a shared `HashMap<LoraKernelKey, String>` would collide when both adapter types target same-shaped weights. Added separate `fused_gatedlora_ptx_kernels` map. Invariant #17.

4. **CRLF line-ending issue on snapshot files** (discovered at Task 1.5 smoke): Windows `core.autocrlf=true` converted `.snap` files to CRLF, breaking every snapshot assertion against LF-emitting PTX. Fixed on main via separate PR adding `.gitattributes` rule `*.snap text eol=lf`.

## Deferred follow-ups

- **B.3.2** -- Fused GatedLoRA backward; deferred with measurement trigger (see STUB doc).
- **B.4+** -- sm_90 WGMMA, `ldmatrix`, `cp.async`, multi-warp-per-tile, perf benchmarking (same as B.3 milestone's deferred items).
- **Deep `kernel_skeleton` refactor** with fusion-callback pattern.
- **FA param-block migration** to `emit_param_block`.
