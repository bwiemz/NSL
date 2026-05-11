# BitNet phase emitters

Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §3 + §4.

## Phase 1 (M35.1) — inference-only

### Public emitters (callable from external subsystems)

- `packed_load.rs` — HBM → SMEM ternary load + on-the-fly unpack.
- `quantized_ternary_gemm.rs` — fused activation-quant prologue + ternary GEMM body.
- `finalize.rs` — dequant + bias/residual add + HBM write.

### Subsystem-internal emitters (`pub(super)`)

- `absmean_quant.rs` — per-row activation quantization to int8. Callable
  from unit tests but external callers should use `quantized_ternary_gemm`.
  (File name historically references "absmean"; actual BitNet b1.58 uses
  per-row **absmax** for activations — weights use absmean. The file name
  is retained for spec traceability; the implementation uses absmax.)
- `ternary_gemm.rs` — bare ternary GEMM body. **Never publicly exposed** —
  has the "activations must be quantized first" precondition that's
  impossible to violate when fused with the activation-quant prologue.
  Discipline pattern: IR-001 (API-shape-enforced invariants).

## Phase 2 (M35.2) — additive (NOT created in Phase 1)

When Phase 2 ships (gated on §1.3's escalation criterion in the spec), it
adds the following files. They are **not** stubs in Phase 1 — Phase 2's
PR creates them.

- `ternary_gemm_backward.rs` — STE backward through forward GEMM.
- `quantize_shadow.rs` — FP32 shadow weights → ternary packed format.
- `absmean_quant_backward.rs` — backward through activation quantization
  (or folded into STE; design choice deferred to Phase 2).
- `orchestrator_train.rs` — training-mode orchestrator composing forward
  + backward phases.

## Audit discipline

Phase 2 PRs MAY ONLY modify files listed under "Phase 2 — additive" above,
plus any net-new file added under that section. Modifications to any
Phase 1 file from a Phase 2 PR constitute a Phase 1 regression and must
be filed as a separate Phase 1 fix PR.
