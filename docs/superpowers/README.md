# docs/superpowers — index

This directory is NSL's institutional memory: the design specs, implementation
plans and measurement findings behind each research subsystem, written at the
time the work was done. It is organised by DATE in `specs/` and `plans/`, which
is the right order for "what happened when" and the wrong order for "what do I
read to understand X". This page is the by-subsystem index (roadmap Doc1).

For the reader-facing description of how the code works today, start with
[`docs/architecture/`](../architecture/README.md) — one document per crate —
and the wiki's [Architecture-Overview](../wiki/Architecture-Overview.md). A
spec here records what was DESIGNED; the code and `STATUS.md` say what
shipped, and where the two disagree the code wins.

Older plans (WRGA fused-PTX and gated-LoRA closeouts, the M52–M62 roadmap
design, the pretraining memory-reduction plan) live in
[`docs/plans/`](../plans/).

Counts: 124 documents in 26 subsystems.

## FASE — fused adaptive step emission (training-loop fusion)

| Date | Kind | Document |
|---|---|---|
| 2026-04-14 | design spec | [`2026-04-14-fase-adamw-bias-correction-design.md`](specs/2026-04-14-fase-adamw-bias-correction-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-deferred-codegen-integration-design.md`](specs/2026-04-14-fase-deferred-codegen-integration-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-deferred-consume-per-param-hook-design.md`](specs/2026-04-14-fase-deferred-consume-per-param-hook-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-numerical-validation-design.md`](specs/2026-04-14-fase-numerical-validation-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-peak-memory-regression-design.md`](specs/2026-04-14-fase-peak-memory-regression-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-training-report-cli-design.md`](specs/2026-04-14-fase-training-report-cli-design.md) |
| 2026-04-14 | design spec | [`2026-04-14-fase-two-phase-grad-clip-design.md`](specs/2026-04-14-fase-two-phase-grad-clip-design.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-adamw-bias-correction.md`](plans/2026-04-14-fase-adamw-bias-correction.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-deferred-codegen-integration.md`](plans/2026-04-14-fase-deferred-codegen-integration.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-deferred-consume-per-param-hook.md`](plans/2026-04-14-fase-deferred-consume-per-param-hook.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-numerical-validation.md`](plans/2026-04-14-fase-numerical-validation.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-peak-memory-regression.md`](plans/2026-04-14-fase-peak-memory-regression.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-training-report-cli.md`](plans/2026-04-14-fase-training-report-cli.md) |
| 2026-04-14 | plan | [`2026-04-14-fase-two-phase-grad-clip.md`](plans/2026-04-14-fase-two-phase-grad-clip.md) |
| 2026-04-15 | design spec | [`2026-04-15-fase-codegen-phase2-design.md`](specs/2026-04-15-fase-codegen-phase2-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-fase-optim-step-dispatch-design.md`](specs/2026-04-15-fase-optim-step-dispatch-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-fase-per-layer-mode-design.md`](specs/2026-04-15-fase-per-layer-mode-design.md) |
| 2026-04-15 | plan | [`2026-04-15-fase-codegen-phase2-implementation.md`](plans/2026-04-15-fase-codegen-phase2-implementation.md) |
| 2026-04-15 | plan | [`2026-04-15-fase-optim-step-dispatch-implementation.md`](plans/2026-04-15-fase-optim-step-dispatch-implementation.md) |
| 2026-04-15 | plan | [`2026-04-15-fase-per-layer-mode-implementation.md`](plans/2026-04-15-fase-per-layer-mode-implementation.md) |

## CPDT — compile-time precision / weight-aware compilation

| Date | Kind | Document |
|---|---|---|
| 2026-04-15 | design spec | [`2026-04-15-cpdt-pipeline-integration-design.md`](specs/2026-04-15-cpdt-pipeline-integration-design.md) |
| 2026-04-15 | plan | [`2026-04-15-cpdt-pipeline-integration-implementation.md`](plans/2026-04-15-cpdt-pipeline-integration-implementation.md) |
| 2026-04-18 | design spec | [`2026-04-18-cpdt-weight-aware-phase1-design.md`](specs/2026-04-18-cpdt-weight-aware-phase1-design.md) |
| 2026-04-18 | findings / notes | [`2026-04-18-cpdt-weight-aware-phase2-stub.md`](specs/2026-04-18-cpdt-weight-aware-phase2-stub.md) |
| 2026-04-18 | plan | [`2026-04-18-cpdt-weight-aware-phase1.md`](plans/2026-04-18-cpdt-weight-aware-phase1.md) |
| 2026-04-19 | design spec | [`2026-04-19-cpdt-calibration-correction-design.md`](specs/2026-04-19-cpdt-calibration-correction-design.md) |
| 2026-04-19 | plan | [`2026-04-19-cpdt-calibration-correction.md`](plans/2026-04-19-cpdt-calibration-correction.md) |
| 2026-04-20 | design spec | [`2026-04-20-cpdt-validate-body-design.md`](specs/2026-04-20-cpdt-validate-body-design.md) |
| 2026-04-20 | design spec | [`2026-04-20-cpdt-weight-aware-opt-out-design.md`](specs/2026-04-20-cpdt-weight-aware-opt-out-design.md) |
| 2026-04-21 | design spec | [`2026-04-21-cpdt-ast-autodetect-design.md`](specs/2026-04-21-cpdt-ast-autodetect-design.md) |

## PCA — position-conditioned attention (Tier A/B, RoPE)

| Date | Kind | Document |
|---|---|---|
| 2026-04-18 | design spec | [`2026-04-18-pca-tier-a-design.md`](specs/2026-04-18-pca-tier-a-design.md) |
| 2026-04-18 | plan | [`2026-04-18-pca-tier-a-implementation.md`](plans/2026-04-18-pca-tier-a-implementation.md) |
| 2026-05-02 | design spec | [`2026-05-02-pca-tier-b-tile-skip-design.md`](specs/2026-05-02-pca-tier-b-tile-skip-design.md) |
| 2026-05-12 | design spec | [`2026-05-12-pca-tier-b-revision-design.md`](specs/2026-05-12-pca-tier-b-revision-design.md) |
| 2026-05-12 | findings / notes | [`2026-05-12-tier-b-smem-probe-findings.md`](specs/2026-05-12-tier-b-smem-probe-findings.md) |
| 2026-05-12 | plan | [`2026-05-12-pca-tier-b-tile-skip-implementation-v2.md`](plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md) |
| 2026-05-13 | design spec | [`2026-05-13-pca-tier-b15-and-b2-design.md`](specs/2026-05-13-pca-tier-b15-and-b2-design.md) |
| 2026-05-13 | findings / notes | [`2026-05-13-tier-b-b15-3-skip-ratio-investigation.md`](specs/2026-05-13-tier-b-b15-3-skip-ratio-investigation.md) |
| 2026-05-13 | findings / notes | [`2026-05-13-tier-b-b2-predicate-verification-findings.md`](specs/2026-05-13-tier-b-b2-predicate-verification-findings.md) |
| 2026-05-13 | findings / notes | [`2026-05-13-tier-b-m2-m6-findings.md`](specs/2026-05-13-tier-b-m2-m6-findings.md) |
| 2026-05-13 | findings / notes | [`2026-05-13-tier-b-measurement-procedure.md`](specs/2026-05-13-tier-b-measurement-procedure.md) |
| 2026-05-13 | plan | [`2026-05-13-pca-tier-b15-and-b2-implementation.md`](plans/2026-05-13-pca-tier-b15-and-b2-implementation.md) |
| 2026-05-14 | design spec | [`2026-05-14-pca-tier-b-dispatch-design.md`](specs/2026-05-14-pca-tier-b-dispatch-design.md) |
| 2026-05-14 | findings / notes | [`2026-05-14-tier-b-dispatch-integration-findings.md`](specs/2026-05-14-tier-b-dispatch-integration-findings.md) |
| 2026-05-14 | plan | [`2026-05-14-pca-tier-b-dispatch-implementation.md`](plans/2026-05-14-pca-tier-b-dispatch-implementation.md) |
| 2026-05-15 | data | [`2026-05-15-tier-b-floor-derivation.csv`](specs/2026-05-15-tier-b-floor-derivation.csv) |
| 2026-05-15 | design spec | [`2026-05-15-pca-tier-b-planner-design.md`](specs/2026-05-15-pca-tier-b-planner-design.md) |
| 2026-05-15 | findings / notes | [`2026-05-15-tier-b-bii-smem-probe-findings.md`](specs/2026-05-15-tier-b-bii-smem-probe-findings.md) |
| 2026-05-15 | findings / notes | [`2026-05-15-tier-b-floor-derivation-findings.md`](specs/2026-05-15-tier-b-floor-derivation-findings.md) |
| 2026-05-15 | findings / notes | [`2026-05-15-tier-b-planner-options-findings.md`](specs/2026-05-15-tier-b-planner-options-findings.md) |
| 2026-05-15 | plan | [`2026-05-15-pca-tier-b-planner-implementation.md`](plans/2026-05-15-pca-tier-b-planner-implementation.md) |
| 2026-05-16 | design spec | [`2026-05-16-pca-rope-position-reset-design.md`](specs/2026-05-16-pca-rope-position-reset-design.md) |
| 2026-05-16 | design spec | [`2026-05-16-pca-strategy-3-per-cta-design.md`](specs/2026-05-16-pca-strategy-3-per-cta-design.md) |
| 2026-05-16 | findings / notes | [`2026-05-16-rope-ffi-scope-findings.md`](specs/2026-05-16-rope-ffi-scope-findings.md) |
| 2026-05-16 | plan | [`2026-05-16-pca-rope-position-reset-implementation.md`](plans/2026-05-16-pca-rope-position-reset-implementation.md) |
| 2026-05-17 | design spec | [`2026-05-17-pca-rope-activation-design.md`](specs/2026-05-17-pca-rope-activation-design.md) |

## WRGA — weight-rewrite / gated-LoRA adapter fusion

| Date | Kind | Document |
|---|---|---|
| 2026-04-12 | design spec | [`2026-04-12-wrga-milestone-b2-design.md`](specs/2026-04-12-wrga-milestone-b2-design.md) |
| 2026-04-12 | design spec | [`2026-04-12-wrga-milestone-b21-design.md`](specs/2026-04-12-wrga-milestone-b21-design.md) |
| 2026-04-12 | plan | [`2026-04-12-wrga-milestone-b2-plan.md`](plans/2026-04-12-wrga-milestone-b2-plan.md) |
| 2026-04-12 | plan | [`2026-04-12-wrga-milestone-b21-plan.md`](plans/2026-04-12-wrga-milestone-b21-plan.md) |
| 2026-04-13 | design spec | [`2026-04-13-wrga-milestone-b3-design.md`](specs/2026-04-13-wrga-milestone-b3-design.md) |
| 2026-04-13 | plan | [`2026-04-13-wrga-milestone-b3-plan.md`](plans/2026-04-13-wrga-milestone-b3-plan.md) |
| 2026-04-19 | design spec | [`2026-04-19-wrga-b32-option3-revised-design.md`](specs/2026-04-19-wrga-b32-option3-revised-design.md) |
| 2026-04-19 | design spec | [`2026-04-19-wrga-b32-option3-source-ad-wiring-design.md`](specs/2026-04-19-wrga-b32-option3-source-ad-wiring-design.md) |
| 2026-04-19 | plan | [`2026-04-19-wrga-b32-option3-source-ad-wiring-plan.md`](plans/2026-04-19-wrga-b32-option3-source-ad-wiring-plan.md) |

## CSHA — compiler-specialized hybrid attention

| Date | Kind | Document |
|---|---|---|
| 2026-04-13 | design spec | [`2026-04-13-csha-tier-a-wiring-design.md`](specs/2026-04-13-csha-tier-a-wiring-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-csha-tier-c-fused-backward-design.md`](specs/2026-04-15-csha-tier-c-fused-backward-design.md) |
| 2026-04-15 | plan | [`2026-04-15-csha-tier-c-fused-backward.md`](plans/2026-04-15-csha-tier-c-fused-backward.md) |
| 2026-04-17 | design spec | [`2026-04-17-csha-gap-i-design.md`](specs/2026-04-17-csha-gap-i-design.md) |
| 2026-04-17 | findings / notes | [`2026-04-17-csha-gap-i-cascade-audit.md`](specs/2026-04-17-csha-gap-i-cascade-audit.md) |

## M35 — BitNet b1.58 ternary and FP8 MMA

| Date | Kind | Document |
|---|---|---|
| 2026-04-15 | design spec | [`2026-04-15-m35-fp8-mma-correctness-design.md`](specs/2026-04-15-m35-fp8-mma-correctness-design.md) |
| 2026-04-15 | plan | [`2026-04-15-m35-fp8-mma-correctness.md`](plans/2026-04-15-m35-fp8-mma-correctness.md) |
| 2026-05-11 | design spec | [`2026-05-11-m35-1-bitnet-ternary-design.md`](specs/2026-05-11-m35-1-bitnet-ternary-design.md) |
| 2026-05-11 | plan | [`2026-05-11-m35-1-bitnet-ternary-implementation.md`](plans/2026-05-11-m35-1-bitnet-ternary-implementation.md) |
| 2026-05-12 | design spec | [`2026-05-12-m35-2a-bitnet-backward-design.md`](specs/2026-05-12-m35-2a-bitnet-backward-design.md) |
| 2026-05-12 | findings / notes | [`2026-05-12-m35-2-ste-baseline-findings.md`](specs/2026-05-12-m35-2-ste-baseline-findings.md) |
| 2026-05-12 | plan | [`2026-05-12-m35-2a-bitnet-backward-implementation.md`](plans/2026-05-12-m35-2a-bitnet-backward-implementation.md) |

## M62 — C ABI export and legacy interop

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m62-legacy-interop-design.md`](specs/2026-03-19-m62-legacy-interop-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-m62-c-wrappers-design.md`](specs/2026-04-15-m62-c-wrappers-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-m62-export-decorator-design.md`](specs/2026-04-15-m62-export-decorator-design.md) |
| 2026-04-15 | design spec | [`2026-04-15-m62-grad-context-bridge-design.md`](specs/2026-04-15-m62-grad-context-bridge-design.md) |
| 2026-04-15 | plan | [`2026-04-15-m62-c-wrappers-implementation.md`](plans/2026-04-15-m62-c-wrappers-implementation.md) |
| 2026-04-15 | plan | [`2026-04-15-m62-export-decorator-implementation.md`](plans/2026-04-15-m62-export-decorator-implementation.md) |
| 2026-04-16 | design spec | [`2026-04-16-m62-weight-loading-design.md`](specs/2026-04-16-m62-weight-loading-design.md) |
| 2026-04-16 | plan | [`2026-04-16-m62-weight-loading-implementation.md`](plans/2026-04-16-m62-weight-loading-implementation.md) |

## M56 — multi-agent

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m56-multi-agent-design.md`](specs/2026-03-19-m56-multi-agent-design.md) |
| 2026-04-23 | design spec | [`2026-04-23-m56-multi-agent-v1-design.md`](specs/2026-04-23-m56-multi-agent-v1-design.md) |
| 2026-04-23 | plan | [`2026-04-23-m56-multi-agent-v1.md`](plans/2026-04-23-m56-multi-agent-v1.md) |

## M57 — FPGA / neuromorphic

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m57-fpga-neuromorphic-design.md`](specs/2026-03-19-m57-fpga-neuromorphic-design.md) |

## M58 — fault tolerance

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m58-fault-tolerance-design.md`](specs/2026-03-19-m58-fault-tolerance-design.md) |

## M59 — topology routing

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m59-topology-routing-design.md`](specs/2026-03-19-m59-topology-routing-design.md) |

## M60 — distributed data

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m60-distributed-data-design.md`](specs/2026-03-19-m60-distributed-data-design.md) |

## M61 — cluster debugging

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-m61-cluster-debugging-design.md`](specs/2026-03-19-m61-cluster-debugging-design.md) |

## Dev tools (phases 1–5)

| Date | Kind | Document |
|---|---|---|
| 2026-04-12 | design spec | [`2026-04-12-nsl-dev-tools-phase1-design.md`](specs/2026-04-12-nsl-dev-tools-phase1-design.md) |
| 2026-04-12 | plan | [`2026-04-12-nsl-dev-tools-phase1.md`](plans/2026-04-12-nsl-dev-tools-phase1.md) |
| 2026-04-13 | design spec | [`2026-04-13-nsl-dev-tools-phase2-5-design.md`](specs/2026-04-13-nsl-dev-tools-phase2-5-design.md) |
| 2026-04-13 | design spec | [`2026-04-13-nsl-dev-tools-phase2-design.md`](specs/2026-04-13-nsl-dev-tools-phase2-design.md) |
| 2026-04-13 | design spec | [`2026-04-13-nsl-dev-tools-phase3-design.md`](specs/2026-04-13-nsl-dev-tools-phase3-design.md) |
| 2026-04-13 | design spec | [`2026-04-13-nsl-dev-tools-phase4-design.md`](specs/2026-04-13-nsl-dev-tools-phase4-design.md) |
| 2026-04-13 | design spec | [`2026-04-13-nsl-dev-tools-phase5-design.md`](specs/2026-04-13-nsl-dev-tools-phase5-design.md) |
| 2026-04-13 | plan | [`2026-04-13-nsl-dev-tools-phase2-5.md`](plans/2026-04-13-nsl-dev-tools-phase2-5.md) |
| 2026-04-13 | plan | [`2026-04-13-nsl-dev-tools-phase2.md`](plans/2026-04-13-nsl-dev-tools-phase2.md) |
| 2026-04-13 | plan | [`2026-04-13-nsl-dev-tools-phase3.md`](plans/2026-04-13-nsl-dev-tools-phase3.md) |
| 2026-04-13 | plan | [`2026-04-13-nsl-dev-tools-phase4.md`](plans/2026-04-13-nsl-dev-tools-phase4.md) |
| 2026-04-13 | plan | [`2026-04-13-nsl-dev-tools-phase5.md`](plans/2026-04-13-nsl-dev-tools-phase5.md) |

## Coder models and RL/SFT data

| Date | Kind | Document |
|---|---|---|
| 2026-04-01 | design spec | [`2026-04-01-coder-rl-sft-data-pipeline-design.md`](specs/2026-04-01-coder-rl-sft-data-pipeline-design.md) |
| 2026-04-01 | plan | [`2026-04-01-coder-rl-sft-data-pipeline.md`](plans/2026-04-01-coder-rl-sft-data-pipeline.md) |

## WGGO — weight-graph global optimization

| Date | Kind | Document |
|---|---|---|
| 2026-04-22 | design spec | [`2026-04-22-wggo-prune-ir-rewrite-design.md`](specs/2026-04-22-wggo-prune-ir-rewrite-design.md) |
| 2026-04-22 | plan | [`2026-04-22-wggo-prune-ir-rewrite-implementation.md`](plans/2026-04-22-wggo-prune-ir-rewrite-implementation.md) |

## AWQ quantization

| Date | Kind | Document |
|---|---|---|
| 2026-04-22 | design spec | [`2026-04-22-awq-real-subprocess-completion-design.md`](specs/2026-04-22-awq-real-subprocess-completion-design.md) |
| 2026-04-22 | plan | [`2026-04-22-awq-real-subprocess-completion-implementation.md`](plans/2026-04-22-awq-real-subprocess-completion-implementation.md) |

## Calibration decoupling (#134)

| Date | Kind | Document |
|---|---|---|
| 2026-05-06 | design spec | [`2026-05-06-134-decouple-calibration-design.md`](specs/2026-05-06-134-decouple-calibration-design.md) |
| 2026-05-06 | plan | [`2026-05-06-134-decouple-calibration-implementation.md`](plans/2026-05-06-134-decouple-calibration-implementation.md) |

## Matmul / cuBLAS

| Date | Kind | Document |
|---|---|---|
| 2026-04-21 | design spec | [`2026-04-21-matmul-cublas-swap-design.md`](specs/2026-04-21-matmul-cublas-swap-design.md) |

## FlashAttention scalar emitter

| Date | Kind | Document |
|---|---|---|
| 2026-04-14 | design spec | [`2026-04-14-fa-scalar-emitter-rewrite-design.md`](specs/2026-04-14-fa-scalar-emitter-rewrite-design.md) |
| 2026-04-14 | plan | [`2026-04-14-fa-scalar-emitter-rewrite.md`](plans/2026-04-14-fa-scalar-emitter-rewrite.md) |

## CFTP

| Date | Kind | Document |
|---|---|---|
| 2026-05-16 | findings / notes | [`2026-05-16-cftp-section-4-4-deferral.md`](specs/2026-05-16-cftp-section-4-4-deferral.md) |

## Training-state identity (2026-08 items 7–8)

| Date | Kind | Document |
|---|---|---|
| 2026-08-18 | design spec | [`2026-08-18-item7-dlpack-output-ownership.md`](specs/2026-08-18-item7-dlpack-output-ownership.md) |
| 2026-08-19 | design spec | [`2026-08-19-item8-resumable-training-state.md`](specs/2026-08-19-item8-resumable-training-state.md) |

## Train-block driver decomposition (roadmap A1)

| Date | Kind | Document |
|---|---|---|
| 2026-09-08 | design spec | [`2026-09-08-a1-train-plan-ir-design.md`](specs/2026-09-08-a1-train-plan-ir-design.md) |

## Runtime C-ABI source of truth (roadmap A3)

| Date | Kind | Document |
|---|---|---|
| 2026-09-08 | design spec | [`2026-09-08-a3-abi-extern-table-design.md`](specs/2026-09-08-a3-abi-extern-table-design.md) |

## CUDA context — one value per device (roadmap A4)

| Date | Kind | Document |
|---|---|---|
| 2026-09-09 | design spec | [`2026-09-09-a4-cuda-context-design.md`](specs/2026-09-09-a4-cuda-context-design.md) |

## Kernel IR v2 and the hand-PTX migration (roadmap A2)

| Date | Kind | Document |
|---|---|---|
| 2026-09-09 | design spec | [`2026-09-09-a2-kir-v2-design.md`](specs/2026-09-09-a2-kir-v2-design.md) |

## Other

| Date | Kind | Document |
|---|---|---|
| 2026-03-19 | design spec | [`2026-03-19-nsl-coder-50m-design.md`](specs/2026-03-19-nsl-coder-50m-design.md) |
| 2026-03-19 | plan | [`2026-03-19-nsl-coder-50m-implementation.md`](plans/2026-03-19-nsl-coder-50m-implementation.md) |
