# BLOCKED: M35.2a implementation gated on V-P1-D measurement

**Spec:** [`docs/superpowers/specs/2026-05-12-m35-2a-bitnet-backward-design.md`](docs/superpowers/specs/2026-05-12-m35-2a-bitnet-backward-design.md) §2.4

**Status:** UNMEASURED

## What this gate is

M35.2a (BitNet b1.58 backward kernel emission) cannot ship implementation until the Phase 1 → Phase 2 escalation criteria from M35.1 spec §1.3 are measured PASS on real workloads:

- **Implementation-quality gate:** ≥80% of paper's claimed inference speedup at the 3B configuration.
- **Implementation-quality gate:** ≥80% of paper's claimed memory reduction at the 3B configuration.
- **Method-quality gate:** trained b1.58 checkpoints achieve perplexity within 5% of equivalent-parameter FP16 baselines on a held-out evaluation set.

All three must pass for M35.2 to proceed.

## What this file does

While this file exists at repo root, the `.github/workflows/design_only_enforcement.yml` CI workflow rejects any PR introducing more than 5 non-comment non-blank lines in:

- `crates/nsl-codegen/src/bitnet/phases/*backward*.rs`
- `crates/nsl-codegen/src/bitnet/phases/*shadow*.rs`
- `crates/nsl-codegen/src/bitnet/orchestrator_train.rs`

Stub-only modules with module docstrings + `//! TODO(M35.2a impl gated on V-P1-D)` markers are permitted (≤5 lines).

## Measurement procedure

Per M35.1 spec §6.6 (logit-match merge gate) + M35.2a spec §2.4:

1. Land M35.1's two Linux follow-on items:
   - `tests/fixtures/bitnet_b158_3b_reference_logits.bin` (~2 MB FP16) generated via bitnet.cpp on the pinned `1bitLLM/bitnet_b1_58-3B@af89e318d78a70802061246bf037199d2fb97020` checkpoint against the 32-prompt fixture.
   - Weight-scale wiring through `crates/nsl-codegen/src/bitnet/phases/finalize.rs::emit` (per the TODO docstring in `loader.rs::LoadedTernaryWeight` from M35.1 PR #156).
2. Un-ignore `bitnet_logit_match::end_to_end_logit_match_against_hf_b158_3b` (remove `#[ignore]` attribute).
3. Run the merge gate on Linux (WSL2 OK per spec §2.5 V-P1-D-prep): `bash scripts/fetch_bitnet_b158_3b.sh && cargo test -p nsl-codegen --test bitnet_logit_match -- --ignored --nocapture`.
4. Measure inference throughput vs FP16 baseline; measure memory footprint vs FP16 baseline.
5. Train a small b1.58 checkpoint on a held-out dataset slice; measure final perplexity vs FP16 baseline.

Record the three measurements with timestamps + evidence (logs, GPU profiler output, perplexity numbers).

## How to unblock

The V-P1-D measurement PR updates this file with the result. After the V-P1-D measurement PR is merged with PASS status, the M35.2a implementation PR opens; its commit 1 deletes this file + retires the CI workflow.

If V-P1-D FAILS, per M35.1 spec §1.3, M35.2 is deferred indefinitely. M35.2a spec remains as permanent artifact per IR-003. This file is then deleted by an M35.2-deferral PR that also retires the M35.2 spec docs.

## Related artifacts

- M35.2a spec: `docs/superpowers/specs/2026-05-12-m35-2a-bitnet-backward-design.md`
- M35.2a plan: `docs/superpowers/plans/2026-05-12-m35-2a-bitnet-backward-implementation.md`
- M35.1 PR #156 (Phase 1 inference, merged)
- IR-003 (institutional rule codifying design-only-vs-measurement-dependent posture): `docs/institutional-rules.md`
