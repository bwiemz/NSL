# NSL Institutional Rules

This document catalogs project-level design principles surfaced across NSL's spec and brainstorm work. Each rule has a stable identifier (IR-NNN) used in spec citations.

## How to read this document

Each rule is one paragraph stating the principle and citing the specs where it was surfaced or applied. Specs cite rules by identifier (e.g., "per IR-003, the verification gate runs before implementation").

## How to add a rule (entry criteria)

A pattern becomes an IR when it satisfies all three:

- Surfaced across at least two distinct specs/brainstorms.
- The pattern's violation produced or would have produced a real failure mode in retrospect.
- The pattern is small enough to cite by identifier and explain in one paragraph.

Patterns that don't satisfy these criteria are documented in the rejecting spec's text as "considered but not codified," not added to the registry. The criteria prevent two failure modes: registry inflation (every spec adding "lessons learned" entries) and registry stagnation (real patterns never codified because nobody knows when to add).

## Rules

### IR-001 — Preconditions enforced by API shape, not docstrings

Where a function has an unstated precondition (a callee must be invoked first, an input must be in a specific state), the API should make violation structurally impossible — keep the unsafe-without-prep function module-internal, expose only the safe composition. Distinct from type-system enforcement; this is about visibility and composition shape.

Cited from:
- `docs/superpowers/specs/2026-04-26-bitnet-phase1-design.md` — `quantized_ternary_gemm` fusion.
- `docs/superpowers/specs/2026-04-29-awq-calibration-backward-pass-design.md` — `weight_index_map`.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §3.4 — `RangeTableAddrs` constructor.

### IR-002 — External references as one-time anchors

External measurements / hardware findings recorded in dated findings docs and cited by path from specs that depend on them. Findings docs append-only with dated re-run log; specs cite by path so the dependency is auditable.

Cited from:
- BitNet HF checkpoint pinning (b1.58 reference logits fixture).
- AWQ calibration (sidecar envelope hashing).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §2 + §3.4 — SMEM probe findings doc dependency.

### IR-003 — Pre-implementation verification of load-bearing assumptions

When a spec relies on cross-module behavior, verify the assumption via grep / probe / measurement before writing the code that depends on it. 15-minute verification beats multi-day rework. **Specialized case: resource-budget arithmetic at spec-pinned fixture dimensions.** When a spec pins specific fixture dimensions for measurement (e.g., gate fixtures), the spec's resource-budget arithmetic (SMEM, register pressure, occupancy) must be instantiated at those specific values, not left generic-over-config — otherwise the spec passes review based on plausible-generic numbers while the spec-pinned fixture has an unverified-specific failure mode.

Cited from:
- WGGO Phase 2 NodeId space (verified before backward-pass integration).
- CSHA Tier B.1 V1/V2/V3 findings (verified pre-B1.2 kernel emission).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §2 — SMEM probe.
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` §11.4 — gate fixture SMEM-budget verification gap (fixture-pinned SMEM exceeded device cap; uncaught because spec's §3.4 arithmetic was generic-over-config rather than instantiated-at-gate-fixture). First instance of the fixture-pinned-resource-budget specialization; if a second instance materializes (e.g., during M35.2a backward SMEM verification), promote to IR-013.

### IR-004 — APIs return aligned, ready-to-use values so consumers don't have to

Offset-computation site owns alignment, padding, and any other consumer-facing invariants; consumer assumes the returned value is ready-to-use. Separation-of-concerns principle: the layout site is the single audit point for layout correctness.

Cited from:
- `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs` — `kv_offset` / `sp_offset` discipline.
- `docs/superpowers/specs/2026-04-26-bitnet-phase1-design.md` — `packed_load.rs` register-resident values.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §3.4 — `tier_b_range_table_offset` `align_up(2)` guarantee.

### IR-005 — Bifurcation of emission paths for the same logical operation requires measurement-driven justification, not architectural preference

v1 prefers uniform emission across the config matrix as a structural property; bifurcation (e.g., "compile-time-unroll at small N, runtime loop at large N") requires explicit performance measurement justifying the dual emission paths. Reason: bifurcation creates non-uniform bug surface (regression in one path only manifests at certain configs) and doubled test infrastructure.

Cited from:
- `docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md` — no producer/consumer split in v1.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §4.7 — no compile-time-unroll / runtime-loop hybrid.

### IR-006 — Distinct failure modes warrant distinct test surfaces

Bundle test concerns only when their failure modes have identical diagnostic shape; otherwise split them. The principle is diagnostic precision: when two distinct failure modes share a single regression surface, bisecting which one fired adds investigation time at every regression.

Cited from:
- CSHA Tier B.1 cost-model correction (standalone PR, separate from kernel-implementation).
- WGGO Phase 2 #134 — per-commit milestone matrix.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §5 — isolation snapshot + integration SASS split.

### IR-007 — PTX emission discipline is pinned at the instruction-pattern level, not the algorithmic level

The ptxas → SASS pipeline is pattern-recognition-driven, not semantically aware. Spec text that says "emit a warp-uniform branch" without pinning the specific PTX patterns (register class, operand shape, predicate construction) that ptxas recognizes as uniform risks emission that's algorithmically correct but performance-degraded. Pin specific patterns.

Cited from:
- FA-2 v2 Tier B.1 cp.async commit-group cadence + wait_group operand.
- BitNet `BFE.U32` for single-instruction trit unpack.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §4.2 — warp-uniformity through specific register-class + operand patterns.

### IR-008 — For long-lived kernels, verification surface investment is the load-bearing cost

Observation across multiple kernel specs: kernel emission is a few hundred lines; verification (snapshots, SASS assertions, parity tests, reference impls) is comparable or larger and is what catches regressions over the kernel's lifetime. Not a rule that prescribes a ratio — a framing that justifies budget allocation when the verification surface seems disproportionate to the emission.

Cited from:
- FA-2 v2 Tier B.1 verification harness (V1/V2/V3 + cost-model snapshot).
- BitNet Phase 1 validation harness + reference implementation.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` overall §3.1 / §3.4 / §6.3 balance.
- `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §8.10 — 1.7× test-LOC-to-implementation-LOC ratio reported observationally per IR-008's original framing (not a budget ceiling); symptom-based v2 triggers (duplicate coverage, speculative tests, over-specification of settled surfaces) are the right shape for future test-surface review, not aggregate metric thresholds.

### IR-009 — Dead-code lifecycle requires explicit removal triggers

Dead-code (feature-flag-disabled-but-not-removed) without an explicit removal trigger becomes permanent. When a spec deprecates a feature via feature-flag disable rather than git-revert, the spec MUST pin (a) a decay timer (e.g., 6 months) by which a spec-level review re-evaluates dead-code maintenance cost vs option-value, (b) revival triggers (workload that re-justifies the feature, breakage that forces fix-or-revert choice), and (c) the breakage-trigger semantics (CUDA toolkit update, dependency drift). Without these, "we'll remove it later if not revived" decays into "it's been here forever and nobody knows why." The institutional pattern: dead-code is fine; permanent-dead-code is institutional debt that compounds.

Cited from:
- `docs/superpowers/specs/2026-04-27-awq-calibration-followup-design.md` — deprecated-shim lifecycle (the original surfacing).
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` §3.4.1 — Tier B revert with 6-month decay timer.

### IR-010 — Measurement-gated decisions specify thresholds, fail semantics, and protocol before the measurement runs

Decisions deferred until after measurement risk post-hoc threshold negotiation — the natural human temptation to soften a gate that the measurement nearly cleared. Pinning all three at design time (pass thresholds with rationale, fail semantics with explicit action, measurement protocol with reproducibility tolerance) converts the gate from "decision support" to "data-driven decision." Post-measurement, only the data is examined; the decision rules are settled. If post-measurement analysis suggests a threshold was wrong, that analysis lands as a separate spec revision PR with explicit justification, not as a measurement-time adjustment. Same shape across BitNet Phase 1 → Phase 2 escalation criteria, CSHA Tier B.1 V1/V2/V3, and PCA Tier B keep/revert.

Cited from:
- `docs/superpowers/specs/2026-04-26-bitnet-phase1-design.md` — Phase 1 → Phase 2 escalation thresholds pinned at spec time.
- `docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md` — V1/V2/V3 verification decision matrices.
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` §3 — Tier B acceptance bar with non-negotiability policy.

### IR-011 — Distinct test surfaces roll up into a richer decision space than single-surface evaluation

Multi-fixture compositions (gate / parity / sensitivity tiers; V1/V2/V3 verifications; per-commit milestone matrices) enable nuanced outcomes that single-surface evaluation forecloses. The composition is the load-bearing property, not the multiple surfaces themselves. Example: a single-fixture gate produces binary keep/revert; a three-tier matrix (gate + parity + sensitivity) produces three outcomes including "keep with sparsity gate" derived from the sensitivity curve. The richer outcome space emerges from composing surfaces whose individual purposes are distinct (correctness vs decision vs diagnostic). This is one level up from IR-006 (distinct test surfaces for distinct failure modes) — it's "distinct test surfaces roll up into outcomes IR-006 alone wouldn't have surfaced."

Cited from:
- `docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md` — V1/V2/V3 multi-tier verification.
- WGGO Phase 2 #134 — per-commit milestone matrix.
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` §4 — gate/parity/sensitivity tiers unlock the "keep with sparsity gate" outcome.
- `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §9 — third instance of distinct test surfaces rolling up into a richer decision space: V-Bii-SMEM probe's five-outcome decision matrix + per-MAX SMEM-headroom recording roll up into a four-way v2 trigger split (path A extend baked max / path B B-iii migration; from MAX=16384 unrestricted vs MAX=8192 restricted), differentiating cost from ~30 min re-probe to ~500 LOC kernel-architecture migration. (First instance: dispatch spec §13.3 sensitivity tier; second instance: this spec's trigger-space split.)

### IR-012 — Measurement-infrastructure contracts are explicit in spec, not implicit in implementation

Shell scripts, CI configurations, and downstream tools encode measurement-binary contracts (output format, exit codes, fixture names, reproducibility seeds) as integration assumptions. Convention-only enforcement decays as the binary evolves; explicit contracts in spec persist. Pinning the contract — output format with stable prefix, exit code semantics, reproducibility seed default — prevents future refactors from silently breaking downstream consumers. Same discipline as CHANGELOG-CALIBRATION enforcement and SMEM-probe findings-doc append-only protocol.

Cited from:
- `docs/superpowers/specs/2026-04-27-awq-calibration-followup-design.md` — CHANGELOG-CALIBRATION enforcement.
- `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` — append-only re-run log.
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` §5.2 — nsl-codegen-bench invocation contract (output line format, exit codes, `--seed`).

### IR-013 — External-caller-context assumptions warrant pre-implementation verification

Specs whose design depends on what callers know, pass through, or have available at call time must verify those caller-context assumptions before implementation invests against them. Distinct from IR-003's general internal-code-graph verification (NodeId space, ABI signatures, predicate operand symmetry) — IR-013 is about **external** caller behavior: the input pipelines that feed your spec's surface. Plausibility-driven design ("callers probably have X at this layer") is the failure mode this rule prevents. The verification protocol mirrors IR-003: 30-45 minute grep / read-upward / classify before the dependent code is written.

Narrow framing per the surfacing spec's §13.2 reasoning: IR-013 is caller-behavior-in-the-same-codebase specifically. Three structurally distinct verification disciplines surfaced across this brainstorm cycle, each potentially warranting its own institutional rule over time:

- **IR-013 (here):** Caller-behavior verification — grep call-graph, classify cases (α/β/γ), document trace.
- **IR-014 candidate (future):** Upstream-reference verification — read external project's code, document its idiom, amend spec (pattern of V-M35.2a-STE; promote on second instance).
- **IR-015 candidate (future):** Hardware-capability or runtime-value verification — emit forced-access probe, sweep configs, measure (pattern of SMEM probe + V-P1-D; promote on second instance).

Collapsing all three into a broad IR-013 would lose diagnostic distinction. The narrow framing preserves precision; the candidate slots reserve room for the other specializations to land as their own rules when they're justified.

Cited from:
- `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` §5 + §14 — V-dispatch-integration of PCA Tier B dispatch callers; surfaced case-(β) finding (α=0, β=3, γ=0) that invalidated the original spec's case-(α) premise and pivoted the work to a follow-on planner spec. The verification ran in 30-45 minutes and prevented ~150 LOC of D-3 implementation work that would have hit integration failure. The canonical first instance.
