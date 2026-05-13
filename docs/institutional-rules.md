# NSL Institutional Rules Registry

Cross-cutting disciplines surfaced through spec brainstorms. Specs cite rules
by stable identifier (e.g., "per IR-001 ..."). Adding a new rule requires
the surfacing spec to introduce it; rules in this registry are load-bearing.

---

## IR-001: API-shape-enforced invariants

**Rule.** Invariants on data flowing between phase emitters (or analogous
internal interfaces) should be enforced by API shape, not docstrings.

**Pattern.** When a phase emitter has a precondition on its inputs that
can't be type-checked statically, make the emitter subsystem-internal and
expose a composed public emitter that establishes the precondition before
invoking the internal one.

**Origin.** Introduced in
`docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §8.1.

**Example (BitNet M35.1).** `ternary_gemm.rs` requires activations to be
quantized first. Rather than document the precondition in a docstring,
`ternary_gemm.rs` is kept subsystem-internal; the public path is
`quantized_ternary_gemm.rs`, which fuses activation-quant + GEMM.
External callers cannot invoke `ternary_gemm.rs` directly.

**Related instances.** The calibration FFI's compile-time `.expect()` for
`weight_index_map`'s pre-population is a related discipline. Type-safe
builders with required fields (e.g., `BitNetKernelConfig`) are another
instance.

---

## IR-002: External references as one-time correctness anchors

**Rule.** External references (bitnet.cpp, HuggingFace checkpoints,
third-party implementations) are one-time correctness anchors during
initial implementation, not ongoing CI dependencies.

**Pattern.** During initial implementation, capture the external
reference's relevant output as committed fixtures. Validate the NSL
implementation against the fixtures, not against the live external
source. Future regressions are caught by fixture comparisons; future
legitimate updates re-anchor against the external source as a one-time
step.

**Origin.** Introduced in
`docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §8.2.

**Examples (BitNet M35.1).**

- Trit-within-byte ordering: bitnet.cpp's `unpack_weights` inspected once;
  high-bits-first convention pinned and asserted by hand-constructed-byte test.
- CPU reference correctness: bitnet.cpp's outputs on 10 fixtures captured
  as JSON; the Rust reference asserts bit-exact match.
- HF model identity: revision SHA pinned; CI fetches from
  `<model_id>@<revision_sha>` (HF revisions are immutable).

**Anti-pattern this prevents.** Ongoing C++ build dependencies in CI;
tests that silently drift when upstream pushes updates; "test failed
because external service changed" failure modes.

---

## IR-003: Separate design-proceeding work from measurement-dependent implementation

**Rule.** Work that can proceed on the platform/state available now should be separated from work that depends on unmeasured criteria. Design proceeds on the platform-independent surface; measurement-dependent implementation waits for the gate.

**Pattern.** Identify the load-bearing measurement (e.g., perplexity gate, hardware-behavior probe, V-P1-D-style escalation criteria). Identify what can be designed without it (architectural decisions, API shape, fixture sets, reference implementations). Ship the design as a permanent artifact; gate the implementation on the measurement.

**Origin.** Introduced in
`docs/superpowers/specs/2026-05-12-m35-2a-bitnet-backward-design.md` §9.2.

**Examples.**

- **AWQ #142 → #134:** design PR landed before #134 measured calibration-PPL outcomes. Design PR was self-contained; implementation cited design PR by SHA; design remained a permanent artifact regardless of measurement outcome.
- **M35.2a (origin spec):** ships design-only with `BLOCKED_ON_V_P1_D.md` + CI enforcement. Implementation waits for V-P1-D pass.
- **CSHA Tier B.1 V1/V2/V3:** verified load-bearing assumptions before kernel implementation began. Caught three substantive spec drifts pre-implementation.
- **V-M35.2a-STE (PR #163):** pre-implementation verification of the M35.2a STE choice. Surveyed community b1.58 sources and found DISCREPANCY (vanilla STE, not clipped). Caught the assumption error before any backward-kernel code was written.

**Discipline pattern: structural enforcement, not convention.** IR-003 is not satisfied by intent alone — it requires CI checks, BLOCKED_ON files, or other tooling that prevents implementation-creep into design-only PRs. Convention-only enforcement degrades with team turnover.

**Anti-pattern this prevents.** Implementation work accumulates while waiting for measurement, then re-litigates design decisions when measurement results arrive. The institutional cost is wasted implementation effort if measurement fails; the institutional value is design-as-permanent-artifact if measurement succeeds OR fails.
