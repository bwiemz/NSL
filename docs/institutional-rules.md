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
