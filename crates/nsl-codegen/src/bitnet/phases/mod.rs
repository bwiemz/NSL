//! BitNet phase emitters. See `README.md` for the public/internal visibility split.

pub mod packed_load;

// absmean_quant is `pub` (not `pub(super)`) so integration tests in
// `crates/nsl-codegen/tests/` can introspect the emitted PTX as structural
// snapshots. Integration tests live in separate compilation units and
// cannot see `pub(crate)`/`pub(super)` items.
//
// The IR-001 invariant ("only fused `quantized_ternary_gemm` should be
// publicly callable for activation-quant + ternary GEMM") is enforced at
// the documentation level: the module docstring of `absmean_quant.rs`
// directs external callers to `quantized_ternary_gemm::emit` instead.
// This mirrors the visibility/discipline pattern already used by
// `packed_load` above.
pub mod absmean_quant;
