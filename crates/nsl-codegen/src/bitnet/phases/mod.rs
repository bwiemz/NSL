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

// Internal: pub only for test introspection (same rationale as absmean_quant).
// IR-001 is documentation-enforced for this emitter; external callers should
// use quantized_ternary_gemm (the fused public path).
pub mod ternary_gemm;

// Public: the fused absmax + ternary GEMM emitter. The only ternary-GEMM
// path that external callers (CSHA-fused mode, future M35.x) should compose
// against per IR-001.
pub mod quantized_ternary_gemm;

// Public: finalize epilogue (FP32 -> F16/BF16 cast, optional bias/residual
// add, HBM write). Composed by `synthesize_kernel` after the fused GEMM.
pub mod finalize;
