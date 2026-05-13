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
// `#[doc(hidden)]` also keeps this off the rustdoc front page so external
// crate users don't see it as a public API surface.
#[doc(hidden)]
pub mod absmean_quant;

// Internal: pub only for test introspection (same rationale as absmean_quant).
// IR-001 is documentation-enforced for this emitter; external callers should
// use quantized_ternary_gemm (the fused public path). `#[doc(hidden)]` hides
// it from rustdoc to reduce the risk of mistaken external composition.
#[doc(hidden)]
pub mod ternary_gemm;

// Public: the fused absmax + ternary GEMM emitter. The only ternary-GEMM
// path that external callers (CSHA-fused mode, future M35.x) should compose
// against per IR-001.
pub mod quantized_ternary_gemm;

// Public: finalize epilogue (FP32 -> F16/BF16 cast, optional bias/residual
// add, HBM write). Composed by `synthesize_kernel` after the fused GEMM.
pub mod finalize;

// M35.2a Phase 2 stubs — gated on V-P1-D pass per BLOCKED_ON_V_P1_D.md.
// Real emit() implementations land in M35.2a implementation Stage D.3 / D.4.
#[doc(hidden)]
pub mod ternary_gemm_backward;

#[doc(hidden)]
pub mod absmax_quant_backward;

// M35.2b stub — intentionally not implemented in M35.2a; lands as part of M35.2b.
#[doc(hidden)]
pub mod quantize_shadow;
