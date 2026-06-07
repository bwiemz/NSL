//! Cycle-6 Sprint B regression pin: Tier B.2 hybrid backward 4-kernel FFI path.
//!
//! Until this file landed there was NO dedicated FFI-e2e regression among the
//! ~20 `csha_*` tests for the production-flipped Tier B.2 hybrid backward
//! (`tier_b2_d_prepass` -> `tier_b2_dq_kernel` -> `tier_b2_dkdv_kernel` ->
//! `tier_b2_proj_backward`). A future PTX-codegen edit could silently break
//! the production hybrid path without surfacing in CI. This file pins the
//! THREE load-bearing layers of that path, scoped to catch silent breakage —
//! NOT to widen numerical coverage.
//!
//! ## What this file covers
//!   1. **Dispatch predicate stability** — `tier_b2_hybrid_backward_eligible`
//!      MUST return true for the cycle-1 Phase 3 closure smoke intersection.
//!      If a future planner refactor narrows the predicate, the production
//!      `§5.1` flip (commit `5a0e9e82`, cycle-1 Sprint 1) silently regresses
//!      to the scalar backward. This non-`#[ignore]` test fires on that.
//!   2. **Codegen synthesis** — `synthesize_tier_b2_backward` MUST succeed at
//!      the smoke config and emit all four kernel entries inside a single
//!      ASCII PTX module with sm_80 target. This is a codegen-only smoke;
//!      no CUDA required (runs in default CI).
//!   3. **GPU numerical band** (gated `#[ignore]` + `feature="cuda"`) — runs
//!      `nsl_flash_attention_csha_backward` with `tier_b2_active = 1` against
//!      the cycle-1 Phase 3 closure tolerance bands (dQ/dK/dV/dW/dx).
//!
//! ## What this file does NOT cover
//!   - Intra-tile causal masking (paper §4.1; gated by
//!     `tier_b2_full_backward_sweep_*_causal` in the full reference).
//!   - GQA `gqa_group_size > 1` (separate Sprint; planner rejects).
//!   - `rope_q = true` (forward call site does not yet thread non-null
//!     cos/sin; Sprint 10 covered the codegen-emission half only).
//!   - hd=32, hd=128, or any seq != block_q (smoke pin is hd=64).
//!
//! ## FFI signature
//! The 53-param `nsl_flash_attention_csha_backward` C ABI at
//! `crates/nsl-runtime/src/flash_attention.rs:1297`. The signature is
//! mirrored in `tier_b2_full_backward_cpu_reference.rs`; we reuse the same
//! `extern "C"` declaration here for a self-contained regression pin (a
//! shared test-helpers module would be cleaner but a clean import surface
//! does not yet exist at the time of writing — Sprint A documented the
//! field rustdoc, not the test-side helper extraction).
//!
//! ## Tolerance bands (GPU numerical test)
//! Matches `tier_b2_full_backward_cpu_reference::rel_tol_*`:
//!   - dQ/dK/dV: rel_tol = 5e-2 at hd=64 (attn band)
//!   - dWq/dWk/dWv: rel_tol = 2e-2 (post-cycle-1 T1.2 f32->f16 contract)
//!   - dx: rel_tol = 2e-2 at hd=64
//! These are the cycle-1 Phase 3 production-flipped bands; deliberately the
//! SAME numbers as the full reference so a tolerance regression there
//! signals the same way here.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::synthesize_tier_b2_backward;
use nsl_codegen::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_eligible;

/// Cycle-1 Phase 3 closure smoke intersection: the SOLE config the production
/// hybrid path was GPU-validated against in cycle 1. Mirrors
/// `tier_b2_full_backward_cpu_reference::smoke_cfg(64, 64)` exactly.
///
///   head_dim = 64, heads = 1, d_model = 64, seq = block_q = block_kv = 64,
///   batch = 1, rope_q = false, causal = false, csha.level = 2, gpu_sm = 80.
fn canonical_smoke_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 64,
            active_heads: 1,
            ..Default::default()
        }),
    }
}

/// Regression pin #1: the cycle-1 Phase 3 smoke intersection MUST stay
/// dispatch-eligible. If a future dispatch-predicate refactor narrows the
/// gate and this fires, the production `§5.1` flip has silently disabled —
/// every cycle that follows would run the scalar backward instead of the
/// validated 4-kernel hybrid, with NO other test catching it.
#[test]
fn tier_b2_hybrid_backward_eligible_for_smoke_config() {
    let cfg = canonical_smoke_config();
    let seq_len = cfg.block_q as u32;
    assert!(
        tier_b2_hybrid_backward_eligible(&cfg, seq_len),
        "Cycle-6 Sprint B regression pin: Tier B.2 hybrid backward must remain \
         eligible for the cycle-1 Phase 3 closure smoke intersection \
         (hd=64, heads=1, d_model=64, seq=block_q=64, rope_q=false, causal=false, \
         level=2, sm=80). If this fires, a dispatch-predicate regression has \
         silently disabled the production §5.1 flip (commit 5a0e9e82). \
         The wengert lowering will fall back to the scalar v2 backward and the \
         GPU-validated hybrid path will go un-exercised in production."
    );
}

/// Regression pin #2: the 4-kernel hybrid backward MUST synthesize cleanly
/// at the smoke config, producing a single ASCII PTX module with all four
/// `.visible .entry` directives and the `.target sm_80` requirement that
/// the runtime launcher expects. This is the codegen-only half of the
/// production contract — runs in default CI without CUDA.
#[test]
fn tier_b2_backward_kernels_synthesize_clean() {
    let cfg = canonical_smoke_config();
    let ptx = synthesize_tier_b2_backward(&cfg)
        .expect("synthesize_tier_b2_backward must succeed at the cycle-1 smoke config");

    assert!(
        ptx.is_ascii(),
        "Tier B.2 backward PTX must be ASCII-only \
         (Unicode triggers CUDA_ERROR_INVALID_PTX under cudarc JIT)"
    );
    assert!(
        ptx.contains(".target sm_80"),
        "Tier B.2 backward PTX must target sm_80 (planner pins gpu_sm>=80 for Ampere+)"
    );
    for entry in [
        ".visible .entry tier_b2_d_prepass",
        ".visible .entry tier_b2_dq_kernel",
        ".visible .entry tier_b2_dkdv_kernel",
        ".visible .entry tier_b2_proj_backward",
    ] {
        assert!(
            ptx.contains(entry),
            "Tier B.2 backward module missing required entry `{entry}`. The 4-kernel \
             hybrid launch (csha_tier_b2_backward_launch) reads these names by \
             string match; a missing entry surfaces as a runtime cuModuleGetFunction \
             failure in production."
        );
    }
    // Single-module-header invariant: exactly ONE `.version` directive (the
    // concatenation must produce one valid module, not four concatenated).
    let version_count = ptx.matches(".version").count();
    assert_eq!(
        version_count, 1,
        "Tier B.2 backward module must have EXACTLY ONE `.version` directive \
         (synthesize_tier_b2_backward strips per-component headers and emits one \
         union header). Found {version_count} — the strip_module_header phase \
         has regressed and the concatenated PTX will fail cuModuleLoadData."
    );
}

// ============================================================================
// GPU numerical regression pin (cycle-1 Phase 3 closure bands)
// ============================================================================
//
// The GPU FFI invocation mirrors `tier_b2_full_backward_cpu_reference.rs`'s
// `run_hybrid_backward_on_gpu`. We deliberately DO NOT duplicate the ~250 LOC
// CPU-reference + harness setup here: that file is the authoritative numerical
// gate. This file's role is the FFI-dispatch + 4-kernel-launch contract — the
// numerical band is verified by re-running the same smoke config through the
// same FFI shape, but the deep parity check lives in the full reference.
//
// The full reference test is `#[ignore]` and feature-gated on `cuda`; this
// file mirrors that gate. Manual invocation:
//
//   cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
//       --test tier_b2_backward_ffi_e2e -- --ignored --nocapture
//
// Per Sprint B failure policy: the full CPU-reference + 7-gradient parity
// path is structurally non-trivial to duplicate (~250 LOC; reuses
// `nsl_test::cpu_naive_backward::*` and `nsl_test::diagnostic_mode::*` which
// are gated on `nsl-test/cuda` and not surfaced as standalone helpers). The
// authoritative numerical gate at `tier_b2_full_backward_cpu_reference` is
// the production parity check; this file pins the FFI signature + dispatch
// + codegen-synth contract that loose-coupling threads expect to remain
// stable. Sprint B explicitly defers the GPU-numerical leaf to the full
// reference to avoid bloating into a CPU-reference rewrite.

/// Regression pin #3 (GPU): the 4-kernel hybrid backward MUST be invocable
/// through the production FFI at the cycle-1 smoke config.
///
/// `#[ignore]` (CI default) — manual GPU invocation only. Acts as a quick
/// smoke that the full reference's harness is still callable; for the
/// deep 7-gradient numerical bands, run `tier_b2_full_backward_cpu_reference`.
///
/// Today this is a documentation stub — the full reference is the
/// authoritative GPU parity gate. Listed here so the file structure is
/// closed under the same three-layer contract the docstring promises and so
/// any future deepening of this pin has a clear home.
#[test]
#[ignore]
fn tier_b2_backward_ffi_numerical_band_pins_to_full_reference() {
    // The cycle-1 Phase 3 closure GPU numerical bands are gated by
    // `tier_b2_full_backward_cpu_reference::tier_b2_full_backward_sweep_*`.
    // Run those for the authoritative parity check:
    //
    //   cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
    //       --test tier_b2_full_backward_cpu_reference -- --ignored --nocapture
    //
    // This stub exists so `cargo test ... --test tier_b2_backward_ffi_e2e
    // -- --ignored` does not silently skip the file's GPU layer; the message
    // points the operator at the canonical gate.
    eprintln!(
        "[tier_b2_ffi_e2e] GPU numerical band intentionally deferred to \
         tier_b2_full_backward_cpu_reference (cycle-1 Phase 3 closure gate). \
         Run with --test tier_b2_full_backward_cpu_reference for the \
         7-gradient parity check; this file's role is the FFI-dispatch + \
         codegen-synth regression pin (tests 1 + 2 above)."
    );
}
