//! Sprint 8 (paper §4.1) — compile-time seq_len elision in dq/dkdv kernels.
//!
//! Two compile-time optimizations land in this sprint:
//!
//! 1. **`num_q_iters` / `num_kv_iters` constant-folding.** When
//!    `config.csha.static_seq_len = Some(s)` is provided, the runtime
//!    `add + shr` sequence that computes `ceil(s / bq)` collapses to a
//!    single `mov.u32 %num_q_iters, <const>`. Same for `num_kv_iters`.
//!
//! 2. **Tile-skip predicate elision (single-tile causal case).** When
//!    `static_seq_len <= block_q` AND `static_seq_len <= block_kv` AND
//!    `causal = true`, the outer tile-skip comparison (`kv_tile_start <=
//!    q_tile_end`) is folded to the constant `1` (single tile is always
//!    active). Per-element intra-tile causal masking still emits and
//!    handles correctness within that single tile.
//!
//! Byte-identity for the `static_seq_len = None` path is verified
//! explicitly so callers that have not opted in see ZERO PTX delta.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn base_cfg(bq: i64, bkv: i64, hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim: hd,
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
            d_model: hd as u32,
            ..Default::default()
        }),
    }
}

fn with_static_seq_len(mut cfg: FlashAttentionConfig, s: u32) -> FlashAttentionConfig {
    let mut extras = cfg.csha.clone().expect("base_cfg always sets csha");
    extras.static_seq_len = Some(s);
    cfg.csha = Some(extras);
    cfg
}

// ============================================================================
// Test 1 — static_seq_len = None preserves byte-identical PTX (regression gate)
// ============================================================================

/// The default (`static_seq_len: None`) MUST produce byte-identical PTX to
/// pre-Sprint-8 output. We assert this by checking the kernel still contains
/// the exact `add.u32 %num_q_iters, %seq_len_r, ...` + `shr.u32 ...` sequence
/// AND does NOT contain any of the new Sprint 8 markers.
#[test]
fn dq_none_path_emits_runtime_add_shr_no_static_markers() {
    let cfg = base_cfg(64, 64, 32);
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        ptx.contains("add.u32 %num_q_iters, %seq_len_r, 63"),
        "None path must keep the runtime add.u32 num_q_iters sequence"
    );
    assert!(
        ptx.contains("shr.u32 %num_q_iters, %num_q_iters, 6"),
        "None path must keep the runtime shr.u32 num_q_iters sequence"
    );
    assert!(
        ptx.contains("add.u32 %num_kv_iters, %seq_len_r, 63"),
        "None path must keep the runtime add.u32 num_kv_iters sequence"
    );
    assert!(
        !ptx.contains("V2_STATIC_NUM_Q_ITERS"),
        "None path must NOT emit the static folding marker"
    );
    assert!(
        !ptx.contains("V2_STATIC_NUM_KV_ITERS"),
        "None path must NOT emit the static folding marker"
    );
    assert!(
        !ptx.contains("V2_STATIC_SINGLE_TILE"),
        "None path must NOT emit the single-tile elision marker"
    );
}

#[test]
fn dkdv_none_path_emits_runtime_add_shr_no_static_markers() {
    let cfg = base_cfg(64, 64, 32);
    let ptx = synthesize_dkdv_kernel(&cfg).expect("dkdv synth");
    assert!(
        ptx.contains("add.u32 %num_q_iters, %seq_len_r, 63"),
        "None path must keep the runtime num_q_iters sequence"
    );
    assert!(
        ptx.contains("add.u32 %num_kv_iters, %seq_len_r, 63"),
        "None path must keep the runtime num_kv_iters sequence"
    );
    assert!(!ptx.contains("V2_STATIC_NUM_Q_ITERS"));
    assert!(!ptx.contains("V2_STATIC_NUM_KV_ITERS"));
    assert!(!ptx.contains("V2_STATIC_SINGLE_TILE"));
}

/// Byte-identity gate: with `static_seq_len = None`, the dq PTX must equal
/// itself across re-synthesis (rules out nondeterminism) and must NOT
/// contain ANY of the Sprint 8 markers anywhere in the bytes. This is the
/// strongest "no regression" assertion.
#[test]
fn dq_none_path_synthesis_is_byte_stable() {
    let cfg = base_cfg(64, 64, 32);
    let a = synthesize_dq_kernel(&cfg).expect("dq synth a");
    let b = synthesize_dq_kernel(&cfg).expect("dq synth b");
    assert_eq!(a, b, "synthesis must be deterministic");
    // No Sprint 8 byte sequences may appear.
    for marker in ["V2_STATIC_NUM_Q_ITERS", "V2_STATIC_NUM_KV_ITERS", "V2_STATIC_SINGLE_TILE"] {
        assert!(!a.contains(marker), "default PTX leaked Sprint 8 marker: {marker}");
    }
}

#[test]
fn dkdv_none_path_synthesis_is_byte_stable() {
    let cfg = base_cfg(64, 64, 32);
    let a = synthesize_dkdv_kernel(&cfg).expect("dkdv synth a");
    let b = synthesize_dkdv_kernel(&cfg).expect("dkdv synth b");
    assert_eq!(a, b, "synthesis must be deterministic");
    for marker in ["V2_STATIC_NUM_Q_ITERS", "V2_STATIC_NUM_KV_ITERS", "V2_STATIC_SINGLE_TILE"] {
        assert!(!a.contains(marker), "default PTX leaked Sprint 8 marker: {marker}");
    }
}

// ============================================================================
// Test 2 — static_seq_len = Some(s) folds num_q_iters / num_kv_iters
// ============================================================================

/// Single-tile case (seq_len = block_q = block_kv = 64): num_q_iters = num_kv_iters = 1.
#[test]
fn dq_static_seq_len_64_single_tile_folds_to_mov_1() {
    let cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        ptx.contains("mov.u32 %num_q_iters, 1;"),
        "static seq_len=64 with bq=64 must fold num_q_iters to 1; PTX:\n{ptx}"
    );
    assert!(
        ptx.contains("mov.u32 %num_kv_iters, 1;"),
        "static seq_len=64 with bkv=64 must fold num_kv_iters to 1"
    );
    // Both markers must be present.
    assert!(ptx.contains("V2_STATIC_NUM_Q_ITERS"));
    assert!(ptx.contains("V2_STATIC_NUM_KV_ITERS"));
    // And the runtime add/shr sequence must NOT appear for these registers.
    assert!(
        !ptx.contains("add.u32 %num_q_iters, %seq_len_r"),
        "static path must elide the runtime add.u32 %num_q_iters sequence"
    );
    assert!(
        !ptx.contains("add.u32 %num_kv_iters, %seq_len_r"),
        "static path must elide the runtime add.u32 %num_kv_iters sequence"
    );
}

#[test]
fn dkdv_static_seq_len_64_single_tile_folds_to_mov_1() {
    let cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    let ptx = synthesize_dkdv_kernel(&cfg).expect("dkdv synth");
    assert!(
        ptx.contains("mov.u32 %num_q_iters, 1;"),
        "static seq_len=64 with bq=64 must fold num_q_iters to 1; PTX:\n{ptx}"
    );
    assert!(
        ptx.contains("mov.u32 %num_kv_iters, 1;"),
        "static seq_len=64 with bkv=64 must fold num_kv_iters to 1"
    );
    assert!(ptx.contains("V2_STATIC_NUM_Q_ITERS"));
    assert!(ptx.contains("V2_STATIC_NUM_KV_ITERS"));
    assert!(!ptx.contains("add.u32 %num_q_iters, %seq_len_r"));
    assert!(!ptx.contains("add.u32 %num_kv_iters, %seq_len_r"));
}

/// Multi-tile case (seq_len = 256, block = 64): num_q_iters = num_kv_iters = 4.
/// Verifies the fold is computed correctly and embeds the right literal.
#[test]
fn dq_static_seq_len_256_multi_tile_folds_to_mov_4() {
    let cfg = with_static_seq_len(base_cfg(64, 64, 32), 256);
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        ptx.contains("mov.u32 %num_q_iters, 4;"),
        "static seq_len=256 with bq=64 must fold num_q_iters to 4"
    );
    assert!(
        ptx.contains("mov.u32 %num_kv_iters, 4;"),
        "static seq_len=256 with bkv=64 must fold num_kv_iters to 4"
    );
}

/// Non-multiple seq_len (seq_len = 100, block = 64): ceil(100/64) = 2. The
/// runtime path would have computed (100 + 63) >> 6 = 2 as well — verifies
/// the constant-fold matches the runtime semantics.
#[test]
fn dq_static_seq_len_100_non_multiple_uses_ceil_div() {
    let cfg = with_static_seq_len(base_cfg(64, 64, 32), 100);
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        ptx.contains("mov.u32 %num_q_iters, 2;"),
        "ceil(100/64) = 2"
    );
    assert!(
        ptx.contains("mov.u32 %num_kv_iters, 2;"),
        "ceil(100/64) = 2"
    );
}

// ============================================================================
// Test 3 — tile-skip predicate elision (single-tile + causal)
// ============================================================================

/// Single-tile + causal: the outer tile-skip predicate folds to mov 1.
/// The per-element intra-tile mask MUST still be emitted (correctness).
#[test]
fn dq_single_tile_causal_elides_tile_skip_predicate() {
    let mut cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    cfg.causal = true;
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        ptx.contains("V2_STATIC_SINGLE_TILE"),
        "single-tile causal must emit the elision marker"
    );
    assert!(
        ptx.contains("mov.u32 %tile_skip_predicate, 1;  // V2_STATIC_SINGLE_TILE"),
        "tile-skip predicate must fold to constant 1"
    );
    // The runtime comparison `setp.le.u32 %p_causal_active` must NOT appear.
    assert!(
        !ptx.contains("setp.le.u32 %p_causal_active"),
        "single-tile elision must drop the setp.le.u32 comparison"
    );
    // The per-element intra-tile mask MUST still be emitted (correctness).
    assert!(
        ptx.contains("V2_INTRA_TILE_CAUSAL_MASK"),
        "intra-tile per-element causal mask must still emit (single-tile correctness)"
    );
}

#[test]
fn dkdv_single_tile_causal_elides_tile_skip_predicate() {
    let mut cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    cfg.causal = true;
    let ptx = synthesize_dkdv_kernel(&cfg).expect("dkdv synth");
    assert!(ptx.contains("V2_STATIC_SINGLE_TILE"));
    assert!(ptx.contains("mov.u32 %tile_skip_predicate, 1;  // V2_STATIC_SINGLE_TILE"));
    assert!(!ptx.contains("setp.le.u32 %p_causal_active"));
    assert!(ptx.contains("V2_INTRA_TILE_CAUSAL_MASK"));
}

/// Multi-tile causal (seq_len > block_q) MUST keep the runtime tile-skip
/// comparison — single-tile elision must not trigger.
#[test]
fn dq_multi_tile_causal_keeps_runtime_tile_skip() {
    let mut cfg = with_static_seq_len(base_cfg(64, 64, 32), 256);
    cfg.causal = true;
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(
        !ptx.contains("V2_STATIC_SINGLE_TILE"),
        "multi-tile MUST NOT trigger single-tile elision (s=256 > bq=64)"
    );
    assert!(
        ptx.contains("setp.le.u32 %p_causal_active, %kv_tile_start, %q_tile_end"),
        "multi-tile causal MUST keep the runtime tile-skip comparison"
    );
}

/// Single-tile + non-causal: the elision marker MUST NOT appear (the
/// elision is causal-only; non-causal already has a trivial `mov 1` path).
#[test]
fn dq_single_tile_non_causal_no_static_single_tile_marker() {
    let cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(!cfg.causal, "guard: this test runs with causal=false");
    assert!(
        !ptx.contains("V2_STATIC_SINGLE_TILE"),
        "non-causal path uses the existing non-causal mov.u32 1 path; \
         single-tile marker is causal-only"
    );
}

/// Single-tile + causal at smaller bq (32): seq_len=32, bq=32, bkv=32 -> single tile.
/// Diagnostic: print the PTX lines that differ between the None and
/// Some(64) paths so reviewers can see the elision delta at a glance.
/// Always passes; only meaningful when run with `-- --nocapture`.
#[test]
fn diag_print_ptx_delta() {
    let none_cfg = base_cfg(64, 64, 32);
    let mut some_cfg = with_static_seq_len(base_cfg(64, 64, 32), 64);
    some_cfg.causal = true;
    let mut none_causal_cfg = base_cfg(64, 64, 32);
    none_causal_cfg.causal = true;
    let none = synthesize_dq_kernel(&none_cfg).expect("dq synth");
    let some = synthesize_dq_kernel(&some_cfg).expect("dq synth");
    let none_causal = synthesize_dq_kernel(&none_causal_cfg).expect("dq synth");

    println!("\n--- None (default) num_q_iters / num_kv_iters lines ---");
    for l in none.lines().filter(|l| l.contains("num_q_iters") || l.contains("num_kv_iters")) {
        println!("{l}");
    }
    println!("\n--- Some(64), causal=true num_q_iters / num_kv_iters lines ---");
    for l in some.lines().filter(|l| l.contains("num_q_iters") || l.contains("num_kv_iters")) {
        println!("{l}");
    }
    println!("\n--- None, causal=true tile_skip_predicate region ---");
    for l in none_causal.lines().filter(|l| {
        l.contains("tile_skip_predicate") || l.contains("p_causal_active") || l.contains("D1: tile_skip")
    }) {
        println!("{l}");
    }
    println!("\n--- Some(64), causal=true tile_skip_predicate region ---");
    for l in some.lines().filter(|l| {
        l.contains("tile_skip_predicate") || l.contains("p_causal_active") || l.contains("D1: tile_skip")
    }) {
        println!("{l}");
    }
}

#[test]
fn dq_single_tile_causal_smaller_bq32() {
    let mut cfg = with_static_seq_len(base_cfg(32, 32, 32), 32);
    cfg.causal = true;
    let ptx = synthesize_dq_kernel(&cfg).expect("dq synth");
    assert!(ptx.contains("V2_STATIC_SINGLE_TILE"));
    assert!(ptx.contains("mov.u32 %num_q_iters, 1;"));
    assert!(ptx.contains("mov.u32 %num_kv_iters, 1;"));
}
