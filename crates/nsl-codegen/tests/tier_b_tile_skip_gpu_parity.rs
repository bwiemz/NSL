//! PCA Tier B tile-skip: GPU parity gate for the Tier-B-on forward kernel.
//!
//! This is the packed-shape parity gate demanded by the Stage-C deferral in
//! `ensure_sdpa_fwd_variant_table` (`PROMOTE_TIER_B`): the benchmark that
//! first launched Tier-B in production found its output deviated ~1e-2 at
//! s=1024/hd=64 while the base segment-masked kernel matched the decomposed
//! oracle at ~3e-6.
//!
//! Strategy: drive `nsl_sdpa_fused_forward` twice on identical inputs —
//! once with the Tier-B sentinel pair zeroed (base segment-masked kernel)
//! and once with the Tier-B-on PTX (the runtime gate selects it whenever
//! the pair is non-null, the segment pointer is non-null, and
//! `seq_len ∈ [TIER_B_SEQ_LEN_FLOOR, TIER_B_MAX_BAKED_SEQ_LEN]`) — and
//! compare (out, lse) across segment patterns that isolate the skip logic:
//!
//!   * `no_skip`   — single document (all ids 0): the disjointness predicate
//!     can never fire, so the Tier-B kernel executes the exact same math as
//!     the base kernel. Any deviation is an emission side effect (register /
//!     SMEM corruption from the preamble or predicate), not skip logic.
//!   * `aligned`   — 192-token documents (boundaries on 64-token tile
//!     edges): the benchmark shape. Whole (q,kv) tile pairs are disjoint.
//!   * `unaligned` — 100-token documents (boundaries mid-tile): mixed tiles
//!     survive the predicate; only strictly-disjoint pairs skip.
//!   * `many_short` — 40-token documents: high skip density.
//!
//! Each pattern also runs a small-batch multi-head shape (B=4, H=8) that
//! mirrors the per-batch-row launch contract, comparing Tier-B against the
//! base kernel (the base kernel is the reference — it has its own oracle
//! gate in `sdpa_fused_forward_gpu_parity.rs`).
//!
//! Requires an NVIDIA GPU + driver; `#[ignore]` by default. Run with:
//!   cargo test -p nsl-codegen --features cuda --test tier_b_tile_skip_gpu_parity --release -- --ignored --nocapture
//!
//! The launch-count proof is serialized through an in-process mutex (see
//! `assert_tier_b_matches_base`), so the exact-delta counter assertion
//! holds even under libtest's default parallel scheduling — but
//! `--test-threads=1` remains the repo-wide GPU-gate convention.

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::shared_mem_bytes_selected;
use nsl_codegen::pca_tier_b::emit_tier_b_variants_for_config;
use nsl_codegen::pca_tilerange::tier_b_range_table_bytes;

use nsl_runtime::flash_attention::{nsl_sdpa_fused_launch_count, nsl_sdpa_fused_forward};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::pca_tier_b_runtime::TIER_B_MAX_BAKED_SEQ_LEN;
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_cuda_jit_log};

const BLOCK_Q: i64 = 64;
const BLOCK_KV: i64 = 64;
const D: usize = 64; // production Stage-C head_dim where the deviation was seen

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    nsl_cuda_init() == 0
}

fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed as u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn production_config() -> FlashAttentionConfig {
    // EXACTLY the config ensure_sdpa_fwd_variant_table admits for hd=64
    // (causal=true, segment_masked=true): 64x64 tiles, plain family.
    FlashAttentionConfig {
        block_q: BLOCK_Q,
        block_kv: BLOCK_KV,
        head_dim: D as i64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 90,
        segment_masked: true,
        csha: None,
        checkpoint: None,
    }
}

fn gpu_tensor(shape: &[i64], vals: &[f32]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "GPU tensor alloc failed");
    let dptr = nsl_tensor_data_ptr(t);
    nsl_test_cuda_h2d(dptr, vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

fn host_segment_tensor(b: usize, s: usize, seg: &[u16]) -> i64 {
    assert_eq!(seg.len(), b * s);
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, b as i64);
    nsl_list_push(shape_list, s as i64);
    let t = nsl_tensor_zeros_on(shape_list, 0);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "CPU tensor alloc failed");
    let dptr = nsl_tensor_data_ptr(t) as *mut f32;
    for (i, &v) in seg.iter().enumerate() {
        unsafe { *dptr.add(i) = v as f32 };
    }
    t
}

/// Segment ids for `[b, s]` where each row packs documents of `doc_len`
/// tokens (last document ragged). Rows get different boundary phases when
/// `stagger` is set so the per-batch-row launch contract is exercised.
fn doc_pattern(b: usize, s: usize, doc_len: usize, stagger: bool) -> Vec<u16> {
    let mut seg = vec![0u16; b * s];
    for row in 0..b {
        let phase = if stagger { (row * doc_len) / (b.max(1)) } else { 0 };
        for i in 0..s {
            seg[row * s + i] = ((i + phase) / doc_len) as u16;
        }
    }
    seg
}

struct FusedResult {
    out: Vec<f32>,
    lse: Vec<f32>,
}

/// Launch the fused forward with or without the Tier-B pair and read back.
fn run_variant(
    b: usize,
    h: usize,
    s: usize,
    seg: &[u16],
    tier_b: bool,
) -> FusedResult {
    let total = b * h * s * D;
    let q_host = det_seq(0xA1B2_C3D4 ^ (s as u32), total);
    let k_host = det_seq(0x1234_5678 ^ (s as u32), total);
    let v_host = det_seq(0xDEAD_BEEF ^ (s as u32), total);

    let q_t = gpu_tensor(&[b as i64, h as i64, s as i64, D as i64], &q_host);
    let k_t = gpu_tensor(&[b as i64, h as i64, s as i64, D as i64], &k_host);
    let v_t = gpu_tensor(&[b as i64, h as i64, s as i64, D as i64], &v_host);
    let seg_t = host_segment_tensor(b, s, seg);

    let config = production_config();
    let emission = emit_tier_b_variants_for_config(&config);

    let prep = |mut ptx: Vec<u8>| {
        while ptx.last() == Some(&0) {
            ptx.pop();
        }
        if ptx.last() != Some(&b'\n') {
            ptx.push(b'\n');
        }
        ptx.push(0);
        ptx
    };
    let base_ptx = prep(emission.base_ptx.clone());
    // PTX entry name inside the Tier-B-on module is the BASE v2 name — the
    // `_tier_b_max{N}` suffix is a codegen-cache key only (Stage-C
    // benchmark discovery, mirrored by ensure_sdpa_fwd_variant_table).
    let kernel_name = CString::new(emission.base_kernel_name.clone()).unwrap();

    // One SMEM value serves both launches, exactly as the production table
    // computes it: base request widened to cover the Tier-B range table at
    // the baked max seq.
    let base_smem = shared_mem_bytes_selected(&config) as i64;
    let tb_off = nsl_codegen::flash_attention_v2::smem_layout::tier_b_range_table_offset(
        &config,
        nsl_codegen::flash_attention_v2::smem_layout::Direction::Forward,
    );
    let smem = base_smem.max((tb_off + tier_b_range_table_bytes(&config, TIER_B_MAX_BAKED_SEQ_LEN)) as i64);

    let (tb_ptx_holder, tb_ptx_ptr, tb_name_ptr) = if tier_b {
        let on = prep(
            emission
                .tier_b_on_ptx
                .clone()
                .expect("production config must admit Tier-B emission"),
        );
        let p = on.as_ptr() as i64;
        (Some(on), p, kernel_name.as_ptr() as i64)
    } else {
        (None, 0i64, 0i64)
    };

    let scale = 1.0f32 / (D as f32).sqrt();
    let list_ptr = nsl_sdpa_fused_forward(
        q_t,
        k_t,
        v_t,
        scale.to_bits() as i64,
        1,
        seg_t,
        base_ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        tb_ptx_ptr,
        tb_name_ptr,
        BLOCK_Q,
        BLOCK_KV,
        smem,
    );
    if list_ptr == 0 {
        let probe = if tier_b {
            tb_ptx_holder.as_ref().unwrap().as_ptr() as i64
        } else {
            base_ptx.as_ptr() as i64
        };
        let log_ptr = nsl_test_cuda_jit_log(probe);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        panic!(
            "nsl_sdpa_fused_forward DECLINED (tier_b={tier_b}, b={b} h={h} s={s}) — JIT log:\n{log}"
        );
    }

    let out_t = nsl_list_get(list_ptr, 0);
    let lse_t = nsl_list_get(list_ptr, 1);
    let mut out = vec![0.0f32; total];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(out_t),
        (total * 4) as i64,
    );
    let total_lse = b * h * s;
    let mut lse = vec![0.0f32; total_lse];
    nsl_test_cuda_d2h(
        lse.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(lse_t),
        (total_lse * 4) as i64,
    );

    nsl_tensor_free(out_t);
    nsl_tensor_free(lse_t);
    nsl_list_free(list_ptr);
    nsl_tensor_free(q_t);
    nsl_tensor_free(k_t);
    nsl_tensor_free(v_t);
    nsl_tensor_free(seg_t);

    // The holder must stay alive through the launch (the FFI copies the
    // PTX into the driver during cuModuleLoadData, which happens inside
    // the call — this is just belt-and-braces against NLL dropping it).
    drop(tb_ptx_holder);

    FusedResult { out, lse }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = (ai - bi).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    (max_abs, max_idx)
}

/// Tolerance for tier-b vs base: both outputs pass through the kernel's
/// f16 output staging, so exact-math-equal paths may still differ by one
/// f16 ulp after widening; with the test's bounded inputs (|out| <= ~0.5)
/// one ulp is <= ~4.9e-4. Skip patterns additionally reorder the online
/// softmax's running max/sum updates at the ~1e-6 level pre-quantization.
/// 5e-4 is therefore ~one-ulp tight (the 2026-07-14 promotion run in fact
/// observed max_abs == 0.0 exactly on every pattern) while the original
/// deferral bug sat at ~1e-2 — 20x above this gate.
const TB_VS_BASE_TOL: f32 = 5e-4;

/// Serializes the (base run, counter snapshot, tier-b run, counter check)
/// window: the launch counter is process-global, so a sibling test's
/// tier-b launch inside the window would make the exact-delta assertion
/// spuriously fail — or, worse, vouch for a run whose own dispatch fell
/// back to base (the very bug the counter proves absent).
static LAUNCH_PROOF_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_tier_b_matches_base(name: &str, b: usize, h: usize, s: usize, seg: &[u16]) {
    let _serial = LAUNCH_PROOF_GATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let base = run_variant(b, h, s, seg, false);
    // Launch-count proof: the Tier-B call below must increment the
    // Tier-B counter (variant 1), or the comparison is vacuous — a
    // dispatch bug silently selecting the base kernel would "pass"
    // bitwise-equal without ever running the Tier-B PTX.
    let tb_launches_before = nsl_sdpa_fused_launch_count(1);
    let tb = run_variant(b, h, s, seg, true);
    let tb_launches_after = nsl_sdpa_fused_launch_count(1);
    assert_eq!(
        tb_launches_after,
        tb_launches_before + 1,
        "{name}: the Tier-B variant was NOT launched (counter {tb_launches_before} -> \
         {tb_launches_after}) — runtime gate declined or dispatch fell back to base"
    );
    let (out_err, out_idx) = max_abs_diff(&base.out, &tb.out);
    let (lse_err, lse_idx) = max_abs_diff(&base.lse, &tb.lse);
    eprintln!(
        "{name} (b={b} h={h} s={s}): out max_abs={out_err:.7} at {out_idx} \
         (base={}, tb={}); lse max_abs={lse_err:.7} at {lse_idx} \
         (base={}, tb={})",
        base.out[out_idx], tb.out[out_idx], base.lse[lse_idx], tb.lse[lse_idx],
    );
    assert!(
        tb.out.iter().all(|x| x.is_finite()),
        "{name}: tier-b out contains non-finite values"
    );
    assert!(
        out_err < TB_VS_BASE_TOL,
        "{name}: tier-b OUT deviates from base kernel: {out_err} at idx {out_idx}"
    );
    assert!(
        lse_err < TB_VS_BASE_TOL,
        "{name}: tier-b LSE deviates from base kernel: {lse_err} at idx {lse_idx}"
    );
}

/// Single document — the skip predicate can never fire. Any deviation here
/// is an emission side effect, NOT skip logic.
#[test]
#[ignore]
fn tier_b_no_skip_single_doc_s1024() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (1, 1, 1024);
    let seg = vec![0u16; b * s];
    assert_tier_b_matches_base("no_skip_single_doc", b, h, s, &seg);
}

/// 192-token documents (tile-aligned boundaries) — the benchmark shape.
#[test]
#[ignore]
fn tier_b_aligned_docs_s1024() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (1, 1, 1024);
    let seg = doc_pattern(b, s, 192, false);
    assert_tier_b_matches_base("aligned_docs_192", b, h, s, &seg);
}

/// 100-token documents (boundaries mid-tile).
#[test]
#[ignore]
fn tier_b_unaligned_docs_s1024() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (1, 1, 1024);
    let seg = doc_pattern(b, s, 100, false);
    assert_tier_b_matches_base("unaligned_docs_100", b, h, s, &seg);
}

/// 40-token documents — high skip density.
#[test]
#[ignore]
fn tier_b_many_short_docs_s1024() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (1, 1, 1024);
    let seg = doc_pattern(b, s, 40, false);
    assert_tier_b_matches_base("many_short_docs_40", b, h, s, &seg);
}

/// Benchmark shape: B=4, H=8, staggered per-row boundaries — exercises the
/// per-batch-row launch loop with row-distinct range tables.
#[test]
#[ignore]
fn tier_b_batch4_heads8_staggered_s1024() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (4, 8, 1024);
    let seg = doc_pattern(b, s, 192, true);
    assert_tier_b_matches_base("batch4_heads8_staggered", b, h, s, &seg);
}

/// Long-sequence spot check at s=4096 (deeper kv sweeps per q-tile).
#[test]
#[ignore]
fn tier_b_aligned_docs_s4096() {
    if !cuda_available() {
        return;
    }
    let (b, h, s) = (1, 1, 4096);
    let seg = doc_pattern(b, s, 640, false);
    assert_tier_b_matches_base("aligned_docs_640_s4096", b, h, s, &seg);
}
