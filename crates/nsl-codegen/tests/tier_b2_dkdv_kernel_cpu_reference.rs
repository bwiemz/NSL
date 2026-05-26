//! Layer-1 dK/dV-kernel parity tests vs CPU reference (Phase 3a Task 9).
//!
//! All GPU tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation:
//!     cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
//!         --test tier_b2_dkdv_kernel_cpu_reference -- --ignored --nocapture
//!
//! Structure:
//!   - Test 0: CPU-only s=1 analytic case (no GPU, no `#[ignore]`).
//!   - Test 1: `tier_b2_dkdv_sweep_cpu_naive` — CpuNaive forward, hd=32/64/128.
//!   - Test 2: `tier_b2_dkdv_sweep_b1_forward` — B.1 GPU forward, hd=32/64/128.
//!
//! Both GPU sweeps compare dV and dK independently against `cpu_naive_backward_dkdv`
//! (which recomputes P/dS from Q/K/V/O). Three guards per output: reference-magnitude
//! floor, zero-output guard, relative-error gate. The relative tolerance is the same
//! tiered schedule as the dQ parity gate (per spec §6.1).
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §6.1

#![cfg(feature = "cuda")]

use std::ffi::{c_void, CString};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::{
    tier_b2_dkdv_total_smem_bytes, tier_b2_effective_bq,
};
use nsl_codegen::flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_test::cpu_naive_backward::cpu_naive_backward_dkdv;
use nsl_test::diagnostic_mode::{
    compute_forward_for_test, generate_d_o, generate_forward_inputs, FSource,
};

extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64,
        name_ptr: i64,
        grid_x: i64,
        grid_y: i64,
        grid_z: i64,
        block_x: i64,
        block_y: i64,
        block_z: i64,
        args_ptr: i64,
        num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("[tier_b2_dkdv] skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("[tier_b2_dkdv] skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// Null-terminate a PTX string for cuModuleLoadData.
fn ptx_to_cstr_bytes(ptx: &str) -> Vec<u8> {
    let mut bytes = ptx.as_bytes().to_vec();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0);
    bytes
}

/// Tiered RELATIVE tolerance for dK/dV parity gate. Same schedule as dQ.
fn rel_tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 {
        8e-2
    } else if hd >= 64 {
        5e-2
    } else {
        3e-2
    }
}

/// dO scale for the dK/dV parity gate. Same as DQ_GATE_DO_SCALE: lifts the signal
/// above the f16 noise floor so the relative gate is non-vacuous.
const DQ_GATE_DO_SCALE: f32 = 1024.0;

fn cfg(bq: i64, hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, d_model: 128, ..Default::default() }),
    }
}

// Test 0 (CPU-only analytic s=1 case) lives in the NON-cuda-gated integration test
// `crates/nsl-test/tests/cpu_naive_backward_dkdv_analytic.rs` so the CPU reference
// math is regression-tested in a standard `cargo test` (no GPU/cuda feature). This
// file (#![cfg(feature="cuda")]) holds only the GPU parity sweeps below.

// ============================================================================
// GPU sweeps
// ============================================================================

#[test]
#[ignore]
fn tier_b2_dkdv_sweep_cpu_naive() {
    // Phase 3a regression gate — CpuNaive forward, hd=32/64/128.
    // Path-A schedule per spec §5.2: hd=128 uses bq=32; hd=32/64 use bq=64.
    for &hd in &[32i64, 64, 128] {
        let bq = if hd == 128 { 32 } else { 64 };
        let seq = if hd >= 128 { 128 } else { 64 };
        validate_dkdv_for_source(&cfg(bq, hd), FSource::CpuNaive, seq);
    }
}

#[test]
#[ignore]
fn tier_b2_dkdv_sweep_b1_forward() {
    // Phase 3a closure gate — B.1 GPU forward + adapter.
    // seq=32 (B.1 single-block precondition: launcher forces block_kv=32).
    //
    // INVOCATION: requires nsl-test's cuda feature, so run with
    //   cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
    //       --test tier_b2_dkdv_kernel_cpu_reference tier_b2_dkdv_sweep_b1_forward \
    //       -- --ignored --nocapture
    for &hd in &[32i64, 64, 128] {
        validate_dkdv_for_source(&cfg(32, hd), FSource::B1Forward, 32);
    }
}

// ============================================================================
// Shared parity helper
// ============================================================================

/// Run the dK/dV-kernel parity check for one forward source at a given (cfg, seq).
///
/// Sources the forward via `compute_forward_for_test`, scales dO identically to the
/// dQ gate (DQ_GATE_DO_SCALE), runs the GPU D pre-pass, then launches the dkdv kernel.
/// Compares both dV and dK independently against `cpu_naive_backward_dkdv` with three
/// guards each: reference-magnitude floor, zero-output guard, relative-error gate.
fn validate_dkdv_for_source(cfg: &FlashAttentionConfig, source: FSource, seq: usize) {
    if !cuda_available() {
        return;
    }
    let batch = 1usize;
    let heads = 1usize;
    let hd = cfg.head_dim as usize;

    let inputs = generate_forward_inputs(cfg, source, seq);
    // Scale dO so that dV and dK are substantial (non-vacuous gate).
    let d_o: Vec<half::f16> = generate_d_o(cfg, seq)
        .iter()
        .map(|x| half::f16::from_f32(x.to_f32() * DQ_GATE_DO_SCALE))
        .collect();

    // Comparator reads Q/K/V/O from fwd (same convention as dQ gate).
    let fwd = compute_forward_for_test(&inputs, cfg, source, seq);

    // D pre-pass (needed as backward input).
    let d_prepass_ptx = synthesize_d_prepass(cfg).expect("d_prepass synth");
    let d = run_d_prepass_on_gpu(&d_prepass_ptx, &d_o, &fwd.o, batch, heads, seq, hd);

    // dK/dV kernel.
    let dkdv_ptx = synthesize_dkdv_kernel(cfg).expect("dkdv_kernel synth");
    let (dv_gpu, dk_gpu) = run_dkdv_kernel_on_gpu(
        &dkdv_ptx, cfg, &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &d_o,
        &fwd.row_max, &fwd.row_sum, &d, batch, heads, seq, hd,
    );

    // CPU reference.
    let (dv_ref, dk_ref) = cpu_naive_backward_dkdv(
        &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, batch, heads, seq, cfg,
    );

    let rel_tol = rel_tol_for_head_dim(hd as u32);

    // --- dV gate ---
    let dv_max_abs = dv_gpu.iter().zip(dv_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dv_max_ref = dv_ref.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let dv_max_gpu = dv_gpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let dv_rel     = if dv_max_ref > 0.0 { dv_max_abs / dv_max_ref } else { dv_max_abs };
    eprintln!(
        "[validate_dkdv FSource={:?}] hd={} bq={} seq={} \
         dV: max|gpu|={:.6e} max|ref|={:.6e} max_abs={:.6e} rel={:.4e} rel_tol={:.4e}",
        source, hd, cfg.block_q, seq, dv_max_gpu, dv_max_ref, dv_max_abs, dv_rel, rel_tol
    );
    assert!(
        dv_max_ref > 1e-3,
        "FSource={:?} hd={}: dV reference |dV|={:.3e} too small (raise DQ_GATE_DO_SCALE)",
        source, hd, dv_max_ref
    );
    assert!(
        dv_max_gpu >= 0.25 * dv_max_ref,
        "FSource={:?} hd={} seq={}: dV-kernel output near-zero \
         (max|dv_gpu|={:.3e} << max|dv_ref|={:.3e}) — ZERO-OUTPUT GUARD",
        source, hd, seq, dv_max_gpu, dv_max_ref
    );
    assert!(
        dv_rel <= rel_tol,
        "FSource={:?} hd={} seq={}: dV relative error {:.4e} > rel_tol {:.4e} \
         (max_abs={:.3e}, max|dv_ref|={:.3e})",
        source, hd, seq, dv_rel, rel_tol, dv_max_abs, dv_max_ref
    );

    // --- dK gate ---
    let dk_max_abs = dk_gpu.iter().zip(dk_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dk_max_ref = dk_ref.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let dk_max_gpu = dk_gpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let dk_rel     = if dk_max_ref > 0.0 { dk_max_abs / dk_max_ref } else { dk_max_abs };
    eprintln!(
        "[validate_dkdv FSource={:?}] hd={} bq={} seq={} \
         dK: max|gpu|={:.6e} max|ref|={:.6e} max_abs={:.6e} rel={:.4e} rel_tol={:.4e}",
        source, hd, cfg.block_q, seq, dk_max_gpu, dk_max_ref, dk_max_abs, dk_rel, rel_tol
    );
    // dK is intrinsically ~3 orders of magnitude smaller than dV: dK = dS @ Q with
    // dS = (1/sqrt(d)) * P * (dP - D), so the 1/sqrt(d) factor AND the (dP - D)
    // difference shrink dK relative to dV (= P @ dO, no such factors). The dV-calibrated
    // 1e-3 floor is therefore wrong for dK — at B1Forward hd=128 the *true* |dK| is
    // ~4.4e-4 (1/sqrt(128) compounding with B.1's small inputs), a correct value the
    // GPU matches to rel 2.9e-3. Use a dK-appropriate 1e-4 floor; the zero-output guard
    // below (0.25x ratio) is the real anti-zero check and still bites hard.
    assert!(
        dk_max_ref > 1e-4,
        "FSource={:?} hd={}: dK reference |dK|={:.3e} too small for a meaningful gate \
         (below the 1e-4 dK floor) — raise DQ_GATE_DO_SCALE",
        source, hd, dk_max_ref
    );
    assert!(
        dk_max_gpu >= 0.25 * dk_max_ref,
        "FSource={:?} hd={} seq={}: dK-kernel output near-zero \
         (max|dk_gpu|={:.3e} << max|dk_ref|={:.3e}) — ZERO-OUTPUT GUARD",
        source, hd, seq, dk_max_gpu, dk_max_ref
    );
    assert!(
        dk_rel <= rel_tol,
        "FSource={:?} hd={} seq={}: dK relative error {:.4e} > rel_tol {:.4e} \
         (max_abs={:.3e}, max|dk_ref|={:.3e})",
        source, hd, seq, dk_rel, rel_tol, dk_max_abs, dk_max_ref
    );
}

// ============================================================================
// GPU launchers
// ============================================================================

/// dK/dV-kernel GPU launcher.
///
/// Entry signature (13 params, in order):
///   q_saved_ptr, k_saved_ptr, v_saved_ptr, d_o_ptr, row_max_ptr, row_sum_ptr,
///   d_ptr, segment_ids_ptr, d_k_out_ptr, d_v_out_ptr, seq_len, heads, batch
///
/// Returns (dv_host, dk_host), both [batch, heads, seq, hd] f32.
#[allow(clippy::too_many_arguments)]
fn run_dkdv_kernel_on_gpu(
    ptx: &str,
    cfg: &FlashAttentionConfig,
    q: &[half::f16],
    k: &[half::f16],
    v: &[half::f16],
    d_o: &[half::f16],
    row_max: &[f32],
    row_sum: &[f32],
    d: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let in_4d_bytes = (batch * heads * seq * hd * 2) as i64; // f16
    let in_3d_bytes = (batch * heads * seq * 4) as i64;       // f32
    let out_bytes   = (batch * heads * seq * hd * 4) as i64;  // f32

    let q_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let k_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let v_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let do_dev   = nsl_test_cuda_alloc(in_4d_bytes);
    let rmax_dev = nsl_test_cuda_alloc(in_3d_bytes);
    let rsum_dev = nsl_test_cuda_alloc(in_3d_bytes);
    let d_dev    = nsl_test_cuda_alloc(in_3d_bytes);
    let dk_dev   = nsl_test_cuda_alloc(out_bytes);
    let dv_dev   = nsl_test_cuda_alloc(out_bytes);
    assert!(
        q_dev != 0 && k_dev != 0 && v_dev != 0 && do_dev != 0
            && rmax_dev != 0 && rsum_dev != 0 && d_dev != 0
            && dk_dev != 0 && dv_dev != 0,
        "dK/dV-kernel device alloc failed"
    );

    nsl_test_cuda_h2d(q_dev,    q.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(k_dev,    k.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(v_dev,    v.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(do_dev,   d_o.as_ptr()     as i64, in_4d_bytes);
    nsl_test_cuda_h2d(rmax_dev, row_max.as_ptr() as i64, in_3d_bytes);
    nsl_test_cuda_h2d(rsum_dev, row_sum.as_ptr() as i64, in_3d_bytes);
    nsl_test_cuda_h2d(d_dev,    d.as_ptr()       as i64, in_3d_bytes);

    let ptx_bytes   = ptx_to_cstr_bytes(ptx);
    let kernel_name = CString::new("tier_b2_dkdv_kernel").unwrap();

    let mut q_p    = q_dev    as u64;
    let mut k_p    = k_dev    as u64;
    let mut v_p    = v_dev    as u64;
    let mut do_p   = do_dev   as u64;
    let mut rmax_p = rmax_dev as u64;
    let mut rsum_p = rsum_dev as u64;
    let mut d_p    = d_dev    as u64;
    let mut seg_p  = 0u64;     // null — no segment masking
    let mut dk_p   = dk_dev   as u64;
    let mut dv_p   = dv_dev   as u64;
    let mut seq_u32   = seq   as u32;
    let mut heads_u32 = heads as u32;
    let mut batch_u32 = batch as u32;

    // 13 params: q, k, v, d_o, row_max, row_sum, d, seg, d_k_out, d_v_out, seq, heads, batch
    let args: [*mut c_void; 13] = [
        &mut q_p    as *mut _ as *mut c_void,
        &mut k_p    as *mut _ as *mut c_void,
        &mut v_p    as *mut _ as *mut c_void,
        &mut do_p   as *mut _ as *mut c_void,
        &mut rmax_p as *mut _ as *mut c_void,
        &mut rsum_p as *mut _ as *mut c_void,
        &mut d_p    as *mut _ as *mut c_void,
        &mut seg_p  as *mut _ as *mut c_void,
        &mut dk_p   as *mut _ as *mut c_void,
        &mut dv_p   as *mut _ as *mut c_void,
        &mut seq_u32   as *mut _ as *mut c_void,
        &mut heads_u32 as *mut _ as *mut c_void,
        &mut batch_u32 as *mut _ as *mut c_void,
    ];

    // Grid: (ceil(seq / effective_bq), heads, batch) — same as dq (bq=bkv here).
    let bq         = tier_b2_effective_bq(cfg) as i64;
    let grid_x     = ((seq as i64) + bq - 1) / bq;
    let smem_bytes = tier_b2_dkdv_total_smem_bytes(cfg) as i64;

    let rc = unsafe {
        nsl_kernel_launch(
            ptx_bytes.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            heads as i64,
            batch as i64,
            128, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            smem_bytes,
        )
    };

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx_bytes.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no JIT log>".into()
        };
        nsl_test_cuda_free(q_dev);
        nsl_test_cuda_free(k_dev);
        nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(do_dev);
        nsl_test_cuda_free(rmax_dev);
        nsl_test_cuda_free(rsum_dev);
        nsl_test_cuda_free(d_dev);
        nsl_test_cuda_free(dk_dev);
        nsl_test_cuda_free(dv_dev);
        panic!("dK/dV-kernel launch failed rc={rc}\nJIT log:\n{log}");
    }

    let mut dk_host = vec![0.0f32; batch * heads * seq * hd];
    let mut dv_host = vec![0.0f32; batch * heads * seq * hd];
    nsl_test_cuda_d2h(dv_host.as_mut_ptr() as i64, dv_dev, out_bytes);
    nsl_test_cuda_d2h(dk_host.as_mut_ptr() as i64, dk_dev, out_bytes);
    nsl_test_cuda_free(q_dev);
    nsl_test_cuda_free(k_dev);
    nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(do_dev);
    nsl_test_cuda_free(rmax_dev);
    nsl_test_cuda_free(rsum_dev);
    nsl_test_cuda_free(d_dev);
    nsl_test_cuda_free(dk_dev);
    nsl_test_cuda_free(dv_dev);
    (dv_host, dk_host)
}

/// D pre-pass GPU launcher (cloned verbatim from tier_b2_dq_kernel_cpu_reference).
fn run_d_prepass_on_gpu(
    ptx: &str,
    d_o: &[half::f16],
    o: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> Vec<f32> {
    assert_eq!(d_o.len(), batch * heads * seq * hd, "d_o size mismatch");
    assert_eq!(o.len(), batch * heads * seq * hd, "o size mismatch");

    let in_bytes  = (batch * heads * seq * hd * 2) as i64;
    let out_bytes = (batch * heads * seq * 4) as i64;

    let d_o_dev   = nsl_test_cuda_alloc(in_bytes);
    let o_dev     = nsl_test_cuda_alloc(in_bytes);
    let d_out_dev = nsl_test_cuda_alloc(out_bytes);
    assert!(
        d_o_dev != 0 && o_dev != 0 && d_out_dev != 0,
        "D pre-pass device alloc failed"
    );

    nsl_test_cuda_h2d(d_o_dev, d_o.as_ptr() as i64, in_bytes);
    nsl_test_cuda_h2d(o_dev,   o.as_ptr()   as i64, in_bytes);

    let ptx_bytes   = ptx_to_cstr_bytes(ptx);
    let kernel_name = CString::new("tier_b2_d_prepass").unwrap();

    let mut d_o_ptr_u64  = d_o_dev   as u64;
    let mut o_ptr_u64    = o_dev     as u64;
    let mut d_out_ptr_u64 = d_out_dev as u64;
    let mut seq_len_u32  = seq   as u32;
    let mut heads_u32    = heads as u32;
    let args: [*mut c_void; 5] = [
        &mut d_o_ptr_u64   as *mut _ as *mut c_void,
        &mut o_ptr_u64     as *mut _ as *mut c_void,
        &mut d_out_ptr_u64 as *mut _ as *mut c_void,
        &mut seq_len_u32   as *mut _ as *mut c_void,
        &mut heads_u32     as *mut _ as *mut c_void,
    ];

    let grid_x = ((seq as i64) + 31) / 32;
    let rc = unsafe {
        nsl_kernel_launch(
            ptx_bytes.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            heads as i64,
            batch as i64,
            32, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0, // no dynamic SMEM
        )
    };

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx_bytes.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no JIT log>".into()
        };
        nsl_test_cuda_free(d_o_dev);
        nsl_test_cuda_free(o_dev);
        nsl_test_cuda_free(d_out_dev);
        panic!("D pre-pass launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    let mut d_host = vec![0.0f32; batch * heads * seq];
    nsl_test_cuda_d2h(d_host.as_mut_ptr() as i64, d_out_dev, out_bytes);
    nsl_test_cuda_free(d_o_dev);
    nsl_test_cuda_free(o_dev);
    nsl_test_cuda_free(d_out_dev);
    d_host
}
