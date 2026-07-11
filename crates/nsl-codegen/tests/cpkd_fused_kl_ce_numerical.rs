//! CPKD fused KL-CE — GPU numerical validation against the f64 reference.
//!
//! Drives the SAME FFIs the compiled distill step dispatches
//! (`nsl_fused_kl_ce_forward` / `nsl_fused_kl_ce_backward`) with
//! deterministic seeded buffers and compares per-row losses, saved LSEs,
//! and all three student gradients against
//! `cpkd_fused_loss::reference_{forward,backward}_f64` (which is itself
//! pinned by finite differences in the module's unit tests).
//!
//! ## Fixture
//! rows = 2*16 = 32, V=512, HS=64, HT=96 (asymmetric on purpose),
//! vocab_tile=128, alpha=0.4, T=2.5, ignore_index -100 at every 8th row.
//!
//! ## Tolerances
//! * forward per-row loss + LSEs: |gpu - ref| / max(|ref|, 1.0) < 1e-3
//! * backward dx_s/dW_s/dbias_s: max abs err < 5e-3
//! * skip identity: loss/lse rows and dx_s rows exactly 0 for -100 rows
//!
//! ## Running
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test cpkd_fused_kl_ce_numerical \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::cpkd_fused_loss::{
    FusedKlCeConfig, reference_backward_f64, reference_forward_f64,
    synthesize_fused_kl_ce_backward_ptx, synthesize_fused_kl_ce_ptx,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_fused_kl_ce_backward, nsl_fused_kl_ce_forward, nsl_test_cuda_alloc,
    nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

const B: usize = 2;
const S: usize = 16;
const ROWS: usize = B * S;
const V: usize = 512;
const HS: usize = 64;
const HT: usize = 96;
const VOCAB_TILE: u32 = 128;
const IGNORE_INDEX: i64 = -100;
const ALPHA: f64 = 0.4;
const TEMP: f64 = 2.5;

fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) * 0.6 - 0.3;
    }
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {rc}");
        false
    } else {
        true
    }
}

fn f32_bits(v: f64) -> i64 {
    (v as f32).to_bits() as i64
}

#[allow(clippy::too_many_lines)]
#[test]
#[ignore]
fn fused_kl_ce_gpu_forward_and_backward_match_reference() {
    if !cuda_available() {
        return;
    }

    let cfg = FusedKlCeConfig {
        vocab_size: V as u32,
        student_hidden: HS as u32,
        teacher_hidden: HT as u32,
        batch_size: B as u32,
        seq_len: S as u32,
        vocab_tile: VOCAB_TILE,
        gpu_sm: 80,
        ignore_index: IGNORE_INDEX,
    };
    cfg.validate().expect("config must be valid");

    // Targets: -100 at every 8th row.
    let mut targets = vec![0i64; ROWS];
    for (i, t) in targets.iter_mut().enumerate() {
        *t = if i % 8 == 0 {
            IGNORE_INDEX
        } else {
            ((i * 37 + 13) % V) as i64
        };
    }
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as i64;

    // Host buffers.
    let mut xs = vec![0f32; ROWS * HS];
    let mut ws = vec![0f32; V * HS];
    let mut bs = vec![0f32; V];
    let mut xt = vec![0f32; ROWS * HT];
    let mut wt = vec![0f32; V * HT];
    let mut bt = vec![0f32; V];
    fill_seeded(&mut xs, 42);
    fill_seeded(&mut ws, 137);
    fill_seeded(&mut bs, 7);
    fill_seeded(&mut xt, 1042);
    fill_seeded(&mut wt, 1137);
    fill_seeded(&mut bt, 1007);

    // f64 reference.
    let to64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let (xs64, ws64, bs64) = (to64(&xs), to64(&ws), to64(&bs));
    let (xt64, wt64, bt64) = (to64(&xt), to64(&wt), to64(&bt));
    let (ref_loss, ref_lse_s1, ref_lse_st, ref_lse_tt, ref_nv) = reference_forward_f64(
        &xs64, &ws64, &bs64, &xt64, &wt64, &bt64, &targets, ROWS, V, HS, HT, ALPHA, TEMP,
        IGNORE_INDEX,
    );
    assert_eq!(ref_nv as i64, num_valid);
    let (ref_dxs, ref_dws, ref_dbs) = reference_backward_f64(
        &xs64, &ws64, &bs64, &xt64, &wt64, &bt64, &targets, ROWS, V, HS, HT, ALPHA, TEMP,
        IGNORE_INDEX, 1.0,
    );

    // PTX.
    let fwd_ptx = synthesize_fused_kl_ce_ptx(&cfg);
    let bwd_ptx = synthesize_fused_kl_ce_backward_ptx(&cfg);
    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Device buffers.
    let alloc = |bytes: usize| nsl_test_cuda_alloc(bytes as i64);
    let xs_dev = alloc(ROWS * HS * 4);
    let ws_dev = alloc(V * HS * 4);
    let bs_dev = alloc(V * 4);
    let xt_dev = alloc(ROWS * HT * 4);
    let wt_dev = alloc(V * HT * 4);
    let bt_dev = alloc(V * 4);
    let tgt_dev = alloc(ROWS * 8);
    let loss_dev = alloc(ROWS * 4);
    let lse_s1_dev = alloc(ROWS * 4);
    let lse_st_dev = alloc(ROWS * 4);
    let lse_tt_dev = alloc(ROWS * 4);
    let dxs_dev = alloc(ROWS * HS * 4);
    let dws_dev = alloc(V * HS * 4);
    let dbs_dev = alloc(V * 4);
    for d in [
        xs_dev, ws_dev, bs_dev, xt_dev, wt_dev, bt_dev, tgt_dev, loss_dev, lse_s1_dev,
        lse_st_dev, lse_tt_dev, dxs_dev, dws_dev, dbs_dev,
    ] {
        assert_ne!(d, 0, "device alloc failed");
    }

    let h2d = |dev: i64, host: *const u8, bytes: usize| {
        nsl_test_cuda_h2d(dev, host as i64, bytes as i64);
    };
    h2d(xs_dev, xs.as_ptr() as *const u8, ROWS * HS * 4);
    h2d(ws_dev, ws.as_ptr() as *const u8, V * HS * 4);
    h2d(bs_dev, bs.as_ptr() as *const u8, V * 4);
    h2d(xt_dev, xt.as_ptr() as *const u8, ROWS * HT * 4);
    h2d(wt_dev, wt.as_ptr() as *const u8, V * HT * 4);
    h2d(bt_dev, bt.as_ptr() as *const u8, V * 4);
    h2d(tgt_dev, targets.as_ptr() as *const u8, ROWS * 8);
    // Gradient outputs must be zero-filled (red.global.add accumulation).
    let zeros_dxs = vec![0f32; ROWS * HS];
    let zeros_dws = vec![0f32; V * HS];
    let zeros_dbs = vec![0f32; V];
    h2d(dxs_dev, zeros_dxs.as_ptr() as *const u8, ROWS * HS * 4);
    h2d(dws_dev, zeros_dws.as_ptr() as *const u8, V * HS * 4);
    h2d(dbs_dev, zeros_dbs.as_ptr() as *const u8, V * 4);

    // ── Forward ──────────────────────────────────────────────────────────
    let rc = nsl_fused_kl_ce_forward(
        fwd_ptx.as_ptr() as i64,
        fwd_name.as_ptr() as i64,
        xs_dev,
        ws_dev,
        bs_dev,
        xt_dev,
        wt_dev,
        bt_dev,
        tgt_dev,
        loss_dev,
        lse_s1_dev,
        lse_st_dev,
        lse_tt_dev,
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(ALPHA),
        f32_bits(TEMP),
        cfg.shared_mem_bytes() as i64,
    );
    assert_eq!(rc, 0, "forward launch failed rc={rc}");

    let d2h = |dev: i64, host: *mut u8, bytes: usize| {
        nsl_test_cuda_d2h(host as i64, dev, bytes as i64);
    };
    let mut gpu_loss = vec![0f32; ROWS];
    let mut gpu_lse_s1 = vec![0f32; ROWS];
    let mut gpu_lse_st = vec![0f32; ROWS];
    let mut gpu_lse_tt = vec![0f32; ROWS];
    d2h(loss_dev, gpu_loss.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_s1_dev, gpu_lse_s1.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_st_dev, gpu_lse_st.as_mut_ptr() as *mut u8, ROWS * 4);
    d2h(lse_tt_dev, gpu_lse_tt.as_mut_ptr() as *mut u8, ROWS * 4);

    let rel = |gpu: f32, r: f64| ((gpu as f64) - r).abs() / r.abs().max(1.0);
    for row in 0..ROWS {
        if targets[row] == IGNORE_INDEX {
            assert_eq!(gpu_loss[row], 0.0, "row {row}: skip row loss must be 0");
            assert_eq!(gpu_lse_s1[row], 0.0, "row {row}: skip row lse_s1 must be 0");
            continue;
        }
        assert!(
            rel(gpu_loss[row], ref_loss[row]) < 1e-3,
            "row {row}: loss gpu={} ref={}",
            gpu_loss[row],
            ref_loss[row]
        );
        assert!(
            rel(gpu_lse_s1[row], ref_lse_s1[row]) < 1e-3,
            "row {row}: lse_s1 gpu={} ref={}",
            gpu_lse_s1[row],
            ref_lse_s1[row]
        );
        assert!(
            rel(gpu_lse_st[row], ref_lse_st[row]) < 1e-3,
            "row {row}: lse_sT gpu={} ref={}",
            gpu_lse_st[row],
            ref_lse_st[row]
        );
        assert!(
            rel(gpu_lse_tt[row], ref_lse_tt[row]) < 1e-3,
            "row {row}: lse_tT gpu={} ref={}",
            gpu_lse_tt[row],
            ref_lse_tt[row]
        );
    }
    eprintln!("forward: {ROWS} rows match reference (rel < 1e-3)");

    // ── Backward ─────────────────────────────────────────────────────────
    let rc = nsl_fused_kl_ce_backward(
        bwd_ptx.as_ptr() as i64,
        bwd_name.as_ptr() as i64,
        f32_bits(1.0),
        xs_dev,
        ws_dev,
        bs_dev,
        xt_dev,
        wt_dev,
        bt_dev,
        tgt_dev,
        lse_s1_dev,
        lse_st_dev,
        lse_tt_dev,
        dxs_dev,
        dws_dev,
        dbs_dev,
        ROWS as i64,
        V as i64,
        HS as i64,
        HT as i64,
        f32_bits(ALPHA),
        f32_bits(TEMP),
        num_valid,
    );
    assert_eq!(rc, 0, "backward launch failed rc={rc}");

    let mut gpu_dxs = vec![0f32; ROWS * HS];
    let mut gpu_dws = vec![0f32; V * HS];
    let mut gpu_dbs = vec![0f32; V];
    d2h(dxs_dev, gpu_dxs.as_mut_ptr() as *mut u8, ROWS * HS * 4);
    d2h(dws_dev, gpu_dws.as_mut_ptr() as *mut u8, V * HS * 4);
    d2h(dbs_dev, gpu_dbs.as_mut_ptr() as *mut u8, V * 4);

    let max_err = |gpu: &[f32], r: &[f64]| {
        gpu.iter()
            .zip(r.iter())
            .map(|(&g, &rr)| ((g as f64) - rr).abs())
            .fold(0f64, f64::max)
    };
    let dxs_err = max_err(&gpu_dxs, &ref_dxs);
    let dws_err = max_err(&gpu_dws, &ref_dws);
    let dbs_err = max_err(&gpu_dbs, &ref_dbs);
    eprintln!("backward max abs err: dx_s={dxs_err:.2e} dW_s={dws_err:.2e} dbias_s={dbs_err:.2e}");
    assert!(dxs_err < 5e-3, "dx_s max err {dxs_err}");
    assert!(dws_err < 5e-3, "dW_s max err {dws_err}");
    assert!(dbs_err < 5e-3, "dbias_s max err {dbs_err}");

    // Skip identity: -100 rows have exactly-zero dx_s.
    for row in 0..ROWS {
        if targets[row] == IGNORE_INDEX {
            for h in 0..HS {
                assert_eq!(
                    gpu_dxs[row * HS + h],
                    0.0,
                    "row {row} h {h}: skip row dx_s must be exactly 0"
                );
            }
        }
    }

    for d in [
        xs_dev, ws_dev, bs_dev, xt_dev, wt_dev, bt_dev, tgt_dev, loss_dev, lse_s1_dev,
        lse_st_dev, lse_tt_dev, dxs_dev, dws_dev, dbs_dev,
    ] {
        nsl_test_cuda_free(d);
    }
}
