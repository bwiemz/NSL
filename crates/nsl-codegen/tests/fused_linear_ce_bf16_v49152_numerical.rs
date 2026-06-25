//! Sprint v5 — bf16 forward+backward GPU numerical validation at the
//! production NSL vocabulary V=49152.
//!
//! Mirror of `fused_linear_ce_fp16_v49152_numerical.rs` for bf16. v4-1
//! bf16 was numerically validated only at V=4096 / V=8192 / V=8193
//! (regression guard); v4 review flagged the lack of a production-scale
//! (V=49152) bf16 numerical pin as the matching blocker for lifting the
//! wengert-level bf16 refusal.
//!
//! ## Path exercised
//! V=49152 routes through `synthesize_large_vocab_forward_ptx`
//! (`emit_large_partials_kernel_bf16` + `emit_large_finalize_kernel_bf16`),
//! launched via `nsl_fused_linear_ce_forward_large` with `dtype_tag = 2`
//! (Bf16). Backward kernel is shared with the v1 path —
//! `emit_bwd_kernel_bf16` — scanning num_tiles=384 tiles per row, launched
//! via `nsl_fused_linear_ce_backward`.
//!
//! ## bf16 numerical pressure
//!
//! Bf16 has the **same exponent range as f32** (8 bits) but only **7
//! explicit + 1 implicit mantissa bits** — vs fp16's 10+1. Per-logit
//! relative noise is therefore ~2^-7 ≈ 8e-3 (vs fp16's 2^-10 ≈ 1e-3),
//! roughly 8× looser than fp16 on individual logits. However:
//!   - The same 1/sqrt(N) central-limit cancellation applies to the
//!     mean_loss aggregate over ~5 × 10^6 logit-rounding events, bringing
//!     the per-aggregate noise down to ~8e-3 / sqrt(5e6) ≈ 4e-6 — still
//!     well under any reasonable tolerance gate.
//!   - The bf16 emitter's tail-zero sentinel uses 0xFF80 (bf16 -INF) per
//!     the v4-1 Finding-3 fix; this test exercises the small-vocab and
//!     large-vocab paths only when V % vocab_tile == 0 (V=49152 / 128 =
//!     384 tiles, exact), so the sentinel path is NOT exercised here.
//!     The V=8193 regression guard in `fused_linear_ce_bf16_numerical.rs`
//!     pins that case separately.
//!
//! ### Empirical measurement (Sprint v5-2)
//!
//! With `x ∈ uniform[-0.5, 0.5]`, `W ∈ normal(0, 0.01)`, `bias = 0`, the
//! seeded NSL-production-shape fixture (B=2, S=64, V=49152, H=128) on an
//! RTX 5080 (sm_120) measured:
//!
//! ```text
//! bf16 V=49152 forward: gpu_loss=10.798112 cpu_loss=10.798112
//!   mean_loss rel_err = 7.37e-9
//!   lse max_abs       = 4.37e-6
//! bf16 V=49152 backward:
//!   dx max_abs        = 4.21e-9
//!   dW max_abs        = 3.96e-9
//!   dbias max_abs     = 7.93e-9
//! ```
//!
//! Bf16 aggregate noise is **comparable** to fp16 here — not 8× worse —
//! because the bf16 emitter casts UP to f32 during dot-product
//! accumulation (`cvt.f32.bf16` on every load) and only HBM storage is
//! bf16. The 8× per-element mantissa penalty therefore applies only to
//! the input rounding, not to the in-register accumulator chain.
//!
//! Tolerances pinned at ~100× the empirical baseline:
//!   - bf16 fwd mean_loss rel_err <= 1e-5  (~1400× headroom)
//!   - bf16 fwd lse max_abs       <= 1e-4  (~23×)
//!   - bf16 bwd dx max_abs        <= 1e-5  (~2400×)
//!   - bf16 bwd dW max_abs        <= 1e-5  (~2500×)
//!   - bf16 bwd dbias max_abs     <= 1e-5  (~1300×)
//!
//! These match the fp16 V=49152 gates exactly — a divergence between
//! fp16 and bf16 at production scale would itself be a smoke signal
//! and warrant investigation.

#![cfg(feature = "cuda")]

mod common;

use common::fused_lce_cpu_f64::{
    cpu_lce_backward_f64, cpu_lce_forward_f64, IGNORE_INDEX as CPU_IGNORE_INDEX,
};
use half::bf16;
use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig, MAX_VOCAB_HARD_CEILING,
    synthesize_fused_linear_ce_backward_ptx, synthesize_fused_linear_ce_ptx,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d,
    nsl_fused_linear_ce_backward, nsl_fused_linear_ce_forward_large,
};

const IGNORE_INDEX: i64 = -100;
const DTYPE_TAG_BF16: i64 = 2;

const _IGNORE_INDEX_MATCH: () = assert!(
    (IGNORE_INDEX as i32) == CPU_IGNORE_INDEX,
    "IGNORE_INDEX must match common::fused_lce_cpu_f64::IGNORE_INDEX"
);

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {rc}");
        false
    } else {
        true
    }
}

fn fill_seeded_uniform(dst: &mut [f32], seed: u64, lo: f32, hi: f32) {
    let mut s = seed;
    let span = hi - lo;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        *x = lo + ((u as f32) / (u32::MAX as f32)) * span;
    }
}

fn fill_seeded_normal(dst: &mut [f32], seed: u64, mean: f32, std: f32) {
    let mut s = seed;
    let mut idx = 0;
    while idx < dst.len() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (((s >> 33) as u32) as f32 / u32::MAX as f32).max(1e-9);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((s >> 33) as u32) as f32 / u32::MAX as f32;
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        dst[idx] = mean + std * z0;
        idx += 1;
        if idx < dst.len() {
            dst[idx] = mean + std * z1;
            idx += 1;
        }
    }
}

fn f32_slice_to_bf16_bits(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = bf16::from_f32(*s).to_bits();
    }
}

fn round_through_bf16(src: &[f32]) -> Vec<f32> {
    src.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

fn f32_to_f64(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&x| x as f64).collect()
}

fn i64_to_i32_targets(src: &[i64]) -> Vec<i32> {
    src.iter()
        .map(|&t| if t == IGNORE_INDEX { CPU_IGNORE_INDEX } else { t as i32 })
        .collect()
}

#[test]
#[ignore]
fn bf16_forward_backward_at_v49152_production_scale() {
    if !cuda_available() {
        return;
    }

    const B: usize = 2;
    const S: usize = 64;
    const V: usize = 49152;
    const H: usize = 128;
    const VOCAB_TILE: u32 = 128;

    let cfg = FusedLinearCEConfig {
        vocab_size: V as u32,
        hidden_size: H as u32,
        seq_len: S as u32,
        batch_size: B as u32,
        vocab_tile: VOCAB_TILE,
        gpu_sm: 80,
        dtype: Dtype::Bf16,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("V=49152 bf16 config must validate");
    assert!(
        cfg.is_large_vocab(),
        "V=49152 must route through the large-vocab bf16 forward path"
    );

    let rows = B * S;

    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        let h = (i.wrapping_mul(2654435761)) & 0xFFFF_FFFF;
        *t = if h % 5 == 0 {
            IGNORE_INDEX
        } else {
            ((i.wrapping_mul(37).wrapping_add(13)) % V) as i64
        };
    }
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as u32;
    eprintln!("V=49152 bf16: rows={rows} num_valid={num_valid}");

    let mut x_f32 = vec![0f32; rows * H];
    let mut w_f32 = vec![0f32; V * H];
    let bias_f32 = vec![0f32; V];
    fill_seeded_uniform(&mut x_f32, 42, -0.5, 0.5);
    fill_seeded_normal(&mut w_f32, 137, 0.0, 0.01);

    let x_ref = round_through_bf16(&x_f32);
    let w_ref = round_through_bf16(&w_f32);
    let bias_ref = round_through_bf16(&bias_f32);

    let mut x_b16 = vec![0u16; rows * H];
    let mut w_b16 = vec![0u16; V * H];
    let mut bias_b16 = vec![0u16; V];
    f32_slice_to_bf16_bits(&x_f32, &mut x_b16);
    f32_slice_to_bf16_bits(&w_f32, &mut w_b16);
    f32_slice_to_bf16_bits(&bias_f32, &mut bias_b16);

    let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    fwd_ptx.push(0u8);
    let mut bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    bwd_ptx.push(0u8);
    let kname_a = format!("{}\0", cfg.large_partials_kernel_name());
    let kname_b = format!("{}\0", cfg.large_finalize_kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    let x_bytes = (rows * H * 2) as i64;
    let w_bytes = (V * H * 2) as i64;
    let bias_bytes = (V * 2) as i64;
    let tgt_bytes = (rows * 8) as i64;
    let loss_bytes = (rows * 4) as i64;
    let lse_bytes = (rows * 4) as i64;
    let dx_bytes = (rows * H * 4) as i64;
    let dw_bytes = (V * H * 4) as i64;
    let dbias_bytes = (V * 4) as i64;
    let partials_bytes = cfg.large_partials_bytes() as i64;

    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let w_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let bias_dev = unsafe { nsl_test_cuda_alloc(bias_bytes) };
    let tgt_dev = unsafe { nsl_test_cuda_alloc(tgt_bytes) };
    let loss_dev = unsafe { nsl_test_cuda_alloc(loss_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let dx_dev = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    let dw_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dbias_dev = unsafe { nsl_test_cuda_alloc(dbias_bytes) };
    let partials_dev = unsafe { nsl_test_cuda_alloc(partials_bytes) };
    assert!(
        x_dev != 0 && w_dev != 0 && bias_dev != 0 && tgt_dev != 0
            && loss_dev != 0 && lse_dev != 0 && dx_dev != 0
            && dw_dev != 0 && dbias_dev != 0 && partials_dev != 0,
        "device alloc failed"
    );

    unsafe {
        nsl_test_cuda_h2d(x_dev, x_b16.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev, w_b16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_b16.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev, targets.as_ptr() as i64, tgt_bytes);
    }

    let smem = cfg.shared_mem_bytes();
    let num_tiles = cfg.num_vocab_tiles();
    eprintln!("bf16 V=49152: SMEM={smem} B/CTA, num_tiles={num_tiles}");

    eprintln!("computing CPU f64 reference...");
    let t0 = std::time::Instant::now();
    let x_ref_f64 = f32_to_f64(&x_ref);
    let w_ref_f64 = f32_to_f64(&w_ref);
    let bias_ref_f64 = f32_to_f64(&bias_ref);
    let targets_i32 = i64_to_i32_targets(&targets);
    let cpu_fwd = cpu_lce_forward_f64(
        &x_ref_f64, &w_ref_f64, &bias_ref_f64, &targets_i32, rows, V, H,
    );
    eprintln!(
        "CPU forward took {:.1}s; mean_loss = {:.6}",
        t0.elapsed().as_secs_f64(),
        cpu_fwd.mean_loss
    );

    let rc = nsl_fused_linear_ce_forward_large(
        fwd_ptx.as_ptr() as i64,
        kname_a.as_ptr() as i64,
        kname_b.as_ptr() as i64,
        x_dev, w_dev, bias_dev, tgt_dev, partials_dev,
        loss_dev, lse_dev,
        B as i64, S as i64, V as i64, H as i64,
        num_tiles as i64,
        smem as i64,
        DTYPE_TAG_BF16,
    );
    assert_eq!(rc, 0, "nsl_fused_linear_ce_forward_large (bf16) failed rc={rc}");

    let mut loss_gpu = vec![0f32; rows];
    let mut lse_gpu = vec![0f32; rows];
    unsafe {
        nsl_test_cuda_d2h(loss_gpu.as_mut_ptr() as i64, loss_dev, loss_bytes);
        nsl_test_cuda_d2h(lse_gpu.as_mut_ptr() as i64, lse_dev, lse_bytes);
    }

    let mut gpu_loss_sum = 0f64;
    let mut gpu_nv = 0usize;
    let mut max_lse_abs = 0f64;
    for (i, (&lv, &lsv)) in loss_gpu.iter().zip(lse_gpu.iter()).enumerate() {
        if targets[i] == IGNORE_INDEX {
            assert_eq!(lv, 0.0, "skip-identity: loss row={i} = {lv}");
            assert_eq!(lsv, 0.0, "skip-identity: lse row={i} = {lsv}");
            continue;
        }
        gpu_loss_sum += lv as f64;
        gpu_nv += 1;
        let d = (lsv as f64 - cpu_fwd.lse_per_row[i]).abs();
        if d > max_lse_abs { max_lse_abs = d; }
    }
    let gpu_mean_loss = gpu_loss_sum / gpu_nv.max(1) as f64;
    let mean_loss_rel = (gpu_mean_loss - cpu_fwd.mean_loss).abs()
        / cpu_fwd.mean_loss.abs().max(1.0);
    eprintln!(
        "bf16 V=49152 forward: gpu_loss={:.6} cpu_loss={:.6} rel_err={:.3e} \
         lse_max_abs={:.3e}",
        gpu_mean_loss, cpu_fwd.mean_loss, mean_loss_rel, max_lse_abs
    );

    // Tolerances match the fp16 V=49152 gates — the cvt.f32.bf16 cast
    // on every dot-product load means the in-register accumulator chain
    // sees the same f32 precision as the fp16 path. A divergence between
    // bf16 and fp16 at production scale would itself be a smoke signal.
    assert!(
        mean_loss_rel < 1e-5,
        "bf16 V=49152 fwd mean_loss rel_err {:.3e} exceeds 1e-5 — \
         empirical baseline 7.37e-9; a real numerical regression (broken \
         bf16 cvt, missing -INF sentinel at 0xFF80, wrong rescale) is the \
         likely cause",
        mean_loss_rel
    );
    assert!(
        max_lse_abs < 1e-4,
        "bf16 V=49152 fwd lse max_abs {:.3e} exceeds 1e-4 — \
         empirical baseline 4.37e-6; a regression in the per-tile \
         rescale chain (Kernel A bf16) is the likely cause",
        max_lse_abs
    );

    // ─── Backward ───────────────────────────────────────────────────────────
    {
        let zeros_dx = vec![0f32; rows * H];
        let zeros_dw = vec![0f32; V * H];
        let zeros_db = vec![0f32; V];
        unsafe {
            nsl_test_cuda_h2d(dx_dev, zeros_dx.as_ptr() as i64, dx_bytes);
            nsl_test_cuda_h2d(dw_dev, zeros_dw.as_ptr() as i64, dw_bytes);
            nsl_test_cuda_h2d(dbias_dev, zeros_db.as_ptr() as i64, dbias_bytes);
        }
    }

    let go_bits = 1.0f32.to_bits() as i64;
    let bwd_rc = unsafe {
        nsl_fused_linear_ce_backward(
            bwd_ptx.as_ptr() as i64,
            bwd_name.as_ptr() as i64,
            go_bits,
            x_dev, w_dev, bias_dev, tgt_dev, lse_dev,
            dx_dev, dw_dev, dbias_dev,
            B as i64, S as i64, V as i64, H as i64,
            num_valid as i64,
            smem as i64,
            DTYPE_TAG_BF16,
        )
    };
    assert_eq!(bwd_rc, 0, "nsl_fused_linear_ce_backward (bf16, V=49152) failed rc={bwd_rc}");

    let mut dx_gpu = vec![0f32; rows * H];
    let mut dw_gpu = vec![0f32; V * H];
    let mut dbias_gpu = vec![0f32; V];
    unsafe {
        nsl_test_cuda_d2h(dx_gpu.as_mut_ptr() as i64, dx_dev, dx_bytes);
        nsl_test_cuda_d2h(dw_gpu.as_mut_ptr() as i64, dw_dev, dw_bytes);
        nsl_test_cuda_d2h(dbias_gpu.as_mut_ptr() as i64, dbias_dev, dbias_bytes);
    }

    eprintln!("computing CPU f64 backward reference...");
    let t1 = std::time::Instant::now();
    let cpu_bwd = cpu_lce_backward_f64(
        &x_ref_f64, &w_ref_f64, &bias_ref_f64, &targets_i32,
        &cpu_fwd.lse_per_row, 1.0, rows, V, H,
    );
    eprintln!("CPU backward took {:.1}s", t1.elapsed().as_secs_f64());

    let mut dx_max_abs = 0f64;
    for row in 0..rows {
        if targets[row] == IGNORE_INDEX {
            for hi in 0..H {
                let val = dx_gpu[row * H + hi];
                assert_eq!(val, 0.0, "bf16 skip-identity dx[{row},{hi}] = {val}");
            }
        } else {
            for hi in 0..H {
                let d = (dx_gpu[row * H + hi] as f64 - cpu_bwd.dx[row * H + hi]).abs();
                if d > dx_max_abs { dx_max_abs = d; }
            }
        }
    }
    let mut dw_max_abs = 0f64;
    for i in 0..V * H {
        let d = (dw_gpu[i] as f64 - cpu_bwd.dw[i]).abs();
        if d > dw_max_abs { dw_max_abs = d; }
    }
    let mut db_max_abs = 0f64;
    for vi in 0..V {
        let d = (dbias_gpu[vi] as f64 - cpu_bwd.dbias[vi]).abs();
        if d > db_max_abs { db_max_abs = d; }
    }
    eprintln!(
        "bf16 V=49152 backward: dx_max_abs={:.3e} dw_max_abs={:.3e} dbias_max_abs={:.3e}",
        dx_max_abs, dw_max_abs, db_max_abs
    );

    assert!(
        dx_max_abs < 1e-5,
        "bf16 V=49152 bwd dx max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 4.21e-9; check the backward bf16 softmax \
         recompute or dx scatter via W^T",
        dx_max_abs
    );
    assert!(
        dw_max_abs < 1e-5,
        "bf16 V=49152 bwd dW max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 3.96e-9; check the dlogits × x outer-product \
         scatter via red.global.add.f32",
        dw_max_abs
    );
    assert!(
        db_max_abs < 1e-5,
        "bf16 V=49152 bwd dbias max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 7.93e-9; check the dlogits row-sum scatter",
        db_max_abs
    );

    unsafe {
        nsl_test_cuda_free(x_dev);
        nsl_test_cuda_free(w_dev);
        nsl_test_cuda_free(bias_dev);
        nsl_test_cuda_free(tgt_dev);
        nsl_test_cuda_free(loss_dev);
        nsl_test_cuda_free(lse_dev);
        nsl_test_cuda_free(dx_dev);
        nsl_test_cuda_free(dw_dev);
        nsl_test_cuda_free(dbias_dev);
        nsl_test_cuda_free(partials_dev);
    }

    eprintln!("bf16 V=49152 production-scale numerical pin: PASS");
}
