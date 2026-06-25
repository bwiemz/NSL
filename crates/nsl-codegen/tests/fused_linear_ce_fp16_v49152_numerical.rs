//! Sprint v5 — fp16 forward+backward GPU numerical validation at the
//! production NSL vocabulary V=49152.
//!
//! ## Why
//! `fused_linear_ce_fp16_numerical.rs` pins fp16 at V=4096 and V=8192 only
//! — both small-vocab routings. v4 review flagged the lack of a
//! production-scale fp16 numerical gate as the blocker for lifting the
//! wengert-level fp16 refusal. This sprint closes that gap.
//!
//! ## Path exercised
//! V=49152 routes through `synthesize_large_vocab_forward_ptx` (Kernel A
//! per-tile partials, Kernel B per-row finalize), launched via
//! `nsl_fused_linear_ce_forward_large` with `dtype_tag = 1` (F16).
//! The backward kernel is shared between v1 and large-vocab paths — it
//! scans `num_tiles = 384` tiles per row — and is dispatched via the
//! standard `nsl_fused_linear_ce_backward` FFI.
//!
//! ## Tolerances
//!
//! Numerical pressure compared to the V=4096 fp16 baseline:
//!   - `num_tiles` at V=49152, vocab_tile=128 is 384 vs 4 at V=4096,
//!     vocab_tile=1024 — a 96× increase in the online-LSE rescale chain.
//!   - Each rescale is a single-ULP `mad.rn.f32` plus `ex2.approx.f32` /
//!     `lg2.approx.f32` (each ~3 ULP) — error accumulates roughly
//!     additively in the worst case, ~96 × 5 ULP ≈ 6e-5 relative on LSE.
//!   - The fp16-input dot product retains 10-bit mantissa precision, so
//!     dot(x,W) loses ~1 part in 1024 ≈ 1e-3 RELATIVE to the f64 reference
//!     PER LOGIT. The max-shifted softmax sum amplifies whichever logit is
//!     within ~10 of the max (entropy ~= ln(V) = 10.8 nats at uniform);
//!     for our seeded inputs entropy stays near uniform, so ~V/e effective
//!     "high-mass" logits each contribute ~1 ULP-fp16 noise to lse.
//!
//! ### Empirical measurement (Sprint v5-2)
//!
//! With `x ∈ uniform[-0.5, 0.5]`, `W ∈ normal(0, 0.01)`, `bias = 0`, the
//! seeded NSL-production-shape fixture (B=2, S=64, V=49152, H=128) on
//! an RTX 5080 (sm_120) measured:
//!
//! ```text
//! fp16 V=49152 forward: gpu_loss=10.798127 cpu_loss=10.798127
//!   mean_loss rel_err = 2.47e-8
//!   lse max_abs       = 1.38e-6
//! fp16 V=49152 backward:
//!   dx max_abs        = 5.87e-9
//!   dW max_abs        = 3.78e-9
//!   dbias max_abs     = 5.39e-9
//! ```
//!
//! The mean-loss reduction averages fp16 quantization noise over
//! `num_valid × V ≈ 5 × 10^6` independent logit-rounding events; the
//! `1/sqrt(N)` central-limit cancellation brings the per-logit ~1e-3
//! relative fp16 noise down to ~1e-7 on the aggregate, matching the
//! observed `rel_err ≈ 2.5e-8`. The backward gradients are likewise
//! aggregates over many independent rounding events.
//!
//! Tolerances are pinned at ~100× the empirical measurement so a real
//! kernel-level regression (broken cast, wrong rescale, missing sentinel)
//! trips immediately but cross-machine RNG/order jitter does not flag.
//!   - fp16 fwd mean_loss rel_err <= 1e-5  (~400× margin over empirical)
//!   - fp16 fwd lse max_abs       <= 1e-4  (~70× margin)
//!   - fp16 bwd dx max_abs        <= 1e-5  (~1700× margin)
//!   - fp16 bwd dW max_abs        <= 1e-5  (~2600× margin)
//!   - fp16 bwd dbias max_abs     <= 1e-5  (~1800× margin)
//!
//! (Wider than V=4096's 1e-3 forward / 1e-3 backward only on the LSE
//! axis where the 384-tile rescale chain genuinely costs precision —
//! the dx/dW/dbias accumulation paths converge faster than ULP × V.)
//!
//! **Audit trail (Finding 5 mitigation)**: empirical baselines above are
//! recorded in `crates/nsl-codegen/tests/data/fused_lce_v49152_baseline.json`
//! along with the GPU SKU, driver version, and PTX-synthesizer entry
//! point at the time of measurement.  Without that artifact the
//! docstring's claimed numbers are not auditable.
//!
//! **Production-distribution caveat (Finding 4)**: this fixture exercises
//! the fp16 kernel against benign inputs that produce a near-uniform
//! softmax (per-logit stddev ≈ 0.033, all 49152 logits clustered within
//! ~±0.1).  It does NOT validate the kernel for real transformer
//! distributions where logit spreads of 5-10 (step 0) to 20+ (converged)
//! put real pressure on the per-tile rescale chain.  Future work should
//! add a second fixture with `W∈N(0,0.05)` + target-class boost to
//! exercise the peaked-softmax regime.

#![cfg(feature = "cuda")]

mod common;

use common::fused_lce_cpu_f64::{
    cpu_lce_backward_f64, cpu_lce_forward_f64, IGNORE_INDEX as CPU_IGNORE_INDEX,
};
use half::f16;
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
const DTYPE_TAG_F16: i64 = 1;

// Compile-time assertion that the local and helper IGNORE_INDEX match —
// catches drift if either side ever changes the sentinel value.
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

/// Deterministic LCG.
fn fill_seeded_uniform(dst: &mut [f32], seed: u64, lo: f32, hi: f32) {
    let mut s = seed;
    let span = hi - lo;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        *x = lo + ((u as f32) / (u32::MAX as f32)) * span;
    }
}

/// Deterministic Gaussian-ish via Box-Muller from LCG uniforms.
/// Sufficient quality for numerical tests — not crypto.
fn fill_seeded_normal(dst: &mut [f32], seed: u64, mean: f32, std: f32) {
    let mut s = seed;
    let mut idx = 0;
    while idx < dst.len() {
        // Generate two uniforms in (0, 1].
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

/// Convert f32 → fp16 bits (round-to-nearest-even via the `half` crate).
fn f32_slice_to_fp16_bits(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s).to_bits();
    }
}

/// Round each f32 through fp16 → f32. Models what the kernel sees in HBM.
fn round_through_fp16(src: &[f32]) -> Vec<f32> {
    src.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

fn f32_to_f64(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&x| x as f64).collect()
}

fn i64_to_i32_targets(src: &[i64]) -> Vec<i32> {
    src.iter()
        .map(|&t| if t == IGNORE_INDEX { CPU_IGNORE_INDEX } else { t as i32 })
        .collect()
}

/// Production-scale fp16 forward+backward numerical pin.
///
/// B=2, S=64, V=49152, H=128, vocab_tile=128.
///   - rows = 128 (20% ignored = 26 rows ignored, 102 valid)
///   - num_tiles = 384 (Kernel A grid = (128 * 384, 1, 1) = 49,152 CTAs)
///   - CPU f64 reference = 102 valid rows * 49152 logits/row * 128 H
///     ≈ 6.4 * 10^8 f64 FMAs — runs in ~30 s on a modern CPU.
#[test]
#[ignore]
fn fp16_forward_backward_at_v49152_production_scale() {
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
        dtype: Dtype::F16,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("V=49152 fp16 config must validate");
    assert!(
        cfg.is_large_vocab(),
        "V=49152 must route through the large-vocab fp16 forward path"
    );

    let rows = B * S;

    // Targets: 20% ignored, valid otherwise.
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        // Hash-mix `i` to spread ignores; ~20% via mod 5 == 0.
        let h = (i.wrapping_mul(2654435761)) & 0xFFFF_FFFF;
        *t = if h % 5 == 0 {
            IGNORE_INDEX
        } else {
            ((i.wrapping_mul(37).wrapping_add(13)) % V) as i64
        };
    }
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as u32;
    eprintln!("V=49152 fp16: rows={rows} num_valid={num_valid}");

    // Host f32 buffers — uniform[-0.5, 0.5] for x; normal(0, 0.01) for W
    // (matches BitNet/transformer init); zeros for bias.
    let mut x_f32 = vec![0f32; rows * H];
    let mut w_f32 = vec![0f32; V * H];
    let bias_f32 = vec![0f32; V];
    fill_seeded_uniform(&mut x_f32, 42, -0.5, 0.5);
    fill_seeded_normal(&mut w_f32, 137, 0.0, 0.01);

    // Round through fp16 — kernel sees these exact rounded values.
    let x_ref = round_through_fp16(&x_f32);
    let w_ref = round_through_fp16(&w_f32);
    let bias_ref = round_through_fp16(&bias_f32);

    // fp16 HBM buffers.
    let mut x_h16 = vec![0u16; rows * H];
    let mut w_h16 = vec![0u16; V * H];
    let mut bias_h16 = vec![0u16; V];
    f32_slice_to_fp16_bits(&x_f32, &mut x_h16);
    f32_slice_to_fp16_bits(&w_f32, &mut w_h16);
    f32_slice_to_fp16_bits(&bias_f32, &mut bias_h16);

    // PTX.
    let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    fwd_ptx.push(0u8);
    let mut bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    bwd_ptx.push(0u8);
    let kname_a = format!("{}\0", cfg.large_partials_kernel_name());
    let kname_b = format!("{}\0", cfg.large_finalize_kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Device buffers.
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
        nsl_test_cuda_h2d(x_dev, x_h16.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev, w_h16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_h16.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev, targets.as_ptr() as i64, tgt_bytes);
    }

    let smem = cfg.shared_mem_bytes();
    let num_tiles = cfg.num_vocab_tiles();
    eprintln!("fp16 V=49152: SMEM={smem} B/CTA, num_tiles={num_tiles}");

    // ─── CPU f64 reference via shared helper ─────────────────────────────────
    eprintln!("computing CPU f64 reference (102 valid rows × 49152 V × 128 H)...");
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

    // ─── GPU forward ────────────────────────────────────────────────────────
    let rc = nsl_fused_linear_ce_forward_large(
        fwd_ptx.as_ptr() as i64,
        kname_a.as_ptr() as i64,
        kname_b.as_ptr() as i64,
        x_dev, w_dev, bias_dev, tgt_dev, partials_dev,
        loss_dev, lse_dev,
        B as i64, S as i64, V as i64, H as i64,
        num_tiles as i64,
        smem as i64,
        DTYPE_TAG_F16,
    );
    assert_eq!(rc, 0, "nsl_fused_linear_ce_forward_large (fp16) failed rc={rc}");

    let mut loss_gpu = vec![0f32; rows];
    let mut lse_gpu = vec![0f32; rows];
    unsafe {
        nsl_test_cuda_d2h(loss_gpu.as_mut_ptr() as i64, loss_dev, loss_bytes);
        nsl_test_cuda_d2h(lse_gpu.as_mut_ptr() as i64, lse_dev, lse_bytes);
    }

    // Forward error metrics.
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
        "fp16 V=49152 forward: gpu_loss={:.6} cpu_loss={:.6} rel_err={:.3e} \
         lse_max_abs={:.3e}",
        gpu_mean_loss, cpu_fwd.mean_loss, mean_loss_rel, max_lse_abs
    );

    assert!(
        mean_loss_rel < 1e-5,
        "fp16 V=49152 fwd mean_loss rel_err {:.3e} exceeds 1e-5 — \
         this is ~400× the empirical 2.47e-8 baseline; a real numerical \
         regression (broken cast, missing -INF sentinel, wrong rescale) \
         would be the most likely cause",
        mean_loss_rel
    );
    assert!(
        max_lse_abs < 1e-4,
        "fp16 V=49152 fwd lse max_abs {:.3e} exceeds 1e-4 — \
         this is ~70× the empirical 1.38e-6 baseline; a regression in \
         the per-tile rescale chain (Kernel A) would be the most likely \
         cause",
        max_lse_abs
    );

    // ─── GPU backward ───────────────────────────────────────────────────────
    // Zero accumulator buffers (red.global.add).
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
            DTYPE_TAG_F16,
        )
    };
    assert_eq!(bwd_rc, 0, "nsl_fused_linear_ce_backward (fp16, V=49152) failed rc={bwd_rc}");

    let mut dx_gpu = vec![0f32; rows * H];
    let mut dw_gpu = vec![0f32; V * H];
    let mut dbias_gpu = vec![0f32; V];
    unsafe {
        nsl_test_cuda_d2h(dx_gpu.as_mut_ptr() as i64, dx_dev, dx_bytes);
        nsl_test_cuda_d2h(dw_gpu.as_mut_ptr() as i64, dw_dev, dw_bytes);
        nsl_test_cuda_d2h(dbias_gpu.as_mut_ptr() as i64, dbias_dev, dbias_bytes);
    }

    // CPU f64 backward reference.
    eprintln!("computing CPU f64 backward reference...");
    let t1 = std::time::Instant::now();
    let cpu_bwd = cpu_lce_backward_f64(
        &x_ref_f64, &w_ref_f64, &bias_ref_f64, &targets_i32,
        &cpu_fwd.lse_per_row, 1.0, rows, V, H,
    );
    eprintln!("CPU backward took {:.1}s", t1.elapsed().as_secs_f64());

    // dx error — skip-identity on ignored rows, max_abs on valid rows.
    let mut dx_max_abs = 0f64;
    for row in 0..rows {
        if targets[row] == IGNORE_INDEX {
            for hi in 0..H {
                let val = dx_gpu[row * H + hi];
                assert_eq!(val, 0.0, "fp16 skip-identity dx[{row},{hi}] = {val}");
            }
        } else {
            for hi in 0..H {
                let d = (dx_gpu[row * H + hi] as f64 - cpu_bwd.dx[row * H + hi]).abs();
                if d > dx_max_abs { dx_max_abs = d; }
            }
        }
    }
    // dW error — every row accumulated.
    let mut dw_max_abs = 0f64;
    for i in 0..V * H {
        let d = (dw_gpu[i] as f64 - cpu_bwd.dw[i]).abs();
        if d > dw_max_abs { dw_max_abs = d; }
    }
    // dbias error.
    let mut db_max_abs = 0f64;
    for vi in 0..V {
        let d = (dbias_gpu[vi] as f64 - cpu_bwd.dbias[vi]).abs();
        if d > db_max_abs { db_max_abs = d; }
    }
    eprintln!(
        "fp16 V=49152 backward: dx_max_abs={:.3e} dw_max_abs={:.3e} dbias_max_abs={:.3e}",
        dx_max_abs, dw_max_abs, db_max_abs
    );

    assert!(
        dx_max_abs < 1e-5,
        "fp16 V=49152 bwd dx max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 5.87e-9; check the backward softmax recompute \
         or dx scatter via W^T",
        dx_max_abs
    );
    assert!(
        dw_max_abs < 1e-5,
        "fp16 V=49152 bwd dW max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 3.78e-9; check the dlogits × x outer-product \
         scatter via red.global.add.f32",
        dw_max_abs
    );
    assert!(
        db_max_abs < 1e-5,
        "fp16 V=49152 bwd dbias max_abs {:.3e} exceeds 1e-5 — \
         empirical baseline 5.39e-9; check the dlogits row-sum scatter",
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

    eprintln!("fp16 V=49152 production-scale numerical pin: PASS");
}
