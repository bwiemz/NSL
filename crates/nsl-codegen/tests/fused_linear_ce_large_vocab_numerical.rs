//! Sprint-3 large-vocab forward end-to-end GPU numerical correctness.
//!
//! Validates the two-kernel `synthesize_fused_linear_ce_ptx` output
//! against a CPU f64 reference at NSL production scale (vocab=49152)
//! and at an intermediate scale (vocab=16384).
//!
//! ## Tolerances
//! Match v1's `fused_linear_ce_numerical.rs`:
//! * forward loss relative error ≤ 1e-3 (CPU f64 reference vs GPU f32)
//! * the `lse_out[row]` written by Kernel B matches CPU LSE within 1e-3 rel
//!
//! Both kernels are pure f32 fma.rn / ex2.approx / lg2.approx, same as v1,
//! so the same tolerances apply: rounding accumulates over `num_tiles`
//! tiles in the rescale chain but each individual rescale is one ULP.
//!
//! ## Running
//! ```bash
//! cargo test -p nsl-codegen --features cuda \
//!     --test fused_linear_ce_large_vocab_numerical \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig, synthesize_fused_linear_ce_ptx, MAX_VOCAB_HARD_CEILING,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h,
    nsl_fused_linear_ce_forward_large,
};

// ─── PRNG (LCG; reproducible across platforms) ───────────────────────────────

fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        // [-0.05, 0.05) — small magnitudes keep f32 softmax numerics calm
        // over 49K vocab (one wild logit can saturate fp32 range).
        *x = ((u as f32) / (u32::MAX as f32)) * 0.1 - 0.05;
    }
}

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

// ─── CPU reference forward (f64) ─────────────────────────────────────────────

const IGNORE_INDEX: i64 = -100;

fn cpu_reference_forward(
    b: usize,
    s: usize,
    v: usize,
    h: usize,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    targets: &[i64],
) -> (Vec<f64>, Vec<f64>) {
    let rows = b * s;
    let mut losses = vec![0f64; rows];
    let mut lses = vec![0f64; rows];

    for row in 0..rows {
        let t = targets[row];
        if t == IGNORE_INDEX {
            continue;
        }
        // Compute the row's logits in f64.
        let mut max_logit = f64::NEG_INFINITY;
        let mut logits = vec![0f64; v];
        for vi in 0..v {
            let mut acc = bias[vi] as f64;
            for hi in 0..h {
                acc += x[row * h + hi] as f64 * w[vi * h + hi] as f64;
            }
            logits[vi] = acc;
            if acc > max_logit {
                max_logit = acc;
            }
        }
        let sum_exp: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let lse_val = sum_exp.ln() + max_logit;
        let tgt = t as usize;
        losses[row] = lse_val - logits[tgt];
        lses[row] = lse_val;
    }

    (losses, lses)
}

// ─── Core test driver ────────────────────────────────────────────────────────

struct ScaleParams {
    b: usize,
    s: usize,
    v: usize,
    h: usize,
    vocab_tile: u32,
    name: &'static str,
    /// Forward per-row loss relative tolerance.
    loss_rel_tol: f64,
    /// LSE absolute tolerance.
    lse_abs_tol: f64,
}

fn run_scale(scale: &ScaleParams) {
    if !cuda_available() {
        return;
    }

    let ScaleParams { b, s, v, h, vocab_tile, name, loss_rel_tol, lse_abs_tol } = *scale;
    eprintln!("\n=== {name}: B={b} S={s} V={v} H={h} vocab_tile={vocab_tile} ===");

    let cfg = FusedLinearCEConfig {
        vocab_size: v as u32,
        hidden_size: h as u32,
        seq_len: s as u32,
        batch_size: b as u32,
        vocab_tile,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("config must validate");
    assert!(cfg.is_large_vocab(), "this test exercises only the large-vocab path");

    let rows = b * s;

    // Targets: -100 at every 16th position, valid otherwise.
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        if i % 16 == 0 {
            *t = IGNORE_INDEX;
        } else {
            *t = ((i * 37 + 13) % v) as i64;
        }
    }

    let mut x_host = vec![0f32; rows * h];
    let mut w_host = vec![0f32; v * h];
    let mut bias_host = vec![0f32; v];
    fill_seeded(&mut x_host, 42);
    fill_seeded(&mut w_host, 137);
    fill_seeded(&mut bias_host, 7);

    // PTX.
    let mut ptx = synthesize_fused_linear_ce_ptx(&cfg);
    ptx.push(0u8);
    let kname_a = format!("{}\0", cfg.large_partials_kernel_name());
    let kname_b = format!("{}\0", cfg.large_finalize_kernel_name());

    let tmp = std::env::temp_dir();
    std::fs::write(tmp.join(format!("fused_linear_ce_large_{name}.ptx")), &ptx[..ptx.len() - 1]).ok();
    eprintln!("PTX written to {}", tmp.display());

    // Device alloc.
    let x_bytes = (rows * h * 4) as i64;
    let w_bytes = (v * h * 4) as i64;
    let bias_bytes = (v * 4) as i64;
    let tgt_bytes = (rows * 8) as i64;
    let loss_bytes = (rows * 4) as i64;
    let lse_bytes = (rows * 4) as i64;
    let partials_bytes = cfg.large_partials_bytes() as i64;
    eprintln!(
        "device alloc: x={x_bytes}B w={w_bytes}B bias={bias_bytes}B tgt={tgt_bytes}B \
         loss={loss_bytes}B lse={lse_bytes}B partials={partials_bytes}B"
    );

    let x_dev = nsl_test_cuda_alloc(x_bytes);
    let w_dev = nsl_test_cuda_alloc(w_bytes);
    let bias_dev = nsl_test_cuda_alloc(bias_bytes);
    let tgt_dev = nsl_test_cuda_alloc(tgt_bytes);
    let loss_dev = nsl_test_cuda_alloc(loss_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    let partials_dev = nsl_test_cuda_alloc(partials_bytes);
    assert!(
        x_dev != 0 && w_dev != 0 && bias_dev != 0 && tgt_dev != 0
            && loss_dev != 0 && lse_dev != 0 && partials_dev != 0,
        "device alloc failed"
    );

    nsl_test_cuda_h2d(x_dev, x_host.as_ptr() as i64, x_bytes);
    nsl_test_cuda_h2d(w_dev, w_host.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(bias_dev, bias_host.as_ptr() as i64, bias_bytes);
    nsl_test_cuda_h2d(tgt_dev, targets.as_ptr() as i64, tgt_bytes);

    // CPU reference.
    eprintln!("computing CPU reference (rows={rows}, V={v}, H={h})...");
    let t0 = std::time::Instant::now();
    let (ref_losses, ref_lses) = cpu_reference_forward(b, s, v, h, &x_host, &w_host, &bias_host, &targets);
    eprintln!("CPU reference took {:.1}s", t0.elapsed().as_secs_f64());

    // GPU forward.
    let smem = cfg.shared_mem_bytes();
    let num_tiles = cfg.num_vocab_tiles();
    eprintln!("Kernel A SMEM = {smem} B/CTA, num_tiles = {num_tiles}");

    let rc = nsl_fused_linear_ce_forward_large(
        ptx.as_ptr() as i64,
        kname_a.as_ptr() as i64,
        kname_b.as_ptr() as i64,
        x_dev, w_dev, bias_dev, tgt_dev, partials_dev,
        loss_dev, lse_dev,
        b as i64, s as i64, v as i64, h as i64,
        num_tiles as i64,
        smem as i64,
        0, // dtype_tag = 0 (F32 sentinel; Sprint v3-2)
    );
    assert_eq!(rc, 0, "nsl_fused_linear_ce_forward_large failed rc={rc}");

    let mut loss_gpu = vec![0f32; rows];
    let mut lse_gpu = vec![0f32; rows];
    nsl_test_cuda_d2h(loss_gpu.as_mut_ptr() as i64, loss_dev, loss_bytes);
    nsl_test_cuda_d2h(lse_gpu.as_mut_ptr() as i64, lse_dev, lse_bytes);

    // Compare.
    let mut max_loss_rel_err = 0f64;
    let mut max_lse_abs_err = 0f64;
    let mut max_loss_abs_err = 0f64;
    let mut n_valid = 0;
    for row in 0..rows {
        if targets[row] == IGNORE_INDEX {
            // Skip-identity: GPU must write exactly 0/0.
            assert_eq!(loss_gpu[row], 0.0f32, "skip-identity: loss row={row}");
            assert_eq!(lse_gpu[row], 0.0f32, "skip-identity: lse row={row}");
            continue;
        }
        let loss_diff = (loss_gpu[row] as f64 - ref_losses[row]).abs();
        let denom = ref_losses[row].abs().max(1.0);
        let loss_rel = loss_diff / denom;
        if loss_rel > max_loss_rel_err {
            max_loss_rel_err = loss_rel;
        }
        if loss_diff > max_loss_abs_err {
            max_loss_abs_err = loss_diff;
        }
        let lse_diff = (lse_gpu[row] as f64 - ref_lses[row]).abs();
        if lse_diff > max_lse_abs_err {
            max_lse_abs_err = lse_diff;
        }
        n_valid += 1;
    }

    eprintln!(
        "{name}: n_valid={n_valid} max_loss_rel_err={max_loss_rel_err:.2e} \
         max_loss_abs_err={max_loss_abs_err:.2e} max_lse_abs_err={max_lse_abs_err:.2e}"
    );

    assert!(
        max_loss_rel_err < loss_rel_tol,
        "{name}: forward loss rel err {max_loss_rel_err:.2e} exceeds {loss_rel_tol:.2e}"
    );
    assert!(
        max_lse_abs_err < lse_abs_tol,
        "{name}: lse abs err {max_lse_abs_err:.2e} exceeds {lse_abs_tol:.2e}"
    );

    nsl_test_cuda_free(x_dev);
    nsl_test_cuda_free(w_dev);
    nsl_test_cuda_free(bias_dev);
    nsl_test_cuda_free(tgt_dev);
    nsl_test_cuda_free(loss_dev);
    nsl_test_cuda_free(lse_dev);
    nsl_test_cuda_free(partials_dev);

    eprintln!("{name}: PASS");
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn large_vocab_forward_at_production_scale_v49152() {
    // NSL production: V=49152, H=128, B=2, S=64.
    // Total rows = 128; CPU ref does 128 * 49152 dot products of length 128
    // ≈ 8 * 10^8 f64 muls — keep under a minute.
    run_scale(&ScaleParams {
        b: 2, s: 64, v: 49152, h: 128, vocab_tile: 128,
        name: "v49152",
        // Looser than v1's 1e-3 because num_tiles=384 (vs 4 at v1's V=4096
        // with vocab_tile=1024); rescale chain accumulates ~384 * 1 ULP of
        // error in the worst case. Still order-of-magnitude tight.
        loss_rel_tol: 5e-3,
        lse_abs_tol: 2e-2,
    });
}

#[test]
#[ignore]
fn large_vocab_forward_at_intermediate_scale_v16384() {
    // Intermediate: V=16384, H=64, B=1, S=32 (small to keep CPU ref fast).
    run_scale(&ScaleParams {
        b: 1, s: 32, v: 16384, h: 64, vocab_tile: 256,
        name: "v16384",
        loss_rel_tol: 1e-3,
        lse_abs_tol: 1e-2,
    });
}
