//! Sprint v3-2 — fp16 forward+backward GPU numerical validation.
//!
//! Mirrors `fused_linear_ce_numerical.rs` but uses `Dtype::F16` end-to-end.
//! HBM tensors `x`, `W`, `bias` are uploaded as fp16; outputs `loss` and
//! `lse` are f32; gradient outputs `dx`, `dW`, `dbias` are f32 (per
//! Sprint v3-2 master-grad convention).
//!
//! Reference: CPU f64 computation that casts inputs to fp16, then expands
//! back to f64 for the reference math — this models exactly what the fp16
//! kernel sees in HBM.
//!
//! ## Tolerances
//!
//! Looser than the f32 path because fp16 input precision dominates (the
//! online-LSE algorithm + dot-product accumulator are still f32):
//!   - Forward loss rel-err <= 1e-3
//!   - Backward dx max-abs   <= 1e-3
//!   - Skip identity: dx[ignore_row, :] == 0.0 exactly
//!
//! ## Coverage
//!
//! Test 1: small vocab (V=4096, H=128, B=2, S=64, vocab_tile=1024).
//! Test 2: vocab boundary at V=8192 (still small-vocab path; the next
//! tile up would route to the large-vocab kernels which are gated by
//! the ptxas test for this sprint).
//!
//! Large-vocab fp16 GPU validation deferred per spec: "the fp16 large-
//! vocab path is supported but adds GPU test setup complexity; gate it
//! via the ptxas test which proves the kernel assembles".

#![cfg(feature = "cuda")]

use half::f16;
use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig,
    synthesize_fused_linear_ce_backward_ptx, synthesize_fused_linear_ce_ptx,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_d2h, nsl_test_cuda_h2d,
    nsl_fused_linear_ce_backward, nsl_fused_linear_ce_forward,
};

const IGNORE_INDEX: i64 = -100;
const DTYPE_TAG_F16: i64 = 1;

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        false
    } else {
        true
    }
}

/// Same deterministic LCG as the F32 test, scaled to keep fp16 in range.
fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        // [-0.3, 0.3) — small magnitudes; safe under fp16 cast.
        *x = ((u as f32) / (u32::MAX as f32)) * 0.6 - 0.3;
    }
}

/// Convert an f32 slice to fp16 bit-representation (`[u16]`) for HBM upload.
/// Round-to-nearest-even is the half crate's default conversion.
fn f32_slice_to_fp16_bits(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s).to_bits();
    }
}

/// Round each f32 element through fp16 → f32 to model the dtype loss the
/// GPU kernel sees. Used to build the CPU reference operating on the
/// same numerical inputs as the kernel.
fn round_through_fp16(src: &[f32]) -> Vec<f32> {
    src.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

/// CPU f64 reference forward operating on already-rounded-through-fp16 inputs.
fn cpu_reference_forward(
    b: usize,
    s: usize,
    v: usize,
    h: usize,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    targets: &[i64],
) -> (f64, Vec<f64>) {
    let rows = b * s;
    let mut lse = vec![0f64; rows];
    let mut total_loss = 0f64;
    let mut num_valid = 0usize;

    for row in 0..rows {
        let t = targets[row];
        if t == IGNORE_INDEX {
            continue;
        }
        // Compute this row's logits in f64.
        let mut row_logits = vec![0f64; v];
        for vi in 0..v {
            let mut acc = bias[vi] as f64;
            for hi in 0..h {
                acc += x[row * h + hi] as f64 * w[vi * h + hi] as f64;
            }
            row_logits[vi] = acc;
        }
        let max_logit = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let lse_val = sum_exp.ln() + max_logit;
        lse[row] = lse_val;
        let tgt = t as usize;
        total_loss += -(row_logits[tgt] - lse_val);
        num_valid += 1;
    }

    let mean_loss = total_loss / num_valid.max(1) as f64;
    (mean_loss, lse)
}

/// CPU f64 reference backward on rounded inputs.
fn cpu_reference_backward(
    b: usize,
    s: usize,
    v: usize,
    h: usize,
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    targets: &[i64],
    lse: &[f64],
    grad_output: f64,
) -> Vec<f32> {
    let rows = b * s;
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
    let inv_nv = grad_output / num_valid.max(1) as f64;
    let mut dx = vec![0f32; rows * h];

    for row in 0..rows {
        let t = targets[row];
        if t == IGNORE_INDEX {
            continue;
        }
        let tgt = t as usize;
        let mut row_logits = vec![0f64; v];
        for vi in 0..v {
            let mut acc = bias[vi] as f64;
            for hi in 0..h {
                acc += x[row * h + hi] as f64 * w[vi * h + hi] as f64;
            }
            row_logits[vi] = acc;
        }
        let lse_val = lse[row];
        let mut dlogits = vec![0f64; v];
        for vi in 0..v {
            let p_v = (row_logits[vi] - lse_val).exp();
            let one_hot = if vi == tgt { 1.0f64 } else { 0.0 };
            dlogits[vi] = (p_v - one_hot) * inv_nv;
        }
        for hi in 0..h {
            let mut acc = 0f64;
            for vi in 0..v {
                acc += dlogits[vi] * w[vi * h + hi] as f64;
            }
            dx[row * h + hi] = acc as f32;
        }
    }
    dx
}

/// Body shared by V=4096 and V=8192 cases.
fn run_fp16_numerical(b: usize, s: usize, v: usize, h: usize, vocab_tile: u32) {
    let cfg = FusedLinearCEConfig {
        vocab_size: v as u32,
        hidden_size: h as u32,
        seq_len: s as u32,
        batch_size: b as u32,
        vocab_tile,
        gpu_sm: 80,
        dtype: Dtype::F16,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: 8192,
    };
    cfg.validate().expect("config must validate");
    assert!(!cfg.is_large_vocab(), "this test exercises the v1 fp16 small path");

    let rows = b * s;

    // Targets: -100 every 16th position.
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        *t = if i % 16 == 0 {
            IGNORE_INDEX
        } else {
            ((i * 37 + 13) % v) as i64
        };
    }
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as u32;

    // Host f32 buffers (used to seed PRNG values then round through fp16).
    let mut x_f32 = vec![0f32; rows * h];
    let mut w_f32 = vec![0f32; v * h];
    let mut bias_f32 = vec![0f32; v];
    fill_seeded(&mut x_f32, 42);
    fill_seeded(&mut w_f32, 137);
    fill_seeded(&mut bias_f32, 7);

    // Reference operates on the EXACT fp16 values the kernel will see.
    let x_ref = round_through_fp16(&x_f32);
    let w_ref = round_through_fp16(&w_f32);
    let bias_ref = round_through_fp16(&bias_f32);

    // Host fp16 buffers (u16 bit storage).
    let mut x_h16 = vec![0u16; rows * h];
    let mut w_h16 = vec![0u16; v * h];
    let mut bias_h16 = vec![0u16; v];
    f32_slice_to_fp16_bits(&x_f32, &mut x_h16);
    f32_slice_to_fp16_bits(&w_f32, &mut w_h16);
    f32_slice_to_fp16_bits(&bias_f32, &mut bias_h16);

    // PTX.
    let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    fwd_ptx.push(0u8);
    let mut bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    bwd_ptx.push(0u8);
    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Device buffers: x/W/bias are fp16 (2 bytes/elem); loss/lse/dx/dW/dbias are f32 (4 bytes).
    let x_bytes     = (rows * h * 2) as i64;
    let w_bytes     = (v * h * 2) as i64;
    let bias_bytes  = (v * 2) as i64;
    let tgt_bytes   = (rows * 8) as i64;
    let loss_bytes  = (rows * 4) as i64;
    let lse_bytes   = (rows * 4) as i64;
    let dx_bytes    = (rows * h * 4) as i64;
    let dw_bytes    = (v * h * 4) as i64;
    let dbias_bytes = (v * 4) as i64;

    let x_dev     = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let w_dev     = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let bias_dev  = unsafe { nsl_test_cuda_alloc(bias_bytes) };
    let tgt_dev   = unsafe { nsl_test_cuda_alloc(tgt_bytes) };
    let loss_dev  = unsafe { nsl_test_cuda_alloc(loss_bytes) };
    let lse_dev   = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let dx_dev    = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    let dw_dev    = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dbias_dev = unsafe { nsl_test_cuda_alloc(dbias_bytes) };
    assert!(x_dev != 0 && w_dev != 0 && bias_dev != 0);
    assert!(tgt_dev != 0 && loss_dev != 0 && lse_dev != 0);
    assert!(dx_dev != 0 && dw_dev != 0 && dbias_dev != 0);

    unsafe {
        nsl_test_cuda_h2d(x_dev,    x_h16.as_ptr()    as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev,    w_h16.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_h16.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev,  targets.as_ptr()  as i64, tgt_bytes);
    }

    let smem = cfg.shared_mem_bytes();
    eprintln!("fp16 fixture: B={b} S={s} V={v} H={h} num_valid={num_valid} SMEM={smem}");

    // CPU reference forward.
    let (ref_loss, ref_lse) = cpu_reference_forward(
        b, s, v, h, &x_ref, &w_ref, &bias_ref, &targets,
    );
    eprintln!("CPU reference loss = {:.6}", ref_loss);

    // GPU forward.
    let fwd_rc = unsafe {
        nsl_fused_linear_ce_forward(
            fwd_ptx.as_ptr() as i64,
            fwd_name.as_ptr() as i64,
            x_dev, w_dev, bias_dev, tgt_dev, loss_dev, lse_dev,
            b as i64, s as i64, v as i64, h as i64,
            smem as i64,
            DTYPE_TAG_F16,
        )
    };
    assert_eq!(fwd_rc, 0, "nsl_fused_linear_ce_forward (fp16) failed rc={}", fwd_rc);

    let mut loss_host = vec![0f32; rows];
    unsafe {
        nsl_test_cuda_d2h(loss_host.as_mut_ptr() as i64, loss_dev, loss_bytes);
    }
    let mut gpu_loss_sum = 0f64;
    let mut gpu_nv = 0usize;
    for (i, &lv) in loss_host.iter().enumerate() {
        if targets[i] != IGNORE_INDEX {
            gpu_loss_sum += lv as f64;
            gpu_nv += 1;
        }
    }
    let gpu_loss_mean = gpu_loss_sum / gpu_nv.max(1) as f64;
    let rel_err = (gpu_loss_mean - ref_loss).abs() / ref_loss.abs().max(1.0);
    eprintln!(
        "fp16 forward loss: gpu={:.6} ref={:.6} rel_err={:.2e}",
        gpu_loss_mean, ref_loss, rel_err
    );
    assert!(
        rel_err < 1e-3,
        "fp16 forward loss rel_err {:.2e} exceeds 1e-3 tolerance (gpu={:.6}, ref={:.6})",
        rel_err, gpu_loss_mean, ref_loss
    );

    // Zero grads before backward (atomics accumulate).
    {
        let zeros_dx    = vec![0f32; rows * h];
        let zeros_dw    = vec![0f32; v * h];
        let zeros_dbias = vec![0f32; v];
        unsafe {
            nsl_test_cuda_h2d(dx_dev,    zeros_dx.as_ptr()    as i64, dx_bytes);
            nsl_test_cuda_h2d(dw_dev,    zeros_dw.as_ptr()    as i64, dw_bytes);
            nsl_test_cuda_h2d(dbias_dev, zeros_dbias.as_ptr() as i64, dbias_bytes);
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
            b as i64, s as i64, v as i64, h as i64,
            num_valid as i64,
            smem as i64,
            DTYPE_TAG_F16,
        )
    };
    assert_eq!(bwd_rc, 0, "nsl_fused_linear_ce_backward (fp16) failed rc={}", bwd_rc);

    let mut dx_gpu = vec![0f32; rows * h];
    unsafe {
        nsl_test_cuda_d2h(dx_gpu.as_mut_ptr() as i64, dx_dev, dx_bytes);
    }

    let dx_ref = cpu_reference_backward(
        b, s, v, h, &x_ref, &w_ref, &bias_ref, &targets, &ref_lse, 1.0,
    );

    let mut max_abs = 0f32;
    let mut n_checked = 0usize;
    for row in 0..rows {
        if targets[row] == IGNORE_INDEX {
            for hi in 0..h {
                let val = dx_gpu[row * h + hi];
                assert_eq!(
                    val, 0.0f32,
                    "fp16 skip-identity violated: dx_gpu[{row},{hi}] = {val}"
                );
            }
        } else {
            for hi in 0..h {
                let err = (dx_gpu[row * h + hi] - dx_ref[row * h + hi]).abs();
                if err > max_abs { max_abs = err; }
                n_checked += 1;
            }
        }
    }
    eprintln!(
        "fp16 backward dx max_abs = {:.2e} over {} elements",
        max_abs, n_checked
    );
    assert!(
        max_abs < 1e-3,
        "fp16 backward dx max_abs {:.2e} exceeds 1e-3 tolerance",
        max_abs
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
    }
}

#[test]
#[ignore]
fn fp16_forward_backward_at_v4096() {
    if !cuda_available() {
        return;
    }
    run_fp16_numerical(2, 64, 4096, 128, 1024);
}

#[test]
#[ignore]
fn fp16_forward_backward_at_v8192_boundary() {
    if !cuda_available() {
        return;
    }
    // V=8192 is exactly the small/large routing threshold (still small).
    run_fp16_numerical(1, 32, 8192, 128, 1024);
}

// ─── Review Finding 2 — large-vocab F16 non-divisor regression guard ──
//
// V=8193 with vocab_tile=128 produces ceil(8193/128) = 65 tiles where
// the last tile has 1 real logit + 127 tail-zero slots.  Pre-fix, every
// tail slot wrote fp16 0x0000 (the broken cvt path), and the per-tile
// max-reduce folded 0.0 into the cross-CTA LSE — corrupting forward
// loss whenever all real logits were negative.  Post-fix the tail
// slots hold fp16 -INF (0xFC00) so the max-reduce ignores them.
//
// We deliberately pick V=8193 (just over the LARGE_VOCAB_THRESHOLD of
// 8192) instead of V=49153 to keep the CPU f64 reference under a
// minute even on modest hosts (rows=32 × 8193 × 128 = ~33M f64 muls).
// Routes through the SAME `emit_large_partials_kernel_f16` path the
// production V=49152 fixture uses — the tail-zero branch and its
// reduction-time consumer are identical.

#[test]
#[ignore]
fn fp16_large_vocab_non_divisor_no_tail_zero_corruption() {
    if !cuda_available() {
        return;
    }

    use nsl_codegen::fused_linear_ce::MAX_VOCAB_HARD_CEILING;
    use nsl_runtime::nsl_fused_linear_ce_forward_large;

    const B: usize = 1;
    const S: usize = 32;
    const V: usize = 8193; // V % vocab_tile != 0 — exercises tail-zero branch
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
    cfg.validate().expect("V=8193 must validate (no V%vocab_tile gate)");
    assert!(
        cfg.is_large_vocab(),
        "V=8193 must route through the large-vocab F16 kernels"
    );

    let rows = B * S;
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        *t = if i % 16 == 0 { IGNORE_INDEX } else { ((i * 37 + 13) % V) as i64 };
    }

    // Seed with NEGATIVE-only logits to expose the bug: pre-fix the
    // tail slots stored fp16 0.0 (the broken cvt path), which would
    // then dominate the max-reduce — corrupting the LSE and the loss.
    // Post-fix the tail stores fp16 -INF, so 0.0 never appears.
    //
    // Negative logits via a tiny mean-shift: subtract a bias from the
    // distribution so the max real logit is comfortably below 0.0.
    let mut x_f32 = vec![0f32; rows * H];
    let mut w_f32 = vec![0f32; V * H];
    let mut bias_f32 = vec![-2.0f32; V]; // baseline negative bias
    fill_seeded(&mut x_f32, 42);
    fill_seeded(&mut w_f32, 137);
    // additive perturbation on top of the negative baseline
    {
        let mut tmp = vec![0f32; V];
        fill_seeded(&mut tmp, 7);
        for (b_, t_) in bias_f32.iter_mut().zip(tmp.iter()) {
            *b_ += *t_ * 0.1;
        }
    }

    let x_ref = round_through_fp16(&x_f32);
    let w_ref = round_through_fp16(&w_f32);
    let bias_ref = round_through_fp16(&bias_f32);

    let mut x_h16 = vec![0u16; rows * H];
    let mut w_h16 = vec![0u16; V * H];
    let mut bias_h16 = vec![0u16; V];
    f32_slice_to_fp16_bits(&x_f32, &mut x_h16);
    f32_slice_to_fp16_bits(&w_f32, &mut w_h16);
    f32_slice_to_fp16_bits(&bias_f32, &mut bias_h16);

    let mut ptx = nsl_codegen::fused_linear_ce::synthesize_fused_linear_ce_ptx(&cfg);
    ptx.push(0u8);
    let kname_a = format!("{}\0", cfg.large_partials_kernel_name());
    let kname_b = format!("{}\0", cfg.large_finalize_kernel_name());

    let x_bytes = (rows * H * 2) as i64;
    let w_bytes = (V * H * 2) as i64;
    let bias_bytes = (V * 2) as i64;
    let tgt_bytes = (rows * 8) as i64;
    let loss_bytes = (rows * 4) as i64;
    let lse_bytes = (rows * 4) as i64;
    let partials_bytes = cfg.large_partials_bytes() as i64;

    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let w_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let bias_dev = unsafe { nsl_test_cuda_alloc(bias_bytes) };
    let tgt_dev = unsafe { nsl_test_cuda_alloc(tgt_bytes) };
    let loss_dev = unsafe { nsl_test_cuda_alloc(loss_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let partials_dev = unsafe { nsl_test_cuda_alloc(partials_bytes) };

    unsafe {
        nsl_test_cuda_h2d(x_dev, x_h16.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev, w_h16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_h16.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev, targets.as_ptr() as i64, tgt_bytes);
    }

    let (ref_mean_loss, _ref_lse) =
        cpu_reference_forward(B, S, V, H, &x_ref, &w_ref, &bias_ref, &targets);
    eprintln!("CPU ref mean loss = {ref_mean_loss:.6}");

    let smem = cfg.shared_mem_bytes();
    let num_tiles = cfg.num_vocab_tiles();
    let rc = nsl_fused_linear_ce_forward_large(
        ptx.as_ptr() as i64,
        kname_a.as_ptr() as i64,
        kname_b.as_ptr() as i64,
        x_dev, w_dev, bias_dev, tgt_dev, partials_dev,
        loss_dev, lse_dev,
        B as i64, S as i64, V as i64, H as i64,
        num_tiles as i64,
        smem as i64,
        DTYPE_TAG_F16,
    );
    assert_eq!(
        rc, 0,
        "nsl_fused_linear_ce_forward_large (fp16, non-divisor V) failed rc={rc}"
    );

    let mut loss_host = vec![0f32; rows];
    unsafe {
        nsl_test_cuda_d2h(loss_host.as_mut_ptr() as i64, loss_dev, loss_bytes);
    }
    let mut gpu_loss_sum = 0f64;
    let mut gpu_nv = 0usize;
    for (i, &lv) in loss_host.iter().enumerate() {
        if targets[i] != IGNORE_INDEX {
            gpu_loss_sum += lv as f64;
            gpu_nv += 1;
        }
    }
    let gpu_mean_loss = gpu_loss_sum / gpu_nv.max(1) as f64;
    let rel_err =
        (gpu_mean_loss - ref_mean_loss).abs() / ref_mean_loss.abs().max(1.0);
    eprintln!(
        "fp16 V=8193 forward: gpu={gpu_mean_loss:.6} ref={ref_mean_loss:.6} rel_err={rel_err:.2e}"
    );
    assert!(
        rel_err < 1e-3,
        "fp16 V=8193 (non-divisor) forward rel_err {rel_err:.2e} exceeds 1e-3 \
         — pre-fix this fixture would have shown corruption when all real \
         logits were negative (tail-zero stored fp16 0.0 instead of -INF, \
         dominating the per-tile max-reduce)"
    );

    unsafe {
        nsl_test_cuda_free(x_dev);
        nsl_test_cuda_free(w_dev);
        nsl_test_cuda_free(bias_dev);
        nsl_test_cuda_free(tgt_dev);
        nsl_test_cuda_free(loss_dev);
        nsl_test_cuda_free(lse_dev);
        nsl_test_cuda_free(partials_dev);
    }
}
