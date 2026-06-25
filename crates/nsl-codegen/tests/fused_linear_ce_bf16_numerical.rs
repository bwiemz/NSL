//! Sprint v4-1 — bf16 forward+backward GPU numerical validation.
//!
//! Mirrors `fused_linear_ce_fp16_numerical.rs` but uses `Dtype::Bf16`
//! end-to-end. HBM tensors `x`, `W`, `bias` are uploaded as bf16; outputs
//! `loss` and `lse` are f32; gradient outputs `dx`, `dW`, `dbias` are f32
//! (master-grad convention — identical to fp16).
//!
//! Reference: CPU f64 computation that casts inputs through bf16 round-
//! trip (`f32 -> bf16 -> f32`), then expands back to f64 for the
//! reference math — this models exactly what the bf16 kernel sees in HBM.
//!
//! ## Tolerances
//!
//! Bf16 has a **larger exponent range** than fp16 (8-bit vs 5-bit) but
//! the **same total storage** (16 bits), which means **fewer mantissa
//! bits** — only 7 explicit + 1 implicit = ~3 decimal digits, vs fp16's
//! 10 explicit (~3.3 decimal digits). The looser tolerances reflect:
//!   - Forward loss rel-err <= 5e-3
//!   - Backward dx max-abs   <= 1e-3 (same as fp16 — dx is in f32 storage)
//!   - Skip identity: dx[ignore_row, :] == 0.0 exactly
//!
//! ## Coverage (mirroring v3-2 fp16)
//!
//! Test 1: small vocab (V=4096, H=128, B=2, S=64, vocab_tile=1024).
//! Test 2: vocab boundary at V=8192 (still small-vocab path).
//! Test 3: Finding-2 regression guard — non-divisor V=8193 (large path)
//!         seeded with all-negative logits so the tail-zero sentinel
//!         silently corrupting to a finite value would surface as
//!         a loss mismatch.

#![cfg(feature = "cuda")]

use half::bf16;
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
// dtype_tag = 2 for Bf16 (0=F32, 1=F16, 2=Bf16). Today the runtime
// dispatch layer is informational — the kernel itself encodes its own
// element-size — so the value is propagated for future use but not yet
// load-bearing. See runtime/src/fused_linear_ce.rs for the `_dtype_tag`
// underscore-prefixed binding that pins the contract.
const DTYPE_TAG_BF16: i64 = 2;

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

/// Same deterministic LCG as the F16 test, scaled to keep bf16 in range.
fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        // [-0.3, 0.3) — small magnitudes; safe under bf16 cast (bf16 has
        // f32-equivalent exponent range so the lower bound here is just
        // about keeping reference + GPU paths numerically comparable).
        *x = ((u as f32) / (u32::MAX as f32)) * 0.6 - 0.3;
    }
}

/// Convert an f32 slice to bf16 bit-representation (`[u16]`) for HBM upload.
/// Round-to-nearest-even is the half crate's default conversion.
fn f32_slice_to_bf16_bits(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = bf16::from_f32(*s).to_bits();
    }
}

/// Round each f32 element through bf16 → f32 to model the dtype loss the
/// GPU kernel sees. Used to build the CPU reference operating on the
/// same numerical inputs as the kernel.
fn round_through_bf16(src: &[f32]) -> Vec<f32> {
    src.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

/// CPU f64 reference forward operating on already-rounded-through-bf16
/// inputs. Structurally identical to the fp16 reference.
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
fn run_bf16_numerical(b: usize, s: usize, v: usize, h: usize, vocab_tile: u32) {
    let cfg = FusedLinearCEConfig {
        vocab_size: v as u32,
        hidden_size: h as u32,
        seq_len: s as u32,
        batch_size: b as u32,
        vocab_tile,
        gpu_sm: 80,
        dtype: Dtype::Bf16,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: 8192,
    };
    cfg.validate().expect("config must validate");
    assert!(!cfg.is_large_vocab(), "this test exercises the v1 bf16 small path");

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

    // Host f32 buffers (used to seed PRNG values then round through bf16).
    let mut x_f32 = vec![0f32; rows * h];
    let mut w_f32 = vec![0f32; v * h];
    let mut bias_f32 = vec![0f32; v];
    fill_seeded(&mut x_f32, 42);
    fill_seeded(&mut w_f32, 137);
    fill_seeded(&mut bias_f32, 7);

    // Reference operates on the EXACT bf16 values the kernel will see.
    let x_ref = round_through_bf16(&x_f32);
    let w_ref = round_through_bf16(&w_f32);
    let bias_ref = round_through_bf16(&bias_f32);

    // Host bf16 buffers (u16 bit storage).
    let mut x_bf16 = vec![0u16; rows * h];
    let mut w_bf16 = vec![0u16; v * h];
    let mut bias_bf16 = vec![0u16; v];
    f32_slice_to_bf16_bits(&x_f32, &mut x_bf16);
    f32_slice_to_bf16_bits(&w_f32, &mut w_bf16);
    f32_slice_to_bf16_bits(&bias_f32, &mut bias_bf16);

    // PTX.
    let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    fwd_ptx.push(0u8);
    let mut bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    bwd_ptx.push(0u8);
    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Device buffers: x/W/bias are bf16 (2 bytes/elem); loss/lse/dx/dW/dbias f32.
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
        nsl_test_cuda_h2d(x_dev,    x_bf16.as_ptr()    as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev,    w_bf16.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_bf16.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev,  targets.as_ptr()  as i64, tgt_bytes);
    }

    let smem = cfg.shared_mem_bytes();
    eprintln!("bf16 fixture: B={b} S={s} V={v} H={h} num_valid={num_valid} SMEM={smem}");

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
            DTYPE_TAG_BF16,
        )
    };
    assert_eq!(fwd_rc, 0, "nsl_fused_linear_ce_forward (bf16) failed rc={}", fwd_rc);

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
        "bf16 forward loss: gpu={:.6} ref={:.6} rel_err={:.2e}",
        gpu_loss_mean, ref_loss, rel_err
    );
    assert!(
        rel_err < 5e-3,
        "bf16 forward loss rel_err {:.2e} exceeds 5e-3 tolerance (gpu={:.6}, ref={:.6})",
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
            DTYPE_TAG_BF16,
        )
    };
    assert_eq!(bwd_rc, 0, "nsl_fused_linear_ce_backward (bf16) failed rc={}", bwd_rc);

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
                    "bf16 skip-identity violated: dx_gpu[{row},{hi}] = {val}"
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
        "bf16 backward dx max_abs = {:.2e} over {} elements",
        max_abs, n_checked
    );
    assert!(
        max_abs < 1e-3,
        "bf16 backward dx max_abs {:.2e} exceeds 1e-3 tolerance",
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
fn bf16_forward_backward_at_v4096() {
    if !cuda_available() {
        return;
    }
    run_bf16_numerical(2, 64, 4096, 128, 1024);
}

#[test]
#[ignore]
fn bf16_forward_backward_at_v8192_boundary() {
    if !cuda_available() {
        return;
    }
    // V=8192 is exactly the small/large routing threshold (still small).
    run_bf16_numerical(1, 32, 8192, 128, 1024);
}

// ─── Sprint v4-1 — large-vocab bf16 non-divisor regression guard ──────
//
// Mirrors the fp16 Finding-2 regression test for bf16. V=8193 with
// vocab_tile=128 produces ceil(8193/128) = 65 tiles where the last tile
// holds 1 real logit + 127 tail-zero slots. The bf16 emitter MUST store
// `0xFF80` (bf16 -INF) directly via `mov.b16 ; st.shared.b16`; any other
// pattern (the f32-sentinel-then-cvt path the fp16 v3-2 review found, or
// accidentally re-using fp16's `0xFC00`) would silently corrupt the
// per-tile max-reduce when all real logits are negative.
//
// Seeded with NEGATIVE-only logits to expose the bug: a healthy tail-zero
// sentinel of bf16 -INF is ignored by the max-reduce; a broken sentinel
// (finite or zero) would dominate the max when all real logits are below
// 0.0, producing an LSE that's catastrophically wrong.

#[test]
#[ignore]
fn bf16_large_vocab_non_divisor_no_tail_zero_corruption() {
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
        dtype: Dtype::Bf16,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: MAX_VOCAB_HARD_CEILING,
    };
    cfg.validate().expect("V=8193 must validate (no V%vocab_tile gate)");
    assert!(
        cfg.is_large_vocab(),
        "V=8193 must route through the large-vocab Bf16 kernels"
    );

    let rows = B * S;
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        *t = if i % 16 == 0 { IGNORE_INDEX } else { ((i * 37 + 13) % V) as i64 };
    }

    // NEGATIVE-only logits via a deep mean-shift: subtract a large bias from
    // the distribution so EVERY real logit (at every tile, every row) is
    // comfortably below 0.0 — not just the average.
    //
    // Adversarial review Finding 4 (MEDIUM): the previous `bias = -2.0 ± 0.03`
    // did NOT reliably produce all-negative logits at every tile. With
    // x,W ∈ [-0.3, 0.3] and H=128, dot stddev ≈ sqrt(128 * (0.3^2/3)^2) ≈ 0.62,
    // so logits ≈ N(-2, 0.62). Across V=8193 logits per row, expected
    // positive-logit count per row ≈ V * P(z > 3.23σ) ≈ ~8 — meaning most
    // per-tile max-reduces saw a positive real logit, and a buggy tail-zero
    // sentinel of 0.0 would NOT corrupt those tiles (max(positive, 0) =
    // positive). Only the tail tile (1 real + 127 tail-zeros) reliably
    // exercised the bug.
    //
    // bias = -10.0 with stddev ≈ 0.62 puts the +3σ upper bound at ~-8.14;
    // P(logit > 0) ≈ Q(-(-10)/0.62) ≈ Q(16) ≈ 10^-58, so expected positive
    // logits per row across V=8193 is ≈ 10^-54. Every per-tile max-reduce
    // sees ONLY negative real logits, so a buggy non-(-INF) sentinel
    // consistently corrupts every tile (not just the tail).
    let mut x_f32 = vec![0f32; rows * H];
    let mut w_f32 = vec![0f32; V * H];
    let mut bias_f32 = vec![-10.0f32; V];
    fill_seeded(&mut x_f32, 42);
    fill_seeded(&mut w_f32, 137);
    {
        let mut tmp = vec![0f32; V];
        fill_seeded(&mut tmp, 7);
        for (b_, t_) in bias_f32.iter_mut().zip(tmp.iter()) {
            *b_ += *t_ * 0.1;
        }
    }

    let x_ref = round_through_bf16(&x_f32);
    let w_ref = round_through_bf16(&w_f32);
    let bias_ref = round_through_bf16(&bias_f32);

    let mut x_bf16 = vec![0u16; rows * H];
    let mut w_bf16 = vec![0u16; V * H];
    let mut bias_bf16 = vec![0u16; V];
    f32_slice_to_bf16_bits(&x_f32, &mut x_bf16);
    f32_slice_to_bf16_bits(&w_f32, &mut w_bf16);
    f32_slice_to_bf16_bits(&bias_f32, &mut bias_bf16);

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
        nsl_test_cuda_h2d(x_dev, x_bf16.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev, w_bf16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_bf16.as_ptr() as i64, bias_bytes);
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
        DTYPE_TAG_BF16,
    );
    assert_eq!(
        rc, 0,
        "nsl_fused_linear_ce_forward_large (bf16, non-divisor V) failed rc={rc}"
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
        "bf16 V=8193 forward: gpu={gpu_mean_loss:.6} ref={ref_mean_loss:.6} rel_err={rel_err:.2e}"
    );
    assert!(
        rel_err < 5e-3,
        "bf16 V=8193 (non-divisor) forward rel_err {rel_err:.2e} exceeds 5e-3 \
         — a finite tail-zero sentinel (instead of bf16 -INF 0xFF80) would \
         dominate the per-tile max-reduce when all real logits are negative"
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
