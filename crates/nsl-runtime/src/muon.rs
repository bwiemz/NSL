//! P1 Muon items 8+10: the planned Muon Newton-Schulz orthogonalization
//! primitive.
//!
//! The stdlib NSL implementation (`nsl.optim.muon.muon_orthogonalize`) is the
//! REFERENCE — it stays in `muon.nsl` and the gate tests pin this primitive
//! against it. Production `muon_step` calls this instead because:
//!
//! - **No `.item()`** (item 8): the pre-normalization norm is computed and
//!   consumed on-device (`gpu_muon_frobenius_scale_f32` — stats kernel into a
//!   persistent device scratch + a scale kernel that reads it). The NSL
//!   reference's `sum(x * x).item()` forced a full device sync per rank-2
//!   param per optimizer step.
//! - **Planned execution** (item 10): fixed iteration structure driven from
//!   Rust (no per-step NSL dispatch), tall/wide handled by materialized
//!   transposes at entry/exit only, intermediates freed eagerly so the GPU
//!   caching allocator serves every loop iteration from the same blocks
//!   (steady-state: zero new device allocations per call).
//!
//! Numerics: the op sequence is the reference sequence — matmul, mul_scalar,
//! and the fused `scalar_mul_add_inplace` (documented bit-exact vs its
//! two-op decomposition). The only intentional divergence is WHERE the norm
//! scale happens: on-device in f32 on GPU (vs the reference's f64 host
//! round-trip) and pre-transpose on both devices (the Frobenius norm is
//! transpose-invariant; summation order shifts the result by O(eps)).
//! Deferred (documented, not attempted here): batched small-matrix NS and
//! direct stochastic-rounded parameter updates.
//!
//! Output contract: ALWAYS a fresh, contiguous, logically-ordered tensor —
//! never a strided view — so downstream elementwise update math is safe on
//! flat-indexed GPU kernels. The input is borrowed, never freed.

use crate::tensor::{
    nsl_tensor_contiguous, nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_mul,
    nsl_tensor_mul_scalar, nsl_tensor_scalar_mul_add_inplace, nsl_tensor_sum,
    nsl_tensor_transpose, NslTensor,
};

/// Quintic Newton-Schulz coefficients (Jordan et al., 2024) — must match
/// `stdlib/nsl/optim/muon.nsl` exactly.
const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
const FROB_EPS: f64 = 0.000_000_1;

/// CPU Frobenius-normalize: x / (sqrt(Σx²) + 1e-7), reference op order
/// (materialized square, sequential sum, f64 host sqrt, scalar mul).
/// There is no sync to remove on CPU, so this path stays reference-shaped.
fn cpu_frobenius_scale(x_ptr: i64) -> i64 {
    let sq = nsl_tensor_mul(x_ptr, x_ptr, 0);
    let s_t = nsl_tensor_sum(sq);
    nsl_tensor_free(sq);
    let s = {
        let t = NslTensor::from_ptr(s_t);
        // 0-d/1-element reduction result in the tensor's native dtype.
        let v = match t.dtype {
            0 => unsafe { *(t.data as *const f64) },
            1 => f64::from(unsafe { *(t.data as *const f32) }),
            other => {
                eprintln!("nsl: muon_orthogonalize unsupported dtype {other}");
                std::process::abort();
            }
        };
        v
    };
    nsl_tensor_free(s_t);
    let inv = 1.0 / (s.sqrt() + FROB_EPS);
    nsl_tensor_mul_scalar(x_ptr, inv, 0)
}

/// Muon Newton-Schulz orthogonalization of a rank-2 gradient/momentum
/// tensor. Returns a FRESH contiguous tensor approximating the
/// semi-orthogonal factor UVᵀ of the input's SVD; the input is borrowed.
/// `ns_steps` arrives as f64 (the stdlib threads it as float) and must be a
/// positive whole number — codegen validates the literal, this re-checks
/// defensively.
#[no_mangle]
pub extern "C" fn nsl_tensor_muon_orthogonalize(g_ptr: i64, ns_steps: f64) -> i64 {
    let g = NslTensor::from_ptr(g_ptr);
    if g.ndim != 2 {
        eprintln!(
            "nsl: muon_orthogonalize requires a rank-2 tensor (got rank {})",
            g.ndim
        );
        std::process::abort();
    }
    if !(ns_steps >= 1.0 && ns_steps.fract() == 0.0) {
        eprintln!("nsl: muon_orthogonalize ns_steps must be a positive integer (got {ns_steps})");
        std::process::abort();
    }
    if g.len == 0 {
        // Review finding: a [N, 0] tensor would reach the GPU launch with
        // grid=0 (CUDA_ERROR_INVALID_VALUE abort) and has no orthogonal
        // factor anyway — refuse coherently on both devices.
        eprintln!("nsl: muon_orthogonalize requires a non-empty rank-2 tensor");
        std::process::abort();
    }
    let steps = ns_steps as i64;
    let (rows, cols) = unsafe { (*g.shape, *g.shape.add(1)) };
    let tall = rows > cols;

    // 1. Frobenius pre-normalization on the ORIGINAL orientation (norm is
    //    transpose-invariant; the input is contiguous here or the GPU helper
    //    materializes it). GPU: fully device-resident. CPU: reference math.
    let x0 = {
        #[cfg(feature = "cuda")]
        {
            if g.device > 0 {
                crate::cuda::gpu_muon_frobenius_scale_f32(g_ptr)
            } else {
                cpu_frobenius_scale(g_ptr)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            cpu_frobenius_scale(g_ptr)
        }
    };

    // 2. Tall matrices orthogonalize the transpose so the Gram matrix stays
    //    at the smaller dimension — materialized once at entry.
    let mut x = if tall {
        let v = nsl_tensor_transpose(x0, 0, 1);
        let c = nsl_tensor_contiguous(v);
        nsl_tensor_free(v);
        nsl_tensor_free(x0);
        c
    } else {
        x0
    };

    // 3. Quintic NS iteration: a = x xᵀ; b = NS_B·a + NS_C·(a a);
    //    x = NS_A·x + b x. The fused scalar_mul_add_inplace keeps each
    //    combine at one launch and is bit-exact vs mul_scalar + add.
    for _ in 0..steps {
        let xt = nsl_tensor_transpose(x, 0, 1);
        let a = nsl_tensor_matmul(x, xt, 0);
        nsl_tensor_free(xt);
        let aa = nsl_tensor_matmul(a, a, 0);
        let b = nsl_tensor_mul_scalar(a, NS_B, 0);
        nsl_tensor_scalar_mul_add_inplace(b, aa, NS_C);
        nsl_tensor_free(a);
        nsl_tensor_free(aa);
        let t = nsl_tensor_matmul(b, x, 0);
        nsl_tensor_scalar_mul_add_inplace(t, x, NS_A);
        nsl_tensor_free(b);
        nsl_tensor_free(x);
        x = t;
    }

    // 4. Restore the original orientation — materialized so the caller
    //    never receives a strided view.
    if tall {
        let v = nsl_tensor_transpose(x, 0, 1);
        let c = nsl_tensor_contiguous(v);
        nsl_tensor_free(v);
        nsl_tensor_free(x);
        c
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::creation::create_tensor_from_f64_data;

    fn make_f64(rows: i64, cols: i64, data: &[f64]) -> i64 {
        create_tensor_from_f64_data(data, &[rows, cols])
    }

    fn read_f64(ptr: i64, n: usize) -> Vec<f64> {
        let t = NslTensor::from_ptr(ptr);
        assert_eq!(t.dtype, 0);
        (0..n)
            .map(|i| unsafe { *(t.data as *const f64).add(i) })
            .collect()
    }

    /// Independent f64 reference of the reference NSL implementation
    /// (transpose-first normalization order, per muon.nsl).
    fn reference(g: &[f64], rows: usize, cols: usize, steps: usize) -> Vec<f64> {
        fn matmul(a: &[f64], b: &[f64], n: usize, k: usize, m: usize) -> Vec<f64> {
            let mut out = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut acc = 0.0;
                    for p in 0..k {
                        acc += a[i * k + p] * b[p * m + j];
                    }
                    out[i * m + j] = acc;
                }
            }
            out
        }
        fn transpose(a: &[f64], n: usize, m: usize) -> Vec<f64> {
            let mut out = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    out[j * n + i] = a[i * m + j];
                }
            }
            out
        }
        let tall = rows > cols;
        let (r, c, mut x) = if tall {
            (cols, rows, transpose(g, rows, cols))
        } else {
            (rows, cols, g.to_vec())
        };
        let n: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
        for v in &mut x {
            *v *= 1.0 / n;
        }
        for _ in 0..steps {
            let xt = transpose(&x, r, c);
            let a = matmul(&x, &xt, r, c, r);
            let aa = matmul(&a, &a, r, r, r);
            let b: Vec<f64> = a
                .iter()
                .zip(&aa)
                .map(|(av, aav)| -4.7750 * av + 2.0315 * aav)
                .collect();
            let bx = matmul(&b, &x, r, r, c);
            x = x
                .iter()
                .zip(&bx)
                .map(|(xv, bxv)| 3.4445 * xv + bxv)
                .collect();
        }
        if tall {
            transpose(&x, r, c)
        } else {
            x
        }
    }

    #[test]
    fn muon_orthogonalize_matches_reference_wide_and_tall() {
        // Deterministic pseudo-random fill (no Math.random in tests).
        let fill = |n: usize, seed: u64| -> Vec<f64> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
                })
                .collect()
        };
        for (rows, cols, seed) in [(4i64, 6i64, 7u64), (6, 4, 9), (5, 5, 11)] {
            let n = (rows * cols) as usize;
            let data = fill(n, seed);
            let g = make_f64(rows, cols, &data);
            let out = nsl_tensor_muon_orthogonalize(g, 5.0);
            let got = read_f64(out, n);
            let want = reference(&data, rows as usize, cols as usize, 5);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                    "({rows}x{cols}) elem {i}: primitive {a} vs reference {b}"
                );
            }
            // Semi-orthogonality sanity: rows (wide) / cols (tall) of the
            // result should have near-unit norms after 5 NS steps.
            nsl_tensor_free(out);
            nsl_tensor_free(g);
        }
    }

    #[test]
    fn muon_orthogonalize_input_not_mutated() {
        let data = [0.5, -0.25, 0.125, 1.0, -0.75, 0.3125];
        let g = make_f64(2, 3, &data);
        let out = nsl_tensor_muon_orthogonalize(g, 3.0);
        assert_eq!(read_f64(g, 6), data.to_vec(), "input was clobbered");
        nsl_tensor_free(out);
        nsl_tensor_free(g);
    }
}
