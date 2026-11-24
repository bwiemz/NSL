//! Silicon gate for fresh gradient accumulators (`nsl_runtime::accum_fresh`).
//!
//! A fresh buffer holds stale bytes and is zero in value. Every case below
//! starts from a buffer full of nonzero stale values, marks it, and holds the
//! writer to the result the same writer gives on a zeroed, unmarked buffer,
//! bit for bit:
//!
//! 1. the fused weight-gradient GEMM, f32 operands (`sgemm_wgrad_accum`) and
//!    bf16 operands (`storage_gemm::wgrad_accum`), at a small shape and at one
//!    on the timed cublasLt plan: the first call overwrites (`beta = 0`), the
//!    second adds;
//! 2. the scaled add of a materialised gradient: zeroed, then added;
//! 3. `nsl_accum_materialize`: a buffer no writer took is zeroed, one a writer
//!    took is left alone;
//! 4. the multi-tensor AdamW step's keep variant: θ, m and v are the zeroing
//!    step's bits, and the accumulator is left as it was.
//!
//! (The end-to-end gate, `nsl-cli`'s `accum_overwrite_gate`, trains the
//! production line with and without the protocol; there a buffer no gradient
//! reaches has stale bytes of zero, so case 3 is what proves the zeroing.)
//!
//!   cargo test -p nsl-runtime --features cuda,test-hooks \
//!       --test accum_fresh_gpu -- --include-ignored --test-threads=1
#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use std::sync::Mutex;

use nsl_runtime::accum_fresh::{counts, nsl_accum_mark_fresh, nsl_accum_materialize};
use nsl_runtime::fase_step::{
    nsl_fase_fused_adamw_step_multi, nsl_fase_fused_adamw_step_multi_keep_mp,
};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::precision_cast::nsl_tensor_to_dtype;
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get_dtype,
    nsl_tensor_scalar_mul_add_inplace, nsl_tensor_to_device, nsl_tensor_wgrad_accum,
};

const DTYPE_F32: i64 = 1;
const DTYPE_BF16: i64 = 3;

/// The marks are process-global; one lock keeps this file's gates apart.
static SERIAL: Mutex<()> = Mutex::new(());

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

fn upload(shape: &[usize], vals: &[f32]) -> i64 {
    assert_eq!(
        shape.iter().product::<usize>(),
        vals.len(),
        "shape/data mismatch"
    );
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let s = nsl_list_new();
    for &d in shape {
        nsl_list_push(s, d as i64);
    }
    let cpu = nsl_tensor_from_static(leaked.as_ptr() as i64, s, DTYPE_F32);
    nsl_list_free(s);
    let dev = nsl_tensor_to_device(cpu, 1);
    nsl_tensor_free(cpu);
    dev
}

fn bits(t: i64, n: usize) -> Vec<u32> {
    let cpu = nsl_tensor_to_device(t, 0);
    assert_eq!(
        nsl_tensor_get_dtype(cpu),
        DTYPE_F32,
        "results come back f32"
    );
    let p = nsl_tensor_data_ptr(cpu);
    let out = unsafe { std::slice::from_raw_parts(p as *const f32, n) }
        .iter()
        .map(|v| v.to_bits())
        .collect();
    nsl_tensor_free(cpu);
    out
}

/// Values in [-1, 1) with a zero every 13th.
fn values(n: usize, salt: f64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i % 13 == 5 {
                0.0
            } else {
                (((i as f64) * 0.618_033_988_749_894_9 + salt).fract() * 2.0 - 1.0) as f32
            }
        })
        .collect()
}

/// A buffer of nonzero stale values: what the keep step leaves behind.
fn stale(shape: &[usize]) -> i64 {
    let n = shape.iter().product();
    upload(
        shape,
        &(0..n).map(|i| 7.0 + i as f32 * 1e-3).collect::<Vec<_>>(),
    )
}

fn zeros(shape: &[usize]) -> i64 {
    upload(shape, &vec![0.0; shape.iter().product()])
}

fn list(ts: &[i64]) -> i64 {
    let l = nsl_list_new();
    for &t in ts {
        nsl_list_push(l, t);
    }
    l
}

/// Case 1, both operand dtypes: the first write to a fresh buffer is the
/// write to a zeroed one, and the second adds exactly as it does there.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_fresh_buffer_s_first_weight_gradient_overwrites_and_the_next_adds() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !cuda_available() {
        return;
    }
    // A small shape (cuBLAS) and one past the timed cublasLt plan's threshold
    // (m·n·k >= 2^28), the route production GEMMs take: its plan is keyed by
    // the shape, so the beta-1 reference tunes it and the beta-0 write replays
    // it.
    for ((n, d_in, d_out), bf16) in [(192usize, 96usize, 80usize), (4096, 2048, 2048)]
        .into_iter()
        .flat_map(|s| [(s, false), (s, true)])
    {
        let x32 = upload(&[n, d_in], &values(n * d_in, 0.1));
        let g32 = upload(&[n, d_out], &values(n * d_out, 0.7));
        let (x, g) = if bf16 {
            (
                nsl_tensor_to_dtype(x32, DTYPE_BF16),
                nsl_tensor_to_dtype(g32, DTYPE_BF16),
            )
        } else {
            (x32, g32)
        };
        let reference = zeros(&[d_in, d_out]);
        let fresh = stale(&[d_in, d_out]);
        let l = list(&[fresh]);
        nsl_accum_mark_fresh(l);
        let (_, claimed0, _) = counts();
        for call in 0..2 {
            nsl_tensor_wgrad_accum(reference, x, g, 0.25);
            nsl_tensor_wgrad_accum(fresh, x, g, 0.25);
            assert_eq!(
                bits(fresh, d_in * d_out),
                bits(reference, d_in * d_out),
                "{n}x{d_in}x{d_out} bf16={bf16} call {call}: the fresh buffer is not the zeroed one's result"
            );
        }
        let (_, claimed1, _) = counts();
        assert_eq!(
            claimed1 - claimed0,
            1,
            "bf16={bf16}: one mark taken by the first GEMM"
        );
        nsl_list_free(l);
        for t in [reference, fresh] {
            nsl_tensor_free(t);
        }
        if bf16 {
            nsl_tensor_free(x);
            nsl_tensor_free(g);
        }
        nsl_tensor_free(x32);
        nsl_tensor_free(g32);
    }
}

/// Cases 2 and 3: the scaled add zeroes a fresh buffer before it adds; the
/// materialize zeroes a buffer no writer took and leaves a taken one alone.
#[test]
#[ignore = "requires CUDA GPU"]
fn an_adding_writer_and_the_read_see_zeros() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !cuda_available() {
        return;
    }
    let len = 1031usize;
    let g = upload(&[len], &values(len, 0.3));
    let reference = zeros(&[len]);
    let added = stale(&[len]);
    let untouched = stale(&[len]);
    let l = list(&[added, untouched]);
    nsl_accum_mark_fresh(l);
    nsl_tensor_scalar_mul_add_inplace(reference, g, 0.125);
    nsl_tensor_scalar_mul_add_inplace(added, g, 0.125);
    assert_eq!(
        bits(added, len),
        bits(reference, len),
        "the scaled add read stale bytes"
    );
    nsl_accum_materialize(l);
    assert_eq!(
        bits(untouched, len),
        vec![0u32; len],
        "the untouched buffer was not zeroed"
    );
    assert_eq!(
        bits(added, len),
        bits(reference, len),
        "a taken buffer was zeroed again"
    );
    nsl_list_free(l);
    for t in [g, reference, added, untouched] {
        nsl_tensor_free(t);
    }
}

/// Case 4: the keep variant of the multi-tensor AdamW step is the zeroing
/// step's arithmetic, and leaves the accumulators alone.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_keep_step_is_the_zeroing_step_but_for_the_accumulators() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !cuda_available() {
        return;
    }
    let lens = [257usize, 1000, 33];
    let make = |salt: f64| -> [Vec<i64>; 4] {
        let mut out: [Vec<i64>; 4] = Default::default();
        for (j, &n) in lens.iter().enumerate() {
            let s = salt + j as f64;
            out[0].push(upload(&[n], &values(n, s)));
            out[1].push(upload(
                &[n],
                &values(n, s + 0.11)
                    .iter()
                    .map(|v| v * 0.01)
                    .collect::<Vec<_>>(),
            ));
            out[2].push(upload(
                &[n],
                &values(n, s + 0.23)
                    .iter()
                    .map(|v| v * v * 1e-4)
                    .collect::<Vec<_>>(),
            ));
            out[3].push(upload(&[n], &values(n, s + 0.37)));
        }
        out
    };
    // The unclipped step (the kernel branches around the scale) and the clip's.
    for mp_scale in [1.0, 0.5] {
        let (zero, keep) = (make(0.0), make(0.0));
        let lists = |t: &[Vec<i64>; 4]| -> [i64; 4] {
            [list(&t[0]), list(&t[1]), list(&t[2]), list(&t[3])]
        };
        let (lz, lk) = (lists(&zero), lists(&keep));
        let before: Vec<Vec<u32>> = keep[3].iter().zip(lens).map(|(&t, n)| bits(t, n)).collect();
        nsl_fase_fused_adamw_step_multi(
            lz[0], lz[1], lz[2], lz[3], 1e-3, 0.9, 0.1, 0.95, 0.05, 1e-8, 0.01, 1.1, 1.2, 0, 0,
            mp_scale,
        );
        nsl_fase_fused_adamw_step_multi_keep_mp(
            lk[0], lk[1], lk[2], lk[3], 1e-3, 0.9, 0.1, 0.95, 0.05, 1e-8, 0.01, 1.1, 1.2, 0, 0,
            mp_scale,
        );
        for (j, &n) in lens.iter().enumerate() {
            for (k, what) in ["theta", "m", "v"].iter().enumerate() {
                assert_eq!(
                    bits(keep[k][j], n),
                    bits(zero[k][j], n),
                    "mp_scale={mp_scale} param {j}: {what} differs from the zeroing step's"
                );
            }
            assert_eq!(
                bits(zero[3][j], n),
                vec![0u32; n],
                "the zeroing step zeroes"
            );
            assert_eq!(
                bits(keep[3][j], n),
                before[j],
                "the keep step wrote the accumulator"
            );
        }
        for l in lz.into_iter().chain(lk) {
            nsl_list_free(l);
        }
        for t in zero.iter().chain(keep.iter()).flatten() {
            nsl_tensor_free(*t);
        }
    }
}
