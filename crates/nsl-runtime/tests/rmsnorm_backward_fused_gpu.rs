//! Silicon gate for the one-pass RMSNorm backward
//! (`nsl_rmsnorm_backward_fused`, `nsl_kir::kernels::rmsnorm_bwd`): one launch
//! writes `dx` and the gamma gradient's per-chunk column sums, the dgamma final
//! pass adds the chunks.
//!
//! The kernel is two existing kernels' arithmetic on one read of the row, so
//! the gate is bit-identity with them, on the card, where ptxas schedules and
//! unrolls what the PTX spells:
//!
//! 1. `dx` against `nsl_rmsnorm_dx_backward` / `nsl_rmsnorm_dx_backward_add`
//!    (the dx kernels), bit for bit;
//! 2. `dgamma` against the three-kernel gamma gradient run at the SAME row
//!    chunking (its partial kernel takes any), bit for bit, at the production
//!    plan and at chunkings with ragged last chunks. (That path is itself held
//!    to an f64 reference within its derived bound by
//!    `rmsnorm_dgamma_backward_gpu.rs`.)
//! 3. the launch census: the production entry ran the fused kernel, so a
//!    fallback cannot pass vacuously.
//!
//! Shapes: tails against the 256-thread block and the column slots (one
//! element, a width under a warp, one past a block, the widest the slots
//! take), and the recipes' shapes (rows = batch × sequence).
//!
//!   cargo test -p nsl-runtime --features cuda,test-hooks \
//!       --test rmsnorm_backward_fused_gpu -- --include-ignored --test-threads=1
#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use std::sync::Mutex;

use nsl_kir::kernels::rmsnorm_bwd::{fused_plan, kernel_name, slots_for};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_rmsnorm_backward_fused, nsl_rmsnorm_dx_backward, nsl_rmsnorm_dx_backward_add,
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get_dtype,
    nsl_tensor_to_device,
};
use nsl_runtime::{
    test_kernel_launch_census_arm, test_kernel_launch_count, test_rmsnorm_backward_fused_chunked,
    test_rmsnorm_dgamma_chunked,
};

const DTYPE_F32: i64 = 1;
const EPS: f64 = 1e-5;

/// The census is process-global; one lock keeps this file's gates apart.
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
    assert_ne!(p, 0, "downloaded tensor has no data pointer");
    let out = unsafe { std::slice::from_raw_parts(p as *const f32, n) }
        .iter()
        .map(|v| v.to_bits())
        .collect();
    nsl_tensor_free(cpu);
    out
}

fn weyl(i: usize, salt: f64) -> f64 {
    ((i as f64) * 0.618_033_988_749_894_9 + salt).fract()
}

/// Activations in [-3, 3) scaled per row by 2^(row % 7 - 3); a zero every 29.
fn activations(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|k| {
            if k % 29 == 11 {
                return 0.0;
            }
            let scale = 2f64.powi((k / cols % 7) as i32 - 3);
            ((weyl(k, 0.1) * 6.0 - 3.0) * scale) as f32
        })
        .collect()
}

/// Gradients: log-uniform magnitude in [1e-9, 1e-6], either sign.
fn gradients(n: usize, salt: f64) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let mag = 10f64.powf(-9.0 + 3.0 * weyl(k, salt));
            let sign = if weyl(k, salt + 0.34) < 0.5 {
                -1.0
            } else {
                1.0
            };
            (sign * mag) as f32
        })
        .collect()
}

/// Gains near 1, as trained norms have them.
fn gains(cols: usize) -> Vec<f32> {
    (0..cols).map(|k| (0.5 + weyl(k, 0.23)) as f32).collect()
}

struct Inputs {
    dy: i64,
    x: i64,
    g: i64,
    res: i64,
}

impl Inputs {
    fn new(rows: usize, cols: usize) -> Inputs {
        let n = rows * cols;
        Inputs {
            dy: upload(&[rows, cols], &gradients(n, 0.37)),
            x: upload(&[rows, cols], &activations(rows, cols)),
            g: upload(&[cols], &gains(cols)),
            res: upload(&[rows, cols], &gradients(n, 0.81)),
        }
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        for t in [self.dy, self.x, self.g, self.res] {
            nsl_tensor_free(t);
        }
    }
}

/// `[rows, cols]`, with and without the residual: the production entry and
/// the chunkings in `chunks`, each against the separate ops bit for bit.
fn check(rows: usize, cols: usize, chunks: &[u64]) {
    let t = Inputs::new(rows, cols);
    let n = rows * cols;
    let slots = slots_for(cols as u64).expect("a width the slots take");
    for res in [0, t.res] {
        let add = res != 0;
        let dx_ref = if add {
            nsl_rmsnorm_dx_backward_add(t.dy, t.x, t.g, res, EPS)
        } else {
            nsl_rmsnorm_dx_backward(t.dy, t.x, t.g, EPS)
        };
        let want_dx = bits(dx_ref, n);
        nsl_tensor_free(dx_ref);

        // The production entry, with its plan.
        test_kernel_launch_census_arm();
        let list = nsl_rmsnorm_backward_fused(t.dy, t.x, t.g, res, EPS);
        assert_eq!(
            test_kernel_launch_count(&kernel_name(add, slots)),
            1,
            "{rows}x{cols} add={add}: fused launch"
        );
        let (dx, dg) = (nsl_list_get(list, 0), nsl_list_get(list, 1));
        nsl_list_free(list);
        assert!(
            bits(dx, n) == want_dx,
            "{rows}x{cols} add={add}: dx differs from the dx kernel's"
        );
        let plan = fused_plan(rows as u64);
        let dg_ref = test_rmsnorm_dgamma_chunked(t.dy, t.x, t.g, EPS as f32, plan.chunk_rows);
        assert_eq!(
            bits(dg, cols),
            bits(dg_ref, cols),
            "{rows}x{cols} add={add}: dgamma at the plan's chunking"
        );
        for v in [dx, dg, dg_ref] {
            nsl_tensor_free(v);
        }

        for &c in chunks {
            let (dx, dg) = test_rmsnorm_backward_fused_chunked(t.dy, t.x, t.g, res, EPS as f32, c);
            assert!(
                bits(dx, n) == want_dx,
                "{rows}x{cols} add={add} chunk {c}: dx differs"
            );
            let dg_ref = test_rmsnorm_dgamma_chunked(t.dy, t.x, t.g, EPS as f32, c);
            assert_eq!(
                bits(dg, cols),
                bits(dg_ref, cols),
                "{rows}x{cols} add={add} chunk {c}: dgamma differs"
            );
            for v in [dx, dg, dg_ref] {
                nsl_tensor_free(v);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_backward_is_both_kernels_bit_for_bit_on_tails() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    for (rows, cols) in [
        (1, 1),
        (3, 7),
        (37, 1),
        (5, 257),
        (130, 260),
        (19, 512),
        (9, 2049),
        (6, 4096),
    ] {
        check(rows, cols, &[1, 3, 8]);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_backward_is_both_kernels_bit_for_bit_at_the_recipe_shapes() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    for (rows, cols) in [(1024, 512), (2048, 1280), (4096, 2048)] {
        check(rows, cols, &[3, 16]);
    }
}

/// Diagnostic, not a gate: the fused kernel and the three-kernel path at the
/// recipe shapes, chunk sizes 2-32, with a 256 MiB write between launches so
/// each starts with cold L2. Read the per-kernel durations from an nsys trace
/// of this test (CUDA events around one launch would time the host's enqueue).
#[test]
#[ignore = "diagnostic: run under nsys"]
fn rmsnorm_backward_kernel_times() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    let flush = upload(&[64 << 20], &vec![0.0; 64 << 20]);
    let evict = || {
        let f = nsl_runtime::tensor::nsl_tensor_add(flush, flush, 0);
        nsl_tensor_free(f);
    };
    for (rows, cols) in [(2048, 1280), (4096, 2048)] {
        let t = Inputs::new(rows, cols);
        for _ in 0..7 {
            evict();
            let dx = nsl_rmsnorm_dx_backward_add(t.dy, t.x, t.g, t.res, EPS);
            evict();
            let dg = test_rmsnorm_dgamma_chunked(t.dy, t.x, t.g, EPS as f32, 128);
            for v in [dx, dg] {
                nsl_tensor_free(v);
            }
            for c in [2, 4, 8, 16, 32] {
                evict();
                let (dx, dg) =
                    test_rmsnorm_backward_fused_chunked(t.dy, t.x, t.g, t.res, EPS as f32, c);
                for v in [dx, dg] {
                    nsl_tensor_free(v);
                }
            }
        }
    }
    nsl_runtime::test_cuda_device_synchronize();
    nsl_tensor_free(flush);
}
