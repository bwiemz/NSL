//! GPU regression for `nsl_kernel_launch_tensors` — the user-`kernel`-block
//! launch path used by `nsl run`.
//!
//! Codegen lowers each tensor argument to its NslTensor *handle* (a host
//! pointer to the struct). A prior version passed those handles straight to
//! `nsl_kernel_launch`, so `cuLaunchKernel` dereferenced host struct pointers
//! as device addresses and every GPU `kernel` launch died with
//! CUDA_ERROR_ILLEGAL_ADDRESS (surfacing at the next DtoH copy). This test
//! reproduces the m17 `vec_add` kernel at the FFI level: it fails on the old
//! marshaling and passes once the handle -> `.data` extraction is correct.
//!
//! Requires a GPU:
//!   cargo test -p nsl-runtime --features cuda --test kernel_launch_tensors_gpu -- --include-ignored

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get, nsl_tensor_to_device,
};

// `nsl_kernel_launch_tensors` is `#[no_mangle] pub extern "C"` in nsl-runtime,
// but the `cuda` module is `pub(crate)`, so link against the C symbol directly
// (same pattern the tier_b GPU tests use for `nsl_kernel_launch`).
extern "C" {
    fn nsl_kernel_launch_tensors(
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

const DTYPE_F32: i64 = 1;

/// Minimal `c[i] = a[i] + b[i]` kernel, ISA 7.0, ASCII only, no `mad.lo.u32`
/// (invalid in ISA 7.0). i = blockIdx.x * blockDim.x + threadIdx.x.
const VEC_ADD_PTX: &str = concat!(
    ".version 7.0\n",
    ".target sm_70\n",
    ".address_size 64\n",
    ".visible .entry vec_add(.param .u64 a, .param .u64 b, .param .u64 c) {\n",
    "    .reg .u32 %r<5>;\n",
    "    .reg .u64 %rd<12>;\n",
    "    .reg .f32 %f<4>;\n",
    "    ld.param.u64 %rd1, [a];\n",
    "    ld.param.u64 %rd2, [b];\n",
    "    ld.param.u64 %rd3, [c];\n",
    "    mov.u32 %r1, %ctaid.x;\n",
    "    mov.u32 %r2, %ntid.x;\n",
    "    mul.lo.u32 %r3, %r1, %r2;\n",
    "    mov.u32 %r4, %tid.x;\n",
    "    add.u32 %r3, %r3, %r4;\n",
    "    cvt.u64.u32 %rd4, %r3;\n",
    "    mul.lo.u64 %rd5, %rd4, 4;\n",
    "    add.u64 %rd6, %rd1, %rd5;\n",
    "    add.u64 %rd7, %rd2, %rd5;\n",
    "    add.u64 %rd8, %rd3, %rd5;\n",
    "    ld.global.f32 %f1, [%rd6];\n",
    "    ld.global.f32 %f2, [%rd7];\n",
    "    add.f32 %f3, %f1, %f2;\n",
    "    st.global.f32 [%rd8], %f3;\n",
    "    ret;\n",
    "}\n",
    "\0"
);

const KERNEL_NAME: &[u8] = b"vec_add\0";

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

fn f32_tensor(vals: &[f32]) -> i64 {
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    nsl_list_push(shape, vals.len() as i64);
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

fn get1(t: i64, i: i64) -> f64 {
    let idx = nsl_list_new();
    nsl_list_push(idx, i);
    let v = nsl_tensor_get(t, idx);
    nsl_list_free(idx);
    v
}

#[test]
#[ignore]
fn kernel_launch_tensors_vec_add_writes_correct_output() {
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    const N: usize = 1024;
    let a = f32_tensor(&[1.0_f32; N]);
    let b = f32_tensor(&[2.0_f32; N]);
    let c = f32_tensor(&[0.0_f32; N]);

    let ga = nsl_tensor_to_device(a, 1);
    let gb = nsl_tensor_to_device(b, 1);
    let gc = nsl_tensor_to_device(c, 1);
    assert!(ga != 0 && gb != 0 && gc != 0, "to_device must produce GPU tensors");

    // args array = NslTensor handles, exactly as codegen builds it.
    let handles: [i64; 3] = [ga, gb, gc];
    let rc = unsafe {
        nsl_kernel_launch_tensors(
            VEC_ADD_PTX.as_ptr() as i64,
            KERNEL_NAME.as_ptr() as i64,
            4, 1, 1,   // grid
            256, 1, 1, // block  => 4*256 = 1024 threads
            handles.as_ptr() as i64,
            handles.len() as i64,
            0,
        )
    };
    assert_eq!(rc, 0, "kernel launch must succeed (rc={rc})");

    let gc_back = nsl_tensor_to_device(gc, 0);
    for i in 0..N {
        let got = get1(gc_back, i as i64);
        assert!(
            (got - 3.0).abs() <= 1e-5,
            "elem {i}: expected 3.0, got {got} (kernel arg marshaling wrong?)"
        );
    }

    nsl_tensor_free(gc_back);
    nsl_tensor_free(gc);
    nsl_tensor_free(gb);
    nsl_tensor_free(ga);
    nsl_tensor_free(c);
    nsl_tensor_free(b);
    nsl_tensor_free(a);
}
