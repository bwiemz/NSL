//! CUDA runtime: context management, kernel launch, module cache.
//! Only compiled when the `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use std::ffi::c_void;

pub(crate) mod kernels;

#[cfg(feature = "cuda")]
pub(crate) mod inner {
    use cudarc::driver::sys::*;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};

    struct CudaState {
        device: CUdevice,
        #[allow(dead_code)]
        context: CUcontext,
        module_cache: HashMap<usize, CUmodule>,
    }

    // SAFETY: CUcontext/CUmodule are opaque pointers managed by the CUDA driver.
    // We only access CudaState through the Mutex, ensuring single-threaded access.
    unsafe impl Send for CudaState {}

    static CUDA_STATE: OnceLock<Mutex<CudaState>> = OnceLock::new();

    /// Ensure CUDA is initialized. Called from FFI exports.
    pub(crate) fn init() {
        ensure_context();
    }

    /// Ensure the CUDA context is current on the calling thread.
    /// Must be called before any CUDA driver API call.
    fn ensure_context() {
        let s = state();
        let guard = s.lock().unwrap();
        unsafe {
            cuCtxSetCurrent(guard.context);
        }
    }

    fn state() -> &'static Mutex<CudaState> {
        CUDA_STATE.get_or_init(|| {
            unsafe {
                let result = cuInit(0);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuInit failed: {:?}",
                    result
                );
                let mut device: CUdevice = 0;
                let result = cuDeviceGet(&mut device, 0);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuDeviceGet failed: {:?}",
                    result
                );
                let mut context: CUcontext = std::ptr::null_mut();
                let result = cuDevicePrimaryCtxRetain(&mut context, device);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuDevicePrimaryCtxRetain failed: {:?}",
                    result
                );
                let result = cuCtxSetCurrent(context);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuCtxSetCurrent failed: {:?}",
                    result
                );
                Mutex::new(CudaState {
                    device,
                    context,
                    module_cache: HashMap::new(),
                })
            }
        })
    }

    /// Allocate unified memory (accessible from both CPU and GPU).
    pub(crate) fn alloc_managed(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAllocManaged(
                &mut ptr,
                size_bytes,
                CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
            );
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAllocManaged({} bytes) failed: {:?}",
                size_bytes,
                result
            );
            ptr as *mut c_void
        }
    }

    /// Free unified memory.
    pub(crate) fn free_managed(ptr: *mut c_void) {
        unsafe {
            let result = cuMemFree_v2(ptr as CUdeviceptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFree failed: {:?}",
                result
            );
        }
    }

    /// Prefetch memory to device. Best-effort: silently ignores NOT_SUPPORTED.
    pub(crate) fn prefetch_to_device(ptr: *mut c_void, size_bytes: usize, device_id: i32) {
        let state = state();
        let _guard = state.lock().unwrap();
        unsafe {
            let location = CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_id,
            };
            let result = cuMemPrefetchAsync_v2(
                ptr as CUdeviceptr,
                size_bytes,
                location,
                0, // flags
                std::ptr::null_mut(), // default stream
            );
            if result != CUresult::CUDA_SUCCESS
                && result != CUresult::CUDA_ERROR_NOT_SUPPORTED
                && result != CUresult::CUDA_ERROR_INVALID_DEVICE
            {
                panic!("cuMemPrefetchAsync failed: {:?}", result);
            }
        }
    }

    /// Launch a PTX kernel. `ptx_ptr` and `name_ptr` must point to null-terminated C strings.
    /// `args` is a slice of pointers to argument values (as required by `cuLaunchKernel`).
    pub(crate) fn kernel_launch(
        ptx_ptr: *const u8,
        name_ptr: *const u8,
        grid: [i64; 3],
        block: [i64; 3],
        args: &[*mut c_void],
    ) -> CUresult {
        let state = state();
        let mut guard = state.lock().unwrap();
        unsafe { cuCtxSetCurrent(guard.context); }

        // Cache modules by PTX pointer address (stable .rodata addresses)
        let cache_key = ptx_ptr as usize;
        let module = if let Some(m) = guard.module_cache.get(&cache_key) {
            *m
        } else {
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                let result = cuModuleLoadData(&mut module, ptx_ptr as *const c_void);
                if result != CUresult::CUDA_SUCCESS {
                    return result;
                }
            }
            guard.module_cache.insert(cache_key, module);
            module
        };

        let mut func: CUfunction = std::ptr::null_mut();
        unsafe {
            let result = cuModuleGetFunction(&mut func, module, name_ptr as *const i8);
            if result != CUresult::CUDA_SUCCESS {
                return result;
            }
        }

        // cuLaunchKernel expects an array of pointers to argument values.
        // The caller passes exactly that, so we just need a mutable copy of the slice.
        let mut kernel_args: Vec<*mut c_void> = args.to_vec();

        unsafe {
            cuLaunchKernel(
                func,
                grid[0] as u32,
                grid[1] as u32,
                grid[2] as u32,
                block[0] as u32,
                block[1] as u32,
                block[2] as u32,
                0,                          // shared memory bytes
                std::ptr::null_mut(),       // default stream
                kernel_args.as_mut_ptr(),
                std::ptr::null_mut(),       // no extra
            )
        }
    }
}

// === GPU op helpers ===

/// GPU elementwise binary op.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU elementwise: length mismatch");

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4); // f32 = 4 bytes
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: a.ndim, len: a.len, refcount: 1,
        device: a.device, dtype: 1,
    });
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args,
    );
    assert_eq!(result as u32, 0, "GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    out_ptr as i64
}

/// GPU elementwise unary op.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_unary(a_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: a.ndim, len: a.len, refcount: 1,
        device: a.device, dtype: 1,
    });
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args,
    );
    assert_eq!(result as u32, 0, "GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    out_ptr as i64
}

/// GPU scalar op (tensor op scalar).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_op(a_ptr: i64, scalar: f32, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: a.ndim, len: a.len, refcount: 1,
        device: a.device, dtype: 1,
    });
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut s_val = scalar;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut s_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args,
    );
    assert_eq!(result as u32, 0, "GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    out_ptr as i64
}

// === FFI exports ===

/// Initialize the CUDA runtime (device 0, primary context).
/// Returns 0 on success. Aborts if CUDA feature is not compiled.
#[no_mangle]
pub extern "C" fn nsl_cuda_init() -> i64 {
    #[cfg(feature = "cuda")]
    {
        inner::init();
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA support not compiled. Rebuild with --features cuda");
        std::process::abort();
    }
}

/// Launch a PTX kernel. All params are i64 for Cranelift ABI compatibility.
///
/// - `ptx_ptr`: pointer to null-terminated PTX source string
/// - `name_ptr`: pointer to null-terminated kernel function name
/// - `grid_x/y/z`: grid dimensions
/// - `block_x/y/z`: block dimensions
/// - `args_ptr`: pointer to array of `*mut c_void` (pointers to argument values)
/// - `num_args`: number of arguments
///
/// Returns 0 (CUDA_SUCCESS) on success, non-zero CUDA error code on failure.
#[no_mangle]
pub extern "C" fn nsl_kernel_launch(
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
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let args_slice = unsafe {
            std::slice::from_raw_parts(args_ptr as *const *mut c_void, num_args as usize)
        };
        let result = inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            args_slice,
        );
        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, name_ptr, grid_x, grid_y, grid_z);
        let _ = (block_x, block_y, block_z, args_ptr, num_args);
        eprintln!("CUDA support not compiled. Rebuild with --features cuda");
        std::process::abort();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    const VEC_ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry vec_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u64 n
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %fs<4>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u64 %rd4, [n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    mov.u32 %r1, %tid.x;
    add.u32 %r3, %r3, %r1;
    cvt.u64.u32 %rd5, %r3;
    setp.ge.u64 %p1, %rd5, %rd4;
    @%p1 bra DONE;

    shl.b64 %rd6, %rd5, 2;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %fs1, [%rd7];
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %fs2, [%rd7];
    add.f32 %fs3, %fs1, %fs2;
    add.u64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %fs3;

DONE:
    ret;
}\0";

    #[test]
    fn test_vec_add_kernel_launch() {
        let n: usize = 1024;
        let size_bytes = n * std::mem::size_of::<f32>();

        // Allocate unified memory
        let a = inner::alloc_managed(size_bytes);
        let b = inner::alloc_managed(size_bytes);
        let c = inner::alloc_managed(size_bytes);

        // Fill a and b on CPU (unified memory allows this)
        let a_slice = unsafe { std::slice::from_raw_parts_mut(a as *mut f32, n) };
        let b_slice = unsafe { std::slice::from_raw_parts_mut(b as *mut f32, n) };
        for i in 0..n {
            a_slice[i] = i as f32;
            b_slice[i] = (i * 2) as f32;
        }

        // Launch kernel
        let n_val = n as u64;
        let mut a_arg = a as u64;
        let mut b_arg = b as u64;
        let mut c_arg = c as u64;
        let mut n_arg = n_val;

        let args: [*mut std::ffi::c_void; 4] = [
            &mut a_arg as *mut _ as *mut std::ffi::c_void,
            &mut b_arg as *mut _ as *mut std::ffi::c_void,
            &mut c_arg as *mut _ as *mut std::ffi::c_void,
            &mut n_arg as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256i64;
        let grid_size = ((n as i64) + block_size - 1) / block_size;

        let result = inner::kernel_launch(
            VEC_ADD_PTX.as_ptr(),
            "vec_add\0".as_ptr(),
            [grid_size, 1, 1],
            [block_size, 1, 1],
            &args.map(|p| p),
        );
        assert_eq!(result as u32, 0, "kernel launch failed");

        // Synchronize to ensure kernel completed
        unsafe {
            let sync = cudarc::driver::sys::cuCtxSynchronize();
            assert_eq!(sync as u32, 0, "sync failed");
        }

        // Verify results on CPU (unified memory)
        let c_slice = unsafe { std::slice::from_raw_parts(c as *const f32, n) };
        for i in 0..n {
            let expected = (i + i * 2) as f32;
            assert_eq!(c_slice[i], expected, "mismatch at index {}", i);
        }

        // Cleanup
        inner::free_managed(a);
        inner::free_managed(b);
        inner::free_managed(c);
    }

    #[test]
    fn test_tensor_to_device_roundtrip() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device};

        // Create a CPU tensor manually: [1.0, 2.0, 3.0, 4.0]
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let shape = vec![4i64];
        let strides = vec![1i64];
        let t = Box::new(NslTensor {
            data: data.as_ptr() as *mut std::ffi::c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: 1,
            device: 0,
            dtype: 0,
        });
        // Leak the vecs so the tensor can use them
        std::mem::forget(data);
        std::mem::forget(shape);
        std::mem::forget(strides);
        let cpu_tensor = Box::into_raw(t) as i64;

        // Transfer CPU → GPU
        let gpu_tensor = nsl_tensor_to_device(cpu_tensor, 1);
        let gpu_t = unsafe { &*(gpu_tensor as *const NslTensor) };
        assert_eq!(gpu_t.device, 1);
        assert_eq!(gpu_t.dtype, 1); // f32

        // Transfer GPU → CPU
        let cpu_back = nsl_tensor_to_device(gpu_tensor, 0);
        let cpu_t = unsafe { &*(cpu_back as *const NslTensor) };
        assert_eq!(cpu_t.device, 0);
        assert_eq!(cpu_t.dtype, 0); // f64

        // Verify values survived the roundtrip (f64 → f32 → f64)
        for i in 0..4 {
            let val = unsafe { *cpu_t.data_f64().add(i) };
            let expected = (i + 1) as f64;
            assert!((val - expected).abs() < 1e-6, "mismatch at {}: {} vs {}", i, val, expected);
        }
    }

    #[test]
    fn test_gpu_elementwise_add() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device, nsl_tensor_add};

        // Create CPU tensors manually
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let a = Box::new(NslTensor {
            data: a_data.as_ptr() as *mut std::ffi::c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1, len: 4, refcount: 1, device: 0, dtype: 0,
        });
        std::mem::forget(a_data); std::mem::forget(shape.clone()); std::mem::forget(strides.clone());
        let a_cpu = Box::into_raw(a) as i64;

        let shape2 = vec![4i64];
        let strides2 = vec![1i64];
        let b = Box::new(NslTensor {
            data: b_data.as_ptr() as *mut std::ffi::c_void,
            shape: shape2.as_ptr() as *mut i64,
            strides: strides2.as_ptr() as *mut i64,
            ndim: 1, len: 4, refcount: 1, device: 0, dtype: 0,
        });
        std::mem::forget(b_data); std::mem::forget(shape2); std::mem::forget(strides2);
        let b_cpu = Box::into_raw(b) as i64;

        // Transfer to GPU
        let a_gpu = nsl_tensor_to_device(a_cpu, 1);
        let b_gpu = nsl_tensor_to_device(b_cpu, 1);

        // Add on GPU
        let c_gpu = nsl_tensor_add(a_gpu, b_gpu);

        // Sync and transfer back
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        let c_cpu = nsl_tensor_to_device(c_gpu, 0);
        let c = unsafe { &*(c_cpu as *const NslTensor) };

        let expected = [11.0, 22.0, 33.0, 44.0];
        for i in 0..4 {
            let val = unsafe { *c.data_f64().add(i) };
            assert!((val - expected[i]).abs() < 0.1, "mismatch at {}: {} vs {}", i, val, expected[i]);
        }
    }
}
