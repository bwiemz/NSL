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

    #[cfg(test)]
    use std::collections::HashMap as TestHashMap;
    #[cfg(test)]
    static CUDA_SIZE_REGISTRY: std::sync::OnceLock<std::sync::Mutex<TestHashMap<usize, usize>>> = std::sync::OnceLock::new();

    #[cfg(test)]
    fn cuda_size_registry() -> &'static std::sync::Mutex<TestHashMap<usize, usize>> {
        CUDA_SIZE_REGISTRY.get_or_init(|| std::sync::Mutex::new(TestHashMap::new()))
    }

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
            #[cfg(test)]
            crate::memory::stats::cuda_alloc(size_bytes);
            #[cfg(test)]
            cuda_size_registry().lock().unwrap().insert(ptr as usize, size_bytes);
            ptr as *mut c_void
        }
    }

    /// Free unified memory.
    pub(crate) fn free_managed(ptr: *mut c_void) {
        ensure_context();
        #[cfg(test)]
        {
            let size = cuda_size_registry().lock().unwrap().remove(&(ptr as usize)).unwrap_or(0);
            crate::memory::stats::cuda_free(size);
        }
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

    /// Allocate device-only memory (not accessible from host without explicit copy).
    pub(crate) fn alloc_device(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAlloc_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
            ptr as *mut c_void
        }
    }

    /// Free device-only memory allocated with `alloc_device`.
    pub(crate) fn free_device(ptr: *mut c_void) {
        ensure_context();
        unsafe {
            let result = cuMemFree_v2(ptr as CUdeviceptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFree_v2 (device) failed: {:?}",
                result
            );
        }
    }

    /// Allocate pinned (page-locked) host memory for fast DMA transfers.
    pub(crate) fn alloc_pinned(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = cuMemAllocHost_v2(&mut ptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAllocHost_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
            ptr
        }
    }

    /// Free pinned host memory allocated with `alloc_pinned`.
    pub(crate) fn free_pinned(ptr: *mut c_void) {
        ensure_context();
        unsafe {
            let result = cuMemFreeHost(ptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFreeHost failed: {:?}",
                result
            );
        }
    }

    /// Copy `size_bytes` bytes from host memory to device memory.
    pub(crate) fn memcpy_htod(dst_device: *mut c_void, src_host: *const c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemcpyHtoD_v2(dst_device as CUdeviceptr, src_host, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyHtoD_v2({} bytes) failed: {:?}",
                size_bytes,
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

    // === cuEvent wrappers for kernel profiler ===

    pub unsafe fn cu_event_create(event: *mut u64) {
        cuEventCreate(event as *mut CUevent, 0); // CU_EVENT_DEFAULT = 0
    }

    pub unsafe fn cu_event_record(event: u64, stream: *mut std::ffi::c_void) {
        cuEventRecord(event as CUevent, stream as CUstream);
    }

    pub unsafe fn cu_event_elapsed_time(ms: *mut f32, start: u64, stop: u64) {
        cuEventElapsedTime_v2(ms, start as CUevent, stop as CUevent);
    }

    pub unsafe fn cu_event_destroy(event: u64) {
        cuEventDestroy_v2(event as CUevent);
    }

    pub unsafe fn cu_ctx_synchronize() {
        cuCtxSynchronize();
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
        let func = {
            let mut guard = state.lock().unwrap();
            unsafe { cuCtxSetCurrent(guard.context); }

            // Cache modules by PTX pointer address (stable .rodata addresses)
            let cache_key = ptx_ptr as usize;
            let module = if let Some(m) = guard.module_cache.get(&cache_key) {
                *m
            } else {
                let mut module: CUmodule = std::ptr::null_mut();
                let res = unsafe { cuModuleLoadData(&mut module, ptx_ptr as *const c_void) };
                if res != CUresult::CUDA_SUCCESS { return res; }
                guard.module_cache.insert(cache_key, module);
                module
            };

            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
            let mut func: CUfunction = std::ptr::null_mut();
            let res = unsafe { cuModuleGetFunction(&mut func, module, name.as_ptr()) };
            if res != CUresult::CUDA_SUCCESS { return res; }

            func
        }; // guard dropped here — no lock held for CUDA calls

        // Profiler: pop event pair (lock-pop-unlock on profiler mutex)
        let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
            crate::kernel_profiler::kernel_profiler_pop_events()
        } else {
            None
        };

        // Record start event before launch
        if let Some((start, _, _)) = &profiler_events {
            unsafe { cuEventRecord(*start as CUevent, std::ptr::null_mut()); }
        }

        // Launch kernel (no lock held)
        let mut kernel_args: Vec<*mut c_void> = args.to_vec();
        let res = unsafe {
            cuLaunchKernel(
                func,
                grid[0] as u32, grid[1] as u32, grid[2] as u32,
                block[0] as u32, block[1] as u32, block[2] as u32,
                0, std::ptr::null_mut(),
                kernel_args.as_mut_ptr(), std::ptr::null_mut(),
            )
        };

        // Record stop event after launch
        if let Some((_, stop, _)) = &profiler_events {
            unsafe { cuEventRecord(*stop as CUevent, std::ptr::null_mut()); }
            // Push trace (lock-push-unlock on profiler mutex)
            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
            let name_str = name.to_str().unwrap_or("unknown");
            crate::kernel_profiler::kernel_profiler_push_trace(
                name_str,
                [grid[0] as u32, grid[1] as u32, grid[2] as u32],
                [block[0] as u32, block[1] as u32, block[2] as u32],
            );
        }

        // NOTE: cuCtxSynchronize removed — unified memory provides coherence,
        // profiler flush performs single sync at program end
        res
    }
}

#[cfg(feature = "cuda")]
pub(crate) use inner::{cu_event_create, cu_event_record, cu_event_elapsed_time, cu_event_destroy, cu_ctx_synchronize};

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
 owns_data: 1,
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
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
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
 owns_data: 1,
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
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

/// GPU matrix multiplication: C[M,N] = A[M,K] @ B[K,N], f32 inputs.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_matmul_f32(a_ptr: i64, b_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };

    assert!(a.ndim >= 2 && b.ndim >= 2, "matmul requires 2D+ tensors");

    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let b_shape = unsafe { std::slice::from_raw_parts(b.shape, b.ndim as usize) };

    let m = a_shape[a.ndim as usize - 2] as u64;
    let k = a_shape[a.ndim as usize - 1] as u64;
    let k2 = b_shape[b.ndim as usize - 2] as u64;
    let n = b_shape[b.ndim as usize - 1] as u64;
    assert_eq!(k, k2, "matmul inner dimension mismatch: {} vs {}", k, k2);

    let out_len = (m * n) as usize;
    let out_data = inner::alloc_managed(out_len * 4); // f32 = 4 bytes

    // Output shape [M, N]
    let shape_layout = std::alloc::Layout::array::<i64>(2).unwrap();
    let shape = unsafe { std::alloc::alloc(shape_layout) as *mut i64 };
    unsafe {
        *shape.add(0) = m as i64;
        *shape.add(1) = n as i64;
    }
    let strides = NslTensor::compute_strides(shape, 2);

    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: 2, len: out_len as i64, refcount: 1,
        device: a.device, dtype: 1,
 owns_data: 1,
    });
    let out_ptr = Box::into_raw(out);

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_data as u64;
    let mut m_val = m;
    let mut n_val = n;
    let mut k_val = k;

    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut m_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut k_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 16i64; // 16x16 thread block
    let grid_x = ((n as i64) + block - 1) / block;
    let grid_y = ((m as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        kernels::MATMUL_F32_PTX.as_ptr(),
        "nsl_matmul_f32\0".as_ptr(),
        [grid_x, grid_y, 1],
        [block, block, 1],
        &args,
    );
    assert_eq!(result as u32, 0, "GPU matmul kernel failed: {}", result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

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
 owns_data: 1,
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
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

// === GPU backward op helpers ===

/// GPU backward binary op: takes grad tensor and a saved tensor, produces output of same shape as grad.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_backward_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU backward: length mismatch between grad and saved tensors");

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4); // f32 = 4 bytes
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: a.ndim, len: a.len, refcount: 1,
        device: a.device, dtype: 1,
        owns_data: 1,
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
    assert_eq!(result as u32, 0, "GPU backward kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    #[allow(unused_unsafe)]
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_relu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::RELU_BACKWARD_F32_PTX,
        "nsl_relu_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_sigmoid_backward(grad: i64, saved_out: i64) -> i64 {
    gpu_backward_binary(
        grad, saved_out,
        kernels::SIGMOID_BACKWARD_F32_PTX,
        "nsl_sigmoid_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_tanh_backward(grad: i64, saved_out: i64) -> i64 {
    gpu_backward_binary(
        grad, saved_out,
        kernels::TANH_BACKWARD_F32_PTX,
        "nsl_tanh_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_gelu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::GELU_BACKWARD_F32_PTX,
        "nsl_gelu_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_silu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::SILU_BACKWARD_F32_PTX,
        "nsl_silu_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_backward(grad: i64, input: i64, min_val: f32, max_val: f32) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(grad as *const NslTensor) };
    let b = unsafe { &*(input as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU clamp_backward: length mismatch");

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor {
        data: out_data, shape, strides,
        ndim: a.ndim, len: a.len, refcount: 1,
        device: a.device, dtype: 1,
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_t.data as u64;
    let mut min_arg = min_val;
    let mut max_arg = max_val;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut min_arg as *mut _ as *mut std::ffi::c_void,
        &mut max_arg as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::CLAMP_BACKWARD_F32_PTX.as_ptr(),
        "nsl_clamp_backward_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args,
    );
    assert_eq!(result as u32, 0, "GPU clamp_backward kernel failed: {}", result as u32);
    #[allow(unused_unsafe)]
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
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
    fn test_alloc_free_device() {
        // Allocate 1024 bytes of device-only memory and free it — no crash expected.
        let ptr = inner::alloc_device(1024);
        assert!(!ptr.is_null(), "alloc_device returned null");
        inner::free_device(ptr);
    }

    #[test]
    fn test_alloc_free_pinned() {
        // Allocate 256 bytes of pinned host memory, write/read CPU-side, then free.
        let ptr = inner::alloc_pinned(256);
        assert!(!ptr.is_null(), "alloc_pinned returned null");

        // Write and read back on the CPU side (pinned memory is host-accessible).
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, 256) };
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        for (i, &byte) in slice.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "pinned memory mismatch at byte {}", i);
        }

        inner::free_pinned(ptr);
    }

    #[test]
    fn test_memcpy_htod() {
        // Copy host data into device memory and free — verifies the copy doesn't crash.
        let host_data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let size_bytes = host_data.len() * std::mem::size_of::<f32>();

        let dev_ptr = inner::alloc_device(size_bytes);
        assert!(!dev_ptr.is_null(), "alloc_device returned null");

        inner::memcpy_htod(dev_ptr, host_data.as_ptr() as *const std::ffi::c_void, size_bytes);

        // Sync to ensure transfer is complete before freeing.
        unsafe {
            let sync = cudarc::driver::sys::cuCtxSynchronize();
            assert_eq!(sync as u32, 0, "cuCtxSynchronize after memcpy_htod failed");
        }

        inner::free_device(dev_ptr);
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
            owns_data: 1,
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
    fn test_gpu_matmul() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device, nsl_tensor_matmul};

        // A = [[1,2,3],[4,5,6]] (2x3)
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2i64, 3];
        let a_strides = vec![3i64, 1];
        let a = Box::new(NslTensor {
            data: a_data.as_ptr() as *mut std::ffi::c_void,
            shape: a_shape.as_ptr() as *mut i64,
            strides: a_strides.as_ptr() as *mut i64,
            ndim: 2, len: 6, refcount: 1, device: 0, dtype: 0,
 owns_data: 1,
        });
        std::mem::forget(a_data); std::mem::forget(a_shape); std::mem::forget(a_strides);
        let a_cpu = Box::into_raw(a) as i64;

        // B = [[7,8],[9,10],[11,12]] (3x2)
        let b_data = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_shape = vec![3i64, 2];
        let b_strides = vec![2i64, 1];
        let b = Box::new(NslTensor {
            data: b_data.as_ptr() as *mut std::ffi::c_void,
            shape: b_shape.as_ptr() as *mut i64,
            strides: b_strides.as_ptr() as *mut i64,
            ndim: 2, len: 6, refcount: 1, device: 0, dtype: 0,
 owns_data: 1,
        });
        std::mem::forget(b_data); std::mem::forget(b_shape); std::mem::forget(b_strides);
        let b_cpu = Box::into_raw(b) as i64;

        // Transfer to GPU
        let a_gpu = nsl_tensor_to_device(a_cpu, 1);
        let b_gpu = nsl_tensor_to_device(b_cpu, 1);

        // Matmul on GPU
        let c_gpu = nsl_tensor_matmul(a_gpu, b_gpu);

        // Sync and transfer back
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        let c_cpu = nsl_tensor_to_device(c_gpu, 0);
        let c = unsafe { &*(c_cpu as *const NslTensor) };

        // Expected: [[58, 64], [139, 154]]
        // 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        let expected = [58.0, 64.0, 139.0, 154.0];
        for i in 0..4 {
            let val = unsafe { *c.data_f64().add(i) };
            assert!((val - expected[i]).abs() < 0.5, "matmul mismatch at {}: {} vs {}", i, val, expected[i]);
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
 owns_data: 1,
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
 owns_data: 1,
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
