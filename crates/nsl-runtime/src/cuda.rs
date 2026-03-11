//! CUDA runtime: context management, kernel launch, module cache.
//! Only compiled when the `cuda` feature is enabled.

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

    pub(crate) fn ensure_init() -> &'static Mutex<CudaState> {
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
                let result = cuCtxCreate_v2(&mut context, 0, device);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuCtxCreate failed: {:?}",
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
        ensure_init();
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
        let state = ensure_init();
        let _guard = state.lock().unwrap();
        unsafe {
            let result = cuMemPrefetchAsync(
                ptr as CUdeviceptr,
                size_bytes,
                device_id,
                std::ptr::null_mut(), // default stream
            );
            if result != CUresult::CUDA_SUCCESS
                && result != CUresult::CUDA_ERROR_NOT_SUPPORTED
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
        let state = ensure_init();
        let mut guard = state.lock().unwrap();

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

// === FFI exports ===

/// Initialize the CUDA runtime (device 0, primary context).
/// Returns 0 on success. Aborts if CUDA feature is not compiled.
#[no_mangle]
pub extern "C" fn nsl_cuda_init() -> i64 {
    #[cfg(feature = "cuda")]
    {
        inner::ensure_init();
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
