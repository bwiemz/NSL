//! CUDA runtime: context management, kernel launch, module cache.
//! Only compiled when the `cuda` feature is enabled.

use std::ffi::c_void;

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
        let _ = state();
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
        state();
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
}
