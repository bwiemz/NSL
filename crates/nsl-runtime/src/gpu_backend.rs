//! M47: Backend-agnostic GPU runtime trait.
//!
//! All GPU backends (CUDA, ROCm, Metal, WebGPU) implement this trait.
//! The compiled binary links against exactly one backend — no runtime dispatch.

use std::ffi::c_void;

/// Opaque device memory pointer.
pub type DevicePtr = *mut c_void;
/// Opaque handle to a loaded kernel module.
pub type ModuleHandle = u64;
/// Opaque handle to a GPU stream/command queue.
pub type StreamHandle = u64;

/// Errors from GPU operations.
#[derive(Debug)]
pub enum GpuError {
    OutOfMemory { requested: usize },
    InvalidPointer,
    KernelLaunchFailed { name: String, code: i32 },
    ModuleLoadFailed { reason: String },
    Unsupported { feature: String },
    DriverError { code: i32, message: String },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::OutOfMemory { requested } => {
                write!(f, "GPU out of memory ({requested} bytes requested)")
            }
            GpuError::InvalidPointer => write!(f, "invalid device pointer"),
            GpuError::KernelLaunchFailed { name, code } => {
                write!(f, "kernel '{name}' launch failed (error {code})")
            }
            GpuError::ModuleLoadFailed { reason } => {
                write!(f, "module load failed: {reason}")
            }
            GpuError::Unsupported { feature } => {
                write!(f, "unsupported feature: {feature}")
            }
            GpuError::DriverError { code, message } => {
                write!(f, "GPU driver error {code}: {message}")
            }
        }
    }
}

/// Kernel launch argument.
#[derive(Debug, Clone)]
pub enum KernelArg {
    Ptr(DevicePtr),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
}

/// Device capability information.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub name: String,
    pub max_threads_per_block: u32,
    pub max_shared_memory: u32,
    pub warp_size: u32,
    pub has_tensor_cores: bool,
    pub has_f16_compute: bool,
    pub has_bf16_compute: bool,
    pub has_atomic_float: bool,
    pub total_memory_bytes: u64,
}

/// Backend-agnostic GPU runtime interface.
///
/// Each backend (CUDA, ROCm, Metal, WebGPU) provides an implementation.
/// The compiled binary links against exactly one backend at build time.
pub trait GpuBackend: Send + Sync {
    /// Allocate device memory. Returns a device pointer.
    fn alloc(&self, bytes: usize) -> Result<DevicePtr, GpuError>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> Result<(), GpuError>;

    /// Copy bytes from host to device.
    fn copy_h2d(&self, host: *const u8, device: DevicePtr, bytes: usize) -> Result<(), GpuError>;

    /// Copy bytes from device to host.
    fn copy_d2h(&self, device: DevicePtr, host: *mut u8, bytes: usize) -> Result<(), GpuError>;

    /// Copy bytes from device to device (no host round-trip).
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<(), GpuError>;

    /// Load a compiled kernel module (PTX, AMDGPU, Metal library, WGSL).
    fn load_module(&self, code: &[u8]) -> Result<ModuleHandle, GpuError>;

    /// Launch a kernel by name from a loaded module.
    fn launch_kernel(
        &self,
        module: ModuleHandle,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        args: &[KernelArg],
    ) -> Result<(), GpuError>;

    /// Wait for all outstanding operations to complete.
    fn synchronize(&self) -> Result<(), GpuError>;

    /// Query device capabilities.
    fn capabilities(&self) -> DeviceCapabilities;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_error_display() {
        let e = GpuError::OutOfMemory { requested: 1024 };
        assert!(format!("{e}").contains("1024"));

        let e = GpuError::KernelLaunchFailed {
            name: "add".into(),
            code: -1,
        };
        assert!(format!("{e}").contains("add"));
    }

    #[test]
    fn kernel_arg_variants() {
        let args = vec![
            KernelArg::Ptr(std::ptr::null_mut()),
            KernelArg::U32(256),
            KernelArg::F32(1.0),
        ];
        assert_eq!(args.len(), 3);
    }
}
