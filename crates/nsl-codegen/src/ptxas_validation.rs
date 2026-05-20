//! PTX validation helper for unit tests.
//!
//! Uses `cudarc::driver::sys::cuModuleLoadData` when a CUDA device is
//! available (i.e., the `cuda` feature is enabled AND a context is current);
//! falls back to shelling out to `nvcc --cubin` otherwise.
//!
//! Returns `Ok(())` when PTX is accepted, `Err(String)` with details
//! when ptxas rejects it.

/// Validate that the given PTX string compiles.
///
/// Returns `Ok(())` on success, `Err(msg)` when the validator rejects the PTX.
/// When no validator is available (no `cuda` feature + no `nvcc` on PATH),
/// the error message will contain `"nvcc not available"`.
pub fn validate_ptx(ptx: &str) -> Result<(), String> {
    // Prefer cudarc when feature is enabled and a CUDA context is current.
    #[cfg(feature = "cuda")]
    {
        if let Some(res) = try_validate_via_cudarc(ptx) {
            return res;
        }
    }
    // Fallback: nvcc --cubin (compile-only, no device required at test time).
    validate_via_nvcc(ptx)
}

/// Try validation via `cuModuleLoadData`.
///
/// Returns `None` when no CUDA context is current (so the caller can fall
/// through to the nvcc path), or `Some(result)` when a context was found
/// and the load attempt completed (success or ptxas error).
#[cfg(feature = "cuda")]
fn try_validate_via_cudarc(ptx: &str) -> Option<Result<(), String>> {
    use cudarc::driver::sys;
    use std::ffi::CString;
    unsafe {
        // cuCtxGetCurrent returns CUDA_SUCCESS and sets *pctx = NULL when no
        // context is bound on this thread.  Skip to nvcc fallback in that case.
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        let query_rc = sys::cuCtxGetCurrent(&mut ctx);
        if query_rc != sys::CUresult::CUDA_SUCCESS || ctx.is_null() {
            return None;
        }

        let ptx_c = match CString::new(ptx) {
            Ok(c) => c,
            // Embedded NUL — treat as invalid PTX.
            Err(_) => return Some(Err("PTX contains embedded NUL byte".to_string())),
        };

        let mut module: sys::CUmodule = std::ptr::null_mut();
        let rc = sys::cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const std::os::raw::c_void);
        if rc == sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            Some(Ok(()))
        } else {
            let mut name_ptr: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(rc, &mut name_ptr);
            let msg = if !name_ptr.is_null() {
                std::ffi::CStr::from_ptr(name_ptr)
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("cudarc rc={:?}", rc)
            };
            Some(Err(format!("cuModuleLoadData rejected PTX: {}", msg)))
        }
    }
}

/// Validate PTX by invoking `ptxas` directly on a temporary file.
///
/// `ptxas` is the PTX assembler shipped with the CUDA toolkit.  Unlike
/// `nvcc --cubin`, it does not require a C compiler (e.g., `cl.exe` on
/// Windows) in PATH — only the `.ptx` source.  When `ptxas` is not on
/// PATH the function returns `Err("nvcc not available: ...")` (the legacy
/// sentinel string Task B2 tests match) so callers can distinguish
/// "validator absent" from "PTX invalid".
fn validate_via_nvcc(ptx: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Infer architecture from the `.target` directive.
    let arch = if ptx.contains(".target sm_90") {
        "sm_90"
    } else if ptx.contains(".target sm_86") {
        "sm_86"
    } else if ptx.contains(".target sm_80") {
        "sm_80"
    } else if ptx.contains(".target sm_75") {
        "sm_75"
    } else {
        // Default: Ampere — covers the RTX 5070 Ti development target.
        "sm_80"
    };

    // Write PTX to a temp file.  Use process id + a random suffix to avoid
    // collisions when tests run in parallel under cargo-nextest.
    let pid = std::process::id();
    // Use a thread-local counter as a cheap disambiguator for parallel tests.
    let seq = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        SEQ.fetch_add(1, Ordering::Relaxed)
    };
    let tmp_ptx = std::env::temp_dir().join(format!("nsl_ptxval_{pid}_{seq}.ptx"));
    let tmp_cubin = std::env::temp_dir().join(format!("nsl_ptxval_{pid}_{seq}.cubin"));
    {
        let mut f = std::fs::File::create(&tmp_ptx).map_err(|e| format!("tmp create: {e}"))?;
        f.write_all(ptx.as_bytes())
            .map_err(|e| format!("tmp write: {e}"))?;
    }

    // ptxas -arch <sm> <input.ptx> -o <output.cubin>
    let output = Command::new("ptxas")
        .args(["-arch", arch])
        .arg(&tmp_ptx)
        .args(["-o"])
        .arg(&tmp_cubin)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    // Clean up temp files regardless of result.
    let _ = std::fs::remove_file(&tmp_ptx);
    let _ = std::fs::remove_file(&tmp_cubin);

    let output = match output {
        Ok(o) => o,
        Err(e) => return Err(format!("nvcc not available: {e}")),
    };

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!("nvcc rejected PTX: {stderr}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid PTX — an empty entry point.  Should compile on any
    /// sm_80-capable toolkit.
    #[test]
    fn validates_trivial_valid_ptx() {
        let ptx = r#".version 7.0
.target sm_80
.address_size 64

.visible .entry nsl_test_trivial ()
{
    ret;
}
"#;
        // If no validator is reachable, skip gracefully rather than fail.
        match validate_ptx(ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] validates_trivial_valid_ptx: no validator available: {msg}");
            }
            Err(msg) => panic!("expected trivial PTX to validate; got: {msg}"),
        }
    }

    /// Obviously invalid PTX — unbalanced braces and garbage syntax.
    /// Should be rejected by both cudarc and nvcc.
    #[test]
    fn rejects_invalid_ptx() {
        let ptx = r#".version 7.0
.target sm_80
.address_size 64

.visible .entry nsl_test_broken ()
{
    this is not valid ptx syntax
"#;
        match validate_ptx(ptx) {
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] rejects_invalid_ptx: no validator available: {msg}");
            }
            Err(_) => {} // expected — ptxas rejects it
            Ok(()) => panic!("expected invalid PTX to be rejected, but it validated"),
        }
    }
}
