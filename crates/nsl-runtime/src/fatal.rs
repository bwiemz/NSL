//! Fatal runtime conditions: reported once, then the process exits with a
//! code that names the condition (roadmap C1).
//!
//! NOT `panic!`. The sites that reach [`die`] sit under `extern "C"` entry
//! points (the allocator behind `nsl_tensor_to_device`, the kernel launcher,
//! the cuBLAS wrappers — several hundred callers), and a Rust panic in such a
//! frame cannot unwind: it becomes `panic in a function that cannot unwind`
//! → SIGABRT → on a GPU box a multi-GB core dump, with the real diagnostic
//! buried under two backtraces (observed 2026-08-31: a 5.5 GB core for a
//! condition that had already been diagnosed in full). Printing, flushing
//! and exiting is the convention `src/cuda/mod.rs`'s OOM path established;
//! this module is that path generalised, so every fatal condition has one
//! chokepoint and its own exit code.
//!
//! The exit codes are a contract with whatever supervises a compiled program
//! (a training driver, `scripts/gpu-guard.sh`, CI): they tell "the card ran
//! out" from "a driver call failed" from "the program crashed" (a panic's
//! 101, SIGABRT's 134) without parsing stderr. They are documented in
//! `docs/architecture/runtime.md` and pinned by the tests below; a new
//! condition takes the next code, an existing code never changes meaning.

use std::io::Write;

/// Exit code for a fatal GPU OOM (the first of these, kept at its original
/// value).
pub const NSL_EXIT_GPU_OOM: i32 = 12;
/// Exit code for a CUDA driver call (`cuMemAlloc`, `cuMemcpy*`, …) that
/// failed for a reason other than out-of-memory.
pub const NSL_EXIT_CUDA_DRIVER: i32 = 13;
/// Exit code for an asynchronous device error surfaced by the
/// `cuCtxSynchronize` that `--cuda-sync` inserts after a kernel or cuBLAS
/// call.
pub const NSL_EXIT_CUDA_ASYNC: i32 = 14;
/// Exit code for a cuBLAS call that failed where no partial result is safe
/// to continue from.
pub const NSL_EXIT_CUBLAS: i32 = 15;
/// Exit code for a device tensor reaching a runtime built without the
/// `cuda` feature.
pub const NSL_EXIT_CUDA_NOT_COMPILED: i32 = 16;
/// Exit code for a tensor operation asked to work on a dtype it does not
/// implement — a compiler/runtime contract violation, not a user error.
pub const NSL_EXIT_UNSUPPORTED_DTYPE: i32 = 17;
/// Exit code for a tensor operation whose operands' shapes do not satisfy
/// its contract (an operand shorter than the other, a dtype mix a fused
/// step cannot take) — a compiler/runtime contract violation, not a user
/// error.
pub const NSL_EXIT_SHAPE_MISMATCH: i32 = 18;
/// Exit code for a runtime operation this build does not implement (a
/// device-to-device transfer, an ONNX export of a block-packed dtype).
pub const NSL_EXIT_UNSUPPORTED: i32 = 19;

/// The fatal conditions the runtime exits on. Each maps to one exit code
/// ([`Fatal::exit_code`]); the diagnostic text is the caller's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fatal {
    /// The device allocator could not satisfy a request even after draining
    /// the pool and retrying.
    GpuOom,
    /// A CUDA driver call failed for a reason other than OOM
    /// (`CUDA_ERROR_ILLEGAL_ADDRESS` after a faulting kernel, a failed
    /// `cuMemcpyHtoD`, …).
    CudaDriver,
    /// The post-launch `cuCtxSynchronize` of `--cuda-sync` reported an
    /// asynchronous device error.
    CudaAsync,
    /// A cuBLAS call returned an error status on an in-place operation.
    Cublas,
    /// A device tensor reached a tensor op in a runtime built without the
    /// `cuda` feature (the program was compiled for a GPU this runtime
    /// cannot drive).
    CudaNotCompiled,
    /// A tensor op was asked to work on a dtype it does not implement: the
    /// compiler emitted a call the runtime's contract does not cover.
    UnsupportedDtype,
    /// A tensor op's operands do not satisfy its shape contract (an operand
    /// too short for the other, a fused step handed mixed dtypes): the
    /// compiler emitted a call the runtime's contract does not cover.
    ShapeMismatch,
    /// A runtime operation this build does not implement was reached (a
    /// device-to-device transfer, an ONNX export of a block-packed dtype).
    Unsupported,
}

impl Fatal {
    /// Every variant, for the exit-code tests and for documentation
    /// generators.
    pub const ALL: [Fatal; 8] = [
        Fatal::GpuOom,
        Fatal::CudaDriver,
        Fatal::CudaAsync,
        Fatal::Cublas,
        Fatal::CudaNotCompiled,
        Fatal::UnsupportedDtype,
        Fatal::ShapeMismatch,
        Fatal::Unsupported,
    ];

    /// The process exit code for this condition.
    pub const fn exit_code(self) -> i32 {
        match self {
            Fatal::GpuOom => NSL_EXIT_GPU_OOM,
            Fatal::CudaDriver => NSL_EXIT_CUDA_DRIVER,
            Fatal::CudaAsync => NSL_EXIT_CUDA_ASYNC,
            Fatal::Cublas => NSL_EXIT_CUBLAS,
            Fatal::CudaNotCompiled => NSL_EXIT_CUDA_NOT_COMPILED,
            Fatal::UnsupportedDtype => NSL_EXIT_UNSUPPORTED_DTYPE,
            Fatal::ShapeMismatch => NSL_EXIT_SHAPE_MISMATCH,
            Fatal::Unsupported => NSL_EXIT_UNSUPPORTED,
        }
    }

    /// The condition's name as it appears on the final stderr line.
    pub const fn as_str(self) -> &'static str {
        match self {
            Fatal::GpuOom => "gpu-oom",
            Fatal::CudaDriver => "cuda-driver",
            Fatal::CudaAsync => "cuda-async",
            Fatal::Cublas => "cublas",
            Fatal::CudaNotCompiled => "cuda-not-compiled",
            Fatal::UnsupportedDtype => "unsupported-dtype",
            Fatal::ShapeMismatch => "shape-mismatch",
            Fatal::Unsupported => "unsupported",
        }
    }
}

/// Report `msg` (verbatim, as an `ERROR` event on the `cuda` target), then a
/// final `[nsl] fatal: <kind>, exiting with code <n>` line, flush stderr,
/// and exit with the condition's code. Never returns and never unwinds.
pub fn die(kind: Fatal, msg: &str) -> ! {
    crate::nsl_log!(ERROR, "cuda", "{msg}");
    crate::nsl_log!(
        ERROR,
        "nsl",
        "[nsl] fatal: {}, exiting with code {}",
        kind.as_str(),
        kind.exit_code()
    );
    let _ = std::io::stderr().flush();
    std::process::exit(kind.exit_code());
}

/// The `#[cfg(not(feature = "cuda"))]` arm of a tensor op that a device
/// tensor reached: one message, one exit code, at every such site.
pub fn cuda_not_compiled() -> ! {
    die(
        Fatal::CudaNotCompiled,
        "CUDA support not compiled. Rebuild with --features cuda",
    )
}

/// `die(Fatal::UnsupportedDtype, ..)` for the "`<op>: unsupported dtype
/// <n>`" family — the same message every site printed as a panic.
pub fn unsupported_dtype(op: &str, dtype: impl std::fmt::Display) -> ! {
    die(Fatal::UnsupportedDtype, &format!("{op}: unsupported dtype {dtype}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn gpu_oom_keeps_its_original_exit_code() {
        // Supervisors have matched on 12 since the OOM path was introduced;
        // the generalisation must not move it.
        assert_eq!(Fatal::GpuOom.exit_code(), 12);
        assert_eq!(NSL_EXIT_GPU_OOM, 12);
    }

    #[test]
    fn every_condition_has_a_distinct_code_outside_the_conventional_ones() {
        let codes: HashSet<i32> = Fatal::ALL.iter().map(|k| k.exit_code()).collect();
        assert_eq!(codes.len(), Fatal::ALL.len(), "two conditions share an exit code");
        for k in Fatal::ALL {
            let c = k.exit_code();
            // 0/1 are success/generic failure, 2 is clap's usage error, 101
            // is a Rust panic, 128+ are signals: a fatal code must be none of
            // those, so a supervisor can attribute it.
            assert!((3..=100).contains(&c), "{k:?} → {c} collides with a conventional code");
        }
    }

    #[test]
    fn the_codes_are_stable() {
        // Existing codes never change meaning; a new condition takes the
        // next one. The table in docs/architecture/runtime.md mirrors this.
        let expected = [
            (Fatal::GpuOom, 12),
            (Fatal::CudaDriver, 13),
            (Fatal::CudaAsync, 14),
            (Fatal::Cublas, 15),
            (Fatal::CudaNotCompiled, 16),
            (Fatal::UnsupportedDtype, 17),
            (Fatal::ShapeMismatch, 18),
            (Fatal::Unsupported, 19),
        ];
        assert_eq!(expected.len(), Fatal::ALL.len());
        for (k, c) in expected {
            assert_eq!(k.exit_code(), c, "{k:?}");
        }
    }

    #[test]
    fn names_are_distinct_and_kebab_case() {
        let names: HashSet<&str> = Fatal::ALL.iter().map(|k| k.as_str()).collect();
        assert_eq!(names.len(), Fatal::ALL.len());
        for n in names {
            assert!(n.bytes().all(|b| b.is_ascii_lowercase() || b == b'-'), "{n}");
        }
    }
}
