//! G3 — Fused linear-CE FFI surface.
//!
//! Exports two C-ABI functions:
//!
//! * `nsl_fused_linear_ce_forward`  — forward pass
//! * `nsl_fused_linear_ce_backward` — backward pass
//!
//! Both functions accept pre-synthesised PTX (null-terminated) plus device
//! pointers for the tensors.  On CPU (device == 0) they fall back to the
//! existing CPU cross-entropy implementation.
//!
//! The PTX is supplied by the caller (typically a test or compiler-emitted
//! call site); this module does NOT synthesise PTX — that lives in
//! `nsl-codegen::fused_linear_ce`.
//!
//! # CFTP v6 — structural buffer-conformance via wengert precision_cast
//!
//! When `dtype_tag != 0` (fp16 or bf16), the caller MUST allocate
//! `x_ptr` / `w_ptr` / `bias_ptr` in HBM with a layout matching the dtype:
//! `[B*S*H]` or `[V*H]` of 16-bit elements (2 bytes per elem) for the
//! 16-bit dtypes.  The PTX kernels do `ld.global.b16` reads on those
//! pointers — supplying a 32-bit-laid-out buffer would read the upper+lower
//! halves of two adjacent f32s as a single bf16/fp16 with NO error and
//! NO diagnostic (silent numerical corruption).
//!
//! As of CFTP v6, the wengert dispatch in
//! `crates/nsl-codegen/src/wengert_lower.rs::lower_fused_linear_ce_forward`
//! (and the symmetric `..._backward_extract`) inserts an explicit
//! `nsl_tensor_to_{bf16,fp16}` cast op into the Cranelift IR before the
//! FFI call whenever `dtype_tag != 0`.
//!
//! ## Adversarial-review remediation (Findings 1 / 6 / 12 — HIGH)
//!
//! The structural cast (`cast_and_publish` in
//! `runtime/src/tensor/precision_cast.rs`) is CPU-only in v6 — it asserts
//! `t.device == 0` and panics on a GPU tensor (with an `extern "C"` abort
//! that ends the process). A real training step lowered through the
//! wengert dispatch passes GPU x/W/bias straight into `nsl_tensor_to_bf16`
//! and so would abort. To keep the FFI safe until the v7 device-side cast
//! kernel lands, this module ALSO enforces the dtype-conformance contract
//! at the FFI boundary by REFUSING `dtype_tag != 0` with a clear
//! diagnostic and a `-1` return code — preserving the Deferral-must-refuse
//! invariant. The refusal is the safe interim per the Findings 1/6/12 fix
//! options (option a: keep the refusal until a real device-side cast kernel
//! lands; the v6 structural cast is INCOMPLETE for the GPU path the FFI
//! requires).
//!
//! Callers that want to exercise the manually-staged direct-FFI numerical
//! path (`fused_linear_ce_{fp16,bf16}_v49152_numerical.rs` allocates
//! compliant 16-bit device buffers explicitly) can set the
//! `NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1` opt-in escape valve — those tests
//! KNOW the buffers are dtype-compliant because they staged them
//! themselves. The wengert-driven production path never sets that env
//! var, so a malformed buffer reaching the kernel produces a clean refusal
//! rather than silent corruption (the v5 invariant restored).

/// CFTP v6 Findings 1/6/12 (HIGH): inverse-deferral guard.
///
/// The Sprint v5 env-guarded refusal was REMOVED in d61e8282 on the
/// assumption that the wengert structural cast closes the gap — but
/// `cast_and_publish` panics on any GPU tensor, so the structural cast
/// cannot run on the production path it is supposed to fix. Restore the
/// refusal as a default-on safety net: when `dtype_tag != 0` the FFI
/// returns `-1` with a clear diagnostic UNLESS the caller has explicitly
/// opted into the manually-staged path via `NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1`.
///
/// Returns `true` if the call should refuse (i.e. caller should return `-1`
/// immediately); `false` if the call should proceed normally.
#[inline]
fn refuse_non_f32_if_unsafe(dtype_tag: i64, ffi_name: &str) -> bool {
    if dtype_tag == 0 {
        return false;
    }
    // Opt-in escape valve for direct-FFI numerical tests that stage
    // dtype-compliant device buffers themselves. The wengert-driven
    // production path NEVER sets this var, so a malformed buffer is
    // refused cleanly rather than silently corrupting the kernel reads.
    if std::env::var_os("NSL_FUSED_LCE_ALLOW_NON_F32_FFI").is_some() {
        return false;
    }
    eprintln!(
        "{ffi_name}: refusing dtype_tag={dtype_tag} — the v6 wengert \
         precision_cast (`nsl_tensor_to_{{bf16,fp16}}`) does not support \
         GPU tensors (asserts t.device==0). Set \
         NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1 to opt into the manually-staged \
         direct-FFI path (callers must allocate dtype-compliant 16-bit HBM \
         buffers themselves). Restoring the v5 default-on refusal per \
         Findings 1/6/12; the silent-corruption hazard the v5 refusal \
         guarded was reintroduced when the structural cast turned out to \
         be CPU-only.",
    );
    true
}

/// Forward: `loss[i] = -log_softmax(x[i] @ W^T + bias)[target[i]]`.
///
/// # Arguments
/// * `ptx_ptr`      — `*const u8`: null-terminated forward PTX string.
/// * `kname_ptr`    — `*const u8`: null-terminated kernel name.
/// * `x_ptr`        — device/host f32 tensor `[B, S, H]`.
/// * `w_ptr`        — device/host f32 tensor `[V, H]`.
/// * `bias_ptr`     — device/host f32 tensor `[V]`.
/// * `targets_ptr`  — device/host i64 tensor `[B*S]`.
/// * `loss_out_ptr` — device/host f32 output `[B*S]` (pre-allocated).
/// * `lse_out_ptr`  — device/host f32 output `[B*S]` (pre-allocated, for backward).
/// * `b, s, v, h`   — dims as u32.
/// * `smem_bytes`   — shared-memory bytes per CTA (from `FusedLinearCEConfig::shared_mem_bytes()`).
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default — preserves pre-Sprint-v3-2 behaviour byte-identically;
/// all existing callers including the Cranelift IR call site in
/// `wengert_lower.rs` pass this sentinel).
///
/// `1` = F16. Callers that pass `1` MUST allocate `x_ptr`, `w_ptr`, `bias_ptr`
/// as fp16 in HBM. The `loss_out_ptr` and `lse_out_ptr` outputs stay f32
/// regardless of `dtype_tag`.
///
/// `2` = Bf16 (Sprint v4-1). Same HBM-layout contract as F16 (16-bit storage
/// for x/W/bias; f32 outputs). The kernel-side bf16 cvt mnemonics need PTX
/// ISA 7.8+ — codegen bumps to `.version 8.0` on the bf16 path so callers
/// don't need to do anything special at this FFI layer.
///
/// Threading actual fp16/bf16 from user code (so the @fused_lm_ce decorator
/// drives `dtype_tag` end-to-end) is a v4-2 follow-on.
///
/// Any value outside {0, 1, 2} is treated as F32 (defensive — preserves
/// forward-compat with un-recompiled callers).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_forward(
    ptx_ptr: i64,
    kname_ptr: i64,
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    loss_out_ptr: i64,
    lse_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v6 Findings 1/6/12: default-on dtype refusal.  The v6 structural
    // cast (`cast_and_publish`) panics on GPU tensors, so it cannot run on
    // the production wengert path; refuse here unless the caller has opted
    // into the manually-staged direct-FFI path via
    // `NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1`.
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_forward") {
        return -1;
    }
    let _dtype_tag = dtype_tag;
    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_forward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            loss_out_ptr as u64,
            lse_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!(
                "nsl_fused_linear_ce_forward: CUDA launch failed rc={}",
                rc
            );
            return -(rc as i64);
        }
        // Synchronise so the test can read results immediately.
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
        }
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, kname_ptr, x_ptr, w_ptr, bias_ptr, targets_ptr,
                 loss_out_ptr, lse_out_ptr, b, s, v, h, smem_bytes);
        eprintln!("nsl_fused_linear_ce_forward: compiled without cuda feature");
        -1
    }
}

/// Sprint-3 large-vocab forward: two-kernel (per-tile partials + per-row finalize).
///
/// Use this FFI instead of `nsl_fused_linear_ce_forward` when
/// `FusedLinearCEConfig::is_large_vocab()` returns true (vocab > 8192).
/// The PTX module pointed at by `ptx_ptr` MUST contain BOTH kernels —
/// the codegen produces this in one byte string via
/// `synthesize_fused_linear_ce_ptx` whenever `is_large_vocab()` is true.
///
/// # Lifecycle of `partials_ptr`
/// Caller-owned: allocate `(B*S) * num_tiles * 2 * sizeof(f32)` bytes of
/// device memory before this call and free it after. The buffer is overwritten
/// by Kernel A and read by Kernel B; its contents on return are
/// implementation-defined (callers should not depend on them).
///
/// # Arguments
/// * `ptx_ptr`            — `*const u8`: null-terminated PTX string with both kernels.
/// * `partials_kname_ptr` — `*const u8`: null-terminated Kernel A name
///   (`FusedLinearCEConfig::large_partials_kernel_name`).
/// * `finalize_kname_ptr` — `*const u8`: null-terminated Kernel B name
///   (`FusedLinearCEConfig::large_finalize_kernel_name`).
/// * `x_ptr`        — device f32 `[B, S, H]`.
/// * `w_ptr`        — device f32 `[V, H]`.
/// * `bias_ptr`     — device f32 `[V]`.
/// * `targets_ptr`  — device i64 `[B*S]`.
/// * `partials_ptr` — device f32 `[B*S, num_tiles, 2]` (caller-owned scratch).
/// * `loss_out_ptr` — device f32 `[B*S]` output (pre-allocated).
/// * `lse_out_ptr`  — device f32 `[B*S]` output (pre-allocated; for backward reuse).
/// * `b, s, v, h`   — dims as i64.
/// * `num_tiles`    — `vocab_size.div_ceil(vocab_tile)` (host-computed).
/// * `smem_bytes`   — Kernel A per-CTA smem budget; Kernel B is launched with 0.
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default; preserves pre-Sprint-v3-2 ABI). `1` = F16 — caller
/// MUST allocate x/W/bias in HBM as fp16. `2` = Bf16 (Sprint v4-1) —
/// same HBM layout as F16 (16-bit storage). `partials_ptr` stays f32-sized
/// across all dtypes (`large_partials_bytes()` is dtype-independent).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_forward_large(
    ptx_ptr: i64,
    partials_kname_ptr: i64,
    finalize_kname_ptr: i64,
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    partials_ptr: i64,
    loss_out_ptr: i64,
    lse_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    num_tiles: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v6 Findings 1/6/12: default-on dtype refusal (see forward).
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_forward_large") {
        return -1;
    }
    let _dtype_tag = dtype_tag;
    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_forward_large(
            ptx_ptr as *const u8,
            partials_kname_ptr as *const u8,
            finalize_kname_ptr as *const u8,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            partials_ptr as u64,
            loss_out_ptr as u64,
            lse_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            num_tiles as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!("nsl_fused_linear_ce_forward_large: CUDA launch failed rc={rc}");
            return -(rc as i64);
        }
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            ptx_ptr, partials_kname_ptr, finalize_kname_ptr,
            x_ptr, w_ptr, bias_ptr, targets_ptr, partials_ptr,
            loss_out_ptr, lse_out_ptr,
            b, s, v, h, num_tiles, smem_bytes,
        );
        eprintln!("nsl_fused_linear_ce_forward_large: compiled without cuda feature");
        -1
    }
}

/// Backward: computes `dx[B,S,H]`, `dW[V,H]`, `dbias[V]` given saved lse.
///
/// # Arguments
/// * `ptx_ptr`        — `*const u8`: null-terminated backward PTX string.
/// * `kname_ptr`      — `*const u8`: null-terminated backward kernel name.
/// * `grad_output`    — upstream scalar gradient (f32, reinterpreted from i64 bits).
/// * `x_ptr`          — device f32 `[B, S, H]`.
/// * `w_ptr`          — device f32 `[V, H]`.
/// * `bias_ptr`       — device f32 `[V]`.
/// * `targets_ptr`    — device i64 `[B*S]`.
/// * `lse_ptr`        — device f32 `[B*S]` saved from forward.
/// * `dx_out_ptr`     — device f32 `[B, S, H]` (zeroed by caller).
/// * `dw_out_ptr`     — device f32 `[V, H]`    (zeroed by caller).
/// * `dbias_out_ptr`  — device f32 `[V]`        (zeroed by caller).
/// * `b, s, v, h`     — dims as i64.
/// * `num_valid`      — count of non-ignore positions (host-computed from targets).
/// * `smem_bytes`     — shared-memory bytes per CTA.
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default; preserves pre-Sprint-v3-2 ABI). `1` = F16 — caller
/// MUST allocate x/W/bias in HBM as fp16. `2` = Bf16 (Sprint v4-1) — same
/// HBM layout as F16 (16-bit storage). `lse_ptr` stays f32 across all
/// dtypes (forward writes f32 regardless of activation dtype); `dx_out` /
/// `dw_out` / `dbias_out` stay f32 regardless of dtype_tag (PyTorch
/// master-gradient convention — the kernel emits `red.global.add.f32`
/// for cross-CTA accumulation).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_backward(
    ptx_ptr: i64,
    kname_ptr: i64,
    grad_output_bits: i64, // f32 bits packed into i64
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    lse_ptr: i64,
    dx_out_ptr: i64,
    dw_out_ptr: i64,
    dbias_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    num_valid: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v6 Findings 1/6/12: default-on dtype refusal (see forward).
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_backward") {
        return -1;
    }
    let grad_output = f32::from_bits(grad_output_bits as u32);
    let _dtype_tag = dtype_tag;

    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_backward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            grad_output,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            lse_ptr as u64,
            dx_out_ptr as u64,
            dw_out_ptr as u64,
            dbias_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            num_valid as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!(
                "nsl_fused_linear_ce_backward: CUDA launch failed rc={}",
                rc
            );
            return -(rc as i64);
        }
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
        }
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, kname_ptr, grad_output, x_ptr, w_ptr, bias_ptr,
                 targets_ptr, lse_ptr, dx_out_ptr, dw_out_ptr, dbias_out_ptr,
                 b, s, v, h, num_valid, smem_bytes);
        eprintln!("nsl_fused_linear_ce_backward: compiled without cuda feature");
        -1
    }
}

#[cfg(test)]
mod v6_default_refusal_tests {
    //! CFTP v6 Findings 1/6/12 (HIGH): the FFI dtype refusal is RESTORED
    //! and now DEFAULT-ON for `dtype_tag != 0`, because the v6 structural
    //! cast (`cast_and_publish`) is CPU-only and panics on GPU tensors —
    //! it cannot run on the production wengert path. The opt-in escape
    //! valve `NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1` lets manually-staged
    //! direct-FFI numerical tests proceed.
    //!
    //! These tests run on BOTH CUDA-enabled and CUDA-disabled lanes — the
    //! refusal short-circuits BEFORE the cuda-gated launcher, so the same
    //! `-1` return is observable on either build profile.  This closes
    //! Finding 5 (LOW): the v5 negative-pin only ran on `cfg(not(cuda))`.

    /// Tests serialize on this mutex so env-var manipulation does not race
    /// across parallel test threads.  std::env::set_var is not thread-safe.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard for the opt-in escape-valve env var.
    struct EnvGuard {
        prev: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        fn set(value: Option<&str>) -> Self {
            let prev = std::env::var_os("NSL_FUSED_LCE_ALLOW_NON_F32_FFI");
            match value {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_ALLOW_NON_F32_FFI", v),
                None => std::env::remove_var("NSL_FUSED_LCE_ALLOW_NON_F32_FFI"),
            }
            EnvGuard { prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_ALLOW_NON_F32_FFI", v),
                None => std::env::remove_var("NSL_FUSED_LCE_ALLOW_NON_F32_FFI"),
            }
        }
    }

    /// Forward dtype_tag=2 (bf16) MUST refuse with `-1` by default on
    /// both cuda-enabled and cuda-disabled lanes.  The refusal fires
    /// BEFORE the `#[cfg(feature="cuda")]` launcher block, so pointer args
    /// are never read — null pointers are safe input.
    #[test]
    fn default_refuses_bf16_dtype_tag_forward_without_env_opt_in() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None); // ensure unset
        let rc = super::nsl_fused_linear_ce_forward(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2,
        );
        assert_eq!(
            rc, -1,
            "dtype_tag=2 (bf16) must refuse without NSL_FUSED_LCE_ALLOW_NON_F32_FFI; \
             the v6 structural cast is CPU-only and cannot ensure conformance."
        );
    }

    /// Mirror for fp16 dtype_tag=1.
    #[test]
    fn default_refuses_fp16_dtype_tag_forward_without_env_opt_in() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None);
        let rc = super::nsl_fused_linear_ce_forward(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        );
        assert_eq!(rc, -1, "dtype_tag=1 (fp16) must refuse without opt-in");
    }

    /// Mirror for backward.
    #[test]
    fn default_refuses_bf16_dtype_tag_backward_without_env_opt_in() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None);
        let rc = super::nsl_fused_linear_ce_backward(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2,
        );
        assert_eq!(rc, -1, "backward dtype_tag=2 must refuse without opt-in");
    }

    /// F32 (dtype_tag=0) MUST NOT trigger the refusal.  On a non-cuda build
    /// the FFI returns `-1` for the compiled-without-cuda reason, but the
    /// refusal helper itself must return `false` (no refusal).  This pins
    /// the contract that the F32 byte-identity path is never blocked.
    #[test]
    fn f32_dtype_tag_does_not_trigger_refusal() {
        assert!(
            !super::refuse_non_f32_if_unsafe(0, "test"),
            "dtype_tag=0 must never refuse — F32 byte-identity contract"
        );
    }

    /// Opt-in escape valve via NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1 allows
    /// dtype_tag=1/2 to proceed past the refusal helper.
    #[test]
    fn env_opt_in_disables_refusal_for_bf16_dtype_tag() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(Some("1"));
        assert!(
            !super::refuse_non_f32_if_unsafe(2, "test"),
            "with NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1 set, dtype_tag=2 must NOT refuse \
             (escape valve for manually-staged direct-FFI numerical tests)."
        );
    }
}
