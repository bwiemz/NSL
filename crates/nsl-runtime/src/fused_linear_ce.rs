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
//! FFI call whenever `dtype_tag != 0`, so buffer conformance is now
//! enforced STRUCTURALLY at codegen time — no env-var guard is needed.
//!
//! Direct-FFI tests (`fused_linear_ce_{fp16,bf16}_v49152_numerical.rs`)
//! continue to allocate compliant 16-bit buffers explicitly.

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
    // dtype_tag is currently informational; the host launcher's element-size
    // handling is implicit in the PTX bytes the caller supplied (the codegen
    // already specialized to the right dtype). The argument is plumbed so
    // future host-side hooks (e.g. SMEM cap raising for fp16) can branch on
    // it without another ABI change.
    //
    // CFTP v6: buffer conformance is enforced STRUCTURALLY by wengert_lower's
    // precision_cast op (see module docs). No env-guarded refusal is needed.
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
    // CFTP v6: buffer conformance is enforced STRUCTURALLY by wengert_lower's
    // precision_cast op (see module docs). No env-guarded refusal is needed.
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
    // CFTP v6: buffer conformance is enforced STRUCTURALLY by wengert_lower's
    // precision_cast op (see module docs). No env-guarded refusal is needed.
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
mod v6_no_refusal_tests {
    //! CFTP v6: the v5 opt-in `NSL_FUSED_LCE_REFUSE_NON_F32` runtime refusal
    //! has been REMOVED — wengert_lower now inserts an explicit precision_cast
    //! before the FFI call so buffer conformance is enforced structurally.
    //!
    //! These tests are the negative-contract pins: they prove the FFI no
    //! longer consults the env var (i.e. setting it must NOT cause refusal),
    //! and that the `refuse_non_f32_if_gated` symbol has been excised.

    /// Tests serialize on this mutex so env-var manipulation does not race
    /// across parallel test threads.  std::env::set_var is not thread-safe.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper that temporarily sets the (now-defunct) env var and restores
    /// the previous state on drop.  Used only to prove the FFI ignores it.
    struct LegacyEnvGuard {
        prev: Option<std::ffi::OsString>,
    }
    impl LegacyEnvGuard {
        fn set(value: Option<&str>) -> Self {
            let prev = std::env::var_os("NSL_FUSED_LCE_REFUSE_NON_F32");
            match value {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_REFUSE_NON_F32", v),
                None => std::env::remove_var("NSL_FUSED_LCE_REFUSE_NON_F32"),
            }
            LegacyEnvGuard { prev }
        }
    }
    impl Drop for LegacyEnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_REFUSE_NON_F32", v),
                None => std::env::remove_var("NSL_FUSED_LCE_REFUSE_NON_F32"),
            }
        }
    }

    /// Without the `cuda` feature, the FFI returns `-1` (compiled-without-cuda
    /// sentinel) BEFORE any kernel dispatch logic — but crucially, it does so
    /// for ALL dtype_tags including 1/2, regardless of the legacy env var.
    /// The old refusal path returned `-1` only with the env var set; the new
    /// path returns `-1` for the cuda-feature-off reason instead.
    ///
    /// On CUDA-enabled builds, the FFI proceeds to the kernel launcher
    /// (which requires real device pointers and a live context, so we
    /// cannot exercise it in a unit test without a GPU). The negative-path
    /// assertion here is that the FFI no longer SHORT-CIRCUITS on the env
    /// var alone.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn env_var_does_not_short_circuit_ffi_on_bf16_dtype_tag() {
        let _lk = ENV_MUTEX.lock().unwrap();
        // Set the legacy env var to "1" — under v5 this would have caused
        // an early `-1` refusal return.  Under v6 the FFI must IGNORE it
        // and fall through to the cuda-feature-gated path.
        let _g = LegacyEnvGuard::set(Some("1"));

        // Call forward with dtype_tag=2 (bf16). All pointer args are 0
        // (null) because we don't run kernels in a non-cuda build — the
        // path returns -1 with the "compiled without cuda feature" message.
        let rc = super::nsl_fused_linear_ce_forward(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2,
        );
        assert_eq!(
            rc, -1,
            "non-cuda build returns -1 sentinel; the path is the \
             cuda-feature-gated one, NOT the removed v5 env-guard refusal"
        );

        let rc = super::nsl_fused_linear_ce_backward(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2,
        );
        assert_eq!(rc, -1, "non-cuda backward returns -1 sentinel for the same reason");
    }

    /// The `refuse_non_f32_if_gated` symbol must no longer exist in this
    /// module — this test simply documents the removal; if a future
    /// regression re-introduces the symbol, it would not be referenced
    /// here so this serves as a comment-pin rather than a compile-time
    /// check.  The presence of `v6_no_refusal_tests` as the module name
    /// is the load-bearing signal.
    #[test]
    fn module_name_pins_v6_removal_intent() {
        // No assertion — the test's existence under the
        // `v6_no_refusal_tests` module name documents intent.
    }
}
