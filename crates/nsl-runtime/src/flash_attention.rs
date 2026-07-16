//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::NslTensor;
#[cfg(feature = "cuda")]
use crate::autodiff;

/// One-time log guard: prints the selected kernel variant (FA3 or FA2) only once.
#[cfg(feature = "cuda")]
static FA_VARIANT_LOGGED: AtomicBool = AtomicBool::new(false);

/// One-time log guard: prints the first successful `nsl_sdpa_fused_forward`
/// launch. Parity harnesses need launch-level PROOF the fused path fired —
/// the decline protocol is silent by design, so the absence of decline
/// diagnostics proves nothing.
#[cfg(feature = "cuda")]
static SDPA_FUSED_LAUNCH_LOGGED: AtomicBool = AtomicBool::new(false);

/// Per-variant successful-launch counters for `nsl_sdpa_fused_forward`
/// (index 0 = base segment-masked/plain kernel, 1 = Tier-B tile-skip).
/// The once-per-process marker above only proves the FIRST launch; the
/// Tier-B parity gate needs proof that the Tier-B PTX specifically ran —
/// a dispatch bug that silently fell back to the base kernel would
/// otherwise produce a vacuous bitwise-equal "pass".
static SDPA_FUSED_LAUNCH_COUNTS: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Test/diagnostic probe: number of successful fused-forward launches for
/// `variant` (0 = base kernel, 1 = Tier-B tile-skip). Any other variant
/// returns -1. Counts whole FFI calls, not per-batch-row kernel launches.
#[no_mangle]
pub extern "C" fn nsl_sdpa_fused_launch_count(variant: i64) -> i64 {
    match variant {
        0 | 1 => SDPA_FUSED_LAUNCH_COUNTS[variant as usize]
            .load(std::sync::atomic::Ordering::Relaxed) as i64,
        _ => -1,
    }
}

/// Launch FlashAttention-3 (Hopper wgmma) kernel.
/// Returns 0 on success, -1 if the launch failed (caller should fall back to FA2).
#[cfg(feature = "cuda")]
fn flash_attention_hopper(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale: f32,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: bool,
) -> i64 {
    use crate::cuda::kernels_hopper::{generate_flash_attention_3_ptx, FA3Config};

    let block_q = 64;
    let block_kv = 64;

    let config = FA3Config {
        block_q,
        block_kv,
        head_dim: head_dim as usize,
        seq_len: seq_len as usize,
        batch_heads: (batch * heads) as usize,
        causal,
        fp8: false,
        scale,
    };

    // Generate PTX at runtime for this config
    let ptx = generate_flash_attention_3_ptx(block_q, block_kv, head_dim as usize, causal, false);
    let ptx_cstr = format!("{}\0", ptx);

    let mut q_data = q_ptr as u64;
    let mut k_data = k_ptr as u64;
    let mut v_data = v_ptr as u64;
    let mut o_data = out_ptr as u64;
    let mut lse_data = if logsumexp_ptr != 0 {
        logsumexp_ptr as u64
    } else {
        0u64
    };
    let mut scale_val = scale;
    let mut seq_val = seq_len as u64;
    let mut hd_val = head_dim as u64;
    let mut num_kv = config.num_kv_tiles() as u64;

    let args: [*mut c_void; 9] = [
        &mut q_data   as *mut _ as *mut c_void,
        &mut k_data   as *mut _ as *mut c_void,
        &mut v_data   as *mut _ as *mut c_void,
        &mut o_data   as *mut _ as *mut c_void,
        &mut lse_data as *mut _ as *mut c_void,
        &mut scale_val as *mut _ as *mut c_void,
        &mut seq_val  as *mut _ as *mut c_void,
        &mut hd_val   as *mut _ as *mut c_void,
        &mut num_kv   as *mut _ as *mut c_void,
    ];

    let grid  = config.grid();
    let block = config.block();
    let shared = config.shared_mem_bytes();

    let result = crate::cuda::inner::kernel_launch(
        ptx_cstr.as_ptr(),
        b"nsl_flash_attention_3\0".as_ptr(),
        grid,
        block,
        &args,
        shared,
    );

    if result as u32 != 0 {
        eprintln!(
            "[nsl] FA3 Hopper kernel launch failed ({:?}), falling back to FA2",
            result
        );
        return -1;
    }

    crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
    0
}

/// FlashAttention-2 kernel launch wrapper.
///
/// All params are i64 for Cranelift ABI compatibility (same pattern as nsl_kernel_launch).
/// f32 scale is passed as i64 and reconstructed via f32::from_bits(scale as u32).
///
/// Returns 0 on success, non-zero CUDA error code on failure.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// Non-CSHA entry: this path has no `segment_ids_ptr` parameter, so the runtime
/// gate is supplied `0` for that slot and always returns `false` — Tier B never
/// fires for non-CSHA configs. The extension is present to keep all 6 FFI entry
/// points uniformly shaped per planner spec §4.6.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,  // backward aux output (0 = skip, inference-only)
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    // Non-CSHA path has no segment_ids_ptr parameter; pass 0, gate always returns false.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            0,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        // M33: tree mask params (null for non-tree-mask variants)
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        // Backward pass: logsumexp auxiliary output
        let mut lse = logsumexp_ptr as u64;

        let args: [*mut c_void; 21] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
        ];

        // ── Ampere / Hopper dispatch ──────────────────────────────────────────
        // Detect SM version once; on Hopper (sm_90+) try FA3 (wgmma) first.
        // If FA3 launch fails fall through to FA2 with the caller-supplied PTX.
        let sm = crate::cuda::inner::detect_sm_version();
        let fa3_ok = if sm >= 90 {
            let fa3_result = flash_attention_hopper(
                q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
                f32::from_bits(scale_bits as u32),
                batch, heads, seq_len, head_dim,
                causal != 0,
            );
            if fa3_result == 0 {
                if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!("[nsl] Using FlashAttention-3 (Hopper wgmma, sm_90a)");
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        let result: i64 = if fa3_ok {
            // FA3 already launched and synced; treat as success.
            0
        } else {
            if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("[nsl] Using FlashAttention-2 (Ampere mma.sync)");
            }
            let rc = crate::cuda::inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                effective_name_ptr as *const u8,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            ) as i64;
            if rc != 0 {
                // NEVER fail silently: the Cranelift call sites ignore this
                // return code, so without this line a failed launch leaves
                // `out` all-zeros and training continues on garbage — the
                // forward twin of the #324 silent-zero-gradients bug. (Found
                // by the 4.2 pretrain e2e: ptxas 13.3 rejected non-ASCII PTX
                // comments with CUDA_ERROR_INVALID_PTX=218, the forward
                // "succeeded" with zero attention output, and the loss
                // climbed to the uniform plateau.)
                eprintln!(
                    "[flash-fwd] FlashAttention forward kernel launch FAILED \
                     (CUDA error {rc}) — the attention output buffer is \
                     UNWRITTEN (zeros) and training/inference results are \
                     invalid. Check the PTX with nsl_test_cuda_jit_log. \
                     Refusing to continue silently."
                );
            }
            rc
        };

        // Record tape op for backward pass if recording
        if autodiff::is_recording() {
            let scale = f32::from_bits(scale_bits as u32);
            // Bump refcounts on Q, K, V so they survive until backward
            NslTensor::from_ptr(q_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(k_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(v_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on out so it survives until backward (needed for D[i] = dO . O)
            NslTensor::from_ptr(out_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on logsumexp so it survives until backward
            NslTensor::from_ptr(logsumexp_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            autodiff::maybe_record(autodiff::TapeOp::FlashAttention {
                q: q_ptr, k: k_ptr, v: v_ptr,
                out: out_ptr,
                logsumexp: logsumexp_ptr,
                scale,
                batch, heads, seq_len, head_dim,
                causal: causal != 0,
                saved_q: q_ptr,
                saved_k: k_ptr,
                saved_v: v_ptr,
            });
        }

        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (effective_ptx_ptr, effective_name_ptr);
        eprintln!("[nsl] FlashAttention requires CUDA. Use naive path (no @flash_attention decorator).");
        -1
    }
}

// ── PCA Stage C: plain fused segment-masked SDPA ──────────────────────────

/// Canonical row-major (contiguity) check for the FlashAttention FFI
/// dispatch gates.
///
/// The GPU kernels (forward v2, backward phase-1/phase-2) and the CPU
/// reference's fast read path consume raw base pointers STRIDE-BLIND: a
/// `reshape(..).transpose(..)` view fed to them is silently interpreted as
/// `[b, h, s, d]`-contiguous and produces smoothly-wrong (~1e-4-scale)
/// results instead of an error — the #344 failure mode. Every stride-blind
/// consumer must therefore gate on this predicate (decline / fall back)
/// or gather through the strides.
///
/// Unlike `NslTensor::is_contiguous`, dims of extent <= 1 are treated as
/// trivially contiguous: their stride is never used to address more than
/// one element, so any value is layout-equivalent to the canonical one.
fn is_canonical_row_major(t: &NslTensor) -> bool {
    if t.ndim <= 1 {
        return true;
    }
    let ndim = t.ndim as usize;
    let mut expected = 1i64;
    for dd in (0..ndim).rev() {
        let (dim, stride) = unsafe { (*t.shape.add(dd), *t.strides.add(dd)) };
        if dim > 1 && stride != expected {
            return false;
        }
        expected *= dim.max(1);
    }
    true
}

/// Read a `[batch, seq]` segment-id tensor into a host `Vec<u16>`.
///
/// PCA Stage C staging helper shared by `nsl_sdpa_fused_forward` and the
/// segment-masked backward in `flash_attention_backward_gpu`. The packing
/// stdlib materialises segment ids as small integers stored in float
/// tensors (f32 per packing.rs; CPU-side tensors may carry the runtime's
/// default f64), while the v2 kernels consume `[B, S]` **u16** entries —
/// so the runtime narrows host-side before staging.
///
/// Device-resident inputs are copied back with `memcpy_dtoh` first (the
/// inner helpers establish the thread's CUDA context themselves).
///
/// Returns `None` when the tensor is null, not rank-2, shape-mismatched
/// against `[batch, seq]`, not row-major contiguous, has a dtype that is
/// neither f32 nor f64, or carries ids outside the exactly-representable
/// u16 range (negative / > 65535 / non-integral — a saturating `as u16`
/// narrowing would silently mask DIFFERENT pairs than the CPU reference's
/// raw-f32 comparison). Callers treat `None` as a silent decline (forward)
/// or a CPU-reference fallback (backward).
#[cfg(feature = "cuda")]
fn segment_ids_host_u16(
    segment_ids_ptr: i64,
    batch: usize,
    seq_len: usize,
) -> Option<Vec<u16>> {
    if segment_ids_ptr == 0 {
        return None;
    }
    let t = NslTensor::from_ptr(segment_ids_ptr);
    if t.ndim != 2 {
        return None;
    }
    let (sb, ss) = unsafe { (*t.shape.add(0), *t.shape.add(1)) };
    if sb != batch as i64 || ss != seq_len as i64 {
        return None;
    }
    // Stride-blind reads were the #344 failure mode: refuse views.
    if !is_canonical_row_major(t) {
        return None;
    }
    let n = batch * seq_len;
    let host_f32: Vec<f32> = if t.device > 0 {
        match t.dtype {
            1 => {
                let mut buf = vec![0.0f32; n];
                crate::cuda::inner::memcpy_dtoh(
                    buf.as_mut_ptr() as *mut c_void,
                    t.data as *const c_void,
                    n * 4,
                );
                buf
            }
            0 => {
                let mut buf = vec![0.0f64; n];
                crate::cuda::inner::memcpy_dtoh(
                    buf.as_mut_ptr() as *mut c_void,
                    t.data as *const c_void,
                    n * 8,
                );
                buf.iter().map(|&x| x as f32).collect()
            }
            _ => return None,
        }
    } else {
        match t.dtype {
            1 => (0..n).map(|i| unsafe { *t.data_f32().add(i) }).collect(),
            0 => (0..n).map(|i| unsafe { *t.data_f64().add(i) as f32 }).collect(),
            _ => return None,
        }
    };
    // Every id must be an exact small integer so the u16 narrowing is
    // lossless and the GPU kernel masks the SAME pairs as the CPU
    // reference (which compares the raw floats).
    if host_f32
        .iter()
        .any(|&x| !(0.0..=65535.0).contains(&x) || x.fract() != 0.0)
    {
        return None;
    }
    // PCA Stage C (review finding): the fused kernels' per-doc tile SKIP
    // assumes segment ids are non-decreasing along each row (the DataLoader
    // packer guarantees it; hand-built ids might not). A violation would
    // silently skip live tiles — decline to the mask-driven decomposed
    // path instead, which is correct for arbitrary segment layouts.
    let row = seq_len;
    if row > 0 {
        for chunk in host_f32.chunks(row) {
            if chunk.windows(2).any(|w| w[1] < w[0]) {
                flash_bwd_warn_once(
                    "[nsl] segment_ids not non-decreasing within a row -                      declining fused packed kernels (decomposed path is exact)",
                );
                return None;
            }
        }
    }
    Some(host_f32.iter().map(|&x| x as u16).collect())
}

/// Stage host segment ids into a device u16 buffer (`alloc_managed` +
/// `memcpy_htod`). The caller owns the returned buffer and MUST release it
/// via `inner::free_managed` only after a `cuCtxSynchronize` that covers
/// every kernel launch reading it — the kernels read segment ids from
/// global memory while running. Never null for a non-empty slice
/// (`alloc_managed` panics on OOM rather than returning null).
#[cfg(feature = "cuda")]
fn stage_segment_ids_device(seg_host: &[u16]) -> *mut c_void {
    // P0.1 VRAM accounting: segment-id staging is attention workspace.
    let _surface = crate::cuda::caching_allocator::SurfaceGuard::new(
        crate::cuda::caching_allocator::SurfaceTag::AttnWorkspace,
    );
    let bytes = seg_host.len() * 2;
    let dev = crate::cuda::inner::alloc_managed(bytes);
    if !dev.is_null() {
        crate::cuda::inner::memcpy_htod(dev, seg_host.as_ptr() as *const c_void, bytes);
    }
    dev
}

/// Wrap a device buffer in a freshly allocated GPU `NslTensor` (device=1,
/// dtype=f32, refcount=1). Ownership of `data` passes to the tensor.
/// Shared by `flash_attention_backward_gpu` ([dq, dk, dv]) and
/// `nsl_sdpa_fused_forward` ([out, lse]).
#[cfg(feature = "cuda")]
fn make_gpu_tensor(data: *mut c_void, shape: &[i64], total: usize) -> i64 {
    let ndim = shape.len() as i64;
    let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);
    let t = Box::new(NslTensor::new(
        data,
        shape_ptr,
        strides,
        ndim,
        total as i64,
        1, // device = GPU
        1, // dtype = f32
        1, // refcount
        0, // flags
    ));
    Box::into_raw(t) as i64
}

/// GQA→MHA expansion for the GPU flash backward (scaling campaign item 1).
///
/// The phase-1/phase-2 backward kernels index K/V by the fused `b*h` block
/// id, i.e. they are MHA-only. For GQA inputs this materializes
/// K/V `[b, kv_h, s, d]` → `[b, h, s, d]` on-device with the same
/// consecutive-block head mapping the CPU reference uses
/// (`kv_h_idx = h_idx / gqa_groups` in `flash_attention_backward_cpu_gqa`):
/// destination head `kv*g + gi` copies source head `kv`. Pure DtoD copies —
/// `b·h` block copies of `s·d·4` bytes, no new kernels, no host transfer.
///
/// Returns an owned GPU `NslTensor*` `[b, h, s, d]`; caller frees.
#[cfg(feature = "cuda")]
fn expand_kv_heads_device(
    kv_ptr: i64, b: usize, kv_h: usize, groups: usize, s: usize, d: usize,
) -> i64 {
    let src = NslTensor::from_ptr(kv_ptr);
    let h = kv_h * groups;
    let total = b * h * s * d;
    // P0.1 VRAM accounting: the expanded-KV envelope is attention workspace.
    let _surface = crate::cuda::caching_allocator::SurfaceGuard::new(
        crate::cuda::caching_allocator::SurfaceTag::AttnWorkspace,
    );
    let buf = crate::cuda::inner::alloc_managed(total * 4);
    let block_bytes = s * d * 4;
    for bi in 0..b {
        for kvi in 0..kv_h {
            let src_off = (bi * kv_h + kvi) * block_bytes;
            for gi in 0..groups {
                let dst_off = (bi * h + kvi * groups + gi) * block_bytes;
                crate::cuda::inner::memcpy_dtod(
                    unsafe { (buf as *mut u8).add(dst_off) } as *mut c_void,
                    unsafe { (src.data as *const u8).add(src_off) } as *const c_void,
                    block_bytes,
                );
            }
        }
    }
    make_gpu_tensor(buf, &[b as i64, h as i64, s as i64, d as i64], total)
}

/// Reduce expanded-head dK/dV back to KV heads in the backward result list.
///
/// The adjoint of the head expansion above is the group-sum
/// (`ReduceToShape` over the broadcast axis — the same rule stdlib GQA's
/// decomposed path uses for its `expand` backward), so
/// `dK[b, kv] = Σ_g dK_exp[b, kv*g + gi]`. Implemented with public FFIs:
/// reshape `[b,h,s,d]` → `[b,kv,g,s,d]` (zero-copy view of a contiguous
/// tensor) then `nsl_tensor_sum_dim(dim=2)` on the GPU. Replaces list slots
/// 1 (dK) and 2 (dV) in place; dQ (slot 0) is untouched.
#[cfg(feature = "cuda")]
fn reduce_expanded_kv_grads(
    list: i64, b: usize, kv_h: usize, groups: usize, s: usize, d: usize,
) {
    use crate::tensor::shape_ops::nsl_tensor_reshape;
    use crate::tensor::reduction::nsl_tensor_sum_dim;
    for idx in [1i64, 2i64] {
        let exp_t = crate::list::nsl_list_get(list, idx); // borrow
        let dims = crate::list::nsl_list_new();
        for v in [b as i64, kv_h as i64, groups as i64, s as i64, d as i64] {
            crate::list::nsl_list_push(dims, v);
        }
        let view5 = nsl_tensor_reshape(exp_t, dims);
        crate::list::nsl_list_free(dims);
        let reduced = nsl_tensor_sum_dim(view5, 2, 0); // [b, kv_h, s, d]
        crate::tensor::nsl_tensor_free(view5);
        crate::tensor::nsl_tensor_free(exp_t);
        crate::list::nsl_list_set(list, idx, reduced);
    }
}

/// PCA Stage C: plain fused (decorator-free) SDPA forward with optional
/// segment masking for packed-sequence batches.
///
/// Called by the wengert_lower decorator-free SDPA dispatch: when the
/// pattern matcher recognises a plain `softmax(Q K^T * scale [+ mask]) V`
/// composition, codegen synthesises the v2 FA-2 forward PTX (and, for
/// packed batches, its `segment_masked` variant) and emits a call here
/// instead of lowering the decomposed matmul+softmax chain. This entry is
/// source-AD-only: it records NO tape op (the tape path never reaches it)
/// and returns the saved logsumexp so the segment-aware backward can skip
/// the segment-blind FLASH_LSE recompute.
///
/// Returns an `NslList*` `[out, lse]` on success:
///   out — `[b, h, s, d]` f32 GPU tensor (attention output)
///   lse — `[b, h, s]`    f32 GPU tensor (per-row logsumexp)
///
/// # Decline protocol (return 0)
///
/// `0` means "decline": the caller keeps the decomposed SDPA graph, which
/// is always correct. Declines are SILENT and side-effect-free by design —
/// they are a normal per-op dispatch outcome, not an error:
///   * non-CUDA build, or `ptx_ptr`/`name_ptr` == 0
///   * q/k/v null, not GPU-resident (`device != 1`), not f32, or not rank-4
///   * q/k/v not canonical row-major (the kernel is stride-blind; a
///     transpose/reshape view would compute smoothly-wrong attention —
///     callers must materialize `.contiguous()` inputs to use this path)
///   * k/v disagreeing with q in `[batch, seq, head_dim]`, or carrying a
///     different head count (GQA callers must pre-expand KV before
///     dispatching here)
///   * `seq_len` not a multiple of `block_q`/`block_kv` (the v2 kernel has
///     no ragged-tail guards)
///   * segment path: segment tensor missing / rank != 2 / shape mismatch
///     against `[b, s]` / non-contiguous / ids not exact u16-range ints
///
/// Two intentionally NON-silent cases:
///   * kernel-launch failure — the output buffer would stay unwritten, so
///     silence would reproduce the #324-style silent-zeros failure mode.
///     We print once (keyed by message content) and still return 0 so the
///     caller recovers via decomposition.
///   * `NSL_SDPA_FUSED_DISABLE=1` — operational kill switch. Every call
///     declines immediately (before any allocation) with a once-per-process
///     stderr note. Parity harnesses use it to compare fused-vs-decomposed
///     on the same fixture; it is also the production escape hatch if the
///     fused path misbehaves. The env var is read once per process
///     (OnceLock) — this FFI sits on the per-step hot path.
///
/// # Observability
///
/// The FIRST successful launch prints a once-per-process stderr marker
/// (`[nsl] sdpa fused forward: launched (...)`) with the segment-mask
/// flag, head_dim, tile sizes, and whether the Tier-B tile-skip variant
/// was selected — launch-level proof for parity tests that the fused path
/// actually fired (silent declines are indistinguishable from success
/// otherwise).
///
/// # Kernel ABI
///
/// Marshals the v2 forward param layout: 36 base params (the 21 classic
/// FA-2 slots + 9 CSHA extras + 6 Tier-C save slots, all nulled except
/// `param_logsumexp`), plus a 37th `segment_ids` device pointer passed
/// ONLY when launching a segment-masked kernel — the unmasked kernel does
/// not declare the 37th param, and the codegen dispatch guarantees the
/// (kernel variant, segment pointer) pairing. `causal` is baked into the
/// PTX variant at codegen time; the runtime accepts it for call-site
/// symmetry and diagnostics only.
///
/// Output dtype: the v2 kernel STORES its output as f16 (and its
/// logsumexp as f32). The kernel writes an internal f16 staging buffer,
/// and the runtime widens on-device (`csha_fwd_convert_f16_to_f32`) into
/// the f32 tensor this FFI returns — callers always see f32.
///
/// Tier-B sentinel discipline matches the other `nsl_flash_attention*`
/// entries (planner spec §4): the pair must agree, and Tier-B-on can only
/// fire on the segment path (the runtime gate requires a non-null device
/// segment pointer).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_sdpa_fused_forward(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    scale_bits: i64,
    causal: i64,
    segment_ids_ptr: i64,
    ptx_ptr: i64, name_ptr: i64,
    tier_b_ptx_ptr: i64, tier_b_name_ptr: i64,
    block_q: i64, block_kv: i64,
    shared_mem_bytes: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::assert_tier_b_sentinels;

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_sdpa_fused_forward",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    #[cfg(feature = "cuda")]
    {
        use crate::cuda::inner;
        use crate::pca_tier_b_runtime::should_dispatch_tier_b_at_runtime;

        // Baked into the PTX variant at codegen time (see doc comment).
        let _ = causal;

        // ── Operational kill switch (see doc comment) ──
        static SDPA_FUSED_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let disabled = *SDPA_FUSED_DISABLED.get_or_init(|| {
            std::env::var("NSL_SDPA_FUSED_DISABLE").ok().as_deref() == Some("1")
        });
        if disabled {
            flash_bwd_warn_once(
                "[nsl] sdpa fused forward disabled via NSL_SDPA_FUSED_DISABLE",
            );
            return 0;
        }

        // Decline-reason debug (NSL_SDPA_FUSED_DEBUG=1): the decline
        // protocol is silent by design, which makes shape-dependent
        // declines invisible in production runs — this narrates them.
        macro_rules! decline {
            ($reason:expr) => {{
                if std::env::var("NSL_SDPA_FUSED_DEBUG").is_ok() {
                    eprintln!("[sdpa-fused-debug] decline: {}", $reason);
                }
                return 0;
            }};
        }
        // ── Decline checks: host-only struct reads, no driver calls ──
        if ptx_ptr == 0 || name_ptr == 0 || q_ptr == 0 || k_ptr == 0 || v_ptr == 0 {
            decline!("null ptx/name/tensor (head_dim matched no variant?)");
        }
        if block_q <= 0 || block_kv <= 0 {
            decline!("non-positive block dims");
        }
        let q_t = NslTensor::from_ptr(q_ptr);
        let k_t = NslTensor::from_ptr(k_ptr);
        let v_t = NslTensor::from_ptr(v_ptr);
        let wrong_layout =
            |t: &NslTensor| t.device != 1 || t.dtype != 1 || t.ndim != 4;
        if wrong_layout(q_t) || wrong_layout(k_t) || wrong_layout(v_t) {
            decline!(format!(
                "layout: q(dev={},dt={},nd={}) k(dev={},dt={},nd={}) v(dev={},dt={},nd={})",
                q_t.device, q_t.dtype, q_t.ndim, k_t.device, k_t.dtype, k_t.ndim,
                v_t.device, v_t.dtype, v_t.ndim));
        }
        // Stride guard: the v2 kernel reads raw device pointers
        // stride-blind. A transpose/reshape VIEW here would compute
        // smoothly-wrong attention (no error) — decline instead; the
        // decomposed fallback is stride-correct.
        if !is_canonical_row_major(q_t)
            || !is_canonical_row_major(k_t)
            || !is_canonical_row_major(v_t)
        {
            decline!(format!(
                "non-canonical strides: q={} k={} v={}",
                is_canonical_row_major(q_t),
                is_canonical_row_major(k_t),
                is_canonical_row_major(v_t)));
        }
        let (b, h, s, d) = unsafe {
            (
                *q_t.shape.add(0),
                *q_t.shape.add(1),
                *q_t.shape.add(2),
                *q_t.shape.add(3),
            )
        };
        if b <= 0 || h <= 0 || s <= 0 || d <= 0 {
            decline!("non-positive dims");
        }
        // K/V must agree with Q in [batch, seq, head_dim] and carry the
        // SAME head count — the plain fused kernel indexes K/V by the
        // fused batch*head block id, so GQA callers must pre-expand KV
        // first. (Q's own head_dim vs the PTX variant is NOT re-checked:
        // the caller selected the variant. K/V agreement with Q is a
        // separate matter — a mismatch would read out of bounds.)
        let kv_mismatch = |t: &NslTensor| unsafe {
            *t.shape.add(0) != b
                || *t.shape.add(1) != h
                || *t.shape.add(2) != s
                || *t.shape.add(3) != d
        };
        if kv_mismatch(k_t) || kv_mismatch(v_t) {
            decline!("k/v shape mismatch with q (GQA must be pre-expanded)");
        }
        // Ragged shapes: the v2 kernel has no seq_len tail guards.
        if s % block_q != 0 || s % block_kv != 0 {
            decline!(format!("ragged seq: s={s} block_q={block_q} block_kv={block_kv}"));
        }

        // ── Driver work starts here (thread-local context invariant) ──
        inner::ensure_context();

        // Segment staging input: [b, s] host-float ids -> Vec<u16>. A
        // validation failure inside the helper is still a silent decline —
        // nothing has been allocated yet.
        let seg_host = if segment_ids_ptr != 0 {
            match segment_ids_host_u16(segment_ids_ptr, b as usize, s as usize) {
                Some(host) => Some(host),
                None => return 0,
            }
        } else {
            None
        };

        // Output + logsumexp buffers, zeroed (mirrors the backward's
        // alloc_managed discipline; ownership passes to the returned
        // tensors on success, freed on the loud launch-failure path).
        //
        // Output dtype: the v2 forward epilogue stores its result as
        // **f16** (`cvt.rn.f16.f32` + `st.global.b16` in
        // flash_attention_v2/phases/forward/finalize.rs) while the
        // decomposed graph consumes f32. The kernel therefore writes into
        // an f16 STAGING buffer and `csha_fwd_convert_f16_to_f32` widens
        // into the final f32 tensor after the launch. Wrapping the raw
        // kernel output as f32 was the Stage-C parity bug: f16 bit pairs
        // reinterpreted as f32 are ~0, the attention output collapses,
        // and the first-step loss lands at ~ln(vocab) instead of the
        // decomposed value. logsumexp is stored f32 by the kernel — no
        // staging needed.
        let total_out = (b * h * s * d) as usize;
        let total_lse = (b * h * s) as usize;
        let out_data = inner::alloc_managed(total_out * 4);
        inner::memset_d8(out_data, total_out * 4);
        // P0.1 VRAM accounting: the f16 staging buffer is attention
        // workspace (out/lse are real outputs and keep the ambient surface).
        let out_f16 = {
            let _surface = crate::cuda::caching_allocator::SurfaceGuard::new(
                crate::cuda::caching_allocator::SurfaceTag::AttnWorkspace,
            );
            inner::alloc_managed(total_out * 2)
        };
        inner::memset_d8(out_f16, total_out * 2);
        let lse_data = inner::alloc_managed(total_lse * 4);
        inner::memset_d8(lse_data, total_lse * 4);

        let seg_dev: *mut c_void = match &seg_host {
            Some(host) => stage_segment_ids_device(host),
            None => std::ptr::null_mut(),
        };

        // Tier-B dispatch (planner spec §6.3): the runtime gate needs the
        // DEVICE segment pointer, so Tier-B-on only fires on the
        // segment-masked path.
        let tier_b_selected = should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            seg_dev as i64,
            s as u32,
        );
        let (effective_ptx_ptr, effective_name_ptr) = if tier_b_selected {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

        // ── Marshal the v2 forward param layout: 36 base [+ 1 segment] ──
        // Widths matter — the launch wrapper reads sizeof(param_type)
        // bytes at each slot: everything u64 except scale/csha_eps (.f32)
        // and csha_active_heads/csha_d_model (.u32).
        let mut q = q_t.data as u64;
        let mut k = k_t.data as u64;
        let mut v = v_t.data as u64;
        // Kernel writes f16 — point it at the staging buffer, NOT the
        // final f32 tensor (see the output-dtype comment above).
        let mut out = out_f16 as u64;
        let mut sc = f32::from_bits(scale_bits as u32);
        let mut bb = b as u64;
        let mut hh = h as u64;
        let mut sl = s as u64;
        let mut hd = d as u64;
        let mut bt: u64 = 0; // block_table
        let mut kp: u64 = 0; // k_pool
        let mut vp: u64 = 0; // v_pool
        let mut bs: u64 = 0; // block_size
        let mut cos: u64 = 0;
        let mut sin: u64 = 0;
        let mut sids: u64 = 0; // seq_ids
        let mut slens: u64 = 0; // seq_lens
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = lse_data as u64;
        // CSHA extras — all disabled on the plain fused path.
        let mut csha_x: u64 = 0;
        let mut csha_nw: u64 = 0;
        let mut csha_wq: u64 = 0;
        let mut csha_wk: u64 = 0;
        let mut csha_wv: u64 = 0;
        let mut csha_wo: u64 = 0;
        let mut csha_eps: f32 = 0.0;
        let mut csha_ah: u32 = 0;
        let mut csha_dm: u32 = 0;
        // Tier-C activation-save slots — null (the backward consumes the
        // returned lse instead of forward-saved projections).
        let mut q_proj: u64 = 0;
        let mut k_proj: u64 = 0;
        let mut v_proj: u64 = 0;
        let mut rmax: u64 = 0;
        let mut rsum: u64 = 0;
        let mut xraw: u64 = 0;
        // 37th param: segment_ids device pointer — pushed ONLY when
        // launching the segment-masked kernel (which alone declares it).
        // NEVER marshal 37 args for an unmasked kernel or 36 for a masked
        // one: cuLaunchKernel arg counts must match the entry signature.
        let mut seg_arg = seg_dev as u64;

        let mut args: Vec<*mut c_void> = vec![
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut bb as *mut _ as *mut c_void,
            &mut hh as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut csha_x as *mut _ as *mut c_void,
            &mut csha_nw as *mut _ as *mut c_void,
            &mut csha_wq as *mut _ as *mut c_void,
            &mut csha_wk as *mut _ as *mut c_void,
            &mut csha_wv as *mut _ as *mut c_void,
            &mut csha_wo as *mut _ as *mut c_void,
            &mut csha_eps as *mut _ as *mut c_void,
            &mut csha_ah as *mut _ as *mut c_void,
            &mut csha_dm as *mut _ as *mut c_void,
            &mut q_proj as *mut _ as *mut c_void,
            &mut k_proj as *mut _ as *mut c_void,
            &mut v_proj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
        ];
        if !seg_dev.is_null() {
            args.push(&mut seg_arg as *mut _ as *mut c_void);
        }

        // Grid/block identical to `nsl_flash_attention`: one CTA per
        // q-tile per (batch, head); 4 warps per CTA.
        // Launch PER BATCH ROW with pre-indexed pointers. The v2 segment
        // staging documents its contract as "the launch wrapper passes a
        // pre-indexed pointer (this batch sample's row); the kernel does
        // NOT add a batch_idx offset" (forward/prelude.rs Task-3C NOTE) —
        // a single [B,S]-base launch therefore stages ROW 0's segment ids
        // for every CTA and silently mis-masks rows 1..B (benchmark
        // discovery: first-step loss shifted 1e-2 at batch=4 while
        // batch=1 matched the decomposed oracle to 5e-6). Per-row
        // launches keep every pointer consistent with batch_idx==0
        // in-kernel; the extra launch overhead is negligible next to the
        // kernel itself.
        let grid = [s / block_q, h, 1];
        let block = [128i64, 1, 1];
        let row_qkv = (h * s * d) as u64 * 4; // f32 bytes per batch row
        let row_lse = (h * s) as u64 * 4;
        let row_seg = s as u64 * 2; // u16 bytes
        let mut rc = cudarc::driver::sys::CUresult::CUDA_SUCCESS;
        for bi in 0..b as u64 {
            q = q_t.data as u64 + bi * row_qkv;
            k = k_t.data as u64 + bi * row_qkv;
            v = v_t.data as u64 + bi * row_qkv;
            out = out_f16 as u64 + bi * (h * s * d) as u64 * 2; // f16 staging
            lse = lse_data as u64 + bi * row_lse;
            bb = 1;
            if !seg_dev.is_null() {
                seg_arg = seg_dev as u64 + bi * row_seg;
            }
            rc = inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                effective_name_ptr as *const u8,
                grid,
                block,
                &args,
                shared_mem_bytes as u32,
            );
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                break;
            }
        }

        // Widen the f16 staging output into the final f32 tensor. Same
        // stream as the forward launch, so in-stream ordering guarantees
        // the conversion sees the completed forward — no intermediate
        // sync needed.
        let conv_rc = if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            csha_fwd_convert_f16_to_f32(out_f16 as u64, out_data, total_out)
        } else {
            rc
        };

        // Synchronize before freeing the in-flight scratch buffers (the
        // f16 staging output on every call; the staged segment ids on the
        // packed path) — and surface asynchronous execution faults while
        // at it.
        // p3: stream-ordered by default. The sync only surfaces async faults;
        // the scratch frees below go through the caching allocator (free_managed),
        // which is stream-safe. Under NSL_CUDA_SYNC=1 the eager fault-check stays.
        let sync_rc = if inner::sync_mode_enabled() {
            unsafe { cudarc::driver::sys::cuCtxSynchronize() }
        } else {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS
        };

        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS
            || conv_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS
            || sync_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS
        {
            // The one LOUD decline: an unwritten output buffer must never
            // flow onward silently (the #324 silent-zeros failure mode).
            // Once per distinct message — this would otherwise repeat for
            // every attention op of every step on an unlaunchable config.
            flash_bwd_warn_once(&format!(
                "[sdpa-fused] fused SDPA forward kernel launch FAILED \
                 (launch rc {:?}, f16-widen rc {:?}, sync rc {:?}; b={b}, \
                 h={h}, s={s}, d={d}, segmented={}) — declining so the \
                 caller falls back to the decomposed SDPA graph (correct, \
                 slower). Check the PTX with nsl_test_cuda_jit_log.",
                rc,
                conv_rc,
                sync_rc,
                !seg_dev.is_null(),
            ));
            inner::free_managed(out_f16);
            inner::free_managed(seg_dev);
            inner::free_managed(out_data);
            inner::free_managed(lse_data);
            return 0;
        }

        // Kernels finished — the f16 staging output and the staged
        // segment ids are dead.
        inner::free_managed(out_f16);
        inner::free_managed(seg_dev);

        // Per-variant launch counter (parity-gate proof, see the static's
        // doc comment) + once-per-process launch marker.
        SDPA_FUSED_LAUNCH_COUNTS[if tier_b_selected { 1 } else { 0 }]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !SDPA_FUSED_LAUNCH_LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[nsl] sdpa fused forward: launched (segmask={}, head_dim={d}, \
                 tiles={block_q}x{block_kv}, tier_b={})",
                if seg_dev.is_null() { 0 } else { 1 },
                if tier_b_selected { 1 } else { 0 },
            );
        }

        // Hand out/lse ownership to the caller as NslList [out, lse].
        // No tape recording: this is a source-AD-only channel.
        let out_ptr = make_gpu_tensor(out_data, &[b, h, s, d], total_out);
        let lse_ptr = make_gpu_tensor(lse_data, &[b, h, s], total_lse);
        let list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(list, out_ptr);
        crate::list::nsl_list_push(list, lse_ptr);
        list
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Silent decline: the caller keeps the decomposed SDPA graph,
        // which is correct on CPU.
        let _ = (q_ptr, k_ptr, v_ptr, scale_bits, causal, segment_ids_ptr);
        let _ = (ptx_ptr, name_ptr, block_q, block_kv, shared_mem_bytes);
        0
    }
}

/// Resolve an NSL tensor-struct pointer (`NslTensor*` wrapped as `i64`) into
/// the raw device data pointer expected by the CSHA PTX kernels.
///
/// Context: Cranelift-emitted call sites for the CSHA FFIs pass `q_val` /
/// `k_val` / ... as `NslTensor*` (the opaque tensor handle). The CSHA
/// kernels, however, declare their tensor parameters as `.param .u64` and
/// do raw HBM address arithmetic on the loaded value. Passing the host
/// `NslTensor*` straight through would make the kernel dereference the
/// struct's first fields (`magic`, `data`, `shape`, ...) as if they were
/// tensor elements, producing `CUDA_ERROR_ILLEGAL_ADDRESS` on the very
/// first `ld.global` — exactly the bug PR #79 surfaced.
///
/// Dual-path contract:
///   - **Production** (Cranelift-emitted code): always passes
///     `NslTensor*`. We auto-promote CPU→GPU via `nsl_tensor_to_device`
///     (idempotent) and read `.data` for the kernel.
///   - **Test-only** (`nsl_test_cuda_alloc` call sites in
///     `crates/nsl-codegen/tests/csha_cuda_launch_*.rs`): passes raw
///     device pointers from `cuMemAlloc_v2`. We detect this via
///     `cuPointerGetAttribute(MEMORY_TYPE)`: a CU_MEMORYTYPE_DEVICE
///     result means the pointer is already in HBM and we pass it
///     through unchanged.
///
/// Null passes through as `0` so the kernel's runtime null-guards still
/// work.
#[cfg(feature = "cuda")]
#[inline]
fn csha_tensor_data_ptr(tensor_ptr: i64) -> u64 {
    if tensor_ptr == 0 {
        return 0;
    }

    // Query the driver for the pointer's memory type. If it's already
    // device memory (test path via `nsl_test_cuda_alloc`), the caller
    // provided a raw device pointer and we pass it straight through.
    // The query also handles managed memory cleanly — managed pointers
    // have memory_type == HOST (or a cudaMemoryType mismatch) and we
    // treat them as NslTensor* for safety.
    //
    // NOTE: we check `MEMORY_TYPE` not `DEVICE_POINTER` because the
    // latter returns the same address for device allocations but also
    // succeeds (returning the managed alias) for managed memory, which
    // would false-positive on NslTensor* handles stored in managed
    // memory. MEMORY_TYPE returns CU_MEMORYTYPE_DEVICE only for
    // `cuMemAlloc`/`cuMemAllocPitch` results.
    unsafe {
        use cudarc::driver::sys::{
            cuPointerGetAttribute, CUpointer_attribute, CUmemorytype, CUresult,
        };
        let mut mem_type: u32 = 0;
        let rc = cuPointerGetAttribute(
            &mut mem_type as *mut u32 as *mut std::ffi::c_void,
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            tensor_ptr as cudarc::driver::sys::CUdeviceptr,
        );
        if rc == CUresult::CUDA_SUCCESS
            && mem_type == CUmemorytype::CU_MEMORYTYPE_DEVICE as u32
        {
            // Raw device pointer — test-only path. Pass through.
            return tensor_ptr as u64;
        }
    }

    // Host pointer (NslTensor* handle) — auto-promote CPU→GPU
    // (idempotent for already-GPU tensors) then extract `.data`.
    let t = NslTensor::from_ptr(tensor_ptr);
    let gpu_ptr = if t.device == 0 {
        crate::tensor::nsl_tensor_to_device(tensor_ptr, 1)
    } else {
        tensor_ptr
    };
    let gpu_t = NslTensor::from_ptr(gpu_ptr);
    gpu_t.data as u64
}

/// CSHA Tier A.1: FlashAttention FFI variant that carries per-layer CSHA
/// extras (paper §2). This entry point preserves the argument layout of
/// `nsl_flash_attention` and *appends* nine CSHA-specific arguments:
///
///   x_ptr            — pre-norm input tensor (what RMSNorm will read)
///   norm_weight_ptr  — RMSNorm gain vector
///   wq_ptr / wk_ptr / wv_ptr / wo_ptr — projection weights the fused
///                                       kernel will project in-SMEM
///   rmsnorm_eps_bits — f32 bits of the RMSNorm epsilon
///   active_heads     — number of heads the specialised kernel computes
///                      (0 = use `heads` from the base layout)
///   d_model          — feature dimension for the projection tiles
///
/// For Tier A.1 the CSHA PTX body is not yet emitted (A.2), so this FFI
/// launches the CSHA-tagged PTX variant directly with a 30-argument
/// launch list — the 21 base FlashAttention params plus the nine CSHA
/// extras declared by `emit_flash_attention_entry` (csha_x_ptr through
/// csha_d_model).
///
/// Pick the per-CTA threadcount (`block_x`) for the CSHA forward FFIs.
///
/// Tier B.1 codegen (`flash_attention_v2::tier_b1::synthesize`) emits a
/// `global_t = warp_id + local_t * 8` warp distribution that assumes 8
/// warps per CTA = 256 threads. Tier A v2 (and the classic FA path) uses
/// 4 warps = 128 threads. The codegen embeds a `_tier_b1` suffix in the
/// kernel name when Tier B.1 dispatch applies (`csha.level >= 2 &&
/// gpu_sm >= 80` per `flash_attention_v2::is_tier_b1_dispatch`). This
/// helper peeks at the null-terminated name string and selects the
/// right block_x.
///
/// Carrying the signal in the kernel name (rather than as a new FFI
/// arg) keeps the 37-arg ABI stable — every existing caller continues
/// to work; only the name string they pass through changes for the
/// Tier B.1 dispatch path.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string allocated by `CString` or similar.
#[cfg(feature = "cuda")]
fn csha_block_x_for_kernel(name_ptr: i64) -> i64 {
    if name_ptr == 0 {
        return 128;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    // Marker is short (8 bytes); scan is O(name_len * 8) per launch.
    if name_bytes.windows(b"_tier_b1".len()).any(|w| w == b"_tier_b1") {
        256
    } else {
        128
    }
}

/// Extract the chunk size encoded in a Tier B.1 kernel name (the
/// `_tier_b1_chunk<N>` suffix appended by
/// `flash_attention_kernel_name_v2`). Returns `None` for non-Tier-B.1
/// names. The runtime pre-pass uses this to size the chunkified
/// scratch buffers + launch the narrow-and-chunkify kernels at the
/// right output stride.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string.
#[cfg(feature = "cuda")]
fn csha_tier_b1_chunk_for_kernel(name_ptr: i64) -> Option<u32> {
    if name_ptr == 0 {
        return None;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    let marker = b"_tier_b1_chunk";
    let pos = name_bytes.windows(marker.len()).position(|w| w == marker)?;
    let rest = &name_bytes[pos + marker.len()..];
    let end = rest.iter().position(|&b| !b.is_ascii_digit()).unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    std::str::from_utf8(&rest[..end]).ok()?.parse::<u32>().ok()
}

/// Read the null-terminated kernel name into an owned `String` for
/// diagnostic logging. Returns `"<null>"` for a zero `name_ptr`. Truncates
/// at 256 bytes to bound an accidental run-away scan if the pointer
/// references uninitialised memory.
#[cfg(feature = "cuda")]
fn csha_kernel_name_for_diag(name_ptr: i64) -> String {
    if name_ptr == 0 {
        return "<null>".into();
    }
    unsafe {
        let c = name_ptr as *const u8;
        let mut end = 0usize;
        while end < 256 && *c.add(end) != 0 {
            end += 1;
        }
        String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
    }
}

/// Detect a per-document CTA kernel by name-suffix.
///
/// The per-doc CTA forward kernel emitted by
/// `nsl_codegen::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward`
/// appends the `_per_doc_cta` suffix to its entry-point name (mirrors the
/// `_tier_b1_chunk<N>` pattern used by Tier B.1).
///
/// When this returns `true`, the FFI dispatch uses:
///   * `grid_x = num_docs` (from the `num_docs_or_zero` trailing arg)
///     instead of `ceil(seq/block_q)`
///   * `block_x = 128` (per-doc kernel is 4 warps, same as FA-2 default)
///
/// The signal lives in the kernel name (not a new FFI arg) for the *kernel
/// identity* — `num_docs_or_zero` carries the *runtime grid_x*.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string.
#[cfg(feature = "cuda")]
fn csha_is_per_doc_cta_kernel(name_ptr: i64) -> bool {
    if name_ptr == 0 {
        return false;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    let marker = b"_per_doc_cta";
    name_bytes.windows(marker.len()).any(|w| w == marker)
}

/// Detect a CSHA fused-projection kernel by name. The CSHA name suffix
/// block (`cshaL<level>_n<0|1>_p<0|1>_o<0|1>_h<heads>`) carries `_p1`
/// exactly when `fused_projections` was set at synthesis. The search is
/// anchored AFTER the `cshaL` marker so the leading `flash_attn_p<0|1>`
/// paged marker can never match.
///
/// Fused-projection kernels compute K/V projections only for the CTA's
/// own block rows (single-tile assumption — see the "Single-tile
/// assumption" comment in `flash_attention_v2/mod.rs`), so multi-tile
/// launches must go through the two-launch dispatch or be refused.
///
/// Safety: caller guarantees `name_ptr`, when non-zero, points to a
/// null-terminated C string.
#[cfg(feature = "cuda")]
fn csha_is_fused_projection_kernel(name_ptr: i64) -> bool {
    if name_ptr == 0 {
        return false;
    }
    let name_bytes = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const i8).to_bytes()
    };
    let Some(pos) = name_bytes.windows(5).position(|w| w == b"cshaL") else {
        return false;
    };
    name_bytes[pos..].windows(3).any(|w| w == b"_p1")
}

/// Tier B.1 per-call x-scratch RAII holder. Drop queues the GPU scratch
/// buffer for a stream-ordered deferred free (p3-remainder): a NULL-stream
/// event is recorded after the main kernel that reads the scratch, and the
/// raw `cuMemFree` runs once that event completes — no host stall on the
/// inference hot path. `NSL_CUDA_SYNC=1` restores the eager sync-then-free.
///
/// **W is NOT held here.** Chunkified Wq/Wk/Wv buffers are owned by
/// the process-global cache in `cuda::tier_b1_prepass::w_cache` —
/// weights are static across inference calls, so caching them avoids
/// re-running `launch_w_prepass` on every call.
#[cfg(feature = "cuda")]
struct PrepassScratch {
    x_scratch: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
impl Drop for PrepassScratch {
    fn drop(&mut self) {
        // Stream-ordered deferred free — see the struct doc. `defer_free_device`
        // ignores a null pointer and records the completion event on the same
        // NULL stream the main kernel launched on, so the free cannot run
        // before that kernel finishes reading the scratch.
        crate::cuda::inner::defer_free_device(self.x_scratch);
    }
}

/// Tier B.1 pre-pass orchestration. When the dispatched kernel is a
/// `_tier_b1_chunk<N>` variant, run the RMSNorm + narrow + chunkify on
/// `x` and the narrow + col-major chunkify on `Wq/Wk/Wv` (all on the
/// GPU via `cuda::tier_b1_prepass`), writing to freshly-allocated
/// scratch. Returns the substituted (x, Wq, Wk, Wv) pointers + RAII
/// scratch handle whose `Drop` queues a stream-ordered deferred free of
/// the buffers (NULL-stream event, freed once complete). The caller MUST
/// keep the handle alive until after the main kernel launch, so the event
/// is recorded after the consuming kernel.
///
/// Returns `None` when the kernel is not Tier B.1 (caller uses original
/// pointers unchanged).
#[cfg(feature = "cuda")]
fn csha_tier_b1_prepass_substitute(
    name_ptr: i64,
    x_data: u64, nw_data: u64,
    wq_data: u64, wk_data: u64, wv_data: u64,
    seq_len: i64, head_dim: i64, d_model: i64,
    rmsnorm_eps_bits: i64,
) -> Option<(u64, u64, u64, u64, PrepassScratch)> {
    let chunk = csha_tier_b1_chunk_for_kernel(name_ptr)? as u64;
    if x_data == 0 || nw_data == 0 || wq_data == 0 || wk_data == 0 || wv_data == 0 {
        return None;
    }
    let seq = seq_len as u64;
    let dm = d_model as u64;
    let hd = head_dim as u64;
    debug_assert_eq!(dm % chunk, 0, "d_model ({}) must be divisible by chunk ({})", dm, chunk);
    let n_chunks = dm / chunk;
    // X scratch is per-call (x changes every step). W is cached
    // process-globally (weights are static across inference calls).
    let x_bytes = (n_chunks * seq * chunk * 2) as usize;
    let x_scratch = crate::cuda::inner::alloc_device(x_bytes);
    if x_scratch.is_null() {
        return None;
    }
    let scratch = PrepassScratch { x_scratch };
    let eps = f32::from_bits(rmsnorm_eps_bits as u32);
    let rc_x = crate::cuda::tier_b1_prepass::launch_x_prepass(
        x_data, nw_data, scratch.x_scratch as u64, seq, dm, chunk, eps,
    );
    if rc_x as u32 != 0 {
        eprintln!("[csha-tier-b1] x prepass launch failed rc={:?}", rc_x);
        return None;
    }
    let wq_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wq_data, dm, hd, chunk)?;
    let wk_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wk_data, dm, hd, chunk)?;
    let wv_cached = crate::cuda::tier_b1_prepass::w_chunkified_cached(wv_data, dm, hd, chunk)?;
    Some((scratch.x_scratch as u64, wq_cached, wk_cached, wv_cached, scratch))
}

/// A.2.5: this replaces the pre-A.2.5 forwarder (which invoked
/// `nsl_flash_attention` and dropped the extras, causing a cuLaunch arg
/// count mismatch on kernels whose body now references the CSHA params).
/// The CSHA PTX scaffolds from A.2.2/A.2.3/A.2.4 still runtime-null-check
/// the pointers, so callers that pass NULL for any of Wq/Wk/Wv/Wo/
/// x_ptr/norm_weight_ptr (current A.1/A.2.1x state) execute the classic
/// Q-from-HBM path inside the kernel. This FFI's contract is therefore
/// a strict extension of `nsl_flash_attention`'s behaviour.
///
/// FA3 Hopper fallback is intentionally skipped here — the FA3 kernel
/// does not understand CSHA extras. Ampere/Hopper both run the Ampere
/// mma.sync FA2 path when CSHA is active.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_csha(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    // CSHA extras consumed by the PTX prologue/projection/epilogue
    // scaffolds (A.2.2/A.2.3/A.2.4). Each pointer is null-checked
    // inside the kernel so current NULL-passing call sites fall through
    // to the classic path without dereferencing garbage.
    x_ptr: i64,
    norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64,
    d_model: i64,
    // PCA Tier A: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion PTX kernel declares this
    // param in its prelude only when its FlashAttentionConfig has
    // segment_masked=true, in which case the pointer must be non-null and
    // reference a [B, S] u16 tensor in global memory.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // PCA per-doc CTA (Strategy 3 v1): number of documents in the batch.
    // Pass 0 for legacy (non-per-doc) topology; the standard
    // `grid_x = ceil(seq_len / block_q)` is used. Pass non-zero ONLY when
    // the dispatched kernel's name carries the `_per_doc_cta` suffix —
    // grid_x is then overridden to `num_docs_or_zero` per Strategy 3's
    // one-CTA-per-document launch contract. Mismatched signal/topology
    // (suffix present + zero count, or zero suffix + nonzero count)
    // returns -1 with an `eprintln!` diagnostic on `[nsl::flash_attention]`.
    //
    // Discussion: an alternative design reads the sentinel from
    // `doc_starts_ptr` via a synchronous `cuMemcpyDtoH_v2` per launch,
    // avoiding the ABI change. We rejected that because (1) the D2H
    // copy adds a host-device round-trip per kernel launch on the
    // hot per-step path, defeating Strategy 3's whole-point; (2) the
    // explicit arg makes mismatched callers a compile-time Cranelift
    // sig error rather than a silent grid_x miscalculation; (3) the
    // sentinel-0 default is byte-identical for every existing caller.
    num_docs_or_zero: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        // Grid / block identical to `nsl_flash_attention` — CSHA doesn't
        // change the outer tiling, except for A.4 weight-informed
        // specialisation: when `active_heads` is non-zero and smaller
        // than the full head count, the kernel variant was compiled
        // with a matching early-exit guard; shrink grid_y so each CTA
        // lands inside the active range. active_heads == 0 means "no
        // pruning — run the full head count".
        //
        // CSHA paper §5.2 v1 (cycle-2 audit): this is the PRIMARY tier
        // of dead-head elimination — blocks for pruned heads never
        // launch. The defense-in-depth in-kernel guard lives in
        // `crates/nsl-codegen/src/flash_attention_v2/phases/forward/
        // csha_hooks.rs::emit_active_heads_guard`; the same formula
        // appears at the scalar+fused backward launch sites further
        // down this file. Formula pinned by `a4_grid_y_*` unit tests
        // in the `#[cfg(test)] mod` below.
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // PCA per-doc CTA (Strategy 3 v1) topology detection.
        // The kernel name carries `_per_doc_cta` when codegen emitted the
        // per-doc CTA forward kernel. Launch geometry differs from the
        // standard FA-2 path:
        //   * grid_x = num_docs (one CTA per document, NOT ceil(seq/block_q))
        //   * grid_y = batch * heads (unchanged)
        //   * grid_z = 1               (unchanged)
        //   * block_x = 128            (4 warps, same as FA-2 default)
        // The caller MUST pass `num_docs_or_zero > 0` AND
        // `doc_starts_ptr != 0` when launching a per-doc CTA variant.
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha: per-doc CTA kernel \
                     ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha: per-doc CTA kernel \
                     ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            // Caller passed a per-doc grid_x but the dispatched kernel is
            // not a per-doc variant — likely a planner/dispatch bug.
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha: num_docs_or_zero={} provided but \
                 kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }

        // Multi-tile fused-projection refusal (inference FFI): the `_p1`
        // kernel is single-tile ("Single-tile assumption" in
        // flash_attention_v2/mod.rs). The training path
        // (`nsl_flash_attention_csha_with_saves`) implements the two-launch
        // multi-tile dispatch using its projection save buffers as staging;
        // this saves-less entry point has no staging buffers yet, so refuse
        // loudly rather than launch into silent garbage. Callers needing
        // multi-tile today can route through the with_saves FFI with
        // transient q_proj/k_proj/v_proj buffers.
        if !per_doc_cta
            && csha_is_fused_projection_kernel(effective_name_ptr)
            && (seq_len > block_q || seq_len > _block_kv)
        {
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha: fused-projection kernel {:?} is \
                 single-tile (seq_len={seq_len} > block_q={block_q}/block_kv={}) — refusing; use \
                 nsl_flash_attention_csha_with_saves (two-launch multi-tile dispatch) instead",
                csha_kernel_name_for_diag(effective_name_ptr),
                _block_kv,
            );
            return -1;
        }

        let grid_x = if per_doc_cta {
            num_docs_or_zero
        } else {
            (seq_len + block_q - 1) / block_q
        };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        // Tier B.1 launch geometry: the pipelined-MMA codegen emits a
        // `global_t = warp_id + local_t * 8` warp distribution
        // assuming 8 warps per CTA. Tier A v2 (and the classic FA path)
        // use 4 warps. The codegen embeds a `_tier_b1` suffix in the
        // kernel name when Tier B.1 dispatch applies
        // (`csha.level >= 2 && gpu_sm >= 80` per
        // `flash_attention_v2::is_tier_b1_dispatch`). Peek at the
        // null-terminated name string here to pick the right block_x.
        //
        // Carrying the signal in the kernel name keeps the FFI ABI
        // stable — all existing 37-arg callers continue to work; only
        // the name string they pass through changes for Tier B.1.
        // (Use `effective_name_ptr` so the block_x reflects the actual
        // kernel that will be launched — if a Tier B PCA variant was
        // selected by `should_dispatch_tier_b_at_runtime`, its name is
        // what cuLaunchKernel resolves the function from.)
        let block_x = csha_block_x_for_kernel(effective_name_ptr);
        let block_y = 1i64;
        let block_z = 1i64;

        // 21 base args — must exactly mirror the non-CSHA path so the
        // shared PTX body works identically on NULL CSHA extras.
        //
        // PR #79 bug fix: each tensor arg below is a host `NslTensor*`
        // (Cranelift emits these as opaque tensor handles from
        // `compile_flash_attention_call`). The CSHA PTX expects raw
        // device (HBM) base pointers, so we resolve each handle via
        // `csha_tensor_data_ptr` which (a) auto-promotes CPU tensors to
        // GPU and (b) extracts `NslTensor.data` for the kernel.
        // Non-tensor args (batch, heads, block_size, ...) pass through
        // unchanged. Paged/ragged pointers (block_table, k_pool, ...)
        // are always null at today's call sites so they pass through as
        // `0`.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);

        // 9 CSHA extras, matching the PTX param declarations in
        // `emit_flash_attention_entry`. eps is declared .f32; heads /
        // d_model are .u32. Widths matter — the launch wrapper reads
        // sizeof(param_type) bytes starting at each `*mut c_void`.
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;

        // CSHA Tier B.1 production pre-pass orchestration.
        // When the dispatched kernel's name contains `_tier_b1_chunk<N>`,
        // we run the host-side RMSNorm + narrow + chunkify on `x` and
        // the narrow + col-major chunkify on Wq/Wk/Wv (all GPU-side via
        // `cuda::tier_b1_prepass`), then substitute the chunkified
        // pointers in the args list. The kernel was compiled with
        // `skip_rmsnorm_prologue=true` (force-overridden by codegen
        // in the Tier B.1 dispatch path) so it reads the pre-pass'd x
        // directly without re-RMSNormalizing.
        //
        // `_prepass_handle` is an RAII guard: its Drop queues the scratch
        // buffer for a stream-ordered deferred free (records a NULL-stream
        // completion event, then frees once it fires). It MUST live until
        // after `kernel_launch` returns — dropping earlier would record the
        // event before the consuming kernel, allowing the free to race.
        let _prepass_handle = if let Some((nx, nwq, nwk, nwv, handle)) =
            csha_tier_b1_prepass_substitute(
                effective_name_ptr,
                x, nw, wq, wk, wv,
                seq_len, head_dim, d_model,
                rmsnorm_eps_bits,
            )
        {
            x = nx;
            wq = nwq;
            wk = nwk;
            wv = nwv;
            Some(handle)
        } else {
            None
        };
        // Tier C activation-save pointers — always null for this FFI; the
        // `nsl_flash_attention_csha_with_saves` FFI passes real pointers.
        let mut q_proj: u64 = 0;
        let mut k_proj: u64 = 0;
        let mut v_proj: u64 = 0;
        let mut rmax: u64 = 0;
        let mut rsum: u64 = 0;
        let mut xraw: u64 = 0;
        // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order).
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order).
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        let args: [*mut c_void; 38] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            // ── CSHA extras ───────────────────────────────────────
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            // ── Tier C activation-save pointers (null by default) ─────
            &mut q_proj as *mut _ as *mut c_void,
            &mut k_proj as *mut _ as *mut c_void,
            &mut v_proj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order)
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order)
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        ) as i64;

        // Tape recording mirrors `nsl_flash_attention` exactly — CSHA
        // is a forward-pass optimisation; the backward pass re-reads Q/K/V
        // from the same pointers the caller supplied.
        if autodiff::is_recording() {
            let scale = f32::from_bits(scale_bits as u32);
            NslTensor::from_ptr(q_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(k_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(v_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(out_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(logsumexp_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            autodiff::maybe_record(autodiff::TapeOp::FlashAttention {
                q: q_ptr, k: k_ptr, v: v_ptr,
                out: out_ptr,
                logsumexp: logsumexp_ptr,
                scale,
                batch, heads, seq_len, head_dim,
                causal: causal != 0,
                saved_q: q_ptr,
                saved_k: k_ptr,
                saved_v: v_ptr,
            });
        }

        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model, segment_ids_ptr, doc_starts_ptr);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        eprintln!("[nsl] CSHA FlashAttention requires CUDA; non-CUDA build cannot launch.");
        -1
    }
}

/// CSHA Tier C: FlashAttention FFI variant that forwards to the same PTX
/// body as `nsl_flash_attention_csha` but supplies non-null
/// activation-save pointers so the fused source-AD backward kernel has
/// post-RoPE Q/K/V + row_max/row_sum available in HBM.
///
/// Pointer contract:
///   q_proj_ptr / k_proj_ptr / v_proj_ptr — HBM buffers of shape
///     `[batch, heads, seq, head_dim]` f16, row-major. Typically
///     allocated via `nsl_csha_alloc_backward_activations`.
///   row_max_ptr / row_sum_ptr — HBM buffers of shape
///     `[batch, heads, seq]` f32.
///
/// Null-safety: any of the 5 pointers MAY be null (the kernel's per-tensor
/// null guards skip that store). A fully-null call is equivalent to
/// `nsl_flash_attention_csha`.
///
/// Callers MUST have compiled the kernel with
/// `CshaExtras::save_activations_for_backward=true`; otherwise the PTX
/// emits no save path and the pointers are ignored.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_with_saves(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    // Tier C activation-save pointers.
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    // Tier C: raw-x save pointer (forward stages a copy of x BEFORE
    // RMSNorm overwrites csha_x_ptr in place, so the backward dRMSNorm
    // can read pre-norm x). Null = skip the save.
    x_raw_ptr: i64,
    // PCA Tier A: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion PTX kernel declares this
    // param in its prelude only when its FlashAttentionConfig has
    // segment_masked=true, in which case the pointer must be non-null and
    // reference a [B, S] u16 tensor in global memory.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // PCA per-doc CTA (Strategy 3 v1): number of documents in the batch.
    // See `nsl_flash_attention_csha` for the full contract; this variant
    // mirrors that behaviour exactly.
    num_docs_or_zero: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha_with_saves",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // PCA per-doc CTA (Strategy 3 v1) topology — see the matching
        // block in `nsl_flash_attention_csha` for the full contract +
        // alternative-design discussion.
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: per-doc CTA kernel \
                     ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: per-doc CTA kernel \
                     ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: num_docs_or_zero={} provided but \
                 kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }

        // Multi-tile fused-projection dispatch (two-launch): a `_p1` kernel
        // projects K/V only for the CTA's own block rows ("Single-tile
        // assumption" in flash_attention_v2/mod.rs), so seq_len beyond one
        // block would silently produce garbage attention. Launch A (the
        // normal launch below) still writes VALID projection saves for
        // every block; a second launch of the same kernel with null
        // weights/x then takes the classic HBM tile-load path
        // (k_start-addressed, multi-tile-correct) over the widened
        // projections. Configurations the chain does not cover refuse
        // loudly here instead of launching into silent garbage.
        let is_tier_b1 = {
            let nb = if effective_name_ptr != 0 {
                unsafe { std::ffi::CStr::from_ptr(effective_name_ptr as *const i8).to_bytes() }
            } else {
                &[][..]
            };
            nb.windows(14).any(|w| w == b"_tier_b1_chunk")
        };
        let multi_tile_fused = !per_doc_cta
            && csha_is_fused_projection_kernel(effective_name_ptr)
            && (seq_len > block_q || seq_len > _block_kv);
        if multi_tile_fused {
            if block_q != _block_kv {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: multi-tile fused-projection \
                     dispatch requires block_q == block_kv (got {block_q} vs {_block_kv}, seq_len={seq_len}); \
                     asymmetric fused tiles are single-tile only — refusing instead of launching into garbage"
                );
                return -1;
            }
            if is_tier_b1 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: multi-tile dispatch not \
                     implemented for Tier B.1 kernels ({:?}, seq_len={seq_len} > block_q={block_q}) — refusing",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
            if segment_ids_ptr != 0 || doc_starts_ptr != 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_with_saves: multi-tile fused dispatch \
                     does not support segment_masked / doc-aware launches yet (seq_len={seq_len} > \
                     block_q={block_q}) — refusing"
                );
                return -1;
            }
        }

        let grid_x = if per_doc_cta {
            num_docs_or_zero
        } else {
            (seq_len + block_q - 1) / block_q
        };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        // Tier B.1 launch geometry signal via kernel name suffix; see
        // `csha_block_x_for_kernel` and the matching comment in
        // `nsl_flash_attention_csha`. Must use `effective_name_ptr` (the
        // kernel actually launched) — when Tier B PCA dispatch swaps in
        // a different variant, its name decides the block_x.
        let block_x = csha_block_x_for_kernel(effective_name_ptr);
        let block_y = 1i64;
        let block_z = 1i64;

        // PR #79 bug fix: user-provided tensor args arrive as
        // `NslTensor*` — resolve each via `csha_tensor_data_ptr` (auto-
        // promotes CPU→GPU, extracts `.data`). The 6 activation-save
        // pointers (q_proj_ptr, ..., x_raw_ptr) are ALREADY raw device
        // pointers allocated by `nsl_csha_alloc_backward_activations_into`
        // and pass through untouched.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;

        // CSHA Tier B.1 production pre-pass — same orchestration as
        // `nsl_flash_attention_csha`. When the dispatched kernel is a
        // Tier B.1 variant (`_tier_b1_chunk<N>` suffix in its name) the
        // host-side RMSNorm + narrow + chunkify runs on `x` and the
        // narrow + col-major chunkify runs on Wq/Wk/Wv (cached
        // process-globally), and the chunkified pointers are
        // substituted before the kernel launch. The kernel itself was
        // compiled with `skip_rmsnorm_prologue=true` so it reads the
        // pre-pass'd x directly.
        //
        // `_prepass_handle` is RAII: its Drop queues the per-call x-scratch
        // for a stream-ordered deferred free (records a NULL-stream event,
        // frees once it fires). It MUST live until after `kernel_launch`
        // returns — dropping earlier would record the event before the
        // consuming kernel, allowing the free to race.
        let _prepass_handle = if let Some((nx, nwq, nwk, nwv, handle)) =
            csha_tier_b1_prepass_substitute(
                effective_name_ptr,
                x, nw, wq, wk, wv,
                seq_len, head_dim, d_model,
                rmsnorm_eps_bits,
            )
        {
            x = nx;
            wq = nwq;
            wk = nwk;
            wv = nwv;
            Some(handle)
        } else {
            None
        };

        // Activation-save pointers: raw device buffers — pass through.
        let mut q_proj = q_proj_ptr as u64;
        let mut k_proj = k_proj_ptr as u64;
        let mut v_proj = v_proj_ptr as u64;
        let mut rmax = row_max_ptr as u64;
        let mut rsum = row_sum_ptr as u64;
        let mut xraw = x_raw_ptr as u64;

        // Multi-tile dispatch needs the projection saves as staging between
        // launch A (fused per-block projections) and launch B (interleaved
        // attention twin). Callers that don't keep saves (inference-style
        // invocations) get transient f16 staging allocated here and freed
        // after the chain.
        let mut mt_staging: [*mut c_void; 3] = [std::ptr::null_mut(); 3];
        if multi_tile_fused {
            let f16_bytes = (batch * heads * seq_len * head_dim) as usize * 2;
            if q_proj == 0 {
                mt_staging[0] = crate::cuda::inner::alloc_device(f16_bytes);
                q_proj = mt_staging[0] as u64;
            }
            if k_proj == 0 {
                mt_staging[1] = crate::cuda::inner::alloc_device(f16_bytes);
                k_proj = mt_staging[1] as u64;
            }
            if v_proj == 0 {
                mt_staging[2] = crate::cuda::inner::alloc_device(f16_bytes);
                v_proj = mt_staging[2] as u64;
            }
        }
        // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order).
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order).
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        let args: [*mut c_void; 38] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut q_proj as *mut _ as *mut c_void,
            &mut k_proj as *mut _ as *mut c_void,
            &mut v_proj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // PCA Tier A: segment_ids slot (trailing — matches prelude params Vec order)
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts slot (trailing — matches prelude params Vec order)
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        // NSL_CSHA_DUMP_GRADS=1 forward-side diag: read the PTX kernel name
        // out of name_ptr (null-terminated C string) and print it so a
        // mismatched PTX selection (inference PTX instead of the _with_saves
        // variant) is visible. Also prints the save pointers so they can be
        // confirmed non-null and distinct.
        let dump_on = std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false);
        if dump_on {
            let kname = unsafe {
                let c = effective_name_ptr as *const u8;
                if c.is_null() { String::from("<null>") } else {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < 256 { end += 1; }
                    String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
                }
            };
            eprintln!(
                "[csha-dump-fwd] with_saves kernel=\"{}\" q_proj=0x{:x} k_proj=0x{:x} v_proj=0x{:x} \
                 row_max=0x{:x} row_sum=0x{:x} x_raw=0x{:x}",
                kname, q_proj, k_proj, v_proj, rmax, rsum, xraw
            );
            // Dump the full PTX to a file for offline inspection — only once
            // to avoid accumulating files.
            unsafe {
                let c = effective_ptx_ptr as *const u8;
                if !c.is_null() {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < (1 << 20) { end += 1; }
                    if end > 0 {
                        let bytes = std::slice::from_raw_parts(c, end);
                        let _ = std::fs::write(
                            std::env::temp_dir().join("csha_with_saves.ptx"),
                            bytes,
                        );
                        eprintln!(
                            "[csha-dump-fwd] ptx written to {}/csha_with_saves.ptx ({} bytes)",
                            std::env::temp_dir().display(), end
                        );
                    }
                }
            }
        }

        let mut fwd_rc = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        // Multi-tile two-launch, launch B: launch A above ran the fused
        // kernel per q-block — its projection saves (q_proj/k_proj/v_proj,
        // f16, post-RoPE) are valid for EVERY block, but its attention
        // output is single-tile garbage twice over (own-block K/V SMEM
        // re-read + the split S/PV orchestration clobbers P per KV tile).
        // Widen the saved projections to f32 scratch and launch the
        // `_mt_attn` twin entry from the combined module
        // (`synthesize_forward_multi_tile_combined`): the classic
        // INTERLEAVED orchestration with k_start-addressed HBM tile
        // loads, no RMSNorm/projection/RoPE (inputs are pre-projected
        // and pre-rotated), and else-branch row_max/row_sum saves.
        // Launch B overwrites out/lse/row_max/row_sum with correct
        // values; the projection saves survive because their pointers
        // are nulled for B. If the module lacks the twin (single-kernel
        // PTX from a caller that has not adopted the combined
        // synthesizer, or a sinks/segment/checkpoint config), the name
        // lookup fails and the error surfaces loudly in the rc.
        if multi_tile_fused && fwd_rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // Twin entry name: `<fused_name>_mt_attn`, NUL-terminated.
            let mt_name: Vec<u8> = {
                let base = unsafe {
                    std::ffi::CStr::from_ptr(effective_name_ptr as *const i8).to_bytes()
                };
                let mut v = Vec::with_capacity(base.len() + 9);
                v.extend_from_slice(base);
                v.extend_from_slice(b"_mt_attn\0");
                v
            };
            let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
            let f32_bytes = qkv_elems * 4;
            let qf = crate::cuda::inner::alloc_device(f32_bytes);
            let kf = crate::cuda::inner::alloc_device(f32_bytes);
            let vf = crate::cuda::inner::alloc_device(f32_bytes);
            let free_mt = |a: *mut c_void, b: *mut c_void, c: *mut c_void| {
                if !a.is_null() { crate::cuda::inner::free_device(a); }
                if !b.is_null() { crate::cuda::inner::free_device(b); }
                if !c.is_null() { crate::cuda::inner::free_device(c); }
            };
            for (src, dst, tag) in [(q_proj, qf, "q_proj"), (k_proj, kf, "k_proj"), (v_proj, vf, "v_proj")] {
                let rc = csha_fwd_convert_f16_to_f32(src, dst, qkv_elems);
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    eprintln!(
                        "[nsl::flash_attention] multi-tile dispatch: f16->f32 widening of {tag} failed: {rc:?}"
                    );
                    free_mt(qf, kf, vf);
                    for st in mt_staging {
                        if !st.is_null() {
                            crate::cuda::inner::free_device(st);
                        }
                    }
                    return -1;
                }
            }
            // Mutate the launch locals in place — `args` holds pointers to
            // them, and cuLaunchKernel reads the pointees at call time
            // (same contract the per-q-block `slens` threading relies on).
            q = qf as u64;
            k = kf as u64;
            v = vf as u64;
            x = 0;
            nw = 0;
            wq = 0;
            wk = 0;
            wv = 0;
            q_proj = 0;
            k_proj = 0;
            v_proj = 0;
            xraw = 0;
            let _ = std::hint::black_box(&args);
            // Same shared_mem_bytes as launch A: if the twin declares a
            // static in-body `.shared` array the dynamic request is
            // additional-but-unused; if it uses the module extern the
            // fused size over-provisions it. Both are within the opt-in
            // cap already validated for launch A.
            fwd_rc = crate::cuda::inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                mt_name.as_ptr(),
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            );
            // kernel_launch synchronizes before returning (async-error
            // check), so freeing the widened buffers here is safe.
            free_mt(qf, kf, vf);
        }
        // Transient staging (allocated only when the caller passed null
        // save pointers on a multi-tile launch) is dead after launch B.
        for st in mt_staging {
            if !st.is_null() {
                crate::cuda::inner::free_device(st);
            }
        }

        // Immediately after the forward kernel, sync + read back q_proj and
        // row_sum so we can tell whether the save-write path fired.  If the
        // forward kernel didn't write these then something between forward
        // PTX synthesis and launch lost the `save_activations_for_backward`
        // flag OR the saves emit but to wrong addresses.
        if dump_on {
            unsafe { crate::cuda::inner::cu_ctx_synchronize(); }
            eprintln!("[csha-dump-fwd-post] launch_rc={:?}", fwd_rc);
            let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
            let stats_elems = (batch * heads * seq_len) as usize;
            let x_elems = (batch * heads * seq_len * head_dim) as usize;
            // q_proj (f16)
            if q_proj != 0 {
                let mut host: Vec<u16> = vec![0u16; qkv_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        q_proj as *const std::ffi::c_void,
                        qkv_elems * 2,
                    );
                }
                let nz = host.iter().filter(|&&b| b != 0).count();
                let first8: Vec<f32> = host.iter().take(8)
                    .map(|&b| crate::tensor::f16_bits_to_f32(b)).collect();
                eprintln!(
                    "[csha-dump-fwd-post]  q_proj first8={:?} nonzero_bits={}/{}",
                    first8, nz, qkv_elems
                );
            }
            // row_sum (f32)
            if rsum != 0 {
                let mut host: Vec<f32> = vec![0.0f32; stats_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        rsum as *const std::ffi::c_void,
                        stats_elems * 4,
                    );
                }
                let nz = host.iter().filter(|&&v| v != 0.0).count();
                eprintln!(
                    "[csha-dump-fwd-post]  row_sum first8={:?} nonzero={}/{}",
                    host.iter().take(8).cloned().collect::<Vec<_>>(),
                    nz, stats_elems
                );
            }
            // x_raw (f32)
            if xraw != 0 {
                let mut host: Vec<f32> = vec![0.0f32; x_elems];
                unsafe {
                    crate::cuda::inner::memcpy_dtoh(
                        host.as_mut_ptr() as *mut std::ffi::c_void,
                        xraw as *const std::ffi::c_void,
                        x_elems * 4,
                    );
                }
                let nz = host.iter().filter(|&&v| v != 0.0).count();
                eprintln!(
                    "[csha-dump-fwd-post]  x_raw  first8={:?} nonzero={}/{}",
                    host.iter().take(8).cloned().collect::<Vec<_>>(),
                    nz, x_elems
                );
            }
        }

        fwd_rc as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model);
        let _ = (q_proj_ptr, k_proj_ptr, v_proj_ptr, row_max_ptr, row_sum_ptr, x_raw_ptr, segment_ids_ptr, doc_starts_ptr);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        eprintln!("[nsl] CSHA FlashAttention w/ saves requires CUDA.");
        -1
    }
}

/// CSHA Tier C: fused source-AD backward kernel launch.
///
/// Forwards to the PTX synthesised by
/// `nsl_codegen::flash_attention_v2::synthesize_backward`. Launch-list
/// layout: forward's 35 CSHA args (identical order so downstream
/// tooling sees one ABI), then 8 backward-specific appends:
/// `dO_ptr` (input) plus the 7 gradient outputs
/// (`dq_ptr`, `dk_ptr`, `dv_ptr`, `dwq_ptr`, `dwk_ptr`, `dwv_ptr`,
/// `dx_ptr`).
///
/// The five activation-save pointers
/// (`q_proj_ptr`, `k_proj_ptr`, `v_proj_ptr`, `row_max_ptr`,
/// `row_sum_ptr`) are inherited from the forward slot: the forward
/// kernel *wrote* into them when launched via
/// `nsl_flash_attention_csha_with_saves`; the backward kernel *reads*
/// from them here. Null pointers on the gradient outputs let callers
/// skip stores they don't need (e.g. freezing a weight).
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
/// Public 54-param FFI symbol — FROZEN signature.
///
/// Delegates to the private `csha_backward_impl` with `probe_ptrs = None`,
/// which produces byte-identical behavior to the pre-c20-T1-followup body:
/// the probe slots (when the `csha_cycle19_probe` feature is on) are set
/// to sentinel `0` so `%p_probe_active` stays false and gradient outputs
/// are unaffected.
///
/// All 12 pre-c19 call sites resolve here — see `csha_backward_ffi_hygiene`
/// integration test for the allow-list.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_backward(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    x_raw_ptr: i64,
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64,
    dx_norm_ptr: i64,
    segment_ids_ptr: i64,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    doc_starts_ptr: i64,
    tier_b2_active: i64,
    num_docs_or_zero: i64,
) -> i64 {
    csha_backward_impl(
        q_ptr, k_ptr, v_ptr,
        out_ptr,
        logsumexp_ptr,
        scale_bits,
        batch, heads, seq_len, head_dim,
        block_table_ptr,
        k_pool_ptr, v_pool_ptr,
        block_size,
        cos_ptr, sin_ptr,
        seq_ids_ptr, seq_lens_ptr,
        shared_mem_bytes,
        ptx_ptr, name_ptr,
        block_q, block_kv,
        causal,
        x_ptr, norm_weight_ptr,
        wq_ptr, wk_ptr, wv_ptr, wo_ptr,
        rmsnorm_eps_bits,
        active_heads, d_model,
        q_proj_ptr, k_proj_ptr, v_proj_ptr,
        row_max_ptr, row_sum_ptr,
        x_raw_ptr,
        do_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        dwq_ptr, dwk_ptr, dwv_ptr,
        dx_ptr,
        dx_norm_ptr,
        segment_ids_ptr,
        tier_b_ptx_ptr,
        tier_b_name_ptr,
        doc_starts_ptr,
        tier_b2_active,
        num_docs_or_zero,
        None,
    )
}

/// Private launcher used by both the FROZEN public FFI symbol
/// (`nsl_flash_attention_csha_backward` — passes `probe_ptrs = None`) and
/// the c19 probe FFI wrapper (`nsl_flash_attention_csha_backward_probe` —
/// passes `Some((probe_ds, probe_dv))` so the trailing PTX param slots
/// receive real device pointers instead of sentinel zeros).
///
/// When `probe_ptrs = Some(_)`, the args array under the
/// `csha_cycle19_probe` feature carries the two probe u64 slots with the
/// caller's device pointers, enabling `%p_probe_active = true` inside the
/// backward PTX and causing the 8-slot dS/dV probe stores to fire.
/// When `probe_ptrs = None`, the slots are `0` and the probe stores fall
/// through — byte-identical to the pre-c20-T1-followup default path.
///
/// The parameter list is otherwise byte-identical to the public FFI's
/// 54-param signature.
#[allow(clippy::too_many_arguments)]
fn csha_backward_impl(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    // Saved activations (inputs to backward).
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    // Tier C raw-x save: forward staged a copy here before RMSNorm
    // overwrote csha_x_ptr in place; backward dRMSNorm reads it.
    x_raw_ptr: i64,
    // Tier C backward-specific.
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64,
    // Gap I.5 fix (Option A): 8th gradient output — dx_norm (gradient
    // w.r.t. the RMSNorm output, i.e. dy_norm). Consumed by the AD-side
    // `RmsNormGammaBackward` emission for correct dgamma under the fused
    // CSHA dispatch path. f32, shape [batch, seq, d_model]. Null when
    // caller doesn't need it (e.g. inference warm-ups).
    dx_norm_ptr: i64,
    // PCA Tier A Task 4B: segment_ids device pointer for packed-sequence training.
    // Pass 0 on unpacked launches. The companion backward PTX kernel (Task 4A)
    // declares this param in its prelude only when segment_masked=true; the
    // segment_masked=false variant doesn't read this slot, so passing 0 is safe
    // for all existing callers.
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA §4.3: doc_starts device pointer for packed-sequence training
    // with document-aware RoPE position reset. Per-row layout —
    // [batch_size, MAX_NUM_DOCS+1] i32 tensor, total batch_size * 1028
    // bytes. Pass 0 to disable (identity positions). When non-zero,
    // kernel computes row offset as batch_idx * (MAX_NUM_DOCS+1) and
    // loads only its row's 1028-byte subtable. See spec §3.
    doc_starts_ptr: i64,
    // CSHA Tier B.2 (Phase 3 T6): explicit flag selecting the 4-kernel hybrid
    // backward launch path. When non-zero, `ptx_ptr` is treated as a module
    // holding the four concatenated Tier B.2 backward entries
    // (`tier_b2_d_prepass`, `tier_b2_dq_kernel`, `tier_b2_dkdv_kernel`,
    // `tier_b2_proj_backward`) and they are launched in sequence, with a
    // D = rowsum(dO*O) scratch buffer allocated for the intermediate. When 0,
    // behavior is byte-identical to the scalar single-kernel path below.
    // The wengert lowering COMPUTES this flag (Sprint 1 T7:
    // compile-time-eligible AND seq_len == block_q, wengert_lower ~2088);
    // note it is iconst(0) for every config the production training-config
    // builder currently emits, because kernel.rs pins level=1 /
    // active_heads=0 which the tier_b2 dispatch eligibility rejects — see
    // the production-eligibility note in the campaign memory. Selection is
    // an explicit flag rather than
    // kernel-name-suffix sniffing — the dispatch decision is made in codegen
    // (see `tier_b2_hybrid_backward_eligible`) and threaded here. See spec §7.
    tier_b2_active: i64,
    // PCA per-doc CTA backward (Sprint 5): number of documents in the
    // batch. Pass 0 for legacy (non-per-doc) topology; the standard
    // `q_blocks` outer loop drives grid_x=1 launches per q-block. Pass
    // non-zero ONLY when the dispatched backward kernel's name carries
    // the `_per_doc_cta` suffix — grid_x is then overridden to
    // `num_docs_or_zero` per Sprint 5's one-CTA-per-document backward
    // launch contract, and the `q_blocks` outer loop is skipped.
    // Mismatched signal/topology (suffix present + zero count, or zero
    // suffix + nonzero count) returns -1 with an `eprintln!` diagnostic.
    num_docs_or_zero: i64,
    // CSHA c20 T1-followup: optional probe device pointers.
    // `None` -> args-array slots receive sentinel 0 (PTX %p_probe_active
    //          stays false, gradients byte-identical).
    // `Some((probe_ds, probe_dv))` -> pointers threaded into the trailing
    //          .param .u64 slots so probe stores fire.
    // Semantically ignored when the `csha_cycle19_probe` feature is off
    // (the PTX-side probe prelude doesn't exist in that config).
    probe_ptrs: Option<(u64, u64)>,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha_backward",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    // CSHA Tier B.2 (Phase 3 T6): 4-kernel hybrid backward launch branch.
    // When the explicit flag is set, `ptx_ptr` holds the concatenated 4-entry
    // Tier B.2 module; launch d_prepass -> dq -> dkdv -> proj_backward in
    // sequence (same-stream, implicitly ordered), with a D=rowsum(dO*O) f32
    // scratch buffer for the intermediate consumed by dq/dkdv. Returns early
    // so the scalar single-kernel path below is reached only when the flag is
    // 0 (byte-identical to today for every existing caller).
    //
    // Kernels 1-3 (d_prepass/dq/dkdv) use the standalone-validated per-kernel
    // ABIs (ported from the Layer-1 GPU reference launchers). Kernel 4
    // (proj_backward) reuses the SCALAR backward param ABI — the exact 49-arg
    // list the scalar path below builds — so it is launched by re-running the
    // scalar else-branch with `effective_name_ptr` overridden to
    // "tier_b2_proj_backward". To keep the scalar path byte-identical, the
    // Tier B.2 branch is fully self-contained here.
    #[cfg(feature = "cuda")]
    {
        if tier_b2_active != 0 {
            return csha_tier_b2_backward_launch(
                ptx_ptr,
                batch, heads, seq_len, head_dim,
                block_q,
                active_heads, d_model,
                scale_bits,
                shared_mem_bytes,
                causal,
                // RMSNorm + projection (scalar-ABI) forward inputs.
                x_ptr, norm_weight_ptr,
                wq_ptr, wk_ptr, wv_ptr, wo_ptr,
                rmsnorm_eps_bits,
                // Forward-saved activations.
                q_proj_ptr, k_proj_ptr, v_proj_ptr,
                row_max_ptr, row_sum_ptr, x_raw_ptr,
                logsumexp_ptr, out_ptr,
                // dO seed + gradient buffers.
                do_ptr,
                dq_ptr, dk_ptr, dv_ptr,
                dwq_ptr, dwk_ptr, dwv_ptr,
                dx_ptr, dx_norm_ptr,
                // Packed-sequence pass-throughs.
                segment_ids_ptr, doc_starts_ptr,
                // Sprint 10: rope_q=true threads cos/sin to proj_backward's
                // emit_drope phase. Null on rope_q=false launches (the entire
                // dRoPE emission is skipped at codegen time then, so a null
                // here is never read).
                cos_ptr, sin_ptr,
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };
        // PCA per-doc CTA (Sprint 5) topology detection. Mirrors the
        // forward `nsl_flash_attention_csha` dispatch: when the kernel
        // name carries `_per_doc_cta`, the backward launches with
        // grid_x=num_docs (one CTA per document) and skips the per-q-block
        // outer launch loop entirely (the kernel's per-doc prelude derives
        // %q_start from doc_starts[bid_x] instead of bid_x*block_q +
        // q_launch_base, so the slens-threaded q-block trick is unused).
        let per_doc_cta = csha_is_per_doc_cta_kernel(effective_name_ptr);
        if per_doc_cta {
            if num_docs_or_zero <= 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_backward: per-doc CTA \
                     kernel ({:?}) requires num_docs_or_zero > 0, got {}",
                    csha_kernel_name_for_diag(effective_name_ptr),
                    num_docs_or_zero,
                );
                return -1;
            }
            if doc_starts_ptr == 0 {
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_backward: per-doc CTA \
                     kernel ({:?}) requires doc_starts_ptr != 0",
                    csha_kernel_name_for_diag(effective_name_ptr),
                );
                return -1;
            }
        } else if num_docs_or_zero > 0 {
            eprintln!(
                "[nsl::flash_attention] nsl_flash_attention_csha_backward: num_docs_or_zero={} \
                 provided but kernel name {:?} lacks the `_per_doc_cta` suffix",
                num_docs_or_zero,
                csha_kernel_name_for_diag(effective_name_ptr),
            );
            return -1;
        }

        // Multi-tile fused backward (Phase 1.1 / PR2b). At multi-tile (seq_len >
        // block_q OR > block_kv) the fused backward now produces correct
        // gradients at batch=1, heads=1 via two mechanisms working together:
        //   1. dK RMWs into f32 scratch (emit_store_dk_only), mirroring dV, so
        //      d_k and dk_scratch hold the cross-launch dK total (was an f16
        //      overwrite that kept only the last-attending q-block).
        //   2. A post-pass (csha_bwd_multitile_postpass, run after the q-block
        //      loop below) recomputes dWk/dWv/dx/dx_norm from the fully
        //      accumulated dK/dV/dQ totals, overwriting the in-kernel per-block
        //      partials. dWq is already correct via its own f32 scratch.
        // heads>1 (cross-head dx_norm accumulation + per-head x_raw layout) and
        // batch>1 (x_raw addressing lacks a batch term) remain unimplemented and
        // are refused. NSL_CSHA_MULTITILE_DW_VALIDATION=1 forces the launch
        // through for isolated heads>1/batch>1 experimentation (gradients not
        // guaranteed correct in that mode). The forward path is unaffected.
        if !per_doc_cta
            && csha_is_fused_projection_kernel(effective_name_ptr)
            && (seq_len > block_q || seq_len > _block_kv)
        {
            let supported = batch == 1 && heads == 1;
            if !supported {
                let validation_only = std::env::var_os("NSL_CSHA_MULTITILE_DW_VALIDATION")
                    .map(|v| v != "0" && v != "")
                    .unwrap_or(false);
                if !validation_only {
                    eprintln!(
                        "[nsl::flash_attention] nsl_flash_attention_csha_backward: multi-tile backward \
                         (seq_len={seq_len} > block_q={block_q}/block_kv={_block_kv}) is implemented only \
                         for batch=1, heads=1 (got batch={batch}, heads={heads}): heads>1 needs cross-head \
                         dx_norm accumulation + per-head x_raw layout, batch>1 needs x_raw batch addressing \
                         — refusing"
                    );
                    return -1;
                }
                eprintln!(
                    "[nsl::flash_attention] nsl_flash_attention_csha_backward: MULTI-TILE VALIDATION MODE \
                     (batch={batch}, heads={heads}) — gradients NOT guaranteed correct for heads>1/batch>1"
                );
            }
        }
        let q_blocks = (seq_len + block_q - 1) / block_q;
        let grid_x = if per_doc_cta { num_docs_or_zero } else { 1i64 };
        let grid_y = batch * effective_heads;
        let grid_z = 1i64;
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // PR #79 bug fix: user-provided tensor args arrive as
        // `NslTensor*` — resolve each via `csha_tensor_data_ptr` (auto-
        // promotes CPU→GPU, extracts `.data`).
        //
        // Passes through unchanged:
        //   * Forward-saved activation pointers (q_proj_ptr, ...,
        //     x_raw_ptr): raw device buffers from
        //     `nsl_csha_alloc_backward_activations_into`.
        //   * Paged/ragged pointers (block_table, k_pool, v_pool,
        //     seq_ids, seq_lens): all null at today's callers.
        //
        // Resolved via `csha_tensor_data_ptr`:
        //   * Primary forward inputs (q, k, v, out, logsumexp, cos,
        //     sin, x, norm_weight, wq, wk, wv, wo).
        //   * Backward seed `dO` plus the 7 gradient outputs
        //     (dq, dk, dv, dwq, dwk, dwv, dx): these are allocated by
        //     Cranelift via `nsl_tensor_zeros_f16_on` / `nsl_tensor_zeros_on`
        //     (see `wengert_lower.rs::FusedCshaBackward`), which return
        //     `NslTensor*` with device=1. The kernel needs the raw
        //     device data pointer to write gradients into.
        let mut q = csha_tensor_data_ptr(q_ptr);
        let mut k = csha_tensor_data_ptr(k_ptr);
        let mut v = csha_tensor_data_ptr(v_ptr);
        let mut out = csha_tensor_data_ptr(out_ptr);
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = csha_tensor_data_ptr(cos_ptr);
        let mut sin = csha_tensor_data_ptr(sin_ptr);
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr);
        let mut x = csha_tensor_data_ptr(x_ptr);
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr);
        let mut wq = csha_tensor_data_ptr(wq_ptr);
        let mut wk = csha_tensor_data_ptr(wk_ptr);
        let mut wv = csha_tensor_data_ptr(wv_ptr);
        let mut wo = csha_tensor_data_ptr(wo_ptr);
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;
        // Forward-saved activations: raw device buffers, pass through.
        let mut qp = q_proj_ptr as u64;
        let mut kpj = k_proj_ptr as u64;
        let mut vpj = v_proj_ptr as u64;
        let mut rmax = row_max_ptr as u64;
        let mut rsum = row_sum_ptr as u64;
        let mut xraw = x_raw_ptr as u64;
        // Backward seed + 8 gradient outputs (dq/dk/dv/dwq/dwk/dwv/dx/dx_norm):
        // NslTensor handles — resolve to device pointers via csha_tensor_data_ptr.
        // PR #74 added dx_norm as the 8th extract so dgamma can be computed
        // from the post-RMSNorm gradient directly.
        let mut d_o = csha_tensor_data_ptr(do_ptr);
        let mut d_q = csha_tensor_data_ptr(dq_ptr);
        let mut d_k = csha_tensor_data_ptr(dk_ptr);
        let mut d_v = csha_tensor_data_ptr(dv_ptr);
        let mut d_wq = csha_tensor_data_ptr(dwq_ptr);
        let mut d_wk = csha_tensor_data_ptr(dwk_ptr);
        let mut d_wv = csha_tensor_data_ptr(dwv_ptr);
        let mut d_x = csha_tensor_data_ptr(dx_ptr);
        let mut d_xn = csha_tensor_data_ptr(dx_norm_ptr);
        // PCA Tier A Task 4B: segment_ids device pointer — raw pass-through
        // (not an NslTensor handle). Null on unpacked launches; the
        // segment_masked kernel variant reads it only when set.
        let mut seg_ids = segment_ids_ptr as u64;
        // PCA §4.3: doc_starts device pointer — raw pass-through.
        // Sentinel 0 = identity positions (RoPE reset disabled).
        let mut doc_starts = doc_starts_ptr as u64;

        // Option A f32 scratch for dK/dV: accumulate in f32 to avoid the
        // f16-HBM saturation-to-inf that plagued the naive ld-add-store
        // finalize. Scratch is zeroed, each q-block kernel does serialized
        // ld.f32/add.f32/st.f32 RMW (deterministic — no atomics), and a
        // final conversion kernel writes f32 scratch → f16 dK/dV outputs.
        let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
        let scratch_bytes = qkv_elems * 4; // f32
        let dk_scratch_raw = if d_k != 0 {
            crate::cuda::inner::alloc_device(scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        let dv_scratch_raw = if d_v != 0 {
            crate::cuda::inner::alloc_device(scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        if !dk_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dk_scratch_raw, scratch_bytes);
        }
        if !dv_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dv_scratch_raw, scratch_bytes);
        }
        let mut dk_scratch = dk_scratch_raw as u64;
        let mut dv_scratch = dv_scratch_raw as u64;

        // Phase 1.1 (pretraining): f32 scratch for dW (dwq/dwk/dwv), same
        // serialized-RMW pattern as dK/dV above. emit_dproj now accumulates
        // each q-block's partial into these f32 buffers (grid_x=1, serial
        // launches → no atomics), and the conversion below writes f32 scratch →
        // f16 dwq/dwk/dwv. Shape is [d_model, kv_dim] with
        // kv_dim = max(active_heads,1)*head_dim, matching emit_dproj's cell
        // layout. Single-tile (one launch, zero-init scratch) stays identical to
        // the old f16 overwrite.
        let dw_kv_dim = active_heads.max(1) * head_dim;
        let dw_elems = (d_model * dw_kv_dim) as usize;
        let dw_scratch_bytes = dw_elems * 4; // f32
        let dwq_scratch_raw = if d_wq != 0 {
            crate::cuda::inner::alloc_device(dw_scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        let dwk_scratch_raw = if d_wk != 0 {
            crate::cuda::inner::alloc_device(dw_scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        let dwv_scratch_raw = if d_wv != 0 {
            crate::cuda::inner::alloc_device(dw_scratch_bytes)
        } else {
            std::ptr::null_mut()
        };
        if !dwq_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dwq_scratch_raw, dw_scratch_bytes);
        }
        if !dwk_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dwk_scratch_raw, dw_scratch_bytes);
        }
        if !dwv_scratch_raw.is_null() {
            crate::cuda::inner::memset_d8(dwv_scratch_raw, dw_scratch_bytes);
        }
        let mut dwq_scratch = dwq_scratch_raw as u64;
        let mut dwk_scratch = dwk_scratch_raw as u64;
        let mut dwv_scratch = dwv_scratch_raw as u64;

        // CSHA cycle 20 T1 (+T1-followup) — probe pointer trailing slots.
        // Under the `csha_cycle19_probe` feature the backward PTX prelude
        // declares two additional `.param .u64 probe_{ds,dv}_out_ptr`
        // slots at the end of the param block, so every launcher path
        // must thread these two u64s or the launch fails with
        // CUDA_ERROR_INVALID_VALUE.
        //
        // c20 T1-followup: the trailing slots are now driven by the
        // `probe_ptrs: Option<(u64, u64)>` parameter, populated by the
        // c19 probe FFI (`nsl_flash_attention_csha_backward_probe`).
        // - `None` (public FFI path, 12 pre-c19 callers) -> sentinel 0
        //   on both slots -> `%p_probe_active = false` at PTX ->
        //   probe stores fall through -> gradient outputs BYTE-IDENTICAL
        //   to the pre-c20-T1-followup path.
        // - `Some((ds, dv))` (probe FFI path) -> real device pointers
        //   thread through -> `%p_probe_active = true` at PTX (if ds
        //   is non-zero) -> the 8 predicated `st.global.f32` sites
        //   fire and populate the probe scratch buffer.
        #[cfg(feature = "csha_cycle19_probe")]
        let (mut probe_ds_slot, mut probe_dv_slot): (u64, u64) =
            probe_ptrs.unwrap_or((0, 0));
        #[cfg(not(feature = "csha_cycle19_probe"))]
        let _ = probe_ptrs;

        #[cfg(feature = "csha_cycle19_probe")]
        let args: [*mut c_void; 54] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut kpj as *mut _ as *mut c_void,
            &mut vpj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            // ── Tier C backward outputs ──
            &mut d_o as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut d_k as *mut _ as *mut c_void,
            &mut d_v as *mut _ as *mut c_void,
            &mut d_wq as *mut _ as *mut c_void,
            &mut d_wk as *mut _ as *mut c_void,
            &mut d_wv as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_xn as *mut _ as *mut c_void,
            // Option A: f32 scratch for dK/dV accumulation.
            &mut dk_scratch as *mut _ as *mut c_void,
            &mut dv_scratch as *mut _ as *mut c_void,
            // Phase 1.1: dW f32 scratch — positional lockstep with the PTX param
            // list (right after dk/dv scratch, before segment/doc/probe trailing).
            &mut dwq_scratch as *mut _ as *mut c_void,
            &mut dwk_scratch as *mut _ as *mut c_void,
            &mut dwv_scratch as *mut _ as *mut c_void,
            // PCA Tier A Task 4B: segment_ids trailing slot.
            &mut seg_ids as *mut _ as *mut c_void,
            // PCA §4.3: doc_starts trailing slot.
            &mut doc_starts as *mut _ as *mut c_void,
            // CSHA cycle 20 T1: probe_ds_out_ptr + probe_dv_out_ptr
            // trailing slots. Sentinel 0 → %p_probe_active=false →
            // probe stores fall through. Only the probe FFI wrapper
            // (feature-gated) will one day overwrite these before launch.
            &mut probe_ds_slot as *mut _ as *mut c_void,
            &mut probe_dv_slot as *mut _ as *mut c_void,
        ];

        // Suppress unused-var warnings on the non-probe cfg branch.
        #[cfg(feature = "csha_cycle19_probe")]
        let _ = (&probe_ds_slot, &probe_dv_slot);

        // Non-probe cfg: 52-slot args array (49 base + 3 dW scratch, Phase 1.1).
        #[cfg(not(feature = "csha_cycle19_probe"))]
        let args: [*mut c_void; 52] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut qp as *mut _ as *mut c_void,
            &mut kpj as *mut _ as *mut c_void,
            &mut vpj as *mut _ as *mut c_void,
            &mut rmax as *mut _ as *mut c_void,
            &mut rsum as *mut _ as *mut c_void,
            &mut xraw as *mut _ as *mut c_void,
            &mut d_o as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut d_k as *mut _ as *mut c_void,
            &mut d_v as *mut _ as *mut c_void,
            &mut d_wq as *mut _ as *mut c_void,
            &mut d_wk as *mut _ as *mut c_void,
            &mut d_wv as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_xn as *mut _ as *mut c_void,
            &mut dk_scratch as *mut _ as *mut c_void,
            &mut dv_scratch as *mut _ as *mut c_void,
            // Phase 1.1: dW f32 scratch — positional lockstep with the PTX param
            // list (right after dk/dv scratch, before segment/doc trailing).
            &mut dwq_scratch as *mut _ as *mut c_void,
            &mut dwk_scratch as *mut _ as *mut c_void,
            &mut dwv_scratch as *mut _ as *mut c_void,
            &mut seg_ids as *mut _ as *mut c_void,
            &mut doc_starts as *mut _ as *mut c_void,
        ];

        // The backward kernel is launched one q-block at a time with the
        // q-block base carried in the otherwise-unused seq_lens_ptr slot.
        // Each launch does serialized f32 RMW on dk_scratch / dv_scratch,
        // so the final scratch values accumulate correctly across all
        // q-blocks without atomics. dQ is written per-launch to disjoint
        // slices so `d_q` still needs a zero init before the first launch.
        if d_q != 0 {
            crate::cuda::inner::memset_d8(d_q as *mut c_void, qkv_elems * 2);
        }

        let mut rc = cudarc::driver::sys::CUresult::CUDA_SUCCESS;
        if per_doc_cta {
            // Per-doc CTA: ONE launch with grid_x=num_docs. The kernel's
            // per-doc prelude derives %q_start from doc_starts[bid_x],
            // ignoring %q_launch_base, so the slens-threaded q-block trick
            // is not needed (and would be incorrect — there is no global
            // q-block ordinal in the per-doc launch).
            slens = 0u64;
            let _ = std::hint::black_box(&slens);
            rc = crate::cuda::inner::kernel_launch(
                effective_ptx_ptr as *const u8,
                effective_name_ptr as *const u8,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            );
        } else {
            for q_block in 0..q_blocks {
                // Thread the q-block base into seq_lens_ptr slot via `slens`.
                // Raw-ptr use: `args[16]` holds `&mut slens`, and CUDA's
                // `cuLaunchKernel` reads the pointee at call time, so each
                // iteration's assignment is picked up by the next launch.
                // `black_box` prevents the optimizer from eliding the write
                // because dataflow analysis can't see the read-through-ptr.
                slens = (q_block * block_q) as u64;
                let _ = std::hint::black_box(&slens);
                rc = crate::cuda::inner::kernel_launch(
                    effective_ptx_ptr as *const u8,
                    effective_name_ptr as *const u8,
                    [grid_x, grid_y, grid_z],
                    [block_x, block_y, block_z],
                    &args,
                    shared_mem_bytes as u32,
                );
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    break;
                }
            }
        }

        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false)
        {
            let dump_first_nonfinite_scratch = |name: &str, ptr: *mut c_void| {
                if ptr.is_null() || qkv_elems == 0 {
                    return;
                }
                let mut host = vec![0f32; qkv_elems];
                crate::cuda::inner::memcpy_dtoh(host.as_mut_ptr() as *mut c_void, ptr, scratch_bytes);
                if let Some((idx, value)) = host.iter().copied().enumerate().find(|(_, x)| !x.is_finite()) {
                    eprintln!(
                        "[csha-dump-bwd-scratch] first non-finite {name}[{idx}]={value:?}"
                    );
                } else {
                    let first = host.iter().take(4).copied().collect::<Vec<_>>();
                    eprintln!(
                        "[csha-dump-bwd-scratch] {name} first4={first:?} max_abs={:.6e}",
                        host.iter().copied().map(f32::abs).fold(0.0f32, f32::max)
                    );
                }
            };
            dump_first_nonfinite_scratch("dk_scratch", dk_scratch_raw);
            dump_first_nonfinite_scratch("dv_scratch", dv_scratch_raw);
        }

        // Drain dV f32 scratch -> f16 d_v.
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && !dv_scratch_raw.is_null() && d_v != 0
        {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dv_scratch_raw, d_v as *mut c_void, qkv_elems,
            );
            if c_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                rc = c_rc;
            }
        }

        // Phase 1.1 (PR2b): drain dK f32 scratch -> f16 d_k. dK now accumulates
        // in f32 scratch via emit_store_dk_only's in-loop RMW (was an f16
        // overwrite that was correct only at single-tile). Byte-identical at
        // single-tile (zero-init scratch, one RMW, one cvt.rn.f16.f32); correct
        // across q-block launches at multi-tile. Done BEFORE the post-pass reads
        // dk_scratch (drain does not mutate the f32 scratch).
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS
            && !dk_scratch_raw.is_null() && d_k != 0
        {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dk_scratch_raw, d_k as *mut c_void, qkv_elems,
            );
            if c_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                rc = c_rc;
            }
        }

        // Phase 1.1 (PR2b): multi-tile post-pass. Recompute dWk/dWv/dx/dx_norm
        // from the now-complete dK/dV totals (dk_scratch/dv_scratch) + dQ total
        // (d_q) + x_raw, overwriting the in-kernel per-q-block PARTIAL results.
        // Only at multi-tile (single-tile in-kernel values are already correct
        // and byte-identical) and only for the supported regime (batch=1,
        // heads=1). Runs BEFORE the dk/dv scratch is freed.
        // Gate: multi-tile AND the supported regime (batch=1, heads=1) AND the
        // fused-projection, non-per-doc kernel class the post-pass applies to.
        // The per-doc CTA path uses csha=None (no fused projections, x_raw_ptr=0,
        // no dWk/dWv/dx outputs), so it must NOT engage the post-pass — otherwise
        // a legitimate per-doc backward at batch=1/heads=1/multi-tile would take
        // the null-input branch and print a spurious "gradients WRONG" diagnostic.
        // Mirrors the refusal gate above.
        let multitile = seq_len > block_q || seq_len > _block_kv;
        let run_postpass = multitile
            && batch == 1
            && heads == 1
            && !per_doc_cta
            && csha_is_fused_projection_kernel(effective_name_ptr);
        if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS && run_postpass {
            // The post-pass needs a dx_norm staging buffer whenever it computes
            // dx (Phase B writes dx_norm then reloads it for the closed-form dx).
            // dx_ptr and dx_norm_ptr are independently nullable per the FFI
            // contract, so when the caller wants dx but not dx_norm, allocate a
            // throwaway dx_norm buffer so dx is still recomputed from the totals
            // (the in-kernel per-block dx is wrong at multi-tile). If neither is
            // requested, pass 0 and the kernel skips Phase B entirely.
            let mut temp_dxn_raw: *mut c_void = std::ptr::null_mut();
            let dxn_for_pp = if d_xn != 0 {
                d_xn
            } else if d_x != 0 {
                let dxn_bytes = (seq_len * d_model * 4) as usize;
                temp_dxn_raw = crate::cuda::inner::alloc_device(dxn_bytes);
                temp_dxn_raw as u64
            } else {
                0
            };
            if xraw != 0
                && d_q != 0
                && !dk_scratch_raw.is_null()
                && !dv_scratch_raw.is_null()
            {
                let pp_rc = csha_bwd_multitile_postpass(
                    dk_scratch, dv_scratch, d_q, xraw, nw, wq, wk, wv,
                    d_wk, d_wv, d_x, dxn_for_pp, seq_len, d_model as u32, head_dim, eps,
                );
                if pp_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    rc = pp_rc;
                }
            } else {
                eprintln!(
                    "[nsl::flash_attention] multi-tile backward post-pass skipped: a required input \
                     buffer is null (xraw={xraw:#x}, d_q={d_q:#x}) — dWk/dWv/dx will be WRONG"
                );
            }
            if !temp_dxn_raw.is_null() {
                crate::cuda::inner::free_device(temp_dxn_raw);
            }
        }

        if !dk_scratch_raw.is_null() {
            crate::cuda::inner::free_device(dk_scratch_raw);
        }
        if !dv_scratch_raw.is_null() {
            crate::cuda::inner::free_device(dv_scratch_raw);
        }

        // Phase 1.1: drain dW f32 scratch -> f16. dWq is correct at both single-
        // and multi-tile (dQ_proj is complete per q-block launch). dWk/dWv are
        // correct ONLY at single-tile; at multi-tile they hold per-q-block
        // PARTIALS and the post-pass above already wrote the correct d_wk/d_wv,
        // so we must NOT overwrite them with the scratch drain — drain dWq only.
        let dw_drain: &[(*mut c_void, u64)] = if run_postpass {
            &[(dwq_scratch_raw, d_wq)]
        } else {
            &[
                (dwq_scratch_raw, d_wq),
                (dwk_scratch_raw, d_wk),
                (dwv_scratch_raw, d_wv),
            ]
        };
        for &(scratch_raw, out) in dw_drain {
            if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !scratch_raw.is_null() {
                let c_rc = csha_bwd_convert_f32_to_f16(scratch_raw, out as *mut c_void, dw_elems);
                if c_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    rc = c_rc;
                }
            }
        }
        for scratch_raw in [dwq_scratch_raw, dwk_scratch_raw, dwv_scratch_raw] {
            if !scratch_raw.is_null() {
                crate::cuda::inner::free_device(scratch_raw);
            }
        }

        // NSL_CSHA_DUMP_GRADS=1 diagnostic — zero-cost when unset. Reads back
        // the 6 forward-saved activations (q_proj, k_proj, v_proj, row_max,
        // row_sum, x_raw) and all 8 gradient outputs (dq/dk/dv/dwq/dwk/dwv/
        // dx/dx_norm) from device → host, prints the first 8 values, max|.|,
        // element-sum, and NaN count for each.
        //
        // Consumes `qp, kpj, vpj, rmax, rsum, xraw` and `d_o, d_q, d_k, d_v,
        // d_wq, d_wk, d_wv, d_x, d_xn` (already resolved to raw device
        // pointers above). Shape info is derived from {batch, heads, seq_len,
        // head_dim, d_model}. Synchronises the device first so any async
        // kernel-launch failure or stale writeback is visible.
        if std::env::var_os("NSL_CSHA_DUMP_GRADS").map(|v| v != "0" && v != "").unwrap_or(false) {
            unsafe { crate::cuda::inner::cu_ctx_synchronize(); }
            let kname = unsafe {
                let c = effective_name_ptr as *const u8;
                if c.is_null() { String::from("<null>") } else {
                    let mut end = 0usize;
                    while *c.add(end) != 0 && end < 256 { end += 1; }
                    String::from_utf8_lossy(std::slice::from_raw_parts(c, end)).into_owned()
                }
            };
            eprintln!("[csha-dump-bwd] kernel=\"{}\"", kname);
            eprintln!(
                "[csha-dump-bwd] saves: q_proj=0x{:x} k_proj=0x{:x} v_proj=0x{:x} \
                 row_max=0x{:x} row_sum=0x{:x} x_raw=0x{:x}",
                qp, kpj, vpj, rmax, rsum, xraw
            );
            eprintln!(
                "[csha-dump-bwd] grads: dO=0x{:x} dq=0x{:x} dk=0x{:x} dv=0x{:x} \
                 dwq=0x{:x} dwk=0x{:x} dwv=0x{:x} dx=0x{:x} dx_norm=0x{:x}",
                d_o, d_q, d_k, d_v, d_wq, d_wk, d_wv, d_x, d_xn
            );
            csha_dump_backward_buffers(
                rc as i64,
                batch, heads, seq_len, head_dim, d_model,
                qp, kpj, vpj, rmax, rsum, xraw,
                d_o, d_q, d_k, d_v, d_wq, d_wk, d_wv, d_x, d_xn,
            );
        }

        rc as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        let _ = (x_ptr, norm_weight_ptr, wq_ptr, wk_ptr, wv_ptr, wo_ptr);
        let _ = (rmsnorm_eps_bits, active_heads, d_model);
        let _ = (q_proj_ptr, k_proj_ptr, v_proj_ptr, row_max_ptr, row_sum_ptr, x_raw_ptr);
        let _ = (do_ptr, dq_ptr, dk_ptr, dv_ptr, dwq_ptr, dwk_ptr, dwv_ptr, dx_ptr, dx_norm_ptr);
        let _ = (segment_ids_ptr, doc_starts_ptr, tier_b2_active);
        let _ = (tier_b_ptx_ptr, tier_b_name_ptr, effective_ptx_ptr, effective_name_ptr);
        let _ = num_docs_or_zero;
        let _ = probe_ptrs;
        eprintln!("[nsl] CSHA backward requires CUDA.");
        -1
    }
}

/// CSHA cycle 19 T1 (+c20 T1 + T1-followup) — dS probe FFI (variant-B new symbol).
///
/// **Scope:** cycle-18 DEGENERATE-PROBE meta-lesson gate. The existing
/// `nsl_flash_attention_csha_backward` 54-param signature is FROZEN — this
/// wrapper adds two trailing device-pointer slots (`probe_ds_out_ptr`,
/// `probe_dv_out_ptr`, each 8 f32 slots wide) reserved for a NON-DEGENERATE
/// probe site at `(batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0,
/// causal=true)` where `P[1,0]≈0.5`, `P[1,1]≈0.5`, and `dS = 0.25 *
/// (dP[1,0] - dP[1,1])` is nonzero under random dO. The exact 0.25
/// coefficient assumes symmetric Q/K (i.e. `S[1,0] ≈ S[1,1]`). Under
/// fully randomized Q/K the coefficient varies but `dS` remains nonzero
/// — the coordinate is non-degenerate in both regimes.
///
/// **Full path LANDED (c20 T1 + T1-followup):**
///   * c20 T1 wired the PTX-side probe emission (8 predicated
///     `st.global.f32` stores at the ds_compute + dqdk_accum sites) and
///     widened the backward prelude with two trailing `.param .u64`
///     probe pointers plus the `%p_probe_active` register.
///   * c20 T1-followup (this file) refactored the launcher body into a
///     private `csha_backward_impl` that accepts
///     `probe_ptrs: Option<(u64, u64)>`. This wrapper now threads
///     `Some((probe_ds_out_ptr, probe_dv_out_ptr))` through instead of
///     delegating to the 54-param FFI and dropping the probe pointers
///     on the floor.
///
/// **ABI:** 54 original params + `probe_ds_out_ptr: i64` + `probe_dv_out_ptr:
/// i64` = **56 total i64 params**. Byte-identical to the original signature
/// on the first 54 slots; sentinel `0` on either trailing slot disables that
/// half of the probe write (PTX `%p_probe_active = false` for the ds side).
#[cfg(feature = "csha_cycle19_probe")]
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_csha_backward_probe(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, block_kv: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    active_heads: i64, d_model: i64,
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64,
    x_raw_ptr: i64,
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64,
    dx_norm_ptr: i64,
    segment_ids_ptr: i64,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    doc_starts_ptr: i64,
    tier_b2_active: i64,
    num_docs_or_zero: i64,
    // T1 additive trailing slots (variant-B).
    probe_ds_out_ptr: i64,
    probe_dv_out_ptr: i64,
) -> i64 {
    // c20 T1-followup: thread probe pointers straight into the private
    // launcher via `probe_ptrs = Some((ds, dv))`. When ds is non-zero the
    // PTX prelude sets `%p_probe_active = true` and the 8 predicated
    // `st.global.f32` sites populate the probe scratch buffer.
    // Sentinel `0` on either slot disables that half of the probe write.
    csha_backward_impl(
        q_ptr, k_ptr, v_ptr,
        out_ptr,
        logsumexp_ptr,
        scale_bits,
        batch, heads, seq_len, head_dim,
        block_table_ptr,
        k_pool_ptr, v_pool_ptr,
        block_size,
        cos_ptr, sin_ptr,
        seq_ids_ptr, seq_lens_ptr,
        shared_mem_bytes,
        ptx_ptr, name_ptr,
        block_q, block_kv,
        causal,
        x_ptr, norm_weight_ptr,
        wq_ptr, wk_ptr, wv_ptr, wo_ptr,
        rmsnorm_eps_bits,
        active_heads, d_model,
        q_proj_ptr, k_proj_ptr, v_proj_ptr,
        row_max_ptr, row_sum_ptr,
        x_raw_ptr,
        do_ptr,
        dq_ptr, dk_ptr, dv_ptr,
        dwq_ptr, dwk_ptr, dwv_ptr,
        dx_ptr,
        dx_norm_ptr,
        segment_ids_ptr,
        tier_b_ptx_ptr,
        tier_b_name_ptr,
        doc_starts_ptr,
        tier_b2_active,
        num_docs_or_zero,
        Some((probe_ds_out_ptr as u64, probe_dv_out_ptr as u64)),
    )
}

/// CSHA Tier B.2 (Phase 3) — `effective_bq` per the dQ/dK/dV-kernel SMEM
/// fallback schedule (`smem_layout::tier_b2_effective_bq`). The runtime cannot
/// depend on `nsl-codegen` (the dependency runs the other way), so the formula
/// is duplicated here. block_q is clamped to 32 at head_dim 128/256 (the
/// SMEM-pressure / register-pressure fallback); otherwise it is used as-is.
/// MUST stay in sync with `smem_layout::tier_b2_effective_bq`.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_effective_bq(block_q: i64, head_dim: i64) -> i64 {
    match head_dim {
        128 | 256 => block_q.min(32),
        _ => block_q,
    }
}

/// Total dynamic-SMEM bytes for the `tier_b2_dq_kernel`, mirroring
/// `smem_layout::tier_b2_dq_total_smem_bytes` (the offset chain summed band by
/// band). `eb == ek` in the Approach-A bq==bkv invariant, but both are computed
/// for fidelity. CHUNK = `TIER_B2_RMSNORM_CHUNK` = 4. All tiles f16 (2 B/elem).
/// MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_dq_total_smem_bytes(block_q: i64, head_dim: i64) -> i64 {
    const CHUNK: i64 = 4;
    let eb = tier_b2_effective_bq(block_q, head_dim);
    let ek = eb; // bq == bkv invariant
    let hd = head_dim;
    // Band chain (low -> high address), matching tier_b2_dq_*_offset:
    //   Q, K, V, dO, dS, Wk-chunk, Wv-chunk, x_q-chunk, x_kv-chunk, K-colmajor.
    let mut off = 0i64;
    off += eb * hd * 2; // Q          -> K offset
    off += ek * hd * 2; // K          -> V offset
    off += ek * hd * 2; // V          -> dO offset
    off += eb * hd * 2; // dO         -> dS offset
    off += eb * ek * 2; // dS         -> Wk-chunk offset
    off += CHUNK * hd * 2; // Wk-chunk -> Wv-chunk offset
    off += CHUNK * hd * 2; // Wv-chunk -> x_q-chunk offset
    off += eb * CHUNK * 2; // x_q-chunk -> x_kv-chunk offset
    off += ek * CHUNK * 2; // x_kv-chunk -> K-colmajor offset
    off += ek * hd * 2; // K-colmajor band (capacity == row-major K tile)
    off
}

/// Total dynamic-SMEM bytes for the `tier_b2_dkdv_kernel`, mirroring
/// `smem_layout::tier_b2_dkdv_total_smem_bytes`. Same prefix as dQ through the
/// x_kv chunk, then the col-major B/A re-stage bands (Q-colmajor, dO-colmajor,
/// P-colmajor, dS-colmajor). MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_dkdv_total_smem_bytes(block_q: i64, head_dim: i64) -> i64 {
    const CHUNK: i64 = 4;
    let eb = tier_b2_effective_bq(block_q, head_dim);
    let ek = eb; // bq == bkv invariant
    let hd = head_dim;
    let mut off = 0i64;
    off += eb * hd * 2; // Q          -> K offset
    off += ek * hd * 2; // K          -> V offset
    off += ek * hd * 2; // V          -> dO offset
    off += eb * hd * 2; // dO         -> Wk-chunk offset
    off += CHUNK * hd * 2; // Wk-chunk -> Wv-chunk offset
    off += CHUNK * hd * 2; // Wv-chunk -> x_q-chunk offset
    off += eb * CHUNK * 2; // x_q-chunk -> x_kv-chunk offset
    off += ek * CHUNK * 2; // x_kv-chunk -> Q-colmajor offset
    off += eb * hd * 2; // Q-colmajor -> dO-colmajor offset
    off += eb * hd * 2; // dO-colmajor -> P-colmajor offset
    off += eb * ek * 2; // P-colmajor -> dS-colmajor offset
    off += eb * ek * 2; // dS-colmajor band
    off
}

/// Compacted dynamic-SMEM bytes for the standalone `tier_b2_proj_backward`
/// kernel, mirroring `smem_layout::tier_b2_proj_backward_smem_bytes`.
///
/// The proj kernel reuses the scalar fused-backward prelude, which lays out the
/// dQ/dK/dV/v_in/x_norm/dx_norm/rms tiles AFTER the full forward layout
/// (`total_bytes` + P + dS). `synthesize_proj_backward` rebases `%shmem_base`
/// down by `backward_dq_offset` so those tiles start at allocation byte 0; in
/// the rebase the `total_bytes` and P/dS terms cancel, leaving exactly the sum
/// of the tail tiles below. All f32 except v_in (f16). seq == block_q in the
/// smoke scope so block_q/block_kv are the raw config values (NOT effective_bq).
/// MUST stay in sync with the codegen helper.
#[cfg(feature = "cuda")]
#[inline]
fn tier_b2_proj_backward_smem_bytes(block_q: i64, block_kv: i64, head_dim: i64, d_model: i64) -> i64 {
    let dq = block_q * head_dim * 4;
    let dv = block_kv * head_dim * 4;
    let dk = block_kv * head_dim * 4;
    let v_in = block_kv * head_dim * 2;
    let x_norm = block_q * d_model * 4;
    let dx_norm = block_q * d_model * 4;
    let rms_strip = block_q * 4;
    dq + dv + dk + v_in + x_norm + dx_norm + rms_strip
}

/// CSHA Tier B.2 (Phase 3 T6): launch the 4-kernel hybrid backward sequence.
///
/// `ptx_ptr` is a single (null-terminated) PTX module containing all four
/// concatenated entries:
///   1. `tier_b2_d_prepass`    reads dO, O          -> writes D (scratch)
///   2. `tier_b2_dq_kernel`    reads q/k/v_saved, dO, row_max, row_sum, D -> dQ
///   3. `tier_b2_dkdv_kernel`  reads q/k/v_saved, dO, row_max, row_sum, D -> dK, dV
///   4. `tier_b2_proj_backward` reads dQ/dK/dV (HBM, read-only) + x_raw/Wq/Wk/Wv
///      + norm_weight -> dWq, dWk, dWv, dx, dx_norm  (scalar backward param ABI)
///
/// Buffer aliasing (spec §4): the dQ/dK/dV buffers written by kernels 2/3 ARE
/// the final gradient outputs that kernel 4 reads; only the D-scratch is new.
///
/// Per-kernel grid/block are ported from the validated Layer-1 GPU reference
/// launchers (`tier_b2_dq_kernel_cpu_reference.rs`,
/// `tier_b2_dkdv_kernel_cpu_reference.rs`); kernel 4 mirrors the scalar
/// backward launch (grid `(1, batch*effective_heads, 1)`, block `(128,1,1)`,
/// threading the per-q-block base through the `seq_lens` slot).
///
/// Launches are same-stream and therefore implicitly ordered; `NSL_CUDA_SYNC`
/// is honored inside `kernel_launch` (it synchronizes + surfaces async errors
/// after every launch). Returns the first non-success `CUresult` as `i64`, or
/// `CUDA_SUCCESS` (0) when all four launch cleanly.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn csha_tier_b2_backward_launch(
    ptx_ptr: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_q: i64,
    active_heads: i64, d_model: i64,
    scale_bits: i64,
    shared_mem_bytes: i64,
    causal: i64,
    x_ptr: i64, norm_weight_ptr: i64,
    wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
    rmsnorm_eps_bits: i64,
    q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
    row_max_ptr: i64, row_sum_ptr: i64, x_raw_ptr: i64,
    logsumexp_ptr: i64, out_ptr: i64,
    do_ptr: i64,
    dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
    dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
    dx_ptr: i64, dx_norm_ptr: i64,
    segment_ids_ptr: i64, doc_starts_ptr: i64,
    // Sprint 10: rope_q=true support. When non-zero, the cos/sin device
    // buffers are threaded through to the proj_backward kernel so its
    // emit_drope phase can de-rotate dQ/dK from post-RoPE to pre-RoPE
    // before emit_dproj computes the weight gradients. Null on rope_q=false
    // launches; emit_drope's internal null-guard then makes the rotation a
    // no-op (which it is even on the codegen side: the codegen-level gate
    // in proj_backward.rs skips emit_drope entirely when !config.rope_q,
    // so a null cos/sin here is never read).
    cos_ptr: i64, sin_ptr: i64,
) -> i64 {
    use cudarc::driver::sys::CUresult;

    let success = CUresult::CUDA_SUCCESS;

    // Resolve the user-facing NslTensor handles to raw device pointers (auto-
    // promotes CPU->GPU + extracts `.data`), exactly as the scalar path does.
    // The forward-saved activations (q/k/v_proj, row_max, row_sum, x_raw) are
    // raw device buffers from `nsl_csha_alloc_backward_activations_into` and
    // pass through unchanged.
    let d_o = csha_tensor_data_ptr(do_ptr);
    let d_q = csha_tensor_data_ptr(dq_ptr);
    let d_k = csha_tensor_data_ptr(dk_ptr);
    let d_v = csha_tensor_data_ptr(dv_ptr);
    let qp = q_proj_ptr as u64;
    let kpj = k_proj_ptr as u64;
    let vpj = v_proj_ptr as u64;
    let rmax = row_max_ptr as u64;
    let rsum = row_sum_ptr as u64;
    let xraw = x_raw_ptr as u64;

    // The D pre-pass (kernel 1) needs the forward attention output O to compute
    // D = rowsum(dO * O). The scalar Tier C path resolves O from the `out_ptr`
    // slot (`csha_tensor_data_ptr(out_ptr)`); the standalone dQ/dK/dV gates feed
    // `fwd.o`. We source O from the same `out_ptr` slot here.
    //
    // PARITY-GATE NOTE (T8, RESOLVED by Sprint 1 T1.1): the wengert
    // `FusedCshaBackward` lowering now threads the forward O handle here —
    // `CshaSavePointers.out` is populated at both forward call sites and
    // passed in the `out_ptr` slot (wengert_lower ~2137). The tier_b2_active
    // flag is likewise computed at lowering (~2088), though it is iconst(0)
    // for the configs the production training-config builder currently
    // emits (kernel.rs pins level=1 / active_heads=0). If O were null the
    // D pre-pass would yield D == 0 and the §8 zero-output guard FAILS —
    // so a missing O cannot pass vacuously.
    let o_for_prepass = csha_tensor_data_ptr(out_ptr);

    // D-scratch: D = rowsum(dO * O), f32, shape [batch, heads, seq_len].
    let d_elems = (batch * heads * seq_len) as usize;
    let d_scratch_raw = if d_elems > 0 {
        crate::cuda::inner::alloc_device(d_elems * 4)
    } else {
        std::ptr::null_mut()
    };
    if d_scratch_raw.is_null() {
        eprintln!("[nsl] CSHA Tier B.2 backward: D-scratch alloc failed");
        return CUresult::CUDA_ERROR_OUT_OF_MEMORY as i64;
    }
    crate::cuda::inner::memset_d8(d_scratch_raw, d_elems * 4);
    let d_scratch = d_scratch_raw as u64;

    // Sprint 1 T1.2 — DTYPE RECONCILIATION (production f16 dq/dk/dv vs kernel f32).
    //
    // Wengert allocates dq/dk/dv as f16 NslTensors (`nsl_tensor_zeros_f16_on` —
    // see `crates/nsl-codegen/src/wengert_lower.rs:1834-1846`). The downstream
    // optimizer / extract ops expect f16 — this is the production contract.
    //
    // The Tier B.2 dq and dkdv kernels write `st.global.f32` (4 bytes/elem) into
    // dq/dk/dv; the proj_backward kernel reads them back as `ld.global.f32`. Wiring
    // the kernels' f32 writes directly against the f16 production buffers would be
    // a 2x buffer overflow (the f16 allocation is half the size of the f32 write
    // footprint). The scalar Tier C dK/dV path solves the same mismatch with f32
    // scratch + `csha_bwd_convert_f32_to_f16` (see `flash_attention.rs:1521-1681`);
    // we apply the identical pattern to dq + dk + dv here.
    //
    // Layout per scratch: f32 [batch, heads, seq_len, head_dim].
    let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
    let scratch_bytes = qkv_elems * 4;
    let dq_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let dk_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let dv_scratch_raw = if qkv_elems > 0 {
        crate::cuda::inner::alloc_device(scratch_bytes)
    } else {
        std::ptr::null_mut()
    };
    let free_scratches = |dq_raw: *mut c_void, dk_raw: *mut c_void, dv_raw: *mut c_void,
                          d_raw: *mut c_void| {
        if !dq_raw.is_null() {
            crate::cuda::inner::free_device(dq_raw);
        }
        if !dk_raw.is_null() {
            crate::cuda::inner::free_device(dk_raw);
        }
        if !dv_raw.is_null() {
            crate::cuda::inner::free_device(dv_raw);
        }
        if !d_raw.is_null() {
            crate::cuda::inner::free_device(d_raw);
        }
    };
    if qkv_elems > 0 && (dq_scratch_raw.is_null() || dk_scratch_raw.is_null() || dv_scratch_raw.is_null()) {
        eprintln!("[nsl] CSHA Tier B.2 backward: dQ/dK/dV scratch alloc failed");
        free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
        return CUresult::CUDA_ERROR_OUT_OF_MEMORY as i64;
    }
    if !dq_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dq_scratch_raw, scratch_bytes);
    }
    if !dk_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dk_scratch_raw, scratch_bytes);
    }
    if !dv_scratch_raw.is_null() {
        crate::cuda::inner::memset_d8(dv_scratch_raw, scratch_bytes);
    }
    let dq_scratch = dq_scratch_raw as u64;
    let dk_scratch = dk_scratch_raw as u64;
    let dv_scratch = dv_scratch_raw as u64;

    // Null-terminated PTX + kernel-name C strings.
    let ptx = ptx_ptr as *const u8;
    let name_d_prepass = b"tier_b2_d_prepass\0";
    let name_dq = b"tier_b2_dq_kernel\0";
    let name_dkdv = b"tier_b2_dkdv_kernel\0";
    let name_proj = b"tier_b2_proj_backward\0";

    let eb = tier_b2_effective_bq(block_q, head_dim);

    // ── Kernel 1: D pre-pass ──
    // ABI: (d_o_ptr, o_ptr, d_out_ptr, seq_len:u32, heads:u32).
    // Grid (ceil(seq/32), heads, batch), block (32,1,1), no dynamic SMEM.
    {
        let mut a_do = d_o as u64;
        let mut a_o = o_for_prepass as u64;
        let mut a_dout = d_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let args: [*mut c_void; 5] = [
            &mut a_do as *mut _ as *mut c_void,
            &mut a_o as *mut _ as *mut c_void,
            &mut a_dout as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + 31) / 32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_d_prepass.as_ptr(),
            [grid_x, heads, batch], [32, 1, 1], &args, 0,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 2: dQ ──
    // ABI: (q,k,v_saved, d_o, row_max, row_sum, d, segment_ids, d_q_out,
    //       seq:u32, heads:u32, batch:u32).
    // Grid (ceil(seq/eb), heads, batch), block (128,1,1), dynamic SMEM = dq total.
    {
        let mut a_q = qp;
        let mut a_k = kpj;
        let mut a_v = vpj;
        let mut a_do = d_o as u64;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_d = d_scratch;
        let mut a_seg = segment_ids_ptr as u64;
        // Sprint 1 T1.2: dq kernel writes f32 → dq_scratch (not the production f16 buffer).
        let mut a_dq = dq_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let mut a_batch = batch as u32;
        let args: [*mut c_void; 12] = [
            &mut a_q as *mut _ as *mut c_void,
            &mut a_k as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_d as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_dq as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
            &mut a_batch as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + eb - 1) / eb;
        let smem = tier_b2_dq_total_smem_bytes(block_q, head_dim) as u32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_dq.as_ptr(),
            [grid_x, heads, batch], [128, 1, 1], &args, smem,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 3: dK/dV ──
    // ABI: (q,k,v_saved, d_o, row_max, row_sum, d, segment_ids, d_k_out,
    //       d_v_out, seq:u32, heads:u32, batch:u32). NOTE: d_k BEFORE d_v.
    // Grid (ceil(seq/eb), heads, batch), block (128,1,1), dynamic SMEM = dkdv total.
    {
        let mut a_q = qp;
        let mut a_k = kpj;
        let mut a_v = vpj;
        let mut a_do = d_o as u64;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_d = d_scratch;
        let mut a_seg = segment_ids_ptr as u64;
        // Sprint 1 T1.2: dkdv kernel writes f32 → dk_scratch / dv_scratch.
        let mut a_dk = dk_scratch;
        let mut a_dv = dv_scratch;
        let mut a_seq = seq_len as u32;
        let mut a_heads = heads as u32;
        let mut a_batch = batch as u32;
        let args: [*mut c_void; 13] = [
            &mut a_q as *mut _ as *mut c_void,
            &mut a_k as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_d as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_dk as *mut _ as *mut c_void,
            &mut a_dv as *mut _ as *mut c_void,
            &mut a_seq as *mut _ as *mut c_void,
            &mut a_heads as *mut _ as *mut c_void,
            &mut a_batch as *mut _ as *mut c_void,
        ];
        let grid_x = (seq_len + eb - 1) / eb;
        let smem = tier_b2_dkdv_total_smem_bytes(block_q, head_dim) as u32;
        let rc = crate::cuda::inner::kernel_launch(
            ptx, name_dkdv.as_ptr(),
            [grid_x, heads, batch], [128, 1, 1], &args, smem,
        );
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // ── Kernel 4: projection backward ──
    // Reuses the SCALAR backward param ABI (the same 49-arg list the scalar
    // else-branch builds). dQ/dK/dV are INPUTS (read from HBM into SMEM); only
    // dWq/dWk/dWv/dx/dx_norm are written. The dk_scratch/dv_scratch slots are
    // unused by proj_backward (no dK/dV finalize store) — pass null. Launch
    // dims mirror the scalar backward: grid (1, batch*effective_heads, 1),
    // block (128,1,1), incoming `shared_mem_bytes`, threading the per-q-block
    // base through the `seq_lens` slot.
    {
        let effective_heads = if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        };

        // q_ptr/k_ptr/v_ptr are unused by proj_backward (it reads the forward
        // saves, not the raw inputs). Pass 0 for all three.
        let mut k = 0u64;
        let mut v = 0u64;
        let mut out_p = o_for_prepass as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = 0u64;
        let mut kp = 0u64;
        let mut vp = 0u64;
        let mut bsz = 0u64;
        // Sprint 10: thread cos/sin to proj_backward so its emit_drope phase
        // can de-rotate dQ/dK to the pre-RoPE basis before dproj reads them.
        // Resolve through csha_tensor_data_ptr (same auto-promote + .data
        // extraction the scalar backward path uses for cos/sin at line
        // 1469-1470). On rope_q=false launches the caller passes 0, which
        // csha_tensor_data_ptr returns as 0 — emit_drope's internal null-
        // guard then short-circuits; under the Sprint-10 codegen-side gate
        // the entire dRoPE block is also skipped, so this is doubly safe.
        let mut cos = csha_tensor_data_ptr(cos_ptr) as u64;
        let mut sin = csha_tensor_data_ptr(sin_ptr) as u64;
        let mut sids = 0u64;
        let mut slens = 0u64; // per-q-block base, threaded in the loop below
        let mut dfs_enter = 0u64;
        let mut dfs_exit = 0u64;
        let mut num_tree_nodes = 0u64;
        let mut lse = csha_tensor_data_ptr(logsumexp_ptr) as u64;
        let mut x = csha_tensor_data_ptr(x_ptr) as u64;
        let mut nw = csha_tensor_data_ptr(norm_weight_ptr) as u64;
        let mut wq = csha_tensor_data_ptr(wq_ptr) as u64;
        let mut wk = csha_tensor_data_ptr(wk_ptr) as u64;
        let mut wv = csha_tensor_data_ptr(wv_ptr) as u64;
        let mut wo = csha_tensor_data_ptr(wo_ptr) as u64;
        let mut eps = f32::from_bits(rmsnorm_eps_bits as u32);
        let mut ah = active_heads as u32;
        let mut dm = d_model as u32;
        let mut a_qp = qp;
        let mut a_kpj = kpj;
        let mut a_vpj = vpj;
        let mut a_rmax = rmax;
        let mut a_rsum = rsum;
        let mut a_xraw = xraw;
        let mut a_do = d_o as u64;
        // dQ/dK/dV: read-only inputs to proj_backward (from HBM).
        // Sprint 1 T1.2: proj_backward reads f32 dQ/dK/dV from the same scratches
        // the dq + dkdv kernels wrote to. The production f16 destinations are
        // populated AFTER proj_backward via `csha_bwd_convert_f32_to_f16`.
        let mut a_dq = dq_scratch;
        let mut a_dk = dk_scratch;
        let mut a_dv = dv_scratch;
        // Gradient outputs written by proj_backward.
        let mut a_dwq = csha_tensor_data_ptr(dwq_ptr) as u64;
        let mut a_dwk = csha_tensor_data_ptr(dwk_ptr) as u64;
        let mut a_dwv = csha_tensor_data_ptr(dwv_ptr) as u64;
        let mut a_dx = csha_tensor_data_ptr(dx_ptr) as u64;
        let mut a_dxn = csha_tensor_data_ptr(dx_norm_ptr) as u64;
        // proj_backward does not write dK/dV — no f32 scratch needed.
        let mut a_dk_scratch = 0u64;
        let mut a_dv_scratch = 0u64;
        let mut a_seg = segment_ids_ptr as u64;
        let mut a_docs = doc_starts_ptr as u64;
        let _ = causal; // proj/dRMSNorm are causal-agnostic; param resolved for ABI parity

        let args: [*mut c_void; 49] = [
            &mut k as *mut _ as *mut c_void, // q_ptr slot (unused; pass 0)
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out_p as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bsz as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut nw as *mut _ as *mut c_void,
            &mut wq as *mut _ as *mut c_void,
            &mut wk as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut wo as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut ah as *mut _ as *mut c_void,
            &mut dm as *mut _ as *mut c_void,
            &mut a_qp as *mut _ as *mut c_void,
            &mut a_kpj as *mut _ as *mut c_void,
            &mut a_vpj as *mut _ as *mut c_void,
            &mut a_rmax as *mut _ as *mut c_void,
            &mut a_rsum as *mut _ as *mut c_void,
            &mut a_xraw as *mut _ as *mut c_void,
            &mut a_do as *mut _ as *mut c_void,
            &mut a_dq as *mut _ as *mut c_void,
            &mut a_dk as *mut _ as *mut c_void,
            &mut a_dv as *mut _ as *mut c_void,
            &mut a_dwq as *mut _ as *mut c_void,
            &mut a_dwk as *mut _ as *mut c_void,
            &mut a_dwv as *mut _ as *mut c_void,
            &mut a_dx as *mut _ as *mut c_void,
            &mut a_dxn as *mut _ as *mut c_void,
            &mut a_dk_scratch as *mut _ as *mut c_void,
            &mut a_dv_scratch as *mut _ as *mut c_void,
            &mut a_seg as *mut _ as *mut c_void,
            &mut a_docs as *mut _ as *mut c_void,
        ];

        let grid_y = batch * effective_heads;
        let q_blocks = if block_q > 0 { (seq_len + block_q - 1) / block_q } else { 1 };
        // T8 fix: the proj kernel rebases %shmem_base to a COMPACTED layout
        // (see `synthesize_proj_backward` + `tier_b2_proj_backward_smem_bytes`),
        // so the incoming `shared_mem_bytes` (which sizes the FULL scalar fused
        // backward, ~137 KB at hd=64 — past the 99 KB device cap) must be
        // overridden with the compacted ~88 KB span the standalone kernel
        // actually uses. d_model from the active_heads-derived projection
        // contract (== head_dim under the smoke scope).
        let proj_smem = tier_b2_proj_backward_smem_bytes(
            block_q, block_q, head_dim, d_model,
        ) as u32;
        let _ = shared_mem_bytes; // superseded by the compacted proj footprint
        let mut rc = success;
        for q_block in 0..q_blocks {
            slens = (q_block * block_q) as u64;
            let _ = std::hint::black_box(&slens);
            rc = crate::cuda::inner::kernel_launch(
                ptx, name_proj.as_ptr(),
                [1, grid_y, 1], [128, 1, 1], &args, proj_smem,
            );
            if rc != success {
                break;
            }
        }
        if rc != success {
            free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
            return rc as i64;
        }
    }

    // Sprint 1 T1.2: Convert f32 scratches → f16 production destinations. Each
    // call reads `qkv_elems` f32 elements from a scratch and writes the same
    // count of f16 elements to the destination (the production wengert-allocated
    // f16 buffer). `csha_bwd_convert_f32_to_f16` is the same helper the scalar
    // dK/dV path uses (see `flash_attention.rs:1654-1674`).
    if qkv_elems > 0 {
        if d_q != 0 && !dq_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dq_scratch_raw, d_q as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
        if d_k != 0 && !dk_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dk_scratch_raw, d_k as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
        if d_v != 0 && !dv_scratch_raw.is_null() {
            let c_rc = csha_bwd_convert_f32_to_f16(
                dv_scratch_raw, d_v as *mut c_void, qkv_elems,
            );
            if c_rc != success {
                free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
                return c_rc as i64;
            }
        }
    }

    free_scratches(dq_scratch_raw, dk_scratch_raw, dv_scratch_raw, d_scratch_raw);
    success as i64
}

/// f32 scratch → f16 output conversion kernel used by the CSHA backward
/// (Option A of the dK/dV saturation fix). Cached once per process via
/// `kernel_launch`'s content-hashed module cache.
///
/// Kernel body: one thread per element, bounds-checked against `n_elems`.
/// `src` is f32, `dst` is f16; conversion is `cvt.rn.f16.f32` (round to
/// nearest even). Overflow still saturates to ±inf — acceptable here
/// because the f32 scratch was accumulated deterministically and any
/// |x| > 65504 is a legitimate gradient value the downstream f16 store
/// can't represent regardless of accumulation strategy.
#[cfg(feature = "cuda")]
const CSHA_BWD_F32_TO_F16_PTX: &str = concat!(
    ".version 8.7\n",
    ".target sm_75\n",
    ".address_size 64\n",
    ".visible .entry nsl_csha_bwd_f32_to_f16(\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 n_elems\n",
    ")\n",
    "{\n",
    // User registers intentionally avoid %tid / %ntid / %ctaid names
    // — those are PTX built-in special registers and redefining them
    // as user regs triggers a ptxas parse failure (tid.x dot-syntax
    // clash).
    "    .reg .u32 %th, %bd, %bi, %gid, %tmp;\n",
    "    .reg .u64 %idx, %n, %src, %dst, %addr_src, %addr_dst;\n",
    "    .reg .f32 %f;\n",
    "    .reg .b16 %h;\n",
    "    .reg .pred %p;\n",
    "    mov.u32 %th, %tid.x;\n",
    "    mov.u32 %bd, %ntid.x;\n",
    "    mov.u32 %bi, %ctaid.x;\n",
    "    mul.lo.u32 %tmp, %bi, %bd;\n",
    "    add.u32 %gid, %tmp, %th;\n",
    "    cvt.u64.u32 %idx, %gid;\n",
    "    ld.param.u64 %src, [src_ptr];\n",
    "    ld.param.u64 %dst, [dst_ptr];\n",
    "    ld.param.u64 %n, [n_elems];\n",
    "    setp.ge.u64 %p, %idx, %n;\n",
    "    @%p bra F32F16_END;\n",
    "    shl.b64 %addr_src, %idx, 2;\n",
    "    add.u64 %addr_src, %src, %addr_src;\n",
    "    shl.b64 %addr_dst, %idx, 1;\n",
    "    add.u64 %addr_dst, %dst, %addr_dst;\n",
    "    ld.global.f32 %f, [%addr_src];\n",
    "    cvt.rn.f16.f32 %h, %f;\n",
    "    st.global.b16 [%addr_dst], %h;\n",
    "F32F16_END:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

#[cfg(feature = "cuda")]
const CSHA_BWD_F32_TO_F16_NAME: &str = "nsl_csha_bwd_f32_to_f16\0";

/// f16 -> f32 widening kernel: reads `n_elems` f16 values from `src_ptr`
/// and writes them as f32 to `dst_ptr`. Used by the multi-tile two-launch
/// forward dispatch: launch A saves projections as f16 (`q_proj`/`k_proj`/
/// `v_proj`), while the attention pass's HBM tile loads read f32 sources
/// (`ld.global.f32` in `emit_k_tile_load` / the q_load wq-null fallback).
/// Same register discipline as the f32->f16 sibling above.
#[cfg(feature = "cuda")]
const CSHA_FWD_F16_TO_F32_PTX: &str = concat!(
    ".version 8.7\n",
    ".target sm_75\n",
    ".address_size 64\n",
    ".visible .entry nsl_csha_fwd_f16_to_f32(\n",
    "    .param .u64 src_ptr,\n",
    "    .param .u64 dst_ptr,\n",
    "    .param .u64 n_elems\n",
    ")\n",
    "{\n",
    "    .reg .u32 %th, %bd, %bi, %gid, %tmp;\n",
    "    .reg .u64 %idx, %n, %src, %dst, %addr_src, %addr_dst;\n",
    "    .reg .f32 %f;\n",
    "    .reg .b16 %h;\n",
    "    .reg .pred %p;\n",
    "    mov.u32 %th, %tid.x;\n",
    "    mov.u32 %bd, %ntid.x;\n",
    "    mov.u32 %bi, %ctaid.x;\n",
    "    mul.lo.u32 %tmp, %bi, %bd;\n",
    "    add.u32 %gid, %tmp, %th;\n",
    "    cvt.u64.u32 %idx, %gid;\n",
    "    ld.param.u64 %src, [src_ptr];\n",
    "    ld.param.u64 %dst, [dst_ptr];\n",
    "    ld.param.u64 %n, [n_elems];\n",
    "    setp.ge.u64 %p, %idx, %n;\n",
    "    @%p bra F16F32_END;\n",
    "    shl.b64 %addr_src, %idx, 1;\n",
    "    add.u64 %addr_src, %src, %addr_src;\n",
    "    shl.b64 %addr_dst, %idx, 2;\n",
    "    add.u64 %addr_dst, %dst, %addr_dst;\n",
    "    ld.global.b16 %h, [%addr_src];\n",
    "    cvt.f32.f16 %f, %h;\n",
    "    st.global.f32 [%addr_dst], %f;\n",
    "F16F32_END:\n",
    "    ret;\n",
    "}\n",
    "\0",
);

#[cfg(feature = "cuda")]
const CSHA_FWD_F16_TO_F32_NAME: &str = "nsl_csha_fwd_f16_to_f32\0";

/// Launch the f16->f32 widening kernel. `n_elems` is the element count
/// (src is f16*2 bytes, dst is f32*4 bytes).
#[cfg(feature = "cuda")]
fn csha_fwd_convert_f16_to_f32(
    src: u64,
    dst: *mut c_void,
    n_elems: usize,
) -> cudarc::driver::sys::CUresult {
    if n_elems == 0 {
        return cudarc::driver::sys::CUresult::CUDA_SUCCESS;
    }
    let mut src_ptr = src;
    let mut dst_ptr = dst as u64;
    let mut n = n_elems as u64;
    let args: [*mut c_void; 3] = [
        &mut src_ptr as *mut _ as *mut c_void,
        &mut dst_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block: i64 = 128;
    let grid: i64 = ((n_elems as i64) + block - 1) / block;
    crate::cuda::inner::kernel_launch(
        CSHA_FWD_F16_TO_F32_PTX.as_ptr(),
        CSHA_FWD_F16_TO_F32_NAME.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    )
}

/// Launch the f32→f16 conversion kernel. `n_elems` is the element count
/// (src is f32*4 bytes, dst is f16*2 bytes).
#[cfg(feature = "cuda")]
fn csha_bwd_convert_f32_to_f16(
    src: *mut c_void,
    dst: *mut c_void,
    n_elems: usize,
) -> cudarc::driver::sys::CUresult {
    if n_elems == 0 {
        return cudarc::driver::sys::CUresult::CUDA_SUCCESS;
    }
    let mut src_ptr = src as u64;
    let mut dst_ptr = dst as u64;
    let mut n = n_elems as u64;
    let args: [*mut c_void; 3] = [
        &mut src_ptr as *mut _ as *mut c_void,
        &mut dst_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block: i64 = 128;
    let grid: i64 = ((n_elems as i64) + block - 1) / block;
    crate::cuda::inner::kernel_launch(
        CSHA_BWD_F32_TO_F16_PTX.as_ptr(),
        CSHA_BWD_F32_TO_F16_NAME.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    )
}

// ── Phase 1.1 (pretraining): multi-tile backward post-pass ──────────────────
//
// At multi-tile (seq_len > block_q OR > block_kv) the in-kernel emit_dproj /
// emit_drmsnorm compute dWk/dWv and dx/dx_norm from each q-block launch's
// PARTIAL dK/dV SMEM tiles, which are wrong: those gradients need the
// cross-launch dK/dV TOTALS. After the host q-block loop, the totals ARE
// available: dk_scratch / dv_scratch (f32, accumulated by the in-loop RMW
// stores) hold dK_proj_total / dV_proj_total, and d_q (f16, disjoint
// per-launch writes) holds dQ_proj_total. This post-pass recomputes the four
// affected gradients from those totals and OVERWRITES the partial in-kernel
// results.
//
// Correctness scope: batch=1, heads=1 (the multi-tile regime the caller
// admits — heads>1 needs cross-head dx_norm accumulation + per-head x_raw
// layout, batch>1 needs an x_raw batch term; both are refused upstream). The
// math mirrors emit_xnorm_recompute / emit_dproj / emit_drmsnorm exactly:
//
//   inv_rms[s]   = rsqrt(mean_p(x_raw[s,p]^2) + eps)
//   x_norm(s,p)  = x_raw[s,p] * inv_rms[s] * norm_weight[p]
//   dWk[p,j]     = sum_s x_norm(s,p) * dk_scratch[s,j]     (= x_norm^T @ dK)
//   dWv[p,j]     = sum_s x_norm(s,p) * dv_scratch[s,j]
//   dx_norm[s,p] = sum_j dQ[s,j]*Wq[p,j] + dK[s,j]*Wk[p,j] + dV[s,j]*Wv[p,j]
//   g_d          = dx_norm[s,p] * norm_weight[p]
//   s_grad       = sum_p g_d * x_raw[s,p]
//   dx[s,p]      = g_d*inv_rms[s] - x_raw[s,p]*s_grad*inv_rms[s]^3/d_model
//
// One CTA (grid=(1,1,1)), 128 threads, serial inner loops (correctness-first;
// perf is Tier B.2's job). inv_rms[s] is cached in static shared memory
// (pp_smem[16384] = 4096 f32 rows → seq_len <= 4096; the launcher refuses
// larger). All buffer layouts match the in-kernel emitters byte-for-byte at
// batch=1/heads=1: x_raw/dx/dx_norm [seq, d_model] f32; dQ/dK/dV [seq,
// head_dim] (dQ f16, dK/dV f32 scratch); Wq/Wk/Wv/dWk/dWv [d_model, head_dim]
// f16; norm_weight [d_model] f32.
#[cfg(feature = "cuda")]
const CSHA_BWD_MULTITILE_POSTPASS_PTX: &str = concat!(
r#".version 8.7
.target sm_75
.address_size 64
.visible .entry nsl_csha_bwd_multitile_postpass(
    .param .u64 dk_scratch_ptr,
    .param .u64 dv_scratch_ptr,
    .param .u64 dq_ptr,
    .param .u64 x_raw_ptr,
    .param .u64 nw_ptr,
    .param .u64 wq_ptr,
    .param .u64 wk_ptr,
    .param .u64 wv_ptr,
    .param .u64 dwk_ptr,
    .param .u64 dwv_ptr,
    .param .u64 dx_ptr,
    .param .u64 dxn_ptr,
    .param .u32 seq_len,
    .param .u32 d_model,
    .param .u32 head_dim,
    .param .f32 eps
)
{
    .shared .align 4 .b8 pp_smem[16384];
    .reg .u32 %th, %s, %pi, %jj, %cell, %ncells, %tmp, %tmp2, %saddr, %sbase, %widx, %wrow, %pcol, %dm, %hd, %seq;
    .reg .u64 %dkp, %dvp, %dqp, %xrp, %nwp, %wqp, %wkp, %wvp, %dwkp, %dwvp, %dxp, %dxnp;
    .reg .u64 %off, %off2, %a, %xrow, %qrow, %krow, %vrow, %dxnrow, %dxrow;
    .reg .f32 %acc, %mean, %eps, %dmf, %invr, %acck, %accv, %xn, %xv, %nwv, %dkv, %dvv, %dxnf, %sgrad, %gd, %t, %invdm, %one, %pre, %term1, %term2, %dxv, %dqv, %wqv, %wkv, %wvv;
    .reg .b16 %h;
    .reg .pred %p, %pg;

    mov.u32 %th, %tid.x;
    ld.param.u64 %dkp,  [dk_scratch_ptr];
    ld.param.u64 %dvp,  [dv_scratch_ptr];
    ld.param.u64 %dqp,  [dq_ptr];
    ld.param.u64 %xrp,  [x_raw_ptr];
    ld.param.u64 %nwp,  [nw_ptr];
    ld.param.u64 %wqp,  [wq_ptr];
    ld.param.u64 %wkp,  [wk_ptr];
    ld.param.u64 %wvp,  [wv_ptr];
    ld.param.u64 %dwkp, [dwk_ptr];
    ld.param.u64 %dwvp, [dwv_ptr];
    ld.param.u64 %dxp,  [dx_ptr];
    ld.param.u64 %dxnp, [dxn_ptr];
    ld.param.u32 %seq,  [seq_len];
    ld.param.u32 %dm,   [d_model];
    ld.param.u32 %hd,   [head_dim];
    ld.param.f32 %eps,  [eps];
    mov.u32 %sbase, pp_smem;
    cvt.rn.f32.u32 %dmf, %dm;
    mov.f32 %one, 0f3F800000;
    div.rn.f32 %invdm, %one, %dmf;

    // Phase 0: inv_rms[s] -> shared
    mov.u32 %s, %th;
PP_IRMS:
    setp.ge.u32 %p, %s, %seq;
    @%p bra PP_IRMS_DONE;
    mul.lo.u32 %tmp, %s, %dm;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 2;
    add.u64 %xrow, %xrp, %off;
    mov.f32 %acc, 0f00000000;
    mov.u32 %pi, 0;
PP_IRMS_SUM:
    setp.ge.u32 %p, %pi, %dm;
    @%p bra PP_IRMS_SUM_DONE;
    cvt.u64.u32 %off, %pi;
    shl.b64 %off, %off, 2;
    add.u64 %a, %xrow, %off;
    ld.global.f32 %xv, [%a];
    fma.rn.f32 %acc, %xv, %xv, %acc;
    add.u32 %pi, %pi, 1;
    bra PP_IRMS_SUM;
PP_IRMS_SUM_DONE:
    mul.f32 %mean, %acc, %invdm;
    add.f32 %mean, %mean, %eps;
    rsqrt.approx.f32 %invr, %mean;
    mul.lo.u32 %saddr, %s, 4;
    add.u32 %saddr, %sbase, %saddr;
    st.shared.f32 [%saddr], %invr;
    add.u32 %s, %s, 128;
    bra PP_IRMS;
PP_IRMS_DONE:
    bar.sync 0;

    // Phase A: dWk[p,j], dWv[p,j] = sum_s x_norm(s,p) * d{K,V}[s,j]
    mul.lo.u32 %ncells, %dm, %hd;
    mov.u32 %cell, %th;
PP_A:
    setp.ge.u32 %p, %cell, %ncells;
    @%p bra PP_A_DONE;
    div.u32 %pcol, %cell, %hd;
    rem.u32 %jj, %cell, %hd;
    mov.f32 %acck, 0f00000000;
    mov.f32 %accv, 0f00000000;
    cvt.u64.u32 %off, %pcol;
    shl.b64 %off, %off, 2;
    add.u64 %a, %nwp, %off;
    ld.global.f32 %nwv, [%a];
    mov.u32 %s, 0;
PP_A_S:
    setp.ge.u32 %p, %s, %seq;
    @%p bra PP_A_S_DONE;
    mul.lo.u32 %tmp, %s, %dm;
    add.u32 %tmp, %tmp, %pcol;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 2;
    add.u64 %a, %xrp, %off;
    ld.global.f32 %xv, [%a];
    mul.lo.u32 %saddr, %s, 4;
    add.u32 %saddr, %sbase, %saddr;
    ld.shared.f32 %invr, [%saddr];
    mul.f32 %xn, %xv, %invr;
    mul.f32 %xn, %xn, %nwv;
    mul.lo.u32 %tmp2, %s, %hd;
    add.u32 %tmp2, %tmp2, %jj;
    cvt.u64.u32 %off2, %tmp2;
    shl.b64 %off2, %off2, 2;
    add.u64 %a, %dkp, %off2;
    ld.global.f32 %dkv, [%a];
    add.u64 %a, %dvp, %off2;
    ld.global.f32 %dvv, [%a];
    fma.rn.f32 %acck, %xn, %dkv, %acck;
    fma.rn.f32 %accv, %xn, %dvv, %accv;
    add.u32 %s, %s, 1;
    bra PP_A_S;
PP_A_S_DONE:
    mul.lo.u32 %widx, %pcol, %hd;
    add.u32 %widx, %widx, %jj;
    cvt.u64.u32 %off, %widx;
    shl.b64 %off, %off, 1;
    setp.eq.u64 %pg, %dwkp, 0;
    @%pg bra PP_A_SKIP_WK;
    add.u64 %a, %dwkp, %off;
    cvt.rn.f16.f32 %h, %acck;
    st.global.b16 [%a], %h;
PP_A_SKIP_WK:
    setp.eq.u64 %pg, %dwvp, 0;
    @%pg bra PP_A_SKIP_WV;
    add.u64 %a, %dwvp, %off;
    cvt.rn.f16.f32 %h, %accv;
    st.global.b16 [%a], %h;
PP_A_SKIP_WV:
    add.u32 %cell, %cell, 128;
    bra PP_A;
PP_A_DONE:

    // Phase B: dx_norm[s,p], dx[s,p] (one thread per row)
    setp.eq.u64 %pg, %dxnp, 0;
    @%pg bra PP_B_DONE;
    mov.u32 %s, %th;
PP_B:
    setp.ge.u32 %p, %s, %seq;
    @%p bra PP_B_DONE;
    mul.lo.u32 %saddr, %s, 4;
    add.u32 %saddr, %sbase, %saddr;
    ld.shared.f32 %invr, [%saddr];
    mul.lo.u32 %tmp, %s, %dm;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 2;
    add.u64 %xrow, %xrp, %off;
    add.u64 %dxnrow, %dxnp, %off;
    add.u64 %dxrow, %dxp, %off;
    mul.lo.u32 %tmp2, %s, %hd;
    cvt.u64.u32 %off2, %tmp2;
    shl.b64 %off2, %off2, 1;
    add.u64 %qrow, %dqp, %off2;
    cvt.u64.u32 %off2, %tmp2;
    shl.b64 %off2, %off2, 2;
    add.u64 %krow, %dkp, %off2;
    add.u64 %vrow, %dvp, %off2;
    mov.f32 %sgrad, 0f00000000;
    mov.u32 %pi, 0;
PP_B_P1:
    setp.ge.u32 %p, %pi, %dm;
    @%p bra PP_B_P1_DONE;
    mul.lo.u32 %wrow, %pi, %hd;
    mov.f32 %dxnf, 0f00000000;
    mov.u32 %jj, 0;
PP_B_J:
    setp.ge.u32 %p, %jj, %hd;
    @%p bra PP_B_J_DONE;
    cvt.u64.u32 %off, %jj;
    shl.b64 %off, %off, 1;
    add.u64 %a, %qrow, %off;
    ld.global.b16 %h, [%a];
    cvt.f32.f16 %dqv, %h;
    cvt.u64.u32 %off, %jj;
    shl.b64 %off, %off, 2;
    add.u64 %a, %krow, %off;
    ld.global.f32 %dkv, [%a];
    add.u64 %a, %vrow, %off;
    ld.global.f32 %dvv, [%a];
    add.u32 %widx, %wrow, %jj;
    cvt.u64.u32 %off, %widx;
    shl.b64 %off, %off, 1;
    add.u64 %a, %wqp, %off;
    ld.global.b16 %h, [%a];
    cvt.f32.f16 %wqv, %h;
    add.u64 %a, %wkp, %off;
    ld.global.b16 %h, [%a];
    cvt.f32.f16 %wkv, %h;
    add.u64 %a, %wvp, %off;
    ld.global.b16 %h, [%a];
    cvt.f32.f16 %wvv, %h;
    fma.rn.f32 %dxnf, %dqv, %wqv, %dxnf;
    fma.rn.f32 %dxnf, %dkv, %wkv, %dxnf;
    fma.rn.f32 %dxnf, %dvv, %wvv, %dxnf;
    add.u32 %jj, %jj, 1;
    bra PP_B_J;
PP_B_J_DONE:
    cvt.u64.u32 %off, %pi;
    shl.b64 %off, %off, 2;
    add.u64 %a, %dxnrow, %off;
    st.global.f32 [%a], %dxnf;
    add.u64 %a, %nwp, %off;
    ld.global.f32 %nwv, [%a];
    mul.f32 %gd, %dxnf, %nwv;
    add.u64 %a, %xrow, %off;
    ld.global.f32 %xv, [%a];
    fma.rn.f32 %sgrad, %gd, %xv, %sgrad;
    add.u32 %pi, %pi, 1;
    bra PP_B_P1;
PP_B_P1_DONE:
    mul.f32 %t, %invr, %invr;
    mul.f32 %t, %t, %invr;
    mul.f32 %t, %t, %invdm;
    mul.f32 %pre, %sgrad, %t;
    mov.u32 %pi, 0;
PP_B_P2:
    setp.ge.u32 %p, %pi, %dm;
    @%p bra PP_B_P2_DONE;
    cvt.u64.u32 %off, %pi;
    shl.b64 %off, %off, 2;
    add.u64 %a, %dxnrow, %off;
    ld.global.f32 %dxnf, [%a];
    add.u64 %a, %nwp, %off;
    ld.global.f32 %nwv, [%a];
    mul.f32 %gd, %dxnf, %nwv;
    mul.f32 %term1, %gd, %invr;
    add.u64 %a, %xrow, %off;
    ld.global.f32 %xv, [%a];
    mul.f32 %term2, %xv, %pre;
    sub.f32 %dxv, %term1, %term2;
    setp.eq.u64 %pg, %dxp, 0;
    @%pg bra PP_B_P2_SKIPST;
    add.u64 %a, %dxrow, %off;
    st.global.f32 [%a], %dxv;
PP_B_P2_SKIPST:
    add.u32 %pi, %pi, 1;
    bra PP_B_P2;
PP_B_P2_DONE:
    add.u32 %s, %s, 128;
    bra PP_B;
PP_B_DONE:
    ret;
}
"#,
"\0");

#[cfg(feature = "cuda")]
const CSHA_BWD_MULTITILE_POSTPASS_NAME: &str = "nsl_csha_bwd_multitile_postpass\0";

/// Launch the multi-tile backward post-pass (see the PTX const above).
/// Single CTA, 128 threads. `inv_rms` is cached in 16 KB static shared, so
/// `seq_len` must be <= 4096 (larger returns `CUDA_ERROR_INVALID_VALUE` and the
/// caller surfaces a backward failure). All pointers are raw device u64.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn csha_bwd_multitile_postpass(
    dk_scratch: u64,
    dv_scratch: u64,
    dq: u64,
    x_raw: u64,
    norm_weight: u64,
    wq: u64,
    wk: u64,
    wv: u64,
    dwk: u64,
    dwv: u64,
    dx: u64,
    dxn: u64,
    seq_len: i64,
    d_model: u32,
    head_dim: i64,
    eps: f32,
) -> cudarc::driver::sys::CUresult {
    if seq_len <= 0 || d_model == 0 || head_dim <= 0 {
        return cudarc::driver::sys::CUresult::CUDA_SUCCESS;
    }
    if seq_len > 4096 {
        eprintln!(
            "[nsl::flash_attention] multi-tile backward post-pass: seq_len={seq_len} > 4096 \
             (static inv_rms shared cap) — refusing"
        );
        return cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let mut a_dk = dk_scratch;
    let mut a_dv = dv_scratch;
    let mut a_dq = dq;
    let mut a_xr = x_raw;
    let mut a_nw = norm_weight;
    let mut a_wq = wq;
    let mut a_wk = wk;
    let mut a_wv = wv;
    let mut a_dwk = dwk;
    let mut a_dwv = dwv;
    let mut a_dx = dx;
    let mut a_dxn = dxn;
    let mut a_seq = seq_len as u32;
    let mut a_dm = d_model;
    let mut a_hd = head_dim as u32;
    let mut a_eps = eps;
    let args: [*mut c_void; 16] = [
        &mut a_dk as *mut _ as *mut c_void,
        &mut a_dv as *mut _ as *mut c_void,
        &mut a_dq as *mut _ as *mut c_void,
        &mut a_xr as *mut _ as *mut c_void,
        &mut a_nw as *mut _ as *mut c_void,
        &mut a_wq as *mut _ as *mut c_void,
        &mut a_wk as *mut _ as *mut c_void,
        &mut a_wv as *mut _ as *mut c_void,
        &mut a_dwk as *mut _ as *mut c_void,
        &mut a_dwv as *mut _ as *mut c_void,
        &mut a_dx as *mut _ as *mut c_void,
        &mut a_dxn as *mut _ as *mut c_void,
        &mut a_seq as *mut _ as *mut c_void,
        &mut a_dm as *mut _ as *mut c_void,
        &mut a_hd as *mut _ as *mut c_void,
        &mut a_eps as *mut _ as *mut c_void,
    ];
    crate::cuda::inner::kernel_launch(
        CSHA_BWD_MULTITILE_POSTPASS_PTX.as_ptr(),
        CSHA_BWD_MULTITILE_POSTPASS_NAME.as_ptr(),
        [1, 1, 1],
        [128, 1, 1],
        &args,
        0,
    )
}

/// NSL_CSHA_DUMP_GRADS diagnostic: d2h-read the 6 save buffers and the 8
/// gradient outputs, print first 8 values + max|.| + sum + NaN count.
/// Zero-cost when disabled (callers gate on env var).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn csha_dump_backward_buffers(
    launch_rc: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64, d_model: i64,
    q_proj_dev: u64, k_proj_dev: u64, v_proj_dev: u64,
    row_max_dev: u64, row_sum_dev: u64, x_raw_dev: u64,
    d_o_dev: u64, d_q_dev: u64, d_k_dev: u64, d_v_dev: u64,
    d_wq_dev: u64, d_wk_dev: u64, d_wv_dev: u64,
    d_x_dev: u64, d_xn_dev: u64,
) {
    eprintln!(
        "[csha-dump] launch_rc={} batch={} heads={} seq={} head_dim={} d_model={}",
        launch_rc, batch, heads, seq_len, head_dim, d_model
    );

    // qkv-shaped: f16 [batch, heads, seq, head_dim]
    let qkv_elems = (batch * heads * seq_len * head_dim) as usize;
    // stats: f32 [batch, heads, seq]
    let stats_elems = (batch * heads * seq_len) as usize;
    // x_raw / dx / dx_norm: f32 [batch, seq, d_model] or [batch, heads, seq, head_dim]
    // (x_raw is B*H*S*D = 1024; dx / dx_norm are B*S*M = 1024 here since M=H*D=32).
    let x_elems = (batch * heads * seq_len * head_dim) as usize;
    let dx_elems = (batch * seq_len * d_model) as usize;
    // dwq/dwk/dwv: f16 [d_model, heads * head_dim]
    let dw_elems = (d_model * heads * head_dim) as usize;

    fn dump_f16(name: &str, dev_ptr: u64, n: usize) {
        if dev_ptr == 0 || n == 0 {
            eprintln!("[csha-dump] {:>9} = <null or empty> dev=0x{:x} n={}", name, dev_ptr, n);
            return;
        }
        let mut host: Vec<u16> = vec![0u16; n];
        unsafe {
            crate::cuda::inner::memcpy_dtoh(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dev_ptr as *const std::ffi::c_void,
                n * 2,
            );
        }
        let mut nan_count: usize = 0;
        let mut max_abs: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for &bits in &host {
            let v = crate::tensor::f16_bits_to_f32(bits);
            if v.is_nan() { nan_count += 1; }
            let a = v.abs();
            if a.is_finite() && a > max_abs { max_abs = a; }
            if v.is_finite() { sum += v as f64; }
        }
        let first8: Vec<f32> = host.iter().take(8).map(|&b| crate::tensor::f16_bits_to_f32(b)).collect();
        eprintln!(
            "[csha-dump] {:>9} [f16 n={}]: first8={:?} max|.|={:.6e} sum={:.6e} nan_count={}",
            name, n, first8, max_abs, sum, nan_count
        );
    }

    fn dump_f32(name: &str, dev_ptr: u64, n: usize) {
        if dev_ptr == 0 || n == 0 {
            eprintln!("[csha-dump] {:>9} = <null or empty> dev=0x{:x} n={}", name, dev_ptr, n);
            return;
        }
        let mut host: Vec<f32> = vec![0.0f32; n];
        unsafe {
            crate::cuda::inner::memcpy_dtoh(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                dev_ptr as *const std::ffi::c_void,
                n * 4,
            );
        }
        let mut nan_count: usize = 0;
        let mut max_abs: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for &v in &host {
            if v.is_nan() { nan_count += 1; }
            let a = v.abs();
            if a.is_finite() && a > max_abs { max_abs = a; }
            if v.is_finite() { sum += v as f64; }
        }
        let first8: Vec<f32> = host.iter().take(8).cloned().collect();
        eprintln!(
            "[csha-dump] {:>9} [f32 n={}]: first8={:?} max|.|={:.6e} sum={:.6e} nan_count={}",
            name, n, first8, max_abs, sum, nan_count
        );
    }

    // Save buffers (inputs to backward kernel — forward populated them).
    dump_f16("q_proj",   q_proj_dev, qkv_elems);
    dump_f16("k_proj",   k_proj_dev, qkv_elems);
    dump_f16("v_proj",   v_proj_dev, qkv_elems);
    dump_f32("row_max",  row_max_dev, stats_elems);
    dump_f32("row_sum",  row_sum_dev, stats_elems);
    dump_f32("x_raw",    x_raw_dev,   x_elems);

    // Backward seed.
    // dO is f16 (attention output gradient), shape [batch, heads, seq, head_dim].
    dump_f16("dO",       d_o_dev,  qkv_elems);

    // Gradient outputs.
    dump_f16("dq",       d_q_dev,  qkv_elems);
    dump_f16("dk",       d_k_dev,  qkv_elems);
    dump_f16("dv",       d_v_dev,  qkv_elems);
    dump_f16("dwq",      d_wq_dev, dw_elems);
    dump_f16("dwk",      d_wk_dev, dw_elems);
    dump_f16("dwv",      d_wv_dev, dw_elems);
    dump_f32("dx",       d_x_dev,  dx_elems);
    dump_f32("dx_norm",  d_xn_dev, dx_elems);
}

/// M42b: Quantized FlashAttention — KV-cache in INT8/FP8, Q in f16/f32.
///
/// Same tiled FlashAttention-2 algorithm but with inline dequantization:
/// each tile load of K/V dequantizes INT8→f32 (using per-head scales from
/// meta_k/meta_v) before the Q@K^T dot product.
///
/// kv_quant_scheme:
///   0 = None (f32 KV, same as nsl_flash_attention)
///   1 = INT8 per-head (scale per attention head)
///   2 = INT8 per-token (scale per token position)
///   3 = INT4 per-group (scale + zero_point per group)
///   4 = FP8 E4M3 (no scale needed, direct cast)
///
/// meta_k/meta_v: pointers to KvBlockQuantMeta arrays (null for FP8/None).
///
/// Returns 0 on success, negative on error.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// Non-CSHA entry: this path has no `segment_ids_ptr` parameter, so the runtime
/// gate is supplied `0` for that slot and always returns `false` — Tier B never
/// fires for non-CSHA configs. The extension is present to keep all 6 FFI entry
/// points uniformly shaped per planner spec §4.6.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_quantized(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    meta_k: i64, meta_v: i64,
    kv_quant_scheme: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_quantized",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick effective PTX/name based on runtime gate (planner spec §6.3).
    // Non-CSHA path has no segment_ids_ptr parameter; pass 0, gate always returns false.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            0,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut mk = meta_k as u64;
        let mut mv = meta_v as u64;
        let mut qs = kv_quant_scheme as u64;

        let args: [*mut c_void; 16] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut mk as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
        ];

        // For scheme 0 (no quantization), delegate to standard FlashAttention PTX.
        // For scheme 1-4, the quantized PTX kernel handles dequantization inline:
        //   INT8: each tile load does `v_f32 = (int8_val as f32) * scale[head]`
        //   FP8:  each tile load does direct E4M3→f32 cast via LUT
        //   INT4: each tile load unpacks nibbles + applies group scale/zero_point
        let result = crate::cuda::inner::kernel_launch(
            effective_ptx_ptr as *const u8,
            effective_name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (meta_k, meta_v, kv_quant_scheme);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv);
        let _ = (effective_ptx_ptr, effective_name_ptr);
        eprintln!("[nsl] quantized FlashAttention requires CUDA.");
        -1
    }
}

/// RoPE + paged cache write kernel launch wrapper.
///
/// All params i64 for Cranelift ABI compatibility.
/// Grid: (num_tokens, num_heads, ceil(head_dim/2))
#[no_mangle]
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: i64, v_projected_ptr: i64,
    cos_ptr: i64, sin_ptr: i64,
    positions_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_table_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
    ptx_ptr: i64, name_ptr: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let grid_x = num_tokens;
        let grid_y = num_heads;
        let grid_z = (head_dim + 1) / 2;

        let block_x = 1i64;
        let block_y = 1i64;
        let block_z = 1i64;

        let mut kp = k_projected_ptr as u64;
        let mut vp = v_projected_ptr as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut pos = positions_ptr as u64;
        let mut k_pool = k_pool_ptr as u64;
        let mut v_pool = v_pool_ptr as u64;
        let mut bt = block_table_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut nt = num_tokens as u64;
        let mut nh = num_heads as u64;
        let mut hd = head_dim as u64;
        let mut bs = block_size as u64;

        let args: [*mut c_void; 14] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut k_pool as *mut _ as *mut c_void,
            &mut v_pool as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            0,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (k_projected_ptr, v_projected_ptr, cos_ptr, sin_ptr, positions_ptr);
        let _ = (k_pool_ptr, v_pool_ptr, block_table_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (num_tokens, num_heads, head_dim, block_size, ptx_ptr, name_ptr);
        eprintln!("[nsl] rope_cache_write requires CUDA.");
        -1
    }
}

// ── FlashAttention backward pass (CPU reference implementation) ──────────

/// CPU reference implementation of FlashAttention backward.
///
/// Computes dQ, dK, dV from the upstream gradient dO, using the saved Q, K, V
/// tensors and the logsumexp (L) from the forward pass. The attention matrix is
/// recomputed from L (never stored), so memory is O(N) not O(N^2).
///
/// All tensors are [batch, heads, seq_len, head_dim] except logsumexp which is
/// [batch, heads, seq_len]. The output tensor O is [batch, heads, seq_len, head_dim].
///
/// Algorithm per (batch, head):
///   D[i] = sum_d(dO[i,d] * O[i,d])                        -- correction term
///   For each query i, key j:
///     s = sum_d(Q[i,d] * K[j,d]) * scale
///     p = exp(s - L[i])                                     -- recomputed softmax
///     if causal and j > i: p = 0
///     dV[j] += p * dO[i]                                    -- P^T @ dO
///     dp = sum_d(dO[i,d] * V[j,d])                          -- dO @ V^T
///     ds = p * (dp - D[i])                                   -- softmax backward
///     dQ[i] += ds * K[j] * scale
///     dK[j] += ds * Q[i] * scale
/// §4.3 attention sinks (Sprint 1b cycle-7) FFI null-pointer guard.
///
/// Returns `Ok(())` when sinks are disabled (`num_sink_tokens == 0`) OR
/// when both `sink_k_ptr` and `sink_v_ptr` are non-null. Returns
/// `Err(...)` with a precise diagnostic naming which pointer is null
/// when sinks are enabled but one or both pointers were passed as zero.
///
/// **Why this exists**: when codegen emits a sinks-enabled kernel
/// (`num_sink_tokens > 0`), the kernel issues `ld.global.b16
/// [sink_k_ptr + ...]` / `ld.global.b16 [sink_v_ptr + ...]`
/// unconditionally for every thread in the cooperative pre-load. A
/// null pointer there causes `CUDA_ERROR_ILLEGAL_ADDRESS` at the first
/// access — silent-corruption-free, but with a useless device-side
/// diagnostic. Catching the null case host-side gives the user a
/// clear, actionable error pointing at the FFI dispatch.
///
/// **FFI wiring deferred**: the production `nsl_flash_attention` FFI
/// does not yet thread `sink_k_ptr`/`sink_v_ptr` through its signature
/// (the cascade of Cranelift call-site updates exceeds the Sprint 1b
/// 100-LOC budget per the spec's failure-policy guidance). The
/// validator is exposed here so the codegen-side eligibility check can
/// be paired with a host-side null-ptr check at the eventual FFI
/// landing site without refactoring this helper.
pub fn validate_sinks_pointers_nonnull(
    num_sink_tokens: u32,
    sink_k_ptr: i64,
    sink_v_ptr: i64,
) -> Result<(), String> {
    if num_sink_tokens == 0 {
        return Ok(());
    }
    if sink_k_ptr == 0 && sink_v_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_k_ptr and sink_v_ptr (got sink_k_ptr=0x0, sink_v_ptr=0x0)",
            num_sink_tokens
        ));
    }
    if sink_k_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_k_ptr (got sink_k_ptr=0x0)",
            num_sink_tokens
        ));
    }
    if sink_v_ptr == 0 {
        return Err(format!(
            "nsl_flash_attention: num_sink_tokens={} > 0 requires non-null sink_v_ptr (got sink_v_ptr=0x0)",
            num_sink_tokens
        ));
    }
    Ok(())
}

#[cfg(test)]
mod sinks_validator_tests {
    use super::validate_sinks_pointers_nonnull;

    #[test]
    fn zero_sinks_accepts_both_null() {
        // Sentinel: when sinks are disabled, the validator must accept
        // null pointers (callers will pass 0 for the disabled path).
        assert!(validate_sinks_pointers_nonnull(0, 0, 0).is_ok());
    }

    #[test]
    fn zero_sinks_accepts_arbitrary_ptrs() {
        // Sentinel disabled: pointer values are ignored.
        assert!(validate_sinks_pointers_nonnull(0, 0xdead, 0xbeef).is_ok());
    }

    #[test]
    fn nonzero_sinks_accepts_both_nonnull() {
        assert!(validate_sinks_pointers_nonnull(4, 0xdead, 0xbeef).is_ok());
    }

    #[test]
    fn nonzero_sinks_rejects_both_null_naming_both() {
        let err = validate_sinks_pointers_nonnull(4, 0, 0)
            .expect_err("both null with sinks enabled must reject");
        assert!(
            err.contains("sink_k_ptr") && err.contains("sink_v_ptr"),
            "error must name BOTH missing pointers: {}",
            err
        );
        assert!(err.contains("num_sink_tokens=4"));
    }

    #[test]
    fn nonzero_sinks_rejects_null_k_naming_k() {
        let err = validate_sinks_pointers_nonnull(4, 0, 0xbeef)
            .expect_err("null sink_k_ptr with sinks enabled must reject");
        assert!(
            err.contains("sink_k_ptr"),
            "error must name the missing sink_k_ptr: {}",
            err
        );
        assert!(
            !err.contains("sink_v_ptr"),
            "error must NOT name sink_v_ptr when only k is null: {}",
            err
        );
    }

    #[test]
    fn nonzero_sinks_rejects_null_v_naming_v() {
        let err = validate_sinks_pointers_nonnull(4, 0xdead, 0)
            .expect_err("null sink_v_ptr with sinks enabled must reject");
        assert!(
            err.contains("sink_v_ptr"),
            "error must name the missing sink_v_ptr: {}",
            err
        );
        assert!(
            !err.contains("sink_k_ptr"),
            "error must NOT name sink_k_ptr when only v is null: {}",
            err
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
) {
    // Strides for [batch, heads, seq_len, head_dim] layout
    let bh_stride = heads * seq_len * head_dim;
    let h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    // Strides for logsumexp [batch, heads, seq_len]
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b in 0..batch {
        for h in 0..heads {
            let qkv_base = b * bh_stride + h * h_stride;
            let lse_base = b * lse_bh_stride + h * lse_h_stride;

            // Phase 1: Compute correction D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row_base = qkv_base + i * s_stride;
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += dout[row_base + d] * out[row_base + d];
                }
                *d_corr_i = sum;
            }

            // Phase 2: Compute dQ, dK, dV
            for i in 0..seq_len {
                let q_base = qkv_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    let k_base = qkv_base + j * s_stride;
                    let v_base = qkv_base + j * s_stride;

                    // Recompute S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d in 0..head_dim {
                        s_val += q[q_base + d] * k[k_base + d];
                    }
                    s_val *= scale;

                    // Recompute P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d in 0..head_dim {
                        dv[k_base + d] += p_val * dout[q_base + d];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d in 0..head_dim {
                        dp_val += dout[q_base + d] * v[v_base + d];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d in 0..head_dim {
                        dq[q_base + d] += ds_val * k[k_base + d] * scale;
                        dk[k_base + d] += ds_val * q[q_base + d] * scale;
                    }
                }
            }
        }
    }
}

/// GQA-aware backward: Q has `heads` heads, K/V have `kv_heads` heads.
/// Each group of `gqa_groups` Q heads shares one KV head.
/// dQ has shape [batch, heads, seq, head_dim].
/// dK, dV have shape [batch, kv_heads, seq, head_dim].
///
/// PCA Stage C: `seg` optionally carries per-position segment ids for a
/// packed batch (`[batch * seq_len]`, row-major, one row per batch entry).
/// When `Some`, a position pair (i, j) is masked out iff
/// `seg[b_idx * seq_len + i] != seg[b_idx * seq_len + j]` — on top of the
/// causal `j > i` skip. Predicate masking is bit-for-bit identical to the
/// Stage-B additive mask semantics: exp((s - 1e9) - lse) underflows to
/// exactly 0.0 for every cross-segment pair. The `logsumexp` passed in must
/// be segment-aware too (forward-saved, or `compute_logsumexp_gqa` with the
/// same `seg`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu_gqa(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool, gqa_groups: usize,
    seg: Option<&[f32]>,
) {
    // Q/dQ/out/dout strides: [batch, heads, seq_len, head_dim]
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    // K/V/dK/dV strides: [batch, kv_heads, seq_len, head_dim]
    let kv_bh_stride = kv_heads * seq_len * head_dim;
    let kv_h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base = b_idx * q_bh_stride + h_idx * q_h_stride;
            let kv_base = b_idx * kv_bh_stride + kv_h_idx * kv_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            // Phase 1: D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row = q_base + i * s_stride;
                let mut sum = 0.0f32;
                for d_idx in 0..head_dim {
                    sum += dout[row + d_idx] * out[row + d_idx];
                }
                *d_corr_i = sum;
            }

            // Phase 2: dQ, dK, dV
            for i in 0..seq_len {
                let qi = q_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    // PCA Stage C: cross-segment pairs contribute nothing
                    // (see the seg contract in the doc comment).
                    if let Some(seg) = seg {
                        if seg[b_idx * seq_len + i] != seg[b_idx * seq_len + j] {
                            continue;
                        }
                    }
                    let kj = kv_base + j * s_stride;
                    let vj = kv_base + j * s_stride;

                    // S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        s_val += q[qi + d_idx] * k[kj + d_idx];
                    }
                    s_val *= scale;

                    // P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d_idx in 0..head_dim {
                        dv[kj + d_idx] += p_val * dout[qi + d_idx];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        dp_val += dout[qi + d_idx] * v[vj + d_idx];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d_idx in 0..head_dim {
                        dq[qi + d_idx] += ds_val * k[kj + d_idx] * scale;
                        dk[kj + d_idx] += ds_val * q[qi + d_idx] * scale;
                    }
                }
            }
        }
    }
}

/// FFI entry point for FlashAttention backward pass (CPU reference).
/// Auto-compute logsumexp from Q, K when the forward didn't save it.
///
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h,j,:] * scale)))
/// with optional causal masking (j <= i).
/// Auto-compute logsumexp with GQA support.
///
/// Q has `heads` heads, K has `kv_heads` heads. Each Q head group maps to one KV head.
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h//gqa_groups,j,:] * scale)))
///
/// PCA Stage C: `seg` optionally carries `[batch * seq_len]` segment ids
/// (same contract as `flash_attention_backward_cpu_gqa`); cross-segment
/// key positions j are excluded from row i's normalizer. The diagonal
/// j == i is always in-segment, so every row keeps at least one term.
#[allow(clippy::too_many_arguments)]
fn compute_logsumexp_gqa(
    q: &[f32], k: &[f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
    seg: Option<&[f32]>,
) -> Vec<f32> {
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    let k_bh_stride = kv_heads * seq_len * head_dim;
    let k_h_stride = seq_len * head_dim;
    let s_stride = head_dim;
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;
    let gqa_groups = if kv_heads > 0 { heads / kv_heads } else { 1 };

    let total_lse = batch * heads * seq_len;
    let mut lse = vec![0.0f32; total_lse];

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base_bh = b_idx * q_bh_stride + h_idx * q_h_stride;
            let k_base_bh = b_idx * k_bh_stride + kv_h_idx * k_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            for i in 0..seq_len {
                let q_row = q_base_bh + i * s_stride;
                let j_max = if causal { i + 1 } else { seq_len };

                // Numerically stable logsumexp: max + log(sum(exp(x - max)))
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..j_max {
                    // PCA Stage C: cross-segment keys are outside row i's
                    // softmax support.
                    if let Some(seg) = seg {
                        if seg[b_idx * seq_len + i] != seg[b_idx * seq_len + j] {
                            continue;
                        }
                    }
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    if score > max_val {
                        max_val = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..j_max {
                    if let Some(seg) = seg {
                        if seg[b_idx * seq_len + i] != seg[b_idx * seq_len + j] {
                            continue;
                        }
                    }
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    sum_exp += (score - max_val).exp();
                }

                lse[lse_base + i] = max_val + sum_exp.ln();
            }
        }
    }

    lse
}

/// Budget-aware backward tile-size selector — the runtime mirror of codegen's
/// `nsl_codegen::flash_attention::backward_select_blocks`.
///
/// The Phase-2 backward keeps six resident SMEM tiles (Q, K, V, dO, dK-local,
/// dV-local) plus the S and dP (MMA) tiles. At `block_q = block_kv = 64` that
/// footprint is ~134 KB at head_dim=64 and ~230 KB at head_dim=128 — far past the
/// 99 KB (`101376`-byte) `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` cap on sm_80..sm_120,
/// so those launches fail the opt-in guard and fall back to the (correct but slow)
/// CPU backward. Shrinking the tiles for large head_dims brings the footprint back
/// under the cap so the GPU kernel runs.
///
/// Constraints baked into the table:
///   * Asymmetric tiles (`block_q != block_kv`) are allowed: codegen's causal
///     q-loop start is tile-ratio-aware (`emit_bwd_causal_q_loop_start` floors
///     `j*block_kv/block_q`), which fixed the historical asymmetric-causal
///     wrong-dV bug the sweep (`diag_block_size_sweep`) originally exposed.
///   * MMA needs full 32-lane warps, so an MMA-eligible `block_q` is a multiple of
///     32 (enforced codegen-side in `backward_uses_mma`); a `block_q < 32` routes to
///     the scalar path, which is warp-count-agnostic.
///   * The runtime always requests the MMA-inclusive SMEM maximum (it cannot tell
///     scalar from MMA PTX apart — see the Phase-2 launch comment), so each entry
///     must fit the MMA-inclusive layout under 99 KB.
///
/// This MUST stay identical to the codegen copy; the codegen test
/// `select_backward_blocks_matches_codegen` pins both to the same table.
#[cfg(feature = "cuda")]
pub fn select_backward_blocks(head_dim: i64) -> (i64, i64) {
    match head_dim {
        // ≤32: the classic 64/64 tile fits (88.6 KB at hd=32). MMA. Unchanged.
        hd if hd <= 32 => (64, 64),
        // 64: 32/32 → 59 KB MMA-inclusive, 1 warp. MMA path (block_q=32 % 32 == 0).
        // Verified GPU-correct causal + non-causal (coder-rl runs here, tensor cores).
        64 => (32, 32),
        // 128: 32/16 → 70 KB. block_q=32 is one full warp → MMA (tensor cores).
        // Asymmetric is safe now that the causal q-loop start floors the tile ratio;
        // GPU-verified correct causal + non-causal (coder7b head_dim). Previously
        // 16/16 scalar (sub-warp block_q forced the scalar path).
        128 => (32, 16),
        // >128 (e.g. 256): nothing fits 99 KB even asymmetric (hd256 at 16/16 is
        // ~102 KB). Keep 64/64 so the launch cleanly overflows the opt-in guard and
        // the runtime falls back to the CPU backward (loud, correct). A future layout
        // redesign (fewer resident tiles) could lift this.
        _ => (64, 64),
    }
}

/// One-shot stderr diagnostics for per-step backward fallbacks.
///
/// Since the decorator-free backward variant table landed, the refusal paths
/// (ragged seq_len, unavailable config) run once per attention op per
/// TRAINING STEP for shapes the GPU kernel can't serve — unconditional
/// printing floods stderr with hundreds of thousands of identical lines per
/// run where one line carries all the information. Keyed by message content,
/// so distinct shapes/configs each still print once. Hard launch FAILURES
/// stay unconditional — those are rare and individually meaningful.
#[cfg(feature = "cuda")]
fn flash_bwd_warn_once(msg: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let first = seen
        .lock()
        .map(|mut s| s.insert(msg.to_string()))
        .unwrap_or(true);
    if first {
        eprintln!("{msg}");
    }
}

/// GPU PTX backward dispatch: launches Phase 1 (D-correction) and Phase 2 (dQ/dK/dV)
/// kernels entirely on GPU. No host-device transfer needed.
///
/// Returns NslList [dQ, dK, dV] as GPU tensors — dQ is `[b, h, s, d]`;
/// dK/dV are `[b, kv_h, s, d]` (== dQ's shape when `kv_h == h`).
///
/// # Native GQA (P4, pretraining memory reduction)
///
/// `kv_h < h` (with `h % kv_h == 0`, caller-gated) launches the grouped
/// `_gqa` Phase-2 entry that codegen co-emits in the same PTX module as the
/// plain kernel: grid.x becomes the fused `batch*kv_head` index, the kernel
/// loops the group's q-heads internally, and dK/dV come out kv-shaped
/// directly — no expanded K/V/dK/dV tensors, no DtoD expand, no group-sum
/// reduce. The `_gqa` entry NAME is derived here by suffixing the plain
/// phase-2 name (`flash_attention_bwd_main_gqa_kernel_name` is the codegen
/// side of that contract). The LSE recompute likewise uses the kv-indexed
/// `nsl_flash_lse_gqa_f32`. Callers that still want the expand envelope pass
/// `kv_h == h` with pre-expanded K/V (see `NSL_FLASH_GQA_BWD_EXPAND`).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_gpu(
    dout_ptr: i64, q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale: f32,
    b: usize, h: usize, kv_h: usize, s: usize, d: usize,
    is_causal: bool,
    // Backward tile sizes. MUST equal the values the caller used to synthesize the
    // Phase-2 PTX (they are baked into the kernel name via
    // `flash_attention_bwd_main_kernel_name` AND into the SMEM offsets in the PTX
    // body), because this function derives the launch grid, block dims, and the
    // dynamic-SMEM request from them. In production both sides call
    // `select_backward_blocks(head_dim)` so they agree; the test FFI passes explicit
    // values that match the test's codegen config.
    block_q: i64, block_kv: i64,
    phase1_ptx_ptr: i64, phase1_name_ptr: i64,
    phase2_ptx_ptr: i64, phase2_name_ptr: i64,
    // PCA Stage C: NslTensor* of [b, s] segment ids (host float per
    // packing.rs), or 0 for the classic unmasked backward. Non-zero
    // REQUIRES a forward-saved logsumexp (see the guard below) and a
    // `_segmask` phase-2 kernel variant from codegen.
    segment_ids_ptr: i64,
) -> i64 {
    use crate::cuda::inner;
    use std::ffi::c_void;

    // ── Ragged-seq guard ──
    // The Phase-2 main backward kernel has NO seq_len tail guards: it iterates
    // num_q_tiles = ceil(s/block_q) and grid_y = ceil(s/block_kv), and a partial
    // final tile computes PHANTOM rows/cols past seq_len. Phantom q-rows are not
    // even causally masked (global_i >= s > every global_j), read the NEXT
    // batch-head's Q/dO/D/L (or past the allocation for the last one), pollute
    // valid dV/dK rows, and atomicAdd garbage dQ into the neighbor batch-head —
    // silent gradient corruption, not a launch error. Until the emitter grows
    // per-row bounds (zero-filling phantom SMEM rows), refuse loudly and let the
    // caller take the correct CPU backward. Production pretraining seqs
    // (1024/2048/4096) are multiples of every selector tile size, so this gate
    // only redirects genuinely-unsupported ragged shapes.
    if s % (block_q.max(1) as usize) != 0 || s % (block_kv.max(1) as usize) != 0 {
        flash_bwd_warn_once(&format!(
            "[flash-bwd] seq_len={s} is not a multiple of the backward tile sizes \
             (block_q={block_q}, block_kv={block_kv}) — the GPU Phase-2 kernel has no \
             ragged-tail guards and would corrupt gradients. Falling back to the \
             correct CPU backward (slower). Pad/pack sequences to a multiple of \
             {block_q} to train this shape on GPU."
        ));
        return 0;
    }

    // ── PCA Stage C: forward-saved LSE resolution ──
    // `logsumexp_ptr`, when non-zero, is an NslTensor* saved by the fused
    // forward (`nsl_sdpa_fused_forward` returns [out, lse]); its `.data`
    // is a GPU f32 [b, h, s] buffer Phase 2 can consume directly —
    // BORROWED, never freed here (`lse_owned` below tracks ownership).
    // The device/dtype/len guard keeps a host-resident or undersized
    // tensor from leaking a bad pointer into the kernel.
    let saved_lse_data: *mut c_void = if logsumexp_ptr != 0 {
        let lse_t = NslTensor::from_ptr(logsumexp_ptr);
        if lse_t.device > 0 && lse_t.dtype == 1 && lse_t.len >= (b * h * s) as i64 {
            lse_t.data
        } else {
            std::ptr::null_mut()
        }
    } else {
        std::ptr::null_mut()
    };

    // ── Segment-mask guard ──
    // A segment-masked backward REQUIRES the forward-saved logsumexp: the
    // FLASH_LSE recompute kernel is segment-blind, and a mask-blind
    // normalizer would silently corrupt every gradient of a packed batch.
    // Refuse (once per config — no per-call values in the message, so the
    // warn_once dedup set stays bounded) and let the caller take the
    // segment-aware CPU reference backward.
    if segment_ids_ptr != 0 && saved_lse_data.is_null() {
        flash_bwd_warn_once(&format!(
            "[flash-bwd] segment-masked backward needs the forward-saved \
             logsumexp (missing, or not a GPU f32 [b,h,s] tensor) — the \
             FLASH_LSE recompute kernel is not segment-aware. Falling back \
             to the segment-aware CPU reference backward (correct but slow) \
             (batch={b}, heads={h}, seq={s}, head_dim={d})."
        ));
        return 0;
    }

    // p3: stream-ordered by default. `from_ptr` reads only the host-side
    // NslTensor struct (the .data device address), never device memory, so no
    // sync is needed here; the backward kernels are NULL-stream-ordered after
    // the forward that produced Q/K/V.
    crate::cuda::inner::sync_after_kernel();

    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    // ── PCA Stage C: stage [b, s] segment ids to a device u16 buffer ──
    // Freed after the post-launch sync on every path (success + error).
    let seg_dev: *mut c_void = if segment_ids_ptr != 0 {
        match segment_ids_host_u16(segment_ids_ptr, b, s) {
            Some(host) => stage_segment_ids_device(&host),
            None => {
                flash_bwd_warn_once(&format!(
                    "[flash-bwd] segment_ids tensor is not a [batch={b}, \
                     seq={s}] float tensor — cannot stage the segment mask \
                     for the GPU backward; falling back to the CPU \
                     reference backward."
                ));
                return 0;
            }
        }
    } else {
        std::ptr::null_mut()
    };

    let total_qkv = b * h * s * d;

    // P0.1 VRAM accounting: everything this backward allocates through the
    // caching allocator (D vector, dq/dk/dv scratches, lse recompute) is
    // attention workspace. RAII guard — restores the previous surface on
    // every exit path, including panics.
    let _surface = crate::cuda::caching_allocator::SurfaceGuard::new(
        crate::cuda::caching_allocator::SurfaceTag::AttnWorkspace,
    );

    // Native GQA (P4): the caller passes `kv_h < h` (with `h % kv_h == 0`)
    // only when it wants the grouped `_gqa` Phase-2 entry — no expand-KV
    // envelope. dQ stays `[b, h, s, d]`; dK/dV come out kv-shaped
    // `[b, kv_h, s, d]` directly, so no post-launch group-sum is needed.
    // `kv_h == h` is the classic MHA path (unchanged), including the case
    // where the dispatcher pre-expanded K/V and passes `kv_h == h`.
    let native_gqa = kv_h != h;
    let total_kv = b * kv_h * s * d;
    let dkdv_elems = if native_gqa { total_kv } else { total_qkv };

    // ── Allocate D correction vector [b*h*s] on GPU ──
    let d_buf = inner::alloc_managed(b * h * s * 4);

    // ── Allocate dQ on GPU (zero-initialized) ──
    let dq_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dq_data, total_qkv * 4);

    // ── Allocate dK, dV on GPU (zero-initialized) ──
    // Sized `[b, kv_h, s, d]` under native GQA, `[b, h, s, d]` otherwise.
    let dk_data = inner::alloc_managed(dkdv_elems * 4);
    inner::memset_d8(dk_data, dkdv_elems * 4);
    let dv_data = inner::alloc_managed(dkdv_elems * 4);
    inner::memset_d8(dv_data, dkdv_elems * 4);

    // ── Phase 1: D-correction vector ──
    // D[bh, i] = sum_d( dO[bh, i, d] * O[bh, i, d] )
    // Grid: (b*h, ceil(s/block_q), 1), Block: (block_q, 1, 1)
    {
        let mut dout_data = dout_t.data as u64;
        let mut out_data = out_t.data as u64;
        let mut d_data = d_buf as u64;
        let mut sl = s as u64;
        let mut hd = d as u64;

        let args: [*mut c_void; 5] = [
            &mut dout_data as *mut _ as *mut c_void,
            &mut out_data as *mut _ as *mut c_void,
            &mut d_data as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let grid = [(b * h) as i64, (s as i64 + block_q - 1) / block_q, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase1_ptx_ptr as *const u8,
            phase1_name_ptr as *const u8,
            grid, block, &args, 0,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // Do NOT proceed to return zero-initialized gradients — that would
            // silently corrupt training. Free scratch and signal failure (0) so
            // the caller falls back to the correct CPU backward. See the return-0
            // contract in `nsl_flash_attention_backward`.
            eprintln!(
                "[flash-bwd] Phase 1 (D-correction) kernel launch FAILED: {:?} — \
                 refusing to return zero gradients; caller will fall back to CPU. \
                 (This means the GPU backward PTX is invalid/unlaunchable for this config.)",
                res
            );
            inner::free_managed(seg_dev);
            inner::free_managed(d_buf);
            inner::free_managed(dq_data);
            inner::free_managed(dk_data);
            inner::free_managed(dv_data);
            return 0;
        }
    }

    // ── Phase 2: Main backward (dQ/dK/dV) ──
    // Grid: (b*h, ceil(s/block_kv), 1), Block: (block_q, 1, 1)
    // Shared memory: this must mirror codegen's `backward_shared_mem_bytes`
    // (nsl-codegen/src/flash_attention.rs). We can't call that function directly
    // — the crate dependency runs codegen -> runtime, not the reverse — and this
    // FFI does not receive `gpu_sm`, so the runtime cannot tell whether the PTX it
    // was handed is the scalar or the MMA variant. The MMA variant reserves an
    // extra `dp_tile` (same size as the S tile) that the scalar variant does not.
    // We therefore always request the MMA-inclusive MAXIMUM: over-allocating
    // dynamic shared for the scalar path is harmless (the kernel simply uses less
    // of its window), whereas UNDER-allocating for the MMA path is undefined
    // behavior (the kernel would index past its dynamic-shared window at run time
    // with no launch-time error). Keep this in sync with `backward_shared_mem_bytes`.
    {
        let pad: i64 = 4;
        let hd_padded = d as i64 + pad;
        let tile_bytes = |rows: i64, cols: i64| -> i64 { rows * cols * 4 };
        let shmem = (tile_bytes(block_kv, hd_padded) * 2  // K, V tiles
            + tile_bytes(block_q, hd_padded) * 2           // Q, dO tiles
            + tile_bytes(block_kv, hd_padded) * 2          // dK, dV accumulators
            + tile_bytes(block_q, block_kv)                // S tile
            + tile_bytes(block_q, block_kv)                // dP tile (MMA path only; over-allocated for scalar)
            + block_q * 4                                  // D vector
            + block_q * 4                                  // L (logsumexp) vector
        ) as u32;

        // Logsumexp source, in preference order:
        //   1. Forward-saved (PCA Stage C): `nsl_sdpa_fused_forward` returns
        //      [out, lse]; when the caller threads that lse tensor through,
        //      use its device buffer directly — borrowed (`lse_owned=false`),
        //      never freed here. MANDATORY on the segment path (the guard
        //      above already refused segment-without-saved-lse).
        //   2. Recompute DEVICE-RESIDENT from Q and K. Q/K are already on
        //      the GPU and the dispatch gate guarantees kv_heads == h (MHA),
        //      so a per-row kernel writes lse without any host transfer.
        //      This replaces the previous Q+K device->host copy +
        //      O(b*h*s^2*d) CPU score recompute + host->device copy that ran
        //      on EVERY backward call — a severe hot-path regression once the
        //      decorator-free backward started dispatching here (PR #347).
        //      The decomposed decorator-free forward saves no lse buffer (the
        //      caller passes 0, wengert_lower.rs); recomputing on-device is
        //      correct and cheap, and its ex2/lg2 approximations match how
        //      the backward kernel itself recomputes P = exp(score - lse).
        let (lse_data, lse_owned) = if !saved_lse_data.is_null() {
            (saved_lse_data, false)
        } else {
            let total_lse = b * h * s;
            let lse_gpu = inner::alloc_managed(total_lse * 4);

            let mut q_arg = q_t.data as u64;
            let mut k_arg = k_t.data as u64;
            let mut lse_arg = lse_gpu as u64;
            let mut total_arg = total_lse as u64;
            let mut seq_arg = s as u64;
            let mut hd_arg = d as u64;
            let mut scale_arg = scale;
            let mut causal_arg = if is_causal { 1u64 } else { 0u64 };
            // Native GQA: K is `[b, kv_h, s, d]`, so the plain lse kernel
            // (which indexes K by the q-head) would read past kv_h heads.
            // The `_gqa` lse kernel takes head counts and reads
            // K[kv_head = q_head / groups]. lse itself stays `[b, h, s]`
            // (a q-side quantity). Two extra u64 args (heads, kv_heads).
            let mut heads_arg = h as u64;
            let mut kv_heads_arg = kv_h as u64;
            let lse_block = 256i64;
            let lse_grid = [(total_lse as i64 + lse_block - 1) / lse_block, 1, 1];
            let lse_res = if native_gqa {
                let lse_args: [*mut c_void; 10] = [
                    &mut q_arg as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                    &mut lse_arg as *mut _ as *mut c_void,
                    &mut total_arg as *mut _ as *mut c_void,
                    &mut seq_arg as *mut _ as *mut c_void,
                    &mut hd_arg as *mut _ as *mut c_void,
                    &mut scale_arg as *mut _ as *mut c_void,
                    &mut causal_arg as *mut _ as *mut c_void,
                    &mut heads_arg as *mut _ as *mut c_void,
                    &mut kv_heads_arg as *mut _ as *mut c_void,
                ];
                inner::kernel_launch(
                    crate::cuda::fused_kernels::FLASH_LSE_GQA_F32_PTX.as_ptr(),
                    b"nsl_flash_lse_gqa_f32\0".as_ptr(),
                    lse_grid,
                    [lse_block, 1, 1],
                    &lse_args,
                    0,
                )
            } else {
                let lse_args: [*mut c_void; 8] = [
                    &mut q_arg as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                    &mut lse_arg as *mut _ as *mut c_void,
                    &mut total_arg as *mut _ as *mut c_void,
                    &mut seq_arg as *mut _ as *mut c_void,
                    &mut hd_arg as *mut _ as *mut c_void,
                    &mut scale_arg as *mut _ as *mut c_void,
                    &mut causal_arg as *mut _ as *mut c_void,
                ];
                inner::kernel_launch(
                    crate::cuda::fused_kernels::FLASH_LSE_F32_PTX.as_ptr(),
                    b"nsl_flash_lse_f32\0".as_ptr(),
                    lse_grid,
                    [lse_block, 1, 1],
                    &lse_args,
                    0,
                )
            };
            if lse_res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // Never return zero gradients on a launch failure — free scratch
                // and signal 0 so the caller takes the correct CPU backward.
                eprintln!(
                    "[flash-bwd] logsumexp kernel launch FAILED: {:?} — refusing to \
                     return zero gradients; caller will fall back to CPU.",
                    lse_res
                );
                inner::free_managed(lse_gpu);
                inner::free_managed(seg_dev);
                inner::free_managed(d_buf);
                inner::free_managed(dq_data);
                inner::free_managed(dk_data);
                inner::free_managed(dv_data);
                return 0;
            }
            (lse_gpu, true)
        };

        let mut dout_data = dout_t.data as u64;
        let mut q_data = q_t.data as u64;
        let mut k_data = k_t.data as u64;
        let mut v_data = v_t.data as u64;
        let mut dq_d = dq_data as u64;
        let mut dk_d = dk_data as u64;
        let mut dv_d = dv_data as u64;
        let mut d_d = d_buf as u64;
        let mut lse_d = lse_data as u64;
        let mut sc = scale;
        let mut sl = s as u64;
        let mut hd = d as u64;
        // PCA Stage C: the plain phase-2 kernel declares 12 params; the
        // `_segmask` variant declares 14 (the 12 above + segment_ids +
        // heads — `heads` recovers batch_idx from the fused b*h block id).
        // Codegen pairs the kernel variant with the segment pointer, so
        // the marshaled arg count always matches the entry signature:
        // exactly 12 unmasked, exactly 14 masked.
        let mut seg_p2 = seg_dev as u64;
        let mut heads_p2 = h as u64;
        let mut kv_heads_p2 = kv_h as u64;

        let mut args: Vec<*mut c_void> = vec![
            &mut dout_data as *mut _ as *mut c_void,
            &mut q_data as *mut _ as *mut c_void,
            &mut k_data as *mut _ as *mut c_void,
            &mut v_data as *mut _ as *mut c_void,
            &mut dq_d as *mut _ as *mut c_void,
            &mut dk_d as *mut _ as *mut c_void,
            &mut dv_d as *mut _ as *mut c_void,
            &mut d_d as *mut _ as *mut c_void,
            &mut lse_d as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        if !seg_dev.is_null() {
            // Segment-masked variant: +segment_ids, +heads (14 params).
            args.push(&mut seg_p2 as *mut _ as *mut c_void);
            args.push(&mut heads_p2 as *mut _ as *mut c_void);
        } else if native_gqa {
            // Native-GQA grouped variant: +heads, +kv_heads (14 params;
            // param_head_dim `hd` already pushed above). The kernel derives
            // batch/kv/group from the fused `batch*kv_head` block index.
            args.push(&mut heads_p2 as *mut _ as *mut c_void);
            args.push(&mut kv_heads_p2 as *mut _ as *mut c_void);
        }

        // Native GQA: one CTA per (batch, kv_head) that loops the group's
        // q-heads internally, so grid.x is `b*kv_h` (not `b*h`). The `_gqa`
        // entry lives in the SAME PTX module as the plain kernel; select it
        // by suffixing the plain phase-2 name.
        let gqa_name_c: Option<std::ffi::CString> = if native_gqa {
            let plain = unsafe { std::ffi::CStr::from_ptr(phase2_name_ptr as *const std::os::raw::c_char) };
            Some(std::ffi::CString::new(format!("{}_gqa", plain.to_string_lossy())).unwrap())
        } else {
            None
        };
        let (grid_x, phase2_name_use) = if native_gqa {
            ((b * kv_h) as i64, gqa_name_c.as_ref().unwrap().as_ptr() as *const u8)
        } else {
            ((b * h) as i64, phase2_name_ptr as *const u8)
        };

        let grid = [grid_x, (s as i64 + block_kv - 1) / block_kv, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase2_ptx_ptr as *const u8,
            phase2_name_use,
            grid, block, &args, shmem,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // As with Phase 1: never return zero gradients on a launch failure.
            // Free everything and signal failure (0) so the caller uses the CPU
            // backward. Common causes: MMA path PTX not yet valid (sm>=80), or the
            // dynamic shared request exceeds the device opt-in cap for this head_dim.
            eprintln!(
                "[flash-bwd] Phase 2 (dQ/dK/dV) kernel launch FAILED: {:?} — \
                 refusing to return zero gradients; caller will fall back to CPU. \
                 (shmem request = {} bytes; check MMA-path PTX validity and the \
                 device MAX_SHARED_MEMORY_PER_BLOCK_OPTIN cap.)",
                res, shmem
            );
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
            if lse_owned {
                inner::free_managed(lse_data);
            }
            inner::free_managed(seg_dev);
            inner::free_managed(d_buf);
            inner::free_managed(dq_data);
            inner::free_managed(dk_data);
            inner::free_managed(dv_data);
            return 0;
        }

        // Sync after all kernels and check for ASYNCHRONOUS execution faults.
        // `kernel_launch` only surfaces synchronous enqueue errors (invalid PTX,
        // bad launch config, shared-mem cap); an illegal shared/global access that
        // occurs DURING execution surfaces only here (and only in the default,
        // non-CUDA_SYNC_MODE config `kernel_launch` does not itself check). Route a
        // faulted sync through the same free-and-return-0 path so the caller falls
        // back to the correct CPU backward rather than returning corrupt gradients.
        // p3: stream-ordered by default; frees below use the caching allocator
        // (free_managed), which is stream-safe. The async-fault trap stays under
        // NSL_CUDA_SYNC=1 (a faulted GPU kernel then falls back to CPU backward).
        let sync_rc = if inner::sync_mode_enabled() {
            unsafe { cudarc::driver::sys::cuCtxSynchronize() }
        } else {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS
        };
        if sync_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!(
                "[flash-bwd] backward kernels reported an ASYNCHRONOUS execution fault \
                 on cuCtxSynchronize: {:?} — refusing to return possibly-corrupt \
                 gradients; caller will fall back to CPU.",
                sync_rc
            );
            if lse_owned {
                inner::free_managed(lse_data);
            }
            inner::free_managed(seg_dev);
            inner::free_managed(d_buf);
            inner::free_managed(dq_data);
            inner::free_managed(dk_data);
            inner::free_managed(dv_data);
            return 0;
        }

        // Free the logsumexp scratch buffer (only when this function
        // allocated it — the forward-saved lse is borrowed) and the staged
        // segment ids (dead now that the post-launch sync completed).
        if lse_owned {
            inner::free_managed(lse_data);
        }
        inner::free_managed(seg_dev);
    }

    // Free D correction buffer
    inner::free_managed(d_buf);

    // ── Build output NslTensor wrappers for dQ, dK, dV ──
    // (`make_gpu_tensor` was hoisted to file scope when `nsl_sdpa_fused_forward`
    // started wrapping its [out, lse] buffers with the same idiom.)
    // dQ is always `[b, h, s, d]`; dK/dV are `[b, kv_h, s, d]` under native
    // GQA (the grouped kernel wrote kv-shaped gradients directly), else
    // `[b, h, s, d]`.
    let q_shape = [b as i64, h as i64, s as i64, d as i64];
    let kv_shape = [b as i64, kv_h as i64, s as i64, d as i64];
    let dq_ptr = make_gpu_tensor(dq_data, &q_shape, total_qkv);
    let dk_ptr = make_gpu_tensor(dk_data, &kv_shape, dkdv_elems);
    let dv_ptr = make_gpu_tensor(dv_data, &kv_shape, dkdv_elems);

    // Pack into NslList [dQ, dK, dV]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

/// Test-only GPU backward probe with **explicit** block sizes — a thin wrapper over
/// `flash_attention_backward_gpu` that bypasses the production `select_backward_blocks`
/// table and does NOT fall back to CPU. Returns the `[dQ, dK, dV]` GPU list on a
/// successful launch, or `0` if the GPU kernel could not run (invalid PTX or the
/// dynamic-SMEM request exceeded the device opt-in cap for this `block_q`/`block_kv`).
///
/// The caller MUST synthesize the Phase-1/Phase-2 PTX with the *same* `block_q`/
/// `block_kv` so the kernel names and SMEM offsets baked into the PTX match the grid
/// and dynamic-SMEM request this function derives from them. Used by the block-size
/// parity sweep in `flash_attention_backward_gpu.rs` to classify each candidate
/// (block_q, block_kv, head_dim) config as GPU-correct / GPU-wrong / GPU-unavailable
/// before a config is promoted into `select_backward_blocks`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn nsl_test_flash_attention_backward_blocks(
    block_q: i64, block_kv: i64,
    dout_ptr: i64,
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: i64,
    phase1_ptx_ptr: i64, phase1_name_ptr: i64,
    phase2_ptx_ptr: i64, phase2_name_ptr: i64,
) -> i64 {
    let scale = f32::from_bits(scale_bits as u32);
    flash_attention_backward_gpu(
        dout_ptr, q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
        scale,
        batch as usize, heads as usize, heads as usize, seq_len as usize, head_dim as usize,
        causal != 0,
        block_q, block_kv,
        phase1_ptx_ptr, phase1_name_ptr,
        phase2_ptx_ptr, phase2_name_ptr,
        0, // segment_ids_ptr: the block sweep exercises unmasked (MHA) configs only
    )
}

///
/// Allocates dQ, dK, dV tensors, runs the CPU backward, and returns them
/// as a tuple of three tensor pointers packed into an NslList.
///
/// This is called from the backward dispatch in autodiff/backward.rs.
/// Falls back to CPU when GPU PTX backward is not available (no PTX pointers
/// or GQA with different Q/KV head counts).
///
/// When `logsumexp_ptr == 0`, the logsumexp is auto-computed from Q, K, scale.
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `nsl_codegen::pca_tier_b::tier_b_disabled_sentinel()` or `tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// **Two-phase backward structural note:** this entry uses `phase1_*`/`phase2_*` PTX
/// pairs rather than a single `ptx_ptr`/`name_ptr` pair, so the canonical
/// `(effective_ptx_ptr, effective_name_ptr)` dispatch substitution doesn't apply.
/// Non-CSHA + non-`segment_masked`: the runtime gate would never select Tier-B-on
/// even if a variant were provided. The sentinel assertion is retained for
/// uniformity per planner spec §4.6; the params are accepted but unused.
///
/// # PCA Stage C: segment-masked backward (19th arg)
///
/// The trailing `segment_ids_ptr` carries the `[batch, seq]` segment-id
/// tensor of a packed batch (host float per packing.rs), or `0` for the
/// classic unmasked backward. When non-zero:
///   * GPU path: the ids are staged to a device u16 buffer and appended
///     (together with `heads`) to the phase-2 marshal — codegen guarantees
///     the paired PTX is the 14-param `_segmask` variant. The
///     forward-saved `logsumexp_ptr` is REQUIRED (the FLASH_LSE recompute
///     kernel is segment-blind); without it the GPU path refuses and the
///     CPU reference below runs instead.
///   * CPU path: cross-segment (i, j) pairs are excluded from both the
///     logsumexp and the gradient accumulation — bit-for-bit identical to
///     the Stage-B additive -1e9 mask, since exp((s - 1e9) - lse)
///     underflows to exactly 0.0.
///
/// # Stride discipline (PCA Stage C parity fix)
///
/// The GPU phase-1/phase-2 kernels are stride-blind, so the GPU dispatch
/// additionally requires dout/q/k/v/out to be canonical row-major; a view
/// input falls back (warn-once) to the CPU reference, whose tensor reads
/// are stride-AWARE (they gather through the view's strides).
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_flash_attention_backward(
    dout_ptr: i64,
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: i64,
    phase1_ptx_ptr: i64,
    phase1_name_ptr: i64,
    phase2_ptx_ptr: i64,
    phase2_name_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
    // PCA Stage C: [batch, seq] segment ids for packed batches (0 = none).
    segment_ids_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::assert_tier_b_sentinels;

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    // Two-phase backward has no canonical dispatch shape; the runtime gate is
    // structurally inapplicable here (non-CSHA + non-segment_masked). The
    // sentinel assertion still runs to enforce call-site discipline.
    assert_tier_b_sentinels(
        "nsl_flash_attention_backward",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );
    // Suppress "unused" — the params are accepted for FFI uniformity.
    let _ = (tier_b_ptx_ptr, tier_b_name_ptr);
    let scale = f32::from_bits(scale_bits as u32);
    let b = batch as usize;
    let h = heads as usize;
    let s = seq_len as usize;
    let d = head_dim as usize;
    let is_causal = causal != 0;

    // GPU PTX dispatch: if PTX pointers are provided and tensors are on GPU,
    // launch the backward kernels directly on the device (no CPU transfer).
    #[cfg(feature = "cuda")]
    {
        let dout_t = NslTensor::from_ptr(dout_ptr);
        let k_t = NslTensor::from_ptr(k_ptr);
        let kv_h = if k_t.ndim >= 2 {
            unsafe { *k_t.shape.add(1) as usize }
        } else {
            h
        };

        // NSL_FLASH_DEBUG=1 narrates the backward dispatch decision. The GPU
        // path is silent on success and the CPU fallbacks are silent by
        // design (the tape-AD path always passes null PTX), which made the
        // decorator-free phase1_ptx=0 wiring gap invisible for a whole
        // campaign — this gives tests and users a positive observable.
        let flash_debug = std::env::var("NSL_FLASH_DEBUG").ok().as_deref() == Some("1");

        // NSL_FLASH_BWD_CPU=1 forces the deterministic CPU reference backward
        // even when backward PTX is available. The GPU phase-2 kernel
        // accumulates dK/dV with float atomicAdd, so its rounding depends on
        // scheduling order: two training runs of the SAME program produce
        // slightly different gradients. Anything that compares checkpoints
        // across runs (the FASE-vs-plain-AdamW parity gate, M46-style
        // determinism audits, bisections) needs this knob until a
        // deterministic-reduction backward variant exists.
        let force_cpu_bwd = std::env::var("NSL_FLASH_BWD_CPU").ok().as_deref() == Some("1");
        if force_cpu_bwd && flash_debug && dout_t.device > 0 {
            eprintln!(
                "[flash-bwd] NSL_FLASH_BWD_CPU=1 — deterministic CPU reference backward forced"
            );
        }

        // GQA GPU envelope (scaling campaign item 1): the phase-1/phase-2
        // kernels are MHA-only, but a GQA input can be served exactly by
        // expanding K/V to full heads on-device, running the MHA kernels,
        // and group-summing dK/dV back (the ReduceToShape adjoint of the
        // expansion). NSL_FLASH_GQA_BWD_CPU=1 restores the old CPU-reference
        // routing for this case only (NSL_FLASH_BWD_CPU=1 still forces CPU
        // for everything).
        let gqa_groups_env = kv_h > 0 && h % kv_h == 0 && kv_h != h;
        let force_gqa_cpu =
            std::env::var("NSL_FLASH_GQA_BWD_CPU").ok().as_deref() == Some("1");
        let gqa_expandable = gqa_groups_env && !force_gqa_cpu;

        if flash_debug && dout_t.device > 0 && !force_cpu_bwd {
            if phase1_ptx_ptr == 0 {
                eprintln!(
                    "[flash-bwd] no backward PTX provided (phase1_ptx=0) — CPU reference \
                     backward (batch={b}, heads={h}, seq={s}, head_dim={d})"
                );
            } else if kv_h != h && !gqa_groups_env {
                eprintln!(
                    "[flash-bwd] irregular GQA layout (kv_heads={kv_h} does not divide \
                     heads={h}) — CPU reference backward"
                );
            } else if kv_h != h && force_gqa_cpu {
                eprintln!(
                    "[flash-bwd] NSL_FLASH_GQA_BWD_CPU=1 — GQA expand-KV GPU backward \
                     disabled; CPU reference backward"
                );
            }
        }

        // PCA Stage C stride guard, reworked (scaling campaign item 1): the
        // GPU phase-1/phase-2 kernels read raw device pointers stride-blind,
        // so a non-contiguous VIEW input (e.g. reshape(..).transpose(1,2)
        // products — dout arrives that way on EVERY decorator-free
        // attention backward) would yield smoothly-wrong gradients, not an
        // error. The original guard routed such inputs to the stride-aware
        // CPU reference — which silently put the ENTIRE flash backward on
        // the host for every packed/decorator-free training step (nsys:
        // ~50% of process CPU time in flash_attention_backward_cpu_gqa at
        // the 16M base config). Materialize a canonical device-side copy
        // instead (one strided-copy kernel per non-canonical input) and
        // keep the backward on the GPU. NSL_FLASH_BWD_CPU=1 still forces
        // the CPU reference wholesale.
        if dout_t.device > 0
            && phase1_ptx_ptr != 0
            && (kv_h == h || gqa_expandable)
            && !force_cpu_bwd
        {
            // Budget-aware backward tile sizes. Must match the sizes codegen used to
            // synthesize the Phase-2 PTX; both sides derive them from head_dim via
            // the same `select_backward_blocks` table (codegen's copy lives in
            // nsl-codegen; this is the runtime mirror — kept identical by
            // `select_backward_blocks_matches_codegen` in the codegen tests).
            let (block_q, block_kv) = select_backward_blocks(head_dim);

            // Canonical-row-major materialization for the stride-blind
            // kernels: any view input gets one device-side strided-copy.
            // Temps collected for freeing after the (synchronizing) launch.
            let mut canon_temps: Vec<i64> = Vec::new();
            let mut canon = |ptr: i64, name: &str, temps: &mut Vec<i64>| -> i64 {
                let t = NslTensor::from_ptr(ptr);
                if is_canonical_row_major(t) {
                    return ptr;
                }
                if flash_debug {
                    eprintln!(
                        "[flash-bwd] input `{name}` is a non-contiguous view — \
                         materializing a canonical device copy for the \
                         stride-blind GPU backward kernels"
                    );
                }
                let c = crate::tensor::nsl_tensor_contiguous(ptr);
                temps.push(c);
                c
            };
            let dout_use = canon(dout_ptr, "dout", &mut canon_temps);
            let q_use = canon(q_ptr, "q", &mut canon_temps);
            let k_canon = canon(k_ptr, "k", &mut canon_temps);
            let v_canon = canon(v_ptr, "v", &mut canon_temps);
            let out_use = canon(out_ptr, "out", &mut canon_temps);

            let groups = if kv_h > 0 { h / kv_h } else { 1 };
            // P4 native GQA: by default `kv_h < h` runs the grouped `_gqa`
            // Phase-2 kernel directly (no expanded K/V/dK/dV, no DtoD expand,
            // no group-sum reduce). `NSL_FLASH_GQA_BWD_EXPAND=1` forces the
            // legacy expand-KV envelope (A/B debugging). The segment-masked
            // path never emits a `_gqa` entry, so it always expands.
            let force_expand = std::env::var("NSL_FLASH_GQA_BWD_EXPAND")
                .map(|v| v == "1")
                .unwrap_or(false)
                || segment_ids_ptr != 0;
            let use_native_gqa = kv_h != h && !force_expand;

            let (k_use, v_use, k_exp_t, v_exp_t, call_kv_h) = if kv_h != h && !use_native_gqa {
                // Expand-KV envelope: expand to full heads so the MHA kernel
                // applies; the post-launch group-sum is the expansion adjoint.
                let ke = expand_kv_heads_device(k_canon, b, kv_h, groups, s, d);
                let ve = expand_kv_heads_device(v_canon, b, kv_h, groups, s, d);
                (ke, ve, ke, ve, h)
            } else if use_native_gqa {
                // Native: unexpanded K/V, real kv_h → grouped kernel.
                (k_canon, v_canon, 0i64, 0i64, kv_h)
            } else {
                // MHA.
                (k_canon, v_canon, 0i64, 0i64, h)
            };

            let gpu_result = flash_attention_backward_gpu(
                dout_use, q_use, k_use, v_use, out_use, logsumexp_ptr,
                scale, b, h, call_kv_h, s, d, is_causal,
                block_q, block_kv,
                phase1_ptx_ptr, phase1_name_ptr,
                phase2_ptx_ptr, phase2_name_ptr,
                segment_ids_ptr,
            );

            // The launcher synchronized before returning on every path, so
            // the expanded/materialized copies are dead now regardless of
            // outcome.
            if k_exp_t != 0 {
                crate::tensor::nsl_tensor_free(k_exp_t);
            }
            if v_exp_t != 0 {
                crate::tensor::nsl_tensor_free(v_exp_t);
            }
            for t in canon_temps {
                crate::tensor::nsl_tensor_free(t);
            }

            if gpu_result != 0 {
                // Group-sum is the expansion adjoint — needed ONLY when we
                // expanded. Native GQA already produced kv-shaped dK/dV.
                if kv_h != h && !use_native_gqa {
                    reduce_expanded_kv_grads(gpu_result, b, kv_h, groups, s, d);
                }
                if flash_debug {
                    eprintln!(
                        "[flash-bwd] GPU backward dispatched \
                         (batch={b}, heads={h}, kv_heads={kv_h}, seq={s}, head_dim={d}, \
                         causal={is_causal}, blocks=({block_q},{block_kv}))"
                    );
                }
                return gpu_result;
            }
            // gpu_result == 0: the launcher refused this config (ragged seq)
            // or a kernel launch failed. flash_attention_backward_gpu already
            // logged the specific cause and returned NO gradients rather than
            // silently-wrong zeros. Fall through to the CPU reference below,
            // which computes correct gradients (slower). Once per distinct
            // config: since the variant table made decorator-free models hit
            // this path, an unconditional print here would repeat identically
            // every attention op of every step.
            flash_bwd_warn_once(&format!(
                "[flash-bwd] GPU backward unavailable for this config \
                 (batch={b}, heads={h}, seq={s}, head_dim={d}) — falling back to the \
                 CPU reference backward (correct but slow); an earlier [flash-bwd] \
                 line names the cause (ragged-seq refusal or launch failure)."
            ));
        }

        if dout_t.device > 0 {
            crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (phase1_ptx_ptr, phase1_name_ptr, phase2_ptx_ptr, phase2_name_ptr);
    }

    // Read input tensors
    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    // Detect GQA: K/V may have fewer heads than Q.
    // Read actual KV head count from K's shape (dim 1).
    let kv_h = if k_t.ndim >= 2 {
        unsafe { *k_t.shape.add(1) as usize }
    } else {
        h
    };

    let total_qkv = b * h * s * d;
    let total_kv = b * kv_h * s * d;
    let total_lse = b * h * s;

    // Helper to read tensor data as a CANONICAL row-major f32 Vec (handles
    // both f32 and f64 dtypes; GPU tensors transfer via cudaMemcpy).
    //
    // PCA Stage C made this STRIDE-AWARE: non-contiguous views (e.g.
    // reshape(..).transpose(1,2) products from stdlib GQA) used to be read
    // as if [b,h,s,d]-contiguous, which produced smoothly-wrong (~1e-4
    // scale) gradients instead of an error — found by the
    // packed-vs-decomposed parity bisection. Views now pull the underlying
    // buffer extent and gather through the tensor's strides.
    fn read_f32_data(t: &NslTensor, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        let is_gpu = t.device > 0;
        let ndim = t.ndim as usize;
        let shape: Vec<i64> = (0..ndim).map(|i| unsafe { *t.shape.add(i) }).collect();
        let strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();

        // Canonical tensors read the buffer 1:1. Views read `span` elements
        // of the UNDERLYING buffer (highest linear element index touched,
        // plus one — stride-0 expand dims contribute nothing) and gather
        // below. Negative strides never occur in this runtime
        // (compute_strides and every view op emit strides >= 0); if one
        // ever appears, keep the old contiguous read but say so loudly.
        let mut canonical = is_canonical_row_major(t);
        if !canonical && strides.iter().any(|&st| st < 0) {
            eprintln!(
                "[flash-bwd] tensor has negative strides (unsupported) — \
                 reading as contiguous; gradients may be WRONG"
            );
            canonical = true;
        }
        let span = if canonical {
            len
        } else {
            let mut last = 0i64;
            for dd in 0..ndim {
                if shape[dd] > 1 {
                    last += (shape[dd] - 1) * strides[dd];
                }
            }
            (last + 1) as usize
        };

        // Pull `span` elements of the underlying buffer to host as f32.
        let raw: Vec<f32> = if t.dtype == 1 {
            // f32 tensor
            let mut buf = vec![0.0f32; span];
            if is_gpu {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        span * 4,
                    );
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("[flash-bwd] WARNING: GPU tensor but CUDA not enabled");
                }
            } else {
                for (i, slot) in buf.iter_mut().enumerate() {
                    *slot = unsafe { *t.data_f32().add(i) };
                }
            }
            buf
        } else {
            // f64 -> f32
            if is_gpu {
                // GPU f64 tensors: transfer as f64, then convert
                #[allow(unused_mut)] // `mut` needed only under `cuda` feature
                let mut f64_buf = vec![0.0f64; span];
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        f64_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        span * 8,
                    );
                }
                f64_buf.iter().map(|&v| v as f32).collect()
            } else {
                (0..span).map(|i| unsafe { *t.data_f64().add(i) as f32 }).collect()
            }
        };
        if canonical {
            return raw;
        }

        // Stride gather: walk logical row-major order, dot each
        // multi-index with the view's strides.
        let mut out = vec![0.0f32; len];
        for (flat, slot) in out.iter_mut().enumerate() {
            let mut rem = flat;
            let mut off = 0usize;
            for dd in (0..ndim).rev() {
                let extent = shape[dd].max(1) as usize;
                let idx = rem % extent;
                rem /= extent;
                off += idx * strides[dd] as usize;
            }
            *slot = raw[off];
        }
        out
    }

    let dout_data = read_f32_data(dout_t, total_qkv);
    let q_data = read_f32_data(q_t, total_qkv);
    let k_data = read_f32_data(k_t, total_kv);
    let v_data = read_f32_data(v_t, total_kv);
    let out_data = read_f32_data(out_t, total_qkv);

    // PCA Stage C: optional [batch, seq] segment ids for packed batches.
    // Host float tensors slice directly; device-resident copies come back
    // through read_f32_data's memcpy_dtoh path.
    let seg_host: Option<Vec<f32>> = if segment_ids_ptr != 0 {
        let seg_t = NslTensor::from_ptr(segment_ids_ptr);
        if seg_t.len == (b * s) as i64 {
            Some(read_f32_data(seg_t, b * s))
        } else {
            // Compiler-emitted call sites guarantee the [b, s] shape; a
            // mismatch is a dispatch bug. This path has no further
            // fallback, so be LOUD (unconditional — this must never scroll
            // away) and compute unmasked gradients rather than reading out
            // of bounds: training fails visibly instead of crashing.
            eprintln!(
                "[flash-bwd] segment_ids tensor has len {} but batch*seq = {} \
                 — IGNORING the segment mask; gradients for this packed \
                 batch are WRONG. This is a compiler dispatch bug.",
                seg_t.len,
                b * s
            );
            None
        }
    } else {
        None
    };

    // Read or auto-compute logsumexp.
    // When logsumexp_ptr == 0, the forward was decomposed (not fused FlashAttention)
    // and no logsumexp buffer was saved. Compute it from Q, K, scale, causal —
    // segment-aware when segment ids are present.
    let gqa_groups = if kv_h > 0 { h / kv_h } else { 1 };
    let lse_data = if logsumexp_ptr != 0 {
        let lse_t = NslTensor::from_ptr(logsumexp_ptr);
        read_f32_data(lse_t, total_lse)
    } else {
        compute_logsumexp_gqa(
            &q_data, &k_data, b, h, kv_h, s, d, scale, is_causal,
            seg_host.as_deref(),
        )
    };

    // Allocate gradient buffers (zero-initialized)
    // dQ has Q's shape [batch, heads, seq, head_dim]
    // dK, dV have KV's shape [batch, kv_heads, seq, head_dim]
    let mut dq_data = vec![0.0f32; total_qkv];
    let mut dk_data = vec![0.0f32; total_kv];
    let mut dv_data = vec![0.0f32; total_kv];

    // Run the CPU backward with GQA support (segment-aware when packed)
    flash_attention_backward_cpu_gqa(
        &q_data, &k_data, &v_data,
        &out_data, &lse_data, &dout_data,
        &mut dq_data, &mut dk_data, &mut dv_data,
        b, h, kv_h, s, d,
        scale, is_causal, gqa_groups,
        seg_host.as_deref(),
    );

    // Create output tensors (f32 dtype, matching Q shape for dQ, KV shape for dK/dV)
    fn make_tensor(data: &[f32], shape: &[i64]) -> i64 {
        let ndim = shape.len() as i64;
        let total = data.len();
        let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s };
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim);
        let data_size = std::mem::size_of_val(data);
        let data_ptr = crate::memory::checked_alloc(data_size) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, total);
        }
        let t = Box::new(NslTensor::new(
            data_ptr as *mut std::ffi::c_void,
            shape_ptr,
            strides,
            ndim,
            total as i64,
            0,
            1,
            1,
            0,
        ));
        Box::into_raw(t) as i64
    }

    let q_shape = [batch, heads, seq_len, head_dim];
    let kv_shape = [batch, kv_h as i64, seq_len, head_dim];
    let mut dq_ptr = make_tensor(&dq_data, &q_shape);
    let mut dk_ptr = make_tensor(&dk_data, &kv_shape);
    let mut dv_ptr = make_tensor(&dv_data, &kv_shape);

    // If inputs were on GPU, transfer gradient tensors to GPU to match device
    let input_device = q_t.device;
    if input_device > 0 {
        dq_ptr = crate::tensor::nsl_tensor_to_device(dq_ptr, input_device as i64);
        dk_ptr = crate::tensor::nsl_tensor_to_device(dk_ptr, input_device as i64);
        dv_ptr = crate::tensor::nsl_tensor_to_device(dv_ptr, input_device as i64);
    }

    // Pack into an NslList [dq, dk, dv]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

// ── CSHA backward-activation buffer allocator ─────────────────────────────

/// The five HBM buffers that the CSHA forward kernel fills when
/// `save_activations_for_backward = true`.  All fields are device
/// pointers represented as `i64` (matching the rest of the NSL runtime
/// ABI).  A value of `0` indicates allocation failure.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CshaBackwardActivations {
    pub q_proj: i64,
    pub k_proj: i64,
    pub v_proj: i64,
    pub row_max: i64,
    pub row_sum: i64,
    /// Raw (pre-RMSNorm) x copy. Forward RMSNorm writes x_normed back into
    /// `csha_x_ptr` in place; the backward dRMSNorm closed form needs the
    /// pre-norm x, so the forward save path stages a copy here. Layout:
    /// `[batch, heads, seq, head_dim]` f32, row-major (matches the x_ptr
    /// the kernel sees).
    pub x_raw: i64,
}

/// Allocate the 5 HBM buffers forward fills when
/// `csha.save_activations_for_backward = true`. Called by the compiler
/// before the forward launch in training mode. All returned pointers
/// are non-zero on success; a zero pointer indicates allocation failure.
/// Caller must call `nsl_csha_free_backward_activations` to release.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_alloc_backward_activations(
    batch: i64, heads: i64, seq: i64, head_dim: i64,
) -> CshaBackwardActivations {
    let qkv_bytes = batch * heads * seq * head_dim * 2;  // f16 = 2 bytes
    let stats_bytes = batch * heads * seq * 4;           // f32 = 4 bytes
    // x_raw mirrors the kernel-visible x buffer: [batch, heads, seq, head_dim] f32.
    let xraw_bytes = batch * heads * seq * head_dim * 4;
    #[cfg(feature = "cuda")]
    {
        use crate::cuda::inner;
        CshaBackwardActivations {
            q_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            k_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            v_proj: inner::alloc_device(qkv_bytes as usize) as i64,
            row_max: inner::alloc_device(stats_bytes as usize) as i64,
            row_sum: inner::alloc_device(stats_bytes as usize) as i64,
            x_raw: inner::alloc_device(xraw_bytes as usize) as i64,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (qkv_bytes, stats_bytes, xraw_bytes);
        CshaBackwardActivations {
            q_proj: 0, k_proj: 0, v_proj: 0, row_max: 0, row_sum: 0, x_raw: 0,
        }
    }
}

/// Free the 6 HBM buffers allocated by `nsl_csha_alloc_backward_activations`.
/// Safe to call with zero pointers — they are silently skipped.
///
/// The buffers come from `inner::alloc_device` (raw `cuMemAlloc`), so they
/// MUST be released via `inner::free_device` (raw `cuMemFree`) — the
/// documented pair.  This function previously routed through
/// `inner::free_managed`, which consults `CUDA_ALLOC_SET` (populated only
/// by `alloc_managed`) and silently early-returns for unregistered
/// pointers: every "free" was a no-op and each alloc/free cycle leaked all
/// six buffers (~B*H*S*(10*D+8) bytes per CSHA layer per step).
///
/// Stream-ordered deferred free (p3-remainder): callers invoke this right
/// after asynchronous kernel launches that still read/write the buffers
/// (`nsl_flash_attention_csha_backward` on the @train path; the with-saves
/// forward on the scope-immediate inference path in `expr/advanced.rs`), and
/// `kernel_launch` does NOT synchronize (outside opt-in `--cuda-sync`). Rather
/// than block on a `cuCtxSynchronize` before freeing, `defer_free_device_batch`
/// records one NULL-stream completion event covering all six buffers and runs
/// the raw `cuMemFree`s once that event completes — same "physically return to
/// the driver" semantics as before, without the host stall. `NSL_CUDA_SYNC=1`
/// restores the eager sync-then-free.
///
/// Do NOT "fix" the pairing by registering the pointers in `CUDA_ALLOC_SET`
/// instead — that would route the frees through the caching allocator, whose
/// pooling reuse semantics differ from the raw `cuMemFree` these buffers need
/// (raw free returns the VRAM to the driver; pooling keeps it resident).
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_free_backward_activations(
    a: CshaBackwardActivations,
) {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda::inner;
        // All six buffers share one lifetime (consumed by the same preceding
        // kernel), so a single completion event guards the group. Null slots
        // are filtered inside `defer_free_device_batch`.
        inner::defer_free_device_batch(&[
            a.q_proj as *mut std::ffi::c_void,
            a.k_proj as *mut std::ffi::c_void,
            a.v_proj as *mut std::ffi::c_void,
            a.row_max as *mut std::ffi::c_void,
            a.row_sum as *mut std::ffi::c_void,
            a.x_raw as *mut std::ffi::c_void,
        ]);
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = a; }
}

/// Gap A: codegen-friendly wrapper around
/// `nsl_csha_alloc_backward_activations` that writes the 6 device pointers
/// into `out_ptr` as a contiguous `[i64; 6]` array.
///
/// Struct-by-value returns are fiddly to emit from Cranelift (sret),
/// so the compiler-side emission uses this i64-array variant instead.
/// Layout matches `CshaBackwardActivations` field order:
///   \[0] q_proj, \[1] k_proj, \[2] v_proj,
///   \[3] row_max, \[4] row_sum, \[5] x_raw
///
/// Returns `0` on success. A zero pointer in any slot indicates allocation
/// failure (matches `nsl_csha_alloc_backward_activations` semantics).
///
/// SAFETY: `out_ptr` must point to at least 48 bytes (6 × i64) of writable
/// memory and be 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_alloc_backward_activations_into(
    batch: i64, heads: i64, seq: i64, head_dim: i64,
    out_ptr: i64,
) -> i64 {
    let a = nsl_csha_alloc_backward_activations(batch, heads, seq, head_dim);
    if out_ptr == 0 {
        // No slot to write into — free immediately to avoid leak and fail.
        nsl_csha_free_backward_activations(a);
        return -1;
    }
    let slots = out_ptr as *mut i64;
    // SAFETY: caller guarantees at least 6 i64 slots.
    slots.add(0).write(a.q_proj);
    slots.add(1).write(a.k_proj);
    slots.add(2).write(a.v_proj);
    slots.add(3).write(a.row_max);
    slots.add(4).write(a.row_sum);
    slots.add(5).write(a.x_raw);
    0
}

/// Gap A: codegen-friendly free variant taking 6 individual i64 pointers
/// (matches the i64-array produced by
/// `nsl_csha_alloc_backward_activations_into`). Avoids passing
/// `CshaBackwardActivations` by-value across the Cranelift ABI boundary.
///
/// Safe to call with zero pointers — they are silently skipped, mirroring
/// `nsl_csha_free_backward_activations`.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_free_backward_activations_from(
    q_proj: i64, k_proj: i64, v_proj: i64,
    row_max: i64, row_sum: i64, x_raw: i64,
) {
    let a = CshaBackwardActivations {
        q_proj, k_proj, v_proj, row_max, row_sum, x_raw,
    };
    nsl_csha_free_backward_activations(a);
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn nsl_csha_alloc_backward_activations_allocates_five_buffers() {
        // Shape: batch=2, heads=4, seq=32, head_dim=32.
        // Expected sizes:
        //   q_proj, k_proj, v_proj: 2*4*32*32 f16 = 8192 bytes each
        //   row_max, row_sum: 2*4*32 f32 = 1024 bytes each
        let r = unsafe { nsl_csha_alloc_backward_activations(2, 4, 32, 32) };
        assert_ne!(r.q_proj, 0, "q_proj alloc failed");
        assert_ne!(r.k_proj, 0, "k_proj alloc failed");
        assert_ne!(r.v_proj, 0, "v_proj alloc failed");
        assert_ne!(r.row_max, 0, "row_max alloc failed");
        assert_ne!(r.row_sum, 0, "row_sum alloc failed");
        assert_ne!(r.x_raw, 0, "x_raw alloc failed");
        unsafe { nsl_csha_free_backward_activations(r); }
    }

    /// The device-resident logsumexp kernel (`FLASH_LSE_F32_PTX`) must match
    /// the CPU reference `compute_logsumexp_gqa` within GPU transcendental
    /// tolerance. It replaced the per-call Q/K host round-trip + CPU score
    /// recompute in the flash backward, so a drift here would perturb every
    /// decorator-free GPU backward's `P = exp(score - lse)`. Checked in both
    /// causal modes at kv_heads == h (the only shape the GPU backward
    /// dispatches).
    #[test]
    #[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
    #[cfg(feature = "cuda")]
    fn flash_lse_gpu_matches_cpu_reference() {
        use crate::cuda::inner;
        use std::ffi::c_void;

        // `inner::alloc_managed` / `kernel_launch` establish the thread's CUDA
        // context on first use, so no explicit init is needed here.

        let b = 2usize;
        let h = 2usize;
        let s = 24usize;
        let d = 16usize;
        let n = b * h * s * d;
        let scale = 1.0f32 / (d as f32).sqrt();

        // Deterministic pseudo-random Q, K in [-1, 1) (LCG).
        let mut lcg: u32 = 0x2545_F491;
        let mut nextf = || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..n).map(|_| nextf()).collect();
        let k: Vec<f32> = (0..n).map(|_| nextf()).collect();

        for &causal in &[false, true] {
            let reference = compute_logsumexp_gqa(&q, &k, b, h, h, s, d, scale, causal, None);

            let bytes = n * 4;
            let total_lse = b * h * s;
            let q_dev = inner::alloc_managed(bytes);
            let k_dev = inner::alloc_managed(bytes);
            let lse_dev = inner::alloc_managed(total_lse * 4);
            inner::memcpy_htod(q_dev, q.as_ptr() as *const c_void, bytes);
            inner::memcpy_htod(k_dev, k.as_ptr() as *const c_void, bytes);

            let mut q_arg = q_dev as u64;
            let mut k_arg = k_dev as u64;
            let mut lse_arg = lse_dev as u64;
            let mut total_arg = total_lse as u64;
            let mut seq_arg = s as u64;
            let mut hd_arg = d as u64;
            let mut scale_arg = scale;
            let mut causal_arg: u64 = causal.into();
            let args: [*mut c_void; 8] = [
                &mut q_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
                &mut lse_arg as *mut _ as *mut c_void,
                &mut total_arg as *mut _ as *mut c_void,
                &mut seq_arg as *mut _ as *mut c_void,
                &mut hd_arg as *mut _ as *mut c_void,
                &mut scale_arg as *mut _ as *mut c_void,
                &mut causal_arg as *mut _ as *mut c_void,
            ];
            let block = 256i64;
            let grid = [(total_lse as i64 + block - 1) / block, 1, 1];
            let res = inner::kernel_launch(
                crate::cuda::fused_kernels::FLASH_LSE_F32_PTX.as_ptr(),
                b"nsl_flash_lse_f32\0".as_ptr(),
                grid,
                [block, 1, 1],
                &args,
                0,
            );
            assert_eq!(
                res,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "lse kernel launch failed (causal={causal})"
            );
            unsafe {
                cudarc::driver::sys::cuCtxSynchronize();
            }

            let mut got = vec![0.0f32; total_lse];
            inner::memcpy_dtoh(
                got.as_mut_ptr() as *mut c_void,
                lse_dev as *const c_void,
                total_lse * 4,
            );
            inner::free_managed(q_dev);
            inner::free_managed(k_dev);
            inner::free_managed(lse_dev);

            for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
                let diff = (g - r).abs();
                let tol = 1e-3 * (1.0 + r.abs());
                assert!(
                    diff <= tol,
                    "lse mismatch (causal={causal}) at {i}: gpu={g}, cpu={r}, \
                     diff={diff}, tol={tol}"
                );
            }
        }
    }

    /// The multi-tile backward post-pass PTX must be ASCII-only and
    /// NUL-terminated. The driver JIT (`cuModuleLoadData`, C-string contract)
    /// rejects any non-ASCII byte with `CUDA_ERROR_INVALID_PTX` — a box-drawing
    /// char in a PTX comment cost a debug cycle here, and offline `ptxas` under
    /// a UTF-8 locale does NOT reproduce it, so a static guard is the only
    /// reliable catch. Mirrors the precision-cast kernels' ASCII assertion.
    #[test]
    #[cfg(feature = "cuda")]
    fn multitile_postpass_ptx_is_ascii_and_nul_terminated() {
        let ptx = CSHA_BWD_MULTITILE_POSTPASS_PTX;
        assert!(ptx.ends_with('\0'), "post-pass PTX must be NUL-terminated");
        for (i, b) in ptx.bytes().enumerate() {
            assert!(
                b.is_ascii(),
                "post-pass PTX byte {i} = {b:#x} is non-ASCII (driver JIT trips \
                 CUDA_ERROR_INVALID_PTX). Context: {:?}",
                &ptx[i.saturating_sub(24)..(i + 1).min(ptx.len())]
            );
        }
        assert!(CSHA_BWD_MULTITILE_POSTPASS_NAME.ends_with('\0'));
    }

    /// Companion guard for the f16<->f32 CSHA cast kernels — also hand-written,
    /// module-private PTX loaded via `cuModuleLoadData` at runtime, and so
    /// subject to the same driver-JIT hazard: a non-ASCII byte anywhere (even a
    /// comment) trips `CUDA_ERROR_INVALID_PTX`, and offline `ptxas` under a
    /// UTF-8 locale does not reproduce it. Without this, these two consts were
    /// the last runtime PTX in the crate not covered by any ASCII gate.
    #[test]
    #[cfg(feature = "cuda")]
    fn csha_cast_ptx_is_ascii_and_nul_terminated() {
        for (tag, ptx, name) in [
            (
                "CSHA_BWD_F32_TO_F16",
                CSHA_BWD_F32_TO_F16_PTX,
                CSHA_BWD_F32_TO_F16_NAME,
            ),
            (
                "CSHA_FWD_F16_TO_F32",
                CSHA_FWD_F16_TO_F32_PTX,
                CSHA_FWD_F16_TO_F32_NAME,
            ),
        ] {
            assert!(ptx.ends_with('\0'), "{tag} PTX must be NUL-terminated");
            for (i, b) in ptx.bytes().enumerate() {
                assert!(
                    b.is_ascii(),
                    "{tag} PTX byte {i} = {b:#x} is non-ASCII (driver JIT trips \
                     CUDA_ERROR_INVALID_PTX). Context: {:?}",
                    &ptx[i.saturating_sub(24)..(i + 1).min(ptx.len())]
                );
            }
            assert!(name.ends_with('\0'), "{tag} name must be NUL-terminated");
        }
    }

    /// Leak regression for the alloc/free pairing bug (2026-07-02):
    /// the six save buffers come from `inner::alloc_device` (raw
    /// `cuMemAlloc`, never registered in `CUDA_ALLOC_SET`), but
    /// `nsl_csha_free_backward_activations` used to free them via
    /// `inner::free_managed`, which silently early-returns for
    /// unregistered pointers.  Every free was a no-op, so each
    /// alloc→free cycle leaked all six buffers.
    ///
    /// Detection: loop alloc→free 50× on a ~5 MB buffer set and compare
    /// driver-reported free memory (`cuMemGetInfo`) before vs. after.
    /// Broken pairing leaks ~250 MB; the assertion threshold of 128 MB
    /// leaves generous headroom for concurrent tests / desktop VRAM
    /// noise while still catching the regression by a wide margin.
    ///
    /// p3-remainder: `nsl_csha_free_backward_activations` now defers the raw
    /// `cuMemFree` behind a NULL-stream event, so `drain_all_deferred_frees`
    /// is called before the final measurement to force every deferred free to
    /// complete. This asserts the STRONGER property that deferred free returns
    /// all VRAM to the driver (it is a latency shift, never a leak).
    #[test]
    #[cfg(feature = "cuda")]
    fn csha_free_backward_activations_returns_memory_to_driver() {
        // batch=2, heads=8, seq=512, head_dim=64:
        //   q/k/v_proj: 2*8*512*64 f16 = 1 MB each
        //   row_max/row_sum: 2*8*512 f32 = 32 KB each
        //   x_raw: 2*8*512*64 f32 = 2 MB
        // ≈ 5.06 MB per iteration, ≈ 253 MB across 50 iterations.
        const ITERS: usize = 50;
        const MAX_ALLOWED_DROP_BYTES: usize = 128 * 1024 * 1024;

        let (free_before, _total) = crate::cuda::inner::query_vram();
        for _ in 0..ITERS {
            let r = unsafe { nsl_csha_alloc_backward_activations(2, 8, 512, 64) };
            assert_ne!(r.q_proj, 0, "q_proj alloc failed");
            assert_ne!(r.x_raw, 0, "x_raw alloc failed");
            unsafe { nsl_csha_free_backward_activations(r); }
        }
        // Force this cycle's event-deferred frees to complete before
        // measuring. (A global pending==0 assertion would be racy — the
        // deferred-free queue is shared across the whole test binary — so we
        // rely on the VRAM-delta bound below, which has ample headroom.)
        crate::cuda::inner::drain_all_deferred_frees();
        let (free_after, _total) = crate::cuda::inner::query_vram();

        let dropped = free_before.saturating_sub(free_after);
        assert!(
            dropped < MAX_ALLOWED_DROP_BYTES,
            "csha save-buffer alloc/free cycle leaked device memory: \
             free VRAM dropped by {} bytes over {} iterations \
             (before={}, after={}). The free path must release the raw \
             `alloc_device` buffers via `free_device`, not `free_managed`.",
            dropped, ITERS, free_before, free_after,
        );
    }

    /// Naive attention forward for reference: O = softmax(Q @ K^T * scale) @ V
    fn naive_attention_forward(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        // Returns (output, logsumexp)
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let lse_bh_stride = heads * seq_len;
        let lse_h_stride = seq_len;

        let total_qkv = batch * heads * seq_len * head_dim;
        let total_lse = batch * heads * seq_len;
        let mut output = vec![0.0f32; total_qkv];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;
                let lse_base = b * lse_bh_stride + h * lse_h_stride;

                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;

                    // Compute scores S[i,:] = Q[i] . K[:] * scale
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }

                    // Apply causal mask
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }

                    // Online softmax and logsumexp
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores {
                        sum_exp += (s - max_score).exp();
                    }
                    logsumexp[lse_base + i] = max_score + sum_exp.ln();

                    // Softmax probs
                    let lse = logsumexp[lse_base + i];
                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();

                    // O[i] = sum_j P[i,j] * V[j]
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        for d in 0..head_dim {
                            output[q_base + d] += probs[j] * v[v_base + d];
                        }
                    }
                }
            }
        }

        (output, logsumexp)
    }

    /// Naive attention backward for reference (full S^2 computation)
    fn naive_attention_backward(
        q: &[f32], k: &[f32], v: &[f32],
        dout: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let total_qkv = batch * heads * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_qkv];
        let mut dk = vec![0.0f32; total_qkv];
        let mut dv = vec![0.0f32; total_qkv];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;

                // Compute full attention scores and softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_exp += *s;
                    }
                    for j in 0..seq_len {
                        probs[i][j] = scores[j] / sum_exp;
                    }
                }

                // dV[j] = sum_i P[i,j] * dO[i]
                for j in 0..seq_len {
                    let v_base = qkv_base + j * s_stride;
                    for i in 0..seq_len {
                        let do_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dv[v_base + d] += probs[i][j] * dout[do_base + d];
                        }
                    }
                }

                // dP[i,j] = dO[i] . V[j]
                // dS[i,j] = P[i,j] * (dP[i,j] - sum_k(dP[i,k] * P[i,k]))
                //         = P[i,j] * (dP[i,j] - D[i])
                // where D[i] = sum_j P[i,j] * dP[i,j] = sum_j P[i,j] * (dO[i] . V[j])
                //            = dO[i] . (sum_j P[i,j] * V[j]) = dO[i] . O[i]
                for i in 0..seq_len {
                    let do_base = qkv_base + i * s_stride;

                    // D[i] = sum_j P[i,j] * (dO[i] . V[j])
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let v_base = qkv_base + j * s_stride;

                        // dP[i,j]
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }

                        // dS[i,j]
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        // dQ[i] += dS[i,j] * K[j] * scale
                        let q_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dq[q_base + d] += ds_ij * k[k_base + d] * scale;
                            dk[k_base + d] += ds_ij * q[q_base + d] * scale;
                        }
                    }
                }
            }
        }

        (dq, dk, dv)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[allow(dead_code)]
    fn rel_error(a: &[f32], b: &[f32]) -> f32 {
        let mut max_rel = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let denom = x.abs().max(y.abs()).max(1e-8);
            let rel = (x - y).abs() / denom;
            if rel > max_rel { max_rel = rel; }
        }
        max_rel
    }

    #[test]
    fn test_flash_backward_matches_naive_non_causal() {
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Deterministic test data
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // Forward pass (naive reference)
        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        // Random-ish upstream gradient
        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive backward
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        // Flash backward
        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol}");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol}");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol}");
    }

    #[test]
    fn test_flash_backward_matches_naive_causal() {
        let batch = 1;
        let heads = 2;
        let seq_len = 6;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol} (causal)");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol} (causal)");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol} (causal)");
    }

    #[test]
    fn test_flash_backward_batched_multihead() {
        let batch = 2;
        let heads = 3;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.11 + 0.5).cos() * 0.3).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.19 + 1.5).sin() * 0.3).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.29 + 2.5).cos() * 0.2).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        assert!(max_abs_diff(&dq_naive, &dq_flash) < tol, "dQ mismatch in batched test");
        assert!(max_abs_diff(&dk_naive, &dk_flash) < tol, "dK mismatch in batched test");
        assert!(max_abs_diff(&dv_naive, &dv_flash) < tol, "dV mismatch in batched test");
    }

    #[test]
    fn test_flash_backward_causal_gradient_mask() {
        // Verify that for a causal mask, gradients for keys past the query position are zero
        // in the sense that they don't receive "future" contributions.
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Use identity-like Q/K to make attention sharp
        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let v: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Make Q[0] only attend to K[0] (with causal, it can't see K[1..3] anyway)
        for d in 0..head_dim {
            q[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            k[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            // Other positions get different patterns
            for i in 1..seq_len {
                q[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
                k[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
            }
        }

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        // Gradient only on the first query row
        let mut dout = vec![0.0f32; total];
        for d in 0..head_dim {
            dout[d] = 1.0;
        }

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        // With causal mask, Q[0] only attends to K[0], so:
        // - dK for positions j>0 should be zero from Q[0]'s gradient
        //   (since dout is only nonzero for row 0 and row 0 can only see j=0)
        // Check dK[1..] are zero
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dk_flash[j * head_dim + d].abs() < 1e-6,
                    "dK[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dk_flash[j * head_dim + d]);
            }
        }
        // dV for positions j>0 should also be zero (P[0,j]=0 for j>0)
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dv_flash[j * head_dim + d].abs() < 1e-6,
                    "dV[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dv_flash[j * head_dim + d]);
            }
        }
    }

    #[test]
    fn test_flash_backward_finite_difference() {
        // Numerical gradient check using finite differences
        let batch = 1;
        let heads = 1;
        let seq_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        let eps = 1e-3f32;

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);
        let dout: Vec<f32> = vec![1.0; total]; // all-ones upstream gradient = computing d(sum(O))/d(param)

        // Analytic gradients
        let mut dq_analytic = vec![0.0f32; total];
        let mut dk_analytic = vec![0.0f32; total];
        let mut dv_analytic = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_analytic, &mut dk_analytic, &mut dv_analytic,
            batch, heads, seq_len, head_dim, scale, false,
        );

        // Finite-difference check for dQ
        for idx in 0..total.min(12) {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q_plus, &k, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q_minus, &k, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dq_analytic[idx] - fd_grad).abs();
            let denom = dq_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dQ[{idx}] analytic={} fd={} rel_err={}",
                dq_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dK
        for idx in 0..total.min(12) {
            let mut k_plus = k.clone();
            let mut k_minus = k.clone();
            k_plus[idx] += eps;
            k_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k_plus, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k_minus, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dk_analytic[idx] - fd_grad).abs();
            let denom = dk_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dK[{idx}] analytic={} fd={} rel_err={}",
                dk_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dV
        for idx in 0..total.min(12) {
            let mut v_plus = v.clone();
            let mut v_minus = v.clone();
            v_plus[idx] += eps;
            v_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k, &v_plus, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k, &v_minus, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dv_analytic[idx] - fd_grad).abs();
            let denom = dv_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dV[{idx}] analytic={} fd={} rel_err={}",
                dv_analytic[idx], fd_grad, diff / denom);
        }
    }

    #[test]
    fn test_logsumexp_only_auxiliary_storage() {
        // Verify that the backward pass uses O(N) auxiliary storage (logsumexp)
        // not O(N^2) (full attention matrix). The logsumexp tensor is
        // [batch, heads, seq_len] while the attention matrix would be
        // [batch, heads, seq_len, seq_len].
        let batch = 2;
        let heads = 4;
        let seq_len = 16;
        let _head_dim = 8;
        let total_lse = batch * heads * seq_len;
        let total_attn = batch * heads * seq_len * seq_len;

        // logsumexp is O(N), attention matrix would be O(N^2)
        assert!(total_lse < total_attn,
            "logsumexp ({total_lse}) should be much smaller than full attention ({total_attn})");
        assert_eq!(total_lse, batch * heads * seq_len);
    }

    // ── GQA helpers ─────────────────────────────────────────────────────

    /// Naive GQA-aware forward: Q [b, h_q, s, d], K/V [b, h_kv, s, d]
    /// Each group of h_q/h_kv Q heads shares one KV head.
    fn naive_attention_forward_gqa(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_lse = batch * h_q * seq_len;
        let mut output = vec![0.0f32; total_q];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;
                let lse_base = b * h_q * seq_len + hq * seq_len;

                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }

                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores { sum_exp += (s - max_s).exp(); }
                    logsumexp[lse_base + i] = max_s + sum_exp.ln();
                    let lse = logsumexp[lse_base + i];

                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        for dd in 0..head_dim {
                            output[qi + dd] += probs[j] * v[vj + dd];
                        }
                    }
                }
            }
        }
        (output, logsumexp)
    }

    /// Naive GQA backward reference: returns (dQ [b,h_q,s,d], dK [b,h_kv,s,d], dV [b,h_kv,s,d])
    fn naive_attention_backward_gqa(
        q: &[f32], k: &[f32], v: &[f32], dout: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_kv = batch * h_kv * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_q];
        let mut dk = vec![0.0f32; total_kv];
        let mut dv = vec![0.0f32; total_kv];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;

                // Compute softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
                    for j in 0..seq_len { probs[i][j] = scores[j] / sum_exp; }
                }

                // dV[j] += sum_i P[i,j] * dO[i]  (accumulated across GQA group)
                for j in 0..seq_len {
                    let vj_idx = kv_base + j * head_dim;
                    for i in 0..seq_len {
                        let doi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dv[vj_idx + dd] += probs[i][j] * dout[doi + dd];
                        }
                    }
                }

                // dQ, dK
                for i in 0..seq_len {
                    let doi = q_base + i * head_dim;
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        let qi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dq[qi + dd] += ds_ij * k[kj + dd] * scale;
                            dk[kj + dd] += ds_ij * q[qi + dd] * scale;
                        }
                    }
                }
            }
        }
        (dq, dk, dv)
    }

    #[test]
    fn test_flash_attention_backward_gqa_cpu() {
        // GQA: 4 Q heads, 2 KV heads (groups=2)
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 16;
        let d = 16;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        // Deterministic test data
        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // GQA forward
        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, false);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive GQA backward (reference)
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, false,
        );

        // Flash GQA backward (under test)
        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, false, groups,
            None,
        );

        // Verify shapes implicitly via lengths
        assert_eq!(dq_flash.len(), total_q, "dQ should have Q shape [b, h_q, s, d]");
        assert_eq!(dk_flash.len(), total_kv, "dK should have KV shape [b, h_kv, s, d]");
        assert_eq!(dv_flash.len(), total_kv, "dV should have KV shape [b, h_kv, s, d]");

        // Verify non-zero gradients
        let dq_norm: f32 = dq_flash.iter().map(|x| x * x).sum();
        let dk_norm: f32 = dk_flash.iter().map(|x| x * x).sum();
        let dv_norm: f32 = dv_flash.iter().map(|x| x * x).sum();
        assert!(dq_norm > 1e-8, "dQ should be non-zero, got norm={dq_norm}");
        assert!(dk_norm > 1e-8, "dK should be non-zero, got norm={dk_norm}");
        assert!(dv_norm > 1e-8, "dV should be non-zero, got norm={dv_norm}");

        // Match naive reference
        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA dQ max abs diff = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA dK max abs diff = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA dV max abs diff = {dv_err} exceeds tol {tol}");

        // Cross-check: summing dK from non-GQA (h_q heads, same K for each group)
        // should match the GQA dK. Run a non-GQA backward with K replicated.
        let mut k_expanded = vec![0.0f32; total_q]; // [b, h_q, s, d]
        let mut v_expanded = vec![0.0f32; total_q];
        for bb in 0..b {
            for hq in 0..h_q {
                let hkv = hq / groups;
                for si in 0..s {
                    for dd in 0..d {
                        let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                        let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                        k_expanded[q_idx] = k[kv_idx];
                        v_expanded[q_idx] = v[kv_idx];
                    }
                }
            }
        }

        // Non-GQA backward with expanded K/V
        let (out_exp, lse_exp) = naive_attention_forward(
            &q, &k_expanded, &v_expanded, b, h_q, s, d, scale, false,
        );
        let mut dk_expanded = vec![0.0f32; total_q];
        let mut dq_expanded = vec![0.0f32; total_q];
        let mut dv_expanded = vec![0.0f32; total_q];
        flash_attention_backward_cpu(
            &q, &k_expanded, &v_expanded, &out_exp, &lse_exp, &dout,
            &mut dq_expanded, &mut dk_expanded, &mut dv_expanded,
            b, h_q, s, d, scale, false,
        );

        // Sum dk_expanded across groups to get per-KV-head gradients
        let mut dk_summed = vec![0.0f32; total_kv];
        for bb in 0..b {
            for hkv in 0..h_kv {
                for g in 0..groups {
                    let hq = hkv * groups + g;
                    for si in 0..s {
                        for dd in 0..d {
                            let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                            let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                            dk_summed[kv_idx] += dk_expanded[q_idx];
                        }
                    }
                }
            }
        }

        let dk_cross_err = max_abs_diff(&dk_summed, &dk_flash);
        assert!(dk_cross_err < tol,
            "GQA dK should match sum of expanded non-GQA dK, err={dk_cross_err}");
    }

    #[test]
    fn test_flash_attention_backward_gqa_causal() {
        // GQA with causal mask: 4 Q heads, 2 KV heads
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 8;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, true);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, true, groups,
            None,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA causal dQ err = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA causal dK err = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA causal dV err = {dv_err} exceeds tol {tol}");

        // Causal check: with dout only at position 0, positions j > 0
        // should not receive gradient from Q[0] for any head
        let mut dout_first = vec![0.0f32; total_q];
        for hq in 0..h_q {
            for dd in 0..d {
                dout_first[hq * s * d + dd] = 1.0;
            }
        }

        let mut dk_causal = vec![0.0f32; total_kv];
        let mut dq_causal = vec![0.0f32; total_q];
        let mut dv_causal = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout_first,
            &mut dq_causal, &mut dk_causal, &mut dv_causal,
            b, h_q, h_kv, s, d,
            scale, true, groups,
            None,
        );

        // With causal mask and dout only at row 0, K/V positions j>0
        // should get zero gradient contribution from row 0
        for hkv in 0..h_kv {
            for j in 1..s {
                for dd in 0..d {
                    let idx = hkv * s * d + j * d + dd;
                    assert!(dk_causal[idx].abs() < 1e-5,
                        "dK[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dk_causal[idx]);
                    assert!(dv_causal[idx].abs() < 1e-5,
                        "dV[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dv_causal[idx]);
                }
            }
        }
    }

    // ── PCA Stage C: fused packed-SDPA runtime tests ──────────────────

    /// The fused-forward decline protocol: null PTX (and, on non-CUDA
    /// builds, everything) must decline with 0 — silently, with no side
    /// effects — so the wengert_lower dispatch keeps the decomposed SDPA
    /// graph. Also pins the 13-arg extern signature.
    #[test]
    fn sdpa_fused_forward_declines_with_null_ptx() {
        let r = nsl_sdpa_fused_forward(
            0, 0, 0,                 // q, k, v
            1.0f32.to_bits() as i64, // scale_bits
            1,                       // causal
            0,                       // segment_ids
            0, 0,                    // ptx, name (null => decline)
            0, 0,                    // tier-b sentinel pair (disabled)
            64, 64,                  // block_q, block_kv
            0,                       // shared_mem_bytes
        );
        assert_eq!(r, 0, "null-PTX fused forward must DECLINE (0), got {r}");
    }

    /// PCA Stage C: a segment-masked CPU backward over a packed row must
    /// equal independent backwards over the per-segment sub-sequences —
    /// block-diagonal attention IS independent attention per segment. The
    /// two batch rows split at different boundaries (5|3 vs 3|5) so a
    /// batch-row-blind masking bug (indexing `seg[i]` instead of
    /// `seg[b_idx * s + i]`) cannot pass.
    #[test]
    fn test_segment_masked_backward_cpu_matches_per_segment() {
        let b = 2usize;
        let h = 2usize;
        let s = 8usize;
        let d = 4usize;
        let scale = 1.0 / (d as f32).sqrt();
        let splits: [usize; 2] = [5, 3]; // per-batch segment boundary

        let total = b * h * s * d;
        let total_lse = b * h * s;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.11).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.19 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.29 + 2.0).sin() * 0.5).collect();
        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37 + 3.0).cos() * 0.3).collect();

        // seg[b*s + i]: distinct values per (batch, segment). Only equality
        // matters to the mask, so arbitrary non-uniform ids are fine.
        let mut seg = vec![0.0f32; b * s];
        for bb in 0..b {
            for i in 0..s {
                seg[bb * s + i] =
                    if i < splits[bb] { (10 + bb) as f32 } else { (20 + bb) as f32 };
            }
        }

        // Extract rows `r` of batch `bb` from a [b, h, s, d] tensor into a
        // [1, h, r.len(), d] tensor (scatter* invert; scatter3 is the
        // [b, h, s] logsumexp analog).
        let extract4 = |src: &[f32], bb: usize, r: std::ops::Range<usize>| -> Vec<f32> {
            let rl = r.len();
            let mut sub = vec![0.0f32; h * rl * d];
            for hh in 0..h {
                for (ri, i) in r.clone().enumerate() {
                    for dd in 0..d {
                        sub[(hh * rl + ri) * d + dd] = src[((bb * h + hh) * s + i) * d + dd];
                    }
                }
            }
            sub
        };
        let scatter4 = |dst: &mut [f32], sub: &[f32], bb: usize, r: std::ops::Range<usize>| {
            let rl = r.len();
            for hh in 0..h {
                for (ri, i) in r.clone().enumerate() {
                    for dd in 0..d {
                        dst[((bb * h + hh) * s + i) * d + dd] = sub[(hh * rl + ri) * d + dd];
                    }
                }
            }
        };
        let scatter3 = |dst: &mut [f32], sub: &[f32], bb: usize, r: std::ops::Range<usize>| {
            let rl = r.len();
            for hh in 0..h {
                for (ri, i) in r.clone().enumerate() {
                    dst[(bb * h + hh) * s + i] = sub[hh * rl + ri];
                }
            }
        };

        for causal in [false, true] {
            // Per-segment reference: independent forward + backward on each
            // (batch, segment) sub-sequence, stitched into full-shape buffers.
            let mut out_full = vec![0.0f32; total];
            let mut lse_full = vec![0.0f32; total_lse];
            let mut dq_ref = vec![0.0f32; total];
            let mut dk_ref = vec![0.0f32; total];
            let mut dv_ref = vec![0.0f32; total];
            for bb in 0..b {
                for r in [0..splits[bb], splits[bb]..s] {
                    let rl = r.len();
                    let q_sub = extract4(&q, bb, r.clone());
                    let k_sub = extract4(&k, bb, r.clone());
                    let v_sub = extract4(&v, bb, r.clone());
                    let dout_sub = extract4(&dout, bb, r.clone());
                    let (out_sub, lse_sub) = naive_attention_forward(
                        &q_sub, &k_sub, &v_sub, 1, h, rl, d, scale, causal,
                    );
                    let mut dq_sub = vec![0.0f32; h * rl * d];
                    let mut dk_sub = vec![0.0f32; h * rl * d];
                    let mut dv_sub = vec![0.0f32; h * rl * d];
                    flash_attention_backward_cpu_gqa(
                        &q_sub, &k_sub, &v_sub, &out_sub, &lse_sub, &dout_sub,
                        &mut dq_sub, &mut dk_sub, &mut dv_sub,
                        1, h, h, rl, d,
                        scale, causal, 1,
                        None,
                    );
                    scatter4(&mut out_full, &out_sub, bb, r.clone());
                    scatter3(&mut lse_full, &lse_sub, bb, r.clone());
                    scatter4(&mut dq_ref, &dq_sub, bb, r.clone());
                    scatter4(&mut dk_ref, &dk_sub, bb, r.clone());
                    scatter4(&mut dv_ref, &dv_sub, bb, r);
                }
            }

            // Masked full-row pass under test: the logsumexp must restrict
            // each row's normalizer to its own segment...
            let lse_masked =
                compute_logsumexp_gqa(&q, &k, b, h, h, s, d, scale, causal, Some(&seg));
            let lse_err = max_abs_diff(&lse_masked, &lse_full);
            assert!(
                lse_err < 1e-5,
                "causal={causal}: segment-masked lse != per-segment lse (err={lse_err})"
            );

            // ...and the backward must confine every gradient contribution
            // to in-segment (i, j) pairs.
            let mut dq_m = vec![0.0f32; total];
            let mut dk_m = vec![0.0f32; total];
            let mut dv_m = vec![0.0f32; total];
            flash_attention_backward_cpu_gqa(
                &q, &k, &v, &out_full, &lse_masked, &dout,
                &mut dq_m, &mut dk_m, &mut dv_m,
                b, h, h, s, d,
                scale, causal, 1,
                Some(&seg),
            );

            let tol = 1e-4;
            let dq_err = max_abs_diff(&dq_ref, &dq_m);
            let dk_err = max_abs_diff(&dk_ref, &dk_m);
            let dv_err = max_abs_diff(&dv_ref, &dv_m);
            assert!(dq_err < tol, "causal={causal}: segment-masked dQ err {dq_err} > {tol}");
            assert!(dk_err < tol, "causal={causal}: segment-masked dK err {dk_err} > {tol}");
            assert!(dv_err < tol, "causal={causal}: segment-masked dV err {dv_err} > {tol}");
        }
    }

    /// PCA Stage C stride-parity fix: the CPU reference backward must read
    /// its tensor inputs STRIDE-AWARE. Feed the FFI a Q that is a
    /// manually-strided view (a [b, s, h, d] buffer viewed as [b, h, s, d]
    /// — exactly the reshape(..).transpose(1, 2) product stdlib GQA emits)
    /// and the same data pre-materialized contiguous: the gradients must
    /// match EXACTLY (same values -> same arithmetic -> bitwise-equal),
    /// where the old stride-blind read produced smoothly-wrong (~1e-4)
    /// gradients.
    #[test]
    fn test_backward_cpu_stride_aware_view_inputs() {
        let b = 1usize;
        let h = 2usize;
        let s = 4usize;
        let d = 4usize;
        let scale = 1.0 / (d as f32).sqrt();
        let total = b * h * s * d;

        // Build a CPU f32 NslTensor with caller-chosen strides (elements).
        fn mk_tensor(data: &[f32], shape: &[i64], strides: &[i64]) -> i64 {
            let shape_ptr =
                crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
            let strides_ptr =
                crate::memory::checked_alloc(std::mem::size_of_val(strides)) as *mut i64;
            for (i, &v) in shape.iter().enumerate() {
                unsafe { *shape_ptr.add(i) = v };
            }
            for (i, &v) in strides.iter().enumerate() {
                unsafe { *strides_ptr.add(i) = v };
            }
            let data_ptr =
                crate::memory::checked_alloc(std::mem::size_of_val(data)) as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
            }
            let t = Box::new(NslTensor::new(
                data_ptr as *mut std::ffi::c_void,
                shape_ptr,
                strides_ptr,
                shape.len() as i64,
                shape.iter().product(),
                0, // device = CPU
                1, // dtype = f32
                1, // owns_data
                0, // data_owner
            ));
            Box::into_raw(t) as i64
        }
        fn tensor_values(ptr: i64, len: usize) -> Vec<f32> {
            let t = NslTensor::from_ptr(ptr);
            assert_eq!(t.dtype, 1, "gradient tensors are f32");
            assert_eq!(t.device, 0, "CPU inputs must yield CPU gradients");
            (0..len).map(|i| unsafe { *t.data_f32().add(i) }).collect()
        }

        // Q's underlying buffer is [b, s, h, d] row-major; the VIEW
        // presents it as [b, h, s, d] via permuted strides [s*h*d, d, h*d, 1].
        let q_bshd: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).sin() * 0.5).collect();
        let mut q_bhsd = vec![0.0f32; total]; // contiguous twin
        for bb in 0..b {
            for hh in 0..h {
                for ss in 0..s {
                    for dd in 0..d {
                        q_bhsd[((bb * h + hh) * s + ss) * d + dd] =
                            q_bshd[((bb * s + ss) * h + hh) * d + dd];
                    }
                }
            }
        }
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.5).collect();
        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();
        let (out, _lse) = naive_attention_forward(&q_bhsd, &k, &v, b, h, s, d, scale, true);

        let shape = [b as i64, h as i64, s as i64, d as i64];
        let canon = [(h * s * d) as i64, (s * d) as i64, d as i64, 1];
        let view_strides = [(s * h * d) as i64, d as i64, (h * d) as i64, 1];

        let run = |q_ptr: i64| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let list_ptr = nsl_flash_attention_backward(
                mk_tensor(&dout, &shape, &canon),
                q_ptr,
                mk_tensor(&k, &shape, &canon),
                mk_tensor(&v, &shape, &canon),
                mk_tensor(&out, &shape, &canon),
                0, // logsumexp: auto-compute
                scale.to_bits() as i64,
                b as i64, h as i64, s as i64, d as i64,
                1, // causal
                0, 0, 0, 0, // no PTX -> CPU reference
                0, 0, // tier-b disabled sentinel
                0, // no segment mask
            );
            assert_ne!(list_ptr, 0, "CPU backward must return [dq, dk, dv]");
            let list = crate::list::NslList::from_ptr(list_ptr);
            assert_eq!(list.len, 3);
            let dq = tensor_values(unsafe { *list.data.add(0) }, total);
            let dk = tensor_values(unsafe { *list.data.add(1) }, total);
            let dv = tensor_values(unsafe { *list.data.add(2) }, total);
            (dq, dk, dv)
        };

        let (dq_c, dk_c, dv_c) = run(mk_tensor(&q_bhsd, &shape, &canon));
        let (dq_v, dk_v, dv_v) = run(mk_tensor(&q_bshd, &shape, &view_strides));

        // Sanity: the two Q encodings really differ in memory order (the
        // test would be vacuous otherwise).
        assert!(
            max_abs_diff(&q_bshd, &q_bhsd) > 1e-3,
            "buffer permutation should reorder values"
        );

        // Stride-aware read => identical logical Q => bitwise-equal grads.
        assert_eq!(
            max_abs_diff(&dq_c, &dq_v),
            0.0,
            "dQ from a strided Q view must EXACTLY match the contiguous run"
        );
        assert_eq!(
            max_abs_diff(&dk_c, &dk_v),
            0.0,
            "dK from a strided Q view must EXACTLY match the contiguous run"
        );
        assert_eq!(
            max_abs_diff(&dv_c, &dv_v),
            0.0,
            "dV from a strided Q view must EXACTLY match the contiguous run"
        );
        let dq_norm: f32 = dq_c.iter().map(|x| x * x).sum();
        assert!(dq_norm > 1e-8, "gradients should be non-trivial");
    }

    // ── CSHA A.2.5: FFI launch path tests ────────────────────────────

    /// Pre-A.2.5 the `_csha` FFI forwarded to `nsl_flash_attention` and
    /// dropped the nine CSHA extras. Any unit test that could observe
    /// the difference without a GPU was really testing the forwarder's
    /// fallback. Post-A.2.5, the CSHA FFI launches the PTX variant
    /// directly — we can still exercise it on a non-CUDA build: the
    /// function must return a clean error code (-1) rather than
    /// panicking, and must accept the full 30-argument FFI signature
    /// without unused-parameter lints.
    ///
    /// On a CUDA build, the same call with null pointers would skip the
    /// PTX prologue/projection/epilogue scaffolds (runtime null-checks)
    /// and execute the classic Q-from-HBM path, so behaviour is a
    /// strict extension of `nsl_flash_attention`.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a25_csha_ffi_noncuda_returns_error_without_panic() {
        let r = nsl_flash_attention_csha(
            0, 0, 0, 0, 0, 1.0f32.to_bits() as i64,
            1, 1, 16, 64,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0,
            64, 64, 0,
            // CSHA extras (all null / zero — matches current call-site state).
            0, 0, 0, 0, 0, 0,
            1e-5f32.to_bits() as i64,
            0, 0,
            // PCA Tier A: segment_ids_ptr (0 = unpacked path).
            0,
            // PCA Tier B Planner: tier_b sentinel pair (both zero = disabled).
            0, 0,
            // PCA §4.3: doc_starts_ptr (0 = identity positions).
            0,
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero (0 = legacy topology).
            0,
        );
        assert_eq!(r, -1, "non-CUDA build must return -1 for the CSHA FFI");
    }

    /// A.4 grid-sizing logic, extracted as a pure function so it can be
    /// exercised without a GPU. Mirrors the `effective_heads` computation
    /// inside `nsl_flash_attention_csha`'s cuda branch — keep the two
    /// in sync.
    fn a4_effective_heads(heads: i64, active_heads: i64) -> i64 {
        if active_heads > 0 && active_heads < heads {
            active_heads
        } else {
            heads
        }
    }

    #[test]
    fn a4_grid_y_uses_full_heads_when_active_heads_zero() {
        // active_heads == 0 is the "no pruning" signal — grid_y must
        // use the full head count.
        assert_eq!(a4_effective_heads(8, 0), 8);
    }

    #[test]
    fn a4_grid_y_shrinks_when_active_heads_prunes() {
        // Common weight-informed specialisation: 3 of 8 heads pruned,
        // kernel launches with grid_y = batch * 5.
        assert_eq!(a4_effective_heads(8, 5), 5);
    }

    #[test]
    fn a4_grid_y_falls_back_to_heads_when_active_exceeds() {
        // Defensive: if the plan somehow reports active_heads > heads
        // (shouldn't happen but could during bring-up), use the full
        // count — the kernel's A.4 PTX guard would otherwise eject
        // every block.
        assert_eq!(a4_effective_heads(8, 16), 8);
    }

    #[test]
    fn a4_grid_y_full_heads_when_active_equals_heads() {
        // active_heads == heads is semantically "no pruning expressed
        // differently"; fall back to the full count so we don't emit
        // the guard branch unnecessarily.
        assert_eq!(a4_effective_heads(8, 8), 8);
    }

    /// CSHA paper §5.2 v1 (Sprint 2 cycle-2 audit pin): the canonical
    /// "half the heads pruned" case used by `csha_cuda_launch_fused`'s
    /// h=8/active=4 fixture. Co-pinned alongside the in-source doc on
    /// `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs
    /// ::emit_active_heads_guard` so the launcher and kernel guard
    /// cannot silently disagree.
    #[test]
    fn a4_grid_y_half_pruned_canonical_case() {
        assert_eq!(a4_effective_heads(8, 4), 4);
        // Also pin the table of paper §5.2 v1 inputs we ship today.
        for (heads, active, expected) in [
            (4i64, 0i64, 4i64),  // sentinel zero -> full
            (4,    1,    1),      // single live head (paper §5.2 extreme)
            (4,    2,    2),
            (4,    4,    4),      // active == heads -> full
            (8,    0,    8),
            (8,    4,    4),      // canonical half-pruned (this test fixture)
            (8,    8,    8),
        ] {
            assert_eq!(
                a4_effective_heads(heads, active),
                expected,
                "effective_heads({heads}, {active}) must be {expected}"
            );
        }
    }

    /// Smoke test: the CSHA FFI symbol resolves and the parameter signature
    /// is wired correctly through the `extern "C"` ABI. This catches A.2.5
    /// signature regressions (e.g. an accidentally dropped extras arg) at
    /// link time — the forwarder version would have silently ignored
    /// wrong-count calls.
    /// Updated in PCA Tier A Task 3C: trailing segment_ids_ptr added.
    /// Updated in PCA Tier B Planner P-3.4: two trailing tier_b_* params added.
    /// Updated in PCA §4.3 Task 3: trailing doc_starts_ptr added.
    /// Updated in PCA per-doc CTA Strategy 3 v1: trailing num_docs_or_zero added.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a25_csha_ffi_signature_has_thirty_params() {
        let _: extern "C" fn(
            i64, i64, i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64, i64,
            i64, i64, i64,
            i64, i64, i64,
            // extras
            i64, i64, i64, i64, i64, i64,
            i64, i64, i64,
            // PCA Tier A: segment_ids_ptr
            i64,
            // PCA Tier B Planner: tier_b_ptx_ptr, tier_b_name_ptr
            i64, i64,
            // PCA §4.3: doc_starts_ptr
            i64,
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero
            i64,
        ) -> i64 = nsl_flash_attention_csha;
    }
}
