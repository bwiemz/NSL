//! Launchers for bf16 parameter storage (`train(..., param_dtype=...)`,
//! `crate::param_storage`): the bf16 members of the multi-tensor AdamW
//! family and the split master's conversion kernels, all built from KIR
//! (`nsl_kir::kernels::optim`).
//!
//! The multi launch is `gpu_fase_fused_adamw_step_multi`'s, table for table:
//! a pinned staging block and device pointer tables cached at capacity, the
//! flat grid's block tables cached per shape list, `blockDim.x` equal to the
//! block the tables were built with. Its own workspace, because it stages six
//! tables (θ, the master or its low half, m, v, mp, the SR counters) where
//! the f32 launch stages four.

use std::ffi::c_void;

use nsl_kir::kernels::optim::Theta;

use super::{BlkTableCache, context, inner, kernels};

/// The f32 hyperparameters of one AdamW launch, in the kernels' order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdamwScalars {
    pub b1: f32,
    pub omb1: f32,
    pub b2: f32,
    pub omb2: f32,
    pub eps: f32,
    pub neg_lr: f32,
    pub neg_lr_wd: f32,
    pub bc1: f32,
    pub bc2: f32,
    pub has_wd: bool,
}

/// One bucket of bf16 parameters, as raw device pointers.
pub(crate) struct ParamBucket<'a> {
    /// The bf16 parameters (2 bytes per element).
    pub theta: &'a [u64],
    /// The f32 masters (`Bf16Master`) or the low halves (`Bf16Split`); empty
    /// for `Bf16Sr`.
    pub master: &'a [u64],
    pub m: &'a [u64],
    pub v: &'a [u64],
    pub mp: &'a [u64],
    /// `Bf16Sr`: each parameter's SR counter base; empty otherwise.
    pub ctr: &'a [u64],
    pub lens: &'a [u32],
}

/// One launch of the `theta` member of the family over `bucket`. `mp_scale`
/// is the clip factor folded into the gradient read (1.0 skips the multiply
/// in-kernel); `sr_key` is read by `Bf16Sr` only.
pub(crate) fn gpu_fase_fused_adamw_step_multi_param(
    theta: Theta,
    m_bf16: bool,
    bucket: &ParamBucket<'_>,
    s: AdamwScalars,
    mp_scale: f32,
    sr_key: u64,
) {
    let k = bucket.theta.len();
    if k == 0 {
        return;
    }
    assert!(
        theta != Theta::F32,
        "the f32 parameters launch through gpu_fase_fused_adamw_step_multi"
    );
    let needs_master = matches!(theta, Theta::Bf16Master(_) | Theta::Bf16Split);
    let needs_ctr = theta == Theta::Bf16Sr;
    assert!(
        [
            bucket.m.len(),
            bucket.v.len(),
            bucket.mp.len(),
            bucket.lens.len()
        ]
        .iter()
        .all(|&n| n == k)
            && bucket.master.len() == if needs_master { k } else { 0 }
            && bucket.ctr.len() == if needs_ctr { k } else { 0 },
        "param-storage multi adamw ({theta:?}): table length mismatch"
    );

    struct Ws {
        cap: usize,
        stage: u64,     // pinned host: 6*cap u64 + cap u32
        tabs: [u64; 6], // device u64 tables: theta, master, m, v, mp, ctr
        ntab: u64,      // device u32 table
        blk_cache: BlkTableCache,
    }
    struct WsSlot(*mut Ws);
    impl Default for WsSlot {
        fn default() -> Self {
            Self(std::ptr::null_mut())
        }
    }

    inner::set_oom_context("fase_fused_adamw_multi_param");
    let ws_raw = context::current().with_workspace(|c: &mut WsSlot| {
        let cur = c.0;
        let need_new = cur.is_null() || unsafe { (*cur).cap } < k;
        unsafe {
            inner::ensure_context();
            // Quiesce before releasing or rewriting anything a previous step's
            // uploads may still be reading.
            let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
            assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
            if need_new {
                if !cur.is_null() {
                    let mut old = Box::from_raw(cur);
                    for t in old.tabs {
                        inner::free_managed(t as *mut c_void);
                    }
                    inner::free_managed(old.ntab as *mut c_void);
                    old.blk_cache.free_all();
                    cudarc::driver::sys::cuMemFreeHost(old.stage as *mut c_void);
                }
                let mut stage: *mut c_void = std::ptr::null_mut();
                let r =
                    cudarc::driver::sys::cuMemAllocHost_v2(&mut stage, (6 * k * 8 + k * 4).max(8));
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "param-storage multi adamw: pinned staging alloc failed"
                );
                let tabs = std::array::from_fn(|_| inner::alloc_managed(k * 8) as u64);
                let ntab = inner::alloc_managed(k * 4) as u64;
                c.0 = Box::into_raw(Box::new(Ws {
                    cap: k,
                    stage: stage as u64,
                    tabs,
                    ntab,
                    blk_cache: BlkTableCache::default(),
                }));
            }
        }
        c.0
    });
    // SAFETY: set or reused above, non-null; this thread's alone while `ws` lives.
    let ws: &mut Ws = unsafe { &mut *ws_raw };

    let tables: [&[u64]; 6] = [
        bucket.theta,
        bucket.master,
        bucket.m,
        bucket.v,
        bucket.mp,
        bucket.ctr,
    ];
    unsafe {
        let base = ws.stage as *mut u64;
        for (slot, src) in tables.iter().enumerate() {
            std::ptr::copy_nonoverlapping(src.as_ptr(), base.add(slot * ws.cap), src.len());
        }
        let nbase = (ws.stage as usize + 6 * ws.cap * 8) as *mut u32;
        std::ptr::copy_nonoverlapping(bucket.lens.as_ptr(), nbase, k);
        let up = |dst: u64, src_off: usize, bytes: usize| {
            crate::host_profile::record_h2d(bytes);
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                (ws.stage as usize + src_off) as *const c_void,
                bytes,
                inner::current_stream(),
            );
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "param-storage multi adamw: table upload failed"
            );
        };
        for (slot, src) in tables.iter().enumerate() {
            if !src.is_empty() {
                up(ws.tabs[slot], slot * ws.cap * 8, k * 8);
            }
        }
        up(ws.ntab, 6 * ws.cap * 8, k * 4);
    }

    // The flat grid: `blockDim.x` must be the block the tables were built for.
    let block = 256i64;
    let (blk_param, blk_base, blk_count) =
        ws.blk_cache
            .get_or_build(bucket.lens, block as u32, "param-storage multi adamw");

    let [mut t0, mut tw, mut tm, mut tv, mut tmp, mut tctr] = ws.tabs;
    let mut tn = ws.ntab;
    let AdamwScalars {
        mut b1,
        mut omb1,
        mut b2,
        mut omb2,
        mut eps,
        mut neg_lr,
        mut neg_lr_wd,
        mut bc1,
        mut bc2,
        has_wd,
    } = s;
    let mut has_wd = u32::from(has_wd);
    let (mut bp, mut bb) = (blk_param, blk_base);
    let mut scale = mp_scale;
    let mut key = sr_key;
    let a = |p: &mut u64| p as *mut u64 as *mut c_void;
    let f = |p: &mut f32| p as *mut f32 as *mut c_void;
    // The kernel's signature (`build_param_multi`): the f32 kernel's, with
    // `wtab` after `ttab` (master variants) or `sr_key, ctrtab` after
    // `has_wd` (SR).
    let mut args: Vec<*mut c_void> = vec![a(&mut t0)];
    if needs_master {
        args.push(a(&mut tw));
    }
    args.extend([a(&mut tm), a(&mut tv), a(&mut tmp), a(&mut tn)]);
    args.extend([
        f(&mut b1),
        f(&mut omb1),
        f(&mut b2),
        f(&mut omb2),
        f(&mut eps),
        f(&mut neg_lr),
        f(&mut neg_lr_wd),
        f(&mut bc1),
        f(&mut bc2),
    ]);
    args.push(&mut has_wd as *mut u32 as *mut c_void);
    if needs_ctr {
        args.push(a(&mut key));
        args.push(a(&mut tctr));
    }
    args.extend([a(&mut bp), a(&mut bb), f(&mut scale)]);

    let ptx = kernels::fase_param_multi_ptx(theta, m_bf16);
    let name = format!("{}\0", nsl_kir::kernels::optim::multi_name(theta, m_bf16));
    let result = inner::kernel_launch(
        ptx.as_ptr(),
        name.as_ptr(),
        [blk_count as i64, 1, 1],
        [block, 1, 1],
        &args,
        0,
    );
    assert_eq!(
        result as u32,
        0,
        "GPU {} failed: {}",
        nsl_kir::kernels::optim::multi_name(theta, m_bf16),
        result as u32
    );
    inner::sync_after_kernel();
}

/// `hi = split_hi(src)`, `lo = split_lo(src)` over `n` f32 elements (raw
/// device buffers): an f32 parameter into the split storage.
pub(crate) fn gpu_param_split(src_f32: u64, hi: u64, lo: u64, n: usize) {
    launch_elementwise(
        kernels::param_split_ptx(),
        b"nsl_param_split_f32\0",
        [src_f32, hi, lo],
        n,
    );
}

/// `dst = join_f32(hi, lo)` over `n` elements (raw device buffers): the
/// exact f32 master of a split parameter.
pub(crate) fn gpu_param_join(hi: u64, lo: u64, dst_f32: u64, n: usize) {
    launch_elementwise(
        kernels::param_join_ptx(),
        b"nsl_param_join_f32\0",
        [hi, lo, dst_f32],
        n,
    );
}

/// One thread per element over `n`: the three pointers, then `n`.
fn launch_elementwise(ptx: &str, name: &[u8], ptrs: [u64; 3], n: usize) {
    if n == 0 {
        return;
    }
    let [mut p0, mut p1, mut p2] = ptrs;
    let mut n_val = n as u64;
    let args = [
        &mut p0 as *mut u64 as *mut c_void,
        &mut p1 as *mut u64 as *mut c_void,
        &mut p2 as *mut u64 as *mut c_void,
        &mut n_val as *mut u64 as *mut c_void,
    ];
    let block = 256i64;
    let grid = (n as i64 + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(),
        name.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU param-storage conversion failed: {}",
        result as u32
    );
    inner::sync_after_kernel();
}
