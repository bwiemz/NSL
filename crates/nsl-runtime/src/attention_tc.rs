//! Launchers for the tensor-core flash attention the runtime builds for
//! itself (`nsl_kir::kernels::flash_attn`): the training path's
//! `scaled_dot_product_attention` forward and backward on f32
//! `[batch, heads, seq, head_dim]` tensors, head_dim 64 or 128, causal or
//! not, K/V with `kv_heads` dividing `heads` (grouped-query attention without
//! expanding K/V), any sequence length.
//!
//! The kernels and their precision contract are documented in
//! `nsl_kir::kernels::flash_attn`. This module builds each variant's PTX
//! once per process, marshals the launch, and owns nothing else: the
//! callers in `flash_attention.rs` allocate and free every buffer.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys::CUresult;
use nsl_kir::kernels::flash_attn::bwd::{self as kbwd, BwdConfig};
use nsl_kir::kernels::flash_attn::fwd::FwdConfig;

use crate::cuda::inner;

/// A variant's PTX module and entry name, both null-terminated, built once
/// and kept for the life of the process (a handful of variants).
struct Built {
    ptx: &'static [u8],
    name: &'static CString,
}

/// Which kernel of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Which {
    Fwd(FwdConfig),
    Pre(u32),
    Dq(BwdConfig),
    Dkdv(BwdConfig),
}

fn built(which: Which) -> &'static Built {
    static CACHE: OnceLock<Mutex<HashMap<Which, &'static Built>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.entry(which).or_insert_with(|| {
        let (ptx, name) = match which {
            Which::Fwd(c) => (nsl_kir::kernels::flash_attn::fwd::ptx(c), c.kernel_name()),
            Which::Pre(hd) => {
                let c = BwdConfig::pre_only(hd);
                (kbwd::pre_ptx(c), c.pre_kernel_name())
            }
            Which::Dq(c) => (kbwd::dq_ptx(c), c.dq_kernel_name()),
            Which::Dkdv(c) => (kbwd::dkdv_ptx(c), c.dkdv_kernel_name()),
        };
        let name = CString::new(name).expect("kernel names have no NUL");
        Box::leak(Box::new(Built { ptx: Box::leak(ptx.into_boxed_slice()), name: Box::leak(Box::new(name)) }))
    })
}

fn launch(which: Which, grid: [u32; 3], threads: u32, args: &[*mut c_void], smem: u32) -> CUresult {
    let b = built(which);
    inner::kernel_launch(
        b.ptx.as_ptr(),
        b.name.as_ptr() as *const u8,
        [grid[0] as i64, grid[1] as i64, grid[2] as i64],
        [threads as i64, 1, 1],
        args,
        smem,
    )
}

/// The largest `gridDim.y` a launch may use.
const MAX_GRID_Y: u64 = 65_535;

/// The compute capability these kernels need: `ldmatrix`, `mma.sync
/// m16n8k16` with f16 operands and the f16x2 packing `cvt.rn.f16x2.f32` are
/// sm_80 instructions, and the PTX is printed for `.target sm_80`.
pub(crate) const MIN_SM: u32 = 80;

/// Whether the current device can run these kernels (compute capability
/// at least [`MIN_SM`]), asked of the driver once per device. Below sm_80
/// the module would fail to JIT on every call before falling back; with
/// this the callers never attempt it.
pub(crate) fn device_supported() -> bool {
    static CACHE: OnceLock<Mutex<HashMap<i32, bool>>> = OnceLock::new();
    let ctx = crate::cuda::context::current();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    *guard.entry(ctx.device).or_insert_with(|| ctx.sm_version() >= MIN_SM)
}

/// Whether these kernels serve a problem of this shape: head_dim 64 or 128,
/// `heads` a positive multiple of `kv_heads`, every dimension positive, the
/// per-head element count and the row count `batch · heads · seq` within
/// `u32` (the kernels index them in 32 bits), and `batch · heads` within a
/// launch's `gridDim.y`.
pub(crate) fn supports(batch: usize, head_dim: usize, heads: usize, kv_heads: usize, seq: usize) -> bool {
    let (b, h, s, d) = (batch as u64, heads as u64, seq as u64, head_dim as u64);
    FwdConfig::supported(head_dim as u32)
        && b > 0
        && s > 0
        && h > 0
        && kv_heads > 0
        && heads.is_multiple_of(kv_heads)
        && s * d < u32::MAX as u64
        && b * h * s < u32::MAX as u64
        && b * h <= MAX_GRID_Y
}

/// Device pointers and shape of one attention problem. `q`/`o` are
/// `[batch, heads, seq, head_dim]`, `k`/`v` `[batch, kv_heads, seq,
/// head_dim]`, `lse` `[batch, heads, seq]`, all f32 and contiguous.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FwdLaunch {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub lse: u64,
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub seq: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub causal: bool,
}

/// Launch the forward over the whole batch (one launch). Writes `o` and
/// `lse`; returns the launch status (asynchronous faults surface at the
/// next synchronisation, as for every kernel).
pub(crate) fn forward(a: &FwdLaunch) -> CUresult {
    forward_with(FwdConfig::new(a.head_dim as u32, a.causal), a)
}

/// [`forward`] with an explicit tiling (`cfg` must match `a`'s head_dim and
/// causality).
pub(crate) fn forward_with(cfg: FwdConfig, a: &FwdLaunch) -> CUresult {
    debug_assert!(supports(a.batch, a.head_dim, a.heads, a.kv_heads, a.seq));
    debug_assert!(cfg.head_dim as usize == a.head_dim && cfg.causal == a.causal);
    let grid = cfg.grid(a.batch as u32, a.heads as u32, a.seq as u32);
    let (mut q, mut k, mut v, mut o, mut lse) = (a.q, a.k, a.v, a.o, a.lse);
    let mut seq = a.seq as u32;
    let mut heads = a.heads as u32;
    let mut kv_heads = a.kv_heads as u32;
    let mut scale = a.scale;
    let args: [*mut c_void; 9] = [
        &mut q as *mut _ as *mut c_void,
        &mut k as *mut _ as *mut c_void,
        &mut v as *mut _ as *mut c_void,
        &mut o as *mut _ as *mut c_void,
        &mut lse as *mut _ as *mut c_void,
        &mut seq as *mut _ as *mut c_void,
        &mut heads as *mut _ as *mut c_void,
        &mut kv_heads as *mut _ as *mut c_void,
        &mut scale as *mut _ as *mut c_void,
    ];
    launch(Which::Fwd(cfg), grid, cfg.threads(), &args, 0)
}

/// Device pointers and shape of one attention backward. Shapes as in
/// [`FwdLaunch`]; `o` and `lse` are the forward's outputs, `dout` the
/// output gradient (`o`'s shape); `dq` (`q`'s shape) and `dk`/`dv` (`k`'s
/// shape) are written in full. `dvec` (`[batch, heads, seq]` f32) and `gmax`
/// (`[batch, heads]` u32) are scratch the caller provides.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BwdLaunch {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub lse: u64,
    pub dout: u64,
    pub dq: u64,
    pub dk: u64,
    pub dv: u64,
    pub dvec: u64,
    pub gmax: u64,
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub seq: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub causal: bool,
}

/// Launch the backward: the D / max|dO| pre-pass, then the dQ kernel and
/// the dK/dV kernel, in stream order. Returns the first failing launch's
/// status.
pub(crate) fn backward(a: &BwdLaunch) -> CUresult {
    backward_with(BwdConfig::new(a.head_dim as u32, a.causal), a, [true, true, true])
}

/// [`backward`] with explicit tilings, launching only the kernels `run`
/// selects (`[pre-pass, dQ, dK/dV]`; the timing diagnostic times them one
/// by one).
pub(crate) fn backward_with(cfg: BwdConfig, a: &BwdLaunch, run: [bool; 3]) -> CUresult {
    debug_assert!(supports(a.batch, a.head_dim, a.heads, a.kv_heads, a.seq));
    debug_assert!(cfg.head_dim as usize == a.head_dim && cfg.causal == a.causal);
    let rows = (a.batch * a.heads * a.seq) as u32;

    let (mut q, mut k, mut v, mut o, mut lse, mut dout) = (a.q, a.k, a.v, a.o, a.lse, a.dout);
    let (mut dq, mut dk, mut dv, mut dvec, mut gmax) = (a.dq, a.dk, a.dv, a.dvec, a.gmax);
    let mut rows_arg = rows;
    let mut seq = a.seq as u32;
    let mut heads = a.heads as u32;
    let mut kv_heads = a.kv_heads as u32;
    let mut scale = a.scale;
    let p = |x: &mut u64| x as *mut u64 as *mut c_void;
    let u = |x: &mut u32| x as *mut u32 as *mut c_void;

    if run[0] {
        // gmax holds a running maximum: it must start at zero.
        inner::memset_d8(a.gmax as *mut c_void, a.batch * a.heads * 4);
        let pre_args = [p(&mut dout), p(&mut o), p(&mut dvec), p(&mut gmax), u(&mut rows_arg), u(&mut seq)];
        let rc = launch(Which::Pre(cfg.head_dim), cfg.pre_grid(rows), cfg.pre_threads(), &pre_args, 0);
        if rc != CUresult::CUDA_SUCCESS {
            return rc;
        }
    }
    let dq_args = [
        p(&mut q),
        p(&mut k),
        p(&mut v),
        p(&mut dout),
        p(&mut lse),
        p(&mut dvec),
        p(&mut dq),
        u(&mut seq),
        u(&mut heads),
        u(&mut kv_heads),
        &mut scale as *mut f32 as *mut c_void,
    ];
    if run[1] {
        let grid = cfg.dq_grid(a.batch as u32, a.heads as u32, a.seq as u32);
        let rc = launch(Which::Dq(cfg), grid, cfg.dq_threads(), &dq_args, 0);
        if rc != CUresult::CUDA_SUCCESS {
            return rc;
        }
    }
    let dkdv_args = [
        p(&mut q),
        p(&mut k),
        p(&mut v),
        p(&mut dout),
        p(&mut lse),
        p(&mut dvec),
        p(&mut gmax),
        p(&mut dk),
        p(&mut dv),
        u(&mut seq),
        u(&mut heads),
        u(&mut kv_heads),
        &mut scale as *mut f32 as *mut c_void,
    ];
    if run[2] {
        let grid = cfg.dkdv_grid(a.batch as u32, a.kv_heads as u32, a.seq as u32);
        return launch(Which::Dkdv(cfg), grid, cfg.dkdv_threads(), &dkdv_args, cfg.dkdv_smem_bytes());
    }
    CUresult::CUDA_SUCCESS
}

#[cfg(test)]
pub(crate) mod tests {
    //! GPU checks of the kernels against f64 references. Each case runs
    //! two references: the exact attention of the f32 inputs (the
    //! tolerance is what f16 operands cost), and the same attention of the
    //! inputs rounded to f16 the way the kernel rounds its operands (a
    //! tight tolerance: what is left is the f16 rounding of `P` and f32
    //! accumulation order), so an indexing or masking error that the
    //! operand tolerance could hide still fails.

    use super::*;

    /// Round to the nearest f16 (ties to even), back to f64.
    pub(crate) fn f16_round(x: f32) -> f64 {
        half_round(x as f64)
    }

    fn half_round(x: f64) -> f64 {
        if !x.is_finite() || x == 0.0 {
            return x;
        }
        let a = x.abs();
        // f16: 10 fraction bits, exponent range [-14, 15], subnormals 2^-24.
        let e = a.log2().floor().max(-14.0) as i32;
        let ulp = 2f64.powi(e - 10);
        let r = (a / ulp).round_ties_even() * ulp;
        let r = if r > 65504.0 { f64::INFINITY } else { r };
        r.copysign(x)
    }

    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        pub fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * amp).collect()
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Shape {
        pub b: usize,
        pub h: usize,
        pub kvh: usize,
        pub s: usize,
        pub d: usize,
        pub causal: bool,
    }

    impl Shape {
        pub fn kv_head(&self, h: usize) -> usize {
            h / (self.h / self.kvh)
        }
    }

    /// f64 attention forward: `(O, lse)` with `round` applied to every Q,
    /// K, V element first.
    pub(crate) fn reference_forward(
        sh: Shape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        scale: f64,
        round: fn(f32) -> f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let Shape { b, h, kvh, s, d, causal } = sh;
        let mut o = vec![0.0; b * h * s * d];
        let mut lse = vec![0.0; b * h * s];
        for bi in 0..b {
            for hi in 0..h {
                let qb = (bi * h + hi) * s * d;
                let kb = (bi * kvh + sh.kv_head(hi)) * s * d;
                for i in 0..s {
                    let n = if causal { i + 1 } else { s };
                    let sc: Vec<f64> = (0..n)
                        .map(|j| {
                            (0..d).map(|x| round(q[qb + i * d + x]) * round(k[kb + j * d + x])).sum::<f64>() * scale
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let p: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                    let l: f64 = p.iter().sum();
                    for x in 0..d {
                        o[qb + i * d + x] = (0..n).map(|j| p[j] * round(v[kb + j * d + x])).sum::<f64>() / l;
                    }
                    lse[(bi * h + hi) * s + i] = m + l.ln();
                }
            }
        }
        (o, lse)
    }

    pub(crate) fn upload(x: &[f32]) -> *mut c_void {
        let p = inner::alloc_managed(x.len().max(1) * 4);
        inner::memcpy_htod(p, x.as_ptr() as *const c_void, x.len() * 4);
        p
    }

    pub(crate) fn download(p: *mut c_void, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        inner::memcpy_dtoh(out.as_mut_ptr() as *mut c_void, p as *const c_void, n * 4);
        out
    }

    /// Fill with a NaN pattern so an unwritten output cannot pass.
    pub(crate) fn poisoned(n: usize) -> *mut c_void {
        let p = inner::alloc_managed(n.max(1) * 4);
        inner::memset_d8_value(p, 0xFF, n.max(1) * 4);
        p
    }

    /// `(max |a - b|, index)` over finite pairs; panics on a non-finite `a`.
    pub(crate) fn max_err(a: &[f32], b: &[f64]) -> (f64, usize) {
        let mut worst = (0.0, 0);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(x.is_finite(), "non-finite output at {i}: {x}");
            let e = (x as f64 - y).abs();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        worst
    }

    fn run_forward(sh: Shape, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> (Vec<f32>, Vec<f32>) {
        let (qd, kd, vd) = (upload(q), upload(k), upload(v));
        let n_o = sh.b * sh.h * sh.s * sh.d;
        let n_l = sh.b * sh.h * sh.s;
        let od = poisoned(n_o);
        let ld = poisoned(n_l);
        let rc = forward(&FwdLaunch {
            q: qd as u64,
            k: kd as u64,
            v: vd as u64,
            o: od as u64,
            lse: ld as u64,
            batch: sh.b,
            heads: sh.h,
            kv_heads: sh.kvh,
            seq: sh.s,
            head_dim: sh.d,
            scale,
            causal: sh.causal,
        });
        assert_eq!(rc, CUresult::CUDA_SUCCESS, "forward launch {sh:?}");
        let sync = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
        assert_eq!(sync, CUresult::CUDA_SUCCESS, "forward execution {sh:?}");
        let o = download(od, n_o);
        let l = download(ld, n_l);
        for p in [qd, kd, vd, od, ld] {
            inner::free_managed(p);
        }
        (o, l)
    }

    fn check_forward(sh: Shape, amp_q: f32, seed: u64) {
        let mut rng = Lcg(seed);
        let q = rng.vec(sh.b * sh.h * sh.s * sh.d, amp_q);
        let k = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let v = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let scale = 1.0 / (sh.d as f32).sqrt();
        let (o, lse) = run_forward(sh, &q, &k, &v, scale);
        let (o_ex, l_ex) = reference_forward(sh, &q, &k, &v, scale as f64, |x| x as f64);
        let (o_rd, l_rd) = reference_forward(sh, &q, &k, &v, scale as f64, f16_round);
        let (eo, io) = max_err(&o, &o_ex);
        let (el, il) = max_err(&lse, &l_ex);
        let (ro, iro) = max_err(&o, &o_rd);
        let (rl, irl) = max_err(&lse, &l_rd);
        eprintln!(
            "{sh:?} amp_q={amp_q}: O vs exact {eo:.2e} (at {io}), lse vs exact {el:.2e} (at {il}); \
             O vs f16-operand model {ro:.2e} (at {iro}), lse {rl:.2e} (at {irl})"
        );
        // f16 operands: |V| <= 1, each operand off by <= 2^-11 relative.
        assert!(eo < 4e-3, "O vs exact: {eo} at {io}");
        assert!(el < 4e-3 * amp_q as f64, "lse vs exact: {el} at {il}");
        // With the operands rounded as the kernel rounds them, what is left
        // is P's f16 rounding (relative 2^-11 per weight, averaged over the
        // row) and f32 accumulation.
        assert!(ro < 5e-4, "O vs f16-operand model: {ro} at {iro}");
        assert!(rl < 2e-5 * (1.0 + amp_q as f64), "lse vs f16-operand model: {rl} at {irl}");
    }

    fn gpu() -> bool {
        if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
            return false;
        }
        inner::ensure_context();
        true
    }


    /// Round to an 11-bit significand without range limits: what an exact
    /// power-of-two prescale followed by f16 rounding does to a gradient in
    /// f16's normal range.
    pub(crate) fn sig11_round(x: f32) -> f64 {
        let x = x as f64;
        if !x.is_finite() || x == 0.0 {
            return x;
        }
        let e = x.abs().log2().floor() as i32;
        let ulp = 2f64.powi(e - 10);
        (x / ulp).round_ties_even() * ulp
    }

    /// f64 attention backward: `(dQ, dK, dV)` for output gradient `dout`,
    /// with `round` applied to Q, K, V and `round_do` to dO. dK/dV of a kv
    /// head sum over its group's query heads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_backward(
        sh: Shape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        dout: &[f32],
        scale: f64,
        round: fn(f32) -> f64,
        round_do: fn(f32) -> f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let Shape { b, h, kvh, s, d, causal } = sh;
        let mut dq = vec![0.0; b * h * s * d];
        let mut dk = vec![0.0; b * kvh * s * d];
        let mut dv = vec![0.0; b * kvh * s * d];
        for bi in 0..b {
            for hi in 0..h {
                let qb = (bi * h + hi) * s * d;
                let kb = (bi * kvh + sh.kv_head(hi)) * s * d;
                for i in 0..s {
                    let n = if causal { i + 1 } else { s };
                    let sc: Vec<f64> = (0..n)
                        .map(|j| {
                            (0..d).map(|x| round(q[qb + i * d + x]) * round(k[kb + j * d + x])).sum::<f64>() * scale
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                    let l: f64 = e.iter().sum();
                    let p: Vec<f64> = e.iter().map(|x| x / l).collect();
                    let o: Vec<f64> =
                        (0..d).map(|x| (0..n).map(|j| p[j] * round(v[kb + j * d + x])).sum::<f64>()).collect();
                    let g: Vec<f64> = (0..d).map(|x| round_do(dout[qb + i * d + x])).collect();
                    let dd: f64 = (0..d).map(|x| g[x] * o[x]).sum();
                    for j in 0..n {
                        let dp: f64 = (0..d).map(|x| g[x] * round(v[kb + j * d + x])).sum();
                        let ds = p[j] * (dp - dd);
                        for x in 0..d {
                            dq[qb + i * d + x] += scale * ds * round(k[kb + j * d + x]);
                            dk[kb + j * d + x] += scale * ds * round(q[qb + i * d + x]);
                            dv[kb + j * d + x] += p[j] * g[x];
                        }
                    }
                }
            }
        }
        (dq, dk, dv)
    }

    /// `(relative Frobenius error, max |error| / max |reference|)`. A
    /// reference that is exactly zero (seq 1: dQ and dK vanish) has no
    /// relative error; the errors are then measured against `zero_scale`
    /// per element instead.
    pub(crate) fn rel_err(got: &[f32], want: &[f64], zero_scale: f64) -> (f64, f64) {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut emax: f64 = 0.0;
        let mut wmax: f64 = 0.0;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "non-finite gradient at {i}: {g}");
            let e = g as f64 - w;
            num += e * e;
            den += w * w;
            emax = emax.max(e.abs());
            wmax = wmax.max(w.abs());
        }
        if den == 0.0 {
            let n = want.len().max(1) as f64;
            return ((num / n).sqrt() / zero_scale, emax / zero_scale);
        }
        ((num / den).sqrt(), emax / wmax)
    }

    pub(crate) struct Grads {
        pub dq: Vec<f32>,
        pub dk: Vec<f32>,
        pub dv: Vec<f32>,
    }

    /// Forward then backward on the GPU with these kernels.
    pub(crate) fn run_backward(sh: Shape, q: &[f32], k: &[f32], v: &[f32], dout: &[f32], scale: f32) -> Grads {
        let n_q = sh.b * sh.h * sh.s * sh.d;
        let n_kv = sh.b * sh.kvh * sh.s * sh.d;
        let n_l = sh.b * sh.h * sh.s;
        let (qd, kd, vd, dod) = (upload(q), upload(k), upload(v), upload(dout));
        let od = poisoned(n_q);
        let ld = poisoned(n_l);
        let rc = forward(&FwdLaunch {
            q: qd as u64,
            k: kd as u64,
            v: vd as u64,
            o: od as u64,
            lse: ld as u64,
            batch: sh.b,
            heads: sh.h,
            kv_heads: sh.kvh,
            seq: sh.s,
            head_dim: sh.d,
            scale,
            causal: sh.causal,
        });
        assert_eq!(rc, CUresult::CUDA_SUCCESS);
        let (dqd, dkd, dvd) = (poisoned(n_q), poisoned(n_kv), poisoned(n_kv));
        let dvec = poisoned(n_l);
        let gmax = poisoned(sh.b * sh.h);
        let rc = backward(&BwdLaunch {
            q: qd as u64,
            k: kd as u64,
            v: vd as u64,
            o: od as u64,
            lse: ld as u64,
            dout: dod as u64,
            dq: dqd as u64,
            dk: dkd as u64,
            dv: dvd as u64,
            dvec: dvec as u64,
            gmax: gmax as u64,
            batch: sh.b,
            heads: sh.h,
            kv_heads: sh.kvh,
            seq: sh.s,
            head_dim: sh.d,
            scale,
            causal: sh.causal,
        });
        assert_eq!(rc, CUresult::CUDA_SUCCESS, "backward launch {sh:?}");
        let sync = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
        assert_eq!(sync, CUresult::CUDA_SUCCESS, "backward execution {sh:?}");
        let g = Grads { dq: download(dqd, n_q), dk: download(dkd, n_kv), dv: download(dvd, n_kv) };
        for p in [qd, kd, vd, dod, od, ld, dqd, dkd, dvd, dvec, gmax] {
            inner::free_managed(p);
        }
        g
    }

    /// One backward case at gradient amplitude `amp` (dO uniform in
    /// `±amp`), checked against the exact f64 gradients and against the
    /// f16-operand model.
    fn check_backward(sh: Shape, amp: f32, seed: u64) {
        let mut rng = Lcg(seed);
        let q = rng.vec(sh.b * sh.h * sh.s * sh.d, 1.0);
        let k = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let v = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let dout = rng.vec(sh.b * sh.h * sh.s * sh.d, amp);
        let scale = 1.0 / (sh.d as f32).sqrt();
        let got = run_backward(sh, &q, &k, &v, &dout, scale);
        let exact = reference_backward(sh, &q, &k, &v, &dout, scale as f64, |x| x as f64, |x| x as f64);
        let model = reference_backward(sh, &q, &k, &v, &dout, scale as f64, f16_round, sig11_round);
        for (name, g, e, m) in [
            ("dQ", &got.dq, &exact.0, &model.0),
            ("dK", &got.dk, &exact.1, &model.1),
            ("dV", &got.dv, &exact.2, &model.2),
        ] {
            let (fe, me) = rel_err(g, e, amp as f64);
            let (fm, mm) = rel_err(g, m, amp as f64);
            eprintln!(
                "{sh:?} amp={amp:e} {name}: vs exact frob {fe:.2e} max {me:.2e}; vs f16-operand model frob {fm:.2e} max {mm:.2e}"
            );
            // f16 operands (relative 2^-11 each) through two chained
            // products: a few 1e-3 in norm. Independent of `amp`: that is
            // what the prescale is for.
            assert!(fe < 5e-3 && me < 2e-2, "{name} vs exact: frob {fe}, max {me}");
            // Against the model, what is left is P's and dS's f16 rounding.
            assert!(fm < 2.5e-3 && mm < 1e-2, "{name} vs model: frob {fm}, max {mm}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
    fn tc_backward_matches_f64_reference_at_realistic_gradient_magnitudes() {
        if !gpu() {
            return;
        }
        let cases = [
            Shape { b: 2, h: 2, kvh: 2, s: 256, d: 64, causal: true },
            Shape { b: 1, h: 4, kvh: 1, s: 200, d: 64, causal: true },
            Shape { b: 1, h: 2, kvh: 2, s: 160, d: 64, causal: false },
            Shape { b: 1, h: 4, kvh: 2, s: 192, d: 128, causal: true },
            Shape { b: 2, h: 2, kvh: 1, s: 77, d: 128, causal: false },
            Shape { b: 1, h: 2, kvh: 2, s: 1, d: 64, causal: true },
        ];
        for (i, sh) in cases.into_iter().enumerate() {
            for amp in [1e-6f32, 1e-7, 1.0] {
                check_backward(sh, amp, 0xb0b + i as u64);
            }
        }
    }

    /// The dQ prescale is per row: a row's dQ depends on that row's dO
    /// alone, bit for bit, however large the other rows' gradients are.
    /// Rows at 1e-8 next to rows at 1e-1 (a ratio no shared scale keeps in
    /// f16's normal range: 1e-8 scaled with the 1e-1 rows lands at ~6e-6,
    /// subnormal) give exactly the dQ they give alone, and that dQ is the
    /// exact gradient to f16-operand precision. dK/dV of the mixed batch
    /// (one scale per group there) still match the exact gradient.
    #[test]
    #[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
    fn tc_backward_small_rows_keep_their_precision() {
        if !gpu() {
            return;
        }
        let sh = Shape { b: 1, h: 2, kvh: 1, s: 192, d: 64, causal: true };
        let mut rng = Lcg(4242);
        let q = rng.vec(sh.b * sh.h * sh.s * sh.d, 1.0);
        let k = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let v = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let base = rng.vec(sh.b * sh.h * sh.s * sh.d, 1.0);
        let big_row = |i: usize| (i / sh.d).is_multiple_of(8);
        let mixed: Vec<f32> =
            base.iter().enumerate().map(|(i, &x)| x * if big_row(i) { 1e-1 } else { 1e-8 }).collect();
        let small_only: Vec<f32> =
            base.iter().enumerate().map(|(i, &x)| if big_row(i) { 0.0 } else { x * 1e-8 }).collect();
        let scale = 1.0 / (sh.d as f32).sqrt();
        let a = run_backward(sh, &q, &k, &v, &mixed, scale);
        let b = run_backward(sh, &q, &k, &v, &small_only, scale);
        for (i, (&x, &y)) in a.dq.iter().zip(&b.dq).enumerate() {
            if !big_row(i) {
                assert_eq!(x.to_bits(), y.to_bits(), "dQ element {i} depends on other rows' dO: {x} vs {y}");
            }
        }
        let ex_b = reference_backward(sh, &q, &k, &v, &small_only, scale as f64, |x| x as f64, |x| x as f64);
        let (fe, me) = rel_err(&b.dq, &ex_b.0, 1e-8);
        eprintln!("small rows alone: dQ frob {fe:.2e} max {me:.2e}");
        assert!(fe < 5e-3 && me < 2e-2, "small-row dQ: frob {fe}, max {me}");
        let ex_a = reference_backward(sh, &q, &k, &v, &mixed, scale as f64, |x| x as f64, |x| x as f64);
        for (name, g, e) in [("dQ", &a.dq, &ex_a.0), ("dK", &a.dk, &ex_a.1), ("dV", &a.dv, &ex_a.2)] {
            let (fe, me) = rel_err(g, e, 1e-1);
            eprintln!("mixed rows {name}: frob {fe:.2e} max {me:.2e}");
            assert!(fe < 5e-3 && me < 2e-2, "{name}: frob {fe}, max {me}");
        }
    }

    /// A GPU f32 tensor of `shape` holding `vals`.
    fn tensor_of(shape: &[i64], vals: &[f32]) -> i64 {
        let list = crate::list::nsl_list_new();
        for &d in shape {
            crate::list::nsl_list_push(list, d);
        }
        let t = crate::tensor::nsl_tensor_zeros_on(list, 1);
        crate::list::nsl_list_free(list);
        let p = crate::tensor::nsl_tensor_data_ptr(t) as *mut c_void;
        inner::memcpy_htod(p, vals.as_ptr() as *const c_void, vals.len() * 4);
        t
    }

    fn tensor_values(t: i64, n: usize) -> Vec<f32> {
        let c = crate::tensor::nsl_tensor_contiguous(t);
        let v = download(crate::tensor::nsl_tensor_data_ptr(c) as *mut c_void, n);
        crate::tensor::nsl_tensor_free(c);
        v
    }

    /// The runtime entry points the compiler calls, end to end: the fused
    /// forward FFI serves an unpacked grouped-query batch whole (K/V not
    /// expanded) and returns `[out, lse]`; the backward FFI, handed the
    /// output gradient as a NON-contiguous view (as the decorator-free
    /// backward hands it) and either the saved lse or none (recomputed),
    /// returns kv-shaped dK/dV. Everything is checked against the f64
    /// reference at a realistic gradient magnitude.
    #[test]
    #[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
    fn tc_ffi_forward_and_backward_match_the_reference() {
        if !gpu() {
            return;
        }
        let sh = Shape { b: 2, h: 4, kvh: 2, s: 150, d: 64, causal: true };
        let mut rng = Lcg(31337);
        let q = rng.vec(sh.b * sh.h * sh.s * sh.d, 1.0);
        let k = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let v = rng.vec(sh.b * sh.kvh * sh.s * sh.d, 1.0);
        let dout = rng.vec(sh.b * sh.h * sh.s * sh.d, 1e-6);
        let scale = 1.0f32 / (sh.d as f32).sqrt();
        let (b, h, kvh, s, d) = (sh.b as i64, sh.h as i64, sh.kvh as i64, sh.s as i64, sh.d as i64);
        let qt = tensor_of(&[b, h, s, d], &q);
        let kt = tensor_of(&[b, kvh, s, d], &k);
        let vt = tensor_of(&[b, kvh, s, d], &v);
        // dO as a [b, s, h, d] buffer viewed [b, h, s, d].
        let mut dout_bshd = vec![0.0f32; dout.len()];
        for bi in 0..sh.b {
            for hi in 0..sh.h {
                for si in 0..sh.s {
                    for x in 0..sh.d {
                        dout_bshd[((bi * sh.s + si) * sh.h + hi) * sh.d + x] =
                            dout[((bi * sh.h + hi) * sh.s + si) * sh.d + x];
                    }
                }
            }
        }
        let dbuf = tensor_of(&[b, s, h, d], &dout_bshd);
        let dview = crate::tensor::nsl_tensor_transpose(dbuf, 1, 2);

        let before = crate::flash_attention::nsl_sdpa_fused_launch_count(0);
        let list = crate::flash_attention::nsl_sdpa_fused_forward(
            qt, kt, vt, scale.to_bits() as i64, 1, 0, 0, 0, 0, 0, 64, 64, 0,
        );
        assert_ne!(list, 0, "the fused forward declined an unpacked GQA batch");
        assert_eq!(crate::flash_attention::nsl_sdpa_fused_launch_count(0), before + 1);
        let out = crate::list::nsl_list_get(list, 0);
        let lse = crate::list::nsl_list_get(list, 1);
        crate::list::nsl_list_free(list);
        let (o_ex, l_ex) = reference_forward(sh, &q, &k, &v, scale as f64, |x| x as f64);
        let (eo, _) = max_err(&tensor_values(out, o_ex.len()), &o_ex);
        let (el, _) = max_err(&tensor_values(lse, l_ex.len()), &l_ex);
        eprintln!("FFI forward: O {eo:.2e}, lse {el:.2e}");
        assert!(eo < 4e-3 && el < 4e-3, "FFI forward: O {eo}, lse {el}");

        let exact = reference_backward(sh, &q, &k, &v, &dout, scale as f64, |x| x as f64, |x| x as f64);
        for (label, lse_arg) in [("saved lse", lse), ("recomputed lse", 0)] {
            let grads = crate::flash_attention::nsl_flash_attention_backward(
                dview, qt, kt, vt, out, lse_arg, scale.to_bits() as i64, b, h, s, d, 1, 0, 0, 0, 0, 0, 0, 0,
            );
            assert_ne!(grads, 0);
            let parts: Vec<i64> = (0..3).map(|i| crate::list::nsl_list_get(grads, i)).collect();
            crate::list::nsl_list_free(grads);
            for (i, (name, want)) in [("dQ", &exact.0), ("dK", &exact.1), ("dV", &exact.2)].into_iter().enumerate() {
                let t = crate::tensor::NslTensor::from_ptr(parts[i]);
                let heads = unsafe { *t.shape.add(1) };
                assert_eq!(heads, if i == 0 { h } else { kvh }, "{label} {name}: wrong head count");
                assert_eq!(t.device, 1, "{label} {name} is not on the GPU (CPU fallback ran)");
                let (fe, me) = rel_err(&tensor_values(parts[i], want.len()), want, 1e-6);
                eprintln!("FFI backward ({label}) {name}: frob {fe:.2e} max {me:.2e}");
                assert!(fe < 5e-3 && me < 2e-2, "FFI backward ({label}) {name}: frob {fe}, max {me}");
                crate::tensor::nsl_tensor_free(parts[i]);
            }
        }
        for t in [qt, kt, vt, dview, dbuf, out, lse] {
            crate::tensor::nsl_tensor_free(t);
        }
    }

    /// Kernel time per call at the production recipes' attention shapes
    /// (Coder-1B: batch 2, 32 heads, seq 2048, head_dim 64; Coder-500M:
    /// batch 2, 20 heads, seq 1024), K/V expanded to every head as the
    /// stdlib passes them and grouped (8 kv heads), causal, over several
    /// tilings of each kernel (the defaults are marked): the measurement the
    /// defaults in `FwdConfig::new` / `BwdConfig::new` come from. Prints;
    /// asserts nothing but successful launches. FLOPs: the forward's two
    /// products; dQ recomputes S and dP and forms dQ (1.5x); dK/dV forms
    /// S, dV, dP and dK (2x).
    #[test]
    #[ignore = "diagnostic: times the tensor-core attention kernels at the production shapes"]
    fn tc_attention_kernel_times() {
        if !gpu() {
            return;
        }
        let shapes = [
            ("1B expanded", Shape { b: 2, h: 32, kvh: 32, s: 2048, d: 64, causal: true }),
            ("1B gqa", Shape { b: 2, h: 32, kvh: 8, s: 2048, d: 64, causal: true }),
            ("500M expanded", Shape { b: 2, h: 20, kvh: 20, s: 1024, d: 64, causal: true }),
            ("hd128", Shape { b: 2, h: 16, kvh: 16, s: 2048, d: 128, causal: true }),
        ];
        let mut warm = true;
        for (label, sh) in shapes {
            let n_q = sh.b * sh.h * sh.s * sh.d;
            let n_kv = sh.b * sh.kvh * sh.s * sh.d;
            let n_l = sh.b * sh.h * sh.s;
            let mut rng = Lcg(9);
            let (qd, kd, vd, dod) = (
                upload(&rng.vec(n_q, 1.0)),
                upload(&rng.vec(n_kv, 1.0)),
                upload(&rng.vec(n_kv, 1.0)),
                upload(&rng.vec(n_q, 1e-6)),
            );
            let (od, ld) = (poisoned(n_q), poisoned(n_l));
            let (dqd, dkd, dvd) = (poisoned(n_q), poisoned(n_kv), poisoned(n_kv));
            let (dvec, gmax) = (poisoned(n_l), poisoned(sh.b * sh.h));
            let fl = FwdLaunch {
                q: qd as u64,
                k: kd as u64,
                v: vd as u64,
                o: od as u64,
                lse: ld as u64,
                batch: sh.b,
                heads: sh.h,
                kv_heads: sh.kvh,
                seq: sh.s,
                head_dim: sh.d,
                scale: 0.125,
                causal: sh.causal,
            };
            let bl = BwdLaunch {
                q: qd as u64,
                k: kd as u64,
                v: vd as u64,
                o: od as u64,
                lse: ld as u64,
                dout: dod as u64,
                dq: dqd as u64,
                dk: dkd as u64,
                dv: dvd as u64,
                dvec: dvec as u64,
                gmax: gmax as u64,
                batch: sh.b,
                heads: sh.h,
                kv_heads: sh.kvh,
                seq: sh.s,
                head_dim: sh.d,
                scale: 0.125,
                causal: sh.causal,
            };
            let time = |f: &dyn Fn() -> CUresult| {
                for _ in 0..3 {
                    assert_eq!(f(), CUresult::CUDA_SUCCESS);
                }
                unsafe { cudarc::driver::sys::cuCtxSynchronize() };
                let n = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    assert_eq!(f(), CUresult::CUDA_SUCCESS);
                }
                assert_eq!(unsafe { cudarc::driver::sys::cuCtxSynchronize() }, CUresult::CUDA_SUCCESS);
                t0.elapsed().as_secs_f64() * 1e3 / n as f64
            };
            if warm {
                // Bring the clocks up (and JIT the default modules) before
                // the first timed configuration.
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_secs_f64() < 1.0 {
                    assert_eq!(forward(&fl), CUresult::CUDA_SUCCESS);
                    assert_eq!(backward(&bl), CUresult::CUDA_SUCCESS);
                    unsafe { cudarc::driver::sys::cuCtxSynchronize() };
                }
                warm = false;
            }
            let flops = 4.0 * (sh.b * sh.h) as f64 * (sh.s as f64).powi(2) * sh.d as f64 / 2.0;
            let hd = sh.d as u32;
            let tilings: &[(u32, u32)] = if hd > 64 {
                &[(8, 64), (8, 32), (4, 64), (4, 32), (8, 16), (4, 16)]
            } else {
                &[(8, 64), (4, 64), (8, 32), (4, 32), (8, 128), (4, 128)]
            };
            let def = FwdConfig::new(hd, sh.causal);
            for &(warps, block_n) in tilings {
                let cfg = FwdConfig { warps, block_n, ..def };
                let t = time(&|| forward_with(cfg, &fl));
                let mark = if cfg == def { " (default)" } else { "" };
                eprintln!("{label}: forward w{warps} n{block_n}: {t:.3} ms, {:.1} TFLOPS{mark}", flops / t / 1e9);
            }
            assert_eq!(forward(&fl), CUresult::CUDA_SUCCESS);
            let bdef = BwdConfig::new(hd, sh.causal);
            let t_pre = time(&|| backward_with(bdef, &bl, [true, false, false]));
            eprintln!("{label}: backward pre-pass {t_pre:.3} ms");
            for &(warps, block_n) in tilings {
                let cfg = BwdConfig { dq_warps: warps, dq_block_n: block_n, ..bdef };
                let t = time(&|| backward_with(cfg, &bl, [false, true, false]));
                let mark = if cfg == bdef { " (default)" } else { "" };
                eprintln!("{label}: dQ w{warps} n{block_n}: {t:.3} ms, {:.1} TFLOPS{mark}", 1.5 * flops / t / 1e9);
            }
            for &(warps, block_q) in tilings {
                let cfg = BwdConfig { dkdv_warps: warps, dkdv_block_q: block_q, ..bdef };
                if cfg.dkdv_smem_bytes() > nsl_kir::kernel_ir::SMEM_DYNAMIC_BUDGET {
                    continue;
                }
                let t = time(&|| backward_with(cfg, &bl, [false, false, true]));
                let mark = if cfg == bdef { " (default)" } else { "" };
                eprintln!("{label}: dK/dV w{warps} q{block_q}: {t:.3} ms, {:.1} TFLOPS{mark}", 2.0 * flops / t / 1e9);
            }
            let tb = time(&|| backward(&bl));
            eprintln!("{label}: whole backward (defaults) {tb:.3} ms");
            for p in [qd, kd, vd, dod, od, ld, dqd, dkd, dvd, dvec, gmax] {
                inner::free_managed(p);
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
    fn tc_forward_matches_f64_reference() {
        if !gpu() {
            return;
        }
        let cases = [
            // whole tiles, MHA, causal
            Shape { b: 2, h: 3, kvh: 3, s: 256, d: 64, causal: true },
            // GQA 4:1, ragged, causal
            Shape { b: 1, h: 4, kvh: 1, s: 200, d: 64, causal: true },
            // non-causal, ragged, hd128
            Shape { b: 2, h: 2, kvh: 2, s: 130, d: 128, causal: false },
            // GQA 2:1, hd128, causal, several q tiles
            Shape { b: 1, h: 4, kvh: 2, s: 384, d: 128, causal: true },
            // shorter than one tile; a single row
            Shape { b: 2, h: 2, kvh: 1, s: 37, d: 64, causal: false },
            Shape { b: 1, h: 2, kvh: 2, s: 1, d: 64, causal: true },
            Shape { b: 1, h: 1, kvh: 1, s: 1, d: 128, causal: false },
            // non-causal hd64 across tiles
            Shape { b: 1, h: 2, kvh: 2, s: 320, d: 64, causal: false },
        ];
        for (i, sh) in cases.into_iter().enumerate() {
            check_forward(sh, 1.0, 0x5eed + i as u64);
        }
        // Large scores: the running max moves by tens across tiles.
        check_forward(Shape { b: 1, h: 2, kvh: 1, s: 300, d: 64, causal: true }, 12.0, 77);
        check_forward(Shape { b: 1, h: 1, kvh: 1, s: 200, d: 128, causal: false }, 12.0, 78);
    }
}
