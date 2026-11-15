//! The bf16-first-moment AdamW kernels (`first_moment_dtype="bf16"`) against
//! an f64 reference of one AdamW step.
//!
//! `nsl_fase_fused_adamw_multi_f32_mbf16` and its per-parameter twin
//! `nsl_fase_fused_adamw_step_f32_mbf16` (built by `nsl_kir::kernels::optim`)
//! run on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`)
//! over the flat grid the runtime launches, for a ragged parameter list with
//! an empty member, at realistic magnitudes: parameters of order 0.05,
//! gradients from 1e-3 down to 1e-6, second moments the squares of those.
//!
//! The reference is one AdamW step in f64 from the same inputs (the f32
//! scalars widened, `m` the widened bf16 value). What the kernel may differ by
//! is derived, not tuned:
//!
//! * the stored `m` is the f32 `m'` rounded ONCE to nearest even in bf16
//!   (8 significant bits): at most 2^-8 of `|m'|`, plus the f32 arithmetic
//!   that produced `m'` (four roundings of 2^-24);
//! * `θ'` is computed from the f32 `m'`, not the rounded one, so it carries
//!   only f32 arithmetic error (each op 2^-24 relative, `div.approx` 2 ulp),
//!   propagated to first order and doubled;
//! * `v'` is the f32 kernel's.
//!
//! Each was run against a deliberately broken kernel: storing `m` truncated
//! instead of rounded fails the `m` bound, and feeding the update the rounded
//! `m` fails the `θ` bound.

use std::collections::HashMap;

use half::bf16;
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{fase_adamw_multi_mbf16_ptx, fase_adamw_step_mbf16_ptx};

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

/// Ragged around the block size, with an empty member that must get no blocks.
const LENS: [usize; 5] = [257, 1, 0, 1000, 255];
const TAIL: usize = 64;
/// Past each f32 buffer's end (finite, so a stray thread would move it).
const F32_TAIL: u32 = 0xC040_0000; // -3.0
/// Past each bf16 first moment's end.
const M_TAIL: u16 = 0x3FC0; // 1.5

const TTAB: u64 = 0x0100_0000;
const MTAB: u64 = 0x0110_0000;
const VTAB: u64 = 0x0120_0000;
const MPTAB: u64 = 0x0130_0000;
const NTAB: u64 = 0x0140_0000;
const BPTAB: u64 = 0x0150_0000;
const BBTAB: u64 = 0x0160_0000;
const DATA: u64 = 0x1000_0000;
const TABLES: usize = 7;

const U: f64 = 1.0 / (1u64 << 24) as f64; // f32 unit roundoff
const UB: f64 = 1.0 / 256.0; // bf16 unit roundoff (8 significant bits)

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s
}

/// Uniform in (-1, 1).
fn unit(s: &mut u64) -> f32 {
    ((lcg(s) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

struct Param {
    theta: Vec<f32>,
    m: Vec<u16>, // bf16 bits
    v: Vec<f32>,
    mp: Vec<f32>,
}

/// Realistic magnitudes: θ ~ 0.05, gradients 1e-3 .. 1e-6, m a past gradient
/// average, v a past squared-gradient average.
fn params(seed: u64) -> Vec<Param> {
    LENS.iter()
        .enumerate()
        .map(|(j, &n)| {
            let mut s = seed + 1000 * j as u64;
            let scale = |s: &mut u64| [1e-3f32, 1e-4, 1e-5, 1e-6][(lcg(s) >> 33) as usize % 4];
            let theta = (0..n).map(|_| 0.05 * unit(&mut s)).collect();
            let m = (0..n)
                .map(|_| {
                    let sc = scale(&mut s);
                    bf16::from_f32(sc * unit(&mut s)).to_bits()
                })
                .collect();
            let v = (0..n)
                .map(|_| {
                    let sc = scale(&mut s);
                    let x = sc * unit(&mut s);
                    x * x
                })
                .collect();
            let mp = (0..n)
                .map(|_| {
                    let sc = scale(&mut s);
                    sc * unit(&mut s)
                })
                .collect();
            Param { theta, m, v, mp }
        })
        .collect()
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn halves(b: &[u8]) -> Vec<u16> {
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = le32(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    b.extend(le32(&[F32_TAIL; TAIL]));
    b
}
fn m_bytes(m: &[u16]) -> Vec<u8> {
    let mut b = le16(m);
    b.extend(le16(&[M_TAIL; TAIL]));
    b
}

/// `fase_step::build_block_tables`, restated.
fn block_tables(lens: &[usize], block: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut bparam, mut bbase) = (vec![], vec![]);
    for (j, &n) in lens.iter().enumerate() {
        for b in 0..n.div_ceil(block) {
            bparam.push(j as u32);
            bbase.push((b * block) as u32);
        }
    }
    (bparam, bbase)
}

#[derive(Debug, Clone, Copy)]
struct Hyper {
    b1: f32,
    omb1: f32,
    b2: f32,
    omb2: f32,
    eps: f32,
    neg_lr: f32,
    neg_lr_wd: f32,
    bc1: f32,
    bc2: f32,
    has_wd: u32,
    mp_scale: f32,
}

impl Hyper {
    fn named(&self) -> Vec<(&'static str, u64)> {
        let f = |x: f32| x.to_bits() as u64;
        vec![
            ("b1", f(self.b1)),
            ("omb1", f(self.omb1)),
            ("b2", f(self.b2)),
            ("omb2", f(self.omb2)),
            ("eps", f(self.eps)),
            ("neg_lr", f(self.neg_lr)),
            ("neg_lr_wd", f(self.neg_lr_wd)),
            ("bc1", f(self.bc1)),
            ("bc2", f(self.bc2)),
            ("has_wd", self.has_wd as u64),
            ("mp_scale", f(self.mp_scale)),
        ]
    }
}

/// The recipes' AdamW (β 0.9 / 0.95, lr 3e-4, decay 0.1) early and late in
/// training, an Adam step without decay, and a clip-scaled gradient.
fn hypers() -> Vec<Hyper> {
    let adamw = |t: i32, has_wd: u32, mp_scale: f32| {
        let (b1, b2, lr, wd) = (0.9f32, 0.95f32, 3e-4f32, 0.1f32);
        Hyper {
            b1,
            omb1: 1.0 - b1,
            b2,
            omb2: 1.0 - b2,
            eps: 1e-8,
            neg_lr: -lr,
            neg_lr_wd: -lr * wd,
            bc1: 1.0 / (1.0 - b1.powi(t)),
            bc2: 1.0 / (1.0 - b2.powi(t)),
            has_wd,
            mp_scale,
        }
    };
    vec![adamw(10, 1, 1.0), adamw(2000, 1, 1.0), adamw(1, 0, 1.0), adamw(50, 1, 0.37)]
}

fn buf(j: usize, k: usize) -> u64 {
    DATA + 0x100_0000 * (4 * j + k) as u64
}

fn run_ctas(prog: &Program, args: &HashMap<String, u64>, global: &mut Vec<Segment>, grid: u32) {
    for cta in 0..grid {
        let mut launch = Launch {
            prog,
            args,
            global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, Order::Ascending);
    }
}

/// The multi kernel over every parameter. Returns global memory in segment
/// order: the seven tables, then θ, m, v, mp of each parameter.
fn run_multi(ps: &[Param], h: &Hyper) -> Vec<Vec<u8>> {
    let prog = parse(&trim(&String::from_utf8(fase_adamw_multi_mbf16_ptx()).expect("ASCII")));
    let (bparam, bbase) = block_tables(&LENS, ELEMENTWISE_BLOCK as usize);
    let table = |k: usize| le64(&(0..LENS.len()).map(|j| buf(j, k)).collect::<Vec<_>>());
    let mut global = vec![
        Segment { base: TTAB, bytes: table(0) },
        Segment { base: MTAB, bytes: table(1) },
        Segment { base: VTAB, bytes: table(2) },
        Segment { base: MPTAB, bytes: table(3) },
        Segment { base: NTAB, bytes: le32(&LENS.map(|n| n as u32)) },
        Segment { base: BPTAB, bytes: le32(&bparam) },
        Segment { base: BBTAB, bytes: le32(&bbase) },
    ];
    for (j, p) in ps.iter().enumerate() {
        global.push(Segment { base: buf(j, 0), bytes: f32_bytes(&p.theta) });
        global.push(Segment { base: buf(j, 1), bytes: m_bytes(&p.m) });
        global.push(Segment { base: buf(j, 2), bytes: f32_bytes(&p.v) });
        global.push(Segment { base: buf(j, 3), bytes: f32_bytes(&p.mp) });
    }
    let mut args: HashMap<String, u64> =
        [("ttab", TTAB), ("mtab", MTAB), ("vtab", VTAB), ("mptab", MPTAB), ("ntab", NTAB), ("bptab", BPTAB), ("bbtab", BBTAB)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    run_ctas(&prog, &args, &mut global, bparam.len() as u32);
    global.into_iter().map(|s| s.bytes).collect()
}

/// The per-parameter kernel on parameter `j` alone. Its gradient is the
/// clip-scaled one the multi kernel reads (`rn(mp * mp_scale)`, which the
/// per-parameter loop applies in place before the step). Returns `[θ, m, v]`.
fn run_single(p: &Param, h: &Hyper, j: usize) -> Vec<Vec<u8>> {
    let prog = parse(&trim(&String::from_utf8(fase_adamw_step_mbf16_ptx()).expect("ASCII")));
    let mp: Vec<f32> = if h.mp_scale == 1.0 { p.mp.clone() } else { p.mp.iter().map(|g| g * h.mp_scale).collect() };
    let mut global = vec![
        Segment { base: buf(j, 0), bytes: f32_bytes(&p.theta) },
        Segment { base: buf(j, 1), bytes: m_bytes(&p.m) },
        Segment { base: buf(j, 2), bytes: f32_bytes(&p.v) },
        Segment { base: buf(j, 3), bytes: f32_bytes(&mp) },
    ];
    let n = p.theta.len();
    let mut args: HashMap<String, u64> =
        [("theta", buf(j, 0)), ("m", buf(j, 1)), ("v", buf(j, 2)), ("mp", buf(j, 3)), ("n", n as u64)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    run_ctas(&prog, &args, &mut global, n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32);
    global.into_iter().take(3).map(|s| s.bytes).collect()
}

/// One element's f64 reference and the bounds the kernel must meet.
struct Expect {
    m: f64,
    m_bound: f64,
    v: f64,
    v_bound: f64,
    theta: f64,
    theta_bound: f64,
}

fn expect(theta0: f32, m0_bits: u16, v0: f32, mp: f32, h: &Hyper) -> Expect {
    let (b1, omb1, b2, omb2) = (h.b1 as f64, h.omb1 as f64, h.b2 as f64, h.omb2 as f64);
    let (eps, neg_lr, neg_lr_wd) = (h.eps as f64, h.neg_lr as f64, h.neg_lr_wd as f64);
    let (bc1, bc2) = (h.bc1 as f64, h.bc2 as f64);
    let th0 = theta0 as f64;
    let m0 = bf16::from_bits(m0_bits).to_f64();
    let v0 = v0 as f64;
    let g = mp as f64 * h.mp_scale as f64;
    let scaled = h.mp_scale != 1.0;

    // m' = β1 m + (1-β1) g: two products, a sum, and (clip) the scaled read.
    let m = b1 * m0 + omb1 * g;
    let e_m = (3.0 + f64::from(u8::from(scaled))) * U * (b1 * m0.abs() + omb1 * g.abs());
    // One round to nearest even into 8 significant bits; the absolute term
    // covers bf16 subnormals (spacing 2^-133).
    let m_bound = UB * (m.abs() + e_m) + e_m + 2f64.powi(-134);

    // v' = β2 v + (1-β2) g²
    let v = b2 * v0 + omb2 * g * g;
    let e_v = (4.0 + 2.0 * f64::from(u8::from(scaled))) * U * (b2 * v0 + omb2 * g * g);

    // θ' from the f32 m' (NOT the rounded one), first-order propagation of
    // every f32 rounding, div.approx at 2 ulp, then doubled.
    let mh = m * bc1;
    let e_mh = e_m * bc1 + U * mh.abs();
    let vh = v * bc2;
    let e_vh = e_v * bc2 + U * vh;
    let sq = vh.sqrt();
    let e_sq = if vh > 0.0 { e_vh / (2.0 * sq) } else { e_vh.sqrt() } + U * sq;
    let den = sq + eps;
    let e_den = e_sq + U * den;
    let q = mh / den;
    let e_q = (e_mh + q.abs() * e_den) / den + 2.0 * U * q.abs();
    let a = neg_lr * q;
    let mut e_a = neg_lr.abs() * e_q + U * a.abs();
    let mut adj = a;
    if h.has_wd != 0 {
        let d = neg_lr_wd * th0;
        adj = a + d;
        e_a += U * d.abs() + U * adj.abs();
    }
    let theta = th0 + adj;
    let theta_bound = 2.0 * (e_a + U * theta.abs());
    Expect { m, m_bound, v, v_bound: 2.0 * e_v, theta, theta_bound }
}

#[test]
fn one_step_matches_the_f64_reference_within_one_bf16_rounding_of_m() {
    let ps = params(7);
    let total: usize = LENS.iter().sum();
    for h in hypers() {
        let out = run_multi(&ps, &h);
        let (mut worst_m, mut worst_t) = (0.0f64, 0.0f64);
        for (j, p) in ps.iter().enumerate() {
            let n = LENS[j];
            let seg = |k: usize| &out[TABLES + 4 * j + k];
            let theta = floats(seg(0));
            let m = halves(seg(1));
            let v = floats(seg(2));
            let mp = floats(seg(3));
            for e in 0..n {
                let x = expect(p.theta[e], p.m[e], p.v[e], p.mp[e], &h);
                let m_got = bf16::from_bits(m[e]).to_f64();
                let dm = (m_got - x.m).abs();
                assert!(
                    dm <= x.m_bound,
                    "{h:?} param {j} elem {e}: m {m_got:e} vs {:e} (|d| {dm:e} > {:e})",
                    x.m,
                    x.m_bound
                );
                let dv = (v[e] as f64 - x.v).abs();
                assert!(dv <= x.v_bound, "{h:?} param {j} elem {e}: v {} vs {:e} ({dv:e} > {:e})", v[e], x.v, x.v_bound);
                let dt = (theta[e] as f64 - x.theta).abs();
                assert!(
                    dt <= x.theta_bound,
                    "{h:?} param {j} elem {e}: theta {} vs {:e} (|d| {dt:e} > {:e})",
                    theta[e],
                    x.theta,
                    x.theta_bound
                );
                assert_eq!(mp[e].to_bits(), 0, "the accumulated gradient is consumed");
                if x.m != 0.0 {
                    worst_m = worst_m.max(dm / x.m.abs());
                }
                worst_t = worst_t.max(dt / x.theta_bound.max(f64::MIN_POSITIVE));
            }
            // Nothing past a parameter's end moves.
            assert!(theta[n..].iter().chain(&v[n..]).chain(&mp[n..]).all(|x| x.to_bits() == F32_TAIL), "param {j} f32 tail moved");
            assert!(m[n..].iter().all(|&x| x == M_TAIL), "param {j} m tail moved");
        }
        // The bound is not slack: m really carries one bf16 rounding (some
        // element is off by more than a quarter of the bf16 unit), so a kernel
        // that stored m in f32 would not be what this test measured.
        assert!(worst_m > UB / 4.0, "{h:?}: worst relative m error {worst_m:e} — m does not look bf16-rounded");
        eprintln!("{h:?}: {total} elements, worst m rel error {worst_m:.3e} (bound ~{UB:.3e}), worst theta |d|/bound {worst_t:.3}");
    }
}

/// The batching entry's contract: each parameter comes out of the multi
/// kernel byte-identical to the per-parameter kernel launched on it alone
/// (the runtime's sequential fallback arm).
#[test]
fn the_multi_kernel_is_the_per_parameter_kernel_batched() {
    let ps = params(11);
    for h in hypers() {
        let out = run_multi(&ps, &h);
        for (j, p) in ps.iter().enumerate() {
            if LENS[j] == 0 {
                continue;
            }
            let single = run_single(p, &h, j);
            for (k, name) in [(0, "theta"), (1, "m"), (2, "v")] {
                assert_eq!(out[TABLES + 4 * j + k], single[k], "{h:?} param {j}: {name} differs");
            }
        }
    }
}
