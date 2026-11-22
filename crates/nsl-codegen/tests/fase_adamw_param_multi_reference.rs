//! The bf16-parameter members of the multi-tensor AdamW family
//! (`nsl_kir::kernels::optim::build_param_multi`), held bit for bit to the
//! f32 kernel they decompose into, on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the flat grid the runtime
//! launches, for a ragged parameter list with an empty member.
//!
//! Each variant computes the f32 kernel's update of an f32 θ, so for the
//! same m, v, gradient, scalars and clip scale:
//!
//! - **master** (`Bf16Master`): the f32 master after the step equals the
//!   f32 kernel's θ', and θ is that θ''s image (RNE, or the split's rounding
//!   for the test-only `HalfTowardTruncation`);
//! - **split** (`Bf16Split`): the stored halves are `split_hi(θ')` and
//!   `split_lo(θ')` of the f32 kernel's θ' run on `join(hi, lo)`, so the
//!   reassembly is exact (every non-NaN pattern; ties are planted, where the
//!   split's image and RNE disagree);
//! - **SR** (`Bf16Sr`): θ' is the host SR rounding of the f32 kernel's θ' run
//!   on `widen(θ)` with the dither of `(sr_key, ctrtab[p] + e)`;
//!
//! and m, v and the zeroed gradient are the f32 kernel's, bytes past each
//! parameter's length untouched, in both block orders. `mp_scale` 1.0 (the
//! multiply branched around) and 0.37 (the clip fold), m f32 and bf16.
//! Mutations this catches, run once each before the commit: the SR tail
//! replaced by `cvt.rn` (M1), the `mp_scale` fold dropped (M2), `mp` not
//! zeroed (M3), the split's carry test `>=` for `>` and its image RNE.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{
    Image, SR_SPLITMIX_GAMMA, Theta, join_f32, multi_name, param_multi_ptx, split_hi, split_lo,
};

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

/// Parameter lengths: ragged around the block size, one empty member.
const LENS: [usize; 5] = [257, 1, 0, 1000, 255];
/// Parameter 3's elements `TIES..TIES + 64` hold planted ties (`lo =
/// 0x8000`, odd and even high halves) with zero moments and gradient, so a
/// step without weight decay leaves them where they are.
const TIES: usize = 900;
const TAIL: usize = 64;
const IN_TAIL: u32 = 0xC040_0000; // -3.0
const HALF_TAIL: u16 = 0x3FC0; // 1.5

const TTAB: u64 = 0x0100_0000;
const WTAB: u64 = 0x0110_0000;
const MTAB: u64 = 0x0120_0000;
const VTAB: u64 = 0x0130_0000;
const MPTAB: u64 = 0x0140_0000;
const NTAB: u64 = 0x0150_0000;
const BPTAB: u64 = 0x0160_0000;
const BBTAB: u64 = 0x0170_0000;
const CTRTAB: u64 = 0x0180_0000;
const DATA: u64 = 0x1000_0000;

fn trim(s: Vec<u8>) -> String {
    String::from_utf8(s)
        .expect("ASCII")
        .trim_end_matches('\0')
        .to_string()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *s
}

const F32_CORNERS: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0001,
    0x0000_0001,
    0x807F_FFFF,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x3F80_0000,
    0x0080_0000,
    0x5F00_0000,
];

/// f32 words: a corner every seventh element, otherwise magnitudes from
/// 1e-9 to 1e1 (gradients and moments at training scale) and 1e-30 / 1e30.
fn f32_words(n: usize, s: &mut u64, nonneg: bool) -> Vec<u32> {
    (0..n)
        .map(|k| {
            let bits = if k % 7 == 3 {
                F32_CORNERS[(k / 7) % F32_CORNERS.len()]
            } else {
                let x = ((lcg(s) >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
                let scale =
                    [1e-30f32, 1e-9, 1e-6, 1e-3, 1e-2, 1.0, 1e1, 1e30][((*s >> 20) % 8) as usize];
                (x * scale).to_bits()
            };
            if nonneg { bits & 0x7FFF_FFFF } else { bits }
        })
        .collect()
}

/// One parameter's inputs. `theta` is the f32 value the update starts from
/// (the master, or the split reassembled, or the bf16 θ widened).
#[derive(Clone)]
struct Param {
    theta: Vec<u32>,
    m: Vec<u32>,
    v: Vec<u32>,
    mp: Vec<u32>,
}

fn params(seed: u64) -> Vec<Param> {
    LENS.iter()
        .enumerate()
        .map(|(j, &n)| {
            let mut s = seed + 100 * j as u64;
            // θ at a weight's scale, with corners.
            let theta = (0..n)
                .map(|k| {
                    if k % 11 == 5 {
                        F32_CORNERS[(k / 11) % F32_CORNERS.len()]
                    } else {
                        let x = ((lcg(&mut s) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                        (x * 0.05).to_bits()
                    }
                })
                .collect();
            let mut p = Param {
                theta,
                m: f32_words(n, &mut s, false),
                v: f32_words(n, &mut s, true),
                mp: f32_words(n, &mut s, false),
            };
            if j == 3 {
                for k in TIES..TIES + 64 {
                    let hi = 0x3C00u32 + k as u32 - TIES as u32; // odd and even truncations
                    p.theta[k] = (hi << 16) | 0x8000;
                    p.m[k] = 0;
                    p.v[k] = 0;
                    p.mp[k] = 0;
                }
            }
            p
        })
        .collect()
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
}

impl Hyper {
    fn named(&self) -> [(&'static str, u64); 10] {
        let f = |x: f32| x.to_bits() as u64;
        [
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
        ]
    }
}

/// AdamW at steps 1 and 1000 with and without decay (the 1B recipe's lr and
/// betas), and one with every scalar distinct so a swapped slot shows.
fn hypers() -> Vec<Hyper> {
    let adamw = |t: i32, has_wd: u32| {
        let (b1, b2) = (0.9f32, 0.95f32);
        Hyper {
            b1,
            omb1: 1.0 - b1,
            b2,
            omb2: 1.0 - b2,
            eps: 1e-8,
            neg_lr: -1.5e-5,
            neg_lr_wd: -1.5e-5 * 0.1,
            bc1: 1.0 / (1.0 - b1.powi(t)),
            bc2: 1.0 / (1.0 - b2.powi(t)),
            has_wd,
        }
    };
    vec![
        adamw(1, 0),
        adamw(1000, 1),
        Hyper {
            b1: 0.7,
            omb1: 0.2,
            b2: 0.95,
            omb2: 0.03,
            eps: 1e-3,
            neg_lr: -0.25,
            neg_lr_wd: -0.125,
            bc1: 1.75,
            bc2: 3.5,
            has_wd: 7,
        },
    ]
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
    b.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn with_tail16(v: &[u16]) -> Vec<u8> {
    let mut b = le16(v);
    b.extend(le16(&[HALF_TAIL; TAIL]));
    b
}
fn with_tail32(v: &[u32]) -> Vec<u8> {
    let mut b = le32(v);
    b.extend(le32(&[IN_TAIL; TAIL]));
    b
}
fn bf16_rn(bits: u32) -> u16 {
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040; // quiet NaN, as cvt.rn does in the interpreter
    }
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn block_tables(lens: &[usize], block: usize) -> (Vec<u32>, Vec<u32>) {
    let mut bparam = vec![];
    let mut bbase = vec![];
    for (j, &n) in lens.iter().enumerate() {
        for b in 0..n.div_ceil(block) {
            bparam.push(j as u32);
            bbase.push((b * block) as u32);
        }
    }
    (bparam, bbase)
}

/// The base address of buffer `k` (0 θ, 1 master/lo, 2 m, 3 v, 4 mp) of
/// parameter `j`.
fn buf(j: usize, k: usize) -> u64 {
    DATA + 0x100_0000 * (5 * j + k) as u64
}

/// The `(sr_key, ctrtab)` of the SR runs: a mixed key over the real
/// registration bases, and bases near `u64::MAX` (a counter wraps).
fn sr_keys() -> [(u64, [u64; 5]); 2] {
    [
        (
            0x9F3A_51C2_77E0_1D4B,
            [0, 1 << 40, 2 << 40, 3 << 40, 4 << 40],
        ),
        (u64::MAX, [u64::MAX - 7, 3, 0, u64::MAX - 100, 1 << 63]),
    ]
}

fn mix64(key: u64, counter: u64) -> u64 {
    let mut z = key.wrapping_add(counter.wrapping_mul(SR_SPLITMIX_GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn sr_round(bits: u32, dither: u16) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    if bits & 0x7F80_0000 == 0x7F80_0000 {
        return if bits & 0x007F_FFFF != 0 {
            sign | 0x7FC0
        } else {
            sign | 0x7F80
        };
    }
    let dithered = bits.wrapping_add(dither as u32);
    if dithered & 0x7F80_0000 == 0x7F80_0000 {
        return sign | 0x7F7F;
    }
    (dithered >> 16) as u16
}

/// What a launch leaves, per parameter: θ (f32 words for the f32 kernel,
/// halves otherwise), the master or low halves, m (words or halves), v, mp.
struct Out {
    theta32: Vec<Vec<u32>>,
    theta16: Vec<Vec<u16>>,
    aux32: Vec<Vec<u32>>,
    aux16: Vec<Vec<u16>>,
    m32: Vec<Vec<u32>>,
    m16: Vec<Vec<u16>>,
    v: Vec<Vec<u32>>,
    mp: Vec<Vec<u32>>,
}

/// Run `theta`'s member of the family (`m_bf16`) over `ps`: θ is laid out as
/// the variant stores it (f32, the master's image beside the f32 master, the
/// split halves, the bf16 θ for SR).
fn run(
    theta: Theta,
    m_bf16: bool,
    ps: &[Param],
    h: &Hyper,
    mp_scale: f32,
    sr: (u64, [u64; 5]),
    order: Order,
) -> Out {
    let prog = parse(&trim(param_multi_ptx(theta, m_bf16)));
    let (bparam, bbase) = block_tables(&LENS, ELEMENTWISE_BLOCK as usize);
    let table = |k: usize| le64(&(0..LENS.len()).map(|j| buf(j, k)).collect::<Vec<_>>());
    let mut global = vec![
        Segment {
            base: TTAB,
            bytes: table(0),
        },
        Segment {
            base: WTAB,
            bytes: table(1),
        },
        Segment {
            base: MTAB,
            bytes: table(2),
        },
        Segment {
            base: VTAB,
            bytes: table(3),
        },
        Segment {
            base: MPTAB,
            bytes: table(4),
        },
        Segment {
            base: NTAB,
            bytes: le32(&LENS.map(|n| n as u32)),
        },
        Segment {
            base: BPTAB,
            bytes: le32(&bparam),
        },
        Segment {
            base: BBTAB,
            bytes: le32(&bbase),
        },
        Segment {
            base: CTRTAB,
            bytes: le64(&sr.1),
        },
    ];
    let first = global.len();
    for (j, p) in ps.iter().enumerate() {
        let (t, w) = match theta {
            Theta::F32 => (with_tail32(&p.theta), vec![]),
            Theta::Bf16Master(_) => (
                with_tail16(&p.theta.iter().map(|&b| bf16_rn(b)).collect::<Vec<_>>()),
                with_tail32(&p.theta),
            ),
            Theta::Bf16Split => (
                with_tail16(&p.theta.iter().map(|&b| split_hi(b)).collect::<Vec<_>>()),
                with_tail16(&p.theta.iter().map(|&b| split_lo(b)).collect::<Vec<_>>()),
            ),
            Theta::Bf16Sr => (
                with_tail16(
                    &p.theta
                        .iter()
                        .map(|&b| (b >> 16) as u16)
                        .collect::<Vec<_>>(),
                ),
                vec![],
            ),
        };
        let m = if m_bf16 {
            with_tail16(&p.m.iter().map(|&b| bf16_rn(b)).collect::<Vec<_>>())
        } else {
            with_tail32(&p.m)
        };
        global.push(Segment {
            base: buf(j, 0),
            bytes: t,
        });
        global.push(Segment {
            base: buf(j, 1),
            bytes: w,
        });
        global.push(Segment {
            base: buf(j, 2),
            bytes: m,
        });
        global.push(Segment {
            base: buf(j, 3),
            bytes: with_tail32(&p.v),
        });
        global.push(Segment {
            base: buf(j, 4),
            bytes: with_tail32(&p.mp),
        });
    }
    let mut args: HashMap<String, u64> = [
        ("ttab", TTAB),
        ("wtab", WTAB),
        ("mtab", MTAB),
        ("vtab", VTAB),
        ("mptab", MPTAB),
        ("ntab", NTAB),
        ("bptab", BPTAB),
        ("bbtab", BBTAB),
        ("ctrtab", CTRTAB),
        ("sr_key", sr.0),
        ("mp_scale", mp_scale.to_bits() as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    let mut ctas: Vec<u32> = (0..bparam.len() as u32).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    let mut out = Out {
        theta32: vec![],
        theta16: vec![],
        aux32: vec![],
        aux16: vec![],
        m32: vec![],
        m16: vec![],
        v: vec![],
        mp: vec![],
    };
    for (j, &n) in LENS.iter().enumerate() {
        let seg = |k: usize| &global[first + 5 * j + k].bytes;
        // Bytes past the length are untouched.
        for k in 0..5 {
            let b = seg(k);
            if b.is_empty() {
                continue;
            }
            let elem = b.len() / (n + TAIL);
            let tail = &b[n * elem..];
            let want = if elem == 2 {
                le16(&[HALF_TAIL; TAIL])
            } else {
                le32(&[IN_TAIL; TAIL])
            };
            assert_eq!(
                tail,
                &want[..],
                "{theta:?}/{m_bf16}: param {j} buffer {k} written past its length"
            );
        }
        let cut16 = |b: &Vec<u8>| halves(&b[..2 * n]);
        let cut32 = |b: &Vec<u8>| words(&b[..4 * n]);
        match theta {
            Theta::F32 => out.theta32.push(cut32(seg(0))),
            _ => out.theta16.push(cut16(seg(0))),
        }
        match theta {
            Theta::Bf16Master(_) => out.aux32.push(cut32(seg(1))),
            Theta::Bf16Split => out.aux16.push(cut16(seg(1))),
            _ => {}
        }
        if m_bf16 {
            out.m16.push(cut16(seg(2)));
        } else {
            out.m32.push(cut32(seg(2)));
        }
        out.v.push(cut32(seg(3)));
        out.mp.push(cut32(seg(4)));
    }
    out
}

/// The f32 kernel's reference, on the f32 θ the variant reassembles.
fn reference(m_bf16: bool, ps: &[Param], h: &Hyper, mp_scale: f32) -> Out {
    run(
        Theta::F32,
        m_bf16,
        ps,
        h,
        mp_scale,
        (0, [0; 5]),
        Order::Ascending,
    )
}

/// The f32 θ each variant's update starts from: the master, the split
/// reassembled, or the bf16 θ widened (SR).
fn effective(theta: Theta, ps: &[Param]) -> Vec<Param> {
    ps.iter()
        .map(|p| {
            let t = match theta {
                Theta::F32 | Theta::Bf16Master(_) => p.theta.clone(),
                Theta::Bf16Split => p
                    .theta
                    .iter()
                    .map(|&b| join_f32(split_hi(b), split_lo(b)))
                    .collect(),
                Theta::Bf16Sr => p.theta.iter().map(|&b| b & 0xFFFF_0000).collect(),
            };
            Param {
                theta: t,
                ..p.clone()
            }
        })
        .collect()
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

fn check(
    theta: Theta,
    m_bf16: bool,
    ps: &[Param],
    h: &Hyper,
    mp_scale: f32,
    sr: (u64, [u64; 5]),
    order: Order,
) {
    let label = format!("{theta:?} m_bf16={m_bf16} {h:?} mp_scale={mp_scale} {order:?}");
    let eff = effective(theta, ps);
    let r = reference(m_bf16, &eff, h, mp_scale);
    let o = run(theta, m_bf16, ps, h, mp_scale, sr, order);
    for (j, &n) in LENS.iter().enumerate() {
        if m_bf16 {
            assert_eq!(o.m16[j], r.m16[j], "{label}: param {j} m");
        } else {
            assert_eq!(o.m32[j], r.m32[j], "{label}: param {j} m");
        }
        assert_eq!(o.v[j], r.v[j], "{label}: param {j} v");
        assert!(
            o.mp[j].iter().all(|&g| g == 0),
            "{label}: param {j} gradient not zeroed"
        );
        let want = &r.theta32[j];
        for e in 0..n {
            let x = want[e];
            match theta {
                Theta::Bf16Master(image) => {
                    assert_eq!(o.aux32[j][e], x, "{label}: param {j}[{e}] master");
                    let img = match image {
                        Image::Rne => bf16_rn(x),
                        Image::HalfTowardTruncation => split_hi(x),
                    };
                    let got = o.theta16[j][e];
                    if is_nan(x) {
                        assert!(
                            is_nan((got as u32) << 16),
                            "{label}: param {j}[{e}] image of NaN {got:#06x}"
                        );
                    } else {
                        assert_eq!(got, img, "{label}: param {j}[{e}] image of {x:#010x}");
                    }
                }
                Theta::Bf16Split => {
                    let (hi, lo) = (o.theta16[j][e], o.aux16[j][e]);
                    assert_eq!(
                        (hi, lo),
                        (split_hi(x), split_lo(x)),
                        "{label}: param {j}[{e}] split of {x:#010x}"
                    );
                    if !is_nan(x) {
                        assert_eq!(join_f32(hi, lo), x, "{label}: param {j}[{e}] reassembly");
                    }
                }
                Theta::Bf16Sr => {
                    let d = mix64(sr.0, sr.1[j].wrapping_add(e as u64)) as u16;
                    assert_eq!(
                        o.theta16[j][e],
                        sr_round(x, d),
                        "{label}: param {j}[{e}] SR of {x:#010x}"
                    );
                }
                Theta::F32 => unreachable!(),
            }
        }
    }
}

const VARIANTS: [Theta; 4] = [
    Theta::Bf16Master(Image::Rne),
    Theta::Bf16Master(Image::HalfTowardTruncation),
    Theta::Bf16Split,
    Theta::Bf16Sr,
];

#[test]
fn every_bf16_parameter_kernel_is_the_f32_step_of_its_exact_master() {
    let ps = params(1);
    for theta in VARIANTS {
        for m_bf16 in [false, true] {
            for (k, h) in hypers().iter().enumerate() {
                for mp_scale in [1.0f32, 0.37] {
                    let order = if k % 2 == 0 {
                        Order::Ascending
                    } else {
                        Order::Descending
                    };
                    check(theta, m_bf16, &ps, h, mp_scale, sr_keys()[k % 2], order);
                }
            }
        }
    }
}

/// The planted ties: without weight decay and with nothing to step, θ' is
/// θ, so the split must hand back each tie unchanged. RNE would carry an odd
/// truncation (and the reassembly would then be one bf16 ulp off).
#[test]
fn the_split_keeps_every_planted_tie_exact() {
    let ps = params(2);
    let h = hypers()[0];
    assert_eq!(h.has_wd, 0);
    let o = run(
        Theta::Bf16Split,
        false,
        &ps,
        &h,
        1.0,
        (0, [0; 5]),
        Order::Ascending,
    );
    let mut odd = 0;
    for k in TIES..TIES + 64 {
        let x = ps[3].theta[k];
        assert_eq!(join_f32(o.theta16[3][k], o.aux16[3][k]), x, "tie {x:#010x}");
        assert_eq!(
            o.theta16[3][k] as u32,
            x >> 16,
            "a tie keeps its truncation: {x:#010x}"
        );
        odd += usize::from(bf16_rn(x) as u32 != x >> 16);
    }
    assert_eq!(odd, 32, "half the planted ties are odd, where RNE carries");
}

/// The names the runtime launches, and each kernel's parameters in the
/// order `cuda/param_store_gpu.rs` passes them: the f32 kernel's, with `wtab`
/// after `ttab` (master, split) or `sr_key, ctrtab` after `has_wd` (SR).
#[test]
fn the_kernels_take_the_arguments_in_the_launchers_order() {
    let scalars = [
        "b1",
        "omb1",
        "b2",
        "omb2",
        "eps",
        "neg_lr",
        "neg_lr_wd",
        "bc1",
        "bc2",
        "has_wd",
    ];
    for theta in VARIANTS {
        let p = trim(param_multi_ptx(theta, false));
        let head = format!(".visible .entry {}(", multi_name(theta, false));
        let sig = p
            .split_once(&head)
            .and_then(|(_, r)| r.split_once(')'))
            .map(|(sig, _)| sig)
            .unwrap_or_else(|| panic!("{theta:?}: no entry\n{p}"));
        let got: Vec<&str> = sig
            .split(',')
            .map(|a| {
                a.trim()
                    .rsplit(' ')
                    .next()
                    .unwrap()
                    .trim_start_matches("param_")
            })
            .collect();
        let mut want: Vec<&str> = vec!["ttab"];
        if matches!(theta, Theta::Bf16Master(_) | Theta::Bf16Split) {
            want.push("wtab");
        }
        want.extend(["mtab", "vtab", "mptab", "ntab"]);
        want.extend(scalars);
        if theta == Theta::Bf16Sr {
            want.extend(["sr_key", "ctrtab"]);
        }
        want.extend(["bptab", "bbtab", "mp_scale"]);
        assert_eq!(got, want, "{theta:?}");
    }
}
