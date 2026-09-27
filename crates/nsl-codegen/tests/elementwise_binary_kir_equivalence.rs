//! The differential equivalence gate for the binary elementwise kernels
//! `nsl_add_f32`, `nsl_sub_f32` and `nsl_mul_f32` (roadmap A2 step 11), and
//! `nsl_div_f32` (new-roadmap item 5, once KIR gained `div.approx`).
//!
//! The runtime carried them as hand-written PTX
//! (`nsl_runtime::cuda::kernels::{ADD,SUB,MUL,DIV}_F32_PTX`); they are now built
//! as KIR by `nsl_kir::kernels::elementwise`. This file runs the frozen hand
//! modules (`tests/fixtures/elementwise_binary_hand.rs`) and the KIR ones
//! side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the grid the runtime's
//! `gpu_elementwise_binary` launches (`ceil(n / 256)` blocks of 256), plus
//! one block more:
//!
//! 1. **Agreement**: under two schedules, the hand and KIR kernels leave
//!    *the same bytes* in all of global memory, out of place and in place
//!    (`c` aliasing `a`, as `gpu_elementwise_binary_inplace` launches).
//! 2. **Correctness**: `c[i]` is the IEEE f32 `a[i] op b[i]` for every
//!    `i < n` (NaN, infinities, signed zeros, subnormals and overflow among
//!    the inputs), and nothing past `n` is written.
//! 3. **The spelling**: the interpreter models `div.approx.f32` as the IEEE
//!    quotient, so `div.approx` → `div.rn` is invisible to execution; on the
//!    machine the two differ by up to 2 ulp. The division's mnemonic is
//!    pinned against the hand module instead.
//! 4. **The gate bites**: relaxing the bound, nudging the element size,
//!    swapping the operands of the subtraction or the division, or reading
//!    the index from block 0 is caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{binary_ptx, BinaryOp, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/elementwise_binary_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const A: u64 = 0x1000_0000;
const B: u64 = 0x2000_0000;
const C: u64 = 0x3000_0000;
/// Elements of `c` past `n`, which no thread may write.
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;

fn kir_ptx(op: BinaryOp) -> String {
    String::from_utf8(binary_ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: BinaryOp) -> String {
    match op {
        BinaryOp::Add => hand::ADD_F32_PTX,
        BinaryOp::Sub => hand::SUB_F32_PTX,
        BinaryOp::Mul => hand::MUL_F32_PTX,
        BinaryOp::Div => hand::DIV_F32_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
}

fn apply(op: BinaryOp, x: f32, y: f32) -> f32 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        // The interpreter's `div.approx` is the IEEE quotient.
        BinaryOp::Div => x / y,
    }
}

/// Bit patterns covering the IEEE corners, then deterministic values of
/// mixed sign and magnitude.
fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 12] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0001, // NaN with a payload
        0x0000_0001, // smallest subnormal
        0x807F_FFFF, // largest negative subnormal
        0x7F7F_FFFF, // f32::MAX
        0xFF7F_FFFF, // f32::MIN
        0x3F80_0000, // 1
        0x0080_0000, // smallest normal
        0x5F00_0000, // 2^63
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
            let scale = [1e-30f32, 1e-3, 1.0, 1e3, 1e30][((s >> 20) % 5) as usize];
            (v * scale).to_bits()
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Launch `ptx` on `n` elements over `ceil(n / 256) + 1` blocks. In place,
/// `c` is `a`'s buffer. Returns all of global memory.
fn run(ptx: &str, n: usize, a: &[u32], b: &[u32], in_place: bool, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut a_bytes = le32(a);
    a_bytes.extend(le32(&vec![POISON; TAIL]));
    let mut global = vec![Segment { base: A, bytes: a_bytes }, Segment { base: B, bytes: le32(b) }];
    if !in_place {
        global.push(Segment { base: C, bytes: le32(&vec![POISON; n + TAIL]) });
    }
    let args: HashMap<String, u64> = [("a", A), ("b", B), ("c", if in_place { A } else { C }), ("n", n as u64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let grid = n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
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
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

const SIZES: [usize; 5] = [1, 255, 256, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for op in BinaryOp::ALL {
        for (k, &n) in SIZES.iter().enumerate() {
            let (a, b) = (values(n, k as u64 + 1), values(n, k as u64 + 50));
            for in_place in [false, true] {
                for order in ORDERS {
                    let hand = run(&hand_ptx(op), n, &a, &b, in_place, order);
                    let kir = run(&kir_ptx(op), n, &a, &b, in_place, order);
                    assert!(hand == kir, "{op:?} n={n} in_place={in_place} {order:?}: global memory differs");
                }
            }
        }
    }
}

/// Same bits, treating any NaN as any NaN: the payload a NaN operation
/// produces is the host's here, not the GPU's, so only NaN-ness is judged.
fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_answer_is_the_ieee_operation() {
    for op in BinaryOp::ALL {
        for (k, &n) in SIZES.iter().enumerate() {
            let (a, b) = (values(n, k as u64 + 1), values(n, k as u64 + 50));
            for in_place in [false, true] {
                let mem = run(&kir_ptx(op), n, &a, &b, in_place, Order::Ascending);
                let c = words(if in_place { &mem[0] } else { &mem[2] });
                for i in 0..n {
                    let want = apply(op, f32::from_bits(a[i]), f32::from_bits(b[i])).to_bits();
                    assert!(same(c[i], want), "{op:?} n={n} i={i}: {:#010x} vs {want:#010x}", c[i]);
                }
                assert!(c[n..].iter().all(|&w| w == POISON), "{op:?} n={n}: wrote past n");
                assert_eq!(words(&mem[1]), b, "{op:?}: b is untouched");
                if !in_place {
                    assert_eq!(words(&mem[0])[..n], a[..], "{op:?}: a is untouched");
                }
            }
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signature_and_entry_names() {
    for op in BinaryOp::ALL {
        let (h, k) = (hand_ptx(op), kir_ptx(op));
        assert_eq!(parse_signature(&k), parse_signature(&h), "{op:?}");
        assert!(k.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
        assert!(h.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` of `op` is told apart from the hand kernel on a
/// ragged size, out of place, under either schedule (or faults).
fn caught(op: BinaryOp, mutate: impl Fn(&str) -> String) -> bool {
    let n = 257;
    let (a, b) = (values(n, 3), values(n, 70));
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    ORDERS.into_iter().any(|order| {
        let expect = run(&hand, n, &a, &b, false, order);
        let (a, b) = (&a, &b);
        match std::panic::catch_unwind(|| run(&mutant, n, a, b, false, order)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// The `i`th line containing `from`, rewritten to `to`.
fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    let at = ptx.lines().enumerate().filter(|(_, l)| l.contains(from)).nth(i).expect("the line").0;
    ptx.lines()
        .enumerate()
        .map(|(k, l)| if k == at { l.replacen(from, to, 1) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in BinaryOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

#[test]
fn relaxing_the_bound_is_caught() {
    for op in BinaryOp::ALL {
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{op:?}");
    }
}

/// f32's size in each of the three addresses.
#[test]
fn nudging_the_element_size_is_caught() {
    for op in BinaryOp::ALL {
        assert_eq!(kir_ptx(op).matches(", 4;").count(), 3, "{op:?}");
        for i in 0..3 {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?}: address {i}");
        }
    }
}

/// `p` with the source operands of every `mnemonic` line swapped.
fn swap_operands(p: &str, mnemonic: &str) -> String {
    p.lines()
        .map(|l| match l.trim_start().strip_prefix(mnemonic) {
            Some(ops) => {
                let v: Vec<&str> = ops.trim_end_matches(';').split(", ").collect();
                format!("    {mnemonic}{}, {}, {};", v[0], v[2], v[1])
            }
            None => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[test]
fn swapping_the_operands_of_the_subtraction_or_the_division_is_caught() {
    assert!(caught(BinaryOp::Sub, |p| swap_operands(p, "sub.f32 ")));
    assert!(caught(BinaryOp::Div, |p| swap_operands(p, "div.approx.f32 ")));
}

/// The division is `div.approx.f32` in both modules, once, and never the
/// IEEE `div.rn.f32`; the interpreter cannot tell the two apart, which is
/// why the spelling is pinned.
#[test]
fn the_division_is_spelled_div_approx_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(BinaryOp::Div), kir_ptx(BinaryOp::Div));
    for p in [&h, &k] {
        assert_eq!(p.matches("div.approx.f32 ").count(), 1);
        assert!(!p.contains("div.rn") && !p.contains("div.full") && !p.contains("rcp."));
    }
    assert!(!caught(BinaryOp::Div, |p| p.replacen("div.approx.f32 ", "div.rn.f32 ", 1)), "a named equivalent mutant");
    assert!(caught(BinaryOp::Div, |p| p.replacen("div.approx.f32 ", "mul.f32 ", 1)), "the operation");
}

#[test]
fn ignoring_the_block_index_is_caught() {
    for op in BinaryOp::ALL {
        assert_eq!(kir_ptx(op).matches("%ctaid.x;").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "{op:?}");
    }
}
