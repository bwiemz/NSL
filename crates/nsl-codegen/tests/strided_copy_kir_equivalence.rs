//! The differential equivalence gate for the strided run-copy kernels
//! (`nsl_scopy_*_f32`, roadmap A2 step 11).
//!
//! The runtime carried these four kernels as a hand-written PTX module,
//! `STRIDED_COPY_RUN_PTX`; they are now built as KIR by
//! `nsl_kir::kernels::strided_copy`. This file runs the frozen hand module
//! (`tests/fixtures/strided_copy_hand.rs`) and the KIR one side by side on
//! the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! over the whole two-dimensional grid `RunPlan::geometry` launches:
//!
//! 1. **Agreement**: under two schedules (threads and CTAs in ascending or
//!    descending order), the hand and KIR arms leave *the same bytes* in all
//!    of global memory.
//! 2. **Correctness**: those bytes are the copy: run `o` of `dst` holds
//!    `src[offsets[o] ..][.. run_len]` (or `src[offsets[o]]` splatted, for a
//!    broadcast arm), every bit of every f32 (NaN payloads and `-0.0`
//!    included), and nothing else of `dst` is written.
//! 3. **The gate bites**: relaxing either bound, dropping the store, nudging
//!    a baked constant, or ignoring either block index is caught.
//!
//! The cases cover a run shorter than its block (idle threads), a run
//! spanning two blocks with a ragged tail, fewer y blocks than runs (the
//! grid-stride walk), overlapping and out-of-order source runs, and each
//! arm.

use std::collections::HashMap;

use nsl_kir::kernels::strided_copy::{ptx as kir_module, ScopyArm};

#[allow(dead_code)]
#[path = "fixtures/strided_copy_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const SRC: u64 = 0x1000_0000;
const DST: u64 = 0x2000_0000;
const OFFSETS: u64 = 0x3000_0000;
/// What `dst` holds before the copy: a byte pattern no source value has.
const POISON: u8 = 0xA5;

#[derive(Debug, Clone)]
struct Case {
    arm: ScopyArm,
    run_len: u64,
    /// One source element offset per run; `outer` is its length.
    offsets: Vec<u64>,
    src_len: usize,
    /// `gridDim.y`; fewer than the runs makes each y block walk several.
    grid_y: u32,
}

impl Case {
    fn outer(&self) -> u64 {
        self.offsets.len() as u64
    }

    /// `RunPlan::geometry`'s block and `grid.x`.
    fn geometry(&self) -> (u32, u32) {
        let units = if self.arm.vec4() { self.run_len / 4 } else { self.run_len };
        let block = units.div_ceil(32).max(1).saturating_mul(32).clamp(32, 256);
        (block as u32, units.div_ceil(block) as u32)
    }
}

/// Deterministic f32 bit patterns, with a NaN carrying a payload and a
/// negative zero mixed in: the copy must move bits, not values.
fn source(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|k| match k % 11 {
            3 => 0x7FC0_1234 + k as u32,
            7 => 0x8000_0000,
            _ => {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                // Never the poison pattern.
                ((s >> 32) as u32) & 0x7F7F_FFFF
            }
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn memory(c: &Case, src: &[u32]) -> Vec<Segment> {
    vec![
        Segment { base: SRC, bytes: le32(src) },
        Segment { base: DST, bytes: vec![POISON; (c.outer() * c.run_len * 4) as usize] },
        Segment { base: OFFSETS, bytes: c.offsets.iter().flat_map(|o| o.to_le_bytes()).collect() },
    ]
}

/// The entry `name` of `module`: from its `.visible .entry` to the next one.
fn kernel_text(module: &str, name: &str) -> String {
    let module = module.trim_end_matches('\0');
    let start = module.find(&format!(".visible .entry {name}(")).unwrap_or_else(|| panic!("no entry {name}"));
    let end = module[start + 1..].find(".visible .entry ").map_or(module.len(), |e| start + 1 + e);
    module[start..end].to_string()
}

fn kir_ptx(arm: ScopyArm) -> String {
    let bytes = kir_module();
    kernel_text(std::str::from_utf8(&bytes).expect("ASCII"), arm.kernel_name())
}

fn hand_ptx(arm: ScopyArm) -> String {
    kernel_text(hand::STRIDED_COPY_RUN_PTX, arm.kernel_name())
}

/// Launch `ptx` over the case's whole grid, as the host does, and return
/// all of global memory.
fn run(ptx: &str, c: &Case, src: &[u32], order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut global = memory(c, src);
    let args: HashMap<String, u64> = [
        ("src", SRC),
        ("dst", DST),
        ("offsets", OFFSETS),
        ("run_len", c.run_len),
        ("outer", c.outer()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let (block, grid_x) = c.geometry();
    let mut ctas: Vec<(u32, u32)> =
        (0..c.grid_y).flat_map(|y| (0..grid_x).map(move |x| (x, y))).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for (x, y) in ctas {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: x,
            ctaid_y: y,
            nctaid_y: c.grid_y,
            ntid: block,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

/// What `dst` must hold after the copy.
fn reference(c: &Case, src: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &off in &c.offsets {
        for i in 0..c.run_len {
            let at = if c.arm.broadcast() { off } else { off + i };
            out.push(src[at as usize]);
        }
    }
    le32(&out)
}

fn cases() -> Vec<Case> {
    use ScopyArm::*;
    vec![
        // A run shorter than its 64-thread block; runs out of order and
        // overlapping, one read twice.
        Case { arm: Run, run_len: 37, offsets: vec![100, 0, 3, 100, 57], src_len: 200, grid_y: 5 },
        // Two x blocks with a ragged tail; two y blocks walking three runs.
        Case { arm: Run, run_len: 300, offsets: vec![0, 301, 650], src_len: 1000, grid_y: 2 },
        // Exactly one full block.
        Case { arm: Run, run_len: 256, offsets: vec![256, 0], src_len: 512, grid_y: 2 },
        // 16 float4 units in a 32-thread block; four y blocks, six runs.
        Case { arm: Run4, run_len: 64, offsets: vec![0, 128, 64, 8, 200, 4], src_len: 300, grid_y: 4 },
        // 260 units: two x blocks, the second nearly idle.
        Case { arm: Run4, run_len: 1040, offsets: vec![1040, 0], src_len: 2100, grid_y: 1 },
        Case { arm: Bcast, run_len: 33, offsets: vec![5, 0, 5, 9], src_len: 16, grid_y: 3 },
        Case { arm: Bcast, run_len: 290, offsets: vec![7, 1], src_len: 8, grid_y: 1 },
        Case { arm: Bcast4, run_len: 48, offsets: vec![1, 2, 3], src_len: 8, grid_y: 2 },
        Case { arm: Bcast4, run_len: 1028, offsets: vec![6, 0], src_len: 8, grid_y: 2 },
    ]
}

#[test]
fn every_arm_is_covered() {
    for arm in ScopyArm::ALL {
        assert!(cases().iter().any(|c| c.arm == arm), "{arm:?} has no case");
    }
    // The grid-stride walk and a second x block are both exercised.
    assert!(cases().iter().any(|c| (c.grid_y as u64) < c.outer()));
    assert!(cases().iter().any(|c| c.geometry().1 > 1));
}

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for (n, c) in cases().iter().enumerate() {
        let src = source(c.src_len, n as u64 + 1);
        for order in [Order::Ascending, Order::Descending] {
            let hand = run(&hand_ptx(c.arm), c, &src, order);
            let kir = run(&kir_ptx(c.arm), c, &src, order);
            assert!(hand == kir, "case {n} ({:?}) under {order:?}: global memory differs", c.arm);
        }
    }
}

#[test]
fn the_answer_is_the_copy() {
    for (n, c) in cases().iter().enumerate() {
        let src = source(c.src_len, n as u64 + 1);
        for order in [Order::Ascending, Order::Descending] {
            let mem = run(&kir_ptx(c.arm), c, &src, order);
            assert_eq!(mem[0], le32(&src), "case {n}: the source is untouched");
            assert!(mem[1] == reference(c, &src), "case {n} ({:?}) under {order:?}: dst is not the copy", c.arm);
            assert_eq!(
                mem[2],
                c.offsets.iter().flat_map(|o| o.to_le_bytes()).collect::<Vec<u8>>(),
                "case {n}: the offsets are untouched"
            );
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for arm in ScopyArm::ALL {
        assert_eq!(parse_signature(&kir_ptx(arm)), parse_signature(&hand_ptx(arm)), "{arm:?}");
    }
    // One module holds all four, as the runtime loads it.
    let bytes = kir_module();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_eq!(text.matches(".visible .entry ").count(), 4);
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Per arm, a case with idle threads in its last x block and fewer y
/// blocks than runs, so every bound and both block indices matter.
fn mutation_case(arm: ScopyArm) -> Case {
    use ScopyArm::*;
    match arm {
        Run => Case { arm, run_len: 300, offsets: vec![0, 301, 650], src_len: 1000, grid_y: 2 },
        Run4 => Case { arm, run_len: 1040, offsets: vec![1040, 0, 8], src_len: 2100, grid_y: 2 },
        Bcast => Case { arm, run_len: 290, offsets: vec![7, 1, 4], src_len: 8, grid_y: 2 },
        Bcast4 => Case { arm, run_len: 1028, offsets: vec![6, 0, 3], src_len: 8, grid_y: 2 },
    }
}

/// Whether `mutate(kir)` of `arm` is told apart from the hand kernel:
/// under either schedule its memory differs, or the interpreter faults.
fn caught(arm: ScopyArm, mutate: impl Fn(&str) -> String) -> bool {
    let c = mutation_case(arm);
    let src = source(c.src_len, 11);
    let hand = hand_ptx(arm);
    let mutant = mutate(&kir_ptx(arm));
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, &c, &src, order);
        let (c, src) = (&c, &src);
        match std::panic::catch_unwind(|| run(&mutant, c, src, order)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// `ptx` with its `i`th line starting with `op` rewritten by `f`
/// (deleted when `f` returns `None`).
fn at_nth(ptx: &str, op: &str, i: usize, f: impl Fn(&str) -> Option<String>) -> String {
    let n = ptx.lines().enumerate().filter(|(_, l)| l.trim_start().starts_with(op)).nth(i).expect("the line").0;
    let mut out: Vec<String> = Vec::new();
    for (k, l) in ptx.lines().enumerate() {
        if k != n {
            out.push(l.to_string());
        } else if let Some(r) = f(l) {
            out.push(r);
        }
    }
    out.join("\n") + "\n"
}

fn count(ptx: &str, op: &str) -> usize {
    ptx.lines().filter(|l| l.trim_start().starts_with(op)).count()
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for arm in ScopyArm::ALL {
        assert!(!caught(arm, |p| p.to_string()), "{arm:?}");
    }
}

/// The two bounds, in emission order: the thread's unit against the run
/// (`i >= units`: exit), and the run index against `outer`. `>=` -> `>`
/// lets one more unit write into the next run (or past `dst`) and one more
/// run read past `offsets`.
#[test]
fn relaxing_either_bound_is_caught() {
    for arm in ScopyArm::ALL {
        assert_eq!(count(&kir_ptx(arm), "setp.ge.u32 "), 2, "{arm:?}");
        for i in 0..2 {
            let relax = |p: &str| at_nth(p, "setp.ge.u32 ", i, |l| Some(l.replacen("setp.ge.u32", "setp.gt.u32", 1)));
            assert!(caught(arm, relax), "{arm:?}: bound {i} relaxed went unnoticed");
        }
    }
}

#[test]
fn dropping_the_store_is_caught() {
    for arm in ScopyArm::ALL {
        let st = if arm.vec4() { "st.global.v4.f32 " } else { "st.global.f32 " };
        assert!(caught(arm, |p| at_nth(p, st, 0, |_| None)), "{arm:?}");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    for arm in ScopyArm::ALL {
        // The offsets' element size, then f32's (both addresses).
        let mut nudges = vec![(", 8;", ", 4;", 0usize), (", 4;", ", 8;", 0), (", 4;", ", 2;", 1)];
        if arm.vec4() {
            // The unit's width in elements, as the x bound's shift and as
            // the element index's.
            nudges.push((", 2;", ", 1;", 0));
            nudges.push((", 2;", ", 3;", 1));
        }
        for (from, to, nth) in nudges {
            let nudge = |p: &str| {
                let n = p.lines().enumerate().filter(|(_, l)| l.contains(from)).nth(nth).expect("the constant").0;
                p.lines()
                    .enumerate()
                    .map(|(k, l)| if k == n { l.replacen(from, to, 1) } else { l.to_string() })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            };
            assert!(caught(arm, nudge), "{arm:?}: `{from}` #{nth} -> `{to}` went unnoticed");
        }
    }
}

/// `%ctaid.x` as 0: the second x block redoes the first one's units and
/// the tail is never written. `%ctaid.y` as 0: every y block walks from
/// run 0, so runs are skipped.
#[test]
fn ignoring_either_block_index_is_caught() {
    for arm in ScopyArm::ALL {
        for special in ["%ctaid.x;", "%ctaid.y;"] {
            assert_eq!(kir_ptx(arm).matches(special).count(), 1, "{arm:?}: {special}");
            assert!(caught(arm, |p| p.replacen(special, "0;", 1)), "{arm:?}: {special} as 0 went unnoticed");
        }
    }
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    let hand = hand_ptx(ScopyArm::Run4);
    let kir = kir_ptx(ScopyArm::Run4);
    for p in [&hand, &kir] {
        assert!(p.contains("%nctaid.y") && p.contains("ld.global.v4.f32 {") && p.contains("st.global.v4.f32 ["));
        parse(p);
    }
    // The hand kernels name their exit on the instruction's own line.
    assert!(hand.contains("SC_RUN4_DONE: ret;"));
}
