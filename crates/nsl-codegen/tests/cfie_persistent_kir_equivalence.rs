//! The differential equivalence gate for the CFIE persistent decode-block
//! kernel, `nsl_cfie_decode_block` (roadmap A2 step 9, the last CFIE
//! slice).
//!
//! `cfie_persistent_ptx` emitted the kernel as hand-assembled PTX; it now
//! builds it as KIR, its attention from `cfie_decode_attention`'s
//! flash-decode sections and its RMSNorms from `cfie_spec_sampler_ptx`'s
//! sum-of-squares tree. This file runs the frozen hand emitter
//! (`tests/fixtures/cfie_persistent_hand.rs`) and the KIR one side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! the spec's level 3:
//!
//! 1. **Agreement**: for every geometry and call below, under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory — the new residual row and the KV pool, whose
//!    `(layer, slot, pos)` record both kernels append.
//! 2. **Correctness**: those bytes are the decode block. The output row
//!    matches `cpu_reference` fed the same f16 weights and the slot's f16
//!    history, and the appended K/V record matches the reference's, rounded
//!    to f16.
//! 3. **The gate bites**: deleting a barrier the kernel needs, nudging a
//!    baked constant, or dropping a tail clamp is caught. The barriers that
//!    are *not* needed are named, with why deleting them cannot change any
//!    execution.
//!
//! As in the sibling gates, the interpreter models `ex2.approx`,
//! `sin.approx` and `cos.approx` by their exact counterparts; the device
//! suites (`cfie_decode_block_gpu_parity.rs` and the generate end-to-end
//! tests) and the `ptxas` gate carry fidelity to the machine.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, emit, DecodeBlockConfig};

/// The two `crate::` paths the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod cfie_decode_attention {
    pub use nsl_codegen::cfie_decode_attention::KERNEL_NAME;
}
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_persistent_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const BLOCK: u32 = 128;

/// The global buffers, in the order `run` returns them, with their bases.
const BUFFERS: [(&str, u64); 12] = [
    ("x_in_ptr", 0x1000_0000),
    ("x_out_ptr", 0x1100_0000),
    ("wq_ptr", 0x2000_0000),
    ("wk_ptr", 0x2100_0000),
    ("wv_ptr", 0x2200_0000),
    ("wo_ptr", 0x2300_0000),
    ("w_gate_ptr", 0x2400_0000),
    ("w_up_ptr", 0x2500_0000),
    ("w_down_ptr", 0x2600_0000),
    ("norm1_w_ptr", 0x3000_0000),
    ("norm2_w_ptr", 0x3100_0000),
    ("kv_base", 0x4000_0000),
];
const X_OUT: usize = 1;
const KV: usize = 11;

#[derive(Debug, Clone)]
struct Geometry {
    d_model: u32,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    d_ff: u32,
    per_slot: u32,
    slots: u32,
    n_layers: u32,
    rope_theta: f32,
    eps: f32,
}

impl Geometry {
    fn cfg(&self) -> DecodeBlockConfig {
        DecodeBlockConfig {
            d_model: self.d_model,
            head_dim: self.head_dim,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            d_ff: self.d_ff,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            n_layers: self.n_layers,
            rope_theta: self.rope_theta,
            eps: self.eps,
        }
    }

    fn hand_cfg(&self) -> hand::DecodeBlockConfig {
        hand::DecodeBlockConfig {
            d_model: self.d_model,
            head_dim: self.head_dim,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            d_ff: self.d_ff,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            n_layers: self.n_layers,
            rope_theta: self.rope_theta,
            eps: self.eps,
            sm_version: 80,
        }
    }

    fn nhd(&self) -> usize {
        (self.n_heads * self.head_dim) as usize
    }

    fn kv_rows(&self) -> usize {
        (self.n_kv_heads * self.head_dim) as usize
    }

    fn max_tokens(&self) -> usize {
        (self.slots * self.per_slot) as usize
    }

    /// Element index of `(layer, plane, token)`'s record in the pool.
    fn record(&self, layer: u32, plane: u32, token: usize) -> usize {
        ((layer as usize * 2 + plane as usize) * self.max_tokens() + token) * self.kv_rows()
    }

    fn pool_elems(&self) -> usize {
        self.n_layers as usize * 2 * self.max_tokens() * self.kv_rows()
    }
}

/// Deterministic values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// One launch's inputs: the residual row, the weights (f16 matrices, f32
/// norm gammas), and the whole pool, every layer and slot filled, so an
/// address that strays reads a different value rather than a zero.
struct Inputs {
    x: Vec<f32>,
    wq: Vec<f16>,
    wk: Vec<f16>,
    wv: Vec<f16>,
    wo: Vec<f16>,
    w_gate: Vec<f16>,
    w_up: Vec<f16>,
    w_down: Vec<f16>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    pool: Vec<f16>,
}

fn inputs(g: &Geometry, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let (d, dff) = (g.d_model as usize, g.d_ff as usize);
    // Weights scaled so a row's dot stays O(1) whatever its length: the
    // softmax and silu then see varied, unsaturated inputs.
    let mut mat = |rows: usize, cols: usize| -> Vec<f16> {
        let s = 1.5 / (cols as f32).sqrt();
        (0..rows * cols).map(|_| f16::from_f32(rng.next() * s)).collect()
    };
    let wq = mat(g.nhd(), d);
    let wk = mat(g.kv_rows(), d);
    let wv = mat(g.kv_rows(), d);
    let wo = mat(d, g.nhd());
    let w_gate = mat(dff, d);
    let w_up = mat(dff, d);
    let w_down = mat(d, dff);
    let x = (0..d).map(|_| rng.next() * 2.0).collect();
    let norm1 = (0..d).map(|_| 1.0 + rng.next() * 0.5).collect();
    let norm2 = (0..d).map(|_| 1.0 + rng.next() * 0.5).collect();
    let pool = (0..g.pool_elems()).map(|_| f16::from_f32(rng.next())).collect();
    Inputs { x, wq, wk, wv, wo, w_gate, w_up, w_down, norm1, norm2, pool }
}

#[derive(Debug, Clone, Copy)]
struct Call {
    layer: u32,
    slot: u32,
    pos: u32,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

/// Global memory after running `ptx` once, in [`BUFFERS`] order.
fn run(ptx: &str, g: &Geometry, input: &Inputs, call: Call, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    // A NaN sentinel: an output element the kernel fails to write shows.
    let out = (0..g.d_model).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect();
    let contents = [
        f32_bytes(&input.x),
        out,
        f16_bytes(&input.wq),
        f16_bytes(&input.wk),
        f16_bytes(&input.wv),
        f16_bytes(&input.wo),
        f16_bytes(&input.w_gate),
        f16_bytes(&input.w_up),
        f16_bytes(&input.w_down),
        f32_bytes(&input.norm1),
        f32_bytes(&input.norm2),
        f16_bytes(&input.pool),
    ];
    let mut global: Vec<Segment> =
        BUFFERS.iter().zip(contents).map(|(&(_, base), bytes)| Segment { base, bytes }).collect();
    let args: HashMap<String, u64> = BUFFERS
        .iter()
        .map(|&(name, base)| (name.to_string(), base))
        .chain([
            ("layer_idx".to_string(), call.layer as u64),
            ("slot_idx".to_string(), call.slot as u64),
            ("pos".to_string(), call.pos as u64),
        ])
        .collect();
    let mut launch = Launch {
        prog: &prog,
        args: &args,
        global: &mut global,
        // Poisoned (0xFF.. is a NaN as f32): a read of shared memory no
        // thread wrote this launch shows in the output.
        shared: vec![0xFF; prog.shared_bytes],
        ctaid: 0,
        ctaid_y: 0,
        nctaid_y: 1,
        ntid: BLOCK,
        steps: 0,
    };
    run_cta(&mut launch, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(g: &Geometry) -> String {
    emit(&g.cfg()).0
}

fn hand_ptx(g: &Geometry) -> String {
    hand::emit_decode_block_ptx(&g.hand_cfg())
}

fn out_f32(mem: &[Vec<u8>]) -> Vec<f32> {
    mem[X_OUT].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn widen(v: &[f16]) -> Vec<f32> {
    v.iter().map(|h| h.to_f32()).collect()
}

/// `cpu_reference` fed the f16 weights, widened, and the slot's first
/// `pos` records of this layer's K and V planes: the new residual row, and
/// the K and V records it appends.
fn reference(g: &Geometry, input: &Inputs, call: Call) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let first = (call.slot * g.per_slot) as usize;
    let history = |plane: u32| -> Vec<f32> {
        let start = g.record(call.layer, plane, first);
        widen(&input.pool[start..start + call.pos as usize * g.kv_rows()])
    };
    let (mut k, mut v) = (history(0), history(1));
    let out = cpu_reference(
        &g.cfg(),
        &input.x,
        &widen(&input.wq),
        &widen(&input.wk),
        &widen(&input.wv),
        &widen(&input.wo),
        &widen(&input.w_gate),
        &widen(&input.w_up),
        &widen(&input.w_down),
        &input.norm1,
        &input.norm2,
        &mut k,
        &mut v,
        call.pos,
    );
    let appended = call.pos as usize * g.kv_rows();
    (out, k[appended..].to_vec(), v[appended..].to_vec())
}

/// The geometries and calls the agreement and correctness gates sweep.
fn cases() -> Vec<(Geometry, Vec<Call>)> {
    let g = |d_model, head_dim, n_heads, n_kv_heads, d_ff, per_slot, slots, n_layers| Geometry {
        d_model,
        head_dim,
        n_heads,
        n_kv_heads,
        d_ff,
        per_slot,
        slots,
        n_layers,
        rope_theta: 10_000.0,
        eps: 1e-5,
    };
    vec![
        // GQA group 2 in a mid-pool layer and slot, with sequence lengths
        // (pos + 1) that cross every attention tile edge: one token, two,
        // a full tile, one over, and two tiles with a ragged tail.
        (
            g(24, 8, 2, 1, 40, 300, 2, 2),
            [0, 1, 127, 128, 200].into_iter().map(|pos| Call { layer: 1, slot: 1, pos }).collect(),
        ),
        // d_model past the block (every strided loop takes two rounds),
        // d_ff over two FFN tiles with a ragged third, GQA group 2.
        (g(136, 16, 4, 2, 300, 64, 1, 1), vec![Call { layer: 0, slot: 0, pos: 5 }]),
        // head_dim == block: every thread owns an attention output element.
        (g(8, 128, 1, 1, 8, 16, 1, 1), vec![Call { layer: 0, slot: 0, pos: 3 }]),
        // 256 Q pairs and 256 V rows (two strided rounds each), MHA-by-2.
        (g(16, 128, 4, 2, 16, 8, 2, 1), vec![Call { layer: 0, slot: 1, pos: 2 }]),
        // The smallest head, MHA, a single FFN row, and a non-default RoPE
        // base and epsilon (both are baked).
        (
            Geometry { rope_theta: 500_000.0, eps: 1e-6, ..g(4, 2, 3, 3, 1, 140, 1, 3) },
            vec![Call { layer: 2, slot: 0, pos: 130 }],
        ),
    ]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for (g, calls) in cases() {
        let (hand, kir) = (hand_ptx(&g), kir_ptx(&g));
        let input = inputs(&g, 0x5eed ^ g.d_model as u64);
        for call in calls {
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, &g, &input, call, order);
                let got = run(&kir, &g, &input, call, order);
                for (i, (e, k)) in expect.iter().zip(&got).enumerate() {
                    assert!(
                        e == k,
                        "{g:?} {call:?} {order:?}: `{}` differs\nhand out: {:?}\nkir out:  {:?}",
                        BUFFERS[i].0,
                        out_f32(&expect),
                        out_f32(&got)
                    );
                }
            }
        }
    }
}

#[test]
fn the_shared_answer_is_the_decode_block() {
    for (g, calls) in cases() {
        let kir = kir_ptx(&g);
        let input = inputs(&g, 0x5eed ^ g.d_model as u64);
        for call in calls {
            let mem = run(&kir, &g, &input, call, Order::Ascending);
            let (want, k_new, v_new) = reference(&g, &input, call);
            // The kernel attends over its appended record *as stored*, in
            // f16, where the reference keeps f32; that rounding bounds the
            // agreement (the cases here land within 3.4e-4).
            for (i, (a, b)) in out_f32(&mem).iter().zip(&want).enumerate() {
                assert!((a - b).abs() <= 2e-3 * (1.0 + b.abs()), "{g:?} {call:?}: out[{i}] = {a}, reference {b}");
            }
            let pool: Vec<f16> =
                mem[KV].chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
            let token = (call.slot * g.per_slot + call.pos) as usize;
            for (plane, want) in [(0, &k_new), (1, &v_new)] {
                let start = g.record(call.layer, plane, token);
                for (e, w) in want.iter().enumerate() {
                    let got = pool[start + e].to_f32();
                    // Half an f16 ulp (2^-11 relative), plus the f32
                    // rounding the two dot orders differ by.
                    assert!(
                        (got - w).abs() <= 1e-3 * (1.0 + w.abs()),
                        "{g:?} {call:?}: plane {plane} element {e} = {got}, reference {w}"
                    );
                }
            }
        }
    }
}

#[test]
fn only_the_output_row_and_one_pool_record_are_written() {
    let (g, calls) = cases().remove(0);
    let input = inputs(&g, 7);
    let call = calls[4];
    let mem = run(&kir_ptx(&g), &g, &input, call, Order::Descending);
    let unchanged = [
        f32_bytes(&input.x),
        f16_bytes(&input.wq),
        f16_bytes(&input.wk),
        f16_bytes(&input.wv),
        f16_bytes(&input.wo),
        f16_bytes(&input.w_gate),
        f16_bytes(&input.w_up),
        f16_bytes(&input.w_down),
        f32_bytes(&input.norm1),
        f32_bytes(&input.norm2),
    ];
    for (i, want) in [0usize, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().zip(unchanged) {
        assert!(mem[i] == want, "the kernel wrote to `{}`", BUFFERS[i].0);
    }
    // The pool differs from its input only inside this token's K and V
    // records.
    let token = (call.slot * g.per_slot + call.pos) as usize;
    let records: Vec<std::ops::Range<usize>> = (0..2)
        .map(|plane| {
            let start = 2 * g.record(call.layer, plane, token);
            start..start + 2 * g.kv_rows()
        })
        .collect();
    let before = f16_bytes(&input.pool);
    for (i, (a, b)) in mem[KV].iter().zip(&before).enumerate() {
        if a != b {
            assert!(records.iter().any(|r| r.contains(&i)), "pool byte {i} changed outside the appended records");
        }
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_entry_name_and_launch_shape() {
    for (g, _) in cases() {
        let (hand, kir) = (parse(&hand_ptx(&g)), parse(&kir_ptx(&g)));
        assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
        let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
        let mut want: Vec<&str> = BUFFERS.iter().map(|&(n, _)| n).collect();
        want.extend(["layer_idx", "slot_idx", "pos"]);
        assert_eq!(names, want);
        assert!(kir_ptx(&g).contains(".visible .entry nsl_cfie_decode_block("));
        // Same shared footprint, one static block.
        assert_eq!(hand.shared_bytes, kir.shared_bytes);
        assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
        let (_, meta) = emit(&g.cfg());
        let (_, hand_meta) = hand::emit(&g.hand_cfg());
        assert_eq!(
            (meta.kernel_name, meta.smem_bytes, meta.block_dim),
            (hand_meta.kernel_name, hand_meta.smem_bytes, hand_meta.block_dim)
        );
    }
}

#[test]
fn the_kir_kernel_keeps_the_hand_kernels_header_lines() {
    for (g, _) in cases() {
        let header = |ptx: &str| -> Vec<String> {
            ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
        };
        assert_eq!(header(&hand_ptx(&g)), header(&kir_ptx(&g)), "{g:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The case every mutation is judged on: two attention tiles with a ragged
/// tail, two FFN tiles with a ragged tail, four heads in GQA pairs, and a
/// `d_model` past the block. Its baked constants are distinct wherever a
/// test nudges one (token stride 20, per-slot 300, kv half 12000, layer
/// 24000, d_ff 200).
fn mutation_case() -> (Geometry, Call) {
    (
        Geometry {
            d_model: 136,
            head_dim: 10,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 200,
            per_slot: 300,
            slots: 2,
            n_layers: 2,
            rope_theta: 10_000.0,
            eps: 1e-5,
        },
        Call { layer: 1, slot: 1, pos: 150 },
    )
}

/// Whether `mutant` is told apart from the hand kernel: under either
/// schedule its memory differs, or the interpreter faults on it.
fn caught(mutant: &str) -> bool {
    let (g, call) = mutation_case();
    let input = inputs(&g, 11);
    let hand = hand_ptx(&g);
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, &g, &input, call, order);
        let input = &input;
        let g = &g;
        let got = std::panic::catch_unwind(|| run(mutant, g, input, call, order));
        match got {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// `ptx` with the `n`th line equal to `line` (trimmed) removed.
fn without_nth(ptx: &str, line: &str, n: usize) -> String {
    let mut seen = 0;
    let mut out = String::new();
    for l in ptx.lines() {
        if l.trim() == line {
            seen += 1;
            if seen == n + 1 {
                continue;
            }
        }
        out.push_str(l);
        out.push('\n');
    }
    assert!(seen > n, "fewer than {} `{line}` lines", n + 1);
    out
}

/// `ptx` with every `mov` line of type `ty` ending in `from` rewritten to
/// end in `to`, after checking there are `expect` of them. The type keeps
/// a u64 stride apart from a u32 loop bound of the same value (the token
/// stride is always the V loop's row count).
fn remat(ptx: &str, ty: &str, from: &str, to: &str, expect: usize) -> String {
    let mov = format!("mov.{ty} ");
    let hit = |l: &str| l.trim_start().starts_with(&mov) && l.ends_with(from);
    assert_eq!(ptx.lines().filter(|l| hit(l)).count(), expect, "`{from}` materialisations");
    ptx.lines().map(|l| if hit(l) { l.replace(from, to) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below: without it, `caught` could
    // be reporting a difference the mutation did not cause.
    let (g, _) = mutation_case();
    assert!(!caught(&kir_ptx(&g)));
}

/// The kernel's barriers, in emission order: the x load (0); RMSNorm1's
/// partial sums (1), tree levels (2..=8) and normalized row (9); the Q/K/V
/// phase (10); the attention tile's scores, softmax and end (11..=13); the
/// head's `l` publish and end (14, 15); the W_o residual (16); RMSNorm2
/// (17..=25); the y zeroing (26); the FFN tile's h and down projection
/// (27, 28).
const BARRIERS: usize = 29;
/// See `the_redundant_barriers_are_named_equivalent_mutants`.
const REDUNDANT: [usize; 4] = [0, 16, 25, 26];

#[test]
fn deleting_any_needed_barrier_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, BARRIERS);
    for n in (0..barriers).filter(|n| !REDUNDANT.contains(n)) {
        assert!(caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

/// Four barriers order nothing, in the hand kernel as in this one. Three
/// follow a strided loop in which thread `t` writes only elements
/// `i ≡ t (mod 128)` of a row that, until the next barrier, only the same
/// thread reads, through a loop with the same stride:
///
/// * after the x load (0): RMSNorm1's partial sums read `x[i]` for the
///   thread's own `i`; the next read of `x` by another thread is none at
///   all (the normalize loop, the W_o residual and the output are all
///   own-element).
/// * after the W_o residual (16): the same for RMSNorm2's partial sums. The
///   residual loop's other read, `ao`, was published by the head loop's end
///   barrier, and RMSNorm2 writes `red` and `xn` only after the partial
///   sums, which every thread finished reading long before (barriers 9-15).
/// * after zeroing `y` (26): the down projection reads and writes `y[i]`
///   for the thread's own `i`, and the output reads it the same way.
///
/// The fourth is RMSNorm2's closing barrier (25), which publishes `xn` to
/// the gate/up projection: the `y` zeroing between them writes nothing
/// another thread reads, so barrier 26 publishes `xn` just as well. Each of
/// the two is redundant only while the other stays; deleting both is
/// caught (`deleting_both_barriers_before_the_ffn_is_caught`).
///
/// A test claiming to catch their deletion would be vacuous, so this pins
/// that it is not caught, and that the kernel still has them (they are
/// what makes the kernel correct should a later change add a reader).
#[test]
fn the_redundant_barriers_are_named_equivalent_mutants() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    for n in REDUNDANT {
        assert!(!caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} is not redundant after all");
    }
}

#[test]
fn deleting_both_barriers_before_the_ffn_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    // Barrier 26 first, then 25: the indices below it are unchanged.
    let mutant = without_nth(&without_nth(&ptx, "bar.sync 0;", 26), "bar.sync 0;", 25);
    assert!(caught(&mutant), "xn published to the gate/up projection by no barrier went unnoticed");
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let token_stride = g.kv_rows() as u64;
    let kv_half = g.max_tokens() as u64 * token_stride;
    let bits = |v: f32| v.to_bits();
    let f = |v: f32| format!(", 0f{:08X};", bits(v));
    let f_next = |v: f32| format!(", 0f{:08X};", bits(v) + 1);
    let inv_sqrt_hd = 1.0f32 / (g.head_dim as f32).sqrt();
    let inv_d = 1.0f32 / g.d_model as f32;
    let c_rope = -(g.rope_theta.log2()) / g.head_dim as f32;
    let neg_log2e = -std::f32::consts::LOG2_E;
    for (ty, from, to, expect) in [
        ("u64", format!(", {token_stride};"), format!(", {};", token_stride + 1), 1),
        ("u64", format!(", {kv_half};"), format!(", {};", kv_half - 1), 1),
        ("u64", format!(", {};", 2 * kv_half), format!(", {};", 2 * kv_half + 1), 1),
        ("u32", format!(", {};", g.per_slot), format!(", {};", g.per_slot - 1), 1),
        ("u32", format!(", {};", g.d_ff), format!(", {};", g.d_ff - 1), 1),
        // The two RMSNorms each bake 1/d_model and eps; the Q and K pair
        // loops each bake the RoPE exponent scale.
        ("f32", f(inv_sqrt_hd), f_next(inv_sqrt_hd), 1),
        ("f32", f(inv_d), f_next(inv_d), 2),
        // eps by one ulp vanishes in `mean + eps` (the mean is O(1)); a
        // doubled eps moves the sum by ~100 ulps.
        ("f32", f(g.eps), f(2.0 * g.eps), 2),
        ("f32", f(c_rope), f_next(c_rope), 2),
        ("f32", f(neg_log2e), f_next(neg_log2e), 1),
    ] {
        assert!(caught(&remat(&ptx, ty, &from, &to, expect)), "`{from}` -> `{to}` went unnoticed");
    }
}

#[test]
fn dropping_a_tail_clamp_is_caught() {
    // `min(remaining, 128)` -> `128`: the attention tile's `tcnt` runs its
    // softmax and P*V over scores no thread wrote; the FFN tile's `fcnt`
    // reads h rows past d_ff.
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
    assert_eq!(clamps.len(), 2, "the attention and FFN tile clamps");
    for clamp in clamps {
        let (dst, rest) = clamp.trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
        let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
        let mutant = ptx.replacen(clamp, &format!("    mov.u32 {dst}, {tile};"), 1);
        assert!(caught(&mutant), "`{clamp}` removed went unnoticed");
    }
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // The decode block is the first CFIE kernel to store f16 (the KV
    // append), take a square root, a remainder, or a sine and cosine; all
    // are modelled rather than skipped.
    let (g, _) = mutation_case();
    let kir = kir_ptx(&g);
    for form in
        ["cvt.rn.f16.f32 ", "st.global.b16 ", "sqrt.rn.f32 ", "rem.u32 ", "sin.approx.f32 ", "cos.approx.f32 ", "ex2.approx.f32 "]
    {
        assert!(kir.lines().any(|l| l.trim_start().starts_with(form)), "{form}");
    }
    parse(&hand_ptx(&g));
    parse(&kir);
}

