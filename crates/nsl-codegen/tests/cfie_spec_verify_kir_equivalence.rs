//! The differential equivalence gate for the CFIE tree-mask verify
//! attention kernel, `nsl_cfie_spec_verify_attn` (roadmap A2 step 9, the
//! speculative-decoding slice's second half).
//!
//! The KIR kernel is the decode-attention kernel run once per tree node,
//! assembled from `cfie_decode_attention`'s sections, plus one masked tile
//! over the appended draft rows. This file runs it and the frozen hand
//! emitter (`tests/fixtures/cfie_speculative_hand.rs`) side by side on the
//! cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), the
//! spec's level 3:
//!
//! 1. **Agreement**: for every geometry, tree, sequence length, layer and
//!    slot below, under two thread schedules, the hand module and the KIR
//!    module leave *the same bytes* in all of global memory.
//! 2. **Correctness**: those bytes are tree attention. They match
//!    `cpu_reference_verify` fed the same f16 K/V, within the tolerance
//!    the decode gate uses (the kernel's exp is `2^(x log2 e)` and it
//!    accumulates in a different order).
//! 3. **The gate bites**: deleting a barrier the kernel needs, nudging a
//!    baked stride or the softmax scale, flipping one bit of one node's
//!    mask, dropping the mask, or dropping the prefix tail clamp is
//!    caught. The barriers that are *not* needed are named, with why
//!    deleting them cannot change any execution.
//!
//! As in the decode gate, the interpreter models `ex2.approx.f32` as
//! `f32::exp2`; the device suite (`cfie_speculative_gpu_parity.rs`) and the
//! `ptxas` gate carry fidelity to the machine.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_speculative::build_tree_mask;
use nsl_codegen::cfie_speculative_ptx::{
    cpu_reference_verify, emit_verify_attention, mask_bits_from_tree, VerifyAttentionConfig,
};

/// The two `crate::` paths the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod cfie_speculative {
    pub use nsl_codegen::cfie_speculative::TreeMask;
}
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_speculative_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const Q_BASE: u64 = 0x1000_0000;
const KV_BASE: u64 = 0x2000_0000;
const OUT_BASE: u64 = 0x3000_0000;
const BLOCK: u32 = 128;

#[derive(Debug, Clone)]
struct Geometry {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    per_slot: u32,
    slots: u32,
    /// One row per tree node: bit `c` of row `r` set iff node `r` attends
    /// node `c`.
    mask: Vec<u64>,
}

impl Geometry {
    fn nodes(&self) -> u32 {
        self.mask.len() as u32
    }

    fn cfg(&self) -> VerifyAttentionConfig {
        VerifyAttentionConfig {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            num_nodes: self.nodes(),
            mask_bits: self.mask.clone(),
        }
    }

    fn hand_cfg(&self) -> hand::VerifyAttentionConfig {
        hand::VerifyAttentionConfig {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            num_nodes: self.nodes(),
            mask_bits: self.mask.clone(),
            sm_version: 80,
        }
    }

    fn max_tokens(&self) -> usize {
        (self.slots * self.per_slot) as usize
    }

    /// Element index of `(layer, plane, token, kv_head, d)` in the pool.
    fn kv_index(&self, layer: u32, plane: u32, token: usize, kv_head: u32, d: u32) -> usize {
        (((layer as usize * 2 + plane as usize) * self.max_tokens() + token) * self.n_kv_heads as usize
            + kv_head as usize)
            * self.head_dim as usize
            + d as usize
    }

    fn pool_elems(&self) -> usize {
        self.n_layers as usize * 2 * self.max_tokens() * (self.n_kv_heads * self.head_dim) as usize
    }

    /// Elements of `q` and of `out`: `[num_nodes][n_heads][head_dim]`.
    fn row_elems(&self) -> usize {
        (self.nodes() * self.n_heads * self.head_dim) as usize
    }
}

/// Deterministic values in [-1, 1), and bits.
struct Lcg(u64);

impl Lcg {
    fn step(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next(&mut self) -> f32 {
        ((self.step() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A tree of `n` nodes in which node `r` attends itself and an arbitrary
/// subset of the earlier nodes: every row differs, and every node but the
/// root has some bits clear, so a mask read from the wrong row or tested
/// at the wrong bit shows.
fn scattered_mask(n: u32, seed: u64) -> Vec<u64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|r| {
            let below = if r == 0 { 0 } else { rng.step() & ((1u64 << r) - 1) };
            below | (1u64 << r)
        })
        .collect()
}

/// A chain: node `r` attends itself and every earlier node.
fn chain_mask(n: u32) -> Vec<u64> {
    (0..n).map(|r| (1u64 << (r + 1)) - 1).collect()
}

struct Inputs {
    q: Vec<f32>,
    /// The whole pool, every layer and slot filled, so an address that
    /// strays reads a different value rather than a zero.
    pool: Vec<f16>,
}

fn inputs(g: &Geometry, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let q = (0..g.row_elems()).map(|_| rng.next()).collect();
    let pool = (0..g.pool_elems()).map(|_| f16::from_f32(rng.next())).collect();
    Inputs { q, pool }
}

#[derive(Debug, Clone, Copy)]
struct Call {
    layer: u32,
    slot: u32,
    seq_len: u32,
}

/// Global memory after running `ptx` over every CTA: `[q, kv, out]`.
fn run(ptx: &str, g: &Geometry, input: &Inputs, call: Call, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let q_bytes: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv_bytes: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    // A NaN sentinel: an output element the kernel fails to write shows.
    let out_bytes: Vec<u8> = (0..g.row_elems()).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect();
    let mut global = vec![
        Segment { base: Q_BASE, bytes: q_bytes },
        Segment { base: KV_BASE, bytes: kv_bytes },
        Segment { base: OUT_BASE, bytes: out_bytes },
    ];
    let args: HashMap<String, u64> = [
        ("q_ptr", Q_BASE),
        ("kv_base", KV_BASE),
        ("out_ptr", OUT_BASE),
        ("layer_idx", call.layer as u64),
        ("slot_idx", call.slot as u64),
        ("seq_len", call.seq_len as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    for ctaid in 0..g.n_heads {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            // Poisoned (0xFF.. is a NaN as f32): a read of shared memory
            // no thread wrote this launch shows in the output.
            shared: vec![0xFF; prog.shared_bytes],
            ctaid,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(g: &Geometry) -> String {
    emit_verify_attention(&g.cfg()).0
}

fn hand_ptx(g: &Geometry) -> String {
    hand::emit_verify_attention_ptx(&g.hand_cfg())
}

fn out_f32(mem: &[Vec<u8>]) -> Vec<f32> {
    mem[2].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Tree attention over the slot's first `seq_len` tokens and the draft
/// rows after them, from `cpu_reference_verify` fed the same f16 values.
fn reference(g: &Geometry, input: &Inputs, call: Call) -> Vec<f32> {
    let first = (call.slot * g.per_slot) as usize;
    let rows = |plane: u32| -> Vec<f32> {
        let mut out = Vec::new();
        for t in 0..(call.seq_len + g.nodes()) as usize {
            for h in 0..g.n_kv_heads {
                for d in 0..g.head_dim {
                    out.push(input.pool[g.kv_index(call.layer, plane, first + t, h, d)].to_f32());
                }
            }
        }
        out
    };
    cpu_reference_verify(&g.cfg(), &input.q, &rows(0), &rows(1), call.seq_len)
}

/// The geometries and calls the agreement and correctness gates sweep.
fn cases() -> Vec<(Geometry, Vec<Call>)> {
    vec![
        // The paper's tree(2,2) (7 BFS nodes), GQA group 2, a mid-pool
        // layer and slot, and prefixes that cross every tile edge: empty,
        // one token, one short of a tile, exactly one, one over, and two
        // tiles with a ragged tail.
        (
            Geometry {
                n_layers: 2,
                n_heads: 2,
                n_kv_heads: 1,
                head_dim: 8,
                per_slot: 300,
                slots: 3,
                mask: mask_bits_from_tree(&build_tree_mask(3, 2)),
            },
            [0, 1, 127, 128, 129, 250]
                .into_iter()
                .map(|seq_len| Call { layer: 1, slot: 1, seq_len })
                .chain([Call { layer: 0, slot: 2, seq_len: 77 }])
                .collect(),
        ),
        // MHA, head_dim not a power of two (threads 40..128 score but own
        // no output), a scattered 9-node mask.
        (
            Geometry {
                n_layers: 1,
                n_heads: 3,
                n_kv_heads: 3,
                head_dim: 40,
                per_slot: 200,
                slots: 1,
                mask: scattered_mask(9, 3),
            },
            vec![Call { layer: 0, slot: 0, seq_len: 5 }, Call { layer: 0, slot: 0, seq_len: 180 }],
        ),
        // The widest tree (33 nodes, K = 32), head_dim == block so every
        // thread owns an output element.
        (
            Geometry {
                n_layers: 1,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 128,
                per_slot: 200,
                slots: 1,
                mask: scattered_mask(33, 11),
            },
            vec![Call { layer: 0, slot: 0, seq_len: 130 }],
        ),
        // head_dim 1, group 4, a chain; and a single node (plain decode
        // with the draft row appended).
        (
            Geometry { n_layers: 3, n_heads: 4, n_kv_heads: 1, head_dim: 1, per_slot: 140, slots: 2, mask: chain_mask(4) },
            vec![Call { layer: 2, slot: 1, seq_len: 129 }],
        ),
        (
            Geometry { n_layers: 1, n_heads: 2, n_kv_heads: 2, head_dim: 16, per_slot: 64, slots: 2, mask: vec![1] },
            vec![Call { layer: 0, slot: 1, seq_len: 0 }, Call { layer: 0, slot: 1, seq_len: 63 }],
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
        let input = inputs(&g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, &g, &input, call, order);
                let got = run(&kir, &g, &input, call, order);
                assert!(
                    expect == got,
                    "{g:?} {call:?} {order:?}: global memory differs\nhand: {:?}\nkir:  {:?}",
                    out_f32(&expect),
                    out_f32(&got)
                );
            }
        }
    }
}

#[test]
fn the_shared_answer_is_tree_attention() {
    for (g, calls) in cases() {
        let kir = kir_ptx(&g);
        let input = inputs(&g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            let got = out_f32(&run(&kir, &g, &input, call, Order::Ascending));
            let want = reference(&g, &input, call);
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                    "{g:?} {call:?}: out[{i}] = {a}, reference {b}"
                );
            }
        }
    }
}

#[test]
fn inputs_are_left_untouched() {
    let (g, calls) = cases().remove(0);
    let input = inputs(&g, 7);
    let mem = run(&kir_ptx(&g), &g, &input, calls[5], Order::Descending);
    let q: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    assert!(mem[0] == q && mem[1] == kv, "the kernel wrote to its inputs");
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_entry_name_and_launch_shape() {
    for (g, _) in cases() {
        let (hand, kir) = (parse(&hand_ptx(&g)), parse(&kir_ptx(&g)));
        assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
        let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["q_ptr", "kv_base", "out_ptr", "layer_idx", "slot_idx", "seq_len"]);
        assert!(kir_ptx(&g).contains(".visible .entry nsl_cfie_spec_verify_attn("));
        // Same shared footprint, one static block.
        assert_eq!(hand.shared_bytes, kir.shared_bytes);
        assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
        let (_, meta) = emit_verify_attention(&g.cfg());
        let (_, hand_meta) = hand::emit_verify_attention(&g.hand_cfg());
        assert_eq!(
            (meta.smem_bytes, meta.block_dim, meta.grid_dim_is_n_heads),
            (hand_meta.smem_bytes, hand_meta.block_dim, hand_meta.grid_dim_is_n_heads)
        );
    }
}

#[test]
fn the_kir_kernel_keeps_the_hand_kernels_header_lines() {
    // The stride lines are compared with the decode kernel's byte for
    // byte (the module's unit test), and the mask lines document the tree.
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

/// The case every mutation is judged on: two prefix tiles with a ragged
/// tail, four nodes whose masks differ, GQA. Its baked constants are
/// distinct wherever a test nudges one (head_dim 8, token stride 16,
/// per-slot 300, kv half 14400, layer 28800, group 2, tile 128, node
/// stride 32, nodes 4, and the mask rows 3, 5 and 13; row 0's mask is the
/// `1` every loop increments by).
fn mutation_case() -> (Geometry, Call) {
    (
        Geometry {
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            per_slot: 300,
            slots: 3,
            mask: vec![0b0001, 0b0011, 0b0101, 0b1101],
        },
        Call { layer: 1, slot: 1, seq_len: 200 },
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

/// `ptx` with every `mov.*` line ending in `from` rewritten to end in `to`,
/// after checking there are `expect` of them.
fn remat(ptx: &str, from: &str, to: &str, expect: usize) -> String {
    let hit = |l: &str| l.trim_start().starts_with("mov.") && l.ends_with(from);
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

/// Each node's barriers, in emission order: the Q row load; the prefix
/// tile's scores, softmax and tile end; the tree tile's scores, softmax
/// and tile end; the `l` publish; and the node's end.
const PER_NODE: usize = 9;
/// The tree tile's closing barrier and the node's end barrier guard
/// nothing: see `the_redundant_barriers_are_named_equivalent_mutants`.
const REDUNDANT: [usize; 2] = [6, 8];

#[test]
fn deleting_any_needed_barrier_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, PER_NODE * g.mask.len(), "nine per node");
    for n in (0..barriers).filter(|n| !REDUNDANT.contains(&(n % PER_NODE))) {
        assert!(caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

/// Two barriers per node order nothing, in the hand kernel as in this one.
///
/// * The tree tile's closing barrier (6) exists in the prefix loop so the
///   next tile's pass 1 cannot overwrite `scores` while a thread is still
///   reading it in pass 3. After the tree tile no pass 1 follows until the
///   next node, which is past the `l` publish barrier (7) anyway; the only
///   shared write between them is thread 0's `l` store, to a region
///   pass 3 does not read.
/// * The node's end barrier (8) keeps the next node's Q row load from
///   overwriting `q` under a reader. Nothing reads `q` after the tree
///   tile's scores barrier (4), and the publish barrier (7) already
///   separates every read of this node's `l` from the next node's
///   thread-0 store, which is several barriers later still.
///
/// A test claiming to catch their deletion would be vacuous, so this pins
/// that it is not caught, and that the kernel still has them (they are
/// what makes the kernel correct should a later change add a reader).
#[test]
fn the_redundant_barriers_are_named_equivalent_mutants() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    for node in 0..g.mask.len() {
        for r in REDUNDANT {
            let n = node * PER_NODE + r;
            assert!(!caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} is not redundant after all");
        }
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let token_stride = (g.n_kv_heads * g.head_dim) as u64;
    let kv_half = g.max_tokens() as u64 * token_stride;
    let scale = (1.0f32 / (g.head_dim as f32).sqrt()).to_bits();
    let node_stride = g.n_heads * g.head_dim;
    for (from, to, expect) in [
        (format!(", {token_stride};"), format!(", {};", token_stride + 1), 1),
        (format!(", {kv_half};"), format!(", {};", kv_half - 1), 1),
        (format!(", {};", 2 * kv_half), format!(", {};", 2 * kv_half + 1), 1),
        (format!(", {};", g.per_slot), format!(", {};", g.per_slot - 1), 1),
        (format!(", 0f{scale:08X};"), format!(", 0f{:08X};", scale + 1), 1),
        // Node 1's Q/out row offset.
        (format!(", {node_stride};"), format!(", {};", node_stride + 1), 1),
    ] {
        assert!(caught(&remat(&ptx, &from, &to, expect)), "`{from}` -> `{to}` went unnoticed");
    }
}

#[test]
fn flipping_one_mask_bit_is_caught() {
    // Each non-root row gains the bit it lacks or loses the one it has,
    // one at a time. Row 0's only bit is its self bit, whose loss the
    // emitter refuses (and the mask immediate `1` is not unique).
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    for (row, flipped) in
        [(0b0011u64, 0b0111u64), (0b0011, 0b0010), (0b0101, 0b0111), (0b0101, 0b0100), (0b1101, 0b1111), (0b1101, 0b1001)]
    {
        let mutant = remat(&ptx, &format!(", {row};"), &format!(", {flipped};"), 1);
        assert!(caught(&mutant), "mask {row:#b} -> {flipped:#b} went unnoticed");
    }
}

#[test]
fn dropping_the_mask_is_caught() {
    // `selp` keeps the score whatever the bit: every draft row counts.
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let selects: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("selp.f32 ")).collect();
    assert_eq!(selects.len(), g.mask.len(), "one mask select per node");
    for (n, sel) in selects.iter().enumerate() {
        let ops: Vec<&str> = sel.trim().trim_start_matches("selp.f32 ").trim_end_matches(';').split(", ").collect();
        let mutant = ptx.replacen(sel, &format!("    mov.f32 {}, {};", ops[0], ops[1]), 1);
        assert!(caught(&mutant), "node {n}'s mask dropped went unnoticed");
    }
}

#[test]
fn dropping_the_prefix_tail_clamp_is_caught() {
    // `tcnt = min(seq_len - tile, TILE)` -> `tcnt = TILE` in one node's
    // prefix loop: its ragged last tile then runs softmax and P*V over
    // scores no thread wrote.
    let (g, _) = mutation_case();
    let ptx = kir_ptx(&g);
    let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
    assert_eq!(clamps.len(), g.mask.len(), "one tail clamp per node");
    for clamp in clamps {
        let (dst, rest) = clamp.trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
        let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
        let mutant = ptx.replacen(clamp, &format!("    mov.u32 {dst}, {tile};"), 1);
        assert!(caught(&mutant), "`{clamp}` removed went unnoticed");
    }
}

#[test]
fn the_interpreter_knows_the_mask_test_forms() {
    // The hand kernel tests a mask bit with `shr.b64` by a u32 register and
    // `and.b64` with an immediate; the KIR one with `shr.u64`, `and.b64`
    // on registers and `selp.f32`. All are modelled rather than skipped.
    let (g, _) = mutation_case();
    let hand = hand_ptx(&g);
    assert!(hand.contains("and.b64 %rd_mb, %rd_mb, 1;"));
    let kir = kir_ptx(&g);
    for form in ["shr.u64 ", "and.b64 ", "selp.f32 "] {
        assert!(kir.lines().any(|l| l.trim_start().starts_with(form)), "{form}");
    }
    parse(&hand);
    parse(&kir);
}
