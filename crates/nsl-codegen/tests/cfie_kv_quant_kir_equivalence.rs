//! The differential equivalence gate for the per-layer KV-quant
//! decode-attention kernels (roadmap A2 step 9, second slice).
//!
//! `cfie_kv_quant_ptx` emitted one hand-assembled kernel per layer, each
//! with its K and V load paths specialized to that layer's precision (f16
//! read directly, or int8 dequantized in registers by a runtime scale). It
//! now builds them as KIR, through the flash-decode builder the base
//! decode-attention kernel uses. This file runs the frozen hand emitter
//! (`tests/fixtures/cfie_kv_quant_hand.rs`) and the KIR one side by side on
//! the cooperative-CTA interpreter `cfie_decode_attn_kir_equivalence.rs`
//! documents (`tests/support/cta_ptx_interp.rs`), the spec's level 3:
//!
//! 1. **Agreement** — for every layer of every mixed-precision pool below,
//!    every sequence length and slot, and under two thread schedules, the
//!    hand module and the KIR module leave *the same bytes* in all of
//!    global memory.
//! 2. **Correctness** — those bytes are attention over the dequantized
//!    cache: they match `cpu_reference_layer`, which mirrors the kernel's
//!    `i8 -> f32 * scale` dequant.
//! 3. **The gate bites** — deleting any one barrier, nudging a baked half
//!    offset, the token stride, the slot base or the softmax scale,
//!    swapping the two halves' dequant scales, or dropping the tail-tile
//!    clamp in the KIR text is caught.
//!
//! The pool is filled everywhere (every layer, both halves, every slot)
//! with values that differ from their neighbours, so an address that
//! strays reads something else rather than a zero; the K and V scales are
//! different numbers, so a half that reads the other half's scale shows.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_kv_quant::KvPrecision;
use nsl_codegen::cfie_kv_quant_ptx::{
    cpu_reference_layer, emit_layer, pool_layout, total_pool_bytes, CpuKvHalf,
    QuantDecodeAttentionConfig,
};

/// The two `crate::` paths the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod cfie_kv_quant {
    pub use nsl_codegen::cfie_kv_quant::KvPrecision;
}
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_kv_quant_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

use KvPrecision::{Fp16 as F, Int8 as I};

const Q_BASE: u64 = 0x1000_0000;
const KV_BASE: u64 = 0x2000_0000;
const OUT_BASE: u64 = 0x3000_0000;
const BLOCK: u32 = 128;

#[derive(Debug, Clone)]
struct Geometry {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    per_slot: u32,
    slots: u32,
    layers: Vec<(KvPrecision, KvPrecision)>,
}

impl Geometry {
    fn cfg(&self) -> QuantDecodeAttentionConfig {
        QuantDecodeAttentionConfig {
            n_layers: self.layers.len() as u32,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            layer_precisions: self.layers.clone(),
        }
    }

    fn hand_cfg(&self) -> hand::QuantDecodeAttentionConfig {
        hand::QuantDecodeAttentionConfig {
            n_layers: self.layers.len() as u32,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            sm_version: 80,
            layer_precisions: self.layers.clone(),
        }
    }

    fn token_elems(&self) -> usize {
        (self.n_kv_heads * self.head_dim) as usize
    }
}

/// Deterministic values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Inputs {
    q: Vec<f32>,
    /// The whole mixed-precision pool, every half filled: f16 halves with
    /// f16 values, int8 halves with every byte value -128..=127 possible.
    pool: Vec<u8>,
    k_scale: f32,
    v_scale: f32,
}

fn inputs(g: &Geometry, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let q = (0..g.n_heads * g.head_dim).map(|_| rng.next()).collect();
    let cfg = g.cfg();
    let mut pool = vec![0u8; total_pool_bytes(&cfg) as usize];
    let half_elems = (g.slots * g.per_slot) as usize * g.token_elems();
    for (o, &(kp, vp)) in pool_layout(&cfg).iter().zip(&g.layers) {
        for (off, p) in [(o.k_offset_bytes, kp), (o.v_offset_bytes, vp)] {
            let off = off as usize;
            for e in 0..half_elems {
                match p {
                    F => pool[off + 2 * e..off + 2 * e + 2]
                        .copy_from_slice(&f16::from_f32(rng.next()).to_bits().to_le_bytes()),
                    _ => pool[off + e] = (rng.next_u64() >> 56) as u8,
                }
            }
        }
    }
    // Different, non-power-of-two scales: a half dequantized with the
    // other half's scale reads different numbers.
    Inputs { q, pool, k_scale: 0.0071 + (seed % 7) as f32 * 1e-4, v_scale: 0.0133 }
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
    // A NaN sentinel: an output element the kernel fails to write shows.
    let out_bytes: Vec<u8> = (0..g.n_heads * g.head_dim).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect();
    let mut global = vec![
        Segment { base: Q_BASE, bytes: q_bytes },
        Segment { base: KV_BASE, bytes: input.pool.clone() },
        Segment { base: OUT_BASE, bytes: out_bytes },
    ];
    let args: HashMap<String, u64> = [
        ("q_ptr", Q_BASE),
        ("kv_base", KV_BASE),
        ("out_ptr", OUT_BASE),
        ("slot_idx", call.slot as u64),
        ("seq_len", call.seq_len as u64),
        ("k_scale", input.k_scale.to_bits() as u64),
        ("v_scale", input.v_scale.to_bits() as u64),
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
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(g: &Geometry, layer: u32) -> String {
    emit_layer(&g.cfg(), layer).0
}

fn hand_ptx(g: &Geometry, layer: u32) -> String {
    hand::emit_layer(&g.hand_cfg(), layer).0
}

fn out_f32(mem: &[Vec<u8>]) -> Vec<f32> {
    mem[2].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What attention over the slot's first `seq_len` tokens of `layer`
/// produces, from `cpu_reference_layer` fed each half as the kernel reads
/// it.
fn reference(g: &Geometry, input: &Inputs, call: Call) -> Vec<f32> {
    let o = pool_layout(&g.cfg())[call.layer as usize];
    let (kp, vp) = g.layers[call.layer as usize];
    let first = (call.slot * g.per_slot) as usize * g.token_elems();
    let n = call.seq_len as usize * g.token_elems();
    let f16s = |off: u64| -> Vec<f32> {
        (first..first + n)
            .map(|e| {
                let at = off as usize + 2 * e;
                f16::from_bits(u16::from_le_bytes([input.pool[at], input.pool[at + 1]])).to_f32()
            })
            .collect()
    };
    let i8s = |off: u64| -> Vec<i8> {
        (first..first + n).map(|e| input.pool[off as usize + e] as i8).collect()
    };
    // Each half decoded only as its own precision: an int8 half read as
    // f16 would run past the end of the pool.
    let decode = |p: KvPrecision, off: u64| match p {
        F => (f16s(off), Vec::new()),
        _ => (Vec::new(), i8s(off)),
    };
    let ((kf, ki), (vf, vi)) = (decode(kp, o.k_offset_bytes), decode(vp, o.v_offset_bytes));
    let k = match kp {
        F => CpuKvHalf::Fp16(&kf),
        _ => CpuKvHalf::Int8 { data: &ki, scale: input.k_scale },
    };
    let v = match vp {
        F => CpuKvHalf::Fp16(&vf),
        _ => CpuKvHalf::Int8 { data: &vi, scale: input.v_scale },
    };
    cpu_reference_layer(&input.q, k, v, g.n_heads, g.n_kv_heads, g.head_dim, call.seq_len)
}

/// The geometries and calls the agreement and correctness gates sweep.
/// Every layer of each pool is run, so every (K, V) precision pairing
/// appears, each at a different baked offset.
fn cases() -> Vec<(Geometry, Vec<Call>)> {
    let every_layer = |g: &Geometry, slot: u32, seqs: &[u32]| -> Vec<Call> {
        (0..g.layers.len() as u32)
            .flat_map(|layer| seqs.iter().map(move |&seq_len| Call { layer, slot, seq_len }))
            .collect()
    };
    let g1 = Geometry {
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        per_slot: 300,
        slots: 3,
        layers: vec![(F, F), (F, I), (I, I), (I, F)],
    };
    // Sequence lengths that cross every tile edge, in a mid-pool slot.
    let mut c1 = every_layer(&g1, 1, &[0, 1, 127, 128, 129, 300]);
    c1.push(Call { layer: 2, slot: 2, seq_len: 77 });
    c1.push(Call { layer: 1, slot: 0, seq_len: 256 });
    // MHA with a head_dim that is not a power of two.
    let g2 = Geometry { n_heads: 3, n_kv_heads: 3, head_dim: 40, per_slot: 200, slots: 1, layers: vec![(I, I), (F, I)] };
    let c2 = every_layer(&g2, 0, &[5, 200]);
    // head_dim == block: every thread owns an output element.
    let g3 = Geometry { n_heads: 1, n_kv_heads: 1, head_dim: 128, per_slot: 130, slots: 1, layers: vec![(I, F), (I, I)] };
    let c3 = every_layer(&g3, 0, &[130]);
    // head_dim 1, group 4 (an even half, so no alignment refusal).
    let g4 = Geometry { n_heads: 4, n_kv_heads: 1, head_dim: 1, per_slot: 129, slots: 2, layers: vec![(I, F), (F, F), (I, I)] };
    let c4 = every_layer(&g4, 1, &[129]);
    vec![(g1, c1), (g2, c2), (g3, c3), (g4, c4)]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for (g, calls) in cases() {
        let input = inputs(&g, 0x5eed ^ g.head_dim as u64);
        for layer in 0..g.layers.len() as u32 {
            let (hand, kir) = (hand_ptx(&g, layer), kir_ptx(&g, layer));
            for call in calls.iter().filter(|c| c.layer == layer) {
                for order in [Order::Ascending, Order::Descending] {
                    let expect = run(&hand, &g, &input, *call, order);
                    let got = run(&kir, &g, &input, *call, order);
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
}

#[test]
fn the_shared_answer_is_attention_over_the_dequantized_cache() {
    for (g, calls) in cases() {
        let input = inputs(&g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            let got = out_f32(&run(&kir_ptx(&g, call.layer), &g, &input, call, Order::Ascending));
            let want = reference(&g, &input, call);
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                // The kernel's exp is 2^(x*log2 e) and it accumulates in a
                // different order from the reference; nothing else differs.
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
    let call = calls[5];
    let mem = run(&kir_ptx(&g, call.layer), &g, &input, call, Order::Descending);
    let q: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(mem[0] == q && mem[1] == input.pool, "the kernel wrote to its inputs");
}

#[test]
fn the_kir_kernels_keep_the_launch_abi_and_entry_names() {
    for (g, _) in cases() {
        for layer in 0..g.layers.len() as u32 {
            let (hand, kir) = (parse(&hand_ptx(&g, layer)), parse(&kir_ptx(&g, layer)));
            assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
            let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
            assert_eq!(names, ["q_ptr", "kv_base", "out_ptr", "slot_idx", "seq_len", "k_scale", "v_scale"]);
            assert!(kir_ptx(&g, layer).contains(&format!(".visible .entry nsl_cfie_decode_attn_l{layer}(")));
            assert_eq!(hand.shared_bytes, kir.shared_bytes);
            assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_hand_kernels_header_lines() {
    for (g, _) in cases() {
        for layer in 0..g.layers.len() as u32 {
            let header = |ptx: &str| -> Vec<String> {
                ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
            };
            assert_eq!(header(&hand_ptx(&g, layer)), header(&kir_ptx(&g, layer)), "{g:?} layer {layer}");
        }
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The multi-tile, ragged-tail case every mutation is judged on: layer 1,
/// both halves int8, so both scales are read. Its baked constants are
/// pairwise distinct — head_dim 8, token stride 16, per-slot 300, group
/// 2, tile 128, K half at byte 43200 and V half at 57600 — so each can be
/// nudged on its own.
fn mutation_case() -> (Geometry, Call) {
    (
        Geometry { n_heads: 4, n_kv_heads: 2, head_dim: 8, per_slot: 300, slots: 3, layers: vec![(F, I), (I, I), (I, F)] },
        Call { layer: 1, slot: 1, seq_len: 300 },
    )
}

/// Whether `mutant` is told apart from the hand kernel: under either
/// schedule its memory differs, or the interpreter faults on it.
fn caught(mutant: &str) -> bool {
    let (g, call) = mutation_case();
    let input = inputs(&g, 11);
    let hand = hand_ptx(&g, call.layer);
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, &g, &input, call, order);
        let (g, input) = (&g, &input);
        match std::panic::catch_unwind(|| run(mutant, g, input, call, order)) {
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

#[test]
fn the_mutation_case_is_what_it_claims() {
    let (g, call) = mutation_case();
    let o = pool_layout(&g.cfg())[call.layer as usize];
    assert_eq!((o.k_offset_bytes, o.v_offset_bytes), (43_200, 57_600));
    assert_eq!(g.layers[call.layer as usize], (I, I));
    // The baseline for every mutation below: without it, `caught` could
    // be reporting a difference the mutation did not cause.
    assert!(!caught(&kir_ptx(&g, call.layer)));
}

#[test]
fn deleting_any_one_barrier_is_caught() {
    let (g, call) = mutation_case();
    let ptx = kir_ptx(&g, call.layer);
    let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, 5, "q load, scores, softmax, tile end, l publish");
    for n in 0..barriers {
        assert!(caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let (g, call) = mutation_case();
    let ptx = kir_ptx(&g, call.layer);
    let o = pool_layout(&g.cfg())[call.layer as usize];
    let token_stride = (g.n_kv_heads * g.head_dim) as u64;
    let scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits());
    let nudged_scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits() + 1);
    for (from, to) in [
        (format!(", {};", o.k_offset_bytes), format!(", {};", o.k_offset_bytes + 1)),
        (format!(", {};", o.v_offset_bytes), format!(", {};", o.v_offset_bytes - 1)),
        (format!(", {};", token_stride), format!(", {};", token_stride + 1)),
        (format!(", {};", g.per_slot), format!(", {};", g.per_slot - 1)),
        (format!(", {scale};"), format!(", {nudged_scale};")),
    ] {
        let hits = ptx.lines().filter(|l| l.trim_start().starts_with("mov.") && l.ends_with(&from)).count();
        assert_eq!(hits, 1, "`{from}` should be materialised once");
        let mutant: String = ptx
            .lines()
            .map(|l| if l.trim_start().starts_with("mov.") && l.ends_with(&from) { l.replace(&from, &to) } else { l.to_string() })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(caught(&mutant), "`{from}` -> `{to}` went unnoticed");
    }
}

#[test]
fn swapping_the_dequant_scales_is_caught() {
    // Each int8 half must be multiplied by its own half's scale.
    let (g, call) = mutation_case();
    let ptx = kir_ptx(&g, call.layer);
    assert_eq!(ptx.matches("[param_k_scale]").count(), 1);
    assert_eq!(ptx.matches("[param_v_scale]").count(), 1);
    let mutant = ptx
        .replace("[param_k_scale]", "[param_SWAP]")
        .replace("[param_v_scale]", "[param_k_scale]")
        .replace("[param_SWAP]", "[param_v_scale]");
    assert!(caught(&mutant), "K and V dequantized with each other's scale went unnoticed");
}

#[test]
fn dropping_the_tail_tile_clamp_is_caught() {
    // `tcnt = min(seq_len - tile, TILE)` -> `tcnt = TILE`. (Pass 1's
    // `tok < seq_len` guard is an equivalent mutant here as in the base
    // kernel's gate, which says why.)
    let (g, call) = mutation_case();
    let ptx = kir_ptx(&g, call.layer);
    let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
    assert_eq!(clamps.len(), 1, "one tail clamp");
    let (dst, rest) = clamps[0].trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
    let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
    let mutant = ptx.replacen(clamps[0], &format!("    mov.u32 {dst}, {tile};"), 1);
    assert!(caught(&mutant), "the tail clamp removed went unnoticed");
}

#[test]
fn the_interpreter_sign_extends_int8() {
    // `ld.global.s8` of 0x80 is -128, not 128: a zero-extending model
    // would let a kernel that loads unsigned bytes pass the gate.
    let ptx = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry t(.param .u64 param_p, .param .u64 param_o) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    .reg .f32 %f<1>;
    ld.param.u64 %rd0, [param_p];
    ld.param.u64 %rd1, [param_o];
    ld.global.s8 %r0, [%rd0];
    cvt.rn.f32.s8 %f0, %r0;
    st.global.f32 [%rd1], %f0;
    ret;
}
";
    let prog = parse(ptx);
    let mut global = vec![Segment { base: 0x100, bytes: vec![0x80] }, Segment { base: 0x200, bytes: vec![0; 4] }];
    let args: HashMap<String, u64> = [("p".to_string(), 0x100), ("o".to_string(), 0x200)].into_iter().collect();
    let mut launch =
        Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid: 0, ctaid_y: 0, nctaid_y: 1, ntid: 1, steps: 0 };
    run_cta(&mut launch, Order::Ascending);
    assert_eq!(f32::from_le_bytes(global[1].bytes[..4].try_into().unwrap()), -128.0);
}
