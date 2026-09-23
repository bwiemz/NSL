//! A cooperative-CTA PTX interpreter for the decode-attention subset, and
//! the differential equivalence gate for roadmap A2 step 9.
//!
//! ## What this proves, and what it does not
//!
//! The migration replaces the hand-assembled `nsl_cfie_decode_attn` module
//! with one built from KIR. The new text must differ — registers and labels
//! come from the allocator, addresses are element indices through
//! `PtrOffset` rather than byte offsets, and shared memory is addressed
//! with 64-bit registers where the hand kernel used 32-bit ones — so the
//! gate executes both rather than comparing them (the spec's level 3).
//!
//! `hand::emit` is the pre-migration emitter itself, frozen verbatim in
//! `tests/fixtures/cfie_decode_attn_hand.rs`. The kernel is generated per
//! configuration (its strides are baked immediates), so the generator is
//! the fixture: a text fixture would freeze one geometry.
//!
//! Three properties are checked:
//!
//! 1. **Agreement** — for every geometry, sequence length, layer and slot
//!    below, and under two thread schedules, the hand module and the KIR
//!    module leave *the same bytes* in all of global memory.
//! 2. **Correctness** — those bytes are attention: they match
//!    `cfie_decode_attention::cpu_reference` fed the same f16 K/V.
//!    Agreement alone would be satisfied by two kernels wrong the same way.
//! 3. **The gate bites** — deleting any one barrier, nudging a baked
//!    stride or the softmax scale, or dropping the tail-tile clamp in the
//!    KIR text is caught.
//!
//! The CTA is executed cooperatively: each thread runs until it reaches a
//! `bar.sync` or `ret`, and a barrier releases only when every thread of
//! the CTA is waiting at one — a thread that exits while others wait is a
//! fault, as a `bar.sync` some threads never reach is a hang on hardware.
//! Every launch runs twice, threads visited in ascending and in descending
//! order: a kernel whose result depends on the order threads run between
//! barriers has a race, and the two schedules expose each barrier this
//! kernel relies on (the mutation tests below prove that, barrier by
//! barrier).
//!
//! The interpreter models `ex2.approx.f32` as `f32::exp2` and `cvt` with
//! `half`, so it does not prove the hardware's approximation — the device
//! parity suite (`cfie_decode_attn_gpu_parity.rs`) and the `ptxas` gate
//! carry fidelity to the machine. What it proves is what the migration put
//! at risk: same control flow, same addressing, same access widths, same
//! floating-point operations in the same order.
//!
//! Any mnemonic or operand form the interpreter does not know is a hard
//! error, never a skip; so is reading a register nothing wrote, and any
//! access outside the buffers a launch provides.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_decode_attention::{
    cpu_reference, emit_decode_attention_ptx, DecodeAttentionConfig,
};

#[allow(dead_code)]
#[path = "fixtures/cfie_decode_attn_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

// ---------------------------------------------------------------------------
// The kernel under test
// ---------------------------------------------------------------------------

const Q_BASE: u64 = 0x1000_0000;
const KV_BASE: u64 = 0x2000_0000;
const OUT_BASE: u64 = 0x3000_0000;
const BLOCK: u32 = 128;

#[derive(Debug, Clone, Copy)]
struct Geometry {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    per_slot: u32,
    slots: u32,
}

impl Geometry {
    fn cfg(self) -> DecodeAttentionConfig {
        DecodeAttentionConfig {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            kv_dtype_bytes: 2,
        }
    }

    fn hand_cfg(self) -> hand::DecodeAttentionConfig {
        hand::DecodeAttentionConfig {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            kv_dtype_bytes: 2,
            sm_version: 80,
        }
    }

    fn token_elems(self) -> usize {
        (self.n_kv_heads * self.head_dim) as usize
    }

    fn max_tokens(self) -> usize {
        (self.slots * self.per_slot) as usize
    }

    /// Element index of `(layer, plane, token, kv_head, d)` in the pool.
    fn kv_index(self, layer: u32, plane: u32, token: usize, kv_head: u32, d: u32) -> usize {
        (((layer as usize * 2 + plane as usize) * self.max_tokens() + token) * self.n_kv_heads as usize
            + kv_head as usize)
            * self.head_dim as usize
            + d as usize
    }

    fn pool_elems(self) -> usize {
        self.n_layers as usize * 2 * self.max_tokens() * self.token_elems()
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

struct Inputs {
    q: Vec<f32>,
    /// The whole pool, every layer and slot filled, so an address that
    /// strays into a neighbouring layer, plane, slot or head reads a
    /// different value rather than a zero.
    pool: Vec<f16>,
}

fn inputs(g: Geometry, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let q = (0..g.n_heads * g.head_dim).map(|_| rng.next()).collect();
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
fn run(ptx: &str, g: Geometry, input: &Inputs, call: Call, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let q_bytes: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv_bytes: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    // A NaN sentinel: an output element the kernel fails to write shows.
    let out_bytes: Vec<u8> = (0..g.n_heads * g.head_dim).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect();
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

fn kir_ptx(g: Geometry) -> String {
    emit_decode_attention_ptx(&g.cfg())
}

fn hand_ptx(g: Geometry) -> String {
    hand::emit_decode_attention_ptx(&g.hand_cfg())
}

fn out_f32(mem: &[Vec<u8>]) -> Vec<f32> {
    mem[2].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What attention over the slot's first `seq_len` tokens produces, from
/// `cpu_reference` fed the same f16 values the kernel reads.
fn reference(g: Geometry, input: &Inputs, call: Call) -> Vec<f32> {
    let first = (call.slot * g.per_slot) as usize;
    let rows = |plane: u32| -> Vec<f32> {
        let mut out = Vec::new();
        for t in 0..call.seq_len as usize {
            for h in 0..g.n_kv_heads {
                for d in 0..g.head_dim {
                    out.push(input.pool[g.kv_index(call.layer, plane, first + t, h, d)].to_f32());
                }
            }
        }
        out
    };
    cpu_reference(&input.q, &rows(0), &rows(1), g.n_heads, g.n_kv_heads, g.head_dim, call.seq_len)
}

/// The geometries and calls the agreement and correctness gates sweep.
fn cases() -> Vec<(Geometry, Vec<Call>)> {
    vec![
        // GQA (group 2), a mid-pool layer and slot, and sequence lengths
        // that cross every tile edge: empty, one token, one short of a
        // tile, exactly one, one over, and three tiles with a ragged tail.
        (
            Geometry { n_layers: 2, n_heads: 2, n_kv_heads: 1, head_dim: 8, per_slot: 300, slots: 3 },
            [0, 1, 127, 128, 129, 300]
                .into_iter()
                .map(|seq_len| Call { layer: 1, slot: 1, seq_len })
                .chain([Call { layer: 0, slot: 2, seq_len: 77 }, Call { layer: 1, slot: 0, seq_len: 256 }])
                .collect(),
        ),
        // MHA with a head_dim that is not a power of two: threads 40..128
        // compute scores but own no output element.
        (
            Geometry { n_layers: 1, n_heads: 3, n_kv_heads: 3, head_dim: 40, per_slot: 200, slots: 1 },
            vec![Call { layer: 0, slot: 0, seq_len: 5 }, Call { layer: 0, slot: 0, seq_len: 200 }],
        ),
        // head_dim == block: every thread owns an output element.
        (
            Geometry { n_layers: 1, n_heads: 1, n_kv_heads: 1, head_dim: 128, per_slot: 130, slots: 1 },
            vec![Call { layer: 0, slot: 0, seq_len: 130 }],
        ),
        // head_dim 1, group 4.
        (
            Geometry { n_layers: 3, n_heads: 4, n_kv_heads: 1, head_dim: 1, per_slot: 129, slots: 2 },
            vec![Call { layer: 2, slot: 1, seq_len: 129 }],
        ),
    ]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for (g, calls) in cases() {
        let (hand, kir) = (hand_ptx(g), kir_ptx(g));
        let input = inputs(g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, g, &input, call, order);
                let got = run(&kir, g, &input, call, order);
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
fn the_shared_answer_is_attention() {
    for (g, calls) in cases() {
        let kir = kir_ptx(g);
        let input = inputs(g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            let got = out_f32(&run(&kir, g, &input, call, Order::Ascending));
            let want = reference(g, &input, call);
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
    let input = inputs(g, 7);
    let mem = run(&kir_ptx(g), g, &input, calls[5], Order::Descending);
    let q: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    assert!(mem[0] == q && mem[1] == kv, "the kernel wrote to its inputs");
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    for (g, _) in cases() {
        let (hand, kir) = (parse(&hand_ptx(g)), parse(&kir_ptx(g)));
        assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
        let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["q_ptr", "kv_base", "out_ptr", "layer_idx", "slot_idx", "seq_len"]);
        assert!(kir_ptx(g).contains(".visible .entry nsl_cfie_decode_attn("));
        // Same shared footprint, one static block.
        let (hand_smem, kir_smem) = (hand.shared_bytes, kir.shared_bytes);
        assert_eq!(hand_smem, kir_smem);
        assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
    }
}

#[test]
fn the_kir_kernel_keeps_the_hand_kernels_baked_header_lines() {
    // Sibling CFIE kernels compare these lines with the decode kernel's
    // byte for byte (see cfie_persistent_ptx / cfie_speculative_ptx).
    for (g, _) in cases() {
        let header = |ptx: &str| -> Vec<String> {
            ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
        };
        assert_eq!(header(&hand_ptx(g)), header(&kir_ptx(g)), "{g:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The multi-tile, ragged-tail case every mutation is judged on. Its
/// baked constants are pairwise distinct (head_dim 8, token stride 16,
/// per-slot 300, kv half 14400, layer 28800, group 2, tile 128), so each
/// can be nudged on its own.
fn mutation_case() -> (Geometry, Call) {
    (
        Geometry { n_layers: 2, n_heads: 4, n_kv_heads: 2, head_dim: 8, per_slot: 300, slots: 3 },
        Call { layer: 1, slot: 1, seq_len: 300 },
    )
}

/// Whether `mutant` is told apart from the hand kernel: under either
/// schedule its memory differs, or the interpreter faults on it.
fn caught(mutant: &str) -> bool {
    let (g, call) = mutation_case();
    let input = inputs(g, 11);
    let hand = hand_ptx(g);
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, g, &input, call, order);
        let input = &input;
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

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below: without it, `caught` could
    // be reporting a difference the mutation did not cause.
    let (g, _) = mutation_case();
    assert!(!caught(&kir_ptx(g)));
}

#[test]
fn deleting_any_one_barrier_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, 5, "q load, scores, softmax, tile end, l publish");
    for n in 0..barriers {
        assert!(caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let cfg = g.cfg();
    let s = nsl_codegen::cfie_decode_attention::kv_strides(&cfg);
    let scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits());
    let nudged_scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits() + 1);
    for (from, to) in [
        (format!(", {};", s.token_stride), format!(", {};", s.token_stride + 1)),
        (format!(", {};", s.kv_half_stride), format!(", {};", s.kv_half_stride - 1)),
        (format!(", {};", s.layer_stride), format!(", {};", s.layer_stride + 1)),
        (format!(", {};", cfg.per_slot_max_tokens), format!(", {};", cfg.per_slot_max_tokens - 1)),
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
fn dropping_the_tail_tile_clamp_is_caught() {
    // `tcnt = min(seq_len - tile, TILE)` -> `tcnt = TILE`: the ragged last
    // tile's softmax and P*V then run over scores no thread wrote.
    //
    // The other tail guard, pass 1's `tok < seq_len`, is deliberately not
    // mutated: loosening or deleting it is an *equivalent* mutant. The
    // extra scores it lets through land at `scores[tcnt..]`, which every
    // later loop bounds away, and the extra K rows it reads stay inside
    // the pool (a K row past the plane's last token is the V plane's
    // first). It is a guard against wasted work, not a correctness one,
    // and no execution can tell it apart — so a test claiming to would be
    // vacuous. The module's own structural test pins that it is present.
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
    assert_eq!(clamps.len(), 1, "one tail clamp");
    let (dst, rest) = clamps[0].trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
    let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
    let mutant = ptx.replacen(clamps[0], &format!("    mov.u32 {dst}, {tile};"), 1);
    assert!(caught(&mutant), "the tail clamp removed went unnoticed");
}

#[test]
fn the_interpreter_refuses_an_instruction_it_does_not_model() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g).replacen("bar.sync 0;", "membar.cta;", 1);
    let err = std::panic::catch_unwind(|| parse(&ptx)).expect_err("an unknown mnemonic must not parse");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("does not know this instruction"), "{msg}");
}

#[test]
fn the_interpreter_faults_on_an_undefined_register() {
    let (g, call) = mutation_case();
    let input = inputs(g, 3);
    // Drop the load of `seq_len`: its first reader must fault, not see 0.
    let ptx: String = kir_ptx(g)
        .lines()
        .filter(|l| !l.contains("[param_seq_len]"))
        .map(|l| format!("{l}\n"))
        .collect();
    let err = std::panic::catch_unwind(|| run(&ptx, g, &input, call, Order::Ascending))
        .expect_err("an undefined register must fault");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("before anything wrote it"), "{msg}");
}
