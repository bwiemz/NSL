//! The fused SDPA backward (PCA Stage C, `synthesize_flash_attention_backward_ptx`)
//! executed on the CPU, instruction by instruction, against f64 gradients —
//! no GPU needed. The forward's twin is `sdpa_fused_forward_interp.rs`,
//! which found the KV-loop race #735 fixed; this file asks the same
//! question of the backward the packed path pairs it with.
//!
//! It runs the production PTX — the two modules `ensure_sdpa_bwd_variant_table`
//! embeds for `(causal, segment_masked)` at head_dim 32 — on
//! `support/cta_ptx_interp.rs`, launched as `nsl_flash_attention_backward`
//! launches them:
//!
//!   * phase 1 (`D = rowsum(dO ∘ O)`): grid `[b*h, s / block_q]`, `block_q`
//!     threads;
//!   * phase 2 (dQ/dK/dV): grid `[b*h, s / block_kv]`, `block_q` threads (no
//!     `_w<N>` suffix on a segment-masked kernel), the runtime's dynamic
//!     shared-memory request (which always reserves the MMA dP tile), dQ
//!     zeroed because the kernel accumulates it with global atomics.
//!
//! Two kernels: the packed (segment-masked) variant Stage C pairs with the
//! fused forward, whose body is single-warp but launches as `block_q / 32`
//! warps, and the unmasked causal variant every decorator-free SDPA
//! backward uses, whose body is partitioned across 4 warps (`_w4`). The
//! first is the one this gate caught: the 4-warp partition (4f8336ca)
//! started each warp's m-loop at its warp id but kept a stride of 1 for the
//! single-warp body, so its second warp repeated m-tiles 1..3 and their dQ,
//! dK and dV atomics landed twice — gradients off by ~100% on every packed
//! step at head_dim <= 32 on sm_80+.
//!
//! O and the logsumexp are the f64 forward's, rounded to f32: the forward
//! is `sdpa_fused_forward_interp.rs`'s business, and feeding the exact
//! values makes every error below the backward's own.
//!
//! Two SMs, two code paths. At sm_75 the backward is scalar f32. From sm_80
//! it runs its five matmuls (S = QKᵀ, dP = dO·Vᵀ, dV += Pᵀ·dO, dQ += dS·K,
//! dK += dSᵀ·Q) on `mma.sync.m16n8k16` with f16 operands, which the
//! interpreter models with the ISA's fragment layout (see its module docs).
//! Two oracles, both f64:
//!
//!   * **exact** — the gradients of causal (and same-document) softmax
//!     attention on the f32 inputs;
//!   * **f16-operand** — the same, with each MMA operand rounded to f16
//!     where the kernel rounds it (Q, K, V, dO, P and dS).
//!
//! The scalar path must match the exact oracle to f32 accuracy; the MMA
//! path must match the f16-operand oracle to f32 accumulation noise, and
//! miss the exact one by f16-sized errors — the same storage story the
//! forward tells, pinned so the change that gives f32 programs f32
//! operands has to flip it deliberately.
//!
//! `ex2.approx` runs as `exp2` on the interpreter: this gate is about the
//! kernel's dataflow, synchronisation and operand precision, not the
//! hardware's approximations.

#[path = "support/cta_ptx_interp.rs"]
#[allow(dead_code)]
mod cta_ptx_interp;

use std::collections::HashMap;

use half::f16;

use cta_ptx_interp::{parse, run_cta, Launch, Order, Program, Segment, SHARED_BASE};
use nsl_codegen::flash_attention::{
    backward_select_blocks, flash_attention_bwd_main_kernel_name, synthesize_flash_attention_backward_ptx,
    FlashAttentionBackwardConfig,
};

/// Every schedule the interpreter offers: thread-interleaved both ways, and
/// warp-serial both ways (one warp runs a whole barrier interval ahead).
const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

const B: usize = 2;
const H: usize = 2;
const S: usize = 128;
const D: usize = 32;

fn config(gpu_sm: u32, segment_masked: bool) -> FlashAttentionBackwardConfig {
    let (block_q, block_kv) = backward_select_blocks(D as i64);
    FlashAttentionBackwardConfig { block_q, block_kv, head_dim: D as i64, causal: true, gpu_sm, segment_masked }
}

/// Deterministic values in about [-2, 2) (the forward gate's generator).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as u32 as f64 / u32::MAX as f64) * 4.0 - 2.0) as f32
        })
        .collect()
}

/// Packed 20-token documents at a different phase per batch row, so every
/// row's documents straddle the 64-wide tiles (the forward gate's layout).
/// Unmasked, every token is in one document: plain causal attention.
fn segments(segment_masked: bool) -> Vec<u16> {
    (0..B).flat_map(|b| (0..S).map(move |i| if segment_masked { ((i + 7 * b) / 20) as u16 } else { 0 })).collect()
}

fn f16_round(v: f64) -> f64 {
    f16::from_f64(v).to_f64()
}

struct Grads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// The forward (O, natural logsumexp) and the gradients, in f64. With
/// `staged`, every operand of the kernel's five MMA products is rounded to
/// f16 first: Q and K in S, dO and V in dP, P and dO in dV, dS and K in dQ,
/// dS and Q in dK. O, D and the logsumexp stay exact — the kernel reads
/// them in f32.
fn oracle(q: &[f32], k: &[f32], v: &[f32], dout: &[f32], seg: &[u16], staged: bool) -> (Vec<f32>, Vec<f32>, Grads) {
    let r = |x: f64| if staged { f16_round(x) } else { x };
    let scale = 1.0 / (D as f64).sqrt();
    let n = B * H * S * D;
    let (mut out, mut lse) = (vec![0f32; n], vec![0f32; B * H * S]);
    let (mut dq, mut dk, mut dv) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for b in 0..B {
        for h in 0..H {
            let base = (b * H + h) * S * D;
            let at = |x: &[f32], i: usize, d: usize| x[base + i * D + d] as f64;
            for i in 0..S {
                let vis: Vec<usize> = (0..=i).filter(|&j| seg[b * S + i] == seg[b * S + j]).collect();
                // The forward is exact: the backward reads O and lse as data.
                let s_exact: Vec<f64> =
                    vis.iter().map(|&j| (0..D).map(|d| at(q, i, d) * at(k, j, d)).sum::<f64>() * scale).collect();
                let m = s_exact.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l = m + s_exact.iter().map(|s| (s - m).exp()).sum::<f64>().ln();
                lse[(b * H + h) * S + i] = l as f32;
                let o: Vec<f64> = (0..D)
                    .map(|d| vis.iter().zip(&s_exact).map(|(&j, s)| (s - l).exp() * at(v, j, d)).sum())
                    .collect();
                for d in 0..D {
                    out[base + i * D + d] = o[d] as f32;
                }
                // The backward, from the f32 O and lse the kernel reads.
                let (o32, l32): (Vec<f64>, f64) = (o.iter().map(|&x| x as f32 as f64).collect(), l as f32 as f64);
                let dd: f64 = (0..D).map(|d| at(dout, i, d) * o32[d]).sum();
                for &j in &vis {
                    let s = (0..D).map(|d| r(at(q, i, d)) * r(at(k, j, d))).sum::<f64>() * scale;
                    let p = (s - l32).exp();
                    let dp: f64 = (0..D).map(|d| r(at(dout, i, d)) * r(at(v, j, d))).sum();
                    let ds = p * (dp - dd);
                    for d in 0..D {
                        dv[base + j * D + d] += r(p) * r(at(dout, i, d));
                        dq[base + i * D + d] += r(ds) * r(at(k, j, d)) * scale;
                        dk[base + j * D + d] += r(ds) * r(at(q, i, d)) * scale;
                    }
                }
            }
        }
    }
    let f = |x: Vec<f64>| x.into_iter().map(|v| v as f32).collect();
    (out, lse, Grads { dq: f(dq), dk: f(dk), dv: f(dv) })
}

const DOUT_BASE: u64 = 0x1000_0000;
const Q_BASE: u64 = 0x1800_0000;
const K_BASE: u64 = 0x2000_0000;
const V_BASE: u64 = 0x2800_0000;
const DQ_BASE: u64 = 0x3000_0000;
const DK_BASE: u64 = 0x3800_0000;
const DV_BASE: u64 = 0x4000_0000;
const DVEC_BASE: u64 = 0x4800_0000;
const LSE_BASE: u64 = 0x5000_0000;
const SEG_BASE: u64 = 0x5800_0000;
const OUT_BASE: u64 = 0x6000_0000;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn module(ptx: &[u8]) -> String {
    std::str::from_utf8(ptx).expect("PTX is UTF-8").trim_end_matches('\0').to_string()
}

/// The phase-2 thread count as `nsl_flash_attention_backward` derives it:
/// `32 * N` for a `_w<N>` kernel name, else `block_q`.
fn phase2_threads(config: &FlashAttentionBackwardConfig) -> u32 {
    let name = flash_attention_bwd_main_kernel_name(config);
    match name.rfind("_w").and_then(|at| name[at + 2..].parse::<u32>().ok()) {
        Some(w) => 32 * w,
        None => config.block_q as u32,
    }
}

/// `nsl_flash_attention_backward`'s phase-2 dynamic shared request: always
/// the MMA-inclusive layout (Q, dO, K, V tiles, S and dP tiles, D and L
/// vectors), plus the dK/dV tiles unless the body is multi-warp (`_w<N>`,
/// which keeps dK/dV in registers).
fn runtime_shmem(config: &FlashAttentionBackwardConfig) -> usize {
    let hd_padded = D + 4;
    let tile = |rows: i64, cols: usize| rows as usize * cols * 4;
    let (bq, bkv) = (config.block_q, config.block_kv);
    let dkdv = if phase2_threads(config) > bq as u32 { 0 } else { tile(bkv, hd_padded) * 2 };
    tile(bkv, hd_padded) * 2 + tile(bq, hd_padded) * 2 + dkdv + tile(bq, bkv as usize) * 2 + bq as usize * 8
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dout: Vec<f32>,
    out: Vec<f32>,
    lse: Vec<f32>,
    seg: Vec<u16>,
}

fn inputs(segment_masked: bool) -> (Inputs, Grads, Grads) {
    let n = B * H * S * D;
    let (q, k, v, dout) = (values(1, n), values(2, n), values(3, n), values(4, n));
    let seg = segments(segment_masked);
    let (out, lse, exact) = oracle(&q, &k, &v, &dout, &seg, false);
    let (_, _, staged) = oracle(&q, &k, &v, &dout, &seg, true);
    (Inputs { q, k, v, dout, out, lse, seg }, exact, staged)
}

/// Launch both phases as the runtime does; return the gradients.
fn run(config: &FlashAttentionBackwardConfig, p1: &Program, p2: &Program, x: &Inputs, order: Order) -> Grads {
    let n = B * H * S * D;
    let mut global = vec![
        Segment { base: DOUT_BASE, bytes: f32_bytes(&x.dout) },
        Segment { base: Q_BASE, bytes: f32_bytes(&x.q) },
        Segment { base: K_BASE, bytes: f32_bytes(&x.k) },
        Segment { base: V_BASE, bytes: f32_bytes(&x.v) },
        // dQ is accumulated with atomics into the runtime's zeroed buffer;
        // dK and dV are stored, so they start poisoned (NaN) and an element
        // the kernel never writes shows.
        Segment { base: DQ_BASE, bytes: vec![0; n * 4] },
        Segment { base: DK_BASE, bytes: vec![0xFF; n * 4] },
        Segment { base: DV_BASE, bytes: vec![0xFF; n * 4] },
        Segment { base: DVEC_BASE, bytes: vec![0xFF; B * H * S * 4] },
        Segment { base: LSE_BASE, bytes: f32_bytes(&x.lse) },
        Segment { base: SEG_BASE, bytes: x.seg.iter().flat_map(|s| s.to_le_bytes()).collect() },
        Segment { base: OUT_BASE, bytes: f32_bytes(&x.out) },
    ];
    let (bq, bkv) = (config.block_q as u32, config.block_kv as u32);
    let bind = |prog: &Program, set: &[(&str, u64)]| -> HashMap<String, u64> {
        let mut args: HashMap<String, u64> = prog.params.iter().map(|(_, n)| (n.clone(), 0)).collect();
        for (name, v) in set {
            assert!(args.contains_key(*name), "the kernel has no parameter `{name}`");
            args.insert(name.to_string(), *v);
        }
        assert_eq!(args.len(), set.len(), "every parameter is bound: {:?}", prog.params);
        args
    };
    let mut launch = |prog: &Program, args: &HashMap<String, u64>, smem: usize, grid_y: u32, threads: u32| {
        for bh in 0..(B * H) as u32 {
            for y in 0..grid_y {
                let mut l = Launch {
                    prog,
                    args,
                    global: &mut global,
                    // Poisoned: a shared read no thread wrote shows as NaN.
                    shared: vec![0xFF; smem],
                    ctaid: bh,
                    ctaid_y: y,
                    nctaid_y: grid_y,
                    ntid: threads,
                    steps: 0,
                };
                run_cta(&mut l, order);
            }
        }
    };

    let a1 = bind(
        p1,
        &[("dout_ptr", DOUT_BASE), ("out_ptr", OUT_BASE), ("d_ptr", DVEC_BASE), ("seq_len", S as u64), ("head_dim", D as u64)],
    );
    launch(p1, &a1, p1.shared_bytes, S as u32 / bq, bq);

    let a2 = bind(
        p2,
        &[
            ("dout", DOUT_BASE),
            ("q", Q_BASE),
            ("k", K_BASE),
            ("v", V_BASE),
            ("dq", DQ_BASE),
            ("dk", DK_BASE),
            ("dv", DV_BASE),
            ("d", DVEC_BASE),
            ("lse", LSE_BASE),
            ("scale", (1.0f32 / (D as f32).sqrt()).to_bits() as u64),
            ("seq_len", S as u64),
            ("head_dim", D as u64),
        ]
        .into_iter()
        .chain(config.segment_masked.then_some([("segment_ids", SEG_BASE), ("heads", H as u64)]).into_iter().flatten())
        .collect::<Vec<_>>(),
    );
    let smem = p2.shared_bytes + runtime_shmem(config);
    assert!(SHARED_BASE as usize + smem < DOUT_BASE as usize);
    launch(p2, &a2, smem, S as u32 / bkv, phase2_threads(config));

    Grads { dq: f32s(&global[4].bytes), dk: f32s(&global[5].bytes), dv: f32s(&global[6].bytes) }
}

/// Phase 1 and phase 2 parsed, plus phase 2's text (for mutants). An
/// unmasked module also carries the native-GQA entry after the MHA one;
/// only the MHA entry — the one an `h == kv_h` launch names — is kept.
fn programs(config: &FlashAttentionBackwardConfig) -> (Program, Program, String) {
    let (p1, p2) = synthesize_flash_attention_backward_ptx(config);
    let mut p2 = module(&p2);
    let entries: Vec<usize> = p2.match_indices(".visible .entry").map(|(at, _)| at).collect();
    assert_eq!(entries.len(), if config.segment_masked { 1 } else { 2 }, "phase-2 entries");
    let name = flash_attention_bwd_main_kernel_name(config);
    assert!(p2[entries[0]..].starts_with(&format!(".visible .entry {name} (")), "the MHA entry is first");
    if let Some(&gqa) = entries.get(1) {
        p2.truncate(gqa);
    }
    (parse(&module(&p1)), parse(&p2), p2)
}

/// Largest error relative to the largest reference magnitude.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale
}

struct Report {
    dq: f32,
    dk: f32,
    dv: f32,
}

impl Report {
    fn of(got: &Grads, want: &Grads) -> Self {
        Report { dq: rel(&got.dq, &want.dq), dk: rel(&got.dk, &want.dk), dv: rel(&got.dv, &want.dv) }
    }
    fn max(&self) -> f32 {
        self.dq.max(self.dk).max(self.dv)
    }
}

fn check(config: &FlashAttentionBackwardConfig, p2: &Program, order: Order) -> (Report, Report) {
    let (p1, _, _) = programs(config);
    let (x, exact, staged) = inputs(config.segment_masked);
    let got = run(config, &p1, p2, &x, order);
    let all = got.dq.iter().chain(&got.dk).chain(&got.dv);
    assert!(all.clone().all(|v| v.is_finite()), "an unwritten or non-finite gradient");
    (Report::of(&got, &exact), Report::of(&got, &staged))
}

fn measure(gpu_sm: u32, segment_masked: bool, order: Order) -> (Report, Report) {
    let config = config(gpu_sm, segment_masked);
    let (_, p2, _) = programs(&config);
    let (e, s) = check(&config, &p2, order);
    eprintln!(
        "{} (sm_{gpu_sm}, {} threads, {order:?}): vs exact dq {:.2e} dk {:.2e} dv {:.2e}; \
         vs f16-operand dq {:.2e} dk {:.2e} dv {:.2e}",
        flash_attention_bwd_main_kernel_name(&config),
        phase2_threads(&config),
        e.dq,
        e.dk,
        e.dv,
        s.dq,
        s.dk,
        s.dv
    );
    (e, s)
}

/// The scalar path is the exact gradient to f32 accuracy.
#[test]
fn scalar_backward_is_the_attention_gradient() {
    for order in ORDERS {
        let (exact, _) = measure(75, true, order);
        assert!(exact.max() < 2e-5, "sm_75 scalar backward: {:.2e} from the exact gradients", exact.max());
    }
}

/// The MMA path is the f16-operand gradient. dV (P and dO rounded, f32
/// accumulation) agrees to f32 noise; dQ and dK go through dS, which the
/// kernel forms in f32 and rounds to f16, so f32-level differences in dS
/// can flip one f16 rounding — a few 1e-4 of the largest gradient at most.
/// Against the exact gradient all three miss by f16-sized errors (the
/// storage story the forward gate tells), and nowhere near the ~100% the
/// double-counted m-tiles produced.
fn assert_is_f16_operand_gradient(name: &str, exact: &Report, staged: &Report) {
    assert!(staged.dv < 2e-5, "{name}: dV {:.2e} from the f16-operand oracle", staged.dv);
    assert!(staged.dq.max(staged.dk) < 5e-4, "{name}: dQ/dK {:.2e} from the f16-operand oracle", staged.dq.max(staged.dk));
    assert!(exact.max() < 5e-3, "{name}: {:.2e} from the exact gradients", exact.max());
    assert!(exact.max() > 1e-4, "{name}: within {:.2e} of exact — no longer f16 operands?", exact.max());
}

#[test]
fn packed_mma_backward_is_the_f16_operand_gradient() {
    for order in ORDERS {
        let (exact, staged) = measure(90, true, order);
        assert_is_f16_operand_gradient("packed sm_90", &exact, &staged);
    }
}

#[test]
fn unmasked_multiwarp_backward_is_the_f16_operand_gradient() {
    assert_eq!(phase2_threads(&config(90, false)), 128, "the unmasked hd32 body is the 4-warp one");
    for order in ORDERS {
        let (exact, staged) = measure(90, false, order);
        assert_is_f16_operand_gradient("unmasked sm_90", &exact, &staged);
    }
}

/// The agreement above is not vacuous: each named break in the packed MMA
/// kernel moves a gradient far from the f16-operand oracle (or faults the
/// interpreter) under at least one schedule. Every occurrence of the
/// pattern is replaced.
///
/// Five of the kernel's ten fences are not here because the next fence
/// covers them — nothing between the two reads what the first protects:
/// "K and V tiles loaded" and "dK/dV zeroed" (the next reads are after
/// "Q, dO, D, L loaded"), "P stored in S_tile" (the dP MMA reads dO and V;
/// P is next read after "dP_tile fully computed"), "dP_tile fully computed"
/// (dP is next read after "dV accumulation done") and "dQ accumulation
/// done" (the dK MMA reads the same dS and Q the dQ MMA did). Removing one
/// of them alone changes nothing, on this model or on hardware.
#[test]
fn packed_kernel_mutants_are_caught() {
    let config = config(90, true);
    let (_, _, text) = programs(&config);
    let barrier = |what: &str| format!("    bar.sync 0;  // {what}\n");
    let mutants: Vec<(&str, Vec<(String, String)>)> = vec![
        (
            // The regression: the two-warp launch walking the m-tiles with the
            // single-warp stride, so warp 1 repeats tiles 1..3.
            "m-loops stride 1 on the two-warp launch (4f8336ca)",
            vec![
                ("add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 2;".into(), "add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;".into()),
                ("add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, 4608;".into(), "add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, 2304;".into()),
                ("add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, 8192;".into(), "add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, 4096;".into()),
            ],
        ),
        (
            "segment mask ignored (a query's own id compared with itself)",
            vec![("setp.ne.u32 %p_seg, %r_seg_i, %r_seg_j;".into(), "setp.ne.u32 %p_seg, %r_seg_i, %r_seg_i;".into())],
        ),
        ("D correction dropped (dS = P * dP)", vec![("sub.f32 %f_tmp, %f_dp, %f_d_val;".into(), "mov.f32 %f_tmp, %f_dp;".into())]),
        ("fence after the Q/dO/D/L loads removed", vec![(barrier("Q, dO, D, L loaded"), String::new())]),
        ("fence after the S MMA removed", vec![(barrier("S_tile fully computed via MMA"), String::new())]),
        ("fence after the dV MMA removed (dS overwrites P under it)", vec![(barrier("dV accumulation done"), String::new())]),
        ("fence after dS removed", vec![(barrier("dS stored in S_tile"), String::new())]),
        ("fence at the end of the q-tile loop removed", vec![(barrier("all steps 3a-3g done (MMA path)"), String::new())]),
    ];
    for (what, edits) in &mutants {
        let mut mutated = text.clone();
        for (from, to) in edits {
            assert!(mutated.contains(from.as_str()), "mutant `{what}`: `{from}` is not in the kernel — resync the gate");
            mutated = mutated.replace(from.as_str(), to);
        }
        let prog = parse(&mutated);
        let caught = ORDERS.iter().any(|&order| {
            std::panic::catch_unwind(|| {
                let (exact, staged) = check(&config, &prog, order);
                eprintln!("mutant `{what}` ({order:?}): {:.2e} from exact, {:.2e} from f16-operand", exact.max(), staged.max());
                staged.max() > 1e-2
            })
            .unwrap_or(true)
        });
        assert!(caught, "mutant `{what}` survived every schedule");
    }
}
