//! The fused SDPA forward (PCA Stage C) executed on the CPU, instruction by
//! instruction, against f64 oracles — no GPU needed.
//!
//! `stage_c_packed_parity.rs`'s GPU gate compares a packed program trained
//! with the fused segment-masked flash forward against the same program on
//! the decomposed matmul/softmax chain, and accepts a checkpoint difference
//! of 2e-2. That number cannot fail on numerics: the fixture trains with
//! AdamW at lr 2e-3 for 8 optimizer steps, and Adam moves each parameter by
//! at most about lr per step, so two runs cannot drift more than ~1.6e-2
//! apart whatever the kernels compute. This file finds out what the
//! difference actually is.
//!
//! It runs the production PTX — `emit_tier_b_variants_for_config`, which is
//! what `ensure_sdpa_fwd_variant_table` embeds, every variant it emits — on
//! `support/cta_ptx_interp.rs`, launched as `nsl_sdpa_fused_forward`
//! launches it: 128 threads, grid `[seq / block_q, heads]`, one launch per
//! batch row with every pointer pre-offset to the row. Two oracles, both in
//! f64:
//!
//!   * **exact** — causal (and, packed, same-document) softmax attention on
//!     the f32 inputs, the function an f32 program asks for;
//!   * **f16-staged** — the same, with K and V first rounded to f16 and the
//!     output rounded to f16 at the end.
//!
//! The kernel reads f32 Q/K/V from global memory. Each thread keeps its row
//! of Q in f32 registers, but K and V are rounded to f16 when they are
//! staged in shared memory (`cvt.rn.f16.f32`, `st.shared.b16`), and the
//! output tile is stored as f16 (`finalize.rs`); the runtime then widens
//! that output back to f32. Everything else is f32 arithmetic. So the
//! kernel's output is within one f16 rounding of the f16-staged oracle and
//! its logsumexp within f32 noise of it, and both miss the exact oracle by
//! f16-sized errors: the whole fused-vs-decomposed differential is the storage format,
//! not a kernel bug, and an f32 program gets f16-precision attention
//! whenever the fused path fires. The assertions below pin both halves, so
//! the change that gives f32 programs an f32-storage kernel has to flip the
//! second one deliberately.
//!
//! Scope: the SEGMENT-MASKED kernels, i.e. the packed path Stage C
//! dispatches (both variants, on every SM — neither uses tensor cores).
//! The unmasked causal configuration synthesises the `mma.sync` m16n8k16
//! f16 tensor-core forward instead; its f16 operands are the instruction's,
//! not a staging choice, and the interpreter does not model MMA fragments.
//!
//! `ex2.approx`, `lg2.approx` and `rcp.approx` run as their exact
//! counterparts on the interpreter (see its module docs): this gate is about
//! the kernel's dataflow and storage, not the hardware's approximations.

#[path = "support/cta_ptx_interp.rs"]
#[allow(dead_code)]
mod cta_ptx_interp;

use std::collections::HashMap;

use half::f16;

use cta_ptx_interp::{parse, run_cta, Launch, Order, Program, Segment, SHARED_BASE};
use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::shared_mem_bytes_selected;
use nsl_codegen::pca_tier_b::emit_tier_b_variants_for_config;

const B: usize = 2;
const H: usize = 2;
const S: usize = 128;
const D: usize = 32;
const BLOCK_Q: usize = 64;
const THREADS: u32 = 128;

/// The configuration the Stage-C fixture's fused dispatch compiles
/// (head_dim 32, 64x64 tiles, causal), as `sdpa_fused_forward_gpu_parity.rs`
/// builds it.
fn config(segment_masked: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: BLOCK_Q as i64,
        block_kv: 64,
        head_dim: D as i64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 90,
        segment_masked,
        csha: None,
        checkpoint: None,
    }
}

/// Deterministic values in about [-2, 2): large enough that the scores
/// spread the softmax, so the output is not a near-uniform average.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as u32 as f64 / u32::MAX as f64) * 4.0 - 2.0) as f32
        })
        .collect()
}

/// Packed documents of 20 tokens, at a different phase per batch row so
/// rows 0 and 1 disagree (the per-row launch is what keeps row 1 from being
/// masked with row 0's ids). With 64-wide KV tiles a document straddles the
/// tile boundary in every row, so the online softmax's cross-tile rescale
/// runs on real data — shorter documents (the Stage-C fixture's 8) leave it
/// nearly idle, and a dropped rescale would go unseen.
fn segments() -> Vec<u16> {
    (0..B).flat_map(|b| (0..S).map(move |i| ((i + 7 * b) / 20) as u16)).collect()
}

fn f16_round(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}

/// f64 attention oracle. With `staged`, K and V are rounded to f16 first and
/// the output is rounded to f16 last — the kernel's storage (Q stays f32:
/// the kernel keeps its row of Q in f32 registers). Returns
/// `(out [B,H,S,D], lse [B,H,S])`, lse the natural logsumexp of each row's
/// scaled visible scores.
fn oracle(q: &[f32], k: &[f32], v: &[f32], seg: Option<&[u16]>, staged: bool) -> (Vec<f32>, Vec<f32>) {
    let load = |x: f32| if staged { f16_round(x) as f64 } else { x as f64 };
    let scale = 1.0 / (D as f64).sqrt();
    let mut out = vec![0f32; B * H * S * D];
    let mut lse = vec![0f32; B * H * S];
    for b in 0..B {
        for h in 0..H {
            let base = (b * H + h) * S * D;
            for i in 0..S {
                let visible: Vec<usize> = (0..=i)
                    .filter(|&j| seg.is_none_or(|s| s[b * S + i] == s[b * S + j]))
                    .collect();
                let score: Vec<f64> = visible
                    .iter()
                    .map(|&j| (0..D).map(|d| q[base + i * D + d] as f64 * load(k[base + j * D + d])).sum::<f64>() * scale)
                    .collect();
                let m = score.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let denom: f64 = score.iter().map(|s| (s - m).exp()).sum();
                for d in 0..D {
                    let acc: f64 = visible
                        .iter()
                        .zip(&score)
                        .map(|(&j, s)| (s - m).exp() / denom * load(v[base + j * D + d]))
                        .sum();
                    out[base + i * D + d] = if staged { f16_round(acc as f32) } else { acc as f32 };
                }
                lse[(b * H + h) * S + i] = (m + denom.ln()) as f32;
            }
        }
    }
    (out, lse)
}

const Q_BASE: u64 = 0x1000_0000;
const K_BASE: u64 = 0x2000_0000;
const V_BASE: u64 = 0x3000_0000;
const OUT_BASE: u64 = 0x4000_0000;
const LSE_BASE: u64 = 0x5000_0000;
const SEG_BASE: u64 = 0x6000_0000;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Launch `prog` as `nsl_sdpa_fused_forward` does and return
/// `(out widened to f32, lse)`.
fn run(prog: &Program, smem: usize, q: &[f32], k: &[f32], v: &[f32], seg: Option<&[u16]>, order: Order) -> (Vec<f32>, Vec<f32>) {
    let seg_bytes: Vec<u8> = seg.map(|s| s.iter().flat_map(|x| x.to_le_bytes()).collect()).unwrap_or_default();
    let mut global = vec![
        Segment { base: Q_BASE, bytes: f32_bytes(q) },
        Segment { base: K_BASE, bytes: f32_bytes(k) },
        Segment { base: V_BASE, bytes: f32_bytes(v) },
        // Poisoned: an output element the kernel never writes reads as NaN.
        Segment { base: OUT_BASE, bytes: vec![0xFF; B * H * S * D * 2] },
        Segment { base: LSE_BASE, bytes: vec![0xFF; B * H * S * 4] },
        Segment { base: SEG_BASE, bytes: seg_bytes },
    ];
    let row_qkv = (H * S * D * 4) as u64;
    let row_out = (H * S * D * 2) as u64;
    let row_lse = (H * S * 4) as u64;
    let row_seg = (S * 2) as u64;
    for b in 0..B as u64 {
        let mut args: HashMap<String, u64> = prog.params.iter().map(|(_, n)| (n.clone(), 0)).collect();
        let mut set = |name: &str, v: u64| {
            assert!(args.contains_key(name), "the kernel has no parameter `{name}`");
            args.insert(name.to_string(), v);
        };
        set("q_ptr", Q_BASE + b * row_qkv);
        set("k_ptr", K_BASE + b * row_qkv);
        set("v_ptr", V_BASE + b * row_qkv);
        set("out_ptr", OUT_BASE + b * row_out);
        set("scale", (1.0f32 / (D as f32).sqrt()).to_bits() as u64);
        set("batch", 1);
        set("heads", H as u64);
        set("seq_len", S as u64);
        set("head_dim", D as u64);
        set("logsumexp", LSE_BASE + b * row_lse);
        if seg.is_some() {
            set("segment_ids_ptr", SEG_BASE + b * row_seg);
        }
        for ctaid_y in 0..H as u32 {
            for ctaid in 0..(S / BLOCK_Q) as u32 {
                let mut launch = Launch {
                    prog,
                    args: &args,
                    global: &mut global,
                    // Poisoned (0xFF.. is NaN as f32 and f16): a read of
                    // shared memory no thread wrote shows in the output.
                    shared: vec![0xFF; smem],
                    ctaid,
                    ctaid_y,
                    nctaid_y: H as u32,
                    ntid: THREADS,
                    steps: 0,
                };
                run_cta(&mut launch, order);
            }
        }
    }
    let out = global[3]
        .bytes
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let lse = global[4].bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (out, lse)
}

/// Every PTX variant the production emission produces for `config`.
fn variants(config: &FlashAttentionConfig) -> Vec<(String, Program, usize)> {
    let emission = emit_tier_b_variants_for_config(config);
    let mut out = vec![(emission.base_kernel_name, emission.base_ptx)];
    if let (Some(name), Some(ptx)) = (emission.tier_b_on_kernel_name, emission.tier_b_on_ptx) {
        out.push((name, ptx));
    }
    let dynamic = shared_mem_bytes_selected(config) as usize;
    out.into_iter()
        .map(|(name, ptx)| {
            // The emitted module is a C string: strip its terminator.
            let text = std::str::from_utf8(&ptx).expect("PTX is UTF-8").trim_end_matches('\0');
            let prog = parse(text);
            // The static blocks, then the dynamic bytes the runtime requests
            // at launch (`shared_mem_bytes_selected`), which the hardware
            // places after them whether or not the module names them.
            let smem = prog.shared_bytes + dynamic;
            assert!(SHARED_BASE as usize + smem < Q_BASE as usize);
            (name, prog, smem)
        })
        .collect()
}

struct Report {
    exact_out: f32,
    staged_out_ulps: f32,
    exact_lse: f32,
    staged_lse: f32,
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Largest difference in units of one f16 step at the reference value.
fn max_f16_ulps(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let step = (f16::from_f32(*w).to_f32().abs().max(f16::MIN_POSITIVE.to_f32())) * f16::EPSILON.to_f32();
            (g - w).abs() / step
        })
        .fold(0.0, f32::max)
}

fn measure(segment_masked: bool, order: Order) -> Vec<(String, Report)> {
    let n = B * H * S * D;
    let (q, k, v) = (values(1, n), values(2, n), values(3, n));
    let seg = segments();
    let seg = segment_masked.then_some(seg.as_slice());
    let (exact_out, exact_lse) = oracle(&q, &k, &v, seg, false);
    let (staged_out, staged_lse) = oracle(&q, &k, &v, seg, true);
    variants(&config(segment_masked))
        .into_iter()
        .map(|(name, prog, smem)| {
            let (out, lse) = run(&prog, smem, &q, &k, &v, seg, order);
            assert!(out.iter().chain(&lse).all(|x| x.is_finite()), "{name}: an unwritten or non-finite result");
            let r = Report {
                exact_out: max_abs(&out, &exact_out),
                staged_out_ulps: max_f16_ulps(&out, &staged_out),
                exact_lse: max_abs(&lse, &exact_lse),
                staged_lse: max_abs(&lse, &staged_lse),
            };
            eprintln!(
                "{name} (segment_masked={segment_masked}, {order:?}): out vs exact {:.3e}, vs f16-staged {:.2} f16 steps; \
                 lse vs exact {:.3e}, vs f16-staged {:.3e}",
                r.exact_out, r.staged_out_ulps, r.exact_lse, r.staged_lse
            );
            (name, r)
        })
        .collect()
}

/// The kernel IS the f16-staged function: its output is within one f16
/// rounding of that oracle, and its logsumexp (stored as f32) agrees with
/// the staged scores to f32 accuracy.
fn assert_is_f16_staged(reports: &[(String, Report)]) {
    for (name, r) in reports {
        assert!(r.staged_out_ulps <= 1.0, "{name}: {:.2} f16 steps from the f16-staged oracle", r.staged_out_ulps);
        assert!(r.staged_lse < 4e-6, "{name}: lse {:.3e} from the f16-staged oracle", r.staged_lse);
    }
}

/// ...and so it is NOT the f32 function an f32 program asks for: both the
/// output and the logsumexp miss the exact oracle by far more than f32
/// arithmetic would (the f32 kernel this gate is waiting for agrees to
/// ~1e-6). When the fused path gains f32 storage for f32 programs, this is
/// the assertion that changes.
fn assert_misses_exact_by_f16_error(reports: &[(String, Report)]) {
    for (name, r) in reports {
        assert!(r.exact_out > 1e-4, "{name}: out within {:.3e} of exact — no longer f16-staged?", r.exact_out);
        assert!(r.exact_lse > 1e-5, "{name}: lse within {:.3e} of exact — no longer f16-staged?", r.exact_lse);
    }
}

#[test]
fn packed_forward_is_f16_staged_attention_in_every_variant() {
    for order in [Order::Ascending, Order::Descending] {
        let reports = measure(true, order);
        assert_eq!(reports.len(), 2, "a segment-masked config emits base + Tier-B variants");
        assert_is_f16_staged(&reports);
        assert_misses_exact_by_f16_error(&reports);
    }
}

/// The agreement above is not vacuous: each named break in the kernel's
/// dataflow moves the result far from the f16-staged oracle (or faults the
/// interpreter). Every occurrence is replaced — the emitter repeats the
/// score/softmax block once per KV tile.
#[test]
fn kernel_mutants_are_caught() {
    let mutants: &[(&str, &str, &str)] = &[
        (
            "segment mask ignored (a key's own id compared with itself)",
            "setp.ne.u16    %p_seg_SEGMASK, %rs_q_SEGMASK, %rs_k_SEGMASK;",
            "setp.ne.u16    %p_seg_SEGMASK, %rs_q_SEGMASK, %rs_q_SEGMASK;",
        ),
        ("softmax scale dropped", "ld.param.f32 %scale, [scale];", "mov.f32 %scale, 0f3F800000;"),
        (
            "running denominator not rescaled",
            "mul.f32 %row_sum, %row_sum, %correction;",
            "mov.f32 %row_sum, %row_sum;",
        ),
        ("output accumulator not rescaled", "mul.f32 %f48, %f48, %correction;", "mov.f32 %f48, %f48;"),
        // The race this gate found: K and V share one SMEM region, and
        // without the end-of-iteration fence a warp that finishes P·V early
        // writes the next K tile over the V tile slower warps still read.
        (
            "KV-loop fence removed (write-after-read race on the K/V region)",
            "    bar.sync 0;  // FENCE: all warps done reading V before the next K tile\n",
            "",
        ),
    ];
    let n = B * H * S * D;
    let (q, k, v) = (values(1, n), values(2, n), values(3, n));
    let seg = segments();
    let (staged_out, staged_lse) = oracle(&q, &k, &v, Some(&seg), true);
    let config = config(true);
    let emission = emit_tier_b_variants_for_config(&config);
    let base = std::str::from_utf8(&emission.base_ptx).expect("UTF-8").trim_end_matches('\0').to_string();
    let smem_dynamic = shared_mem_bytes_selected(&config) as usize;
    for (what, from, to) in mutants {
        assert!(base.contains(from), "mutant `{what}`: `{from}` is not in the kernel — resync the gate");
        let text = base.replace(from, to);
        let caught = std::panic::catch_unwind(|| {
            let prog = parse(&text);
            let (out, lse) = run(&prog, prog.shared_bytes + smem_dynamic, &q, &k, &v, Some(&seg), Order::Ascending);
            let ulps = max_f16_ulps(&out, &staged_out);
            let lse_err = max_abs(&lse, &staged_lse);
            eprintln!("mutant `{what}`: out {ulps:.1} f16 steps, lse {lse_err:.3e} from the f16-staged oracle");
            !(ulps.is_finite() && lse_err.is_finite()) || ulps > 16.0 || lse_err > 1e-3
        })
        .unwrap_or(true);
        assert!(caught, "mutant `{what}` survived");
    }
}
