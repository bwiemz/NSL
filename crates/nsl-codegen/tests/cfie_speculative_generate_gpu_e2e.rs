//! CFIE Cycle 13 (G15 draft-model-in-binary): GPU end-to-end proof of
//! the speculative decode driver `nsl_cfie_speculative_generate`.
//!
//! Drives the REAL path: synthetic weights -> temp .safetensors ->
//! nsl_model_create -> register kinds 1/2/4/6/7/8 -> kv_slots_init ->
//! kv_pool_alloc -> finalize -> bind_model (TARGET) -> bind_draft_model
//! (DRAFT) -> draft_pool_alloc -> nsl_cfie_speculative_generate.
//!
//! Proofs:
//!   (a) LOSSLESS SELF-SPECULATION (the determinism contract): with the
//!       SAME toy model bound as target AND draft and greedy sampling,
//!       speculative_generate(k=3) EXACTLY equals nsl_cfie_generate for
//!       the same prompt/seed/max_new.  Zero tolerance — any off-by-one
//!       in prefill/round/rollback indexing breaks it.  Run twice
//!       (reproducibility + the single-flight guard releases).
//!   (b) DRAFT != TARGET: a second toy draft model (different weight
//!       seeds, same dims); the ENTIRE speculative loop is mirrored on
//!       CPU (draft cpu chain + cpu_reference_draft_sample + target cpu
//!       chain + cpu_reference_verify_probs rows +
//!       cfie_speculative_ptx::cpu_reference_reject with the engine's
//!       per-round seed) -> exact token-sequence equality vs the GPU
//!       FFI, with the mirror REQUIRED to see >= 1 rejection so the
//!       rollback path is genuinely exercised (pattern printed).
//!   (c) EOS + capacity edges: EOS mid-sequence stops with exactly the
//!       truncated prefix; a small KV slot (PER_SLOT 6 < prompt + K
//!       rounds) forces the round-start capacity probe to fail and the
//!       TAIL FALLBACK to run — output must STILL equal
//!       nsl_cfie_generate's capacity-truncated output (the Verify-1
//!       must-fix: no draft-KV write past the pool, no token drift).
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda \
//!     --test cfie_speculative_generate_gpu_e2e -- --ignored \
//!     --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, emit, DecodeBlockConfig};
use nsl_codegen::cfie_sample_ptx::{
    cpu_reference as cpu_reference_sample, emit as emit_sampler, FusedSampleKernelConfig,
};
use nsl_codegen::cfie_spec_sampler_ptx::{
    cpu_reference_draft_sample, cpu_reference_verify_probs, emit_draft_sample, emit_verify_probs,
    SpecSamplerConfig,
};
use nsl_codegen::cfie_speculative_ptx::{
    cpu_reference_reject, emit_rejection_kernel, RejectionConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_bind_draft_model, nsl_cfie_bind_model, nsl_cfie_draft_pool_alloc,
    nsl_cfie_draft_reset, nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_generate,
    nsl_cfie_generate_reset, nsl_cfie_kv_pool_alloc, nsl_cfie_register_kernel,
    nsl_cfie_speculative_generate, nsl_cfie_weights_reset,
};
use nsl_runtime::cfie::ffi::nsl_cfie_kv_slots_init;
use nsl_runtime::nsl_cuda_init;

// ---------------------------------------------------------------------------
// CUDA availability guard (house convention: per-file copy)
// ---------------------------------------------------------------------------

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (zero-mean mapping, per-file copy of the
// cfie_generate_gpu_e2e fixture).
// ---------------------------------------------------------------------------

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (seed >> 32) as u32;
        *x = ((u as f64) / 4294967296.0) as f32 - 0.5;
    }
}

// ---------------------------------------------------------------------------
// Toy dims (the Cycle-10 generate fixture's shapes: GQA + non-trivial
// FFN + 128 vocab).  PER_SLOT is per-test (capacity edges); the DRAFT
// shares every dim (test (b) differs only in WEIGHTS — "same dims"
// is the ABI-sanctioned simple case).
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const D_FF: usize = 128;
const MAX_SLOTS: usize = 2;
const N_LAYERS: usize = 2;
const VOCAB: usize = 128;
const TOP_K: u32 = 8;
const K_TOKENS: usize = 3;

fn block_cfg(per_slot: usize, max_slots: usize) -> DecodeBlockConfig {
    DecodeBlockConfig {
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        d_ff: D_FF as u32,
        per_slot_max_tokens: per_slot as u32,
        max_slots: max_slots as u32,
        n_layers: N_LAYERS as u32,
        rope_theta: 10000.0,
        eps: 1e-5,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
    }
}

fn target_pool_bytes(per_slot: usize) -> usize {
    N_LAYERS * 2 * (MAX_SLOTS * per_slot) * N_KV_HEADS * HEAD_DIM * 2
}

/// Draft pool: max_slots = 1, per-slot = the TARGET's (sizing contract).
fn draft_pool_bytes(per_slot: usize) -> usize {
    N_LAYERS * 2 * per_slot * N_KV_HEADS * HEAD_DIM * 2
}

// ---------------------------------------------------------------------------
// Toy model (seed-parameterized so the DRAFT can differ from the TARGET)
// ---------------------------------------------------------------------------

struct LayerWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
}

fn seeded_layer_weights(seed: u64) -> LayerWeights {
    let mut w = LayerWeights {
        wq: vec![0f32; N_HEADS * HEAD_DIM * D_MODEL],
        wk: vec![0f32; N_KV_HEADS * HEAD_DIM * D_MODEL],
        wv: vec![0f32; N_KV_HEADS * HEAD_DIM * D_MODEL],
        wo: vec![0f32; D_MODEL * N_HEADS * HEAD_DIM],
        w_gate: vec![0f32; D_FF * D_MODEL],
        w_up: vec![0f32; D_FF * D_MODEL],
        w_down: vec![0f32; D_MODEL * D_FF],
        norm1: vec![0f32; D_MODEL],
        norm2: vec![0f32; D_MODEL],
    };
    fill_seeded(&mut w.wq, seed);
    fill_seeded(&mut w.wk, seed.wrapping_add(1));
    fill_seeded(&mut w.wv, seed.wrapping_add(2));
    fill_seeded(&mut w.wo, seed.wrapping_add(3));
    fill_seeded(&mut w.w_gate, seed.wrapping_add(4));
    fill_seeded(&mut w.w_up, seed.wrapping_add(5));
    fill_seeded(&mut w.w_down, seed.wrapping_add(6));
    fill_seeded(&mut w.norm1, seed.wrapping_add(7));
    fill_seeded(&mut w.norm2, seed.wrapping_add(8));
    for m in [
        &mut w.wq, &mut w.wk, &mut w.wv, &mut w.wo, &mut w.w_gate, &mut w.w_up, &mut w.w_down,
    ] {
        for v in m.iter_mut() {
            *v *= 0.35;
        }
    }
    for g in w.norm1.iter_mut().chain(w.norm2.iter_mut()) {
        *g = 1.0 + 0.1 * *g;
    }
    w
}

struct RoundedLayer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
}

fn round_layer(w: &LayerWeights) -> RoundedLayer {
    let r = |m: &[f32]| -> Vec<f32> { m.iter().map(|&x| f16_round(x)).collect() };
    RoundedLayer {
        wq: r(&w.wq),
        wk: r(&w.wk),
        wv: r(&w.wv),
        wo: r(&w.wo),
        w_gate: r(&w.w_gate),
        w_up: r(&w.w_up),
        w_down: r(&w.w_down),
        norm1: w.norm1.clone(),
        norm2: w.norm2.clone(),
    }
}

struct ToyModel {
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>, // [vocab][d_model]
    embed: Vec<f32>,   // [vocab][d_model]
}

/// `seed_base` differentiates target (0xA0 family — the Cycle-10
/// fixture's exact weights) from the draft (any other base).
fn build_toy_model(seed_base: u64) -> ToyModel {
    let layers = vec![
        seeded_layer_weights(seed_base),
        seeded_layer_weights(seed_base + 1),
    ];
    let mut final_norm = vec![0f32; D_MODEL];
    fill_seeded(&mut final_norm, seed_base + 0x51);
    for g in final_norm.iter_mut() {
        *g = 1.0 + 0.1 * *g;
    }
    let mut lm_head = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut lm_head, seed_base + 0x52);
    let mut embed = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut embed, seed_base + 0x43);
    for v in embed.iter_mut() {
        *v *= 0.7;
    }
    ToyModel {
        layers,
        final_norm,
        lm_head,
        embed,
    }
}

fn ln(name: &str, i: usize) -> String {
    format!("model.layers.{i}.{name}")
}

fn write_toy_safetensors(m: &ToyModel) -> tempfile::TempPath {
    let mut owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    let bytes_of = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    for (i, l) in m.layers.iter().enumerate() {
        owned.push((
            ln("self_attn.q_proj.weight", i),
            bytes_of(&l.wq),
            vec![N_HEADS * HEAD_DIM, D_MODEL],
        ));
        owned.push((
            ln("self_attn.k_proj.weight", i),
            bytes_of(&l.wk),
            vec![N_KV_HEADS * HEAD_DIM, D_MODEL],
        ));
        owned.push((
            ln("self_attn.v_proj.weight", i),
            bytes_of(&l.wv),
            vec![N_KV_HEADS * HEAD_DIM, D_MODEL],
        ));
        owned.push((
            ln("self_attn.o_proj.weight", i),
            bytes_of(&l.wo),
            vec![D_MODEL, N_HEADS * HEAD_DIM],
        ));
        owned.push((
            ln("mlp.gate_proj.weight", i),
            bytes_of(&l.w_gate),
            vec![D_FF, D_MODEL],
        ));
        owned.push((
            ln("mlp.up_proj.weight", i),
            bytes_of(&l.w_up),
            vec![D_FF, D_MODEL],
        ));
        owned.push((
            ln("mlp.down_proj.weight", i),
            bytes_of(&l.w_down),
            vec![D_MODEL, D_FF],
        ));
        owned.push((
            ln("input_layernorm.weight", i),
            bytes_of(&l.norm1),
            vec![D_MODEL],
        ));
        owned.push((
            ln("post_attention_layernorm.weight", i),
            bytes_of(&l.norm2),
            vec![D_MODEL],
        ));
    }
    owned.push((
        "model.norm.weight".to_string(),
        bytes_of(&m.final_norm),
        vec![D_MODEL],
    ));
    owned.push((
        "lm_head.weight".to_string(),
        bytes_of(&m.lm_head),
        vec![VOCAB, D_MODEL],
    ));
    owned.push((
        "model.embed_tokens.weight".to_string(),
        bytes_of(&m.embed),
        vec![VOCAB, D_MODEL],
    ));

    let data: HashMap<String, safetensors::tensor::TensorView<'_>> = owned
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                bytes.as_slice(),
            )
            .expect("TensorView::new");
            (name.clone(), view)
        })
        .collect();
    let serialized = safetensors::tensor::serialize(&data, &None).expect("safetensors serialize");

    let tmp = tempfile::Builder::new()
        .prefix("cfie_spec_toy_")
        .suffix(".safetensors")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), &serialized).expect("write safetensors");
    tmp.into_temp_path()
}

// ---------------------------------------------------------------------------
// Engine setup: kinds 1/2/4/6/7/8 + slots + pools + finalize + bindings
// ---------------------------------------------------------------------------

fn sampler_program() -> nsl_codegen::cfie_fused_sample::FusedSampleProgram {
    let sampler_params = SamplingParams {
        strategy: SamplingStrategy::Greedy,
        temperature: 0.0,
        top_k: TOP_K,
        ..Default::default()
    };
    emit_program(
        sampler_params,
        LmHeadShape {
            d_model: D_MODEL as u32,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            dtype_bytes: 2,
        },
    )
}

fn spec_sampler_cfg() -> SpecSamplerConfig {
    SpecSamplerConfig {
        d_model: D_MODEL as u32,
        vocab_size: VOCAB as u32,
        vocab_tile: 128,
        sm_version: 80,
    }
}

fn register(kind: i64, ptx: &str, name: &str, grid: i64, block: i64) {
    let rc = nsl_cfie_register_kernel(
        kind,
        0,
        ptx.as_ptr() as i64,
        ptx.len() as i64,
        name.as_ptr() as i64,
        name.len() as i64,
        grid,
        block,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind {kind}) refused");
}

/// Register kinds 1/2/4/6/7/8, init slots (MAX_SLOTS x per_slot), alloc
/// the TARGET pool, finalize.  The kind-6 draft block bakes max_slots=1
/// with the SAME per-slot capacity (the sizing contract).
fn setup_spec_engine(per_slot: usize) {
    assert_eq!(nsl_cfie_engine_destroy(), 0);

    let (blk_ptx, blk_meta) = emit(&block_cfg(per_slot, MAX_SLOTS));
    register(2, &blk_ptx, &blk_meta.kernel_name, 1, blk_meta.block_dim as i64);

    let (dblk_ptx, dblk_meta) = emit(&block_cfg(per_slot, 1));
    register(6, &dblk_ptx, &dblk_meta.kernel_name, 1, dblk_meta.block_dim as i64);

    let program = sampler_program();
    let (s_ptx, s_meta) = emit_sampler(
        &program,
        &FusedSampleKernelConfig {
            d_model: D_MODEL as u32,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            top_k: TOP_K,
            sm_version: 80,
            grammar_states: 0,
        },
    );
    register(1, &s_ptx, &s_meta.kernel_name, 1, s_meta.block_dim as i64);

    let (r_ptx, r_meta) = emit_rejection_kernel(&RejectionConfig {
        k_tokens: K_TOKENS as u32,
        vocab_size: VOCAB as u32,
        sm_version: 80,
    });
    register(4, &r_ptx, &r_meta.kernel_name, 1, r_meta.block_dim as i64);

    let (d7_ptx, d7_meta) = emit_draft_sample(&spec_sampler_cfg());
    register(7, &d7_ptx, &d7_meta.kernel_name, 1, d7_meta.block_dim as i64);

    let (v8_ptx, v8_meta) = emit_verify_probs(&spec_sampler_cfg());
    register(8, &v8_ptx, &v8_meta.kernel_name, 1, v8_meta.block_dim as i64);

    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, per_slot as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(target_pool_bytes(per_slot) as i64), 0);
    let n = nsl_cfie_engine_finalize();
    assert_eq!(n, 6, "finalize must resolve all six kinds (got {n})");
}

fn bind_target(model_handle: i64) {
    let rc = nsl_cfie_bind_model(
        model_handle,
        N_LAYERS as i64,
        D_MODEL as i64,
        N_HEADS as i64,
        N_KV_HEADS as i64,
        HEAD_DIM as i64,
        D_FF as i64,
        VOCAB as i64,
    );
    assert_eq!(rc, 0, "bind_model refused (rc {rc})");
}

fn bind_draft(model_handle: i64, per_slot: usize) {
    let rc = nsl_cfie_bind_draft_model(
        model_handle,
        N_LAYERS as i64,
        D_MODEL as i64,
        N_HEADS as i64,
        N_KV_HEADS as i64,
        HEAD_DIM as i64,
        D_FF as i64,
        VOCAB as i64,
    );
    assert_eq!(rc, 0, "bind_draft_model refused (rc {rc})");
    assert_eq!(
        nsl_cfie_draft_pool_alloc(draft_pool_bytes(per_slot) as i64),
        0,
        "draft_pool_alloc refused"
    );
}

fn teardown(handles: &[i64]) {
    for &h in handles {
        nsl_runtime::c_api::nsl_model_destroy(h);
    }
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);
}

// ---------------------------------------------------------------------------
// CPU mirror machinery
// ---------------------------------------------------------------------------

struct CpuModel {
    rounded: Vec<RoundedLayer>,
    final_norm: Vec<f32>,
    lm_rounded: Vec<f32>,
    embed: Vec<f32>,
}

fn cpu_model(m: &ToyModel) -> CpuModel {
    CpuModel {
        rounded: m.layers.iter().map(round_layer).collect(),
        final_norm: m.final_norm.clone(),
        lm_rounded: m.lm_head.iter().map(|&x| f16_round(x)).collect(),
        embed: m.embed.clone(),
    }
}

/// One sequence's per-layer KV chains, with by-position rollback
/// (truncation mirrors the engine's overwrite-by-position semantics:
/// positions are always re-fed in order after a rollback).
struct CpuSeq {
    kv: Vec<(Vec<f32>, Vec<f32>)>,
}

impl CpuSeq {
    fn new() -> Self {
        CpuSeq {
            kv: (0..N_LAYERS).map(|_| (Vec::new(), Vec::new())).collect(),
        }
    }
    /// Embed-gather + full block chain at `pos`; returns the final
    /// hidden state and appends one KV row per layer.
    fn feed(&mut self, m: &CpuModel, cfg: &DecodeBlockConfig, token: i64, pos: usize) -> Vec<f32> {
        let row = token as usize * D_MODEL;
        let mut x = m.embed[row..row + D_MODEL].to_vec();
        for l in 0..N_LAYERS {
            let (kv_k, kv_v) = &mut self.kv[l];
            x = cpu_reference(
                cfg,
                &x,
                &m.rounded[l].wq,
                &m.rounded[l].wk,
                &m.rounded[l].wv,
                &m.rounded[l].wo,
                &m.rounded[l].w_gate,
                &m.rounded[l].w_up,
                &m.rounded[l].w_down,
                &m.rounded[l].norm1,
                &m.rounded[l].norm2,
                kv_k,
                kv_v,
                pos as u32,
            );
        }
        x
    }
    fn rollback_to(&mut self, new_len: usize) {
        for (k, v) in self.kv.iter_mut() {
            k.truncate(new_len * N_KV_HEADS * HEAD_DIM);
            v.truncate(new_len * N_KV_HEADS * HEAD_DIM);
        }
    }
}

/// FULL CPU mirror of nsl_cfie_speculative_generate: prefill both
/// models, then rounds of (capacity probe -> draft K -> verify K rows
/// -> cpu_reference_reject -> emit/rollback), with the tail fallback
/// and the all-accept bonus step — the engine's exact control flow.
/// Returns (tokens, accept/reject pattern strings, rejection count).
#[allow(clippy::too_many_arguments)]
fn cpu_speculative_mirror(
    target: &CpuModel,
    draft: &CpuModel,
    cfg: &DecodeBlockConfig,
    prompt: &[i64],
    max_new: usize,
    eos: i64,
    rng_seed: i64,
    per_slot: usize,
) -> (Vec<u32>, Vec<String>, usize) {
    let program = sampler_program();
    let scfg = spec_sampler_cfg();
    let reject_cfg = RejectionConfig {
        k_tokens: K_TOKENS as u32,
        vocab_size: VOCAB as u32,
        sm_version: 80,
    };
    let mut tseq = CpuSeq::new();
    let mut dseq = CpuSeq::new();
    let mut out: Vec<u32> = Vec::new();
    let mut pattern: Vec<String> = Vec::new();
    let mut rejections = 0usize;

    // record() mirror: clamped writes don't matter here (cap = max_new).
    let record = |tok: i64, out: &mut Vec<u32>| -> bool {
        out.push(tok as u32);
        tok == eos || out.len() >= max_new
    };

    // Phase 1+2: prefill both models (target sampler matters only at
    // the last prompt position).
    let mut t0: u32 = 0;
    for (pos, &t) in prompt.iter().enumerate() {
        let x = tseq.feed(target, cfg, t, pos);
        if pos == prompt.len() - 1 {
            t0 = cpu_reference_sample(&program, &x, &target.final_norm, &target.lm_rounded, None, rng_seed as u64);
        }
        dseq.feed(draft, cfg, t, pos);
    }
    if record(t0 as i64, &mut out) {
        return (out, pattern, rejections);
    }

    let mut target_pos = prompt.len();
    let mut draft_pos = prompt.len();
    let mut last = t0 as i64;
    let mut round: i64 = 0;

    'rounds: loop {
        // Capacity probe (the engine's kv_slot_advance(slot, K) probe).
        if target_pos + K_TOKENS > per_slot {
            // Tail fallback: plain decode steps to the -2 stop.
            loop {
                if target_pos + 1 > per_slot {
                    break 'rounds;
                }
                let x = tseq.feed(target, cfg, last, target_pos);
                target_pos += 1;
                let t = cpu_reference_sample(&program, &x, &target.final_norm, &target.lm_rounded, None, rng_seed as u64);
                if record(t as i64, &mut out) {
                    break 'rounds;
                }
                last = t as i64;
            }
        }

        // Draft K greedily.
        let mut dtoks: Vec<i64> = Vec::with_capacity(K_TOKENS);
        let mut dprobs: Vec<f32> = Vec::with_capacity(K_TOKENS);
        let mut prev = last;
        for _ in 0..K_TOKENS {
            let x = dseq.feed(draft, cfg, prev, draft_pos);
            draft_pos += 1;
            let (tok, p) = cpu_reference_draft_sample(&scfg, &x, &draft.final_norm, &draft.lm_rounded);
            dtoks.push(tok as i64);
            dprobs.push(p);
            prev = tok as i64;
        }

        // Target verify rows (row j's input is the PREVIOUS token).
        let mut rows = vec![0f32; K_TOKENS * VOCAB];
        for j in 0..K_TOKENS {
            let inp = if j == 0 { last } else { dtoks[j - 1] };
            let x = tseq.feed(target, cfg, inp, target_pos);
            target_pos += 1;
            let row = cpu_reference_verify_probs(&scfg, &x, &target.final_norm, &target.lm_rounded);
            rows[j * VOCAB..(j + 1) * VOCAB].copy_from_slice(&row);
        }

        // ONE rejection step, engine seed schedule: rng_seed + round.
        let dtoks_u32: Vec<u32> = dtoks.iter().map(|&t| t as u32).collect();
        let (accepted, correction) = cpu_reference_reject(
            &reject_cfg,
            &rows,
            &dprobs,
            &dtoks_u32,
            rng_seed.wrapping_add(round) as u64,
        );
        round += 1;

        if correction == u32::MAX {
            pattern.push(format!("round {}: all-accept ({})", round - 1, K_TOKENS));
            for &t in &dtoks {
                if record(t, &mut out) {
                    break 'rounds;
                }
            }
            // Bonus step (plain target decode_step; -2 stops cleanly).
            if target_pos + 1 > per_slot {
                break 'rounds;
            }
            let inp = dtoks[K_TOKENS - 1];
            let x = tseq.feed(target, cfg, inp, target_pos);
            target_pos += 1;
            let bonus = cpu_reference_sample(&program, &x, &target.final_norm, &target.lm_rounded, None, rng_seed as u64);
            // Draft KV sync.
            dseq.feed(draft, cfg, inp, draft_pos);
            draft_pos += 1;
            if record(bonus as i64, &mut out) {
                break 'rounds;
            }
            last = bonus as i64;
        } else {
            rejections += 1;
            pattern.push(format!(
                "round {}: accepted {} of {}, correction {}",
                round - 1,
                accepted,
                K_TOKENS,
                correction
            ));
            for &t in &dtoks[..accepted as usize] {
                if record(t, &mut out) {
                    break 'rounds;
                }
            }
            let stale = K_TOKENS - accepted as usize - 1;
            target_pos -= stale;
            draft_pos -= stale;
            tseq.rollback_to(target_pos);
            dseq.rollback_to(draft_pos);
            if record(correction as i64, &mut out) {
                break 'rounds;
            }
            last = correction as i64;
        }
    }
    (out, pattern, rejections)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const RNG_SEED: i64 = 0x5EED_1234;
const PROMPT: [i64; 2] = [3, 17];

fn run_generate(max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    let mut out = [0i64; 32];
    let got = nsl_cfie_generate(
        PROMPT.as_ptr() as i64,
        PROMPT.len() as i64,
        max_new as i64,
        eos,
        RNG_SEED,
        out.as_mut_ptr() as i64,
        out.len() as i64,
    );
    let n = got.clamp(0, out.len() as i64) as usize;
    (got, out[..n].iter().map(|&t| t as u32).collect())
}

fn run_speculative(max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    let mut out = [0i64; 32];
    let got = nsl_cfie_speculative_generate(
        PROMPT.as_ptr() as i64,
        PROMPT.len() as i64,
        max_new as i64,
        eos,
        RNG_SEED,
        K_TOKENS as i64,
        out.as_mut_ptr() as i64,
        out.len() as i64,
    );
    let n = got.clamp(0, out.len() as i64) as usize;
    (got, out[..n].iter().map(|&t| t as u32).collect())
}

/// (a) LOSSLESS SELF-SPECULATION: same model as target AND draft ->
/// speculative output EXACTLY equals plain generate.  Run twice.
#[test]
#[ignore = "requires CUDA GPU"]
fn lossless_self_speculation_matches_generate_exactly() {
    if !cuda_available() {
        return;
    }
    let per_slot = 16usize;
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);

    let model = build_toy_model(0xA0);
    let st = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);

    setup_spec_engine(per_slot);
    bind_target(handle);
    bind_draft(handle, per_slot); // SAME model: the lossless anchor

    let max_new = 6usize;
    let (g_rc, g_toks) = run_generate(max_new, 9999);
    assert_eq!(g_rc, max_new as i64);
    let (s_rc, s_toks) = run_speculative(max_new, 9999);
    eprintln!("plain generate:      {g_toks:?}");
    eprintln!("self-speculative(1): {s_toks:?}");
    assert_eq!(s_rc, max_new as i64, "speculative must return max_new");
    assert_eq!(
        s_toks, g_toks,
        "LOSSLESS SELF-SPECULATION: speculative output must EXACTLY equal generate"
    );

    // Second run: reproducible AND the single-flight guard released.
    let (s_rc2, s_toks2) = run_speculative(max_new, 9999);
    eprintln!("self-speculative(2): {s_toks2:?}");
    assert_eq!(s_rc2, max_new as i64);
    assert_eq!(s_toks2, g_toks, "second speculative run must reproduce");

    teardown(&[handle]);
    drop(st);
}

/// (b) DRAFT != TARGET: exact token equality vs the full CPU mirror,
/// with >= 1 REJECTION required (the rollback path genuinely runs).
#[test]
#[ignore = "requires CUDA GPU"]
fn draft_differs_from_target_matches_cpu_mirror_with_rejections() {
    if !cuda_available() {
        return;
    }
    let per_slot = 64usize;
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);

    let target = build_toy_model(0xA0);
    let draft = build_toy_model(0xB7); // different weights, same dims
    let st_t = write_toy_safetensors(&target);
    let st_d = write_toy_safetensors(&draft);
    let pc_t = std::ffi::CString::new(st_t.to_str().unwrap()).unwrap();
    let pc_d = std::ffi::CString::new(st_d.to_str().unwrap()).unwrap();
    let h_t = nsl_runtime::c_api::nsl_model_create(pc_t.as_ptr() as i64);
    let h_d = nsl_runtime::c_api::nsl_model_create(pc_d.as_ptr() as i64);
    assert!(h_t != 0 && h_d != 0);

    setup_spec_engine(per_slot);
    bind_target(h_t);
    bind_draft(h_d, per_slot);

    let max_new = 12usize;
    let (rc, gpu_toks) = run_speculative(max_new, 9999);
    assert_eq!(rc, max_new as i64, "speculative must return max_new");

    let (cpu_toks, pattern, rejections) = cpu_speculative_mirror(
        &cpu_model(&target),
        &cpu_model(&draft),
        &block_cfg(per_slot, MAX_SLOTS),
        &PROMPT,
        max_new,
        9999,
        RNG_SEED,
        per_slot,
    );
    eprintln!("GPU speculative: {gpu_toks:?}");
    eprintln!("CPU mirror:      {cpu_toks:?}");
    eprintln!("accept/reject pattern:");
    for p in &pattern {
        eprintln!("  {p}");
    }
    assert!(
        rejections >= 1,
        "the fixture must exercise the rejection/rollback path (got 0 rejections; \
         change the draft seed)"
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "GPU speculative output must EXACTLY equal the CPU mirror of the whole loop"
    );

    teardown(&[h_t, h_d]);
    drop(st_t);
    drop(st_d);
}

/// (c1) EOS mid-sequence stops the speculative path with exactly the
/// truncated prefix.
#[test]
#[ignore = "requires CUDA GPU"]
fn speculative_eos_stops_early() {
    if !cuda_available() {
        return;
    }
    let per_slot = 16usize;
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);

    let model = build_toy_model(0xA0);
    let st = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);

    setup_spec_engine(per_slot);
    bind_target(handle);
    bind_draft(handle, per_slot);

    // Reference (eos unreachable), then re-run with eos = ref[1].
    let max_new = 6usize;
    let (_, ref_toks) = run_speculative(max_new, 9999);
    assert!(ref_toks.len() >= 2);
    let eos = ref_toks[1] as i64;
    let (rc, toks) = run_speculative(max_new, eos);
    eprintln!("EOS-stop tokens: {toks:?} (eos={eos})");
    assert_eq!(rc, 2, "EOS at emitted-index 1 must stop with exactly 2 tokens");
    assert_eq!(toks, ref_toks[..2].to_vec());

    teardown(&[handle]);
    drop(st);
}

/// (c2) CAPACITY EDGE (the Verify-1 must-fix): PER_SLOT 6 makes the
/// round-start probe fail after the first round, forcing the TAIL
/// FALLBACK — the draft phase never writes past the pool, and the
/// output still EXACTLY equals plain generate truncated at capacity.
#[test]
#[ignore = "requires CUDA GPU"]
fn speculative_capacity_tail_fallback_matches_generate() {
    if !cuda_available() {
        return;
    }
    let per_slot = 6usize; // prompt 2 + one K=3 round + bonus fills it
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);

    let model = build_toy_model(0xA0);
    let st = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);

    setup_spec_engine(per_slot);
    bind_target(handle);
    bind_draft(handle, per_slot);

    // max_new 8 wants 8 tokens; capacity supports fewer — BOTH paths
    // must truncate identically (the -2 clean-stop contract).
    let max_new = 8usize;
    let (g_rc, g_toks) = run_generate(max_new, 9999);
    assert!(
        (g_rc as usize) < max_new,
        "fixture must actually hit capacity (got {g_rc} of {max_new})"
    );
    let (s_rc, s_toks) = run_speculative(max_new, 9999);
    eprintln!("plain generate @cap6:  rc={g_rc} {g_toks:?}");
    eprintln!("speculative @cap6:     rc={s_rc} {s_toks:?}");
    assert_eq!(
        (s_rc, &s_toks),
        (g_rc, &g_toks),
        "capacity-truncated speculative output must equal generate's"
    );

    teardown(&[handle]);
    drop(st);
}
