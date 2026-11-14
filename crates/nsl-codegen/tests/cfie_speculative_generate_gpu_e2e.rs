//! GPU end-to-end tests of the speculative decode driver
//! `nsl_cfie_speculative_generate`, and of what both drivers sample.
//!
//! Drives the REAL path: synthetic weights -> temp .safetensors ->
//! nsl_model_create -> register kinds 1/2/6/7 -> kv_slots_init ->
//! kv_pool_alloc -> finalize -> bind_model (TARGET) -> bind_draft_model
//! (DRAFT) -> draft_pool_alloc -> the drivers.
//!
//! What is checked:
//!   (a) THE SAMPLED DISTRIBUTION, against exact probabilities. A model
//!       built so that every position has the same known logits is
//!       decoded for 1,500 tokens at temperature 0.7, top-k 8, top-p 0.9;
//!       the token counts must match the probabilities this file computes
//!       in f64 from those settings (five standard deviations). The same
//!       request through the speculative driver, with a draft model that
//!       disagrees with the target, must give the same tokens.
//!       Until November 2026 every token of a request was drawn at one
//!       quantile, and speculative verification sampled a temperature-1
//!       softmax with an acceptance rule that over-accepted the draft.
//!   (b) SPECULATIVE == PLAIN, token for token, for a sampling target and
//!       for a greedy one, with the draft equal to the target and with a
//!       different draft; the fixture must make the different draft
//!       mispredict at least once, so the round-ending path really runs.
//!   (c) EOS + capacity edges: EOS mid-sequence stops with exactly the
//!       truncated prefix; a small KV slot forces the tail to finish with
//!       plain steps, and the output still equals plain decoding's.
//!
//! The drivers are compared on the same draws by giving both the same
//! request index (`set_next_request_index`).
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
use nsl_codegen::cfie_sample_ptx::{emit as emit_sampler, FusedSampleKernelConfig};
use nsl_codegen::cfie_spec_sampler_ptx::{
    cpu_reference_draft_sample, emit_draft_sample, SpecSamplerConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_bind_draft_model, nsl_cfie_bind_model, nsl_cfie_draft_pool_alloc,
    nsl_cfie_draft_reset, nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_generate,
    nsl_cfie_generate_reset, nsl_cfie_kv_pool_alloc, nsl_cfie_register_kernel,
    nsl_cfie_speculative_generate, nsl_cfie_weights_reset, set_next_request_index,
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
    let serialized = safetensors::tensor::serialize(&data, None).expect("safetensors serialize");

    let tmp = tempfile::Builder::new()
        .prefix("cfie_spec_toy_")
        .suffix(".safetensors")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), &serialized).expect("write safetensors");
    tmp.into_temp_path()
}

// ---------------------------------------------------------------------------
// Engine setup: kinds 1/2/6/7 + slots + pools + finalize + bindings
// ---------------------------------------------------------------------------

/// The target's sampler settings for a test.
#[derive(Clone, Copy)]
enum Target {
    Greedy,
    /// Temperature 0.7, top-k 8, top-p 0.9.
    Nucleus,
}

const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.9;

fn sampler_program(target: Target) -> nsl_codegen::cfie_fused_sample::FusedSampleProgram {
    let sampler_params = match target {
        Target::Greedy => SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: TOP_K,
            ..Default::default()
        },
        Target::Nucleus => SamplingParams {
            strategy: SamplingStrategy::TopKTopP,
            temperature: TEMPERATURE,
            top_k: TOP_K,
            top_p: TOP_P,
            ..Default::default()
        },
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
        norm_eps: 1e-5,
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

/// Register kinds 1/2/6/7, init slots (MAX_SLOTS x per_slot), alloc the
/// TARGET pool, finalize.  The kind-6 draft block bakes max_slots=1 with
/// the SAME per-slot capacity (the sizing contract).
fn setup_spec_engine(per_slot: usize, target: Target) {
    assert_eq!(nsl_cfie_engine_destroy(), 0);

    let (blk_ptx, blk_meta) = emit(&block_cfg(per_slot, MAX_SLOTS));
    register(2, &blk_ptx, &blk_meta.kernel_name, 1, blk_meta.block_dim as i64);

    let (dblk_ptx, dblk_meta) = emit(&block_cfg(per_slot, 1));
    register(6, &dblk_ptx, &dblk_meta.kernel_name, 1, dblk_meta.block_dim as i64);

    let program = sampler_program(target);
    let (s_ptx, s_meta) = emit_sampler(
        &program,
        &FusedSampleKernelConfig {
            d_model: D_MODEL as u32,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            top_k: TOP_K,
            grammar_states: 0,
            norm_eps: 1e-5,
        },
    );
    register(1, &s_ptx, &s_meta.kernel_name, 1, s_meta.block_dim as i64);

    let (d7_ptx, d7_meta) = emit_draft_sample(&spec_sampler_cfg());
    register(7, &d7_ptx, &d7_meta.kernel_name, 1, d7_meta.block_dim as i64);

    // The kinds that no longer exist are refused, not ignored.
    for gone in [0i64, 3, 4, 5, 8] {
        let rc = nsl_cfie_register_kernel(
            gone,
            0,
            d7_ptx.as_ptr() as i64,
            d7_ptx.len() as i64,
            d7_meta.kernel_name.as_ptr() as i64,
            d7_meta.kernel_name.len() as i64,
            1,
            128,
            0,
        );
        assert_eq!(rc, -1, "kind {gone} must be refused");
    }

    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, per_slot as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(target_pool_bytes(per_slot) as i64), 0);
    let n = nsl_cfie_engine_finalize();
    assert_eq!(n, 4, "finalize must resolve all four kinds (got {n})");
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

/// How many emitted tokens the DRAFT model, fed the true sequence, would
/// have predicted wrongly (greedy).  The first such position is one the
/// driver's draft really mispredicted — within a round the draft is
/// conditioned on the true prefix until its first wrong guess — so a
/// non-zero count means a round ended early in the run under test.  This
/// uses the CPU reference of the draft chain to characterise the FIXTURE;
/// it is not the oracle for any token.
fn draft_mispredictions(draft: &CpuModel, cfg: &DecodeBlockConfig, prompt: &[i64], emitted: &[u32]) -> usize {
    let scfg = spec_sampler_cfg();
    let mut seq = CpuSeq::new();
    let mut wrong = 0;
    let all: Vec<i64> = prompt.iter().copied().chain(emitted.iter().map(|&t| t as i64)).collect();
    for (pos, &tok) in all.iter().enumerate().take(all.len() - 1) {
        let x = seq.feed(draft, cfg, tok, pos);
        if pos + 1 >= prompt.len() {
            let (guess, _) = cpu_reference_draft_sample(&scfg, &x, &draft.final_norm, &draft.lm_rounded);
            if guess as i64 != all[pos + 1] {
                wrong += 1;
            }
        }
    }
    wrong
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const RNG_SEED: i64 = 0x5EED_1234;
const PROMPT: [i64; 2] = [3, 17];

/// Plain decoding as request number `request`.
fn run_generate_as(request: u64, max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    let mut out = vec![0i64; max_new.max(32)];
    set_next_request_index(request);
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

/// Speculative decoding as request number `request`.
fn run_speculative_as(request: u64, max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    let mut out = vec![0i64; max_new.max(32)];
    set_next_request_index(request);
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

fn run_generate(max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    run_generate_as(0, max_new, eos)
}

fn run_speculative(max_new: usize, eos: i64) -> (i64, Vec<u32>) {
    run_speculative_as(0, max_new, eos)
}

fn reset_all() {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_draft_reset(), 0);
}

fn load(model: &ToyModel) -> (i64, tempfile::TempPath) {
    let st = write_toy_safetensors(model);
    let path_c = std::ffi::CString::new(st.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);
    (handle, st)
}

// ---- (a) the sampled distribution ----------------------------------------

/// The logits every position of the i.i.d. model has, before the final
/// norm's scale: eight live tokens (all exact in f16 after the division
/// by D_MODEL below) and -6 for the rest.
const LIVE_LOGITS: [(usize, f32); 8] =
    [(3, 2.0), (17, 1.5), (40, 1.25), (77, 1.0), (100, 0.5), (5, 0.25), (64, 0.0), (127, -0.5)];

/// A model whose next-token logits are the same at every position
/// whatever came before: every embedding row is the same vector, and each
/// layer's output projections (`wo`, `w_down`) are zero, so the residual
/// stream reaching the final norm is that vector unchanged.  The LM-head
/// row of token t is `logit_t / D_MODEL` in every column, so the logit is
/// `logit_t * mean(normed hidden)`.
fn iid_model() -> ToyModel {
    let mut m = build_toy_model(0xC3);
    for l in m.layers.iter_mut() {
        l.wo.iter_mut().for_each(|w| *w = 0.0);
        l.w_down.iter_mut().for_each(|w| *w = 0.0);
    }
    m.embed.iter_mut().for_each(|e| *e = 0.5);
    m.final_norm.iter_mut().for_each(|g| *g = 1.0);
    let mut logits = vec![-6.0f32; VOCAB];
    for (tok, v) in LIVE_LOGITS {
        logits[tok] = v;
    }
    for (t, &l) in logits.iter().enumerate() {
        for d in 0..D_MODEL {
            m.lm_head[t * D_MODEL + d] = l / D_MODEL as f32;
        }
    }
    m
}

/// The exact next-token distribution of [`iid_model`] under the Nucleus
/// settings, in f64, from the definition: RMSNorm of the constant hidden
/// row, the LM head, the temperature, the top-k, the softmax, and the
/// smallest descending prefix whose mass exceeds top-p, renormalised.
fn iid_exact_distribution() -> Vec<f64> {
    // hidden = 0.5 everywhere: mean square 0.25, so each normed element is
    // 0.5 / sqrt(0.25 + 1e-5), and the logit of token t is that times
    // logit_t (D_MODEL columns of logit_t / D_MODEL).
    let normed = 0.5f64 / (0.25f64 + 1e-5).sqrt();
    let mut logits = vec![-6.0f64 * normed; VOCAB];
    for (tok, v) in LIVE_LOGITS {
        logits[tok] = v as f64 * normed;
    }
    let mut order: Vec<usize> = (0..VOCAB).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let kept = &order[..TOP_K as usize];
    let weights: Vec<f64> = kept.iter().map(|&t| (logits[t] / TEMPERATURE as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let mut cum = 0.0;
    let mut cut = probs.len();
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if cum > TOP_P as f64 {
            cut = i + 1;
            break;
        }
    }
    probs[cut..].iter_mut().for_each(|p| *p = 0.0);
    let mass: f64 = probs.iter().sum();
    let mut out = vec![0.0; VOCAB];
    for (&tok, p) in kept.iter().zip(probs) {
        out[tok] = p / mass;
    }
    out
}

fn assert_frequencies(tokens: &[u32], exact: &[f64], what: &str) {
    let n = tokens.len() as f64;
    let mut counts = vec![0usize; exact.len()];
    for &t in tokens {
        assert!((t as usize) < exact.len(), "{what}: token {t} is outside the vocabulary");
        counts[t as usize] += 1;
    }
    for (tok, (&c, &p)) in counts.iter().zip(exact).enumerate() {
        if p == 0.0 {
            assert_eq!(c, 0, "{what}: token {tok} has probability 0 and was drawn {c} times");
            continue;
        }
        let (mean, sd) = (n * p, (n * p * (1.0 - p)).sqrt());
        eprintln!("  token {tok:3}: {c:4} drawn, {mean:7.1} expected (sd {sd:.1})");
        assert!(
            (c as f64 - mean).abs() <= 5.0 * sd,
            "{what}: token {tok} was drawn {c} times of {n}; {mean:.1} expected (p = {p:.4}, sd {sd:.1})"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn decoding_samples_the_configured_distribution_with_and_without_a_draft() {
    if !cuda_available() {
        return;
    }
    const DRAWS: usize = 1500;
    let per_slot = PROMPT.len() + DRAWS + 8;
    reset_all();

    let exact = iid_exact_distribution();
    let kept: Vec<usize> = (0..VOCAB).filter(|&t| exact[t] > 0.0).collect();
    assert_eq!(kept, [3, 17, 40, 77, 100], "the nucleus of the fixture");

    let (h_t, st_t) = load(&iid_model());
    // A draft with ordinary random weights: its greedy guess is one fixed
    // function of the context and is wrong for most draws.
    let (h_d, st_d) = load(&build_toy_model(0xB7));
    setup_spec_engine(per_slot, Target::Nucleus);
    bind_target(h_t);
    bind_draft(h_d, per_slot);

    let (rc, plain) = run_generate_as(0, DRAWS, 9999);
    assert_eq!(rc, DRAWS as i64);
    eprintln!("plain decoding, {DRAWS} tokens:");
    assert_frequencies(&plain, &exact, "plain decoding");
    let distinct: std::collections::BTreeSet<u32> = plain.iter().copied().collect();
    assert_eq!(distinct.len(), 5, "all five nucleus tokens appear: {distinct:?}");

    // The same request through the speculative driver: the same tokens.
    let (rc, spec) = run_speculative_as(0, DRAWS, 9999);
    assert_eq!(rc, DRAWS as i64);
    assert_eq!(spec, plain, "speculative decoding must emit plain decoding's tokens");

    // Another request with the same seed is another sample of the same
    // distribution, not a replay; the same request again is a replay.
    let (_, other) = run_speculative_as(1, DRAWS, 9999);
    assert_ne!(other, plain, "request 1 replayed request 0");
    eprintln!("speculative decoding, request 1:");
    assert_frequencies(&other, &exact, "speculative decoding, second request");
    let (_, again) = run_generate_as(0, DRAWS, 9999);
    assert_eq!(again, plain, "a (seed, request) pair is reproducible");

    teardown(&[h_t, h_d]);
    drop((st_t, st_d));
}

// ---- (b) speculative == plain --------------------------------------------

#[test]
#[ignore = "requires CUDA GPU"]
fn self_speculation_matches_plain_decoding_exactly() {
    if !cuda_available() {
        return;
    }
    let per_slot = 16usize;
    for target in [Target::Greedy, Target::Nucleus] {
        reset_all();
        let (handle, st) = load(&build_toy_model(0xA0));
        setup_spec_engine(per_slot, target);
        bind_target(handle);
        bind_draft(handle, per_slot); // SAME model

        let max_new = 6usize;
        let (g_rc, g_toks) = run_generate(max_new, 9999);
        assert_eq!(g_rc, max_new as i64);
        let (s_rc, s_toks) = run_speculative(max_new, 9999);
        eprintln!("plain generate:      {g_toks:?}");
        eprintln!("self-speculative(1): {s_toks:?}");
        assert_eq!(s_rc, max_new as i64, "speculative must return max_new");
        assert_eq!(s_toks, g_toks, "speculative output must EXACTLY equal generate");

        // Second run: reproducible AND the single-flight guard released.
        let (s_rc2, s_toks2) = run_speculative(max_new, 9999);
        assert_eq!(s_rc2, max_new as i64);
        assert_eq!(s_toks2, g_toks, "second speculative run must reproduce");

        teardown(&[handle]);
        drop(st);
    }
}

/// DRAFT != TARGET: the output is still plain decoding's, for a greedy
/// target and for a sampling one, and the draft really mispredicts.  (The
/// old driver's output with a different draft was checked only against a
/// CPU copy of its own acceptance rule.)
#[test]
#[ignore = "requires CUDA GPU"]
fn a_different_draft_does_not_change_the_output() {
    if !cuda_available() {
        return;
    }
    let per_slot = 64usize;
    for target in [Target::Greedy, Target::Nucleus] {
        reset_all();
        let target_model = build_toy_model(0xA0);
        let draft_model = build_toy_model(0xB7); // different weights, same dims
        let (h_t, st_t) = load(&target_model);
        let (h_d, st_d) = load(&draft_model);
        setup_spec_engine(per_slot, target);
        bind_target(h_t);
        bind_draft(h_d, per_slot);

        let max_new = 24usize;
        let (g_rc, plain) = run_generate(max_new, 9999);
        assert_eq!(g_rc, max_new as i64);
        let (rc, spec) = run_speculative(max_new, 9999);
        assert_eq!(rc, max_new as i64, "speculative must return max_new");
        eprintln!("plain:       {plain:?}");
        eprintln!("speculative: {spec:?}");
        assert_eq!(spec, plain, "a different draft changed the output");

        let wrong = draft_mispredictions(&cpu_model(&draft_model), &block_cfg(per_slot, 1), &PROMPT, &spec);
        eprintln!("the draft would mispredict {wrong} of {} tokens", spec.len() - 1);
        assert!(wrong >= 1, "the fixture must make the draft mispredict (change the draft seed)");

        teardown(&[h_t, h_d]);
        drop((st_t, st_d));
    }
}

// ---- (c) EOS and capacity ------------------------------------------------

/// EOS mid-sequence stops the speculative path with exactly the truncated
/// prefix.
#[test]
#[ignore = "requires CUDA GPU"]
fn speculative_eos_stops_early() {
    if !cuda_available() {
        return;
    }
    let per_slot = 16usize;
    reset_all();
    let (handle, st) = load(&build_toy_model(0xA0));
    setup_spec_engine(per_slot, Target::Greedy);
    bind_target(handle);
    bind_draft(handle, per_slot);

    // Reference (eos unreachable), then re-run with eos = a token whose
    // first occurrence is past the start.
    let max_new = 6usize;
    let (_, ref_toks) = run_speculative(max_new, 9999);
    assert!(ref_toks.len() >= 2);
    let eos = ref_toks[1] as i64;
    let stop = ref_toks.iter().position(|&t| t as i64 == eos).unwrap() + 1;
    let (rc, toks) = run_speculative(max_new, eos);
    eprintln!("EOS-stop tokens: {toks:?} (eos={eos})");
    assert_eq!(rc, stop as i64, "EOS must stop at its first occurrence, emitted");
    assert_eq!(toks, ref_toks[..stop].to_vec());

    teardown(&[handle]);
    drop(st);
}

/// CAPACITY EDGE: a slot of 6 rows leaves room for one K=3 round after
/// the 2-token prompt, so the tail is finished with plain steps — the
/// draft never writes past its pool, and the output still equals plain
/// decoding truncated at capacity.
#[test]
#[ignore = "requires CUDA GPU"]
fn speculative_capacity_tail_matches_plain_decoding() {
    if !cuda_available() {
        return;
    }
    let per_slot = 6usize;
    for target in [Target::Greedy, Target::Nucleus] {
        reset_all();
        let (h_t, st_t) = load(&build_toy_model(0xA0));
        let (h_d, st_d) = load(&build_toy_model(0xB7));
        setup_spec_engine(per_slot, target);
        bind_target(h_t);
        bind_draft(h_d, per_slot);

        // max_new 8 wants 8 tokens; capacity supports fewer — BOTH paths
        // must truncate identically.
        let max_new = 8usize;
        let (g_rc, g_toks) = run_generate(max_new, 9999);
        assert!((g_rc as usize) < max_new, "fixture must actually hit capacity (got {g_rc} of {max_new})");
        // 6 rows: positions 0..5 are fed; the first new token is drawn at
        // position 1, the last at position 5: five tokens.
        assert_eq!(g_rc, 5, "one token per row from the last prompt row on");
        let (s_rc, s_toks) = run_speculative(max_new, 9999);
        eprintln!("plain generate @cap6:  rc={g_rc} {g_toks:?}");
        eprintln!("speculative @cap6:     rc={s_rc} {s_toks:?}");
        assert_eq!((s_rc, &s_toks), (g_rc, &g_toks), "capacity-truncated speculative output must equal generate's");

        teardown(&[h_t, h_d]);
        drop((st_t, st_d));
    }
}
