//! CFIE Cycle 10: GPU end-to-end proof of the host generation driver.
//!
//! The flagship runtime piece — a prompt + a bound model turned into
//! GENERATED tokens on the GPU.  Drives the REAL binding + generation
//! path:
//!
//!   synthetic weights -> temp .safetensors (HF-Llama names) ->
//!   nsl_model_create -> register kinds 1+2 -> kv_slots_init ->
//!   kv_pool_alloc -> finalize -> nsl_cfie_bind_model -> nsl_cfie_generate.
//!
//! The generation loop, per the ABI, gathers the input token's embedding
//! row (host f32) into the x_a device buffer BEFORE each decode_step (the
//! decode-block kernel consumes a hidden-state vector, NOT a token id —
//! confirmed against cfie_persistent_ptx.rs: `x_in` is the residual
//! stream, no embedding lookup on-device).
//!
//! Proofs:
//!   1. FLAGSHIP: nsl_cfie_generate(prompt_len=2, max_new_tokens=4) writes
//!      a deterministic generated token sequence; it must EXACTLY equal a
//!      CPU reference that drives the SAME loop (embed gather +
//!      cfie_persistent_ptx::cpu_reference per layer per pos +
//!      cfie_sample_ptx::cpu_reference per token, feeding f16-rounded
//!      weights — the exact rounding nsl_cfie_bind_model applies via
//!      `half::f16::from_f32`).
//!   2. EOS mid-generation stops early: set eos to the token the reference
//!      emits at generated-index 1; generate must stop with exactly 2
//!      tokens written.
//!   3. out_cap smaller than the true generated count clamps writes
//!      without overrunning the output buffer, while the return value is
//!      the TRUE generated count.
//!   4. generate_reset + weights_reset + re-bind reproduces the exact same
//!      sequence (binding is reconstructible; no stale state leaks).
//!
//! Exact token equality is REQUIRED (greedy argmax of an f16 lm-head is a
//! discrete, reproducible function of the f16-rounded weights + the
//! host-exact embedding gather); a mismatch is a real bug in embed-gather
//! / prefill-boundary / EOS-index / cast, never something to loosen.
//!
//! Run (GPU required, one file at a time — CUDA driver singleton +
//! process-global CFIE engine):
//!
//!   cargo test -p nsl-codegen --features cuda --test cfie_generate_gpu_e2e \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, emit, DecodeBlockConfig};
use nsl_codegen::cfie_sample_ptx::{
    cpu_reference as cpu_reference_sample, emit as emit_sampler, FusedSampleKernelConfig,
};
use nsl_runtime::cfie::engine::{
    nsl_cfie_bind_model, nsl_cfie_engine_destroy, nsl_cfie_engine_finalize, nsl_cfie_generate,
    nsl_cfie_generate_reset, nsl_cfie_kv_pool_alloc, nsl_cfie_register_kernel,
    nsl_cfie_weights_reset,
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

// ---------------------------------------------------------------------------
// f16 rounding: MUST match nsl_cfie_upload_weight_f16, which rounds via
// `half::f16::from_f32(x).to_bits()`.  The CPU reference consumes the same
// f16-rounded f32 masters the GPU kernel reads from the uploaded buffers.
// ---------------------------------------------------------------------------

fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (corrected ZERO-MEAN mapping, copied verbatim from
// cfie_decode_block_gpu_parity.rs — the -0.25-biased precedent saturates
// every SwiGLU gate in a decode block).
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
// Synthetic toy model: 2 layers, small dims (mirrors the decode_step GPU
// parity fixture so the shapes exercise GQA + a non-trivial FFN + a small
// vocab).  Per-slot 8 tokens so the 6-step (prompt 2 + new 4) loop never
// hits the capacity refusal in the flagship path.
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const D_FF: usize = 128;
const PER_SLOT: usize = 8;
const MAX_SLOTS: usize = 2;
const N_LAYERS: usize = 2;
const VOCAB: usize = 128;
const TOP_K: u32 = 8;

// KV pool: [n_layers][2][max_tokens][n_kv_heads][head_dim] f16.
const POOL_BYTES: usize = N_LAYERS * 2 * (MAX_SLOTS * PER_SLOT) * N_KV_HEADS * HEAD_DIM * 2;

fn block_cfg() -> DecodeBlockConfig {
    DecodeBlockConfig {
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        d_ff: D_FF as u32,
        per_slot_max_tokens: PER_SLOT as u32,
        max_slots: MAX_SLOTS as u32,
        n_layers: N_LAYERS as u32,
        rope_theta: 10000.0,
        eps: 1e-5,
        sm_version: 80, // driver JITs sm_80 PTX forward to the local Blackwell
    }
}

/// One layer's host f32 master weights (the decode_block parity fixture's
/// scaling: ~1/sqrt(fan_in) matmuls + norm gammas near 1.0, so the
/// residual stream stays O(1) and the greedy argmax is well-separated).
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

/// The f16-rounded copy of a layer's weights the CPU reference consumes
/// (the 7 matmuls round to f16; the two norm gammas pass through exact —
/// the kernel loads them as f32).
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

/// One CPU decode_block pass (f16-rounded weights, exact norms) through
/// cfie_persistent_ptx::cpu_reference, advancing this layer's KV chain.
#[allow(clippy::too_many_arguments)]
fn cpu_block(
    cfg: &DecodeBlockConfig,
    x: &[f32],
    w: &RoundedLayer,
    kv_k: &mut Vec<f32>,
    kv_v: &mut Vec<f32>,
    pos: u32,
) -> Vec<f32> {
    cpu_reference(
        cfg, x, &w.wq, &w.wk, &w.wv, &w.wo, &w.w_gate, &w.w_up, &w.w_down, &w.norm1, &w.norm2,
        kv_k, kv_v, pos,
    )
}

// ---------------------------------------------------------------------------
// The whole synthetic model: 2 layers + final norm + lm_head + embed.
// ---------------------------------------------------------------------------

struct ToyModel {
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>, // [vocab][d_model]
    embed: Vec<f32>,   // [vocab][d_model]
}

fn build_toy_model() -> ToyModel {
    let layers = vec![seeded_layer_weights(0xA0), seeded_layer_weights(0xA1)];
    let mut final_norm = vec![0f32; D_MODEL];
    fill_seeded(&mut final_norm, 0xF1);
    for g in final_norm.iter_mut() {
        *g = 1.0 + 0.1 * *g;
    }
    let mut lm_head = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut lm_head, 0xF2);
    // Embedding rows kept O(1) so the residual stream after the first
    // block is well-conditioned (same 0.35-ish scale as the block inputs
    // the parity fixture seeds).
    let mut embed = vec![0f32; VOCAB * D_MODEL];
    fill_seeded(&mut embed, 0xE3);
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

/// Layer weight name for the HF-Llama convention bind_model resolves.
fn ln(name: &str, i: usize) -> String {
    format!("model.layers.{i}.{name}")
}

/// Serialize the toy model to a temp .safetensors with the HF-Llama names
/// nsl_cfie_bind_model expects.  Weights are f32 [out][in] row-major
/// (safetensors/PyTorch nn.Linear convention).
fn write_toy_safetensors(m: &ToyModel) -> tempfile::TempPath {
    // Owned (name, le-bytes, shape) tuples; TensorView borrows the bytes.
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
        .prefix("cfie_toy_model_")
        .suffix(".safetensors")
        .tempfile()
        .expect("tempfile");
    std::fs::write(tmp.path(), &serialized).expect("write safetensors");
    tmp.into_temp_path()
}

// ---------------------------------------------------------------------------
// Sampler program shared by the GPU sampler + the CPU reference.
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

/// Register kinds 1 (fused_sample) + 2 (decode_block), init slots, alloc
/// the pool, finalize.  Returns the decode-block PTX (for a JIT-log panic
/// on a finalize failure).
fn setup_engine() -> String {
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    let (blk_ptx, blk_meta) = emit(&block_cfg());
    let rc = nsl_cfie_register_kernel(
        2, // decode_block
        0,
        blk_ptx.as_ptr() as i64,
        blk_ptx.len() as i64,
        blk_meta.kernel_name.as_ptr() as i64,
        blk_meta.kernel_name.len() as i64,
        1,
        blk_meta.block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 2) refused");

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
    let rc = nsl_cfie_register_kernel(
        1, // fused_sample
        0,
        s_ptx.as_ptr() as i64,
        s_ptx.len() as i64,
        s_meta.kernel_name.as_ptr() as i64,
        s_meta.kernel_name.len() as i64,
        1,
        s_meta.block_dim as i64,
        0,
    );
    assert_eq!(rc, 0, "register_kernel(kind 1) refused");

    assert_eq!(nsl_cfie_kv_slots_init(MAX_SLOTS as i64, PER_SLOT as i64), 0);
    assert_eq!(nsl_cfie_kv_pool_alloc(POOL_BYTES as i64), 0);
    let n = nsl_cfie_engine_finalize();
    assert_eq!(n, 2, "finalize must resolve both kinds (got {n})");
    blk_ptx
}

fn bind(model_handle: i64) {
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

// ---------------------------------------------------------------------------
// CPU reference generation: the SAME loop nsl_cfie_generate drives.
//
// For pos in 0..(prompt_len + max_new): input = prompt[pos] while
// prefilling, else the last CPU-sampled token; gather embed[input] (exact
// host f32) as the hidden state; run both decode_block layers (f16-rounded
// weights) advancing per-layer KV chains; sample with the fused-sample
// reference (f16-rounded lm_head, exact final norm).  Recording starts at
// pos == prompt_len - 1 (the last prefill step produces the first new
// token), with EOS + max_new_tokens stops mirroring the FFI.
// ---------------------------------------------------------------------------

fn cpu_generate(
    m: &ToyModel,
    prompt: &[i64],
    max_new_tokens: usize,
    eos: i64,
    rng_seed: u64,
) -> Vec<u32> {
    let cfg = block_cfg();
    let rounded: Vec<RoundedLayer> = m.layers.iter().map(round_layer).collect();
    let lm_rounded: Vec<f32> = m.lm_head.iter().map(|&x| f16_round(x)).collect();
    let program = sampler_program();

    let mut kv: Vec<(Vec<f32>, Vec<f32>)> =
        (0..N_LAYERS).map(|_| (Vec::new(), Vec::new())).collect();

    let mut generated: Vec<u32> = Vec::new();
    let mut last_sampled: i64 = 0;
    let prompt_len = prompt.len();
    let total_steps = prompt_len + max_new_tokens;
    for pos in 0..total_steps {
        let input_token = if pos < prompt_len {
            prompt[pos]
        } else {
            last_sampled
        };
        // Embedding gather: exact host f32 row (no rounding — matches the
        // generate driver's host->device memcpy).
        let row_start = input_token as usize * D_MODEL;
        let mut x = m.embed[row_start..row_start + D_MODEL].to_vec();
        for l in 0..N_LAYERS {
            let (kv_k, kv_v) = &mut kv[l];
            x = cpu_block(&cfg, &x, &rounded[l], kv_k, kv_v, pos as u32);
        }
        let sampled = cpu_reference_sample(
            &program,
            &x,
            &m.final_norm,
            &lm_rounded,
            None,
            rng_seed,
        );
        if pos >= prompt_len - 1 {
            generated.push(sampled);
            if sampled as i64 == eos {
                break;
            }
            if generated.len() >= max_new_tokens {
                break;
            }
        }
        last_sampled = sampled as i64;
    }
    generated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const RNG_SEED: i64 = 0x5EED_1234;
const PROMPT: [i64; 2] = [3, 17];

#[test]
#[ignore = "requires CUDA GPU"]
fn generate_token_sequence_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }
    // Clean slate.
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);

    let model = build_toy_model();
    let st_path = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st_path.to_str().expect("utf8 path")).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0, "nsl_model_create returned 0 (load failed)");

    setup_engine();
    bind(handle);

    let max_new = 4usize;
    let eos = 9999i64; // unused id — generation runs the full max_new
    let mut out = [0i64; 8];
    let got = nsl_cfie_generate(
        PROMPT.as_ptr() as i64,
        PROMPT.len() as i64,
        max_new as i64,
        eos,
        RNG_SEED,
        out.as_mut_ptr() as i64,
        out.len() as i64,
    );
    assert_eq!(got, max_new as i64, "generate must return max_new tokens");
    let gpu_tokens: Vec<u32> = out[..max_new].iter().map(|&t| t as u32).collect();

    let cpu_tokens = cpu_generate(&model, &PROMPT, max_new, eos, RNG_SEED as u64);
    eprintln!("GPU generated: {gpu_tokens:?}");
    eprintln!("CPU reference: {cpu_tokens:?}");
    assert_eq!(
        gpu_tokens, cpu_tokens,
        "generated token sequence must EXACTLY equal the CPU reference"
    );

    nsl_runtime::c_api::nsl_model_destroy(handle);
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    drop(st_path);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn eos_mid_generation_stops_early() {
    if !cuda_available() {
        return;
    }
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);

    let model = build_toy_model();
    let st_path = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st_path.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);
    setup_engine();
    bind(handle);

    // The unconstrained reference sequence (eos unreachable).
    let max_new = 4usize;
    let ref_seq = cpu_generate(&model, &PROMPT, max_new, 9999, RNG_SEED as u64);
    assert!(
        ref_seq.len() >= 2,
        "need at least 2 reference tokens to test a mid-sequence EOS"
    );
    // Set EOS to the token the reference emits at generated-index 1: the
    // driver records index 0, then index 1 == eos and breaks -> exactly 2
    // tokens written (the eos token is recorded, then generation stops).
    let eos = ref_seq[1] as i64;

    let mut out = [0i64; 8];
    let got = nsl_cfie_generate(
        PROMPT.as_ptr() as i64,
        PROMPT.len() as i64,
        max_new as i64,
        eos,
        RNG_SEED,
        out.as_mut_ptr() as i64,
        out.len() as i64,
    );
    assert_eq!(
        got, 2,
        "EOS at generated-index 1 must stop generation with exactly 2 tokens"
    );
    assert_eq!(out[0] as u32, ref_seq[0], "first token unchanged");
    assert_eq!(out[1] as u32, ref_seq[1], "second token is the EOS token");
    eprintln!("EOS-stop tokens: [{}, {}] (eos={eos})", out[0], out[1]);

    nsl_runtime::c_api::nsl_model_destroy(handle);
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    drop(st_path);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn out_cap_smaller_than_generated_clamps_without_overrun() {
    if !cuda_available() {
        return;
    }
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);

    let model = build_toy_model();
    let st_path = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st_path.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);
    setup_engine();
    bind(handle);

    let max_new = 4usize;
    let ref_seq = cpu_generate(&model, &PROMPT, max_new, 9999, RNG_SEED as u64);
    assert_eq!(ref_seq.len(), max_new, "reference must run the full max_new");

    // out_cap = 2 < 4 generated.  A sentinel guard word past the capacity
    // catches an overrun: it must be untouched, the first 2 slots hold the
    // first 2 tokens, and the return value is the TRUE count (4).
    let out_cap = 2usize;
    let mut out = [0i64; 3]; // [0..2) = capacity, [2] = overrun sentinel
    let sentinel = -424242i64;
    out[out_cap] = sentinel;
    let got = nsl_cfie_generate(
        PROMPT.as_ptr() as i64,
        PROMPT.len() as i64,
        max_new as i64,
        9999,
        RNG_SEED,
        out.as_mut_ptr() as i64,
        out_cap as i64,
    );
    assert_eq!(
        got, max_new as i64,
        "return value must be the TRUE generated count, not the written count"
    );
    assert_eq!(out[0] as u32, ref_seq[0], "clamped write slot 0");
    assert_eq!(out[1] as u32, ref_seq[1], "clamped write slot 1");
    assert_eq!(
        out[out_cap], sentinel,
        "no write past out_cap (overrun sentinel intact)"
    );
    eprintln!(
        "out_cap clamp: wrote [{}, {}], returned true count {got}",
        out[0], out[1]
    );

    nsl_runtime::c_api::nsl_model_destroy(handle);
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    drop(st_path);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn regenerate_after_reset_and_rebind_reproduces() {
    if !cuda_available() {
        return;
    }
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);

    let model = build_toy_model();
    let st_path = write_toy_safetensors(&model);
    let path_c = std::ffi::CString::new(st_path.to_str().unwrap()).unwrap();
    let handle = nsl_runtime::c_api::nsl_model_create(path_c.as_ptr() as i64);
    assert!(handle != 0);
    setup_engine();
    bind(handle);

    let max_new = 4usize;
    let run = || -> Vec<u32> {
        let mut out = [0i64; 8];
        let got = nsl_cfie_generate(
            PROMPT.as_ptr() as i64,
            PROMPT.len() as i64,
            max_new as i64,
            9999,
            RNG_SEED,
            out.as_mut_ptr() as i64,
            out.len() as i64,
        );
        assert_eq!(got, max_new as i64);
        out[..max_new].iter().map(|&t| t as u32).collect()
    };

    let first = run();
    // Reset the binding + free the uploaded weights, then re-bind from the
    // same model handle and regenerate: the sequence must reproduce.
    assert_eq!(nsl_cfie_generate_reset(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    bind(handle);
    let second = run();
    assert_eq!(
        first, second,
        "regeneration after reset + re-bind must reproduce the sequence"
    );
    // And it must still match the CPU reference.
    let cpu = cpu_generate(&model, &PROMPT, max_new, 9999, RNG_SEED as u64);
    assert_eq!(second, cpu, "reproduced sequence must match the CPU reference");
    eprintln!("reproduced sequence: {second:?}");

    nsl_runtime::c_api::nsl_model_destroy(handle);
    assert_eq!(nsl_cfie_engine_destroy(), 0);
    assert_eq!(nsl_cfie_weights_reset(), 0);
    assert_eq!(nsl_cfie_generate_reset(), 0);
    drop(st_path);
}
