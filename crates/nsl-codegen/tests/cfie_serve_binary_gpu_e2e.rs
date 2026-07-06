//! CFIE Cycle 12: THE FULLY-FUNCTIONAL PROOF — the paper's serve /
//! @endpoint / generate() syntax, built into a real executable by the
//! real CLI, RUN on the local GPU, producing decoded TEXT that exactly
//! matches an independent CPU reference.
//!
//! Pipeline under test (nothing mocked):
//!
//!   .nsl serve program (weights:/tokenizer:/prompt:/... keys)
//!     -> `cargo run -p nsl-cli --features cuda -- build` (real CLI,
//!        cuda-featured runtime staticlib linked — the M1-60 r4
//!        mechanism that made GPU programs run)
//!     -> the produced executable runs: serve init registers the
//!        compiled kernels, finalizes the engine, loads + binds the toy
//!        safetensors model, loads the REAL tokenizer, runtime-encodes
//!        the prompt (nsl_tokenizer_encode -> nsl_cfie_tensor_to_tokens),
//!        drives nsl_cfie_generate on the GPU, then decodes the
//!        generated ids (nsl_cfie_tokens_to_tensor ->
//!        nsl_tokenizer_decode) and prints the TEXT + the count.
//!
//! Parity: the expected text is computed in-process — the Cycle-10 CPU
//! reference chain (embed gather + cfie_persistent_ptx::cpu_reference
//! per layer + cfie_sample_ptx::cpu_reference per token over
//! f16-rounded weights) produces the expected token ids, and the SAME
//! tokenizer decodes them through the SAME runtime FFIs.  Exact string
//! equality is REQUIRED: greedy argmax over f16 weights is a discrete
//! reproducible function, and WordLevel decode is deterministic — a
//! mismatch is a real bug (wrong prompt path, wrong seed, wrong
//! boundary), never something to loosen.
//!
//! The tokenizer is a hand-written minimal HF tokenizers JSON
//! (WordLevel, vocab t0..t127 -> ids 0..127, Whitespace pre-tokenizer)
//! so every generated id < the toy vocab (128) decodes to a real token
//! and the runtime-encoded prompt "t3 t17" is exactly [3, 17] — the
//! byte-baked fallback would be [116, 51, 32, ...], so a wrongly-taken
//! fallback path changes the generation and fails parity loudly.
//!
//! Run (GPU required, one file at a time — CUDA driver singleton):
//!
//!   cargo test -p nsl-codegen --features cuda \
//!     --test cfie_serve_binary_gpu_e2e -- --ignored --nocapture \
//!     --test-threads=1

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::ffi::CStr;
use std::path::PathBuf;
use std::process::Command;

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_persistent_ptx::{cpu_reference, DecodeBlockConfig};
use nsl_codegen::cfie_sample_ptx::cpu_reference as cpu_reference_sample;
use nsl_runtime::cfie::bridge::{nsl_cfie_tensor_to_tokens, nsl_cfie_tokens_to_tensor};
use nsl_runtime::nsl_cuda_init;
use nsl_runtime::string::nsl_string_free;
use nsl_runtime::tensor::nsl_tensor_free;
use nsl_runtime::tokenizer::{nsl_tokenizer_decode, nsl_tokenizer_encode, nsl_tokenizer_load};

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
// Toy model: copied from cfie_generate_gpu_e2e.rs (house per-file
// convention).  Same seeds => same weights => the serve config below
// MUST mirror these dims or the plan bakes different kernels.
// ---------------------------------------------------------------------------

const D_MODEL: usize = 64;
const HEAD_DIM: usize = 32;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const D_FF: usize = 128;
const N_LAYERS: usize = 2;
const VOCAB: usize = 128;
const TOP_K: u32 = 8;

/// Serve-config geometry: prompt 2 + max_new 4 = 6 steps fit per_slot 8.
const MAX_SEQ: usize = 8;
const MAX_BATCH: usize = 2;
const MAX_NEW: usize = 4;
const EOS: i64 = 999; // outside the 128-id vocab — generation runs full max_new
const PROMPT_TEXT: &str = "t3 t17";
const PROMPT_IDS: [i64; 2] = [3, 17];

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (seed >> 32) as u32;
        *x = ((u as f64) / 4294967296.0) as f32 - 0.5;
    }
}

fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

fn block_cfg() -> DecodeBlockConfig {
    DecodeBlockConfig {
        d_model: D_MODEL as u32,
        head_dim: HEAD_DIM as u32,
        n_heads: N_HEADS as u32,
        n_kv_heads: N_KV_HEADS as u32,
        d_ff: D_FF as u32,
        per_slot_max_tokens: MAX_SEQ as u32,
        max_slots: MAX_BATCH as u32,
        n_layers: N_LAYERS as u32,
        rope_theta: 10000.0,
        eps: 1e-5,
        sm_version: 80, // target_gpu "a100"; driver JITs sm_80 PTX forward
    }
}

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

fn ln(name: &str, i: usize) -> String {
    format!("model.layers.{i}.{name}")
}

/// Serialize the toy model to `<dir>/model.safetensors` with the
/// HF-Llama names nsl_cfie_bind_model resolves (f32 [out][in]).
fn write_toy_safetensors(m: &ToyModel, dir: &std::path::Path) -> PathBuf {
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
    let path = dir.join("model.safetensors");
    std::fs::write(&path, &serialized).expect("write safetensors");
    path
}

// ---------------------------------------------------------------------------
// Tokenizer: a minimal REAL HF tokenizers JSON (WordLevel) whose ids
// cover the toy vocab exactly — t{i} -> id i for i in 0..128.  Loaded by
// the runtime's own nsl_tokenizer_load in the built binary AND in this
// test process (same surface, same semantics).
// ---------------------------------------------------------------------------

fn write_tokenizer_json(dir: &std::path::Path) -> PathBuf {
    let vocab: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
    let json = format!(
        r#"{{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {{ "type": "Whitespace" }},
  "post_processor": null,
  "decoder": null,
  "model": {{
    "type": "WordLevel",
    "vocab": {{ {} }},
    "unk_token": "t0"
  }}
}}"#,
        vocab.join(", ")
    );
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, json).expect("write tokenizer.json");
    path
}

// ---------------------------------------------------------------------------
// CPU reference generation: the same loop nsl_cfie_generate drives
// (copied from cfie_generate_gpu_e2e.rs).
// ---------------------------------------------------------------------------

fn sampler_program() -> nsl_codegen::cfie_fused_sample::FusedSampleProgram {
    // MUST mirror what the serve plan derives from the sampling: section
    // below — temperature 0.0 -> Greedy (cfie_serve::sampling_params),
    // top_k 8, defaults elsewhere; kernel re-emitted at vocab_tile 128.
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
// In-process runtime tokenizer helpers (the SAME FFIs the binary calls).
// ---------------------------------------------------------------------------

fn load_tokenizer(path: &std::path::Path) -> i64 {
    let c = std::ffi::CString::new(path.to_str().expect("utf8 path")).unwrap();
    nsl_tokenizer_load(c.as_ptr() as i64)
}

fn encode_ids(tok: i64, text: &str) -> Vec<i64> {
    let c = std::ffi::CString::new(text).unwrap();
    let enc = nsl_tokenizer_encode(tok, c.as_ptr() as i64);
    assert!(enc != 0, "tokenizer encode returned null tensor");
    let mut buf = [0i64; 64];
    let n = nsl_cfie_tensor_to_tokens(enc, buf.as_mut_ptr() as i64, buf.len() as i64);
    nsl_tensor_free(enc);
    assert!(
        n > 0 && (n as usize) <= buf.len(),
        "encode -> tensor_to_tokens must produce 1..=64 ids (got {n})"
    );
    buf[..n as usize].to_vec()
}

fn decode_text(tok: i64, ids: &[i64]) -> String {
    let t = nsl_cfie_tokens_to_tensor(ids.as_ptr() as i64, ids.len() as i64);
    assert!(t != 0, "tokens_to_tensor refused valid ids");
    let text_ptr = nsl_tokenizer_decode(tok, t);
    assert!(text_ptr != 0, "tokenizer decode returned null");
    let s = unsafe { CStr::from_ptr(text_ptr as *const std::os::raw::c_char) }
        .to_str()
        .expect("decode utf8")
        .to_string();
    nsl_string_free(text_ptr);
    nsl_tensor_free(t);
    s
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

/// Workspace root: crates/nsl-codegen -> crates -> root.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Forward slashes so the path drops into an NSL string literal without
/// escapes (the grammar-e2e precedent in cfie_serve_end_to_end.rs).
fn nsl_path(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "/")
}

/// The serve program: the paper's syntax, toy-model shape keys, and the
/// Cycle-12 endpoint wiring (weights/tokenizer/prompt/...).  Greedy
/// sampling (temperature 0.0) keeps the parity discrete; target_gpu
/// "a100" bakes sm_80 PTX the local RTX 5070 Ti driver JITs forward
/// (the GPU parity tests' precedent).
fn serve_program(weights: &std::path::Path, tokenizer: &std::path::Path) -> String {
    format!(
        r#"serve ToyServe:
    max_batch: {MAX_BATCH}
    max_seq: {MAX_SEQ}
    kv_layout: "static"
    kv_quant: "uniform_fp16"
    target_gpu: "a100"

    n_layers: {N_LAYERS}
    n_heads: {N_HEADS}
    n_kv_heads: {N_KV_HEADS}
    head_dim: {HEAD_DIM}
    d_model: {D_MODEL}
    d_ff: {D_FF}
    vocab_size: {VOCAB}

    weights: "{weights}"
    tokenizer: "{tokenizer}"
    prompt: "{PROMPT_TEXT}"
    max_new_tokens: {MAX_NEW}
    eos_token_id: {EOS}

    sampling:
        temperature: 0.0
        top_k: {TOP_K}
        fused: true

    @endpoint
    fn complete(prompt: str) -> str:
        let count = generate("ToyServe", prompt, 0)
        print(count)
"#,
        weights = nsl_path(weights),
        tokenizer = nsl_path(tokenizer),
    )
}

/// Build the serve program with the REAL CLI, cuda-featured so the
/// linked runtime staticlib can actually drive the GPU (nsl-cli's
/// `cuda` feature -> nsl-codegen/cuda -> nsl-runtime/cuda; linker.rs
/// then selects the cuda-fingerprinted nsl_runtime staticlib).
fn build_serve_binary(prog: &std::path::Path, exe: &std::path::Path) -> std::process::Output {
    let root = workspace_root();
    Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", "cuda", "--"])
        .arg("build")
        .arg(prog)
        .arg("-o")
        .arg(exe)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn cargo run -p nsl-cli")
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA GPU + builds the CLI"]
fn built_serve_binary_generates_expected_text_on_gpu() {
    if !cuda_available() {
        return;
    }

    // Temp workspace for model + tokenizer + program + executable.
    let dir = std::env::temp_dir().join(format!(
        "nsl_cfie_serve_binary_e2e_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let model = build_toy_model();
    let weights_path = write_toy_safetensors(&model, &dir);
    let tok_path = write_tokenizer_json(&dir);

    // ---- Expected output, computed independently in-process ----
    // 1. The prompt the binary will runtime-encode: same tokenizer, same
    //    runtime FFI chain (encode -> tensor_to_tokens).
    let tok = load_tokenizer(&tok_path);
    let prompt_ids = encode_ids(tok, PROMPT_TEXT);
    assert_eq!(
        prompt_ids,
        PROMPT_IDS.to_vec(),
        "WordLevel vocab must encode '{PROMPT_TEXT}' to exactly {PROMPT_IDS:?}"
    );
    // 2. CPU reference generation (seed 0 = the compiled generate()'s
    //    baked demo seed; greedy sampling makes it seed-independent).
    let cpu_tokens = cpu_generate(&model, &prompt_ids, MAX_NEW, EOS, 0);
    assert_eq!(
        cpu_tokens.len(),
        MAX_NEW,
        "reference must run the full max_new (EOS {EOS} is unreachable)"
    );
    assert!(
        cpu_tokens.iter().all(|&t| (t as usize) < VOCAB),
        "reference ids must stay inside the toy vocab: {cpu_tokens:?}"
    );
    // 3. Decode the reference ids through the SAME runtime tokenizer.
    let cpu_ids_i64: Vec<i64> = cpu_tokens.iter().map(|&t| t as i64).collect();
    let expected_text = decode_text(tok, &cpu_ids_i64);
    eprintln!("CPU reference tokens: {cpu_tokens:?}");
    eprintln!("expected decoded text: {expected_text:?}");
    assert!(
        !expected_text.is_empty(),
        "full-coverage vocab must decode to non-empty text"
    );

    // ---- Build the serve binary with the real CLI ----
    let prog_path = dir.join("toy_serve.nsl");
    std::fs::write(&prog_path, serve_program(&weights_path, &tok_path))
        .expect("write serve program");
    let exe_path = dir.join(if cfg!(windows) {
        "toy_serve_bin.exe"
    } else {
        "toy_serve_bin"
    });
    let build = build_serve_binary(&prog_path, &exe_path);
    assert!(
        build.status.success(),
        "nsl build failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(exe_path.exists(), "built executable missing");
    // The compile-time CFIE report must tell the Cycle-12 truth.
    let build_err = String::from_utf8_lossy(&build.stderr);
    assert!(
        build_err.contains("runtime-encoded"),
        "build report must state the runtime-encoded prompt + text decode:\n{build_err}"
    );

    // ---- RUN it on the GPU ----
    let run = Command::new(&exe_path)
        .output()
        .expect("run built serve binary");
    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    eprintln!("--- binary stdout ---\n{stdout}");
    eprintln!("--- binary stderr ---\n{stderr}");
    assert!(
        run.status.success(),
        "serve binary must exit successfully (status {:?})",
        run.status
    );
    // No CFIE refusal anywhere on stderr (refused/refusing).
    assert!(
        !stderr.contains("refus"),
        "binary stderr must contain no CFIE refusal lines:\n{stderr}"
    );

    // ---- THE PROOF: decoded text + count, exactly as predicted ----
    let lines: Vec<&str> = stdout.lines().collect();
    let text_line = lines
        .iter()
        .position(|l| *l == expected_text)
        .unwrap_or_else(|| {
            panic!(
                "stdout must contain the decoded text line {expected_text:?}; got:\n{stdout}"
            )
        });
    let count_line = lines
        .iter()
        .position(|l| *l == MAX_NEW.to_string())
        .unwrap_or_else(|| {
            panic!("stdout must contain the count line \"{MAX_NEW}\"; got:\n{stdout}")
        });
    assert!(
        text_line < count_line,
        "generate() prints the decoded text before the endpoint prints the count"
    );

    // Cleanup (temp artifacts only; the tokenizer handle is a
    // process-lifetime Box by the runtime's handle convention).
    let _ = std::fs::remove_dir_all(&dir);
}
