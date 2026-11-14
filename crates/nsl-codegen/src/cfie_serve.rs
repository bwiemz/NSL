//! CFIE Tier-A wiring (audit gaps G1/G3/G4/G6): extract CFIE
//! configuration from a `serve` block, resolve the model/GPU inputs the
//! planners need from *real* sources (serve config keys, the loaded
//! `--weights` WeightMap, the gpu_specs database), and assemble the
//! [`crate::cfie::CfieInput`] the orchestrator consumes.
//!
//! Source priority for model-architecture inputs, per field:
//!   1. explicit serve config keys (`n_layers`, `n_kv_heads`, `head_dim`,
//!      `n_heads`, `d_model`, `d_ff`, `vocab_size`),
//!   2. inference from the `--weights` WeightMap (layer count from
//!      parameter names, vocab/d_model from the LM-head/embedding shape,
//!      total weight bytes for the KV budget),
//!   3. the M29 serve worker defaults (32 layers, 32 KV heads, 128
//!      head_dim) — flagged as `defaults` in the provenance string so
//!      the build report never silently passes them off as measured.

use nsl_ast::block::{ServeBlock, ServeConfigEntry, ServeSubBlock};
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use crate::cfie::{CfieInput, CfieMode};
use crate::cfie_fused_sample::{LmHeadShape, SamplingParams, SamplingStrategy};
use crate::cfie_kv_plan::{KvBudget, KvShape};
use crate::cfie_kv_quant::KvQuantConfig;
use crate::cfie_persistent::PersistentModel;
use crate::cfie_speculative::SpeculativeConfig;
use crate::error::CodegenError;
use crate::gpu_specs::GpuSpec;
use crate::weight_aware::WeightMap;

/// Paper default: CUDA driver + workspace reserve subtracted from HBM
/// before planning the KV budget (§2's worked example uses 800 MB).
const CUDA_RUNTIME_OVERHEAD_BYTES: u64 = 800 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Serve-block extraction
// ---------------------------------------------------------------------------

/// `sampling:` section values (paper §4).
#[derive(Debug, Clone, Default)]
pub struct SamplingSection {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub fused: Option<bool>,
    /// `seed: <int>` — the seed every request's draws are keyed by
    /// (with the request's index and the token's position).  Default 0.
    pub seed: Option<i64>,
}

/// `speculative:` section values (paper §5).
#[derive(Debug, Clone, Default)]
pub struct SpeculativeSection {
    pub draft_path: Option<String>,
    pub tokens: Option<i64>,
    pub method: Option<String>,
    pub tree_width: Option<i64>,
    pub temperature: Option<f64>,
    /// CFIE Cycle 13 (G15): the draft checkpoint compiled into the SAME
    /// binary (`draft_weights: "<path.safetensors>"`).  When set (and
    /// the spec kernels are emitted) the serve wiring emits the draft
    /// decode-block + kind-7/8 sampler registrations, the serve-init
    /// draft binding, and the `nsl_cfie_speculative_generate` driver.
    pub draft_weights: Option<String>,
    /// Draft model architecture keys (vocab is the target's — shared
    /// vocab is a speculative-decoding invariant).  Any key left unset
    /// defaults to the TARGET's resolved value; a mismatch against the
    /// actual draft checkpoint refuses loudly at serve init
    /// (`nsl_cfie_bind_draft_model` validates every weight's shape).
    pub draft_n_layers: Option<i64>,
    pub draft_d_model: Option<i64>,
    pub draft_n_heads: Option<i64>,
    pub draft_n_kv_heads: Option<i64>,
    pub draft_head_dim: Option<i64>,
    pub draft_d_ff: Option<i64>,
}

/// Everything CFIE-relevant found in one serve block.
#[derive(Debug, Clone, Default)]
pub struct CfieServeConfig {
    pub kv_layout: Option<String>,
    pub kv_quant: Option<String>,
    pub max_seq: Option<i64>,
    pub max_batch: Option<i64>,
    pub target_gpu: Option<String>,
    // Explicit architecture keys.
    pub n_layers: Option<i64>,
    pub n_kv_heads: Option<i64>,
    pub n_heads: Option<i64>,
    pub head_dim: Option<i64>,
    pub d_model: Option<i64>,
    pub d_ff: Option<i64>,
    pub vocab_size: Option<i64>,
    /// RoPE base for the decode-block kernel's baked rotation
    /// constants (default 10000.0 — Llama-2-family; set 500000.0 for
    /// Llama-3-family models or the baked angles are wrong).
    pub rope_theta: Option<f64>,
    /// RMSNorm epsilon baked into the decode-block kernel (default 1e-5).
    pub norm_eps: Option<f64>,
    pub sampling: Option<SamplingSection>,
    pub speculative: Option<SpeculativeSection>,
    pub grammar_schema: Option<String>,
    /// Tokenizer vocab path (`.txt` = one token per line, `.json` =
    /// string array) — required to token-project the schema (G12).
    pub grammar_tokenizer: Option<String>,
    // ── Cycle 11 endpoint-wiring keys ────────────────────────────────
    /// `weights: "<path.safetensors>"` — the model to serve.  Falls back
    /// to the `--weights` CLI path when absent (resolved in serve.rs).
    pub weights_path: Option<String>,
    /// `tokenizer: "<path.json>"` — optional; when present the demo
    /// decodes the generated tokens back to text via
    /// `nsl_tokenizer_load` + the endpoint-body `tokenizer_decode` call.
    pub tokenizer_path: Option<String>,
    /// `prompt: "<text>"` — optional demo prompt.  When a tokenizer is
    /// also configured the prompt is tokenized offline (compile time) so
    /// the baked token array feeds the runtime `nsl_cfie_generate`
    /// (whose prompt ABI is a host i64 array, NOT an f64 tokenizer
    /// tensor — see the Cycle 11 note in serve.rs).
    pub prompt: Option<String>,
    /// `max_new_tokens: <int>` — decode budget for the one-shot demo
    /// (default 64).
    pub max_new_tokens: Option<i64>,
    /// `eos_token_id: <int>` — stop token (default -1 = none).
    pub eos_token_id: Option<i64>,
}

impl CfieServeConfig {
    /// Whether this serve block opts into CFIE at all.  The paper's
    /// rule: omitting every CFIE knob leaves the M29 dynamic
    /// PagedAttention path untouched.
    pub fn requests_cfie(&self) -> bool {
        self.kv_layout.is_some()
            || self.kv_quant.is_some()
            || self
                .sampling
                .as_ref()
                .is_some_and(|s| s.fused.unwrap_or(false))
            || self.speculative.is_some()
            || self.grammar_schema.is_some()
    }
}

fn entry_string(
    entry: &ServeConfigEntry,
    resolve: &dyn Fn(Symbol) -> String,
) -> Option<String> {
    match &entry.value.kind {
        ExprKind::StringLiteral(s) => Some(s.clone()),
        ExprKind::Ident(sym) => Some(resolve(*sym)),
        _ => None,
    }
}

fn entry_int(entry: &ServeConfigEntry) -> Option<i64> {
    match &entry.value.kind {
        ExprKind::IntLiteral(v) => Some(*v),
        _ => None,
    }
}

fn entry_float(entry: &ServeConfigEntry) -> Option<f64> {
    match &entry.value.kind {
        ExprKind::FloatLiteral(f) => Some(*f),
        ExprKind::IntLiteral(v) => Some(*v as f64),
        _ => None,
    }
}

fn entry_bool(entry: &ServeConfigEntry) -> Option<bool> {
    match &entry.value.kind {
        ExprKind::BoolLiteral(b) => Some(*b),
        _ => None,
    }
}

/// Extract CFIE-relevant config from a parsed serve block.  Unknown
/// keys/sections are ignored here — nsl-semantic already diagnosed
/// them; codegen extraction is deliberately permissive so it never
/// double-reports.
pub fn extract(serve: &ServeBlock, resolve: &dyn Fn(Symbol) -> String) -> CfieServeConfig {
    let mut cfg = CfieServeConfig::default();

    for entry in &serve.config {
        match resolve(entry.key).as_str() {
            "kv_layout" => cfg.kv_layout = entry_string(entry, resolve),
            "kv_quant" => cfg.kv_quant = entry_string(entry, resolve),
            // Accept both the paper's `max_seq` and the M29 `max_seq_len`.
            "max_seq" | "max_seq_len" => cfg.max_seq = entry_int(entry),
            "max_batch" => cfg.max_batch = entry_int(entry),
            "target_gpu" => cfg.target_gpu = entry_string(entry, resolve),
            "n_layers" => cfg.n_layers = entry_int(entry),
            "n_kv_heads" | "kv_heads" => cfg.n_kv_heads = entry_int(entry),
            "n_heads" => cfg.n_heads = entry_int(entry),
            "head_dim" => cfg.head_dim = entry_int(entry),
            "d_model" => cfg.d_model = entry_int(entry),
            "d_ff" => cfg.d_ff = entry_int(entry),
            "vocab_size" => cfg.vocab_size = entry_int(entry),
            "rope_theta" => cfg.rope_theta = entry_float(entry),
            "norm_eps" => cfg.norm_eps = entry_float(entry),
            // Cycle 11 endpoint-wiring keys.
            "weights" => cfg.weights_path = entry_string(entry, resolve),
            "tokenizer" => cfg.tokenizer_path = entry_string(entry, resolve),
            "prompt" => cfg.prompt = entry_string(entry, resolve),
            "max_new_tokens" => cfg.max_new_tokens = entry_int(entry),
            "eos_token_id" => cfg.eos_token_id = entry_int(entry),
            _ => {}
        }
    }

    for sub in &serve.sub_blocks {
        match resolve(sub.key).as_str() {
            "sampling" => cfg.sampling = Some(extract_sampling(sub, resolve)),
            "speculative" => cfg.speculative = Some(extract_speculative(sub, resolve)),
            "grammar" => {
                for entry in &sub.entries {
                    match resolve(entry.key).as_str() {
                        "schema" => cfg.grammar_schema = entry_string(entry, resolve),
                        "tokenizer" => cfg.grammar_tokenizer = entry_string(entry, resolve),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    cfg
}

fn extract_sampling(sub: &ServeSubBlock, resolve: &dyn Fn(Symbol) -> String) -> SamplingSection {
    let mut s = SamplingSection::default();
    for entry in &sub.entries {
        match resolve(entry.key).as_str() {
            "temperature" => s.temperature = entry_float(entry),
            "top_k" => s.top_k = entry_int(entry),
            "top_p" => s.top_p = entry_float(entry),
            "fused" => s.fused = entry_bool(entry),
            "seed" => s.seed = entry_int(entry),
            _ => {}
        }
    }
    s
}

fn extract_speculative(
    sub: &ServeSubBlock,
    resolve: &dyn Fn(Symbol) -> String,
) -> SpeculativeSection {
    let mut s = SpeculativeSection::default();
    for entry in &sub.entries {
        match resolve(entry.key).as_str() {
            "draft" => s.draft_path = entry_string(entry, resolve),
            "tokens" => s.tokens = entry_int(entry),
            "method" => s.method = entry_string(entry, resolve),
            "tree_width" => s.tree_width = entry_int(entry),
            "temperature" => s.temperature = entry_float(entry),
            // Cycle 13 (G15): draft-model-in-binary keys.
            "draft_weights" => s.draft_weights = entry_string(entry, resolve),
            "draft_n_layers" => s.draft_n_layers = entry_int(entry),
            "draft_d_model" => s.draft_d_model = entry_int(entry),
            "draft_n_heads" => s.draft_n_heads = entry_int(entry),
            "draft_n_kv_heads" => s.draft_n_kv_heads = entry_int(entry),
            "draft_head_dim" => s.draft_head_dim = entry_int(entry),
            "draft_d_ff" => s.draft_d_ff = entry_int(entry),
            _ => {}
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Model-shape resolution (audit gap G6)
// ---------------------------------------------------------------------------

/// Model architecture with per-field provenance.
#[derive(Debug, Clone)]
pub struct ResolvedModelShape {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub n_heads: u32,
    pub head_dim: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub vocab_size: u32,
    pub dtype_bytes: u32,
    /// Total model weight bytes (from the WeightMap when available).
    pub weights_bytes: Option<u64>,
    /// Human-readable source summary, e.g.
    /// `"n_layers=weights, head_dim=serve-config, d_model=defaults"`.
    pub provenance: String,
}

/// M29 serve worker defaults, kept in sync with `serve.rs`.
const DEFAULT_N_LAYERS: u32 = 32;
const DEFAULT_N_KV_HEADS: u32 = 32;
const DEFAULT_HEAD_DIM: u32 = 128;
const DEFAULT_D_FF_MULT: u32 = 4;
const DEFAULT_VOCAB: u32 = 32_000;

/// Infer `n_layers` from weight-parameter names: the maximum decimal
/// index in a `layers.N.` / `blocks.N.` / `h.N.` / `layer.N.` segment,
/// plus one.
fn infer_n_layers(weights: &WeightMap) -> Option<u32> {
    let mut max_idx: Option<u32> = None;
    for name in weights.names() {
        let mut parts = name.split('.').peekable();
        while let Some(part) = parts.next() {
            if matches!(part, "layers" | "blocks" | "h" | "layer")
                && let Some(next) = parts.peek()
                && let Ok(idx) = next.parse::<u32>()
            {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
        }
    }
    max_idx.map(|m| m + 1)
}

/// Infer `(vocab_size, d_model)` from the LM-head or token-embedding
/// weight shape (`[vocab, d_model]` by convention in every supported
/// checkpoint family).
fn infer_vocab_d_model(weights: &WeightMap) -> Option<(u32, u32)> {
    const CANDIDATES: &[&str] = &[
        "lm_head.weight",
        "model.lm_head.weight",
        "output.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
    ];
    for cand in CANDIDATES {
        if let Some(entry) = weights.get(cand)
            && let [vocab, d_model] = entry.shape[..]
        {
            return Some((vocab as u32, d_model as u32));
        }
    }
    // Fall back to any name ending in a recognisable embedding suffix.
    for name in weights.names() {
        if (name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight"))
            && let Some(entry) = weights.get(name)
            && let [vocab, d_model] = entry.shape[..]
        {
            return Some((vocab as u32, d_model as u32));
        }
    }
    None
}

/// Resolve the model shape from config keys, weights, and defaults.
pub fn resolve_model_shape(
    cfg: &CfieServeConfig,
    weights: Option<&WeightMap>,
) -> ResolvedModelShape {
    let mut sources: Vec<(&'static str, &'static str)> = Vec::new();
    let pick_u32 = |field: &'static str,
                        explicit: Option<i64>,
                        inferred: Option<(u32, &'static str)>,
                        default: u32,
                        sources: &mut Vec<(&'static str, &'static str)>|
     -> u32 {
        if let Some(v) = explicit
            && v > 0
        {
            sources.push((field, "serve-config"));
            return v as u32;
        }
        if let Some((v, label)) = inferred {
            sources.push((field, label));
            return v;
        }
        sources.push((field, "defaults"));
        default
    };

    let inferred_layers = weights.and_then(infer_n_layers);
    let inferred_vd = weights.and_then(infer_vocab_d_model);

    let n_layers = pick_u32(
        "n_layers",
        cfg.n_layers,
        inferred_layers.map(|v| (v, "weights")),
        DEFAULT_N_LAYERS,
        &mut sources,
    );
    let head_dim = pick_u32("head_dim", cfg.head_dim, None, DEFAULT_HEAD_DIM, &mut sources);
    let d_model = pick_u32(
        "d_model",
        cfg.d_model,
        inferred_vd.map(|(_, d)| (d, "weights")),
        head_dim * DEFAULT_N_KV_HEADS,
        &mut sources,
    );
    let n_heads = pick_u32(
        "n_heads",
        cfg.n_heads,
        Some(d_model / head_dim.max(1))
            .filter(|&h| h > 0)
            .map(|h| (h, "derived(d_model/head_dim)")),
        DEFAULT_N_KV_HEADS,
        &mut sources,
    );
    let n_kv_heads = pick_u32(
        "n_kv_heads",
        cfg.n_kv_heads,
        None,
        n_heads.min(DEFAULT_N_KV_HEADS),
        &mut sources,
    );
    let d_ff = pick_u32(
        "d_ff",
        cfg.d_ff,
        None,
        d_model * DEFAULT_D_FF_MULT,
        &mut sources,
    );
    let vocab_size = pick_u32(
        "vocab_size",
        cfg.vocab_size,
        inferred_vd.map(|(v, _)| (v, "weights")),
        DEFAULT_VOCAB,
        &mut sources,
    );

    let provenance = sources
        .iter()
        .map(|(f, s)| format!("{f}={s}"))
        .collect::<Vec<_>>()
        .join(", ");

    ResolvedModelShape {
        n_layers,
        n_kv_heads,
        n_heads,
        head_dim,
        d_model,
        d_ff,
        vocab_size,
        dtype_bytes: 2, // BF16/FP16 serving default
        weights_bytes: weights.map(|w| w.total_bytes()),
        provenance,
    }
}

// ---------------------------------------------------------------------------
// CfieInput assembly
// ---------------------------------------------------------------------------

/// The largest `top_k` the fused sampler supports: its candidate list is
/// a serial structure in shared memory.
pub const MAX_TOP_K: i64 = 64;

/// The fused sampler's parameters from the `sampling:` section.
///
/// Nothing the user did not write is filled in. A section that sets a
/// non-zero temperature must set `top_k`: the kernel keeps a candidate
/// list of at most [`MAX_TOP_K`] tokens and cannot sample the whole
/// vocabulary, so there is no honest meaning for "no top_k". `top_p` is
/// optional, and without it no nucleus filter runs. (A section with only
/// `temperature` used to get `top_k = 50` and `top_p = 0.9` silently.)
/// With no section at all the documented defaults apply and the build
/// report names them as defaults.
fn sampling_params(section: Option<&SamplingSection>) -> Result<SamplingParams, CodegenError> {
    let Some(s) = section else {
        return Ok(SamplingParams::default());
    };
    let wrote_anything = s.temperature.is_some() || s.top_k.is_some() || s.top_p.is_some();
    if !wrote_anything {
        // `sampling: fused: true` alone: the defaults, by request.
        return Ok(SamplingParams::default());
    }
    if let Some(k) = s.top_k
        && !(1..=MAX_TOP_K).contains(&k)
    {
        return Err(CodegenError::new(format!(
            "CFIE sampling: top_k {k} is not supported — the fused sampler keeps at most \
             {MAX_TOP_K} candidates. Use a top_k in 1..={MAX_TOP_K}."
        )));
    }
    let temperature = s.temperature.unwrap_or(1.0) as f32;
    if temperature == 0.0 {
        // Greedy: top_k and top_p cannot change an argmax.
        return Ok(SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: s.top_k.unwrap_or(1) as u32,
            ..SamplingParams::default()
        });
    }
    let Some(top_k) = s.top_k else {
        return Err(CodegenError::new(format!(
            "CFIE sampling: a non-zero temperature needs `top_k` (1..={MAX_TOP_K}). The fused \
             sampler draws from a candidate list of at most {MAX_TOP_K} tokens and cannot sample \
             the whole vocabulary; it will not pick a top_k for you. Add `top_k:`, or set \
             `temperature: 0.0` for greedy decoding."
        )));
    };
    Ok(match s.top_p {
        Some(top_p) => SamplingParams {
            strategy: SamplingStrategy::TopKTopP,
            temperature,
            top_k: top_k as u32,
            top_p: top_p as f32,
            ..SamplingParams::default()
        },
        None => SamplingParams {
            strategy: SamplingStrategy::TopK,
            temperature,
            top_k: top_k as u32,
            top_p: 1.0,
            ..SamplingParams::default()
        },
    })
}

/// What the `kv_quant` key asks for. The KV pool is FP16 in every layer:
/// that is the only precision the decode block writes and reads.
///
/// * absent, `"uniform_fp16"` / `"fp16"`, `"off"`: FP16, which is what
///   the build does. `Ok(false)`.
/// * `"auto"`: FP16 as well, plus the advisory per-layer ranking in the
///   build report, labelled as not applied. `Ok(true)`.
/// * `"uniform_int8"` / `"int8"`: refused. No kernel stores INT8 keys or
///   values, so an INT8 pool could only be read, never filled.
fn kv_quant_advisory(choice: Option<&str>) -> Result<bool, CodegenError> {
    match choice {
        Some("uniform_int8") | Some("int8") => Err(CodegenError::new(
            "CFIE kv_quant: INT8 KV is not supported — no kernel quantizes keys or values \
             into the cache, so an INT8 pool could never be filled. Use \
             `kv_quant: \"uniform_fp16\"` (or drop the key)."
                .to_string(),
        )),
        Some("auto") => Ok(true),
        _ => Ok(false),
    }
}

/// Resolve whether CFIE runs for this serve block, and in which mode.
///
/// Precedence: CLI `--cfie <mode>` override > `@cfie(mode=...)`
/// decorator > implicit activation via serve-block CFIE keys (paper
/// rule: omitting every CFIE knob leaves the M29 dynamic path
/// untouched).  `Ok(None)` means "CFIE not requested".
pub fn resolve_mode(
    cli_override: Option<&str>,
    decorator: Option<CfieMode>,
    cfg: &CfieServeConfig,
) -> Result<Option<CfieMode>, CodegenError> {
    if let Some(s) = cli_override {
        return match CfieMode::parse(s) {
            Some(m) => Ok(Some(m)),
            None => Err(CodegenError::new(format!(
                "--cfie: unknown mode '{s}' (expected full, sampling, or off)"
            ))),
        };
    }
    if let Some(m) = decorator {
        return Ok(Some(m));
    }
    Ok(if cfg.requests_cfie() {
        Some(CfieMode::Full)
    } else {
        None
    })
}

/// The `speculative:` section as the one configuration that has a driver:
/// a greedy draft model compiled into the binary (`draft_weights`),
/// proposing a linear chain of `tokens` tokens. Everything else the
/// section can say is refused, because nothing would honour it.
fn speculative_config(sp: &SpeculativeSection) -> Result<SpeculativeConfig, CodegenError> {
    let refuse = |what: String| Err(CodegenError::new(format!("CFIE speculative: {what}")));
    if let Some(draft) = sp.draft_path.as_deref() {
        return refuse(format!(
            "the `draft` key is not supported ('{draft}'): nothing loads it. Name the draft \
             checkpoint with `draft_weights: \"<path.safetensors>\"` plus the draft_* shape \
             keys (draft_n_layers/draft_d_model/draft_n_heads/draft_n_kv_heads/\
             draft_head_dim/draft_d_ff)."
        ));
    }
    if let Some(method) = sp.method.as_deref()
        && method != "standard"
    {
        return refuse(format!(
            "method \"{method}\" is not supported. The driver drafts a linear chain of tokens \
             (method \"standard\"); tree, medusa and lookahead drafting have no driver. Use \
             `method: \"standard\"` or drop the key."
        ));
    }
    if sp.tree_width.is_some() {
        return refuse(
            "`tree_width` is not supported: there is no tree drafting. Remove the key."
                .to_string(),
        );
    }
    if let Some(t) = sp.temperature
        && t != 0.0
    {
        return refuse(format!(
            "`temperature: {t}` is not supported: the draft model is greedy. Remove the key \
             (the TARGET's sampling is set in the sampling: section and is what the output \
             follows)."
        ));
    }
    if sp.draft_weights.is_none() {
        return refuse(
            "the section needs `draft_weights: \"<path.safetensors>\"`. Without a draft model \
             there is nothing to speculate with, and generate() would run plain decoding \
             while the build report described speculation. Add the draft checkpoint or \
             remove the section."
                .to_string(),
        );
    }
    let k_tokens = sp.tokens.unwrap_or(SpeculativeConfig::default().k_tokens as i64);
    if !(1..=32).contains(&k_tokens) {
        return refuse(format!("`tokens: {k_tokens}` is outside 1..=32."));
    }
    Ok(SpeculativeConfig { k_tokens: k_tokens as u32 })
}

/// Everything `compile_serve_block` needs to run the orchestrator.
pub struct PreparedCfie<'a> {
    pub input: CfieInput<'a>,
    pub shape: ResolvedModelShape,
}

/// Assemble the [`CfieInput`] from extracted config + resolved
/// environment.  Errors on configuration that would require kernel
/// machinery that is not yet wired — a loud refusal is the project
/// convention for unimplemented transformations.
pub fn prepare<'a>(
    cfg: &CfieServeConfig,
    mode: CfieMode,
    gpu: &'a GpuSpec,
    weights: Option<&'a WeightMap>,
) -> Result<PreparedCfie<'a>, CodegenError> {
    if let Some(schema) = cfg.grammar_schema.as_deref() {
        return Err(CodegenError::new(format!(
            "CFIE grammar: grammar-constrained decoding is not supported (schema \
             '{schema}'). The decode loop does not advance the grammar's automaton from \
             token to token and has no end-of-sequence rule, so a `grammar:` section would \
             hold every token to the set allowed at the start. Remove the section."
        )));
    }
    let speculative = match cfg.speculative.as_ref() {
        None => None,
        // Only full mode plans speculation. In `sampling` mode the section
        // would be dropped and generate() would run plain decoding.
        Some(_) if mode != CfieMode::Full => {
            return Err(CodegenError::new(format!(
                "CFIE speculative: the section needs CFIE mode `full`; mode `{}` does not \
                 plan speculative decoding. Remove the section or the mode override.",
                mode.as_str()
            )));
        }
        Some(sp) => Some(speculative_config(sp)?),
    };

    let shape = resolve_model_shape(cfg, weights);

    let max_seq = cfg.max_seq.unwrap_or(4096).max(1) as u32;
    let max_batch = cfg.max_batch.unwrap_or(32).max(1) as u32;

    let kv_shape = KvShape {
        n_layers: shape.n_layers,
        n_kv_heads: shape.n_kv_heads,
        head_dim: shape.head_dim,
        dtype_bytes: shape.dtype_bytes,
    };
    let kv_budget = KvBudget {
        hbm_bytes: (gpu.vram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
        weights_bytes: shape.weights_bytes.unwrap_or(0),
        runtime_overhead_bytes: CUDA_RUNTIME_OVERHEAD_BYTES,
        max_seq,
        max_batch,
        block_size: 256,
        // G24a: coalescing granularity from the GPU's warp-group width
        // (128 on Hopper/Blackwell, 32 on older parts).  The paper's
        // 256-token block clears the 128-token floor unchanged.
        snap_granularity: gpu.warp_group_size(),
    };
    let lm_head = LmHeadShape {
        d_model: shape.d_model,
        vocab_size: shape.vocab_size,
        vocab_tile: 256,
        dtype_bytes: shape.dtype_bytes,
    };
    let persistent_model = PersistentModel {
        d_model: shape.d_model,
        head_dim: shape.head_dim,
        n_layers: shape.n_layers,
        n_heads: shape.n_heads,
        n_kv_heads: shape.n_kv_heads,
        d_ff: shape.d_ff,
        dtype_bytes: shape.dtype_bytes,
    };
    let gpu_budget = crate::cfie_persistent::gpu_budget_from_spec(gpu);

    let input = CfieInput {
        mode,
        target_gpu: gpu.name,
        kv_shape,
        kv_budget,
        lm_head,
        sampling: sampling_params(cfg.sampling.as_ref())?,
        sampling_configured: cfg.sampling.as_ref().is_some_and(|s| {
            s.temperature.is_some() || s.top_k.is_some() || s.top_p.is_some()
        }),
        persistent_model,
        gpu_budget,
        max_active_requests: max_batch,
        kv_quant: KvQuantConfig::new(shape.n_layers, shape.n_kv_heads, shape.head_dim),
        kv_quant_advisory: kv_quant_advisory(cfg.kv_quant.as_deref())?,
        weights,
        speculative,
    };

    Ok(PreparedCfie { input, shape })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::find_gpu;

    fn cfg_with(kv_layout: Option<&str>) -> CfieServeConfig {
        CfieServeConfig {
            kv_layout: kv_layout.map(str::to_string),
            max_seq: Some(2048),
            max_batch: Some(64),
            n_layers: Some(8),
            n_kv_heads: Some(4),
            head_dim: Some(128),
            d_model: Some(512),
            d_ff: Some(1408),
            vocab_size: Some(49_152),
            ..Default::default()
        }
    }

    #[test]
    fn requests_cfie_only_with_cfie_surface() {
        assert!(!CfieServeConfig::default().requests_cfie());
        assert!(cfg_with(Some("static")).requests_cfie());
        let sampling_only = CfieServeConfig {
            sampling: Some(SamplingSection {
                fused: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(sampling_only.requests_cfie());
        let sampling_unfused = CfieServeConfig {
            sampling: Some(SamplingSection {
                fused: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!sampling_unfused.requests_cfie());
    }

    #[test]
    fn explicit_config_shapes_win_and_are_attributed() {
        let shape = resolve_model_shape(&cfg_with(Some("static")), None);
        assert_eq!(shape.n_layers, 8);
        assert_eq!(shape.n_kv_heads, 4);
        assert_eq!(shape.d_model, 512);
        assert!(shape.provenance.contains("n_layers=serve-config"));
    }

    #[test]
    fn defaults_are_flagged_in_provenance() {
        let shape = resolve_model_shape(&CfieServeConfig::default(), None);
        assert_eq!(shape.n_layers, DEFAULT_N_LAYERS);
        assert!(shape.provenance.contains("n_layers=defaults"));
    }

    #[test]
    fn prepare_builds_paper_example_input() {
        let gpu = find_gpu("H100").unwrap();
        let prepared = prepare(&cfg_with(Some("static")), CfieMode::Full, gpu, None).unwrap();
        // 2 * (K+V) * 4 heads * 128 dim * 2 bytes * 8 layers = 16 KiB/token
        assert_eq!(prepared.input.kv_shape.bytes_per_token(), 16_384);
        assert_eq!(prepared.input.max_active_requests, 64);
        assert_eq!(prepared.input.gpu_budget.num_sms, 132);
        let plan = crate::cfie::run(prepared.input);
        assert!(plan.kv.uses_direct_indexing(), "H100 fits static layout");
    }


    fn refusal(cfg: &CfieServeConfig) -> String {
        let gpu = find_gpu("H100").unwrap();
        match prepare(cfg, CfieMode::Full, gpu, None) {
            Err(e) => e.message,
            Ok(_) => panic!("this configuration must be refused"),
        }
    }

    #[test]
    fn a_grammar_section_is_refused() {
        let mut cfg = cfg_with(Some("static"));
        cfg.grammar_schema = Some("schema.json".to_string());
        cfg.grammar_tokenizer = Some("vocab.txt".to_string());
        let msg = refusal(&cfg);
        assert!(msg.contains("grammar-constrained decoding is not supported"), "{msg}");
        assert!(msg.contains("schema.json"), "{msg}");
    }

    #[test]
    fn int8_kv_is_refused_and_auto_is_advisory() {
        let mut cfg = cfg_with(Some("static"));
        for spelling in ["uniform_int8", "int8"] {
            cfg.kv_quant = Some(spelling.to_string());
            let msg = refusal(&cfg);
            assert!(msg.contains("INT8 KV is not supported"), "{msg}");
        }
        let gpu = find_gpu("H100").unwrap();
        for (choice, advisory) in [
            (None, false),
            (Some("uniform_fp16"), false),
            (Some("fp16"), false),
            (Some("off"), false),
            (Some("auto"), true),
        ] {
            cfg.kv_quant = choice.map(str::to_string);
            let prepared = prepare(&cfg, CfieMode::Full, gpu, None).unwrap();
            assert_eq!(prepared.input.kv_quant_advisory, advisory, "{choice:?}");
        }
    }

    fn section(temperature: Option<f64>, top_k: Option<i64>, top_p: Option<f64>) -> SamplingSection {
        SamplingSection { temperature, top_k, top_p, fused: Some(true), seed: None }
    }

    #[test]
    fn sampling_section_maps_to_params_without_filling_anything_in() {
        // temperature 0: greedy, whatever else is written.
        let p = sampling_params(Some(&section(Some(0.0), Some(50), Some(0.9)))).unwrap();
        assert_eq!(p.strategy, SamplingStrategy::Greedy);

        let p = sampling_params(Some(&section(Some(0.7), Some(50), Some(0.9)))).unwrap();
        assert_eq!(p.strategy, SamplingStrategy::TopKTopP);
        assert_eq!((p.temperature, p.top_k, p.top_p), (0.7, 50, 0.9));

        // No top_p: no nucleus filter. It used to become 0.9.
        let p = sampling_params(Some(&section(Some(0.7), Some(40), None))).unwrap();
        assert_eq!(p.strategy, SamplingStrategy::TopK);
        assert_eq!((p.temperature, p.top_k), (0.7, 40));
        let prog = crate::cfie_fused_sample::emit_program(
            p,
            LmHeadShape { d_model: 8, vocab_size: 128, vocab_tile: 128, dtype_bytes: 2 },
        );
        assert!(
            !prog.ops.iter().any(|op| matches!(op, crate::cfie_fused_sample::FusedSampleOp::NucleusFilter { .. })),
            "no nucleus filter without top_p"
        );

        // top_k without a temperature: temperature 1.
        let p = sampling_params(Some(&section(None, Some(8), None))).unwrap();
        assert_eq!((p.strategy, p.temperature, p.top_k), (SamplingStrategy::TopK, 1.0, 8));

        // No section, or a section that only says `fused: true`: the
        // documented defaults.
        let d = SamplingParams::default();
        for p in [sampling_params(None).unwrap(), sampling_params(Some(&section(None, None, None))).unwrap()] {
            assert_eq!((p.strategy, p.temperature, p.top_k, p.top_p), (d.strategy, d.temperature, d.top_k, d.top_p));
        }
    }

    #[test]
    fn a_temperature_without_top_k_is_refused() {
        let err = sampling_params(Some(&section(Some(0.7), None, None))).unwrap_err();
        assert!(err.message.contains("a non-zero temperature needs `top_k`"), "{}", err.message);
        let err = sampling_params(Some(&section(Some(0.7), None, Some(0.9)))).unwrap_err();
        assert!(err.message.contains("needs `top_k`"), "{}", err.message);
    }

    #[test]
    fn a_top_k_the_sampler_cannot_hold_is_refused() {
        let err = sampling_params(Some(&section(Some(0.7), Some(100), Some(0.9)))).unwrap_err();
        assert!(err.message.contains("top_k 100 is not supported"), "{}", err.message);
        assert!(sampling_params(Some(&section(Some(0.7), Some(64), None))).is_ok());
    }

    fn draft_section() -> SpeculativeSection {
        SpeculativeSection {
            tokens: Some(3),
            method: Some("standard".to_string()),
            draft_weights: Some("draft.safetensors".to_string()),
            draft_n_layers: Some(2),
            ..Default::default()
        }
    }

    #[test]
    fn a_speculative_section_with_a_draft_model_is_accepted() {
        let mut cfg = cfg_with(Some("static"));
        cfg.speculative = Some(draft_section());
        let gpu = find_gpu("H100").unwrap();
        let prepared = prepare(&cfg, CfieMode::Full, gpu, None).unwrap();
        assert_eq!(prepared.input.speculative, Some(SpeculativeConfig { k_tokens: 3 }));
        // `temperature: 0.0` says what the draft does anyway.
        cfg.speculative = Some(SpeculativeSection { temperature: Some(0.0), ..draft_section() });
        assert!(prepare(&cfg, CfieMode::Full, gpu, None).is_ok());
    }

    /// `--cfie sampling` (or `@cfie(mode=sampling)`) plans no speculation;
    /// a `speculative:` section must not be dropped without a word.
    #[test]
    fn a_speculative_section_outside_full_mode_is_refused() {
        let mut cfg = cfg_with(Some("static"));
        cfg.speculative = Some(draft_section());
        let gpu = find_gpu("H100").unwrap();
        let err = match prepare(&cfg, CfieMode::Sampling, gpu, None) {
            Err(e) => e.message,
            Ok(_) => panic!("must be refused"),
        };
        assert!(err.contains("needs CFIE mode `full`; mode `sampling`"), "{err}");
        cfg.speculative = None;
        assert!(prepare(&cfg, CfieMode::Sampling, gpu, None).is_ok());
    }

    #[test]
    fn speculative_requests_nothing_honours_are_refused() {
        let cases: [(SpeculativeSection, &str); 6] = [
            (SpeculativeSection { method: Some("tree".into()), ..draft_section() }, "method \"tree\" is not supported"),
            (SpeculativeSection { method: Some("medusa".into()), ..draft_section() }, "method \"medusa\" is not supported"),
            (SpeculativeSection { tree_width: Some(2), ..draft_section() }, "`tree_width` is not supported"),
            (SpeculativeSection { temperature: Some(0.8), ..draft_section() }, "`temperature: 0.8` is not supported"),
            (SpeculativeSection { draft_weights: None, ..draft_section() }, "needs `draft_weights"),
            (SpeculativeSection { draft_path: Some("draft.nslm".into()), ..draft_section() }, "the `draft` key is not supported"),
        ];
        for (section, expect) in cases {
            let mut cfg = cfg_with(Some("static"));
            cfg.speculative = Some(section);
            let msg = refusal(&cfg);
            assert!(msg.contains(expect), "expected `{expect}` in: {msg}");
        }
    }
}
