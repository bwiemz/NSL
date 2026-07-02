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
use crate::cfie_grammar::{dfa_from_json_schema, CompiledDfa, TokenVocab};
use crate::cfie_kv_plan::{KvBudget, KvShape};
use crate::cfie_kv_quant::KvQuantConfig;
use crate::cfie_persistent::{GpuBudget, PersistentModel};
use crate::cfie_speculative::{DraftMethod, SpeculativeConfig};
use crate::error::CodegenError;
use crate::gpu_specs::GpuSpec;
use crate::weight_aware::WeightMap;

/// Paper default: CUDA driver + workspace reserve subtracted from HBM
/// before planning the KV budget (§2's worked example uses 800 MB).
const CUDA_RUNTIME_OVERHEAD_BYTES: u64 = 800 * 1024 * 1024;

/// Default acceptance rate assumed for speculative-speedup estimates
/// until measurement-driven acceptance lands (audit gap G15).
const DEFAULT_SPECULATIVE_ACCEPTANCE: f32 = 0.6;

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
}

/// `speculative:` section values (paper §5).
#[derive(Debug, Clone, Default)]
pub struct SpeculativeSection {
    pub draft_path: Option<String>,
    pub tokens: Option<i64>,
    pub method: Option<String>,
    pub tree_width: Option<i64>,
    pub temperature: Option<f64>,
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
    pub sampling: Option<SamplingSection>,
    pub speculative: Option<SpeculativeSection>,
    pub grammar_schema: Option<String>,
    /// Tokenizer vocab path (`.txt` = one token per line, `.json` =
    /// string array) — required to token-project the schema (G12).
    pub grammar_tokenizer: Option<String>,
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
            if matches!(part, "layers" | "blocks" | "h" | "layer") {
                if let Some(next) = parts.peek() {
                    if let Ok(idx) = next.parse::<u32>() {
                        max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                    }
                }
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
        if let Some(entry) = weights.get(cand) {
            if let [vocab, d_model] = entry.shape[..] {
                return Some((vocab as u32, d_model as u32));
            }
        }
    }
    // Fall back to any name ending in a recognisable embedding suffix.
    for name in weights.names() {
        if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
            if let Some(entry) = weights.get(name) {
                if let [vocab, d_model] = entry.shape[..] {
                    return Some((vocab as u32, d_model as u32));
                }
            }
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
        if let Some(v) = explicit {
            if v > 0 {
                sources.push((field, "serve-config"));
                return v as u32;
            }
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

fn parse_draft_method(name: &str) -> Option<DraftMethod> {
    match name {
        "standard" => Some(DraftMethod::Standard),
        "tree" => Some(DraftMethod::Tree),
        "medusa" => Some(DraftMethod::Medusa),
        "lookahead" => Some(DraftMethod::Lookahead),
        _ => None,
    }
}

fn sampling_params(section: Option<&SamplingSection>) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(s) = section {
        if let Some(t) = s.temperature {
            params.temperature = t as f32;
        }
        if let Some(k) = s.top_k {
            params.top_k = k.max(1) as u32;
        }
        if let Some(p) = s.top_p {
            params.top_p = p as f32;
        }
        params.strategy = if params.temperature == 0.0 {
            SamplingStrategy::Greedy
        } else if s.top_p.is_some() {
            SamplingStrategy::TopKTopP
        } else if s.top_k.is_some() {
            SamplingStrategy::TopK
        } else {
            SamplingStrategy::TopKTopP
        };
    }
    params
}

/// KV-quant thresholds for the requested choice.  `auto` uses the
/// planner's tuned thresholds; the uniform choices force every layer to
/// one side of the sensitivity bands (scores live in [0, 1]).
fn kv_quant_config(choice: Option<&str>, shape: &ResolvedModelShape) -> KvQuantConfig {
    let mut cfg = KvQuantConfig::new(shape.n_layers, shape.n_kv_heads, shape.head_dim);
    match choice {
        Some("uniform_fp16") | Some("fp16") => {
            // Every layer scores above `high_threshold` => stays FP16.
            cfg.high_threshold = -1.0;
            cfg.low_threshold = -2.0;
        }
        Some("uniform_int8") | Some("int8") => {
            // Every layer scores below `low_threshold` => full INT8.
            cfg.low_threshold = 2.0;
            cfg.high_threshold = 3.0;
        }
        _ => {}
    }
    cfg
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
    // Grammar (G12): a schema needs a tokenizer vocab to be projected
    // from the byte-level DFA to token level — without one there is
    // nothing kernel-consumable, so refuse rather than half-compile.
    let grammar_dfa: Option<CompiledDfa> = match (&cfg.grammar_schema, &cfg.grammar_tokenizer) {
        (Some(schema), Some(tokenizer)) => {
            let schema_src = std::fs::read_to_string(schema).map_err(|e| {
                CodegenError::new(format!(
                    "CFIE grammar: cannot read schema '{schema}': {e}"
                ))
            })?;
            let vocab = TokenVocab::load(std::path::Path::new(tokenizer))
                .map_err(|e| CodegenError::new(format!("CFIE grammar: {e}")))?;
            let dfa = dfa_from_json_schema(&schema_src, &vocab).map_err(|e| {
                CodegenError::new(format!(
                    "CFIE grammar fusion (schema '{schema}'): {e}"
                ))
            })?;
            Some(dfa)
        }
        (Some(schema), None) => {
            return Err(CodegenError::new(format!(
                "CFIE grammar fusion: JSON-schema '{schema}' cannot be \
                 token-projected without a tokenizer vocabulary (audit \
                 gap G12). Add `tokenizer: \"vocab.txt\"` (one token \
                 per line) or `tokenizer: \"vocab.json\"` (JSON string \
                 array) to the grammar: section, or remove the section."
            )));
        }
        _ => None,
    };
    if let Some(draft) = cfg.speculative.as_ref().and_then(|s| s.draft_path.as_deref()) {
        // Accepting a draft-model path that is never loaded would be a
        // silent failure; refuse until compiled-speculative draft
        // loading lands (audit gaps G13/G15).  `speculative:` without
        // `draft` stays a compile-time plan (structure/accounting only)
        // and is labelled as such in the build report.
        return Err(CodegenError::new(format!(
            "CFIE compiled speculative: draft model '{draft}' cannot be \
             loaded yet — draft+verify kernel compilation is not wired \
             (audit gaps G13/G15). Remove the `draft` key (plan-only \
             speculative structure) or use the runtime @speculative \
             decorator path until it lands."
        )));
    }

    let shape = resolve_model_shape(cfg, weights);

    // The sampler kernel's grammar-mask row stride is baked from the
    // LM-head vocab; a tokenizer of a different size would silently
    // index the wrong rows once the mask is bound (G16 seam) — enforce
    // equality at the only point that sees both numbers.
    if let Some(dfa) = grammar_dfa.as_ref() {
        if dfa.vocab_size != shape.vocab_size {
            return Err(CodegenError::new(format!(
                "CFIE grammar fusion: tokenizer vocabulary has {} tokens but the \
                 model's vocab_size is {} — the baked mask row stride requires \
                 them to match. Fix the tokenizer file or the vocab_size key.",
                dfa.vocab_size, shape.vocab_size
            )));
        }
    }

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
    let gpu_budget = GpuBudget {
        smem_per_sm: (gpu.l1_cache_kb as u32) * 1024,
        num_sms: gpu.num_sms,
        kernel_launch_us: gpu.kernel_launch_overhead_ns as f64 / 1000.0,
    };

    let speculative = cfg.speculative.as_ref().map(|s| {
        let mut spec = SpeculativeConfig::default();
        if let Some(m) = s.method.as_deref().and_then(parse_draft_method) {
            spec.method = m;
        }
        if let Some(k) = s.tokens {
            spec.k_tokens = k.clamp(1, 32) as u32;
        }
        if let Some(w) = s.tree_width {
            spec.tree_width = w.clamp(1, 8) as u32;
        }
        if let Some(t) = s.temperature {
            spec.draft_temperature = t as f32;
        }
        spec
    });

    let input = CfieInput {
        mode,
        target_gpu: gpu.name,
        kv_shape,
        kv_budget,
        lm_head,
        sampling: sampling_params(cfg.sampling.as_ref()),
        persistent_model,
        gpu_budget,
        max_active_requests: max_batch,
        kv_quant: kv_quant_config(cfg.kv_quant.as_deref(), &shape),
        weights,
        speculative,
        speculative_acceptance: DEFAULT_SPECULATIVE_ACCEPTANCE,
        grammar: None,
        grammar_dfa,
    };

    Ok(PreparedCfie { input, shape })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_kv_quant::plan as plan_kv_quant;
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

    #[test]
    fn grammar_schema_without_tokenizer_refuses_loudly() {
        let mut cfg = cfg_with(Some("static"));
        cfg.grammar_schema = Some("schema.json".to_string());
        let gpu = find_gpu("H100").unwrap();
        let err = match prepare(&cfg, CfieMode::Full, gpu, None) {
            Err(e) => e,
            Ok(_) => panic!("schema without tokenizer cannot be token-projected"),
        };
        assert!(
            err.message.contains("G12") && err.message.contains("tokenizer"),
            "refusal must cite the audit gap and the fix: {}",
            err.message
        );
    }

    #[test]
    fn grammar_schema_with_tokenizer_compiles_token_dfa() {
        let dir = tempfile::tempdir().unwrap();
        let schema = dir.path().join("schema.json");
        std::fs::write(&schema, r#"{"type": "boolean"}"#).unwrap();
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, "true\nfalse\nx\n").unwrap();

        let mut cfg = cfg_with(Some("static"));
        // The mask row stride is baked from the LM-head vocab, so the
        // tokenizer size must equal the model vocab_size.
        cfg.vocab_size = Some(3);
        cfg.grammar_schema = Some(schema.display().to_string());
        cfg.grammar_tokenizer = Some(vocab.display().to_string());
        let gpu = find_gpu("H100").unwrap();
        let prepared = prepare(&cfg, CfieMode::Full, gpu, None).unwrap();
        let dfa = prepared
            .input
            .grammar_dfa
            .as_ref()
            .expect("schema + tokenizer must yield a token DFA");
        assert_eq!(dfa.vocab_size, 3);
        // Tokens "true"/"false" valid from start; "x" is not.
        assert!(dfa.is_valid(dfa.start_state, 0));
        assert!(dfa.is_valid(dfa.start_state, 1));
        assert!(!dfa.is_valid(dfa.start_state, 2));

        let plan = crate::cfie::run(prepared.input);
        assert!(plan.grammar.is_some());
        assert!(plan.sampling.params.grammar_masked);
    }

    #[test]
    fn grammar_tokenizer_vocab_size_mismatch_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let schema = dir.path().join("schema.json");
        std::fs::write(&schema, r#"{"type": "boolean"}"#).unwrap();
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, "true\nfalse\nx\n").unwrap();

        let mut cfg = cfg_with(Some("static"));
        // cfg_with sets vocab_size = 49152; the 3-token tokenizer must
        // refuse instead of silently mis-striding the baked mask.
        cfg.grammar_schema = Some(schema.display().to_string());
        cfg.grammar_tokenizer = Some(vocab.display().to_string());
        let gpu = find_gpu("H100").unwrap();
        let err = match prepare(&cfg, CfieMode::Full, gpu, None) {
            Err(e) => e,
            Ok(_) => panic!("vocab-size mismatch must refuse"),
        };
        assert!(
            err.message.contains("vocab_size"),
            "refusal must explain the mismatch: {}",
            err.message
        );
    }

    #[test]
    fn grammar_missing_files_refuse_with_context() {
        let mut cfg = cfg_with(Some("static"));
        cfg.grammar_schema = Some("does_not_exist.json".to_string());
        cfg.grammar_tokenizer = Some("also_missing.txt".to_string());
        let gpu = find_gpu("H100").unwrap();
        let err = match prepare(&cfg, CfieMode::Full, gpu, None) {
            Err(e) => e,
            Ok(_) => panic!("missing schema file must refuse"),
        };
        assert!(
            err.message.contains("does_not_exist.json"),
            "error must name the file: {}",
            err.message
        );
    }

    #[test]
    fn uniform_kv_quant_choices_land_all_layers_on_one_side() {
        let shape = resolve_model_shape(&cfg_with(None), None);

        let int8 = plan_kv_quant(&kv_quant_config(Some("uniform_int8"), &shape), None);
        assert_eq!(int8.int8_layer_count() as u32, shape.n_layers);

        let fp16 = plan_kv_quant(&kv_quant_config(Some("uniform_fp16"), &shape), None);
        assert_eq!(fp16.int8_layer_count(), 0);
    }

    #[test]
    fn sampling_section_maps_to_params() {
        let s = SamplingSection {
            temperature: Some(0.0),
            top_k: Some(50),
            top_p: Some(0.9),
            fused: Some(true),
        };
        let p = sampling_params(Some(&s));
        assert_eq!(p.strategy, SamplingStrategy::Greedy);

        let s2 = SamplingSection {
            temperature: Some(0.7),
            top_k: Some(50),
            top_p: Some(0.9),
            fused: Some(true),
        };
        let p2 = sampling_params(Some(&s2));
        assert_eq!(p2.strategy, SamplingStrategy::TopKTopP);
        assert_eq!(p2.top_k, 50);
    }

    #[test]
    fn speculative_section_maps_to_config() {
        let mut cfg = cfg_with(Some("static"));
        cfg.speculative = Some(SpeculativeSection {
            draft_path: None,
            tokens: Some(5),
            method: Some("tree".to_string()),
            tree_width: Some(2),
            temperature: Some(0.0),
        });
        let gpu = find_gpu("H100").unwrap();
        let prepared = prepare(&cfg, CfieMode::Full, gpu, None).unwrap();
        let spec = prepared.input.speculative.expect("speculative config");
        assert_eq!(spec.method, DraftMethod::Tree);
        assert_eq!(spec.k_tokens, 5);
        assert_eq!(spec.tree_width, 2);
    }

    #[test]
    fn speculative_draft_path_refuses_loudly() {
        let mut cfg = cfg_with(Some("static"));
        cfg.speculative = Some(SpeculativeSection {
            draft_path: Some("draft.nslm".to_string()),
            tokens: Some(5),
            method: Some("tree".to_string()),
            tree_width: Some(2),
            temperature: Some(0.0),
        });
        let gpu = find_gpu("H100").unwrap();
        let err = match prepare(&cfg, CfieMode::Full, gpu, None) {
            Err(e) => e,
            Ok(_) => panic!("draft path must refuse until G13/G15 land"),
        };
        assert!(
            err.message.contains("G13"),
            "refusal must cite the audit gaps: {}",
            err.message
        );
    }
}
