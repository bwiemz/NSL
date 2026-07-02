//! CFIE — Compiler-Fused Inference Engine: driver + report.
//!
//! Composes the six CFIE passes (KV plan, fused sample, speculative,
//! persistent decode + scheduler, KV quant, grammar) into a single
//! [`CfiePlan`].  Invoked from `serve.rs::run_cfie_for_serve` (Tier-A
//! wiring) with inputs assembled by `cfie_serve.rs`.  Today the plan
//! drives the build report and the request-ring init call; kernel-side
//! consumers land with audit gaps G7/G9/G11/G13/G16/G18.  Produces a
//! human-readable report matching paper §8's sample output.

use serde::Serialize;

use crate::cfie_fused_sample::{emit_program as emit_sample, FusedSampleProgram, LmHeadShape, SamplingParams};
use crate::cfie_grammar::{compile as compile_dfa, minimise as minimise_dfa, CompiledDfa, GrammarSpec};
use crate::cfie_kv_plan::{plan as plan_kv, KvBudget, KvLayoutPlan, KvShape};
use crate::cfie_kv_quant::{plan as plan_kv_quant, KvQuantConfig, KvQuantPlan};
use crate::cfie_persistent::{plan as plan_persistent, GpuBudget, PersistentModel, PersistentPlan};
use crate::cfie_speculative::{emit_program as emit_spec, SpeculativeConfig, SpeculativeProgram};
use crate::weight_aware::WeightMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CfieMode {
    /// All six passes active.
    Full,
    /// KV plan + fused sample only (lowest-risk subset).
    Sampling,
    /// Bypass CFIE entirely — dynamic dispatch fallback.
    Off,
}

impl CfieMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CfieMode::Full => "full",
            CfieMode::Sampling => "sampling",
            CfieMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(CfieMode::Full),
            "sampling" => Some(CfieMode::Sampling),
            "off" | "disable" | "disabled" => Some(CfieMode::Off),
            _ => None,
        }
    }
}

/// Driver input.
pub struct CfieInput<'a> {
    pub mode: CfieMode,
    pub target_gpu: &'a str,
    pub kv_shape: KvShape,
    pub kv_budget: KvBudget,
    pub lm_head: LmHeadShape,
    pub sampling: SamplingParams,
    pub persistent_model: PersistentModel,
    pub gpu_budget: GpuBudget,
    pub max_active_requests: u32,
    pub kv_quant: KvQuantConfig,
    pub weights: Option<&'a WeightMap>,
    pub speculative: Option<SpeculativeConfig>,
    pub speculative_acceptance: f32,
    pub grammar: Option<GrammarSpec>,
    /// Pre-compiled token-level DFA (the G12 schema->token pipeline in
    /// `cfie_grammar::dfa_from_json_schema`).  Takes precedence over
    /// `grammar` when both are set.
    pub grammar_dfa: Option<CompiledDfa>,
}

#[derive(Debug, Clone)]
pub struct CfiePlan {
    pub mode: CfieMode,
    pub target_gpu: String,
    pub kv: KvLayoutPlan,
    pub sampling: FusedSampleProgram,
    pub persistent: PersistentPlan,
    pub kv_quant: KvQuantPlan,
    pub speculative: Option<SpeculativeProgram>,
    pub grammar: Option<CompiledDfa>,
    pub kernel_launches_per_token_baseline: u32,
    pub kernel_launches_per_token_cfie: u32,
    pub solve_us: u64,
    /// Feature 1 (G7): the direct-indexing decode-attention kernel,
    /// emitted by the serve wiring when the KV plan selects a static
    /// layout and the v1 emitter's preconditions hold.  Consumed by
    /// the decode-loop lowering when it lands (G16).
    pub decode_attention_kernel: Option<String>,
    pub decode_attention_ptx: Option<String>,
    /// Feature 6 (G11): the initialized `.global` PTX fragment baking
    /// the grammar's valid-token bitmask into the module image.
    /// Emitted by the serve wiring when `grammar` is set; the decode
    /// loop binds its device address to the sampler's
    /// `grammar_mask_ptr` param when it lands (G16).
    pub grammar_mask_ptx: Option<String>,
}

impl CfiePlan {
    pub fn launch_reduction(&self) -> f64 {
        if self.kernel_launches_per_token_baseline == 0 {
            return 0.0;
        }
        1.0 - (self.kernel_launches_per_token_cfie as f64
            / self.kernel_launches_per_token_baseline as f64)
    }

    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== CFIE Inference Build Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target: {}", self.target_gpu).unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Optimizations applied:").unwrap();
        writeln!(s, "  [1] KV layout: {} ({})", self.kv.kind.as_str(), self.kv.rationale).unwrap();
        if let Some(kernel) = self.decode_attention_kernel.as_ref() {
            writeln!(
                s,
                "      direct-index decode attention: {} emitted ({} bytes PTX, launch wiring pending G16)",
                kernel,
                self.decode_attention_ptx.as_ref().map_or(0, |p| p.len())
            )
            .unwrap();
        }
        writeln!(
            s,
            "  [2] Fused decode-sample: {} ops ({:.1} KB HBM saved per token)",
            self.sampling.len(),
            self.sampling.hbm_bytes_saved as f64 / 1024.0
        )
        .unwrap();
        if let Some(spec) = self.speculative.as_ref() {
            writeln!(
                s,
                "  [3] Compiled speculative: method={}, K={}, expected speedup={:.2}×",
                spec.config.method.as_str(),
                spec.config.k_tokens,
                spec.expected_speedup
            )
            .unwrap();
        } else {
            writeln!(s, "  [3] Compiled speculative: disabled").unwrap();
        }
        writeln!(
            s,
            "  [4] Persistent decode: fusion={}, {} launches/layer (vs {} baseline)",
            self.persistent.fusion.as_str(),
            self.persistent.persistent_launches_per_layer,
            self.persistent.baseline_launches_per_layer
        )
        .unwrap();
        writeln!(
            s,
            "  [5] Per-layer KV quant: {} INT8 layers, {:.1}% memory savings",
            self.kv_quant.int8_layer_count(),
            100.0 * self.kv_quant.memory_savings_ratio()
        )
        .unwrap();
        if let Some(dfa) = self.grammar.as_ref() {
            writeln!(
                s,
                "  [6] Grammar DFA: {} states, {} live transitions ({:.2}% density)",
                dfa.num_states,
                dfa.live_transitions,
                100.0 * dfa.density()
            )
            .unwrap();
            if let Some(mask) = self.grammar_mask_ptx.as_ref() {
                writeln!(
                    s,
                    "      mask baked into module image ({} bytes PTX, launch wiring pending G16)",
                    mask.len()
                )
                .unwrap();
            }
        } else {
            writeln!(s, "  [6] Grammar DFA: disabled").unwrap();
        }
        writeln!(s).unwrap();
        writeln!(
            s,
            "Kernel launches per token: {} (vs {} baseline, {:.1}% reduction)",
            self.kernel_launches_per_token_cfie,
            self.kernel_launches_per_token_baseline,
            100.0 * self.launch_reduction()
        )
        .unwrap();
        writeln!(s, "Continuous batching: on ({} active slots, ring capacity {})",
                 self.persistent.scheduler.max_active,
                 self.persistent.scheduler.ring_buffer.capacity).unwrap();
        writeln!(s, "Solve time: {:.2} ms", self.solve_us as f64 / 1000.0).unwrap();
        s
    }
}

pub fn run(input: CfieInput) -> CfiePlan {
    let t0 = std::time::Instant::now();

    if input.mode == CfieMode::Off {
        return CfiePlan {
            mode: CfieMode::Off,
            target_gpu: input.target_gpu.to_string(),
            kv: plan_kv(&input.kv_shape, &input.kv_budget),
            sampling: emit_sample(input.sampling, input.lm_head),
            persistent: plan_persistent(
                &input.persistent_model,
                &input.gpu_budget,
                input.max_active_requests,
            ),
            kv_quant: KvQuantPlan::default(),
            speculative: None,
            grammar: None,
            kernel_launches_per_token_baseline: 500,
            kernel_launches_per_token_cfie: 500,
            solve_us: t0.elapsed().as_micros() as u64,
            decode_attention_kernel: None,
            decode_attention_ptx: None,
            grammar_mask_ptx: None,
        };
    }

    // 1. KV layout.
    let kv = plan_kv(&input.kv_shape, &input.kv_budget);

    // 2. Fused sampler.  Enable grammar masking when a grammar is set.
    let mut sampling_params = input.sampling;
    sampling_params.grammar_masked = input.grammar.is_some() || input.grammar_dfa.is_some();
    let sampling = emit_sample(sampling_params, input.lm_head);

    // 3. Persistent decode + scheduler.
    let persistent = plan_persistent(
        &input.persistent_model,
        &input.gpu_budget,
        input.max_active_requests,
    );

    // 4. Per-layer KV quantisation.
    let kv_quant = if input.mode == CfieMode::Full {
        plan_kv_quant(&input.kv_quant, input.weights)
    } else {
        KvQuantPlan::default()
    };

    // 5. Compiled speculative (optional).
    let speculative = if input.mode == CfieMode::Full {
        input
            .speculative
            .as_ref()
            .map(|cfg| emit_spec(cfg.clone(), input.speculative_acceptance))
    } else {
        None
    };

    // 6. Grammar DFA (optional).  Pre-compiled token DFAs (G12 serve
    // pipeline) win over edge-list specs; both get Hopcroft-minimised
    // (the audit flagged `minimise` as implemented-but-never-called) —
    // fewer states means a smaller baked mask, same language.
    let grammar = if input.mode == CfieMode::Full {
        input
            .grammar_dfa
            .clone()
            .or_else(|| input.grammar.as_ref().and_then(|spec| compile_dfa(spec).ok()))
            .map(|dfa| minimise_dfa(&dfa))
    } else {
        None
    };

    // Kernel-launch accounting.
    let baseline = 500u32; // paper §1's "~500-1000 launches/token" figure
    let saved_persistent = (persistent.baseline_launches_per_layer
        .saturating_sub(persistent.persistent_launches_per_layer))
        * input.persistent_model.n_layers;
    // Sampling + residual layers: one fused kernel versus six.
    let saved_sampling = if sampling.is_fused() { 5 } else { 0 };
    let cfie_launches = baseline.saturating_sub(saved_persistent + saved_sampling);

    CfiePlan {
        mode: input.mode,
        target_gpu: input.target_gpu.to_string(),
        kv,
        sampling,
        persistent,
        kv_quant,
        speculative,
        grammar,
        kernel_launches_per_token_baseline: baseline,
        kernel_launches_per_token_cfie: cfie_launches,
        solve_us: t0.elapsed().as_micros() as u64,
        decode_attention_kernel: None,
        decode_attention_ptx: None,
        grammar_mask_ptx: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{LmHeadShape, SamplingParams};
    use crate::cfie_kv_plan::{KvBudget, KvShape};
    use crate::cfie_persistent::{GpuBudget, PersistentModel};

    fn nslcoder_input<'a>() -> CfieInput<'a> {
        CfieInput {
            mode: CfieMode::Full,
            target_gpu: "H100",
            kv_shape: KvShape {
                n_layers: 8,
                n_kv_heads: 4,
                head_dim: 128,
                dtype_bytes: 2,
            },
            kv_budget: KvBudget::default(),
            lm_head: LmHeadShape {
                d_model: 512,
                vocab_size: 32_000,
                vocab_tile: 256,
                dtype_bytes: 2,
            },
            sampling: SamplingParams::default(),
            persistent_model: PersistentModel {
                d_model: 512,
                head_dim: 64,
                n_layers: 8,
                n_heads: 8,
                n_kv_heads: 4,
                d_ff: 1408,
                dtype_bytes: 2,
            },
            gpu_budget: GpuBudget::default(),
            max_active_requests: 32,
            kv_quant: KvQuantConfig::new(8, 4, 128),
            weights: None,
            speculative: Some(SpeculativeConfig::default()),
            speculative_acceptance: 0.6,
            grammar: None,
            grammar_dfa: None,
        }
    }

    #[test]
    fn full_mode_populates_every_pass() {
        let plan = run(nslcoder_input());
        assert_eq!(plan.mode, CfieMode::Full);
        assert!(plan.kv.direct.is_some() || plan.kv.bump.is_some());
        assert!(plan.sampling.is_fused());
        assert!(plan.speculative.is_some());
        assert!(!plan.kv_quant.layers.is_empty());
    }

    #[test]
    fn off_mode_does_not_emit_speculative_or_grammar() {
        let mut input = nslcoder_input();
        input.mode = CfieMode::Off;
        let plan = run(input);
        assert!(plan.speculative.is_none());
        assert!(plan.grammar.is_none());
    }

    #[test]
    fn grammar_when_supplied_activates_mask_and_dfa() {
        let mut input = nslcoder_input();
        input.grammar = Some(GrammarSpec::sequence(&[5, 6, 7], 32_000));
        let plan = run(input);
        assert!(plan.grammar.is_some());
        // The sampler must have grammar masking turned on.
        assert!(plan.sampling.params.grammar_masked);
    }

    #[test]
    fn grammar_dfa_input_wins_and_is_minimised() {
        use crate::cfie_grammar::GrammarEdge;
        // 4-state DFA with two merge-equivalent middle states; Hopcroft
        // must collapse them (proves cfie::run actually calls minimise).
        let spec = GrammarSpec {
            num_states: 4,
            vocab_size: 32_000,
            start_state: 0,
            accept_states: vec![3],
            edges: vec![
                GrammarEdge { from: 0, token_id: 0, to: 1 },
                GrammarEdge { from: 0, token_id: 1, to: 2 },
                GrammarEdge { from: 1, token_id: 0, to: 3 },
                GrammarEdge { from: 2, token_id: 0, to: 3 },
            ],
        };
        let dfa = crate::cfie_grammar::compile(&spec).unwrap();
        let mut input = nslcoder_input();
        input.grammar_dfa = Some(dfa.clone());
        let plan = run(input);
        let planned = plan.grammar.expect("grammar_dfa must reach the plan");
        assert!(
            planned.num_states < dfa.num_states,
            "minimise must merge equivalent states: {} vs {}",
            planned.num_states,
            dfa.num_states
        );
        assert!(plan.sampling.params.grammar_masked);
    }

    #[test]
    fn report_includes_baked_mask_line_when_present() {
        let mut input = nslcoder_input();
        input.grammar = Some(GrammarSpec::sequence(&[5, 6, 7], 32_000));
        let mut plan = run(input);
        let dfa = plan.grammar.as_ref().unwrap();
        plan.grammar_mask_ptx = Some(crate::cfie_grammar_ptx::emit_mask_global(dfa));
        let rep = plan.render_report();
        assert!(rep.contains("mask baked into module image"));
    }

    #[test]
    fn launch_reduction_positive_on_h100() {
        let plan = run(nslcoder_input());
        assert!(plan.kernel_launches_per_token_cfie < plan.kernel_launches_per_token_baseline);
        assert!(plan.launch_reduction() > 0.0);
    }

    #[test]
    fn report_contains_all_six_sections() {
        let plan = run(nslcoder_input());
        let rep = plan.render_report();
        assert!(rep.contains("CFIE Inference Build Report"));
        assert!(rep.contains("[1] KV layout"));
        assert!(rep.contains("[2] Fused decode-sample"));
        assert!(rep.contains("[3] Compiled speculative"));
        assert!(rep.contains("[4] Persistent decode"));
        assert!(rep.contains("[5] Per-layer KV quant"));
        assert!(rep.contains("[6] Grammar DFA"));
        assert!(rep.contains("Continuous batching: on"));
    }

    #[test]
    fn deterministic_across_calls() {
        let r1 = run(nslcoder_input()).render_report();
        let r2 = run(nslcoder_input()).render_report();
        let strip = |s: &str| {
            s.lines()
                .filter(|l| !l.contains("Solve time"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip(&r1), strip(&r2));
    }

    #[test]
    fn mode_parse_roundtrip() {
        for m in [CfieMode::Full, CfieMode::Sampling, CfieMode::Off] {
            assert_eq!(CfieMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CfieMode::parse("auto"), Some(CfieMode::Full));
        assert!(CfieMode::parse("nonsense").is_none());
    }

    #[test]
    fn sampling_mode_skips_quant_and_speculative() {
        let mut input = nslcoder_input();
        input.mode = CfieMode::Sampling;
        let plan = run(input);
        assert!(plan.speculative.is_none());
        assert!(plan.kv_quant.layers.is_empty());
    }
}
