//! CFIE — Compiler-Fused Inference Engine: driver + report.
//!
//! Composes the six CFIE passes (KV plan, fused sample, speculative,
//! persistent decode + scheduler, KV quant, grammar) into a single
//! [`CfiePlan`].  Invoked from `serve.rs::run_cfie_for_serve` (Tier-A
//! wiring) with inputs assembled by `cfie_serve.rs`.
//!
//! What the plan drives today (post-Cycle-6): all five kernel families
//! emit real PTX (G7/G13/G16/G18/F2), and on the monolithic serve path
//! `serve.rs` embeds that PTX and emits the `nsl_cfie_register_kernel` /
//! `nsl_cfie_kv_pool_alloc` / `nsl_cfie_engine_finalize` calls, so the
//! chosen kernel family is registered with the runtime engine at serve
//! init.  Per-token launches then go through the host decode loop
//! (`nsl_cfie_decode_step`).  Still pending: endpoint-driven generation
//! needs model binding — serve blocks carry no model reference yet, so
//! the compiled decode loop is exercised by tests/host callers until
//! that binding lands.  The disaggregated serve path emits none of the
//! runtime wiring and the report says so (`runtime_wiring_emitted`).
//! Produces a human-readable report matching paper §8's sample output.

use serde::Serialize;

use crate::cfie_cost::CfieCostEstimate;
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

/// Host launch geometry for one emitted CFIE kernel — the values the
/// compiled serve binary passes to `nsl_cfie_register_kernel` at init
/// (Cycle 6 frozen ABI).  1-D launches only: every current CFIE kernel
/// is `grid=(grid_x,1,1) block=(block_x,1,1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfieLaunchSpec {
    pub grid_x: u32,
    pub block_x: u32,
    /// DYNAMIC shared-memory bytes for the launch.  0 for every current
    /// CFIE kernel — they all declare static `.shared` inside the
    /// module — the field exists for future extern-`.shared` kernels.
    pub smem_dyn_bytes: u32,
}

/// Which kernel family a build registers with the runtime engine.
/// Exactly ONE family registers per build: the quant family bakes the
/// mixed-precision pool layout from `cfie_kv_quant_ptx::pool_layout`,
/// which is incompatible with the uniform-f16 pool the decode-attn /
/// decode-block / spec kernels bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfieKernelFamily {
    /// decode_attn / decode_block / spec kernels (uniform f16 KV pool)
    /// plus the fused sampler.
    Uniform,
    /// per-layer KV-quant attention kernels (mixed-precision KV pool)
    /// plus the fused sampler.
    Quant,
}

impl CfieKernelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            CfieKernelFamily::Uniform => "uniform",
            CfieKernelFamily::Quant => "quant",
        }
    }
}

/// Compile-time family chooser (pure): quant wins whenever the
/// per-layer quant kernels were emitted, because their pool layout is
/// mutually exclusive with the uniform pool (see [`CfieKernelFamily`]).
pub fn choose_kernel_family(plan: &CfiePlan) -> CfieKernelFamily {
    if plan.quant_attention_kernels.is_empty() {
        CfieKernelFamily::Uniform
    } else {
        CfieKernelFamily::Quant
    }
}

/// One `nsl_cfie_register_kernel` call the compiled serve binary makes
/// at init.  Kinds per the frozen Cycle-6 ABI: 0=decode_attn,
/// 1=fused_sample, 2=decode_block, 3=spec_verify, 4=spec_reject,
/// 5=quant_attn.  `layer_idx` is meaningful only for kind 5 (0 for all
/// other kinds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfieKernelReg {
    pub kind: u8,
    pub layer_idx: u32,
    pub name: String,
    pub ptx: String,
    pub grid_x: u32,
    pub block_x: u32,
    pub smem_dyn: u32,
}

/// Build the registration list for the chosen kernel family (pure over
/// the plan).  Kernels register only when BOTH the PTX and a launch
/// spec exist; the fused sampler (kind 1) registers in both families —
/// it is the host decode loop's (`nsl_cfie_decode_step`) second stage.
pub fn kernel_registrations(plan: &CfiePlan) -> Vec<CfieKernelReg> {
    fn reg(
        kind: u8,
        layer_idx: u32,
        name: &str,
        ptx: &str,
        spec: CfieLaunchSpec,
    ) -> CfieKernelReg {
        CfieKernelReg {
            kind,
            layer_idx,
            name: name.to_string(),
            ptx: ptx.to_string(),
            grid_x: spec.grid_x,
            block_x: spec.block_x,
            smem_dyn: spec.smem_dyn_bytes,
        }
    }
    let mut regs = Vec::new();
    match choose_kernel_family(plan) {
        CfieKernelFamily::Quant => {
            if let Some(spec) = plan.quant_attention_launch {
                for (idx, (name, ptx)) in plan.quant_attention_kernels.iter().enumerate() {
                    regs.push(reg(5, idx as u32, name, ptx, spec));
                }
            }
        }
        CfieKernelFamily::Uniform => {
            if let (Some(name), Some(ptx), Some(spec)) = (
                plan.decode_attention_kernel.as_ref(),
                plan.decode_attention_ptx.as_ref(),
                plan.decode_attention_launch,
            ) {
                regs.push(reg(0, 0, name, ptx, spec));
            }
            if let (Some(name), Some(ptx), Some(spec)) = (
                plan.decode_block_kernel.as_ref(),
                plan.decode_block_ptx.as_ref(),
                plan.decode_block_launch,
            ) {
                regs.push(reg(2, 0, name, ptx, spec));
            }
            if let (Some(name), Some(ptx), Some(spec)) = (
                plan.spec_verify_kernel.as_ref(),
                plan.spec_verify_ptx.as_ref(),
                plan.spec_verify_launch,
            ) {
                regs.push(reg(3, 0, name, ptx, spec));
            }
            if let (Some(name), Some(ptx), Some(spec)) = (
                plan.spec_reject_kernel.as_ref(),
                plan.spec_reject_ptx.as_ref(),
                plan.spec_reject_launch,
            ) {
                regs.push(reg(4, 0, name, ptx, spec));
            }
        }
    }
    if let (Some(name), Some(ptx), Some(spec)) = (
        plan.fused_sample_kernel.as_ref(),
        plan.fused_sample_ptx.as_ref(),
        plan.fused_sample_launch,
    ) {
        regs.push(reg(1, 0, name, ptx, spec));
    }
    regs
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
    /// layout and the v1 emitter's preconditions hold.  Registered with
    /// the runtime engine at serve init (uniform family) and launched by
    /// the host decode loop (`nsl_cfie_decode_step`); still awaiting
    /// endpoint model binding to be driven by real requests.
    pub decode_attention_kernel: Option<String>,
    pub decode_attention_ptx: Option<String>,
    /// Feature 4 (G16): the persistent decode-block kernel (one CTA =
    /// one layer's decode step for one token).  Emitted by the serve
    /// wiring only when `persistent.fusion` is Level3 — the full-block
    /// claim; Level1/2 stay plan-level — and the emitter preconditions
    /// hold.  Registered at serve init (uniform family) and launched by
    /// the host decode loop; awaiting endpoint model binding.
    pub decode_block_kernel: Option<String>,
    pub decode_block_ptx: Option<String>,
    /// Feature 3 (G13/G14): compiled speculative verification kernels.
    /// `spec_verify_*` is the tree-mask verification attention (each
    /// node row's ancestor mask baked as a u64 immediate — no mask
    /// tensor parameter); `spec_reject_*` is the rejection-sampling
    /// epilogue (Leviathan residual + xorshift64* PRNG).  Emitted by
    /// the serve wiring when `speculative` is planned AND the KV plan
    /// selected the static direct-index layout both kernels bake their
    /// pool strides from.  Registered at serve init (uniform family) and
    /// launched by the host decode loop; awaiting endpoint model binding.
    pub spec_verify_kernel: Option<String>,
    pub spec_verify_ptx: Option<String>,
    pub spec_reject_kernel: Option<String>,
    pub spec_reject_ptx: Option<String>,
    /// Feature 6 (G11): the initialized `.global` PTX fragment baking
    /// the grammar's valid-token bitmask into the module image.  Emitted
    /// by the serve wiring when `grammar` is set and spliced into the
    /// fused-sampler module; the host resolves its device address via
    /// `cuModuleGetGlobal` at `nsl_cfie_engine_finalize` so the sampler
    /// reads it directly (no `grammar_mask_ptr` parameter needed).
    pub grammar_mask_ptx: Option<String>,
    /// Feature 5 (G18): per-layer decode-attention kernels with the
    /// KV-quant plan's precision baked into each layer's load path
    /// (`nsl_cfie_decode_attn_l{N}` — INT8 layers dequantize in
    /// registers, FP16 layers load directly; no runtime dispatch).
    /// `(kernel_name, ptx)` per layer.  NOTE: the quant kernel family
    /// uses the mixed-precision pool layout from
    /// `cfie_kv_quant_ptx::pool_layout`, which differs from the
    /// uniform-f16 pool the base/block/verify kernels bake; the
    /// compile-time chooser ([`choose_kernel_family`]) registers ONE
    /// family per build.
    pub quant_attention_kernels: Vec<(String, String)>,
    /// Feature 2 (F2): the fused decode-sample kernel (RMSNorm +
    /// LM-head matvec + top-k/top-p + multinomial in ONE launch, only
    /// the token id touches HBM).  Emitted by the serve wiring when
    /// the sampler program is fused and the emitter preconditions hold
    /// (d_model 1..=8192, vocab_size a positive multiple of the
    /// 128-wide tile, top_k 1..=64).  When a grammar mask exists the
    /// initialized `.global` fragment is spliced into this module so
    /// the host resolves `nsl_cfie_grammar_mask` via
    /// `cuModuleGetGlobal` at engine finalize.
    pub fused_sample_kernel: Option<String>,
    pub fused_sample_ptx: Option<String>,
    /// Launch geometry per kernel family, registered with
    /// `nsl_cfie_register_kernel` at serve init.  Set alongside the
    /// matching kernel/ptx pair; the quant layers share one spec.
    pub decode_attention_launch: Option<CfieLaunchSpec>,
    pub fused_sample_launch: Option<CfieLaunchSpec>,
    pub decode_block_launch: Option<CfieLaunchSpec>,
    pub spec_verify_launch: Option<CfieLaunchSpec>,
    pub spec_reject_launch: Option<CfieLaunchSpec>,
    pub quant_attention_launch: Option<CfieLaunchSpec>,
    /// Whether the compiled binary actually carries the Cycle-6 runtime
    /// wiring (register/pool/finalize/destroy emission).  Monolithic
    /// serve sets this; the disaggregated path does NOT emit the wiring
    /// and the report must not claim it (`false` for hand-built plans).
    pub runtime_wiring_emitted: bool,
    /// G22: estimated decode latency + throughput from the explicit
    /// roofline cost model (`cfie_cost`).  Set by `run_cfie_for_serve`
    /// (which has the resolved shape + weights precision + GPU spec);
    /// hand-built plans and `cfie::run` leave it `None` and the report
    /// simply omits the two estimate lines.
    pub cost_estimate: Option<CfieCostEstimate>,
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
        // Wiring suffix, truthful per serve path: monolithic serve
        // emits the register/pool/finalize calls; the disaggregated
        // path (and hand-built plans) does not, so the report must not
        // claim registration there.
        let wiring = if self.runtime_wiring_emitted {
            "registered at serve init; host decode loop = nsl_cfie_decode_step"
        } else {
            "runtime wiring NOT emitted on this serve path (report-only)"
        };
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
                "      direct-index decode attention: {} emitted ({} bytes PTX, {wiring})",
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
        if let Some(kernel) = self.fused_sample_kernel.as_ref() {
            writeln!(
                s,
                "      fused sampler kernel: {} emitted ({} bytes PTX, {wiring})",
                kernel,
                self.fused_sample_ptx.as_ref().map_or(0, |p| p.len())
            )
            .unwrap();
        }
        if let Some(spec) = self.speculative.as_ref() {
            writeln!(
                s,
                "  [3] Compiled speculative: method={}, K={}, expected speedup={:.2}×",
                spec.config.method.as_str(),
                spec.config.k_tokens,
                spec.expected_speedup
            )
            .unwrap();
            if let Some(kernel) = self.spec_verify_kernel.as_ref() {
                writeln!(
                    s,
                    "      verify attention: {} emitted ({} bytes PTX, tree mask baked, {wiring})",
                    kernel,
                    self.spec_verify_ptx.as_ref().map_or(0, |p| p.len())
                )
                .unwrap();
            }
            if let Some(kernel) = self.spec_reject_kernel.as_ref() {
                writeln!(
                    s,
                    "      rejection epilogue: {} emitted ({} bytes PTX, {wiring})",
                    kernel,
                    self.spec_reject_ptx.as_ref().map_or(0, |p| p.len())
                )
                .unwrap();
            }
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
        if let Some(kernel) = self.decode_block_kernel.as_ref() {
            writeln!(
                s,
                "      persistent decode block: {} emitted ({} bytes PTX, {wiring})",
                kernel,
                self.decode_block_ptx.as_ref().map_or(0, |p| p.len())
            )
            .unwrap();
        }
        writeln!(
            s,
            "  [5] Per-layer KV quant: {} INT8 layers, {:.1}% memory savings",
            self.kv_quant.int8_layer_count(),
            100.0 * self.kv_quant.memory_savings_ratio()
        )
        .unwrap();
        if !self.quant_attention_kernels.is_empty() {
            writeln!(
                s,
                "      per-layer decode-attention kernels: {} emitted \
                 (precision baked per layer)",
                self.quant_attention_kernels.len()
            )
            .unwrap();
            // Family exclusivity: the chooser registers the quant
            // family INSTEAD OF decode-block/spec — never both.  Only
            // claimed when a launch spec exists (i.e. the serve wiring
            // actually selected the family, not a hand-built plan) and
            // worded per path so it never contradicts the family line.
            if self.quant_attention_launch.is_some() {
                if self.runtime_wiring_emitted {
                    writeln!(
                        s,
                        "      quant family registered exclusively at serve init; \
                         decode-block/spec kernels NOT registered \
                         (mixed-precision pool layout is incompatible)"
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "      quant family selected exclusively (report-only on \
                         this path); decode-block/spec kernels excluded \
                         (mixed-precision pool layout is incompatible)"
                    )
                    .unwrap();
                }
            }
        }
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
                if self.fused_sample_ptx.is_some() {
                    writeln!(
                        s,
                        "      mask baked into module image ({} bytes PTX, spliced into the fused sampler module; host binds it via cuModuleGetGlobal at engine finalize)",
                        mask.len()
                    )
                    .unwrap();
                } else {
                    writeln!(
                        s,
                        "      mask baked into module image ({} bytes PTX, fused sampler not emitted this build - mask unbound)",
                        mask.len()
                    )
                    .unwrap();
                }
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
        // G22: estimated latency + throughput from the roofline cost
        // model.  Present only when the serve wiring computed it (it has
        // the shape + GPU spec); hand-built plans omit these two lines.
        // ASCII-only: the paper's microsecond renders as "us", never a
        // Unicode micro sign.
        if let Some(est) = self.cost_estimate.as_ref() {
            writeln!(
                s,
                "Estimated decode latency: {:.1}us/token (vs {:.1}us baseline) \
                 [bandwidth-roofline model @ seq={}]",
                est.cfie_us_per_token, est.baseline_us_per_token, est.kv_seq
            )
            .unwrap();
            writeln!(
                s,
                "Estimated throughput (batch={}): {} tok/s \
                 [same model; weights amortized across batch]",
                est.batch,
                group_thousands(est.throughput_tok_s.round() as u64)
            )
            .unwrap();
            writeln!(s, "  cost-model assumptions: {}", est.assumptions).unwrap();
        }
        writeln!(s, "Continuous batching: on ({} active slots, ring capacity {})",
                 self.persistent.scheduler.max_active,
                 self.persistent.scheduler.ring_buffer.capacity).unwrap();
        let regs = kernel_registrations(self);
        if regs.is_empty() {
            writeln!(s, "Kernel family: none registered (no kernels emitted this build)")
                .unwrap();
        } else {
            writeln!(
                s,
                "Kernel family: {} — {} kernel(s); {wiring}",
                choose_kernel_family(self).as_str(),
                regs.len()
            )
            .unwrap();
        }
        writeln!(
            s,
            "Note: endpoint-driven generation still requires model binding — \
             serve blocks carry no model reference yet, so the compiled \
             decode loop is driven by tests/host callers until binding lands."
        )
        .unwrap();
        writeln!(s, "Solve time: {:.2} ms", self.solve_us as f64 / 1000.0).unwrap();
        s
    }
}

/// Group an integer with ASCII commas every three digits (the paper
/// prints throughput as `18,400 tok/s`).  ASCII-only by construction.
fn group_thousands(n: u64) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    let len = bytes.len();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
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
            decode_block_kernel: None,
            decode_block_ptx: None,
            spec_verify_kernel: None,
            spec_verify_ptx: None,
            spec_reject_kernel: None,
            spec_reject_ptx: None,
            grammar_mask_ptx: None,
            quant_attention_kernels: Vec::new(),
            fused_sample_kernel: None,
            fused_sample_ptx: None,
            decode_attention_launch: None,
            fused_sample_launch: None,
            decode_block_launch: None,
            spec_verify_launch: None,
            spec_reject_launch: None,
            quant_attention_launch: None,
            runtime_wiring_emitted: false,
            cost_estimate: None,
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
        decode_block_kernel: None,
        decode_block_ptx: None,
        spec_verify_kernel: None,
        spec_verify_ptx: None,
        spec_reject_kernel: None,
        spec_reject_ptx: None,
        grammar_mask_ptx: None,
        quant_attention_kernels: Vec::new(),
        fused_sample_kernel: None,
        fused_sample_ptx: None,
        decode_attention_launch: None,
        fused_sample_launch: None,
        decode_block_launch: None,
        spec_verify_launch: None,
        spec_reject_launch: None,
        quant_attention_launch: None,
        runtime_wiring_emitted: false,
        cost_estimate: None,
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

    // ── Cycle 6: family chooser + registration list (pure fns) ──────

    const SPEC_1X128: CfieLaunchSpec = CfieLaunchSpec {
        grid_x: 1,
        block_x: 128,
        smem_dyn_bytes: 0,
    };

    #[test]
    fn family_chooser_quant_wins_when_quant_kernels_exist() {
        let mut plan = run(nslcoder_input());
        assert_eq!(choose_kernel_family(&plan), CfieKernelFamily::Uniform);
        plan.quant_attention_kernels = vec![("k".into(), "p".into())];
        assert_eq!(choose_kernel_family(&plan), CfieKernelFamily::Quant);
    }

    #[test]
    fn kernel_registrations_uniform_family_lists_emitted_kernels_plus_sampler() {
        let mut plan = run(nslcoder_input());
        plan.decode_attention_kernel = Some("attn".into());
        plan.decode_attention_ptx = Some("ptx_attn".into());
        plan.decode_attention_launch = Some(CfieLaunchSpec {
            grid_x: 8,
            block_x: 128,
            smem_dyn_bytes: 0,
        });
        plan.decode_block_kernel = Some("blk".into());
        plan.decode_block_ptx = Some("ptx_blk".into());
        plan.decode_block_launch = Some(SPEC_1X128);
        plan.spec_reject_kernel = Some("rej".into());
        plan.spec_reject_ptx = Some("ptx_rej".into());
        plan.spec_reject_launch = Some(CfieLaunchSpec {
            grid_x: 1,
            block_x: 32,
            smem_dyn_bytes: 0,
        });
        plan.fused_sample_kernel = Some("samp".into());
        plan.fused_sample_ptx = Some("ptx_samp".into());
        plan.fused_sample_launch = Some(SPEC_1X128);

        let regs = kernel_registrations(&plan);
        assert_eq!(
            regs.iter().map(|r| r.kind).collect::<Vec<_>>(),
            vec![0, 2, 4, 1],
            "uniform family: whichever of 0/2/3/4 exist, plus kind 1"
        );
        assert!(regs.iter().all(|r| r.layer_idx == 0));
        assert!(regs.iter().all(|r| r.smem_dyn == 0));
        let attn = &regs[0];
        assert_eq!((attn.grid_x, attn.block_x), (8, 128));
        let rej = regs.iter().find(|r| r.kind == 4).unwrap();
        assert_eq!((rej.grid_x, rej.block_x), (1, 32));
    }

    #[test]
    fn kernel_registrations_quant_family_excludes_uniform_kv_kernels() {
        let mut plan = run(nslcoder_input());
        // Uniform kernels emitted on the plan...
        plan.decode_attention_kernel = Some("attn".into());
        plan.decode_attention_ptx = Some("ptx_attn".into());
        plan.decode_attention_launch = Some(SPEC_1X128);
        plan.decode_block_kernel = Some("blk".into());
        plan.decode_block_ptx = Some("ptx_blk".into());
        plan.decode_block_launch = Some(SPEC_1X128);
        // ...but the quant family wins registration.
        plan.quant_attention_kernels = vec![
            ("q0".into(), "ptx_q0".into()),
            ("q1".into(), "ptx_q1".into()),
        ];
        plan.quant_attention_launch = Some(CfieLaunchSpec {
            grid_x: 4,
            block_x: 128,
            smem_dyn_bytes: 0,
        });
        plan.fused_sample_kernel = Some("samp".into());
        plan.fused_sample_ptx = Some("ptx_samp".into());
        plan.fused_sample_launch = Some(SPEC_1X128);

        let regs = kernel_registrations(&plan);
        assert_eq!(
            regs.iter().map(|r| (r.kind, r.layer_idx)).collect::<Vec<_>>(),
            vec![(5, 0), (5, 1), (1, 0)],
            "quant family: kind 5 per layer + kind 1; kinds 0/2/3/4 excluded"
        );
        assert_eq!(regs[0].name, "q0");
        assert_eq!(regs[1].name, "q1");
    }

    #[test]
    fn kernel_registrations_skip_kernels_without_launch_spec() {
        let mut plan = run(nslcoder_input());
        plan.decode_attention_kernel = Some("attn".into());
        plan.decode_attention_ptx = Some("ptx_attn".into());
        // decode_attention_launch deliberately left None.
        assert!(kernel_registrations(&plan).is_empty());
    }

    #[test]
    fn report_family_line_none_without_kernels_and_named_with() {
        let mut plan = run(nslcoder_input());
        assert!(plan
            .render_report()
            .contains("Kernel family: none registered"));
        plan.fused_sample_kernel = Some("nsl_cfie_fused_sample".into());
        plan.fused_sample_ptx = Some("ptx".into());
        plan.fused_sample_launch = Some(SPEC_1X128);
        let rep = plan.render_report();
        assert!(rep.contains("Kernel family: uniform"));
        assert!(rep.contains("fused sampler kernel: nsl_cfie_fused_sample emitted"));
        assert!(!rep.contains("launch wiring pending"));
    }
}
