//! CFIE — Compiler-Fused Inference Engine: driver + report.
//!
//! Composes the CFIE planning passes (KV layout, fused sampler, decode
//! block, per-layer KV precision, speculative decoding) into a single
//! [`CfiePlan`].  Invoked from `serve.rs::run_cfie_for_serve` with inputs
//! assembled by `cfie_serve.rs`.
//!
//! What a plan drives: on the monolithic serve path `serve.rs` emits the
//! decode-block kernel and the fused sampler (and, with a
//! `speculative:` section, the draft model's two kernels), embeds their
//! PTX and emits the `nsl_cfie_register_kernel` /
//! `nsl_cfie_kv_pool_alloc` / `nsl_cfie_engine_finalize` calls, so they
//! are registered with the runtime engine at serve init.  The endpoint's
//! `generate()` compiles to serve-init model binding (`nsl_model_create`
//! + `nsl_cfie_bind_model`) and the `nsl_cfie_generate` decode driver,
//! with the served model resolved from the serve `weights:` key /
//! `--weights`.  With `tokenizer:` configured the `prompt:` text is
//! runtime-encoded and the generated ids are decoded and printed.  The
//! disaggregated serve path emits none of the runtime wiring and the
//! report says so (`runtime_wiring_emitted`).
//!
//! The report states what the build contains.  It prints no latency or
//! throughput figure: the repository has no measurement of this engine,
//! and the cost model that used to supply those lines divided by the
//! whole GPU's memory bandwidth for kernels that run on one thread block.

use serde::Serialize;

use crate::cfie_fused_sample::{emit_program as emit_sample, FusedSampleProgram, LmHeadShape, SamplingParams};
use crate::cfie_kv_plan::{plan as plan_kv, KvBudget, KvLayoutPlan, KvShape};
use crate::cfie_kv_quant::{plan as plan_kv_quant, KvQuantConfig, KvQuantPlan};
use crate::cfie_persistent::{plan as plan_persistent, GpuBudget, PersistentModel, PersistentPlan};
use crate::cfie_speculative::SpeculativeConfig;
use crate::pass_trace::{DeclineReason, PassDisposition};
use crate::weight_aware::WeightMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CfieMode {
    /// All passes active.
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
/// compiled serve binary passes to `nsl_cfie_register_kernel` at init.
/// 1-D launches only: every CFIE kernel is
/// `grid=(grid_x,1,1) block=(block_x,1,1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfieLaunchSpec {
    pub grid_x: u32,
    pub block_x: u32,
    /// DYNAMIC shared-memory bytes for the launch.  0 for every current
    /// CFIE kernel — they all declare static `.shared` inside the
    /// module — the field exists for future extern-`.shared` kernels.
    pub smem_dyn_bytes: u32,
}

/// Registration kinds, as `nsl_runtime::cfie::engine` numbers them.
pub const KIND_FUSED_SAMPLE: u8 = 1;
pub const KIND_DECODE_BLOCK: u8 = 2;
pub const KIND_DRAFT_BLOCK: u8 = 6;
pub const KIND_DRAFT_SAMPLE: u8 = 7;

/// One `nsl_cfie_register_kernel` call the compiled serve binary makes
/// at init.  `layer_idx` is 0 for every kind there is today.
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

/// Build the registration list (pure over the plan).  Every kernel listed
/// here is launched by a generation driver: the decode block and the
/// fused sampler by `nsl_cfie_decode_step`, the draft pair by
/// `nsl_cfie_speculative_generate`.  The list is empty unless the decode
/// path is complete — a sampler with no decode block (or the reverse)
/// would be registered and never launched.
pub fn kernel_registrations(plan: &CfiePlan) -> Vec<CfieKernelReg> {
    if !plan.decode_path_emitted() {
        return Vec::new();
    }
    fn reg(kind: u8, name: &str, ptx: &str, spec: CfieLaunchSpec) -> CfieKernelReg {
        CfieKernelReg {
            kind,
            layer_idx: 0,
            name: name.to_string(),
            ptx: ptx.to_string(),
            grid_x: spec.grid_x,
            block_x: spec.block_x,
            smem_dyn: spec.smem_dyn_bytes,
        }
    }
    let mut regs = Vec::new();
    let mut push = |kind: u8, k: &CfieKernel| {
        if let Some(e) = k.emitted.as_ref() {
            regs.push(reg(kind, &e.name, &e.ptx, e.launch));
        }
    };
    push(KIND_DECODE_BLOCK, &plan.decode_block);
    push(KIND_DRAFT_BLOCK, &plan.draft_block);
    push(KIND_DRAFT_SAMPLE, &plan.draft_sample);
    push(KIND_FUSED_SAMPLE, &plan.fused_sample);
    regs
}

/// A kernel the serve wiring emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedKernel {
    pub name: String,
    pub ptx: String,
    pub launch: CfieLaunchSpec,
}

/// One kernel slot of the plan: the kernel, or why this build has none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CfieKernel {
    pub emitted: Option<EmittedKernel>,
    /// Set by the serve wiring when it decided not to emit the kernel:
    /// the reason, in words a user can act on.  `None` with no kernel
    /// means the wiring did not consider it (a hand-built plan, or a
    /// draft kernel with no `speculative:` section).
    pub not_emitted: Option<String>,
}

impl CfieKernel {
    pub fn emitted(name: String, ptx: String, launch: CfieLaunchSpec) -> Self {
        Self {
            emitted: Some(EmittedKernel { name, ptx, launch }),
            not_emitted: None,
        }
    }
    pub fn refused(why: impl Into<String>) -> Self {
        Self {
            emitted: None,
            not_emitted: Some(why.into()),
        }
    }
    pub fn is_emitted(&self) -> bool {
        self.emitted.is_some()
    }
}

/// Everything the serve wiring resolved about the draft model compiled
/// into the binary — the checkpoint path, the draft architecture (vocab
/// is the target's; shared-vocab invariant), the draft KV pool size
/// (`draft_n_layers * 2 * per_slot * draft_n_kv_heads * draft_head_dim *
/// 2` f16 bytes, with per_slot = the TARGET per-slot capacity per the
/// engine's sizing contract), and the speculative K.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfieSpeculativeDraft {
    pub weights_path: String,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_ff: u32,
    /// The shared vocab (== the target's resolved vocab_size).
    pub vocab_size: u32,
    /// Draft KV pool bytes for `nsl_cfie_draft_pool_alloc`.
    pub pool_bytes: i64,
    /// Speculative K (`speculative.tokens`, 1..=32) — passed to
    /// `nsl_cfie_speculative_generate`.
    pub k_tokens: u32,
    /// The TARGET model's layer count (for the report's per-round
    /// launch accounting).
    pub target_n_layers: u32,
}

/// Driver input.
pub struct CfieInput<'a> {
    pub mode: CfieMode,
    pub target_gpu: &'a str,
    pub kv_shape: KvShape,
    pub kv_budget: KvBudget,
    pub lm_head: LmHeadShape,
    pub sampling: SamplingParams,
    /// Whether `sampling` came from a `sampling:` section (false: the
    /// defaults, which the report then names as defaults).
    pub sampling_configured: bool,
    pub persistent_model: PersistentModel,
    pub gpu_budget: GpuBudget,
    pub max_active_requests: u32,
    pub kv_quant: KvQuantConfig,
    /// Whether to run the advisory per-layer KV precision ranking
    /// (`kv_quant: "auto"`).
    pub kv_quant_advisory: bool,
    pub weights: Option<&'a WeightMap>,
    pub speculative: Option<SpeculativeConfig>,
}

/// Launches the unfused pipeline the plan is compared with would make
/// for the sampling tail (RMSNorm, LM-head matmul, softmax, top-k, top-p,
/// multinomial).
const UNFUSED_SAMPLING_LAUNCHES: u32 = 6;

#[derive(Debug, Clone)]
pub struct CfiePlan {
    pub mode: CfieMode,
    pub target_gpu: String,
    pub kv: KvLayoutPlan,
    pub sampling: FusedSampleProgram,
    /// False when no `sampling:` section set the sampler's parameters.
    pub sampling_configured: bool,
    pub persistent: PersistentPlan,
    /// Advisory per-layer KV precision ranking.  NOT applied: every layer
    /// of the KV pool is FP16, the only precision the decode block reads
    /// and writes.  Empty unless `kv_quant: "auto"` asked for it.
    pub kv_quant: KvQuantPlan,
    pub speculative: Option<SpeculativeConfig>,
    /// The target model's layer count.
    pub n_layers: u32,
    /// Launches per generated token of the nominal unfused pipeline:
    /// `baseline_launches_per_layer * n_layers + 6`.  A count of
    /// operations, not a measurement of any engine.
    pub kernel_launches_per_token_baseline: u32,
    pub solve_us: u64,
    /// The decode-block kernel (one launch = one layer's decode step for
    /// one token), emitted when the planner sustains whole-layer fusion,
    /// the KV layout is static, and the emitter's preconditions hold.
    pub decode_block: CfieKernel,
    /// The fused decode-sample kernel (RMSNorm + LM-head matvec +
    /// top-k/top-p + multinomial in ONE launch; only the token id is
    /// written back).
    pub fused_sample: CfieKernel,
    /// The draft model's decode block and greedy sampler, emitted with a
    /// `speculative:` section; `nsl_cfie_speculative_generate` drives
    /// them.
    pub draft_block: CfieKernel,
    pub draft_sample: CfieKernel,
    /// Resolved draft-model wiring info — set by the serve wiring exactly
    /// when the two draft kernels are emitted.  Drives the serve-init
    /// draft binding emission and the `generate()` rewrite to
    /// `nsl_cfie_speculative_generate`.
    pub speculative_draft: Option<CfieSpeculativeDraft>,
    /// Whether the compiled binary actually carries the runtime wiring
    /// (register/pool/finalize/destroy emission).  Monolithic serve sets
    /// this; the disaggregated path does NOT emit the wiring and the
    /// report must not claim it (`false` for hand-built plans).
    pub runtime_wiring_emitted: bool,
}

impl CfiePlan {
    /// Whether this build can run `generate()`: the runtime's decode loop
    /// needs the decode block and the fused sampler, both.
    pub fn decode_path_emitted(&self) -> bool {
        self.decode_block.is_emitted() && self.fused_sample.is_emitted()
    }

    /// Why `generate()` cannot run in this build, when it cannot.
    pub fn generate_unavailable(&self) -> Option<String> {
        let mut why = Vec::new();
        for (what, k) in [
            ("the decode-block kernel", &self.decode_block),
            ("the fused sampler kernel", &self.fused_sample),
        ] {
            if !k.is_emitted() {
                why.push(format!(
                    "{what} was not emitted: {}",
                    k.not_emitted.as_deref().unwrap_or("it was not planned")
                ));
            }
        }
        if why.is_empty() {
            None
        } else {
            Some(why.join("; "))
        }
    }

    /// Launches per generated token in this build: one decode-block
    /// launch per layer, then one sampler launch.  `None` when the decode
    /// path was not emitted.
    pub fn kernel_launches_per_token(&self) -> Option<u32> {
        self.decode_path_emitted().then_some(self.n_layers + 1)
    }

    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        // Wiring suffix, truthful per serve path: monolithic serve
        // emits the register/pool/finalize calls; the disaggregated
        // path (and hand-built plans) does not, so the report must not
        // claim registration there.
        let wiring = if !self.runtime_wiring_emitted {
            "runtime wiring NOT emitted on this serve path (report-only)"
        } else if self.decode_path_emitted() {
            "registered at serve init"
        } else {
            "NOT registered: the decode path is incomplete, so nothing would launch it"
        };
        let kernel_line = |k: &CfieKernel| -> String {
            match (k.emitted.as_ref(), k.not_emitted.as_ref()) {
                (Some(e), _) => format!("{} emitted ({} bytes PTX, {wiring})", e.name, e.ptx.len()),
                (None, Some(why)) => format!("NOT emitted — {why}"),
                (None, None) => "not planned in this build".to_string(),
            }
        };
        let mut s = String::new();
        writeln!(s, "=== CFIE Inference Build Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target: {}", self.target_gpu).unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Plan:").unwrap();
        writeln!(s, "  [1] KV layout: {} ({})", self.kv.kind.as_str(), self.kv.rationale).unwrap();

        let p = &self.sampling.params;
        let source = if self.sampling_configured {
            "from the sampling: section"
        } else {
            "DEFAULTS — no sampling: section set them"
        };
        let settings = match p.strategy {
            crate::cfie_fused_sample::SamplingStrategy::Greedy => "greedy (temperature 0)".to_string(),
            crate::cfie_fused_sample::SamplingStrategy::TopKTopP => {
                format!("temperature {}, top_k {}, top_p {}", p.temperature, p.top_k, p.top_p)
            }
            _ => format!("temperature {}, top_k {}, no top_p", p.temperature, p.top_k),
        };
        writeln!(s, "  [2] Sampler: {settings} ({source})").unwrap();
        writeln!(s, "      fused sampler kernel: {}", kernel_line(&self.fused_sample)).unwrap();

        match (self.speculative.as_ref(), self.speculative_draft.as_ref()) {
            (Some(_), Some(draft)) => {
                writeln!(
                    s,
                    "  [3] Speculative decoding: K={}, greedy draft model from '{}' ({} layers, d_model {}; draft KV pool {} bytes)",
                    draft.k_tokens, draft.weights_path, draft.n_layers, draft.d_model, draft.pool_bytes
                )
                .unwrap();
                writeln!(s, "      draft decode block: {}", kernel_line(&self.draft_block)).unwrap();
                writeln!(s, "      draft greedy sampler: {}", kernel_line(&self.draft_sample)).unwrap();
                // Per round: K draft steps (draft layers + the draft
                // sampler each) and at most K target steps (target
                // layers + the fused sampler each).
                let k = draft.k_tokens as u64;
                let per_round =
                    k * (draft.n_layers as u64 + 1) + k * (draft.target_n_layers as u64 + 1);
                writeln!(
                    s,
                    "      output: the tokens plain decoding gives for the same seed (each is drawn by the target's sampler; the draft only predicts it)"
                )
                .unwrap();
                writeln!(
                    s,
                    "      cost: up to {per_round} launches per round for at most {k} tokens, against {} per token without a draft — SLOWER than plain decoding; the target verifies one position per step, not K in one pass",
                    draft.target_n_layers as u64 + 1
                )
                .unwrap();
            }
            _ => writeln!(s, "  [3] Speculative decoding: off").unwrap(),
        }

        writeln!(
            s,
            "  [4] Decode block (planner: fusion={}): {}",
            self.persistent.fusion.as_str(),
            kernel_line(&self.decode_block)
        )
        .unwrap();

        writeln!(
            s,
            "  [5] KV precision: FP16 in every layer (the only precision the decode block reads and writes)"
        )
        .unwrap();
        if !self.kv_quant.layers.is_empty() {
            writeln!(
                s,
                "      advisory, NOT applied: {} of {} layers rank as INT8-tolerant. The ranking is relative — each layer's weight spectrum is scaled between this model's smallest and largest, so some layer always ranks lowest and some highest whatever their real sensitivity.",
                self.kv_quant.int8_layer_count(),
                self.kv_quant.layers.len()
            )
            .unwrap();
        }
        writeln!(s).unwrap();

        match self.kernel_launches_per_token() {
            Some(n) => writeln!(
                s,
                "Kernel launches per generated token: {n} ({} decode-block + 1 sampler; a prompt token takes {} — its prediction is not sampled). The unfused pipeline this replaces would make {} ({} per layer + {UNFUSED_SAMPLING_LAUNCHES} for the sampling tail): a count of operations, not a measurement.",
                self.n_layers,
                self.n_layers,
                self.kernel_launches_per_token_baseline,
                self.persistent.baseline_launches_per_layer
            )
            .unwrap(),
            None => writeln!(
                s,
                "Kernel launches per generated token: none — the decode path is not in this build."
            )
            .unwrap(),
        }
        writeln!(
            s,
            "Speed: not measured. Each kernel runs on one thread block; this report estimates no latency or throughput."
        )
        .unwrap();
        writeln!(
            s,
            "Requests: one sequence at a time ({} KV slot(s) planned; there is no batching scheduler).",
            self.persistent.scheduler.max_active
        )
        .unwrap();

        let regs = kernel_registrations(self);
        if regs.is_empty() {
            writeln!(s, "Kernels: none registered in this build.").unwrap();
        } else {
            writeln!(
                s,
                "Kernels: {} ({}); {wiring}.",
                regs.len(),
                regs.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join(", ")
            )
            .unwrap();
        }
        match self.generate_unavailable() {
            Some(why) => writeln!(
                s,
                "generate(): NOT available in this build — {why}. An endpoint that calls generate() does not compile."
            )
            .unwrap(),
            None => {
                let driver = if self.speculative_draft.is_some() {
                    "nsl_cfie_speculative_generate"
                } else {
                    "nsl_cfie_generate"
                };
                writeln!(
                    s,
                    "generate(): compiles to serve-init model binding (nsl_model_create + nsl_cfie_bind_model) and the {driver} driver; the served model is bound from the serve `weights:` key / --weights. It returns the generated-token count. With `tokenizer:` configured, the `prompt:` text is runtime-encoded through that tokenizer (falling back to baked byte-level ids if the encode yields nothing) and the generated ids are decoded and printed as TEXT; without a tokenizer the prompt stays byte-baked and only the count is printed."
                )
                .unwrap();
            }
        }
        writeln!(s, "Solve time: {:.2} ms", self.solve_us as f64 / 1000.0).unwrap();
        s
    }
}

pub fn run(input: CfieInput) -> CfiePlan {
    crate::pass_trace::record("CFIE");
    let t0 = std::time::Instant::now();

    let kv = plan_kv(&input.kv_shape, &input.kv_budget);
    let sampling = emit_sample(input.sampling, input.lm_head);
    let persistent = plan_persistent(
        &input.persistent_model,
        &input.gpu_budget,
        input.max_active_requests,
    );
    let n_layers = input.persistent_model.n_layers;
    let baseline = persistent.baseline_launches_per_layer * n_layers + UNFUSED_SAMPLING_LAUNCHES;
    let full = input.mode == CfieMode::Full;

    // Advisory per-layer KV precision ranking.
    let kv_quant = if full && input.kv_quant_advisory {
        plan_kv_quant(&input.kv_quant, input.weights)
    } else {
        KvQuantPlan::default()
    };
    let speculative = if full { input.speculative.clone() } else { None };

    if input.mode == CfieMode::Off {
        crate::pass_trace::record_disposition("CFIE", PassDisposition::Declined {
            reason: DeclineReason::ModeOff,
        });
    } else {
        // CFIE's product is launch collapse; the rewrite count is the
        // number of per-token launches the plan removes from the nominal
        // unfused pipeline (CFIE never sees a WengertList, which is why
        // the registry files it as OutOfBand).
        let planned = persistent.persistent_launches_per_layer * n_layers
            + if sampling.is_fused() { 1 } else { UNFUSED_SAMPLING_LAUNCHES };
        let launches_eliminated = baseline.saturating_sub(planned) as usize;
        if launches_eliminated == 0 {
            crate::pass_trace::record_disposition("CFIE", PassDisposition::Declined {
                reason: DeclineReason::NoCandidates(
                    "neither the persistent decode nor the fused sampler removed a launch",
                ),
            });
        } else {
            crate::pass_trace::record_disposition("CFIE", PassDisposition::Applied {
                rewrites: launches_eliminated,
            });
        }
    }
    CfiePlan {
        mode: input.mode,
        target_gpu: input.target_gpu.to_string(),
        kv,
        sampling,
        sampling_configured: input.sampling_configured,
        persistent,
        kv_quant,
        speculative,
        n_layers,
        kernel_launches_per_token_baseline: baseline,
        solve_us: t0.elapsed().as_micros() as u64,
        decode_block: CfieKernel::default(),
        fused_sample: CfieKernel::default(),
        draft_block: CfieKernel::default(),
        draft_sample: CfieKernel::default(),
        speculative_draft: None,
        runtime_wiring_emitted: false,
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
            sampling_configured: true,
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
            kv_quant_advisory: true,
            weights: None,
            speculative: Some(SpeculativeConfig::default()),
        }
    }

    fn launch(grid_x: u32, block_x: u32) -> CfieLaunchSpec {
        CfieLaunchSpec { grid_x, block_x, smem_dyn_bytes: 0 }
    }

    /// The plan the serve wiring produces for a build with the decode
    /// path emitted.
    fn emitted_plan() -> CfiePlan {
        let mut plan = run(nslcoder_input());
        plan.decode_block = CfieKernel::emitted("nsl_cfie_decode_block".into(), "BLOCK".into(), launch(1, 128));
        plan.fused_sample = CfieKernel::emitted("nsl_cfie_fused_sample".into(), "SAMPLE".into(), launch(1, 128));
        plan.runtime_wiring_emitted = true;
        plan
    }

    #[test]
    fn full_mode_populates_every_pass() {
        let plan = run(nslcoder_input());
        assert_eq!(plan.mode, CfieMode::Full);
        assert!(plan.kv.direct.is_some());
        assert!(plan.sampling.is_fused());
        assert!(plan.speculative.is_some());
        assert!(!plan.kv_quant.layers.is_empty());
    }

    #[test]
    fn off_and_sampling_modes_plan_no_speculation_and_no_advisory() {
        for mode in [CfieMode::Off, CfieMode::Sampling] {
            let mut input = nslcoder_input();
            input.mode = mode;
            let plan = run(input);
            assert!(plan.speculative.is_none(), "{mode:?}");
            assert!(plan.kv_quant.layers.is_empty(), "{mode:?}");
        }
    }

    /// The baseline is the planner's own count for THIS model, not the
    /// proposal's "~500" (a 32-layer figure): 12 launches per layer for 8
    /// layers, plus 6 for the sampling tail.
    #[test]
    fn the_baseline_is_counted_from_the_model() {
        let plan = run(nslcoder_input());
        assert_eq!(plan.persistent.baseline_launches_per_layer, 12);
        assert_eq!(plan.kernel_launches_per_token_baseline, 12 * 8 + 6);
        let mut input = nslcoder_input();
        input.persistent_model.n_layers = 32;
        assert_eq!(run(input).kernel_launches_per_token_baseline, 12 * 32 + 6);
    }

    /// The launch count is what the decode loop does — one decode-block
    /// launch per layer and one sampler launch — and exists only when
    /// both kernels are in the build.  The report used to print 407 for
    /// this model.
    #[test]
    fn launches_per_token_are_the_decode_loops() {
        let plan = run(nslcoder_input());
        assert_eq!(plan.kernel_launches_per_token(), None, "nothing emitted yet");
        let plan = emitted_plan();
        assert_eq!(plan.kernel_launches_per_token(), Some(9));
        let rep = plan.render_report();
        assert!(rep.contains("Kernel launches per generated token: 9 (8 decode-block + 1 sampler"), "{rep}");
        assert!(rep.contains("would make 102 (12 per layer + 6 for the sampling tail)"), "{rep}");
        assert!(!rep.contains("407") && !rep.contains("500"), "{rep}");
    }

    #[test]
    fn the_report_claims_no_speed_and_no_batching() {
        let rep = emitted_plan().render_report();
        for gone in ["us/token", "tok/s", "Estimated", "Continuous batching", "speedup", "baseline,"] {
            assert!(!rep.contains(gone), "the report still prints `{gone}`:\n{rep}");
        }
        assert!(rep.contains("Speed: not measured."), "{rep}");
        assert!(rep.contains("Requests: one sequence at a time"), "{rep}");
    }

    /// One statement per kernel: emitted and registered, or not emitted
    /// and why.  The report used to say the decode block was "registered
    /// at serve init" two lines above "decode-block/spec kernels NOT
    /// registered".
    #[test]
    fn the_report_says_one_thing_about_each_kernel() {
        let rep = emitted_plan().render_report();
        assert!(rep.contains("[4] Decode block (planner: fusion=level3): nsl_cfie_decode_block emitted (5 bytes PTX, registered at serve init)"), "{rep}");
        assert!(rep.contains("Kernels: 2 (nsl_cfie_decode_block, nsl_cfie_fused_sample); registered at serve init."), "{rep}");
        for absent in ["NOT emitted", "NOT registered", "NOT available"] {
            assert!(!rep.contains(absent), "`{absent}` in:\n{rep}");
        }

        let mut plan = emitted_plan();
        plan.decode_block = CfieKernel::refused("static shared memory of 70000 bytes exceeds the 49152-byte limit of one thread block");
        let rep = plan.render_report();
        assert!(rep.contains("[4] Decode block (planner: fusion=level3): NOT emitted — static shared memory of 70000 bytes"), "{rep}");
        assert!(!rep.contains("nsl_cfie_decode_block emitted"), "{rep}");
        // The sampler was built, but with no decode block nothing would
        // launch it, so it is not registered and the report says so.
        assert!(rep.contains("nsl_cfie_fused_sample emitted (6 bytes PTX, NOT registered: the decode path is incomplete"), "{rep}");
        assert!(rep.contains("Kernels: none registered in this build."), "{rep}");
        assert!(kernel_registrations(&plan).is_empty());
        assert!(rep.contains("generate(): NOT available in this build — the decode-block kernel was not emitted: static shared memory"), "{rep}");
        assert!(rep.contains("Kernel launches per generated token: none"), "{rep}");
    }

    #[test]
    fn generate_needs_both_kernels() {
        let plan = run(nslcoder_input());
        let why = plan.generate_unavailable().unwrap();
        assert!(why.contains("the decode-block kernel was not emitted"), "{why}");
        assert!(why.contains("the fused sampler kernel was not emitted"), "{why}");
        let mut plan = emitted_plan();
        assert_eq!(plan.generate_unavailable(), None);
        plan.fused_sample = CfieKernel::refused("top_k 100 is outside 1..=64");
        let why = plan.generate_unavailable().unwrap();
        assert_eq!(why, "the fused sampler kernel was not emitted: top_k 100 is outside 1..=64");
    }

    #[test]
    fn the_report_names_defaulted_sampler_settings() {
        let mut input = nslcoder_input();
        input.sampling_configured = false;
        let rep = run(input).render_report();
        assert!(rep.contains("[2] Sampler: temperature 0.7, top_k 50, top_p 0.9 (DEFAULTS — no sampling: section set them)"), "{rep}");
        let rep = run(nslcoder_input()).render_report();
        assert!(rep.contains("(from the sampling: section)"), "{rep}");
    }

    #[test]
    fn the_kv_precision_line_is_fp16_and_the_ranking_is_labelled() {
        let rep = run(nslcoder_input()).render_report();
        assert!(rep.contains("[5] KV precision: FP16 in every layer"), "{rep}");
        assert!(rep.contains("advisory, NOT applied:"), "{rep}");
        assert!(rep.contains("The ranking is relative"), "{rep}");
        assert!(!rep.contains("memory savings"), "{rep}");
        let mut input = nslcoder_input();
        input.kv_quant_advisory = false;
        let rep = run(input).render_report();
        assert!(!rep.contains("advisory"), "{rep}");
    }

    #[test]
    fn the_speculative_section_states_the_cost_not_a_speedup() {
        let mut plan = emitted_plan();
        plan.draft_block = CfieKernel::emitted("nsl_cfie_decode_block".into(), "D".into(), launch(1, 128));
        plan.draft_sample = CfieKernel::emitted("nsl_cfie_draft_sample".into(), "S".into(), launch(1, 128));
        plan.speculative_draft = Some(CfieSpeculativeDraft {
            weights_path: "draft.safetensors".into(),
            n_layers: 2,
            d_model: 256,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 128,
            d_ff: 704,
            vocab_size: 32_000,
            pool_bytes: 1 << 20,
            k_tokens: 4,
            target_n_layers: 8,
        });
        let rep = plan.render_report();
        // 4 draft steps of 2 + 1 launches, 4 target steps of 8 + 1.
        assert!(rep.contains("up to 48 launches per round for at most 4 tokens, against 9 per token without a draft"), "{rep}");
        assert!(rep.contains("SLOWER than plain decoding"), "{rep}");
        assert!(rep.contains("nsl_cfie_speculative_generate"), "{rep}");
        let regs = kernel_registrations(&plan);
        assert_eq!(regs.iter().map(|r| r.kind).collect::<Vec<_>>(), [2, 6, 7, 1]);
    }

    #[test]
    fn deterministic_across_calls() {
        let r1 = run(nslcoder_input()).render_report();
        let r2 = run(nslcoder_input()).render_report();
        // Strip the trailing "Solve time" line which varies.
        let strip = |s: &str| {
            s.lines()
                .filter(|l| !l.starts_with("Solve time"))
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
    fn kernel_registrations_list_emitted_kernels_only() {
        let plan = run(nslcoder_input());
        assert!(kernel_registrations(&plan).is_empty());
        let regs = kernel_registrations(&emitted_plan());
        assert_eq!(regs.len(), 2);
        assert_eq!((regs[0].kind, regs[0].name.as_str()), (2, "nsl_cfie_decode_block"));
        assert_eq!((regs[1].kind, regs[1].name.as_str()), (1, "nsl_cfie_fused_sample"));
        assert!(regs.iter().all(|r| r.layer_idx == 0 && r.smem_dyn == 0));
        assert_eq!((regs[0].grid_x, regs[0].block_x), (1, 128));
    }

    #[test]
    fn report_without_wiring_does_not_claim_registration() {
        let mut plan = emitted_plan();
        plan.runtime_wiring_emitted = false;
        let rep = plan.render_report();
        assert!(!rep.contains("registered at serve init"), "{rep}");
        assert!(rep.contains("runtime wiring NOT emitted on this serve path (report-only)"), "{rep}");
    }
}
