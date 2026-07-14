use std::collections::HashMap;

use cranelift_module::{DataDescription, DataId, Module};

use nsl_ast::decl::{Decorator, ModelMember};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Stmt, StmtKind};

use super::{Compiler, FlashAttentionCompileContext, SdpaBwdVariant};
use crate::error::CodegenError;

/// M57.1 §3.2: v1-permanent redirect error message for `nsl build --target fpga`
/// invocations on non-model kernels. The test `tests/fpga_target_redirect.rs`
/// pins this exact text — keep them in sync.
pub const FPGA_TARGET_REDIRECT_MSG: &str =
    "`--target fpga` for general kernels is not supported in v1. \
     Use `nsl fpga-compile <source>` for model-block FPGA compilation.";

/// Parse the numeric SM version from a target string like `"sm_90"` → `90`.
///
/// Accepts the generic `"cuda"` alias (used as `CompileOptions::default()`)
/// and maps it to sm_80 (Ampere, the earliest SM that has every feature
/// the current PTX emitters need). Concrete targets like `sm_75` always
/// take precedence when the user passes one explicitly.
///
/// Panics with a clear message on an unrecognised format so that a
/// misconfigured compile target is caught at compile time rather than
/// silently routing to the wrong PTX path.
pub(crate) fn parse_gpu_sm_from_target(target: &str) -> u32 {
    if target == "cuda" {
        return 80;
    }
    target
        .strip_prefix("sm_")
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("invalid compile target: {target}"))
}

/// Gap B: does the top-level statement list contain at least one
/// `@train` block? Drives whether `compile_flash_attention_kernels`
/// synthesizes the extra CSHA-with-saves forward PTX and the fused
/// backward PTX.  Recursively scans `Decorated` wrappers because
/// train blocks often sit inside test/bench decorators.
fn stmts_contain_train_block(stmts: &[Stmt]) -> bool {
    for s in stmts {
        if contains_train_block_stmt(s) {
            return true;
        }
    }
    false
}

fn contains_train_block_stmt(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::TrainBlock(_) => true,
        // CPKD: a distill block is a training loop over the student — the
        // CSHA training-PTX synthesis (with-saves forward + fused backward)
        // must fire for it exactly as for `train`.
        StmtKind::DistillBlock(_) => true,
        StmtKind::Decorated { stmt: inner, .. } => contains_train_block_stmt(inner),
        _ => false,
    }
}

/// Extract the trailing dim of a `Call{ones|zeros|...}([N])` /
/// `Call{ones|...}([N, M])` shape literal so we can read the
/// d_model from a model layer's `init` expression.
///
/// For `w_norm: Tensor = ones([32])` we want `32` (last dim).
/// For `wq:     Tensor = ones([32, 32])` either dim works since it's a
/// square projection (in_features=out_features=d_model). We take the
/// FIRST dim (in_features) — that's the one that must equal d_model
/// for `x @ wq` to typecheck.
fn first_dim_from_init_expr(expr: &nsl_ast::expr::Expr) -> Option<i64> {
    if let ExprKind::Call { args, .. } = &expr.kind {
        let first_arg = args.first()?;
        if let ExprKind::ListLiteral(elems) = &first_arg.value.kind {
            if let Some(first) = elems.first() {
                if let ExprKind::IntLiteral(v) = first.kind {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn last_dim_from_init_expr(expr: &nsl_ast::expr::Expr) -> Option<i64> {
    if let ExprKind::Call { args, .. } = &expr.kind {
        let first_arg = args.first()?;
        if let ExprKind::ListLiteral(elems) = &first_arg.value.kind {
            if let Some(last) = elems.last() {
                if let ExprKind::IntLiteral(v) = last.kind {
                    return Some(v);
                }
            }
        }
    }
    None
}

impl Compiler<'_> {
    // ── Compile kernel definitions (PTX → .rodata, before functions) ──

    pub fn compile_kernels(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::KernelDef(kernel) => {
                    self.compile_single_kernel(kernel)?;
                }
                StmtKind::Decorated {
                    decorators,
                    stmt: inner,
                } => {
                    if let StmtKind::KernelDef(kernel) = &inner.kind {
                        let has_autotune = decorators.iter().any(|d| {
                            d.name.len() == 1
                                && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
                        });

                        if has_autotune {
                            let params = self.extract_autotune_params(decorators)?;
                            let kernel_name = self
                                .interner
                                .resolve(kernel.name.0)
                                .unwrap_or("unknown")
                                .to_string();

                            if self.compile_options.no_autotune {
                                // --no-autotune: use middle values, skip benchmarking
                                eprintln!(
                                    "[nsl] autotune: --no-autotune, using middle values for {}",
                                    kernel_name
                                );
                                let middle = crate::autotune::select_middle_values(&params);
                                let const_map: HashMap<String, i64> = middle.into_iter().collect();
                                self.compile_single_kernel_with_constants(kernel, &const_map)?;
                            } else {
                                // M26: cost-model-based autotune variant selection
                                let winning_constants =
                                    self.autotune_select_best(&kernel_name, kernel, &params)?;
                                self.compile_single_kernel_with_constants(
                                    kernel,
                                    &winning_constants,
                                )?;
                            }
                        } else {
                            self.compile_single_kernel(kernel)?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Compile a single kernel definition: generate target-specific GPU code and embed in .rodata.
    ///
    /// Dispatches based on `--target` CLI flag:
    /// - CUDA (default): direct AST->PTX via KernelCompiler
    /// - ROCm/Metal/WebGPU: AST->KIR->backend lowerer
    fn compile_single_kernel(
        &mut self,
        kernel: &nsl_ast::block::KernelDef,
    ) -> Result<(), CodegenError> {
        use crate::gpu_target::GpuTarget;

        let target = self.gpu_target();
        let kernel_name = self
            .interner
            .resolve(kernel.name.0)
            .unwrap_or("__kernel")
            .to_string();

        let kernel_bytes = match target {
            GpuTarget::Cuda => {
                // Original path: direct AST -> PTX
                crate::kernel::KernelCompiler::compile(kernel, self.interner)?
            }
            GpuTarget::Fpga => {
                // M57.1 §3.2: `nsl build --target fpga` parses successfully (parse_target
                // recognizes "fpga"), but general-kernel FPGA compilation is not in v1's
                // scope. Model-block compilation routes through `nsl fpga-compile`; this
                // arm rejects non-model invocations with a redirecting error.
                return Err(crate::error::CodegenError::new(FPGA_TARGET_REDIRECT_MSG));
            }
            GpuTarget::Rocm | GpuTarget::Metal | GpuTarget::WebGpu => {
                // M47b: AST -> KIR -> backend-specific lowerer
                let kir = crate::kernel_lower::lower_kernel_to_ir(kernel, self.interner, target)?;

                // Validate that the kernel's required features are supported by the target
                let missing = target.features().missing(kir.required_features);
                if !missing.is_empty() {
                    return Err(CodegenError::new(format!(
                        "kernel '{}' requires features not supported by {}: {}",
                        kernel_name,
                        target.name(),
                        missing.names().join(", ")
                    )));
                }

                let code = match target {
                    GpuTarget::Rocm => {
                        eprintln!(
                            "[nsl] Generated AMDGPU ISA for kernel '{}' (runtime execution requires M47c)",
                            kernel_name
                        );
                        crate::backend_amdgpu::lower_kir_to_amdgpu(&kir)
                    }
                    GpuTarget::Metal => {
                        eprintln!(
                            "[nsl] Generated MSL for kernel '{}' (runtime execution requires M47c)",
                            kernel_name
                        );
                        crate::backend_metal::lower_kir_to_msl(&kir)
                    }
                    GpuTarget::WebGpu => {
                        eprintln!(
                            "[nsl] Generated WGSL for kernel '{}' (runtime execution requires M47c)",
                            kernel_name
                        );
                        crate::backend_wgsl::lower_kir_to_wgsl(&kir)
                    }
                    GpuTarget::Cuda | GpuTarget::Fpga => unreachable!(),
                };

                // Null-terminate for consistency with PTX path
                let mut bytes = code;
                if bytes.last() != Some(&0) {
                    bytes.push(0);
                }
                bytes
            }
        };

        // Embed kernel code bytes in .rodata
        let data_label = format!("__nsl_kernel_{}_{}", target.name(), kernel_name);
        let kernel_data_id = self
            .module
            .declare_data(&data_label, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare kernel data for '{}' ({}): {e}",
                    kernel_name,
                    target.name()
                ))
            })?;
        let mut data_desc = cranelift_module::DataDescription::new();
        data_desc.define(kernel_bytes.into_boxed_slice());
        self.module
            .define_data(kernel_data_id, &data_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define kernel data for '{}' ({}): {e}",
                    kernel_name,
                    target.name()
                ))
            })?;

        // Embed kernel name (null-terminated) in .rodata
        let mut name_bytes = kernel_name.as_bytes().to_vec();
        name_bytes.push(0);
        let name_label = format!("__nsl_kernel_name_{}_{}", target.name(), kernel_name);
        let name_data_id = self
            .module
            .declare_data(&name_label, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare name data for kernel '{}': {e}",
                    kernel_name
                ))
            })?;
        let mut name_desc = cranelift_module::DataDescription::new();
        name_desc.define(name_bytes.into_boxed_slice());
        self.module
            .define_data(name_data_id, &name_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define name data for kernel '{}': {e}",
                    kernel_name
                ))
            })?;

        self.kernels
            .kernel_ptx_data
            .insert(kernel_name, (kernel_data_id, name_data_id));
        Ok(())
    }

    /// Extract tuning parameters from @autotune decorator arguments.
    ///
    /// Expected form: `@autotune(block_size=[64, 128, 256], warps=[2, 4, 8])`
    /// Returns a list of (parameter_name, candidate_values) pairs.
    fn extract_autotune_params(
        &self,
        decorators: &[Decorator],
    ) -> Result<crate::autotune::TuningParams, CodegenError> {
        let autotune_deco = decorators
            .iter()
            .find(|d| {
                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
            })
            .ok_or_else(|| CodegenError::new("@autotune decorator not found".to_string()))?;

        let mut params = Vec::new();
        if let Some(ref args) = autotune_deco.args {
            for arg in args {
                let name = arg
                    .name
                    .as_ref()
                    .and_then(|s| self.interner.resolve(s.0))
                    .unwrap_or("unnamed")
                    .to_string();
                let values = match &arg.value.kind {
                    ExprKind::ListLiteral(items) => items
                        .iter()
                        .filter_map(|item| {
                            if let ExprKind::IntLiteral(v) = &item.kind {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    _ => vec![],
                };
                params.push((name, values));
            }
        }
        Ok(params)
    }

    // ── @autotune variant generation + cost-model selection (M26) ──────

    /// Compile a single kernel with constant substitutions applied.
    ///
    /// Delegates to `KernelCompiler::compile_with_constants` for CUDA targets,
    /// or to the normal `compile_single_kernel` path for non-CUDA targets.
    fn compile_single_kernel_with_constants(
        &mut self,
        kernel: &nsl_ast::block::KernelDef,
        constants: &HashMap<String, i64>,
    ) -> Result<(), CodegenError> {
        use crate::gpu_target::GpuTarget;

        let target = self.gpu_target();
        let kernel_name = self
            .interner
            .resolve(kernel.name.0)
            .unwrap_or("__kernel")
            .to_string();

        let kernel_bytes = match target {
            GpuTarget::Cuda => {
                // AST -> constant substitution -> PTX
                crate::kernel::KernelCompiler::compile_with_constants(
                    kernel,
                    self.interner,
                    constants,
                )?
            }
            _ => {
                // Non-CUDA: constant substitution not yet supported for KIR path.
                // Fall through to the normal compile path.
                return self.compile_single_kernel(kernel);
            }
        };

        // Embed kernel code bytes in .rodata (same as compile_single_kernel)
        let data_label = format!("__nsl_kernel_{}_{}", target.name(), kernel_name);
        let kernel_data_id = self
            .module
            .declare_data(&data_label, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare kernel data for '{}' ({}): {e}",
                    kernel_name,
                    target.name()
                ))
            })?;
        let mut data_desc = cranelift_module::DataDescription::new();
        data_desc.define(kernel_bytes.into_boxed_slice());
        self.module
            .define_data(kernel_data_id, &data_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define kernel data for '{}' ({}): {e}",
                    kernel_name,
                    target.name()
                ))
            })?;

        // Embed kernel name (null-terminated)
        let mut name_bytes = kernel_name.as_bytes().to_vec();
        name_bytes.push(0);
        let name_label = format!("__nsl_kernel_name_{}_{}", target.name(), kernel_name);
        let name_data_id = self
            .module
            .declare_data(&name_label, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare name data for kernel '{}': {e}",
                    kernel_name
                ))
            })?;
        let mut name_desc = cranelift_module::DataDescription::new();
        name_desc.define(name_bytes.into_boxed_slice());
        self.module
            .define_data(name_data_id, &name_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define name data for kernel '{}': {e}",
                    kernel_name
                ))
            })?;

        self.kernels
            .kernel_ptx_data
            .insert(kernel_name, (kernel_data_id, name_data_id));
        Ok(())
    }

    /// Run cost-model-based autotune variant selection for a kernel.
    ///
    /// Generates the Cartesian product of tuning parameters, estimates cost for
    /// each variant using the roofline model, and returns the winning constant map.
    /// Integrates with the autotune cache; respects `--autotune-fresh`.
    fn autotune_select_best(
        &self,
        kernel_name: &str,
        kernel: &nsl_ast::block::KernelDef,
        tuning_params: &crate::autotune::TuningParams,
    ) -> Result<HashMap<String, i64>, CodegenError> {
        let fresh = self.compile_options.autotune_fresh;

        // Compute a cache key from the kernel AST + tuning params + target GPU
        let gpu = crate::gpu_specs::default_gpu();
        let ast_bytes = format!("{:?}", kernel.body).into_bytes();
        let cache_hash = crate::autotune::hash_kernel_ast(
            kernel_name,
            &ast_bytes,
            tuning_params,
            &[], // no specific input shapes at compile time
            gpu.name,
            &format!("{}", gpu.sm_version),
            gpu.num_sms,
        );

        // PTX generator closure: compile kernel with substituted constants
        let interner = self.interner;
        let ptx_generator = |variant: &crate::autotune::Variant| -> Result<String, String> {
            let const_map: HashMap<String, i64> = variant.iter().cloned().collect();
            let ptx_bytes =
                crate::kernel::KernelCompiler::compile_with_constants(kernel, interner, &const_map)
                    .map_err(|e| e.to_string())?;
            // Convert to string (PTX is null-terminated UTF-8)
            let ptx_str = String::from_utf8_lossy(&ptx_bytes).to_string();
            Ok(ptx_str)
        };

        // Cost estimator closure: use roofline model as proxy for timing
        let cost_estimator = |variant: &crate::autotune::Variant| -> Result<f64, String> {
            // Heuristic: larger block sizes generally improve memory coalescing but
            // risk lower occupancy. We model this as elementwise cost over a
            // representative workload, scaled by the variant's block parameters.
            //
            // For a more precise model, we'd analyze the kernel AST to extract
            // matmul dimensions, loop bounds, etc. For now, use a simple proxy:
            // assume 1M elements, and the cost is inversely proportional to the
            // product of block parameters (larger blocks = fewer launches = less overhead).
            let representative_elements: u64 = 1_048_576; // 1M elements
            let dtype_bytes: u64 = 4; // fp32

            // Compute the "block factor" — product of all tuning param values.
            // Higher block factors generally mean fewer kernel launches and better
            // memory coalescing, so estimated time decreases.
            let block_factor: f64 = variant.iter().map(|(_, v)| *v as f64).product();

            // Scale elements by inverse of block factor (normalized to 128 baseline)
            let effective_elements =
                (representative_elements as f64 * (128.0 / block_factor.max(1.0))) as u64;

            let (flops, bytes_read, bytes_written) =
                crate::cost_model::elementwise_unary_cost(effective_elements, dtype_bytes);

            let time_us = crate::cost_model::estimate_time_us(
                flops,
                bytes_read,
                bytes_written,
                gpu,
                dtype_bytes as usize,
            );

            Ok(time_us)
        };

        let winner = crate::autotune::find_best_variant_cost_model(
            kernel_name,
            tuning_params,
            &cache_hash,
            fresh,
            &ptx_generator,
            &cost_estimator,
        )
        .map_err(|e| CodegenError::new(format!("autotune failed for '{}': {}", kernel_name, e)))?;

        eprintln!(
            "[nsl] autotune: selected {:?} for kernel '{}'{}",
            winner,
            kernel_name,
            if fresh { " (fresh)" } else { "" },
        );

        Ok(winner.into_iter().collect())
    }

    // ── FlashAttention kernel synthesis ──────────────────────────────

    /// Walk function definitions for `@flash_attention` decorator, synthesize PTX,
    /// embed it in .rodata, and store the `FlashAttentionCompileContext`.
    pub fn compile_flash_attention_kernels(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        // WGGO-before-kernel-synthesis: solve the per-train-block plans NOW,
        // before any kernel decision, so `maybe_synthesize_csha_training_ptx`
        // (per-doc admission, packing fork) can consume them and the train
        // blocks can reuse them ahead of FASE recipe selection. This is the
        // single choke point every entry point funnels through, which is what
        // makes the ordering guarantee structural rather than per-call-site.
        //
        // Calibrated importance exception: a plan solved here needs the
        // calibration sidecar to score with `importance != Magnitude`. The
        // `compile_and_calibrate` wrapper now fires the calibration harness
        // ahead of this call, so the sidecar is present and the pre-pass plans
        // under the same calibrated scorer the in-place planner uses. But any
        // entry point that never runs the harness leaves the sidecar `None`;
        // there `wggo_prepass_deferred_pending_sidecar` returns true and we
        // defer, keeping planning in-place exactly as before this restructure
        // (planning under a degraded scorer would diverge from the in-place
        // planner and its pre-plan would be fingerprint-rejected anyway).
        // Packing honesty signal for the plan's constraint defaults —
        // computed BEFORE the pre-pass deferral gate (it is AST-only and
        // independent of the calibration sidecar): the in-place planning
        // fallback reads it even when the pre-pass defers, so computing it
        // only on the non-deferred arm made a calibration-deferred build
        // plan packing_mode=0 for a genuinely packed module (review
        // finding).
        self.features.packing_supported_in_module =
            crate::pca_activation::detect_packing_for_stmts(stmts, self.interner)
                || crate::wggo_prepass::stmts_contain_masked_sdpa(stmts, self.interner)
                || crate::wggo_prepass::fn_bodies_contain_masked_sdpa(
                    self.models
                        .model_method_bodies
                        .values()
                        .flat_map(|m| m.values()),
                    self.interner,
                );
        // PCA Stage C note: `packed_sdpa_in_module` is deliberately NOT set
        // here. An AST scan over model method bodies would over-claim — the
        // stdlib GQA model always CARRIES forward_packed, so every GQA module
        // would report the fused consumption channel even when only
        // forward_masked is called. The flag is instead derived from the
        // EXTRACTED training graph (actually-reached ops) in
        // `wggo_prepass::build_preplan`, which is also the only context where
        // the verdict that consumes it can fire.
        let deferred =
            crate::wggo_prepass::wggo_prepass_deferred_pending_sidecar(&self.compile_options);
        if deferred {
            // Note only when planning would otherwise happen: `--wggo off`
            // arrives as Some("off"), so `mode.is_some()` is NOT "enabled"
            // (review finding — the off mode must stay chatter-free).
            if crate::wggo_prepass::wggo_mode_enabled(&self.compile_options) {
                eprintln!(
                    "[wggo] pre-pass deferred: calibrated importance scoring requested but \
                     its calibration sidecar is not yet available; planning stays in-place \
                     (kernel admission unplanned)"
                );
            }
        } else {
            self.wggo_preplans = crate::wggo_prepass::run(self, stmts);
        }

        for stmt in stmts {
            let decorators = match &stmt.kind {
                StmtKind::Decorated { decorators, stmt } => {
                    if matches!(&stmt.kind, StmtKind::FnDef(_)) {
                        decorators
                    } else {
                        continue;
                    }
                }
                // Check model layer declarations AND model methods for
                // `@flash_attention`.  Pre-Gap-F, only `ModelMember::LayerDecl`
                // decorators were scanned; decorators on `ModelMember::Method`
                // were silently dropped, so `@flash_attention fn forward(...)`
                // inside a `model` block had no effect and the SDPA call
                // lowered through the naive transpose/matmul/softmax path.
                // DOC-GAP F.1 fix: also descend into `ModelMember::Method`
                // so the natural "decorate the method that does attention"
                // idiom actually reaches `try_build_flash_context`.
                StmtKind::ModelDef(md) => {
                    for member in &md.members {
                        let decorators = match member {
                            ModelMember::LayerDecl { decorators, .. } => decorators,
                            ModelMember::Method(_fn_def, decorators) => decorators,
                        };
                        if let Some(ctx) = self.try_build_flash_context(decorators)? {
                            self.kernels.flash_attention_context = Some(ctx);
                            self.maybe_synthesize_csha_training_ptx(stmts)?;
                            return Ok(());
                        }
                    }
                    continue;
                }
                _ => continue,
            };

            if let Some(ctx) = self.try_build_flash_context(decorators)? {
                self.kernels.flash_attention_context = Some(ctx);
                self.maybe_synthesize_csha_training_ptx(stmts)?;
                return Ok(());
            }
        }
        // Reaching here means the decorator scan built no `@flash_attention`
        // context (every decorated path returns early above), i.e. this is the
        // decorator-free attention path. Surface whether a WGGO plan is being
        // dropped for it.
        self.report_decorator_free_plan_reachability(stmts);
        Ok(())
    }

    /// Resolve the layer's `d_model` (= input feature dim of RMSNorm) by
    /// scanning the top-level statement list for a `model` whose method
    /// carries `@flash_attention`, then inspecting that model's peer
    /// `LayerDecl`s.
    ///
    /// Resolution order:
    ///   1. `w_norm` (RMSNorm gamma) — the literal first/last dim of its
    ///      shape IS d_model.
    ///   2. `wq` (or `q_proj`) — first dim of `[d_model, d_kv_heads]`.
    ///   3. `wk`, `wv`, `q_proj`, `k_proj`, `v_proj` — same.
    ///
    /// Returns `None` when no model is found, when no peer LayerDecl has
    /// a literal-init shape, or when the init expression isn't a simple
    /// `ones([...])` / `zeros([...])` call.  Callers should keep the
    /// previous `d_model: 0` placeholder in that case so non-CSHA paths
    /// stay unchanged (they don't read d_model).
    fn resolve_csha_d_model_from_stmts(&self, stmts: &[Stmt]) -> Option<u32> {
        // Names we'll probe in priority order.
        const NORM_NAMES: &[&str] = &["w_norm", "norm_weight", "gamma"];
        const PROJ_NAMES: &[&str] = &["wq", "wk", "wv", "q_proj", "k_proj", "v_proj"];

        for stmt in stmts {
            let StmtKind::ModelDef(md) = &stmt.kind else {
                continue;
            };
            // Only care about models that actually carry a
            // `@flash_attention` method — otherwise this is an unrelated
            // model (e.g. a quant target) and its layer dims are
            // irrelevant to the CSHA training config.
            let mut has_flash = false;
            for member in &md.members {
                if let ModelMember::Method(_, decos) = member {
                    if decos.iter().any(|d| {
                        d.name.len() == 1
                            && self.interner.resolve(d.name[0].0).unwrap_or("")
                                == "flash_attention"
                    }) {
                        has_flash = true;
                        break;
                    }
                }
            }
            if !has_flash {
                continue;
            }

            // Phase 1 — try `w_norm` (RMSNorm gamma) first. Both first
            // and last dim equal d_model for a 1-D gamma; we use first.
            for member in &md.members {
                if let ModelMember::LayerDecl { name, init: Some(init), .. } = member {
                    let nm = self.interner.resolve(name.0).unwrap_or("");
                    if NORM_NAMES.contains(&nm) {
                        if let Some(v) = first_dim_from_init_expr(init) {
                            if v > 0 {
                                return Some(v as u32);
                            }
                        }
                    }
                }
            }
            // Phase 2 — fall back to a Q/K/V projection.  Convention:
            // `wq` is `[d_model, d_kv]` (input rows, output cols), so the
            // FIRST dim is d_model.
            for member in &md.members {
                if let ModelMember::LayerDecl { name, init: Some(init), .. } = member {
                    let nm = self.interner.resolve(name.0).unwrap_or("");
                    if PROJ_NAMES.contains(&nm) {
                        if let Some(v) = first_dim_from_init_expr(init) {
                            if v > 0 {
                                return Some(v as u32);
                            }
                            // Last-dim fallback in case the user wrote
                            // `[d_kv, d_model]` (transposed convention).
                        }
                        if let Some(v) = last_dim_from_init_expr(init) {
                            if v > 0 {
                                return Some(v as u32);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Gap B: when the module contains a `@train` block AND a flash-attention
    /// context was established, synthesize two extra PTX strings and attach
    /// them to the context:
    ///
    ///   1. Forward PTX with `csha.save_activations_for_backward=true` —
    ///      needed so `emit_save_activations_subset` actually generates the
    ///      HBM save-writes.  The non-CSHA forward PTX has
    ///      `csha=None`, which suppresses save emission at PTX level.
    ///   2. Fused-backward PTX via `synthesize_backward(config)` — Gap C/D
    ///      reads the resulting DataIds off `CshaSavePointers` to launch
    ///      `nsl_flash_attention_csha_backward`.
    ///
    /// Both run against a CSHA-tweaked clone of the forward config.  We
    /// use the minimum-invasive CSHA preset: `level=1`, no projection
    /// fusion, no output-proj fusion, `save_activations_for_backward=true`.
    /// That keeps the forward kernel's SMEM budget identical to the
    /// inference baseline while wiring up the Tier C save codepaths.
    ///
    /// On any synthesis failure (e.g. backward SMEM validator rejects the
    /// config) we log a diagnostic and leave the fields `None` — the
    /// forward inference PTX still works, and the Gap A save-pointer
    /// allocation continues to run (just into a kernel that ignores
    /// them).  Non-regression is preserved.
    /// Surface the WGGO plan's (non-)reachability on the decorator-free
    /// attention training path.
    ///
    /// `maybe_synthesize_csha_training_ptx` — home of the @pca/WGGO packing
    /// admission fork — only runs when a `@flash_attention` context was built.
    /// The stdlib GQA models real pretraining uses call the plain
    /// `scaled_dot_product_attention` builtin with no decorator, so that
    /// synthesis (and the admission fork that consumes the plan's per-layer
    /// packing decisions) never fires and the attention op lowers through the
    /// decomposed/plain-flash path instead. WGGO still SOLVES a plan for the
    /// train block and the `--wggo` report renders it — which is misleading,
    /// because on this path only the per-layer FASE overrides actually reach
    /// codegen. Emit a one-time verdict so the plan's non-application is
    /// visible rather than silent.
    ///
    /// Behaviour-neutral: this only prints. It deliberately does NOT build a
    /// `flash_attention_context` for the plain op — that field gates the
    /// forward FFI dispatch and (via its optional backward DataIds) the
    /// decorator-free backward's per-head_dim variant-table path (PR #347), so
    /// populating it here would change dispatch, not just diagnostics. Wiring
    /// the packing decision into the plain SDPA dispatch is a later stage; see
    /// the WGGO-pretraining-orchestration design.
    fn report_decorator_free_plan_reachability(&self, stmts: &[Stmt]) {
        // Chatter-free unless WGGO is actually enabled (`--wggo off` arrives as
        // Some("off"), so `mode.is_some()` is not "enabled").
        if !crate::wggo_prepass::wggo_mode_enabled(&self.compile_options) {
            return;
        }
        // A decorator built a context => the admission fork ran normally; nothing
        // is being dropped, so there is nothing to report.
        if self.kernels.flash_attention_context.is_some() {
            return;
        }
        if !stmts_contain_train_block(stmts) {
            return;
        }
        // Only report when a plan was actually solved for the governing (first)
        // train block: if extraction failed there is no plan being dropped.
        let Some(pre) = self
            .wggo_preplans
            .iter()
            .find(|p| p.is_first_train_block)
        else {
            return;
        };
        // A plan is solved for ANY train block (an MLP's matmuls yield one too),
        // but this verdict is specifically about attention-only CSHA/PCA
        // decisions — so it must fire ONLY when the training graph actually
        // computes attention. `contains_attention` is derived from the
        // extracted (inlined) graph, so it holds for real GQA models whose SDPA
        // lives in an imported stdlib layer, not just direct SDPA callers.
        if !pre.contains_attention {
            return;
        }
        let packing_detected =
            crate::pca_activation::detect_packing_for_stmts(stmts, self.interner);
        let plan_prefers_segment_id =
            crate::wggo_prepass::plan_prefers_segment_id(&pre.overrides);
        // PCA Stage C changed what this verdict must claim: the decorator-free
        // forward now carries a fused dispatch (plain flash forward with LSE
        // saves; segment-masked family for the packed builtin), so packing IS
        // applied on this path when the module uses
        // `scaled_dot_product_attention_packed`. CSHA fusion levels remain
        // decorator-gated (unchanged).
        if self.features.packed_sdpa_in_module {
            eprintln!(
                "[wggo] plan-reachability: decorator-free attention with the PACKED \
                 builtin — the plan's segment_id packing decision lowers through the \
                 Stage-C fused segment-masked kernel family; CSHA fusion levels remain \
                 decorator-gated (layers_planned={}, packing_detected={packing_detected}, \
                 plan_prefers_segment_id={plan_prefers_segment_id}).",
                pre.overrides.per_layer.len(),
            );
        } else {
            eprintln!(
                "[wggo] plan-reachability: this train block lowers attention through the \
                 decorator-free `scaled_dot_product_attention` path (no @flash_attention \
                 context) — the fused plain forward applies, but the plan's per-layer \
                 CSHA/PCA packing decisions are NOT lowered here (packing needs the \
                 packed builtin, e.g. GQA forward_packed; CSHA needs the decorator) \
                 (layers_planned={}, packing_detected={packing_detected}, \
                 plan_prefers_segment_id={plan_prefers_segment_id}); per-layer FASE \
                 overrides reach codegen on this path.",
                pre.overrides.per_layer.len(),
            );
        }
    }

    fn maybe_synthesize_csha_training_ptx(
        &mut self,
        stmts: &[Stmt],
    ) -> Result<(), CodegenError> {
        if !stmts_contain_train_block(stmts) {
            return Ok(());
        }
        let base_config = match self.kernels.flash_attention_context.as_ref() {
            Some(ctx) => ctx.config.clone(),
            None => return Ok(()),
        };

        // PCA Tier A activation (spec 2026-05-25): emit segment-masked + RoPE-reset
        // training kernels when the train block's dataset is packed. RoPE-reset rides
        // on the existing `segment_masked && rope_q` gate (forward/prelude.rs:113); no
        // separate flag. Inference (`base_config`) stays unmasked → byte-identical.
        let segment_masked = crate::pca_activation::detect_packing_for_stmts(stmts, self.interner);

        // Resolve `d_model` from the layer's RMSNorm gamma / Q/K/V
        // projection shapes.  Without this the AD emitter allocates
        // `dx_dev`/`dxn_dev` as `[batch, seq, 0]` and the first
        // downstream elementwise op hard-faults with
        //   `tensor shape mismatch in elementwise op (dim 3: 0 vs <hd>)`.
        // Falling back to head_dim keeps the older single-head smokes
        // (where d_model == head_dim) working when no recognisable
        // `w_norm`/`wq`/etc. peer LayerDecl is present.
        let resolved_d_model = self
            .resolve_csha_d_model_from_stmts(stmts)
            .unwrap_or(base_config.head_dim as u32);
        // Cycle-12 T1 wire-up: derive @checkpoint(policy=Full) presence
        // from `compile_options.checkpoint_policies`. v1 ships a single
        // policy variant (Full) per cycle-10 Refuter 3; we use ANY entry
        // in the map as the trigger for routing the backward through
        // `synthesize_backward_with_recompute`. The map being non-empty
        // is the in-tree signal that the user wrote `@checkpoint` on a
        // training-loop function.
        //
        // The fn-name-keyed lookup (which fn carries @checkpoint) is
        // deferred to a follow-on cycle where the compile_flash_attention
        // pipeline gains per-fn context — today the CSHA training PTX
        // emission is module-scoped (one training_config per module
        // owning the @train block).
        let checkpoint_full_active = self
            .compile_options
            .checkpoint_policies
            .values()
            .any(|p| matches!(p, nsl_semantic::effects::CheckpointPolicy::Full));
        let training_checkpoint = if checkpoint_full_active {
            Some(crate::flash_attention::CheckpointExtras::full())
        } else {
            None
        };

        // Cycle-12 T4: when `@checkpoint(policy=Full)` is active, the
        // backward kv-recompute path requires the forward to stage
        // x_raw + Wk/Wv. The cycle-12 R3 augmentation refuses configs
        // that don't satisfy `fused_projections=true`, so we MUST flip
        // the preset to the fused-projections preset when checkpoint is
        // Some. Otherwise (the pre-cycle-12 default), keep the cycle-11
        // minimum-invasive preset with `save_activations_for_backward`
        // alone.
        let csha_extras = if checkpoint_full_active {
            let mut x = crate::flash_attention::CshaExtras::level1_with_fused_proj(1e-5);
            x.d_model = resolved_d_model;
            // T4 (defense-in-depth): `level1_with_fused_proj` already
            // sets save_activations_for_backward=true, but reassert here
            // so a future builder refactor doesn't silently break the
            // x_raw save channel (the cycle-11 silent-garbage bug).
            x.save_activations_for_backward = true;
            x
        } else {
            crate::flash_attention::CshaExtras {
                level: 1,
                fused_rmsnorm: false,
                fused_projections: false,
                fused_output_proj: false,
                active_heads: 0,
                rmsnorm_eps: 1e-5,
                d_model: resolved_d_model,
                save_activations_for_backward: true,
                skip_rmsnorm_prologue: false,
                static_seq_len: None,
            }
        };

        // WGGO plan consultation (Path A). The training-PTX synthesis is
        // module-scoped while the plan is per-layer, so the consultation is
        // conservative: the plan's requested_csha_level is READ and answered
        // honestly rather than blindly applied.
        //   - Level1 requests align with the synthesis preset — confirmed
        //     with a one-line note (previously the plan was ignored outright
        //     and the hardcoded preset happened to coincide).
        //   - Level2/Level3 requests are REFUSED with a diagnostic: on
        //     sm>=80 they divert the with-saves forward to the Tier B.1
        //     pipelined emitter, which emits NO save_activations — honoring
        //     the plan would silently drop the backward's saved activations
        //     (the cycle-11 silent-garbage failure mode). Lift only when
        //     Tier B.1 gains an activation-save channel.
        //   - Reduced active_heads stays unapplied on this path regardless
        //     of the plan: the fused backward's dRMSNorm dx_norm writes are
        //     non-atomic and assume heads=1 (csha_hooks_backward), and the
        //     plan-level [cep] advisory already reports pruned heads as
        //     CSHA-dispatch-only. CshaExtras.active_heads=0 (= all) is the
        //     only safe training value today.
        if let Some(pre) = self.wggo_preplans.iter().find(|p| p.is_first_train_block) {
            use crate::cfie_persistent::FusionLevel;
            let mut over_level1 = 0usize;
            let mut level1 = 0usize;
            let mut level0 = 0usize;
            for l in &pre.overrides.per_layer {
                match l.requested_csha_level {
                    Some(FusionLevel::Level2) | Some(FusionLevel::Level3) => over_level1 += 1,
                    Some(FusionLevel::Level1) => level1 += 1,
                    Some(FusionLevel::None) => level0 += 1,
                    None => {}
                }
            }
            if over_level1 > 0 {
                eprintln!(
                    "[csha] wggo-override-rejected: {over_level1} layer(s) requested \
                     csha level>=2 for the training path; the Tier B.1 forward emits \
                     no activation saves, so honoring it would corrupt the backward — \
                     clamped to level 1 (with-saves synthesis)"
                );
            }
            if level1 > 0 {
                eprintln!(
                    "[csha] plan-aligned: {level1} layer(s) requested csha level 1; \
                     training-PTX synthesis honors it (with-saves forward + fused backward)"
                );
            }
            if level0 > 0 {
                eprintln!(
                    "[csha] wggo-override-rejected: {level0} layer(s) requested csha \
                     level 0 (no fusion), but training-PTX synthesis is gated by the \
                     @flash_attention/@csha decorators, not the plan — level-1 \
                     with-saves kernels are still synthesized (plan cannot veto a \
                     user decorator; remove the decorator to disable CSHA)"
                );
            }
        }
        // Tier C's backward emitter (ds_compute/dqdk_accum/dv_accum)
        // currently hard-asserts `block_kv=32` (T3.3–T3.5 landed with
        // that single tile width; T3.6+ is planned to generalise).
        // The forward inference config inherits block_kv from the
        // user's `@flash_attention` / `@autotune` args — which often
        // set it to 64.  We only use `training_config` to synthesize
        // the with-saves forward PTX and the fused-backward PTX; the
        // primary forward PTX was already embedded from `base_config`.
        // So: clamp block_kv to 32 here, independent of the user's
        // forward tile width, so the backward validator + emitter
        // both accept the config.  The saved activations use block_kv
        // only as a stride, and 32 is a valid stride for any forward
        // block_kv (the runtime dispatcher reads sequence length from
        // the tensor shape, not the PTX-baked block_kv).
        let backward_block_kv: i64 = 32;
        // Backward block_q clamp (fix for 4KB-HtoD ILLEGAL_ADDRESS):
        //
        // The Tier C backward PTX is tiled for `block_q` Q rows per CTA
        // and its phases (q_load, ds_compute, dv_accum, finalize's dq
        // writeback, csha_hooks_backward's dxn writeback + dx Phase 2)
        // address HBM buffers with indices up to `block_q * head_dim`
        // / `block_q * d_model`. When `block_q > seq_len` (the common
        // case for short toy programs — e2e smoke runs seq_len=32 with
        // the default block_q=64 from the forward inference config),
        // the backward HBM writes overflow the gradient output buffers
        // (which are allocated as `[batch, seq, ...]`), tripping
        // CUDA_ERROR_ILLEGAL_ADDRESS inside the kernel.
        //
        // Clamping to 32 matches the block_kv clamp above and the
        // T3.3–T3.5 smoke scope; shorter sequences fit in a single CTA
        // and longer sequences launch `ceil(seq_len/32)` CTAs per
        // batch*head (via the runtime `grid_x = (seq_len + block_q -
        // 1) / block_q` calculation in `nsl_flash_attention_csha_*`).
        // The forward inference PTX still uses the user's
        // (auto)tuned block_q; only the training / fused-backward PTX
        // is tiled for 32 rows.
        //
        // NOTE: this does not fix bounds checking in the kernel — any
        // config where the runtime seq_len is not a multiple of 32
        // would still over-read/write the tail tile. For seq_len that
        // IS a multiple of 32 (the e2e smoke's 32 qualifies), every
        // warp covers exactly one valid row and the HBM indices land
        // in-buffer. Full seq_len bounds-guards in the kernel phases
        // are a follow-up (documented in the Tier C close-out memory).
        let backward_block_q: i64 = 32;
        let training_config = crate::flash_attention::FlashAttentionConfig {
            csha: Some(csha_extras),
            // Cycle-12 T1: route @checkpoint(policy=Full) into the
            // backward synthesizer. None preserves cycle-11 baseline
            // (kv_load HBM path); Some routes the dispatch fork into
            // `synthesize_backward_with_recompute` -> cycle-11
            // `emit_kv_recompute` substitution.
            checkpoint: training_checkpoint,
            block_q: backward_block_q,
            block_kv: backward_block_kv,
            segment_masked,
            ..base_config
        };

        // ── CFTP §4.3 G2 Strategy 3 (Item 4): per-doc CTA admission ─
        // If the user wrote `@pca(strategy=per_document)` AND the
        // train block's dataset is packed AND `pca_per_doc::admit`
        // accepts the (detection, fa_config, packing_cfg) triplet,
        // we route the with-saves forward and the fused backward to
        // the per-document CTA PTX synthesisers in place of the
        // segment-masked Tier A path. Admission failure (e.g. CSHA
        // fused_projections=true, max_doc_len > block_q, num_docs >
        // 256) falls through to the standard Tier A synthesis below.
        //
        // The `_per_doc_cta` kernel-name suffix is the ABI signal the
        // FFI dispatcher (`nsl_flash_attention_csha_with_saves` /
        // `nsl_flash_attention_csha_backward`) already detects to walk
        // `doc_starts` and override `grid_x = num_docs`.
        // WGGO-before-kernels: the pre-pass makes the plan visible HERE, at
        // kernel-synthesis time, so a packing preference can finally flip an
        // admission decision instead of only being validated after the fact
        // (PR #334's deliberate deferral). Semantics: @pca(per_document)
        // AUTHORIZES the per-doc CTA lowering; when the first train block's
        // plan unambiguously prefers segment_id packing (every layer with a
        // nonzero packing_mode chose mode 1), honor that preference and keep
        // the segment-masked Tier-B pair — the post-hoc [pca] verdict layer
        // then reports wggo-override-consumed against the kernels that were
        // actually built. Mixed, zero, or absent preferences change nothing:
        // masking itself is a correctness requirement for packed data and is
        // never plan-negotiable.
        // Consult the FIRST train block's plan specifically (not merely the
        // first block that HAPPENED to plan): kernel synthesis serves the
        // whole module, and when the first block's extraction failed there
        // is no plan entitled to govern it — a later block's preference must
        // not flip admission on the wrong evidence (review finding).
        let plan_prefers_segment_id = self
            .wggo_preplans
            .iter()
            .find(|p| p.is_first_train_block)
            .is_some_and(|pre| crate::wggo_prepass::plan_prefers_segment_id(&pre.overrides));
        let per_doc_authorized = self
            .pca_user_strategies
            .iter()
            .any(|s| s.is_per_document())
            && segment_masked
            && !training_config
                .csha
                .as_ref()
                .is_some_and(|c| c.fused_projections);
        let per_doc_plan = if per_doc_authorized && plan_prefers_segment_id {
            eprintln!(
                "[pca-per-doc] deferred to plan preference: wggo packing_mode=segment_id \
                 prefers the segment-masked Tier-B kernels — skipping per-doc CTA admission"
            );
            None
        } else if per_doc_authorized {
            if let Some(packing_cfg) =
                crate::pca_activation::resolve_packing_config_for_stmts(stmts, self.interner)
            {
                let detection = crate::pca_detect::detect(
                    &packing_cfg,
                    &crate::pca_detect::PcaDetectConfig::default(),
                    2,
                );
                // The user wrote `@pca(strategy=per_document)` explicitly —
                // don't let the statistics-only auto-detect heuristic
                // silently downgrade that opt-in. See
                // `apply_user_strategy_override`'s doc comment.
                let detection =
                    crate::pca_detect::apply_user_strategy_override(detection, true);
                let admit_cfg = crate::pca_per_doc::PerDocAdmitConfig {
                    enable_per_doc_cta: true,
                    ..crate::pca_per_doc::PerDocAdmitConfig::default()
                };
                match crate::pca_per_doc::admit(
                    &detection,
                    &training_config,
                    &packing_cfg,
                    &admit_cfg,
                ) {
                    Ok(plan) => {
                        eprintln!(
                            "[pca-per-doc] admitted: {}",
                            plan.reason
                        );
                        Some(plan)
                    }
                    Err(reason) => {
                        eprintln!(
                            "warning: @pca(strategy=per_document) was requested but could not \
                             be honored ({reason:?}) — falling through to the segment_id_masked \
                             Tier A kernel"
                        );
                        None
                    }
                }
            } else {
                eprintln!(
                    "warning: @pca(strategy=per_document) was requested but the train block's \
                     dataset packing config could not be resolved — falling through to the \
                     segment_id_masked Tier A kernel"
                );
                None
            }
        } else {
            None
        };

        // ── Forward with saves ─────────────────────────────
        // PCA Tier B Planner (planner spec §5): use the emission helper so the
        // call site picks up the Tier-B-on variant when the training config has
        // `segment_masked=true`. For non-`segment_masked` configs the helper
        // returns `None` for the Tier-B-on PTX, preserving the previous
        // single-PTX behaviour byte-identically (base PTX is
        // `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`, which
        // is exactly what the selector previously routed to).
        //
        // Selector pre-flight: the selector's `select_emitter_with_diag` runs
        // its SMEM budget check; preserve that side effect by invoking the
        // existing kernel-name selector once (which delegates to v2 + retains
        // the diag-emit contract for API stability).
        let mut diags = Vec::<String>::new();
        let _ =
            crate::flash_attention_selector::flash_attention_kernel_name_selected_with_diag(
                &training_config, &mut diags,
            );
        for d in diags { eprintln!("warning: {d}"); }
        // Item 4: when per-doc CTA admission succeeds, route the
        // with-saves forward PTX to the per-document CTA emitter and
        // SKIP Tier-B-on variant emission (per-doc is its own dispatch
        // tier — see runtime `csha_is_per_doc_cta_kernel`). When
        // admission fails or the user did not request `per_document`,
        // emit the standard Tier A / Tier B-on variants below.
        let (fwd_kernel_name, fwd_ptx_id, fwd_name_id, fwd_tier_b_ptx_id, fwd_tier_b_name_id) =
            if let Some(plan) = per_doc_plan.as_ref() {
                let pd_name = crate::flash_attention_v2::per_doc_cta::per_doc_cta_kernel_name(
                    &plan.fa_config,
                );
                let pd_ptx_bytes =
                    crate::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward(
                        plan,
                        &training_config,
                    );
                let pd_ptx_id = self.embed_raw_data(
                    &format!("__nsl_flash_ptx_csha_saves_{}", pd_name),
                    pd_ptx_bytes,
                )?;
                let mut pd_name_bytes = pd_name.as_bytes().to_vec();
                pd_name_bytes.push(0);
                let pd_name_id = self.embed_raw_data(
                    &format!("__nsl_flash_name_csha_saves_{}", pd_name),
                    pd_name_bytes,
                )?;
                (pd_name, pd_ptx_id, pd_name_id, None, None)
            } else {
                let fwd_emission =
                    crate::pca_tier_b::emit_tier_b_variants_for_config(&training_config);
                let fwd_kernel_name = fwd_emission.base_kernel_name.clone();

                // Multi-tile twin: embed the combined module (fused kernel +
                // `_mt_attn` interleaved twin) instead of the bare fused PTX
                // so the runtime's two-launch dispatch can serve
                // seq_len > block_q at run time (the fused orchestration is
                // single-tile by construction). For configs the twin cannot
                // serve (segment_masked / sinks / checkpoint / paged /
                // non-fused) `synthesize_forward_multi_tile_combined` returns
                // the fused module unchanged, so this is byte-identical to
                // the previous `fwd_emission.base_ptx` embedding there.
                let base_ptx_combined =
                    crate::flash_attention_v2::synthesize_forward_multi_tile_combined(
                        &training_config,
                    );
                let _ = fwd_emission.base_ptx;
                let fwd_ptx_id = self.embed_raw_data(
                    &format!("__nsl_flash_ptx_csha_saves_{}", fwd_kernel_name),
                    base_ptx_combined,
                )?;
                let mut fwd_name_bytes = fwd_kernel_name.as_bytes().to_vec();
                fwd_name_bytes.push(0);
                let fwd_name_id = self.embed_raw_data(
                    &format!("__nsl_flash_name_csha_saves_{}", fwd_kernel_name),
                    fwd_name_bytes,
                )?;

                // Tier-B-on variant: embed only when the emission helper produced one.
                let (fwd_tier_b_ptx_id, fwd_tier_b_name_id) = match (
                    fwd_emission.tier_b_on_ptx,
                    fwd_emission.tier_b_on_kernel_name,
                ) {
                    (Some(on_ptx), Some(on_name)) => {
                        let on_ptx_id = self.embed_raw_data(
                            &format!("__nsl_flash_ptx_csha_saves_{}", on_name),
                            on_ptx,
                        )?;
                        let mut on_name_bytes = on_name.as_bytes().to_vec();
                        on_name_bytes.push(0);
                        let on_name_id = self.embed_raw_data(
                            &format!("__nsl_flash_name_csha_saves_{}", on_name),
                            on_name_bytes,
                        )?;
                        (Some(on_ptx_id), Some(on_name_id))
                    }
                    _ => (None, None),
                };
                (
                    fwd_kernel_name,
                    fwd_ptx_id,
                    fwd_name_id,
                    fwd_tier_b_ptx_id,
                    fwd_tier_b_name_id,
                )
            };

        // ── Fused backward ─────────────────────────────────
        // The Tier C fused-backward validator has a tighter SMEM budget
        // (adds dQ/dK/dV tiles).  If it rejects the training config we
        // leave the backward IDs None — Gap C/D will detect this and
        // fall through to the legacy tape-op backward path.
        //
        // Item 4: when per-doc CTA admission succeeded for the forward,
        // route the backward through `synthesize_per_doc_cta_backward`
        // as well. The runtime detects the `_per_doc_cta` suffix on the
        // backward name buffer and launches with `grid_x = num_docs`
        // (see `flash_attention.rs:1543`). The Tier-B-on backward
        // variant is skipped (per-doc is its own tier).
        //
        // Sprint 1 T1.4: outside the per-doc branch, `synthesize_backward_combined`
        // returns a single PTX module containing BOTH the scalar v2 backward entry
        // AND the four Tier B.2 hybrid entries (`tier_b2_d_prepass`,
        // `tier_b2_dq_kernel`, `tier_b2_dkdv_kernel`,
        // `tier_b2_proj_backward`) when the config is hybrid-eligible.
        // The runtime branches on `tier_b2_active` (computed at Wengert
        // lowering from the runtime seq_len) to choose which path to
        // launch from this one loaded module. For ineligible configs
        // the output is byte-identical to plain `synthesize_backward`.
        let (bwd_ptx_id, bwd_name_id, bwd_tier_b_ptx_id, bwd_tier_b_name_id) =
            if let Some(plan) = per_doc_plan.as_ref() {
                match crate::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_backward(
                    plan,
                    &training_config,
                ) {
                    Ok(pd_bwd_ptx) => {
                        let pd_bwd_name =
                            crate::flash_attention_v2::per_doc_cta::per_doc_cta_backward_kernel_name(
                                &plan.fa_config,
                            );
                        let pd_bwd_ptx_id = self.embed_raw_data(
                            &format!("__nsl_flash_ptx_csha_bwd_{}", pd_bwd_name),
                            pd_bwd_ptx,
                        )?;
                        let mut pd_bwd_name_bytes = pd_bwd_name.as_bytes().to_vec();
                        pd_bwd_name_bytes.push(0);
                        let pd_bwd_name_id = self.embed_raw_data(
                            &format!("__nsl_flash_name_csha_bwd_{}", pd_bwd_name),
                            pd_bwd_name_bytes,
                        )?;
                        (Some(pd_bwd_ptx_id), Some(pd_bwd_name_id), None, None)
                    }
                    Err(e) => {
                        eprintln!(
                            "[pca-per-doc] backward PTX synthesis failed — falling back to no \
                             backward: {e}"
                        );
                        (None, None, None, None)
                    }
                }
            } else {
            match crate::flash_attention_v2::synthesize_backward_combined(&training_config) {
                Ok(bwd_ptx_string) => {
                    // IMPORTANT: the backward PTX's `.visible .entry` is
                    // generated by `phases::backward::prelude::kernel_name`,
                    // which produces `flash_attn_backward_<rest>` — NOT
                    // `{fwd_kernel_name}_bwd`. Using the wrong name here
                    // makes `cuModuleGetFunction` return
                    // `CUDA_ERROR_NOT_FOUND` at launch time, the kernel
                    // never runs, and every gradient tensor stays at its
                    // zero-initialised value — which is exactly the
                    // "all-zero deltas after SGD" symptom this file was
                    // exposing.
                    let bwd_kernel_name =
                        crate::flash_attention_v2::phases::backward::prelude::kernel_name(
                            &training_config,
                        );
                    let bwd_ptx_id = self.embed_raw_data(
                        &format!("__nsl_flash_ptx_csha_bwd_{}", fwd_kernel_name),
                        bwd_ptx_string.into_bytes(),
                    )?;
                    let mut name_bytes = bwd_kernel_name.as_bytes().to_vec();
                    name_bytes.push(0);
                    let bwd_name_id = self.embed_raw_data(
                        &format!("__nsl_flash_name_csha_bwd_{}", fwd_kernel_name),
                        name_bytes,
                    )?;

                    // PCA Tier B Planner: emit Tier-B-on backward variant when
                    // `segment_masked=true` (planner spec §5.4). The Tier B
                    // arg uses the M2/M6-measured `SegmentResidency::Shared`
                    // and the conservative-max `TIER_B_MAX_BAKED_SEQ_LEN`.
                    let (tier_b_ptx_id_opt, tier_b_name_id_opt) =
                        if crate::pca_tier_b::should_emit_tier_b_at_codegen(&training_config) {
                            let tier_b_args = Some((
                                crate::pca_tier_b::TIER_B_MAX_BAKED_SEQ_LEN,
                                crate::pca_segment::SegmentResidency::Shared,
                            ));
                            match crate::flash_attention_v2::synthesize_backward_with_tier_b(
                                &training_config, tier_b_args,
                            ) {
                                Ok(on_ptx) => {
                                    let on_name = format!(
                                        "{}_tier_b_max{}",
                                        bwd_kernel_name,
                                        crate::pca_tier_b::TIER_B_MAX_BAKED_SEQ_LEN,
                                    );
                                    let on_ptx_id = self.embed_raw_data(
                                        &format!("__nsl_flash_ptx_csha_bwd_{}", on_name),
                                        on_ptx.into_bytes(),
                                    )?;
                                    let mut on_name_bytes = on_name.as_bytes().to_vec();
                                    on_name_bytes.push(0);
                                    let on_name_id = self.embed_raw_data(
                                        &format!("__nsl_flash_name_csha_bwd_{}", on_name),
                                        on_name_bytes,
                                    )?;
                                    (Some(on_ptx_id), Some(on_name_id))
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[csha-gap-b] Tier-B-on backward PTX synthesis skipped — \
                                         validator rejected: {e}"
                                    );
                                    (None, None)
                                }
                            }
                        } else {
                            (None, None)
                        };

                    (Some(bwd_ptx_id), Some(bwd_name_id), tier_b_ptx_id_opt, tier_b_name_id_opt)
                }
                Err(e) => {
                    eprintln!(
                        "[csha-gap-b] backward PTX synthesis skipped — validator rejected: {e}"
                    );
                    (None, None, None, None)
                }
            }
            };

        if let Some(ctx) = self.kernels.flash_attention_context.as_mut() {
            ctx.csha_forward_with_saves_ptx_id = Some(fwd_ptx_id);
            ctx.csha_forward_with_saves_name_id = Some(fwd_name_id);
            ctx.csha_backward_ptx_data_id = bwd_ptx_id;
            ctx.csha_backward_name_data_id = bwd_name_id;
            ctx.csha_training_config = Some(training_config);
            // PCA Tier B Planner: Tier-B-on variant data IDs (None for non-
            // segment_masked configs).
            ctx.csha_with_saves_tier_b_on_ptx_id = fwd_tier_b_ptx_id;
            ctx.csha_with_saves_tier_b_on_name_id = fwd_tier_b_name_id;
            ctx.csha_backward_tier_b_on_ptx_id = bwd_tier_b_ptx_id;
            ctx.csha_backward_tier_b_on_name_id = bwd_tier_b_name_id;
        }
        Ok(())
    }

    /// Try to build a `FlashAttentionCompileContext` from a set of decorators.
    /// Returns `None` if `@flash_attention` is not present.
    fn try_build_flash_context(
        &mut self,
        decorators: &[Decorator],
    ) -> Result<Option<FlashAttentionCompileContext>, CodegenError> {
        let has_flash = decorators.iter().any(|d| {
            d.name.len() == 1
                && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
        });
        if !has_flash {
            return Ok(None);
        }

        // Extract variant flags from sibling decorators
        let mut causal = true; // default
        let mut paged = false;
        let mut paged_block_size: i64 = 16; // default @paged_kv block_size
        let mut rope_q = false;
        let mut rope_style = crate::flash_attention::RopeStyle::HalfSplit;
        let mut gqa_group_size: u32 = 1;
        // Paper §4 tree-mask: bare `@tree_mask` decorator flips `config.tree_mask`,
        // which in turn (a) gates the PTX-side DFS-enter/DFS-exit ancestor check
        // shipped in M33 and (b) appends a variant tag to the kernel name so the
        // runtime dispatch picks the tree_mask kernel. v1 is a bare decorator;
        // any args are rejected (mirrors `@paged_kv` block_size validation).
        let mut tree_mask = false;
        // Paper §4.3: attention sinks. Number of always-attended initial
        // tokens for streaming attention with rolling KV cache. v0 API
        // surface (Sprint 2 cycle-4): the decorator + config field are
        // wired end-to-end, but the SMEM-layout codegen that would
        // materialize the sink cache is DEFERRED to a future sprint.
        // `0` is the sentinel for "sinks disabled".
        //
        // Sprint 2 cycle-5: the only previous assignment site was replaced
        // with an unconditional refusal in the `attention_sink` arm below.
        // Sprint 1b cycle-7 LIFTS that refusal for the narrow Tier A
        // forward single-tile config — the decorator-extraction arm now
        // assigns `num_sink_tokens = N` for any `tokens=N` literal, and
        // each FlashAttentionConfig construction site below consults
        // `attention_sinks_v1_eligible` to refuse non-narrow configs with
        // an axis-specific blocking reason + future Sprint citation
        // (cycle-5 `feedback_deferral_must_refuse` invariant).
        let mut num_sink_tokens: u32 = 0;
        // DOC-GAP F.2: optional `head_dim` argument on `@flash_attention`.
        // Default 64 matches historical behaviour; set explicitly via
        // `@flash_attention(head_dim=32)` to pick a config that fits the
        // Tier C fused-backward SMEM budget.  Validated against
        // `ALLOWED_HEAD_DIM` before being threaded through.  `None` means
        // "use the historical default of 64" — keeps existing code paths
        // byte-identical when the arg is omitted.
        let mut head_dim_override: Option<i64> = None;

        for deco in decorators {
            if deco.name.len() != 1 {
                continue;
            }
            let dname = self.interner.resolve(deco.name[0].0).unwrap_or("");
            match dname {
                "flash_attention" => {
                    // Extract causal= / head_dim= args if present
                    if let Some(ref args) = deco.args {
                        for arg in args {
                            let aname = arg
                                .name
                                .as_ref()
                                .and_then(|s| self.interner.resolve(s.0))
                                .unwrap_or("");
                            if aname == "causal" {
                                if let ExprKind::BoolLiteral(b) = arg.value.kind {
                                    causal = b;
                                }
                            } else if aname == "head_dim" {
                                if let ExprKind::IntLiteral(v) = arg.value.kind {
                                    // Validate against the v2 emitter's
                                    // allowed set. Everything downstream
                                    // (SMEM validator, kernel-name encoder,
                                    // backward synthesizer) assumes this.
                                    if !crate::flash_attention_v2::smem_layout::ALLOWED_HEAD_DIM
                                        .contains(&v)
                                    {
                                        return Err(CodegenError::new(format!(
                                            "@flash_attention(head_dim={}) is not allowed — must be one of {:?}",
                                            v,
                                            crate::flash_attention_v2::smem_layout::ALLOWED_HEAD_DIM,
                                        )));
                                    }
                                    head_dim_override = Some(v);
                                }
                            }
                        }
                    }
                }
                "paged_kv" => {
                    paged = true;
                    // Extract block_size from @paged_kv args (default 16)
                    if let Some(ref args) = deco.args {
                        for arg in args {
                            let aname = arg
                                .name
                                .as_ref()
                                .and_then(|s| self.interner.resolve(s.0))
                                .unwrap_or("");
                            if aname == "block_size" {
                                if let ExprKind::IntLiteral(v) = arg.value.kind {
                                    paged_block_size = v;
                                }
                            }
                        }
                    }
                }
                "rope" => {
                    rope_q = true;
                    if let Some(ref args) = deco.args {
                        for arg in args {
                            let aname = arg
                                .name
                                .as_ref()
                                .and_then(|s| self.interner.resolve(s.0))
                                .unwrap_or("");
                            if aname == "style" {
                                if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                                    if s == "adjacent" {
                                        rope_style = crate::flash_attention::RopeStyle::Adjacent;
                                    }
                                }
                            }
                        }
                    }
                }
                "gqa" => {
                    if let Some(ref args) = deco.args {
                        for arg in args {
                            let aname = arg
                                .name
                                .as_ref()
                                .and_then(|s| self.interner.resolve(s.0))
                                .unwrap_or("");
                            if aname == "groups" {
                                if let ExprKind::IntLiteral(v) = arg.value.kind {
                                    gqa_group_size = v as u32;
                                }
                            }
                        }
                    }
                }
                "tree_mask" => {
                    // Paper §4 v1: bare decorator only. Reject any args
                    // (positional or named) with a clear error so future args
                    // (e.g. a max-depth budget) require an explicit spec
                    // update + extraction-site change rather than silently
                    // being ignored.
                    if let Some(ref args) = deco.args {
                        if !args.is_empty() {
                            return Err(CodegenError::new(
                                "@tree_mask takes no arguments in v1 (bare decorator only)".to_string(),
                            ));
                        }
                    }
                    tree_mask = true;
                }
                "attention_sink" => {
                    // Sprint 2 cycle-4 paper §4.3 API-surface landing.
                    // Sprint 2 cycle-5 added an unconditional refusal for
                    // `tokens>0` to close the silent-correctness gap that
                    // v0 left open (decorator parsed, but no SMEM emission).
                    // Sprint 1b cycle-7 (this commit) LIFTS that refusal
                    // for the narrow Tier A forward single-tile config —
                    // assign `num_sink_tokens = N` here and let the
                    // per-construction-site `attention_sinks_v1_eligible`
                    // check below refuse non-narrow configs with an axis-
                    // specific blocking reason + future Sprint citation.
                    //
                    // Semantic validation (`stmt.rs`) catches negative /
                    // zero / non-literal / unknown-arg cases up front,
                    // so this loop only handles the happy path. tokens=0
                    // cannot be written (semantic rejects it); users who
                    // want sinks disabled simply OMIT the decorator
                    // (num_sink_tokens stays at the local-default 0 and
                    // flows through eligibility unchanged via the early
                    // `if config.num_sink_tokens == 0` return).
                    if let Some(ref args) = deco.args {
                        for arg in args {
                            if let Some(ref name_sym) = arg.name {
                                let aname = self
                                    .interner
                                    .resolve(name_sym.0)
                                    .unwrap_or("");
                                if aname == "tokens" {
                                    if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                        if *n > 0 {
                                            num_sink_tokens = *n as u32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // ── @autotune handling for FlashAttention ──────────────────────
        //
        // When @autotune(block_q=[32, 64, 128], block_kv=[32, 64]) is present
        // alongside @flash_attention, we generate |block_q| x |block_kv| PTX
        // variants. Each variant's shared memory is validated against the 48KB
        // sm_52 limit. When paged, block_kv must be a multiple of block_size.
        // The "primary" context uses the middle-value fallback (or the single
        // default when @autotune is absent).

        // Sprint 1b cycle-7: helper to reject @attention_sink configs that
        // are not v1-narrow. Called at EACH FlashAttentionConfig
        // construction site below (one for each path: autotune-variant,
        // autotune-primary, single-config) so the eligibility check sees
        // the FULLY-MATERIALIZED config and the error message names the
        // exact failing axis + future Sprint that lifts it.
        //
        // Header format makes the user's input visible up front (`@attention_sink(tokens={n})`)
        // and the body cites paper §4.3 v1 narrowness + the eligibility
        // predicate's blocking_reason (axis + Sprint). Disabled-sinks
        // (num_sink_tokens==0) configs short-circuit eligibility and never
        // hit this path.
        fn reject_if_sinks_ineligible(
            config: &crate::flash_attention::FlashAttentionConfig,
        ) -> Result<(), CodegenError> {
            if config.num_sink_tokens == 0 {
                return Ok(());
            }
            let (eligible, why) =
                crate::flash_attention_v2::sinks::attention_sinks_v1_eligible(config);
            if eligible {
                return Ok(());
            }
            let blocking = why.unwrap_or("(unknown axis)");
            Err(CodegenError::new(format!(
                "@attention_sink(tokens={n}) - v1 narrowness violation: {blocking}.\n\n\
                 v1 supports: Tier A scalar forward, single-tile \
                 (block_q==block_kv==seq_len), causal=false, rope_q=false, \
                 gqa_group_size=1, paged=false, segment_masked=false, \
                 tree_mask=false, no @train (no backward), no \
                 csha.fused_projections.",
                n = config.num_sink_tokens,
                blocking = blocking,
            )))
        }

        let has_autotune = decorators.iter().any(|d| {
            d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
        });

        // DOC-GAP F.2: allow `@flash_attention(head_dim=N)` to override
        // the baked-in 64.  Without this, the Tier C fused-backward SMEM
        // validator rejects every block_q/block_kv tuple at hd=64
        // (713 KB > 99 KB cap) and no source-level NSL program can reach
        // the fused backward — hd=32 is the only config reachable today.
        let default_head_dim: i64 = head_dim_override.unwrap_or(64);

        if has_autotune {
            let params = self.extract_autotune_params(decorators)?;

            // Extract block_q and block_kv lists from autotune params
            let block_q_values: Vec<i64> = params
                .iter()
                .find(|(name, _)| name == "block_q")
                .map(|(_, vals)| vals.clone())
                .unwrap_or_else(|| vec![64]);
            let block_kv_values: Vec<i64> = params
                .iter()
                .find(|(name, _)| name == "block_kv")
                .map(|(_, vals)| vals.clone())
                .unwrap_or_else(|| vec![64]);

            // Build tuning params for Cartesian product and middle-value selection
            let tune_params: crate::autotune::TuningParams = vec![
                ("block_q".to_string(), block_q_values.clone()),
                ("block_kv".to_string(), block_kv_values.clone()),
            ];

            if self.compile_options.no_autotune {
                eprintln!("[nsl] autotune: --no-autotune, using middle values for flash_attention");
            }

            // Generate all (block_q, block_kv) combinations
            let variants = crate::autotune::cartesian_product(&tune_params);

            // Validate and synthesize PTX for each variant
            for variant in &variants {
                let bq = variant
                    .iter()
                    .find(|(n, _)| n == "block_q")
                    .map(|(_, v)| *v)
                    .unwrap_or(64);
                let bkv = variant
                    .iter()
                    .find(|(n, _)| n == "block_kv")
                    .map(|(_, v)| *v)
                    .unwrap_or(64);

                let test_config = crate::flash_attention::FlashAttentionConfig {
                    block_q: bq,
                    block_kv: bkv,
                    head_dim: default_head_dim,
                    causal,
                    paged,
                    rope_q,
                    rope_style,
                    gqa_group_size,
                    tree_mask,
                    num_sink_tokens,
                    gpu_sm: parse_gpu_sm_from_target(&self.compile_options.target),
                    segment_masked: false,
                    csha: None,
                    checkpoint: None,
                };

                // Sprint 1b cycle-7: @attention_sink v1 narrowness check.
                reject_if_sinks_ineligible(&test_config)?;

                // Shared memory validation: (block_q + block_kv) * head_dim * 2 <= 49152 (48KB)
                let mut diags = Vec::<String>::new();
                let shmem = crate::flash_attention_selector::shared_mem_bytes_selected_with_diag(
                    &test_config, &mut diags,
                );
                for d in diags { eprintln!("warning: {d}"); }
                if shmem > 49152 {
                    return Err(CodegenError::new(format!(
                        "@autotune variant (block_q={}, block_kv={}) requires {}KB shared memory, exceeds 48KB limit for sm_52",
                        test_config.block_q, test_config.block_kv, shmem / 1024
                    )));
                }

                // Paged block_kv alignment validation
                if paged && bkv % paged_block_size != 0 {
                    return Err(CodegenError::new(format!(
                        "@autotune block_kv={} is not a multiple of @paged_kv block_size={}",
                        bkv, paged_block_size
                    )));
                }

                // PCA Tier B Planner (planner spec §5): use the emission helper.
                // Autotune variants set `segment_masked: false`, so
                // `should_emit_tier_b_at_codegen` returns false; the helper
                // returns only `base_ptx` and the Tier-B-on PTX is `None`.
                // Output is byte-identical to the pre-migration selector path
                // because both ultimately call
                // `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`.
                //
                // Retain the selector kernel-name call for its diagnostic side
                // effect (API stability).
                let mut diags = Vec::<String>::new();
                let _ = crate::flash_attention_selector::flash_attention_kernel_name_selected_with_diag(
                    &test_config, &mut diags,
                );
                for d in diags { eprintln!("warning: {d}"); }
                let emission =
                    crate::pca_tier_b::emit_tier_b_variants_for_config(&test_config);
                let variant_kernel_name = emission.base_kernel_name.clone();
                self.embed_flash_ptx(&variant_kernel_name, emission.base_ptx)?;
                if let (Some(on_ptx), Some(on_name)) =
                    (emission.tier_b_on_ptx, emission.tier_b_on_kernel_name)
                {
                    // Defensive: autotune variants are non-segment_masked, but
                    // honour the emission helper's output uniformly so the
                    // (currently dead) Tier-B-on path is wired identically
                    // to the single-config and CSHA-training paths.
                    self.embed_flash_ptx(&on_name, on_ptx)?;
                }
            }

            // Select the middle values as the primary config (fallback / default dispatch)
            let fallback = crate::autotune::select_middle_values(&tune_params);
            let primary_bq = fallback
                .iter()
                .find(|(n, _)| n == "block_q")
                .map(|(_, v)| *v)
                .unwrap_or(64);
            let primary_bkv = fallback
                .iter()
                .find(|(n, _)| n == "block_kv")
                .map(|(_, v)| *v)
                .unwrap_or(64);

            let config = crate::flash_attention::FlashAttentionConfig {
                block_q: primary_bq,
                block_kv: primary_bkv,
                head_dim: default_head_dim,
                causal,
                paged,
                rope_q,
                rope_style,
                gqa_group_size,
                tree_mask,
                num_sink_tokens,
                gpu_sm: parse_gpu_sm_from_target(&self.compile_options.target),
                segment_masked: false,
                csha: None,
                checkpoint: None,
            };

            // Sprint 1b cycle-7: @attention_sink v1 narrowness check.
            reject_if_sinks_ineligible(&config)?;

            let mut diags = Vec::<String>::new();
            let kernel_name = crate::flash_attention_selector::flash_attention_kernel_name_selected_with_diag(
                &config, &mut diags,
            );
            for d in diags { eprintln!("warning: {d}"); }

            // The primary variant's PTX was already embedded in the loop above.
            // Look up its .rodata IDs from kernel_ptx_data (stored by embed_flash_ptx).
            let (ptx_data_id, name_data_id) = self
                .kernels
                .kernel_ptx_data
                .get(&format!("flash_{}", kernel_name))
                .copied()
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "primary flash variant '{}' not found after autotune embedding",
                        kernel_name
                    ))
                })?;

            // Embed backward PTX alongside forward.
            // Backward tiles are budget-selected per head_dim (decoupled from the
            // forward's tiles) so hd64/hd128 fit the 99 KB SMEM opt-in cap and run on
            // GPU rather than falling back to CPU. The runtime mirrors this via
            // `select_backward_blocks(head_dim)`; the two must stay in agreement.
            let (bwd_block_q, bwd_block_kv) =
                crate::flash_attention::backward_select_blocks(config.head_dim);
            let bwd_config = crate::flash_attention::FlashAttentionBackwardConfig {
                block_q: bwd_block_q,
                block_kv: bwd_block_kv,
                head_dim: config.head_dim,
                causal: config.causal,
                gpu_sm: config.gpu_sm,
                segment_masked: false,
            };
            let (bwd_p1, bwd_p2) =
                crate::flash_attention::synthesize_flash_attention_backward_ptx(&bwd_config);
            let bwd_p1_id = self.embed_raw_data("__nsl_flash_bwd_p1", bwd_p1)?;
            let bwd_p2_id = self.embed_raw_data("__nsl_flash_bwd_p2", bwd_p2)?;

            // Embed backward kernel name strings as null-terminated C strings
            let bwd_p1_name = crate::flash_attention::flash_attention_bwd_d_kernel_name(&bwd_config);
            let mut bwd_p1_name_bytes = bwd_p1_name.into_bytes();
            bwd_p1_name_bytes.push(0); // null-terminate
            let bwd_p1_name_id = self.embed_raw_data("__nsl_flash_bwd_p1_name", bwd_p1_name_bytes)?;

            let bwd_p2_name = crate::flash_attention::flash_attention_bwd_main_kernel_name(&bwd_config);
            let mut bwd_p2_name_bytes = bwd_p2_name.into_bytes();
            bwd_p2_name_bytes.push(0); // null-terminate
            let bwd_p2_name_id = self.embed_raw_data("__nsl_flash_bwd_p2_name", bwd_p2_name_bytes)?;

            // PCA Tier B Planner: primary autotune variant's Tier-B-on IDs
            // (looked up from `kernel_ptx_data` if `embed_flash_ptx` registered
            // one — non-segment_masked configs never register one, so this is
            // always None for the autotune path today).
            let primary_tier_b_on_name = format!(
                "flash_{}_tier_b_max{}",
                kernel_name,
                crate::pca_tier_b::TIER_B_MAX_BAKED_SEQ_LEN,
            );
            let (tier_b_on_ptx_data_id, tier_b_on_name_data_id) = self
                .kernels
                .kernel_ptx_data
                .get(&primary_tier_b_on_name)
                .copied()
                .map(|(p, n)| (Some(p), Some(n)))
                .unwrap_or((None, None));

            Ok(Some(FlashAttentionCompileContext {
                ptx_data_id,
                name_data_id,
                config,
                bwd_phase1_data_id: Some(bwd_p1_id),
                bwd_phase2_data_id: Some(bwd_p2_id),
                bwd_phase1_name_data_id: Some(bwd_p1_name_id),
                bwd_phase2_name_data_id: Some(bwd_p2_name_id),
                bwd_config: Some(bwd_config),
                // Gap B: these are populated in `compile_flash_attention_kernels`
                // after the context is built, if the module contains a `@train`
                // block. The autotune path here doesn't handle training
                // specialisation — Gap B wiring uses the single-config fallback
                // below for both autotune and non-autotune programs.
                csha_forward_with_saves_ptx_id: None,
                csha_forward_with_saves_name_id: None,
                csha_backward_ptx_data_id: None,
                csha_backward_name_data_id: None,
                csha_training_config: None,
                // PCA Tier B Planner: primary forward Tier-B-on IDs (always None
                // for autotune path's non-segment_masked configs).
                tier_b_on_ptx_data_id,
                tier_b_on_name_data_id,
                csha_with_saves_tier_b_on_ptx_id: None,
                csha_with_saves_tier_b_on_name_id: None,
                csha_backward_tier_b_on_ptx_id: None,
                csha_backward_tier_b_on_name_id: None,
            }))
        } else {
            // No @autotune — single-config path (original behaviour)
            let config = crate::flash_attention::FlashAttentionConfig {
                block_q: 64,
                block_kv: 64,
                head_dim: default_head_dim,
                causal,
                paged,
                rope_q,
                rope_style,
                gqa_group_size,
                tree_mask,
                num_sink_tokens,
                gpu_sm: parse_gpu_sm_from_target(&self.compile_options.target),
                segment_masked: false,
                csha: None,
                checkpoint: None,
            };

            // Sprint 1b cycle-7: @attention_sink v1 narrowness check.
            reject_if_sinks_ineligible(&config)?;

            // PCA Tier B Planner (planner spec §5): use the emission helper.
            // Single-config path sets `segment_masked: false`, so
            // `should_emit_tier_b_at_codegen` returns false; the helper returns
            // only `base_ptx`. Output is byte-identical to the pre-migration
            // selector path because both ultimately call
            // `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`.
            //
            // Retain the selector kernel-name call for its diagnostic side
            // effect (API stability).
            let mut diags = Vec::<String>::new();
            let _ = crate::flash_attention_selector::flash_attention_kernel_name_selected_with_diag(
                &config, &mut diags,
            );
            for d in diags { eprintln!("warning: {d}"); }
            let emission = crate::pca_tier_b::emit_tier_b_variants_for_config(&config);
            let kernel_name = emission.base_kernel_name.clone();
            let ptx_bytes = emission.base_ptx;

            // Embed PTX bytes in .rodata
            let ptx_data_id = self
                .module
                .declare_data(
                    &format!("__nsl_flash_ptx_{}", kernel_name),
                    cranelift_module::Linkage::Local,
                    false,
                    false,
                )
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to declare flash PTX data for '{}': {e}",
                        kernel_name
                    ))
                })?;
            let mut data_desc = DataDescription::new();
            data_desc.define(ptx_bytes.into_boxed_slice());
            self.module
                .define_data(ptx_data_id, &data_desc)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to define flash PTX data for '{}': {e}",
                        kernel_name
                    ))
                })?;

            // Embed kernel name (null-terminated) in .rodata
            let mut name_bytes = kernel_name.as_bytes().to_vec();
            name_bytes.push(0);
            let name_data_id = self
                .module
                .declare_data(
                    &format!("__nsl_flash_name_{}", kernel_name),
                    cranelift_module::Linkage::Local,
                    false,
                    false,
                )
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to declare flash name data for '{}': {e}",
                        kernel_name
                    ))
                })?;
            let mut name_desc = DataDescription::new();
            name_desc.define(name_bytes.into_boxed_slice());
            self.module
                .define_data(name_data_id, &name_desc)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to define flash name data for '{}': {e}",
                        kernel_name
                    ))
                })?;

            // Tier-B-on variant: embed only when the emission helper produced one.
            // Single-config sets `segment_masked: false` so this branch is dead
            // today; wired for parity with the CSHA training path.
            let (tier_b_on_ptx_id, tier_b_on_name_id) = match (
                emission.tier_b_on_ptx,
                emission.tier_b_on_kernel_name,
            ) {
                (Some(on_ptx), Some(on_name)) => {
                    let on_ptx_id = self.embed_raw_data(
                        &format!("__nsl_flash_ptx_{}", on_name),
                        on_ptx,
                    )?;
                    let mut on_name_bytes = on_name.as_bytes().to_vec();
                    on_name_bytes.push(0);
                    let on_name_id = self.embed_raw_data(
                        &format!("__nsl_flash_name_{}", on_name),
                        on_name_bytes,
                    )?;
                    (Some(on_ptx_id), Some(on_name_id))
                }
                _ => (None, None),
            };

            // Embed backward PTX alongside forward.
            // Backward tiles are budget-selected per head_dim (decoupled from the
            // forward's tiles) so hd64/hd128 fit the 99 KB SMEM opt-in cap and run on
            // GPU rather than falling back to CPU. The runtime mirrors this via
            // `select_backward_blocks(head_dim)`; the two must stay in agreement.
            let (bwd_block_q, bwd_block_kv) =
                crate::flash_attention::backward_select_blocks(config.head_dim);
            let bwd_config = crate::flash_attention::FlashAttentionBackwardConfig {
                block_q: bwd_block_q,
                block_kv: bwd_block_kv,
                head_dim: config.head_dim,
                causal: config.causal,
                gpu_sm: config.gpu_sm,
                segment_masked: false,
            };
            let (bwd_p1, bwd_p2) =
                crate::flash_attention::synthesize_flash_attention_backward_ptx(&bwd_config);
            let bwd_p1_id = self.embed_raw_data("__nsl_flash_bwd_p1", bwd_p1)?;
            let bwd_p2_id = self.embed_raw_data("__nsl_flash_bwd_p2", bwd_p2)?;

            // Embed backward kernel name strings as null-terminated C strings
            let bwd_p1_name = crate::flash_attention::flash_attention_bwd_d_kernel_name(&bwd_config);
            let mut bwd_p1_name_bytes = bwd_p1_name.into_bytes();
            bwd_p1_name_bytes.push(0);
            let bwd_p1_name_id = self.embed_raw_data("__nsl_flash_bwd_p1_name", bwd_p1_name_bytes)?;

            let bwd_p2_name = crate::flash_attention::flash_attention_bwd_main_kernel_name(&bwd_config);
            let mut bwd_p2_name_bytes = bwd_p2_name.into_bytes();
            bwd_p2_name_bytes.push(0);
            let bwd_p2_name_id = self.embed_raw_data("__nsl_flash_bwd_p2_name", bwd_p2_name_bytes)?;

            Ok(Some(FlashAttentionCompileContext {
                ptx_data_id,
                name_data_id,
                config,
                bwd_phase1_data_id: Some(bwd_p1_id),
                bwd_phase2_data_id: Some(bwd_p2_id),
                bwd_phase1_name_data_id: Some(bwd_p1_name_id),
                bwd_phase2_name_data_id: Some(bwd_p2_name_id),
                bwd_config: Some(bwd_config),
                // Gap B: filled in by `compile_flash_attention_kernels` after
                // this method returns — it decides whether to synthesize the
                // CSHA with-saves forward PTX and the fused backward PTX
                // based on a `@train`-block pre-scan of the top-level stmts.
                csha_forward_with_saves_ptx_id: None,
                csha_forward_with_saves_name_id: None,
                csha_backward_ptx_data_id: None,
                csha_backward_name_data_id: None,
                csha_training_config: None,
                // PCA Tier B Planner: primary forward Tier-B-on IDs (`None` for
                // non-segment_masked single-config; helper returned `None`
                // above).
                tier_b_on_ptx_data_id: tier_b_on_ptx_id,
                tier_b_on_name_data_id: tier_b_on_name_id,
                csha_with_saves_tier_b_on_ptx_id: None,
                csha_with_saves_tier_b_on_name_id: None,
                csha_backward_tier_b_on_ptx_id: None,
                csha_backward_tier_b_on_name_id: None,
            }))
        }
    }

    /// Backward variant table for decorator-free SDPA (per-head_dim, PR #335
    /// wiring gap). Lazily synthesizes and embeds the classic two-phase
    /// backward PTX for each supported head_dim under the op's static
    /// `causal` flag, so `wengert_lower` can select the matching variant on
    /// the RUNTIME head_dim value. Emission is a pure function of
    /// (blocks, head_dim, causal, gpu_sm); blocks come from the same
    /// `backward_select_blocks` table the runtime re-derives from head_dim,
    /// which is what makes compile-time synthesis + runtime selection agree.
    ///
    /// head_dims not in the table (and GQA ops, gated runtime-side on
    /// kv_h == h) select null pointers and keep today's CPU fallback.
    /// hd=256 is omitted on purpose: its (64,64) tiles deliberately overflow
    /// the 99 KB SMEM opt-in cap, so a variant could never launch.
    pub(crate) fn ensure_sdpa_bwd_variant_table(
        &mut self,
        causal: bool,
        segment_masked: bool,
    ) -> Result<Vec<SdpaBwdVariant>, CodegenError> {
        const SDPA_BWD_VARIANT_HEAD_DIMS: &[i64] = &[32, 64, 128];

        if let Some(table) = self.kernels.sdpa_bwd_variants.get(&(causal, segment_masked)) {
            return Ok(table.clone());
        }
        // The classic backward PTX is CUDA-only, and `parse_gpu_sm_from_target`
        // PANICS on anything but "cuda"/"sm_<N>". The decorated path only
        // reaches it behind an @flash_attention decorator, but this lazy path
        // fires for EVERY decorator-free SDPA train compile — including
        // `--target rocm|metal|webgpu|fpga`, which compiled fine before the
        // variant table existed (null pointers → CPU backward). Keep exactly
        // that behavior for non-CUDA targets: an empty table lowers to null
        // pointers, no panic.
        let target = self.compile_options.target.as_str();
        if target != "cuda" && !target.starts_with("sm_") {
            self.kernels
                .sdpa_bwd_variants
                .insert((causal, segment_masked), Vec::new());
            return Ok(Vec::new());
        }
        let gpu_sm = parse_gpu_sm_from_target(&self.compile_options.target);
        let c = i64::from(causal);
        let sm = i64::from(segment_masked);
        let mut table = Vec::with_capacity(SDPA_BWD_VARIANT_HEAD_DIMS.len());
        for &head_dim in SDPA_BWD_VARIANT_HEAD_DIMS {
            let (block_q, block_kv) = crate::flash_attention::backward_select_blocks(head_dim);
            let bwd_config = crate::flash_attention::FlashAttentionBackwardConfig {
                block_q,
                block_kv,
                head_dim,
                causal,
                gpu_sm,
                segment_masked,
            };
            let (p1, p2) =
                crate::flash_attention::synthesize_flash_attention_backward_ptx(&bwd_config);
            let phase1_ptx =
                self.embed_raw_data(&format!("__nsl_sdpa_bwd_p1_c{c}_s{sm}_hd{head_dim}"), p1)?;
            let phase2_ptx =
                self.embed_raw_data(&format!("__nsl_sdpa_bwd_p2_c{c}_s{sm}_hd{head_dim}"), p2)?;

            let mut p1_name =
                crate::flash_attention::flash_attention_bwd_d_kernel_name(&bwd_config)
                    .into_bytes();
            p1_name.push(0);
            let phase1_name = self.embed_raw_data(
                &format!("__nsl_sdpa_bwd_p1_name_c{c}_s{sm}_hd{head_dim}"),
                p1_name,
            )?;

            let mut p2_name =
                crate::flash_attention::flash_attention_bwd_main_kernel_name(&bwd_config)
                    .into_bytes();
            p2_name.push(0);
            let phase2_name = self.embed_raw_data(
                &format!("__nsl_sdpa_bwd_p2_name_c{c}_s{sm}_hd{head_dim}"),
                p2_name,
            )?;

            table.push(SdpaBwdVariant {
                head_dim,
                phase1_ptx,
                phase1_name,
                phase2_ptx,
                phase2_name,
            });
        }
        self.kernels
            .sdpa_bwd_variants
            .insert((causal, segment_masked), table.clone());
        Ok(table)
    }

    /// PCA Stage C: fused-forward variant table for decorator-free SDPA.
    ///
    /// Synthesizes the v2 scalar forward (csha: None — the same emitter
    /// family every decorated plain model launches) per supported head_dim,
    /// under the op's static `(causal, segment_masked)`, and embeds
    /// PTX + entry-name bytes in .rodata. Tile sizes are probed against the
    /// v2 scalar matrix per head_dim, largest-first, so each entry uses the
    /// biggest admissible tiles; head_dims with no admissible tiling are
    /// skipped (runtime head_dim then matches nothing → null PTX → the
    /// lowering's decomposed fallback, exactly like the backward table).
    ///
    /// For `segment_masked` tables, `emit_tier_b_variants_for_config` also
    /// returns the Tier-B tile-skip variant (the codegen gate is
    /// `should_emit_tier_b_at_codegen == segment_masked`); its pair is
    /// embedded alongside so the runtime's 4-condition gate can promote the
    /// launch to the skipping kernel.
    pub(crate) fn ensure_sdpa_fwd_variant_table(
        &mut self,
        causal: bool,
        segment_masked: bool,
    ) -> Result<Vec<super::SdpaFwdVariant>, CodegenError> {
        const SDPA_FWD_VARIANT_HEAD_DIMS: &[i64] = &[32, 64, 128];
        // Largest-first tile probe order. (64,64) is the decorated-path
        // default; smaller tiles admit larger head_dims within the v2
        // scalar matrix / SMEM budget.
        const TILE_CANDIDATES: &[(i64, i64)] = &[(64, 64), (32, 32), (32, 16), (16, 16)];

        if let Some(table) = self.kernels.sdpa_fwd_variants.get(&(causal, segment_masked)) {
            return Ok(table.clone());
        }
        let target = self.compile_options.target.as_str();
        if target != "cuda" && !target.starts_with("sm_") {
            self.kernels
                .sdpa_fwd_variants
                .insert((causal, segment_masked), Vec::new());
            return Ok(Vec::new());
        }
        let gpu_sm = parse_gpu_sm_from_target(&self.compile_options.target);
        let c = i64::from(causal);
        let sm = i64::from(segment_masked);
        let mut table = Vec::with_capacity(SDPA_FWD_VARIANT_HEAD_DIMS.len());
        for &head_dim in SDPA_FWD_VARIANT_HEAD_DIMS {
            let mut admitted = None;
            for &(bq, bkv) in TILE_CANDIDATES {
                let config = crate::flash_attention::FlashAttentionConfig {
                    block_q: bq,
                    block_kv: bkv,
                    head_dim,
                    causal,
                    paged: false,
                    rope_q: false,
                    rope_style: crate::flash_attention::RopeStyle::HalfSplit,
                    gqa_group_size: 1,
                    tree_mask: false,
                    num_sink_tokens: 0,
                    gpu_sm,
                    segment_masked,
                    csha: None,
                    checkpoint: None,
                };
                if crate::flash_attention_v2::smem_layout::validate_scalar_v2_config(
                    &config,
                    crate::flash_attention_v2::smem_layout::Direction::Forward,
                )
                .is_ok()
                {
                    admitted = Some(config);
                    break;
                }
            }
            let Some(config) = admitted else { continue };

            let emission = crate::pca_tier_b::emit_tier_b_variants_for_config(&config);
            let mut diags = Vec::<String>::new();
            let mut smem_bytes = crate::flash_attention_selector::shared_mem_bytes_selected_with_diag(
                &config, &mut diags,
            ) as i64;
            for d in diags {
                eprintln!("warning: {d}");
            }
            if emission.tier_b_on_ptx.is_some() {
                // Tier-B-on kernels build their per-tile segment range table
                // in the DYNAMIC shmem window at `tier_b_range_table_offset`
                // (base + seg arena), i.e. BEYOND the base kernel's request —
                // an under-sized launch is an in-kernel ILLEGAL_ADDRESS
                // (benchmark discovery). One smem value serves both launches,
                // so size for the worst admissible seq (the runtime gate caps
                // at TIER_B_MAX_BAKED_SEQ_LEN); the base kernel simply gets a
                // slightly larger window than it needs.
                let tb = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
                    &config,
                    crate::flash_attention_v2::smem_layout::Direction::Forward,
                ) + crate::pca_tilerange::tier_b_range_table_bytes(
                    &config,
                    crate::pca_tier_b::TIER_B_MAX_BAKED_SEQ_LEN,
                );
                smem_bytes = smem_bytes.max(tb as i64);
            }

            let base_ptx = self.embed_raw_data(
                &format!("__nsl_sdpa_fwd_c{c}_s{sm}_hd{head_dim}"),
                emission.base_ptx,
            )?;
            let base_kernel_name = emission.base_kernel_name.clone();
            let mut base_name_bytes = emission.base_kernel_name.into_bytes();
            base_name_bytes.push(0);
            let base_name = self.embed_raw_data(
                &format!("__nsl_sdpa_fwd_name_c{c}_s{sm}_hd{head_dim}"),
                base_name_bytes,
            )?;

            // PROMOTED (2026-07-14): Tier-B tile-skip passed the
            // packed-shape parity gate demanded by the original Stage-C
            // deferral. `tier_b_tile_skip_gpu_parity.rs` drives the Tier-B
            // PTX (launch-count-proven via `nsl_sdpa_fused_launch_count`)
            // against the base segment-masked kernel across single-doc /
            // tile-aligned / mid-tile / high-density patterns at
            // s∈{1024,4096} and b=4/h=8 — BITWISE-identical outputs and
            // LSE everywhere, which is what correct tile-skip predicts (a
            // fully-masked KV tile is an exact no-op in the online
            // softmax: its -1e9 mass rescales to +0.0 once any visible
            // tile follows, and the causal diagonal guarantees one). The
            // benchmark-era "~1e-2 deviation" that motivated the deferral
            // was an artifact of the then-live per-batch segment-staging
            // bug, not a Tier-B defect. Historical launcher discoveries
            // (still true): (a) the emission helper's tier-b kernel NAME
            // never exists as a PTX entry (the table stores the base
            // name), (b) the dynamic-SMEM launch must cover the range
            // table (sized above). Operational escape hatch:
            // NSL_SDPA_TIER_B_DISABLE=1 forces the base kernel at runtime.
            const PROMOTE_TIER_B: bool = true;
            let (tier_b_ptx, tier_b_name) =
                match (emission.tier_b_on_ptx, emission.tier_b_on_kernel_name) {
                    (Some(on_ptx), Some(_on_name)) if PROMOTE_TIER_B => {
                        let tb_ptx = self.embed_raw_data(
                            &format!("__nsl_sdpa_fwd_tb_c{c}_s{sm}_hd{head_dim}"),
                            on_ptx,
                        )?;
                        // The v2 prelude names EVERY kernel with the base v2
                        // name — `emission.tier_b_on_kernel_name`'s
                        // `_tier_b_max{N}` suffix is a codegen-cache key that
                        // does NOT exist as a PTX entry (pre-existing helper
                        // inconsistency; the Tier-B-on path was dead in
                        // production until Stage C, so cuModuleGetFunction
                        // never saw it). Store the name that is actually IN
                        // the PTX; the runtime distinguishes the two modules
                        // by PTX pointer, not by entry name.
                        let mut tb_name_bytes = base_kernel_name.clone().into_bytes();
                        tb_name_bytes.push(0);
                        let tb_name = self.embed_raw_data(
                            &format!("__nsl_sdpa_fwd_tb_name_c{c}_s{sm}_hd{head_dim}"),
                            tb_name_bytes,
                        )?;
                        (Some(tb_ptx), Some(tb_name))
                    }
                    _ => (None, None),
                };

            table.push(super::SdpaFwdVariant {
                head_dim,
                block_q: config.block_q,
                block_kv: config.block_kv,
                smem_bytes,
                base_ptx,
                base_name,
                tier_b_ptx,
                tier_b_name,
            });
        }
        self.kernels
            .sdpa_fwd_variants
            .insert((causal, segment_masked), table.clone());
        Ok(table)
    }

    /// Embed raw bytes in .rodata and return the DataId.
    fn embed_raw_data(&mut self, label: &str, data: Vec<u8>) -> Result<DataId, CodegenError> {
        let data_id = self
            .module
            .declare_data(label, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| CodegenError::new(format!("failed to declare data '{}': {e}", label)))?;
        let mut desc = DataDescription::new();
        desc.define(data.into_boxed_slice());
        self.module
            .define_data(data_id, &desc)
            .map_err(|e| CodegenError::new(format!("failed to define data '{}': {e}", label)))?;
        Ok(data_id)
    }

    /// Embed a FlashAttention PTX variant in .rodata and record its DataIds
    /// in `kernel_ptx_data` under the key `flash_{kernel_name}`.
    fn embed_flash_ptx(
        &mut self,
        kernel_name: &str,
        ptx_bytes: Vec<u8>,
    ) -> Result<(DataId, DataId), CodegenError> {
        let ptx_data_id = self
            .module
            .declare_data(
                &format!("__nsl_flash_ptx_{}", kernel_name),
                cranelift_module::Linkage::Local,
                false,
                false,
            )
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare flash PTX data for '{}': {e}",
                    kernel_name
                ))
            })?;
        let mut data_desc = DataDescription::new();
        data_desc.define(ptx_bytes.into_boxed_slice());
        self.module
            .define_data(ptx_data_id, &data_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define flash PTX data for '{}': {e}",
                    kernel_name
                ))
            })?;

        let mut name_bytes = kernel_name.as_bytes().to_vec();
        name_bytes.push(0);
        let name_data_id = self
            .module
            .declare_data(
                &format!("__nsl_flash_name_{}", kernel_name),
                cranelift_module::Linkage::Local,
                false,
                false,
            )
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to declare flash name data for '{}': {e}",
                    kernel_name
                ))
            })?;
        let mut name_desc = DataDescription::new();
        name_desc.define(name_bytes.into_boxed_slice());
        self.module
            .define_data(name_data_id, &name_desc)
            .map_err(|e| {
                CodegenError::new(format!(
                    "failed to define flash name data for '{}': {e}",
                    kernel_name
                ))
            })?;

        self.kernels.kernel_ptx_data.insert(
            format!("flash_{}", kernel_name),
            (ptx_data_id, name_data_id),
        );
        Ok((ptx_data_id, name_data_id))
    }
}
