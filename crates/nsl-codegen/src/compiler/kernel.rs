use std::collections::HashMap;

use cranelift_module::{DataDescription, DataId, Module};

use nsl_ast::decl::{Decorator, ModelMember};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Stmt, StmtKind};

use super::{Compiler, FlashAttentionCompileContext};
use crate::error::CodegenError;

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
                crate::kernel::KernelCompiler::compile(kernel, self.interner)
            }
            GpuTarget::Rocm | GpuTarget::Metal | GpuTarget::WebGpu => {
                // M47b: AST -> KIR -> backend-specific lowerer
                let kir = crate::kernel_lower::lower_kernel_to_ir(kernel, self.interner, target);

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
                    GpuTarget::Cuda => unreachable!(),
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
                )
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
                crate::kernel::KernelCompiler::compile_with_constants(kernel, interner, &const_map);
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
        for stmt in stmts {
            let decorators = match &stmt.kind {
                StmtKind::Decorated { decorators, stmt } => {
                    if matches!(&stmt.kind, StmtKind::FnDef(_)) {
                        decorators
                    } else {
                        continue;
                    }
                }
                // Check model layer declarations for @flash_attention
                // (model methods with decorators are stored on LayerDecl members)
                StmtKind::ModelDef(md) => {
                    for member in &md.members {
                        if let ModelMember::LayerDecl { decorators, .. } = member {
                            if let Some(ctx) = self.try_build_flash_context(decorators)? {
                                self.kernels.flash_attention_context = Some(ctx);
                                return Ok(());
                            }
                        }
                    }
                    continue;
                }
                _ => continue,
            };

            if let Some(ctx) = self.try_build_flash_context(decorators)? {
                self.kernels.flash_attention_context = Some(ctx);
                return Ok(());
            }
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

        for deco in decorators {
            if deco.name.len() != 1 {
                continue;
            }
            let dname = self.interner.resolve(deco.name[0].0).unwrap_or("");
            match dname {
                "flash_attention" => {
                    // Extract causal= arg if present
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

        let has_autotune = decorators.iter().any(|d| {
            d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
        });

        let default_head_dim: i64 = 64; // runtime extracts actual head_dim from tensor shape

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
                    tree_mask: false,
                    gpu_sm: 80,
                };

                // Shared memory validation: (block_q + block_kv) * head_dim * 2 <= 49152 (48KB)
                let shmem = crate::flash_attention::shared_mem_bytes(&test_config);
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

                // Synthesize and embed PTX for this variant
                let ptx_bytes =
                    crate::flash_attention::synthesize_flash_attention_ptx(&test_config);
                let variant_kernel_name =
                    crate::flash_attention::flash_attention_kernel_name(&test_config);
                self.embed_flash_ptx(&variant_kernel_name, ptx_bytes)?;
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
                tree_mask: false,
                gpu_sm: 80,
            };

            let kernel_name = crate::flash_attention::flash_attention_kernel_name(&config);

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

            Ok(Some(FlashAttentionCompileContext {
                ptx_data_id,
                name_data_id,
                config,
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
                tree_mask: false,
                gpu_sm: 80,
            };

            let ptx_bytes = crate::flash_attention::synthesize_flash_attention_ptx(&config);
            let kernel_name = crate::flash_attention::flash_attention_kernel_name(&config);

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

            Ok(Some(FlashAttentionCompileContext {
                ptx_data_id,
                name_data_id,
                config,
            }))
        }
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
