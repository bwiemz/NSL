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
                StmtKind::Decorated { decorators, stmt: inner } => {
                    if let StmtKind::KernelDef(kernel) = &inner.kind {
                        let has_autotune = decorators.iter().any(|d| {
                            d.name.len() == 1
                                && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
                        });

                        if has_autotune {
                            let params = self.extract_autotune_params(decorators)?;
                            let kernel_name = self.interner.resolve(kernel.name.0)
                                .unwrap_or("unknown")
                                .to_string();

                            if self.compile_options.no_autotune {
                                eprintln!(
                                    "[nsl] autotune: --no-autotune, using middle values for {}",
                                    kernel_name
                                );
                            }

                            // For now, always use middle-value fallback.
                            // TODO(M26): add GPU benchmarking path when CUDA is available.
                            let _fallback = crate::autotune::select_middle_values(&params);

                            // Compile the kernel normally (constant substitution deferred to M26)
                            self.compile_single_kernel(kernel)?;
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

    /// Compile a single kernel definition: generate PTX and embed in .rodata.
    fn compile_single_kernel(
        &mut self,
        kernel: &nsl_ast::block::KernelDef,
    ) -> Result<(), CodegenError> {
        let ptx_bytes = crate::kernel::KernelCompiler::compile(kernel, self.interner);
        let kernel_name = self.interner.resolve(kernel.name.0).unwrap_or("__kernel").to_string();

        // Embed PTX bytes in .rodata
        let ptx_data_id = self.module
            .declare_data(
                &format!("__nsl_ptx_{}", kernel_name),
                cranelift_module::Linkage::Local,
                false,
                false,
            )
            .map_err(|e| CodegenError::new(format!(
                "failed to declare PTX data for kernel '{}': {e}", kernel_name
            )))?;
        let mut data_desc = cranelift_module::DataDescription::new();
        data_desc.define(ptx_bytes.into_boxed_slice());
        self.module
            .define_data(ptx_data_id, &data_desc)
            .map_err(|e| CodegenError::new(format!(
                "failed to define PTX data for kernel '{}': {e}", kernel_name
            )))?;

        // Embed kernel name (null-terminated) in .rodata
        let mut name_bytes = kernel_name.as_bytes().to_vec();
        name_bytes.push(0);
        let name_data_id = self.module
            .declare_data(
                &format!("__nsl_ptx_name_{}", kernel_name),
                cranelift_module::Linkage::Local,
                false,
                false,
            )
            .map_err(|e| CodegenError::new(format!(
                "failed to declare name data for kernel '{}': {e}", kernel_name
            )))?;
        let mut name_desc = cranelift_module::DataDescription::new();
        name_desc.define(name_bytes.into_boxed_slice());
        self.module
            .define_data(name_data_id, &name_desc)
            .map_err(|e| CodegenError::new(format!(
                "failed to define name data for kernel '{}': {e}", kernel_name
            )))?;

        self.kernel_ptx_data.insert(kernel_name, (ptx_data_id, name_data_id));
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
                d.name.len() == 1
                    && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
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
                                self.flash_attention_context = Some(ctx);
                                return Ok(());
                            }
                        }
                    }
                    continue;
                }
                _ => continue,
            };

            if let Some(ctx) = self.try_build_flash_context(decorators)? {
                self.flash_attention_context = Some(ctx);
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
            d.name.len() == 1
                && self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
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
                eprintln!(
                    "[nsl] autotune: --no-autotune, using middle values for flash_attention"
                );
            }

            // Generate all (block_q, block_kv) combinations
            let variants = crate::autotune::cartesian_product(&tune_params);

            // Validate and synthesize PTX for each variant
            for variant in &variants {
                let bq = variant.iter().find(|(n, _)| n == "block_q").map(|(_, v)| *v).unwrap_or(64);
                let bkv = variant.iter().find(|(n, _)| n == "block_kv").map(|(_, v)| *v).unwrap_or(64);

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
                let ptx_bytes = crate::flash_attention::synthesize_flash_attention_ptx(&test_config);
                let variant_kernel_name = crate::flash_attention::flash_attention_kernel_name(&test_config);
                self.embed_flash_ptx(&variant_kernel_name, ptx_bytes)?;
            }

            // Select the middle values as the primary config (fallback / default dispatch)
            let fallback = crate::autotune::select_middle_values(&tune_params);
            let primary_bq = fallback.iter().find(|(n, _)| n == "block_q").map(|(_, v)| *v).unwrap_or(64);
            let primary_bkv = fallback.iter().find(|(n, _)| n == "block_kv").map(|(_, v)| *v).unwrap_or(64);

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
            };

            let kernel_name = crate::flash_attention::flash_attention_kernel_name(&config);

            // The primary variant's PTX was already embedded in the loop above.
            // Look up its .rodata IDs from kernel_ptx_data (stored by embed_flash_ptx).
            let (ptx_data_id, name_data_id) = self
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

        self.kernel_ptx_data
            .insert(format!("flash_{}", kernel_name), (ptx_data_id, name_data_id));
        Ok((ptx_data_id, name_data_id))
    }
}
