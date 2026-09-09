use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{BlockArg, InstBuilder};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::ExprKind;
use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

use crate::types::{is_block_filled, nsl_type_to_cl};
use cranelift_codegen::ir::Value;

// P0.1 per-surface VRAM accounting: the wire values of
// `nsl_gpu_set_alloc_surface` / `nsl_gpu_get_alloc_surface`. Declared once in
// `nsl_abi::wire::surface` (roadmap A3), which is also where the runtime's
// `SurfaceTag` (#[repr(u8)]) takes its discriminants — so the two cannot
// drift. Widened to i64 here because they are emitted as `iconst I64`.
// Each train-block bracket sets a surface for its allocation region and
// restores the caller's surface afterwards (get/set — nesting-safe).
pub(crate) const SURFACE_WEIGHTS: i64 = nsl_abi::wire::surface::SURFACE_WEIGHTS as i64;
pub(crate) const SURFACE_OPTIM_M: i64 = nsl_abi::wire::surface::SURFACE_OPTIM_M as i64;
pub(crate) const SURFACE_OPTIM_V: i64 = nsl_abi::wire::surface::SURFACE_OPTIM_V as i64;
pub(crate) const SURFACE_M_PARTIAL: i64 = nsl_abi::wire::surface::SURFACE_M_PARTIAL as i64;
pub(crate) const SURFACE_GRADS: i64 = nsl_abi::wire::surface::SURFACE_GRADS as i64;
pub(crate) const SURFACE_ACTIVATIONS: i64 = nsl_abi::wire::surface::SURFACE_ACTIVATIONS as i64;

/// FASE hook: one parameter's primal Value and its `accum_list` /
/// `param_list` index, keyed by its adjoint gradient VarId in the driver's
/// `adj_vid_to_hook_entry` (the CSLA save phase copies the indices into
/// its pending carrier; the FASE Deferred arm consumes the entries).
pub(crate) struct ParamHookEntry {
    pub(crate) primal_val: Value,
    // i64 index into accum_list (== param_list index for this param)
    pub(crate) accum_idx: i64,
}

/// Item C: how ONE parameter's optimizer moments are allocated under
/// `--zero-stage 3`, decided from its `ParameterPlan` entry and consumed by
/// `Compiler::emit_deferred_moment_fill`. Section 4 allocates nothing under
/// stage 3 (the plan and the runtime carve both post-date it), so this is
/// the single place the three shapes are spelled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MomentFill {
    /// `--zero-elementwise` eligible: a persistent 1/world_size SLICE on
    /// every rank, sized by the runtime from the carved shard.
    Elementwise,
    /// Tensor-granular sharded: the owner allocates the full moment, the
    /// rest keep the null placeholder (the stages-1/2 machinery, reused).
    OwnerGated,
    /// Replicated (tied / view-rooted / epilogue): full m/v on every rank,
    /// because every rank updates it from all-reduced gradients.
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceAdParamDiagnosticKind {
    Trainable,
    IgnoredConfig,
    IgnoredNonTensor,
}

/// HOW a train block arrived at its `grad_accumulation` window — kept apart
/// from the window VALUE because a diagnostic that says "1" needs to say why.
///
/// A `NonLiteral` variant used to live here: `train(..., grad_accumulation=GA)`
/// with a `const GA = 4` parsed, type-checked, and then lowered a window of
/// **1** with no diagnostic. The Training Configuration Contract
/// (`nsl_semantic::train_config`) now REFUSES a non-literal window — the
/// same contract `distill` always had — so a Literal here means the window
/// is exactly what the source says.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GradAccumulationDecl {
    /// No `grad_accumulation=` in the train block's config at all.
    Omitted,
    /// Present as an integer literal — the window is what it says.
    Literal,
}

/// Item 12: result of analyzing whether a train-loop callback body touches
/// the streamed model θ (see `Compiler::analyze_callback_model_touch`).
#[derive(Default, Debug)]
struct CallbackModelTouch {
    /// The callback references the model (a field read, a method call, an
    /// `Ident` passed to `model_save`/a helper, …). Requires a scoped upload
    /// under `--weight-stream` or its reads launch on evicted (null) data.
    touches: bool,
    /// The callback may MUTATE θ (an assignment rooted at the model, or a
    /// method call on the model/one of its fields). Drives writeback=1 on the
    /// closing re-evict so the mutation survives the next window's upload.
    may_write: bool,
    /// First model-rooted access path seen, for the compile-time diagnostic.
    first_path: Option<String>,
}

/// Item 11 calibration: fixed DMA issue+completion latency (μs) added to every
/// pack-transfer estimate — small PCIe copies are latency-bound, not
/// bandwidth-bound, so a bytes/BW model alone would price a 4 KiB pack at
/// ~0.1 μs and activate overlap that cannot pay. Combined with the target
/// `GpuSpec`'s `pcie_bandwidth_gbps` / `kernel_launch_overhead_ns` /
/// `peak_bandwidth_gbs`, this closes the deferred WGGO-ILP cost integration:
/// each prefetch edge is priced per-range compute μs vs pack-byte transfer μs.
pub(crate) const WS_PCIE_FIXED_LAT_US: f64 = 10.0;

/// Fallback for UNPRICED packs (a member's shape is not statically concrete —
/// e.g. a bare `Tensor` field annotation): the v1 structural heuristic, which
/// the GPU bit-exactness gates shipped and validated under. A priced edge is
/// calibrated-safe; an unpriced edge is merely heuristic (review M3: never
/// treat a 0-byte pricing as a real transfer estimate).
pub(crate) const WS_PREFETCH_MIN_OPS_PER_RANGE: usize = 4;

/// `pub(crate)` for the drift gate in `source_ad.rs`: constant-folding a
/// config field's `.item()` is sound only for leaves this function keeps OUT
/// of the parameter list, and that implication is asserted against the real
/// function rather than a restatement of it.
pub(crate) fn is_trainable_param_leaf_name(param_name: &str) -> bool {
    let leaf_name = param_name.rsplit('.').next().unwrap_or(param_name);
    !leaf_name.starts_with('_') && leaf_name != "inv_freq"
}

/// CFTP v10 (item 5): return the declared tensor rank of `ty` when it is
/// unambiguously a tensor with a non-empty shape.
///
/// `nsl_semantic::types::Shape::unknown()` and `Shape::scalar()` both
/// produce `Shape { dims: vec![] }`, so we cannot distinguish an
/// unannotated `Tensor` from a genuine rank-0 scalar tensor.  We treat
/// empty shape as UNKNOWN (`None`) so the matcher preserves its
/// conservative-fire behaviour for unannotated code — the load-bearing
/// rank check runs only when the frontend gave us a rank ≥ 1 to check.
/// A `Borrow(Tensor)` is unwrapped so annotated `&Tensor<[V,H]>`
/// parameters (common for `W` in NSL step signatures) participate too.
pub(crate) fn resolvable_tensor_rank(ty: &Type) -> Option<usize> {
    let inner = match ty {
        Type::Borrow(inner) => inner.as_ref(),
        other => other,
    };
    let rank = match inner {
        Type::Tensor { shape, .. }
        | Type::Param { shape, .. }
        | Type::Buffer { shape, .. }
        | Type::Sparse { shape, .. } => shape.rank(),
        _ => return None,
    };
    if rank == 0 {
        None
    } else {
        Some(rank)
    }
}

/// Allowlist of legal `data:` section config keys. Mirrors
/// `nsl-semantic/src/checker/block.rs::DATA_SECTION_KEYS` — both must stay
/// in sync. A key present here but missing in the semantic table will reach
/// `compile_assign` and fail with an undefined-variable error; a key
/// present in the semantic table but missing here will reach
/// `compile_assign` with the same failure mode. v8 ships with a single
/// canonical key (`source`); future keys should land in both places.
const DATA_SECTION_KEYS: &[&str] = &["source"];

/// Returns true iff `stmt` is a `data:` section config pair of the form
/// `<allowlisted-key> = <expr>` (plain `Assign`, plain ident target). These
/// are PCA-detection metadata consumed via the AST walker in
/// `pca_activation.rs`; they must not be lowered as variable assignments.
pub(crate) fn is_data_section_config_pair(stmt: &Stmt, interner: &nsl_lexer::Interner) -> bool {
    let StmtKind::Assign {
        target,
        op: AssignOp::Assign,
        ..
    } = &stmt.kind
    else {
        return false;
    };
    let ExprKind::Ident(name_sym) = target.kind else {
        return false;
    };
    let name = match interner.resolve(name_sym.0) {
        Some(n) => n,
        None => return false,
    };
    DATA_SECTION_KEYS.contains(&name)
}

/// Dev Tools Phase 4 Task 4: extract a layer index from a parameter path.
/// Finds the last numeric segment (e.g. "blocks.3.attn.wq" -> 3).  Returns
/// `u32::MAX` when no numeric segment is present.
pub(crate) fn parse_layer_idx_for_health(path: &str) -> u32 {
    path.split('.')
        .rev()
        .find_map(|seg| seg.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}

pub(crate) fn classify_source_ad_param_name(
    param_name: &str,
    tensor_param_paths: &std::collections::HashSet<String>,
) -> SourceAdParamDiagnosticKind {
    if tensor_param_paths.contains(param_name) {
        if is_trainable_param_leaf_name(param_name) {
            SourceAdParamDiagnosticKind::Trainable
        } else {
            SourceAdParamDiagnosticKind::IgnoredConfig
        }
    } else {
        SourceAdParamDiagnosticKind::IgnoredNonTensor
    }
}

impl Compiler<'_> {
    /// Compile one statement. An error raised anywhere beneath it — by this
    /// dispatcher, an expression, or a helper that never sees a span — leaves
    /// here pointing at the innermost statement or expression that was being
    /// compiled (`CodegenError::with_span_if_unset`: the first node on the
    /// way out to attach a span wins, so nested statements keep theirs).
    pub fn compile_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        self.compile_stmt_dispatch(builder, state, stmt)
            .map_err(|e| e.with_span_if_unset(stmt.span))
    }

    fn compile_stmt_dispatch(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        if let Some(block) = state.current_block
            && is_block_filled(builder, block)
        {
            return Ok(());
        }

        // Clear any stale lambda capture count from a previous statement
        // (only VarDecl should consume this; if it leaks past a statement boundary it's a bug)
        self.registry.last_lambda_capture_count = None;

        // ELTLS v2a: dispatch-fresh membership is a PER-STATEMENT fact
        // (see TensorCleanupState::dispatch_fresh) — a value that survives
        // into the next statement is variable-bound, and SSA use_var can
        // hand a later dispatch that very Value; tracking it there would
        // free the live binding.
        state.cleanup.dispatch_fresh.clear();

        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                match &pattern.kind {
                    PatternKind::Ident(sym) => {
                        let sym = *sym;
                        if let Some(expr) = value
                            && self.expr_is_borrowed_batch_handle(state, expr)
                        {
                            return Err(CodegenError::new(format!(
                                "cannot bind DataLoader batch handle '{}' directly; access batch fields instead",
                                self.resolve_sym(sym)
                            )));
                        }
                        let init_val = if let Some(expr) = value {
                            // M36: Check if this variable is slab-planned for zero-alloc
                            let slab_result =
                                self.try_compile_slab_tensor(builder, state, &sym, expr);
                            match slab_result {
                                Ok(Some(val)) => val,                          // Slab allocation succeeded
                                _ => self.compile_expr(builder, state, expr)?, // Normal path
                            }
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };

                        let cl_type = if let Some(expr) = value {
                            let nsl_ty = self.node_type(expr.id).clone();
                            // M56 Task 18: when the semantic pass returns Error/Unknown
                            // (e.g. for vars in @pipeline_agent bodies where agent
                            // bindings are synthesised at codegen time), fall back to
                            // the actual Cranelift type of the compiled Value so that
                            // variable declaration never mismatches the init_val type.
                            if matches!(
                                nsl_ty,
                                nsl_semantic::types::Type::Unknown
                                    | nsl_semantic::types::Type::Error
                            ) {
                                builder.func.dfg.value_type(init_val)
                            } else {
                                nsl_type_to_cl(&nsl_ty)
                            }
                        } else {
                            cl_types::I64
                        };

                        if let Some((var, _)) = state.variables.get(&sym).copied() {
                            if state.dataloader_symbols.contains(&sym) {
                                return Err(CodegenError::new(format!(
                                    "redeclaring DataLoader handle '{}' is unsupported; use a fresh symbol instead",
                                    self.resolve_sym(sym)
                                )));
                            }
                            // ELTLS §6.5: unified slot clear before rebind.
                            self.eltls_clear_old_slot(builder, state, sym);
                            // Dict twin of the slot clear: scan-admitted
                            // loop-body dict locals free the previous
                            // iteration's dict (values + structure) here.
                            // The tensor clear above no-ops on them (their
                            // Dict type fails its filter and they are not
                            // in eltls_loop_predeclared); iteration one
                            // frees the predeclared 0, a no-op. Keyed off
                            // dict_loop_predeclared — slots the predeclare
                            // actually created — NOT the plan set: a decl
                            // shadowing a function parameter is in the
                            // plan but keeps the caller's slot (review
                            // HIGH-1 on d114b5d7, reproduced corruption).
                            if state.dict_loop_predeclared.contains(&sym) {
                                let old_val = builder.use_var(var);
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_dict_free_tensor_values",
                                    &[old_val],
                                );
                            }
                            builder.def_var(var, init_val);
                        } else {
                            let var = builder.declare_var(cl_type);
                            builder.def_var(var, init_val);
                            state.variables.insert(sym, (var, cl_type));
                        }

                        // Record semantic type for step-variable cleanup
                        if let Some(expr) = value {
                            state
                                .variable_types
                                .insert(sym, self.node_type(expr.id).clone());
                            // Function-valued binding (lambda, fn alias,
                            // call returning a fn): register scoped
                            // liveness for the shadow-dispatch guard.
                            if matches!(
                                self.node_type(expr.id),
                                nsl_semantic::types::Type::Function { .. }
                            ) {
                                state.register_fn_binding(sym);
                            }
                            if self.expr_is_dataloader_handle(state, expr) {
                                state.dataloader_symbols.insert(sym);
                            } else {
                                state.dataloader_symbols.remove(&sym);
                            }
                        } else {
                            state.dataloader_symbols.remove(&sym);
                        }
                        self.update_non_owning_binding(state, sym, value.as_ref());

                        // M50: Track sparse tensor variables for end-to-end dispatch.
                        // Check if the RHS is a call to a sparse function or has Type::Sparse.
                        if let Some(expr) = value {
                            let is_sparse_type = matches!(
                                self.node_type(expr.id),
                                nsl_semantic::types::Type::Sparse { .. }
                            );
                            let is_sparse_call = if let ExprKind::Call { callee, .. } = &expr.kind {
                                if let ExprKind::Ident(fn_sym) = &callee.kind {
                                    let fn_name = self.resolve_sym(*fn_sym);
                                    fn_name.contains("sparse") || fn_name == "from_dense"
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if is_sparse_type || is_sparse_call {
                                state.ownership.sparse_vars.insert(sym);
                            }
                        }

                        // Free intermediate tensor temporaries (keep init_val which is now owned by the variable)
                        self.free_tensor_temporaries(builder, state, Some(init_val));
                        // M38b: Free linear tensors consumed during this let-binding's RHS
                        self.free_linear_consumes(builder, state, Some(init_val));

                        // If the value was a closure lambda, record capture count for indirect call dispatch
                        if let Some(count) = self.registry.last_lambda_capture_count.take() {
                            state.set_closure_info(sym, Some(count));
                        } else {
                            // A rebind to anything that is NOT a capturing
                            // lambda must clear the stale entry, or the call
                            // site reads the new bare function pointer as a
                            // closure struct (probed: silent death). The
                            // scoped setter records an undo entry so a
                            // block-local rebind cannot outlive its scope
                            // (review HIGH on cb1bd16f: a dead if-arm's
                            // rebind deleted the outer closure's entry).
                            state.set_closure_info(sym, None);
                        }
                    }
                    PatternKind::Tuple(sub_patterns) | PatternKind::List(sub_patterns) => {
                        let tuple_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new(
                                "tuple/list destructuring requires a value",
                            ));
                        };
                        let tuple_ty = value.as_ref().map(|expr| self.node_type(expr.id).clone());

                        self.compile_destructure_patterns(
                            builder,
                            state,
                            sub_patterns,
                            tuple_val,
                            tuple_ty.as_ref(),
                        )?;
                        // Review F2 on 67b9ba13: destructuring arms had no
                        // statement-end drain — a sub-expression temp (the
                        // inner `t * 2.0` of a tuple element `t * 2.0 + 1.0`;
                        // the element itself transfers into the tuple)
                        // straddled into the train step loop and was freed
                        // once per step. The destructured value is kept
                        // defensively: if an indeterminate-typed RHS ever
                        // lands it in the list, freeing a list pointer via
                        // nsl_tensor_free would abort on the magic probe.
                        self.free_tensor_temporaries(builder, state, Some(tuple_val));
                        self.free_linear_consumes(builder, state, Some(tuple_val));
                    }
                    PatternKind::Struct { fields, .. } => {
                        // Top-level struct destructuring: let { x, y } = expr
                        let struct_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new("struct destructuring requires a value"));
                        };
                        let struct_ty = value.as_ref().map(|expr| self.node_type(expr.id).clone());
                        for field in fields {
                            let field_name = self.resolve_sym(field.name).to_string();
                            if !self.string_pool.contains_key(field_name.as_str()) {
                                self.intern_string(&field_name)?;
                            }
                            let key_str = self.compile_string_literal(builder, &field_name)?;
                            let field_ty =
                                self.destructure_field_type(struct_ty.as_ref(), field.name);
                            let mut field_val = self.compile_call_by_name(
                                builder,
                                "nsl_dict_get_str",
                                &[struct_val, key_str],
                            )?;
                            if field_ty.as_ref().map(|ty| ty.is_tensor()).unwrap_or(false) {
                                field_val = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_clone",
                                    &[field_val],
                                )?;
                            }
                            if let Some(ref pat) = field.pattern {
                                match &pat.kind {
                                    PatternKind::Ident(sym) => {
                                        let var = builder.declare_var(cl_types::I64);
                                        builder.def_var(var, field_val);
                                        state.variables.insert(*sym, (var, cl_types::I64));
                                        if let Some(field_ty) = field_ty.clone() {
                                            state.variable_types.insert(*sym, field_ty);
                                        }
                                    }
                                    PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                                        self.compile_destructure_patterns(
                                            builder,
                                            state,
                                            nested,
                                            field_val,
                                            field_ty.as_ref(),
                                        )?;
                                    }
                                    PatternKind::Wildcard => {}
                                    _ => {
                                        return Err(CodegenError::new(format!(
                                            "unsupported pattern in struct field '{}'",
                                            field_name
                                        )));
                                    }
                                }
                            } else {
                                let var = builder.declare_var(cl_types::I64);
                                builder.def_var(var, field_val);
                                state.variables.insert(field.name, (var, cl_types::I64));
                                if let Some(field_ty) = field_ty {
                                    state.variable_types.insert(field.name, field_ty);
                                }
                            }
                        }
                        // Review F2 on 67b9ba13 — same statement-end drain as
                        // the tuple/list destructure arm above; the bound
                        // field values are clones (never listed) and the
                        // struct value itself is kept defensively.
                        self.free_tensor_temporaries(builder, state, Some(struct_val));
                        self.free_linear_consumes(builder, state, Some(struct_val));
                    }
                    _ => {
                        return Err(CodegenError::new(
                            "only ident, tuple, list, and struct patterns supported",
                        ))
                    }
                }
            }

            StmtKind::Assign { target, op, value } => {
                self.compile_assign(builder, state, target, *op, value)?;
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    if self.expr_is_dataloader_handle(state, e) {
                        return Err(CodegenError::new(
                            "cannot return a DataLoader handle directly; create and consume loaders within the same function",
                        ));
                    }
                    if self.expr_is_borrowed_batch_handle(state, e) {
                        return Err(CodegenError::new(
                            "cannot return a DataLoader batch handle directly; return batch fields instead",
                        ));
                    }
                    let mut val = self.compile_expr(builder, state, e)?;
                    // ELTLS §6.5: consult return-value ownership and emit the
                    // correct transfer/retain path for tensor-typed returns.
                    // Require Cranelift value to be I64 AND semantic type to be
                    // a real tensor (NOT indeterminate — BYOD dtype method
                    // returns have Unknown semantic type but are scalars).
                    // Also skip inside dtype methods entirely.
                    let ret_ty = self.node_type(e.id).clone();
                    let val_is_ptr = builder.func.dfg.value_type(val) == cl_types::I64;
                    if val_is_ptr && ret_ty.is_tensor() && !state.flags.in_dtype_method {
                        use crate::ownership_expr::Ownership;
                        // Upgrade Unknown to Owned when the return expr is a
                        // call contractually returning an owning ref (see
                        // expr_call_returns_owning_ref) — the conservative
                        // retain below would double-own it and strand one
                        // reference per call.
                        let mut own = self.get_ownership(state, val);
                        if matches!(own, Ownership::Unknown)
                            && self.expr_call_returns_owning_ref(e)
                        {
                            own = Ownership::Owned;
                        }
                        match own {
                            Ownership::Owned => {
                                self.consume_ownership(state, val);
                            }
                            Ownership::BorrowedFromVar(_) | Ownership::BorrowedWeight => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                            }
                            Ownership::TapeHeld => {
                                nsl_log::nsl_log!(WARN, "codegen", 
                                    "ELTLS warning: returning TapeHeld tensor — semantic error"
                                );
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                            }
                            Ownership::Unknown => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                                self.note_unknown_fallback(state, val);
                            }
                        }
                    }
                    // Free intermediate tensor temporaries before returning (keep return value)
                    self.free_tensor_temporaries(builder, state, Some(val));
                    // M38b: Free linear tensors consumed during the return expression
                    self.free_linear_consumes(builder, state, Some(val));
                    // Free non-parameter tensor locals (see emit_return_local_sweep).
                    //
                    // Tensor returns: safe because the retain above compensates
                    // for the returned value aliasing a swept local.
                    //
                    // SCALAR returns: also swept, and previously missed. Note
                    // what the return type does and does not buy us. A
                    // `-> f64` / `-> int` / `-> bool` result is a value, not a
                    // handle, so it cannot alias a swept local and needs no
                    // retain to protect it. The precondition the SWEEP itself
                    // needs is different and orthogonal: no swept local may
                    // have ESCAPED. That is carried by `non_owning_symbols`
                    // (member reads, borrowed batch handles and — since the
                    // 2026-07-24 review — non-Dict subscripts are all marked
                    // non-owning and skipped), exactly as it already is for
                    // the `-> void` and fall-through return paths this merely
                    // brings into line. Skipping the sweep here stranded one
                    // tensor per call for the extremely common
                    // `fn loss(...) -> f64: let d = a - b; return sum(d*d).item()`
                    // shape (measured: live_blocks 5 -> 11 over 3 -> 9 calls,
                    // while the identical `-> void` twin stayed flat at 2).
                    //
                    // Aggregate returns (list/dict/tuple/str/model) stay
                    // excluded: they can alias a local without a retain.
                    let ret_is_scalar = matches!(
                        ret_ty,
                        Type::Int
                            | Type::Float
                            | Type::Bool
                            | Type::F32
                            | Type::F64
                            | Type::Int8
                            | Type::Int16
                            | Type::Int32
                            | Type::Int64
                            | Type::Uint8
                    );
                    if (ret_is_scalar || (val_is_ptr && ret_ty.is_tensor()))
                        && !state.flags.in_dtype_method
                        && !state.flags.in_tape_region
                    {
                        self.emit_return_local_sweep(builder, state);
                    }
                    self.cleanup_active_loop_batches(builder, state);
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.flags.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    // In element-wise unpack methods, bitcast f64→i64 for the return value
                    if state.flags.dtype_unpack_ret_bitcast {
                        let vt = builder.func.dfg.value_type(val);
                        if vt == cranelift_codegen::ir::types::F64 {
                            val = builder.ins().bitcast(
                                cranelift_codegen::ir::types::I64,
                                cranelift_codegen::ir::MemFlagsData::new(),
                                val,
                            );
                        }
                    }
                    builder.ins().return_(&[val]);
                } else {
                    // Bare `return`: same local sweep as the implicit path.
                    if !state.flags.in_dtype_method && !state.flags.in_tape_region {
                        self.emit_return_local_sweep(builder, state);
                    }
                    self.cleanup_active_loop_batches(builder, state);
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.flags.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    builder.ins().return_(&[]);
                }
            }

            StmtKind::Expr(expr) => {
                // compile_nested_expr so a discarded owning result (bare
                // `user_fn(x)`, `y.sum()`) registers as a temporary and the
                // sweep below frees it.
                let _ = self.compile_nested_expr(builder, state, expr)?;
                // Free all tensor temporaries from this expression (none are kept)
                self.free_tensor_temporaries(builder, state, None);
                // M38b: Free linear tensors consumed during this expression statement
                self.free_linear_consumes(builder, state, None);
            }

            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.compile_if_stmt(
                    builder,
                    state,
                    condition,
                    then_block,
                    elif_clauses,
                    else_block,
                )?;
            }

            StmtKind::While { condition, body } => {
                self.compile_while(builder, state, condition, body)?;
            }

            StmtKind::For {
                pattern,
                iterable,
                body,
            } => {
                self.compile_for(builder, state, pattern, iterable, body)?;
            }

            StmtKind::Match { subject, arms } => {
                self.compile_match(builder, state, subject, arms)?;
            }

            StmtKind::Break => {
                let exit = state
                    .loop_stack
                    .last()
                    .map(|lc| lc.exit_block)
                    .ok_or_else(|| CodegenError::new("break outside loop"))?;
                // Free tensor temporaries from current loop iteration before jumping out
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(exit, &[]);
            }

            StmtKind::Continue => {
                let cont = state
                    .loop_stack
                    .last()
                    .map(|lc| lc.continue_block)
                    .ok_or_else(|| CodegenError::new("continue outside loop"))?;
                // Free tensor temporaries from current loop iteration before restarting
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(cont, &[]);
            }

            StmtKind::FnDef(fn_def) => {
                // Nested function definition: declare, compile, and bind name
                let base_name = self.resolve_sym(fn_def.name).to_string();
                let unique_name = format!("__nsl_nested_{}_{}", base_name, self.next_func_index());
                let sig = self.build_fn_signature(fn_def);
                let func_id = self
                    .module
                    .declare_function(&unique_name, cranelift_module::Linkage::Local, &sig)
                    .map_err(|e| {
                        CodegenError::new(format!("failed to declare nested fn '{base_name}': {e}"))
                    })?;
                // Temporarily insert under base_name for compile_fn_def lookup, then restore
                let prev_entry = self.registry.functions.remove(&base_name);
                self.registry
                    .functions
                    .insert(base_name.clone(), (func_id, sig.clone()));

                // Compile the nested function body. closure_info now lives
                // on FuncState (per-function), so the nested body's own
                // closure metadata cannot touch this function's — the
                // earlier compiler-global map needed a snapshot here
                // (review HIGH on 44c011c1).
                self.compile_fn_def(fn_def)?;

                // Remove temp entry and restore any previous function with the same name
                self.registry.functions.remove(&base_name);
                if let Some(prev) = prev_entry {
                    self.registry.functions.insert(base_name, prev);
                }

                // Bind function name as a variable holding the function pointer
                let func_ref = self.module.declare_func_in_func(func_id, builder.func);
                let addr = builder
                    .ins()
                    .func_addr(crate::types::pointer_type(), func_ref);
                let var = builder.declare_var(cl_types::I64);
                builder.def_var(var, addr);
                state.variables.insert(fn_def.name, (var, cl_types::I64));
                // Scoped liveness for the shadow-dispatch guard — the
                // flat `variables` entry above never unbinds, but the
                // checker scopes this fn to the enclosing block.
                state.register_fn_binding(fn_def.name);
            }

            StmtKind::GradBlock(grad) => {
                self.compile_grad_block(builder, state, grad)?;
            }

            StmtKind::TrainBlock(train) => {
                // CFTP v10 (item 3): thread the enclosing `Stmt.id` so
                // `compile_train_block` can look up its `@fused_lm_ce`
                // config by AST NodeId instead of hitting
                // `fused_ce_configs.first()`.
                self.compile_train_block(builder, state, train, stmt.id)?;
            }

            StmtKind::DistillBlock(distill) => {
                // CPKD: distillation training loop with a structurally
                // frozen teacher (I-11); delegates into the train-block
                // lowering with an `active_distill_context` installed.
                self.compile_distill_block(builder, state, distill, stmt.id)?;
            }

            StmtKind::StructDef(_)
            | StmtKind::ModelDef(_)
            | StmtKind::EnumDef(_)
            | StmtKind::TraitDef(_)
            | StmtKind::Import(_)
            | StmtKind::FromImport(_)
            | StmtKind::DatasetDef(_)
            | StmtKind::TokenizerDef(_)
            // M56 Task 17: agent declarations are compiled by the dedicated
            // collect_agents / declare_agent_methods / compile_agent_methods
            // passes in entry_points.rs — not inline in stmt compilation.
            | StmtKind::AgentDef(_) => {}

            StmtKind::DatatypeDef(_) => {
                // M23: custom datatype codegen — implemented in Task 9
            }

            StmtKind::ServeBlock(serve) => {
                self.compile_serve_block(builder, state, serve)?;
            }

            StmtKind::KernelDef(_) => {
                // Kernels are compiled in the compile_kernels pass (before functions).
            }

            StmtKind::QuantBlock(quant) => {
                self.compile_quant_block(builder, state, quant)?;
            }

            StmtKind::WhileLet {
                pattern,
                expr,
                body,
            } => {
                self.compile_while_let(builder, state, pattern, expr, body)?;
            }

            StmtKind::Decorated { decorators, stmt } => {
                // Module-scoped decorator configs that apply regardless of
                // the inner stmt kind (i.e. not FnDef-specific). `@cpdt`
                // wraps a TrainBlock but the `weight_aware` kwarg is global
                // compiler state: nsl-semantic enforces exactly-one-@cpdt-
                // per-program so the single-writer semantics are safe.
                // See docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.
                for d in decorators {
                    if d.name.len() == 1 && self.resolve_sym(d.name[0]) == "cpdt"
                        && let Some(args) = &d.args
                    {
                        for arg in args {
                            if let Some(name_sym) = arg.name
                                && self.resolve_sym(name_sym) == "weight_aware"
                                && let nsl_ast::expr::ExprKind::BoolLiteral(b) =
                                    arg.value.kind
                            {
                                self.cpdt_weight_aware = b;
                            }
                        }
                    }

                    // CFIE Tier-A wiring (audit gap G4): capture
                    // `@cfie(mode=..., target=...)` on a serve block so
                    // `compile_serve_block` consumes it instead of the
                    // config being validated-then-dropped.
                    if d.name.len() == 1
                        && self.resolve_sym(d.name[0]) == "cfie"
                        && matches!(stmt.kind, StmtKind::ServeBlock(_))
                    {
                        // A bare `@cfie` means "enable, full mode".
                        self.cfie_decorator_mode = Some(crate::cfie::CfieMode::Full);
                        if let Some(args) = &d.args {
                            for arg in args {
                                let Some(name_sym) = arg.name else { continue };
                                let aname = self.resolve_sym(name_sym).to_string();
                                match (aname.as_str(), &arg.value.kind) {
                                    ("mode", nsl_ast::expr::ExprKind::Ident(sym)) => {
                                        let m = self.resolve_sym(*sym).to_string();
                                        self.cfie_decorator_mode =
                                            crate::cfie::CfieMode::parse(&m);
                                    }
                                    ("mode", nsl_ast::expr::ExprKind::StringLiteral(s)) => {
                                        self.cfie_decorator_mode =
                                            crate::cfie::CfieMode::parse(s);
                                    }
                                    ("target", nsl_ast::expr::ExprKind::Ident(sym)) => {
                                        self.cfie_decorator_target =
                                            Some(self.resolve_sym(*sym).to_string());
                                    }
                                    ("target", nsl_ast::expr::ExprKind::StringLiteral(s)) => {
                                        self.cfie_decorator_target = Some(s.clone());
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    // Milestone A: capture `@fase(...)` on a train block so
                    // the FASE planner consumes it instead of the config
                    // being validated-then-dropped (the checker's arm at
                    // nsl-semantic/checker/stmt.rs validated and discarded
                    // it; same defect class as the @cfie gap above).
                    if d.name.len() == 1
                        && self.resolve_sym(d.name[0]) == "fase"
                        && matches!(stmt.kind, StmtKind::TrainBlock(_))
                    {
                        let interner = self.interner;
                        let resolve = |s: nsl_ast::Symbol| -> String {
                            interner.resolve(s.0).unwrap_or("").to_string()
                        };
                        let mut diags = Vec::new();
                        if let Some(cfg) = nsl_semantic::cftp::validate_fase_decorator(
                            d, &resolve, &mut diags,
                        ) {
                            // Invalid configs never get here: nsl-semantic
                            // already failed the compile during analysis.
                            if diags.is_empty() {
                                self.fase_decorator = Some(cfg);
                            }
                        }
                    }
                }

                // Check for @no_grad and @fuse on nested function definitions
                if let StmtKind::FnDef(fn_def) = &stmt.kind {
                    for d in decorators {
                        if d.name.len() == 1 {
                            let dname = self.resolve_sym(d.name[0]);
                            if dname == "no_grad" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.registry.no_grad_fns.insert(fname);
                            } else if dname == "fp8_compute" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.features.fp8_compute_fns.insert(fname);
                            } else if dname == "fuse" {
                                self.validate_fuse_body(fn_def)?;
                                // Extract the op chain from the function body's return expression
                                // and register it for fused kernel launch at call sites.
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                let num_params = fn_def.params.len();
                                if let Some(ret_expr) =
                                    fn_def.body.stmts.iter().rev().find_map(|s| match &s.kind {
                                        StmtKind::Return(Some(e)) => Some(e),
                                        StmtKind::Expr(e) => Some(e),
                                        _ => None,
                                    })
                                {
                                    let interner = self.interner;
                                    let resolve = |sym: nsl_ast::Symbol| -> Option<String> {
                                        interner.resolve(sym.0).map(|s| s.to_string())
                                    };
                                    if let Some((ops, _inputs)) =
                                        crate::fusion::analyze_fusible_chain(ret_expr, &resolve)
                                        && ops.len() >= 2
                                    {
                                        self.fusion
                                            .fused_fns
                                            .insert(fname.clone(), (ops, num_params));
                                    }
                                }
                                // Still compile the function normally as fallback (CPU or when
                                // fusion is disabled). The fused path is selected at call site.
                            } else if dname == "grammar" {
                                // M44: @grammar decorator on nested function
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                let mut start_rule = String::new();
                                let mut grammar_source = String::new();
                                if let Some(ref dargs) = d.args {
                                    for arg in dargs {
                                        if let Some(name_sym) = arg.name {
                                            let arg_name = self.resolve_sym(name_sym).to_string();
                                            if arg_name == "start_rule"
                                                && let nsl_ast::expr::ExprKind::StringLiteral(s) =
                                                    &arg.value.kind
                                            {
                                                start_rule = s.clone();
                                            }
                                        } else if let nsl_ast::expr::ExprKind::StringLiteral(s) =
                                            &arg.value.kind
                                        {
                                            grammar_source = s.clone();
                                        }
                                    }
                                }
                                self.features.grammar_configs.insert(
                                    fname,
                                    crate::compiler::GrammarInfo {
                                        start_rule,
                                        grammar_source,
                                    },
                                );
                            }
                        }
                    }
                }
                self.compile_stmt(builder, state, stmt)?;

                // Phase 5 Task 7: after the inner VarDecl has bound the
                // target, emit @inspect hooks.  Only active when
                // `compile_options.dev_tools.inspect_enabled` is true and the stmt
                // is a `let x = ...`.
                if self.compile_options.dev_tools.inspect_enabled
                    && let StmtKind::VarDecl { pattern, .. } = &stmt.kind
                    && let PatternKind::Ident(target_sym) = &pattern.kind
                {
                    for d in decorators {
                        if d.name.len() == 1
                            && self.resolve_sym(d.name[0]) == "inspect"
                        {
                            self.emit_inspect_hook(builder, state, d, *target_sym)?;
                        }
                    }
                }
            }

            _ => {
                return Err(CodegenError::new(
                    "unsupported statement in M3 codegen".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// ELTLS Task 16.1: pre-scan a loop body block for top-level VarDecl
    /// statements that have a simple Ident target, returning the list of
    /// symbols. These symbols are candidates for pre-declaration in the
    /// pre-loop scope so that their second-and-later rebinds are detected
    /// as reassignments (firing eltls_clear_old_slot and freeing the
    /// previous iteration's value).
    ///
    /// Conservative: does NOT filter by type here — nsl_tensor_free_if_valid
    /// handles non-tensor values as runtime no-ops. Does NOT descend into
    /// nested blocks (those are separate scopes and won't share state).
    /// Skips Decorated VarDecls by unwrapping one level of Decorated.
    pub(crate) fn eltls_collect_loop_let_idents(
        &self,
        stmts: &[nsl_ast::stmt::Stmt],
    ) -> Vec<nsl_ast::Symbol> {
        let mut out = Vec::new();
        for stmt in stmts {
            let inner = match &stmt.kind {
                nsl_ast::stmt::StmtKind::Decorated { stmt, .. } => &stmt.kind,
                other => other,
            };
            if let nsl_ast::stmt::StmtKind::VarDecl { pattern, value, .. } = inner {
                // Only top-level simple Ident targets
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    // Skip declarations with no initializer — nothing useful
                    // to free, and they wouldn't trigger eltls anyway.
                    if value.is_some() {
                        out.push(*sym);
                    }
                }
            }
        }
        out
    }

    /// ELTLS Task 16.1: pre-declare the collected loop-body let-ident
    /// symbols in the current (pre-loop) scope. Each slot is initialized
    /// to an i64 zero, which nsl_tensor_free_if_valid treats as a no-op.
    /// Also records the symbol in state.eltls_loop_predeclared so that
    /// eltls_clear_old_slot unlocks the slot-free path without requiring
    /// a variable_types entry (which is only recorded once the VarDecl
    /// has been compiled).
    ///
    /// Skips symbols already present in state.variables (parameter,
    /// outer-scope let, etc.) — those are not loop-local rebinds and
    /// should not be touched here.
    /// Returns the symbols ACTUALLY inserted — callers must remove exactly
    /// these from state.eltls_loop_predeclared after the body compiles.
    /// Returning skipped (already-declared) syms would let an inner
    /// same-name loop's removal strip a sym the OUTER loop armed.
    pub(crate) fn eltls_predeclare_loop_lets(
        &self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        syms: &[nsl_ast::Symbol],
    ) -> Vec<nsl_ast::Symbol> {
        let mut inserted = Vec::new();
        if syms.is_empty() {
            return inserted;
        }
        let zero = builder.ins().iconst(cl_types::I64, 0);
        for &sym in syms {
            if state.variables.contains_key(&sym) {
                continue;
            }
            if state.param_symbols.contains(&sym) {
                continue;
            }
            let var = builder.declare_var(cl_types::I64);
            builder.def_var(var, zero);
            state.variables.insert(sym, (var, cl_types::I64));
            state.eltls_loop_predeclared.insert(sym);
            inserted.push(sym);
        }
        inserted
    }

    /// Collect + ownership-vet + predeclare loop-body let symbols in one
    /// step. This is the only entry point loop lowerings should use.
    ///
    /// Predeclaring a symbol activates a loop-top free of the loop-carried
    /// value at its first rebind (eltls_clear_old_slot). That is only sound
    /// when EVERY binding of the symbol anywhere in the loop body yields an
    /// OWNING reference: ident copies and member reads hand out the
    /// referent's own pointer with no retain, and freeing a loop-carried
    /// borrow frees the referent itself (a model weight, another local's
    /// tensor). Symbols with any unowned binding are left out — they keep
    /// the old declare-in-body behavior (worst case a status-quo strand,
    /// never a double-free).
    ///
    /// Returns the predeclared symbols; callers must remove them from
    /// state.eltls_loop_predeclared after compiling the body.
    pub(crate) fn eltls_predeclare_loop_lets_checked(
        &self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        body: &nsl_ast::stmt::Block,
    ) -> Vec<nsl_ast::Symbol> {
        let syms: Vec<_> = self
            .eltls_collect_loop_let_idents(&body.stmts)
            .into_iter()
            .filter(|s| self.sym_bindings_all_owning_in_block(body, *s))
            .collect();
        let inserted = self.eltls_predeclare_loop_lets(builder, state, &syms);

        // Dict twin (dict_lifetime.rs): zero-predeclare scan-admitted
        // loop-body dict locals whose decl sits at THIS body's direct
        // level, so the rebind clear and the return sweep always see a
        // defined slot (0 on iteration one / when the loop never runs —
        // free_dict_impl no-ops on 0). Deliberately NOT recorded in
        // eltls_loop_predeclared: that set arms the TENSOR clear, whose
        // free_if_valid must keep no-oping on dict handles; the dict
        // rebind clear keys off state.dict_loop_predeclared, written
        // below only when this pass creates the slot. The scan only
        // admits decls in top-level loops, so this emission point
        // dominates every later return.
        let dict_syms: Vec<_> = body
            .stmts
            .iter()
            .filter_map(|s| {
                let nsl_ast::stmt::StmtKind::VarDecl { pattern, .. } = &s.kind else {
                    return None;
                };
                let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind else {
                    return None;
                };
                (state.dict_loop_rebind.contains(sym)
                    && !state.variables.contains_key(sym)
                    && !state.param_symbols.contains(sym))
                .then_some(*sym)
            })
            .collect();
        if !dict_syms.is_empty() {
            let zero = builder.ins().iconst(cl_types::I64, 0);
            for sym in dict_syms {
                let var = builder.declare_var(cl_types::I64);
                builder.def_var(var, zero);
                state.variables.insert(sym, (var, cl_types::I64));
                // The rebind clear keys off this set — ONLY slots this
                // predeclare created may be dict-freed at rebind. A
                // plan sym skipped here (e.g. it shadows a function
                // parameter, whose slot already exists and holds the
                // caller's value) must never reach the clear (review
                // HIGH-1 on d114b5d7).
                state.dict_loop_predeclared.insert(sym);
            }
        }
        inserted
    }

    /// ELTLS (spec §6.5): clear a tensor-typed variable's old value before
    /// reassignment. Emits nsl_tensor_free on the old pointer and purges all
    /// tracking queues (new AND legacy). Skipped for initial let bindings
    /// (symbol not yet in state.variables), parameters, non-owning symbols,
    /// borrowed batch handles, and non-tensor variables.
    ///
    /// Deliberately does NOT touch tape_held — if the old value had an active
    /// tape lease, the nsl_tensor_free here just decrements the variable-slot
    /// refcount and the tape's retained lease keeps the storage alive until
    /// free_tape_held_tensors runs at tape-region exit.
    pub(crate) fn eltls_clear_old_slot(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        sym: nsl_ast::Symbol,
    ) {
        // Dtype method bodies run on scalar-valued slots — do NOT emit
        // tensor frees on them, even if the slot type says indeterminate.
        if state.flags.in_dtype_method {
            return;
        }
        // Only act on reassignments: the symbol must already exist.
        let Some((var, cl_type)) = state.variables.get(&sym).copied() else {
            return; // initial let — slot uninitialized, use_var would be UB
        };
        if cl_type != cl_types::I64 {
            return; // non-tensor variable
        }
        if state.param_symbols.contains(&sym) {
            return; // parameter — caller owns it
        }
        if state.non_owning_symbols.contains(&sym) {
            return; // view or borrow alias
        }
        if state.borrowed_batch_symbols.contains(&sym) {
            return; // DataLoader batch handle — freed by loader teardown
        }
        if state.dataloader_symbols.contains(&sym) {
            return; // DataLoader handle itself
        }
        // Additional semantic filter: only emit free if the variable's type is
        // actually a tensor or indeterminate, OR the slot was pre-declared by
        // the loop-predeclare pass (which records the sym in
        // eltls_loop_predeclared). Values stored in I64 slots that are
        // integers, booleans, lists, or dicts should NOT be freed via
        // nsl_tensor_free — but nsl_tensor_free_if_valid handles them as
        // no-ops by probing the magic field, so loop-predeclared slots are
        // safe to unconditionally attempt the free on.
        let is_loop_predeclared = state.eltls_loop_predeclared.contains(&sym);
        if !is_loop_predeclared {
            match state.variable_types.get(&sym) {
                Some(ty) if ty.is_tensor() || ty.is_indeterminate() => {}
                _ => return,
            }
        }
        // Don't emit frees into a filled block.
        if let Some(block) = state.current_block
            && is_block_filled(builder, block)
        {
            return;
        }
        // Read the old value and free it via the safe variant that handles
        // null pointers, invalid magic, and non-tensor i64 values as no-ops.
        // This is required for loop-pre-declared slots that start at zero on
        // the first iteration, and is a safer default for reassignment
        // paths generally (a stale non-tensor would otherwise crash).
        let old_val = builder.use_var(var);
        let _ = self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[old_val]);
        // Purge tracking queues so the statement/function cleanup paths don't
        // try to free this value again.
        state.cleanup.expr_ownership.remove(&old_val);
        state.cleanup.owned_temporaries.retain(|&v| v != old_val);
        state.cleanup.tensor_temporaries.retain(|&v| v != old_val);
        // DO NOT touch state.cleanup.tape_held — see doc comment above.
    }

    /// Free intermediate tensor temporaries accumulated during expression compilation.
    /// `keep` is the final result value that should NOT be freed (it's owned by a variable).
    /// All other temporaries are intermediates from compound expressions (e.g. `a + b` in `a + b + c`).
    /// Free all I64-typed non-parameter local variables (potential tensors)
    /// at a function-return point. Twin of the implicit-return sweep in
    /// func.rs — until 2026-07 only the fall-off-the-end path had it, so
    /// every `let`-bound tensor local in a function ending with an explicit
    /// `return <expr>` leaked its final reference (`mse_loss`'s `diff`, the
    /// residual-block `f`, the loop-carried `h` — the tape-mode per-step
    /// leak AND the @no_grad inference leak were this hole).
    ///
    /// Safe against the returned value aliasing a swept variable: the
    /// Return arm retains BorrowedFromVar/Unknown returns before calling
    /// this, so the variable's own free leaves the returned reference
    /// intact. `nsl_tensor_free_if_valid` skips non-tensor pointers and
    /// already-poisoned boxes, so aliased vars double-swept in sequence
    /// are no-ops on the second hit.
    pub(crate) fn emit_return_local_sweep(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        // Name order: this sweep is a call per variable, and its order
        // is the function's text (see `variables_in_name_order`).
        let locals: Vec<_> = self
            .variables_in_name_order(state)
            .into_iter()
            .filter(|sym| !state.param_symbols.contains(sym))
            .filter(|sym| !state.non_owning_symbols.contains(sym))
            // Semantic-type filter: only tensor (or indeterminate) locals.
            // free_if_valid's pointer probes are NOT sufficient for plain
            // integers — a large 8-aligned int (e.g. a byte count from
            // gpu_peak_bytes()) passes the null/low/alignment checks and the
            // magic probe DEREFERENCES it, segfaulting on unmapped memory.
            .filter(|sym| {
                matches!(state.variable_types.get(sym),
                         Some(ty) if ty.is_tensor() || ty.is_indeterminate())
            })
            .filter_map(|sym| {
                let (var, cl_type) = state.variables[&sym];
                if cl_type == cl_types::I64 {
                    Some(var)
                } else {
                    None
                }
            })
            .collect();
        for var in locals {
            let val = builder.use_var(var);
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[val]);
        }

        // Dict-local pass (the aggregate-lifetime gap, 2026-07-28). A
        // tensor-valued dict local owns its stored tensors outright — every
        // read CLONES (compile_subscript's Dict arm), so no borrow of a
        // stored tensor exists anywhere and freeing the dict with its
        // values cannot free anything reachable. Only symbols the
        // conservative usage scan admitted are freed here (single top-level
        // call binding — so the slot dominates every return that can see it
        // in `state.variables` — subscript reads only, never
        // returned/passed/stored-into); see dict_lifetime.rs for the veto
        // rules. An unscanned body has an empty set: status quo, the dict
        // strands (leak, not crash).
        let dict_locals: Vec<_> = self
            .variables_in_name_order(state)
            .into_iter()
            .filter(|sym| state.sweepable_dict_locals.contains(sym))
            .filter(|sym| !state.param_symbols.contains(sym))
            .filter_map(|sym| {
                let (var, cl_type) = state.variables[&sym];
                if cl_type == cl_types::I64 {
                    Some(var)
                } else {
                    None
                }
            })
            .collect();
        for var in dict_locals {
            let val = builder.use_var(var);
            let _ =
                self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[val]);
        }
    }

    pub(crate) fn free_tensor_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        // Drain only the temporaries registered ABOVE the innermost loop-scope
        // mark. A `mem::take` of the whole list here stole entries that a loop
        // CONDITION / for-ITERABLE registered before `temp_scope_stack` pushed
        // its mark: the first body statement drained them (emitting their
        // frees INSIDE the body = once per iteration), left `scope_start`
        // pointing past the now-empty list, and the loop-exit cleanups'
        // `[scope_start..]` slices panicked at compile time — a pre-existing
        // ICE (method-form condition temps hit it on main) made trivially
        // reachable once nested-arg tracking covered every dispatch arm
        // (review HIGH on 3b7f085f, red-proven with a 6-line while loop).
        // Below-mark entries stay listed; the loop statement's own
        // statement-end cleanup frees the final condition evaluation's temp
        // after the loop exits.
        let start = state
            .cleanup
            .temp_scope_stack
            .last()
            .copied()
            .unwrap_or(0)
            .min(state.cleanup.tensor_temporaries.len());
        let temps = state.cleanup.tensor_temporaries.split_off(start);
        // Tape ID identity: intermediates can now be safely freed during tape recording
        // because TapeOps use monotonic tape_ids as identity keys (not raw pointers).
        for temp in &temps {
            if Some(*temp) == keep {
                continue;
            }
            // Emit nsl_tensor_free(temp) — but only if block is not already filled
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*temp]);
        }
    }

    /// Free the tensor temporaries a loop CONDITION registered during one
    /// evaluation, inside the block that evaluates it (the loop header),
    /// and drain them from `tensor_temporaries` so the loop statement's
    /// end-of-statement cleanup does not free them a second time.
    ///
    /// `base` is the list length snapshotted immediately before the
    /// condition compiled — everything at or above it was registered by
    /// this evaluation. `keep` is the condition's own result value
    /// (while-let binds it into the body; excluded defensively — a
    /// tracked keep would otherwise become freed-then-read).
    ///
    /// Emitting the frees in the HEADER is what makes this per-iteration:
    /// the header re-executes before every body entry AND before the
    /// exit branch, so the final evaluation's temps are freed exactly
    /// once too. Entries BELOW `base` (an outer statement's temps) stay
    /// listed and untouched.
    pub(crate) fn free_condition_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        base: usize,
        keep: Value,
    ) {
        let base = base.min(state.cleanup.tensor_temporaries.len());
        let temps = state.cleanup.tensor_temporaries.split_off(base);
        for temp in &temps {
            if *temp == keep {
                continue;
            }
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*temp]);
        }
    }

    /// ELTLS: free all TapeHeld tensors accumulated during the current tape
    /// region. Called after nsl_tape_backward runs and before the block's
    /// normal scope cleanup. See spec §7.3.
    ///
    /// Tape-held tensors were promoted by set_ownership_from_op or
    /// promote_to_tape_held when a DataRequired op touched them during
    /// forward pass. The tape holds raw pointers to their data for backward;
    /// we must not free until backward completes.
    pub(crate) fn free_tape_held_tensors(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        let held = std::mem::take(&mut state.cleanup.tape_held);
        for val in held {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[val]);
            state.cleanup.expr_ownership.remove(&val);
        }
    }

    /// M38b: Free linear tensors that were consumed during the current statement.
    /// Called after `free_tensor_temporaries` at each statement boundary.
    /// `keep` is the value being assigned to a variable (should NOT be freed).
    ///
    /// Only active when `state.ownership.lowering.is_some()` — the pending list
    /// is empty otherwise so the loop is a no-op.
    pub(crate) fn free_linear_consumes(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        if state.ownership.linear_consume_pending.is_empty() {
            return;
        }
        // Don't free inside tape-recorded regions — backward needs the data alive.
        // Tape ID identity: linear consumes can now be freed during tape recording.
        let pending = std::mem::take(&mut state.ownership.linear_consume_pending);
        for val in &pending {
            if Some(*val) == keep {
                continue;
            }
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*val]);
        }
    }

    /// Emit nsl_tensor_free calls for all tensor temporaries accumulated since the
    /// current loop scope started. Used at break/continue points.
    /// CRITICAL: Does NOT truncate tensor_temporaries — that only happens at natural scope exit.
    fn emit_loop_scope_cleanup(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(&scope_start) = state.cleanup.temp_scope_stack.last() {
            // Clamp defensively: the statement-end drain above now preserves
            // below-mark entries, but a stale mark must degrade to a no-op,
            // never a slice panic.
            let scope_start = scope_start.min(state.cleanup.tensor_temporaries.len());
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block
                    && is_block_filled(builder, block)
                {
                    break;
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
        }
    }

    /// Emit cleanup AND truncate temporaries at natural loop exit.
    pub(crate) fn cleanup_loop_scope(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(scope_start) = state.cleanup.temp_scope_stack.pop() {
            // Clamp defensively — see emit_loop_scope_cleanup.
            let scope_start = scope_start.min(state.cleanup.tensor_temporaries.len());
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block
                    && is_block_filled(builder, block)
                {
                    break;
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
            state.cleanup.tensor_temporaries.truncate(scope_start);
        }
    }

    fn cleanup_active_loop_batches(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        for &batch_var in state.cleanup.active_batch_vars.iter().rev() {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let batch_ptr = builder.use_var(batch_var);
            let _ = self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_ptr]);
        }
        for loop_ctx in state.loop_stack.iter().rev() {
            let Some(batch_var) = loop_ctx.batch_var else {
                continue;
            };
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let batch_ptr = builder.use_var(batch_var);
            let _ = self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_ptr]);
        }
    }

    /// Emit nsl_dataloader_stop + nsl_dataloader_free for all DataLoaders
    /// created in this scope. Called before function returns to prevent
    /// thread leaks and resource leaks.
    pub(crate) fn teardown_dataloaders(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        let loaders = std::mem::take(&mut state.cleanup.dataloader_vars);
        for dl in &loaders {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_stop", &[*dl]);
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_free", &[*dl]);
        }
    }

    pub(crate) fn expr_is_dataloader_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Call { callee, .. } => {
                if let ExprKind::Ident(fn_sym) = &callee.kind {
                    self.resolve_sym(*fn_sym) == "DataLoader"
                } else {
                    false
                }
            }
            ExprKind::Ident(sym) => state.dataloader_symbols.contains(sym),
            ExprKind::Paren(inner) => self.expr_is_dataloader_handle(state, inner),
            ExprKind::BlockExpr(block) => block
                .stmts
                .last()
                .is_some_and(|stmt| matches!(&stmt.kind, StmtKind::Expr(expr) if self.expr_is_dataloader_handle(state, expr))),
            ExprKind::IfExpr { then_expr, else_expr, .. } => {
                self.expr_is_dataloader_handle(state, then_expr)
                    && self.expr_is_dataloader_handle(state, else_expr)
            }
            _ => false,
        }
    }

    pub(crate) fn expr_is_borrowed_batch_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Ident(sym) => state.borrowed_batch_symbols.contains(sym),
            ExprKind::Paren(inner) => self.expr_is_borrowed_batch_handle(state, inner),
            ExprKind::BlockExpr(block) => block
                .stmts
                .last()
                .is_some_and(|stmt| matches!(&stmt.kind, StmtKind::Expr(expr) if self.expr_is_borrowed_batch_handle(state, expr))),
            ExprKind::IfExpr { then_expr, else_expr, .. } => {
                self.expr_is_borrowed_batch_handle(state, then_expr)
                    && self.expr_is_borrowed_batch_handle(state, else_expr)
            }
            _ => false,
        }
    }

    /// Check if an iterable expression is a real DataLoader handle.
    pub(crate) fn is_dataloader_iterable(&self, state: &FuncState, iterable: &nsl_ast::expr::Expr) -> bool {
        self.expr_is_dataloader_handle(state, iterable)
    }

    /// Emit a `zeros_like` allocation for ONE optimizer-moment buffer (m or
    /// v), honoring the offload / CPDT-precision variants. `precision_list` is
    /// the per-param dtype-code list for THIS moment when a precision plan is
    /// active (`None` = verbatim f32). Factored out of the optimizer-state
    /// init loop so the D3-v2 owner-gated (ZeRO-1 shard) path can reuse the
    /// exact same allocation logic inside its owned branch.
    pub(crate) fn emit_moment_zeros_like(
        &mut self,
        builder: &mut FunctionBuilder,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        let buf = if offload {
            if let Some(code_list) = precision_list {
                // P0.3 composition: HOST-resident state at the planned
                // reduced-precision dtype.
                let code = self.compile_call_by_name(builder, "nsl_list_get", &[code_list, idx])?;
                self.compile_call_by_name(
                    builder,
                    "nsl_tensor_zeros_like_host_dtype",
                    &[param_i, code],
                )?
            } else {
                // Offload-only: HOST-resident f32 state.
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like_host_f32", &[param_i])?
            }
        } else if let Some(code_list) = precision_list {
            let code = self.compile_call_by_name(builder, "nsl_list_get", &[code_list, idx])?;
            self.compile_call_by_name(builder, "nsl_tensor_zeros_like_dtype", &[param_i, code])?
        } else {
            self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[param_i])?
        };
        Ok(buf)
    }

    /// D3 v2 (ZeRO-1): emit an owner-gated optimizer-moment allocation. When
    /// this rank owns `idx` (owner = idx % world_size, decided at RUNTIME by
    /// nsl_zero_owns_param — the same predicate the update gate uses) it
    /// allocates the real zeros_like buffer and records its element count
    /// (nsl_zero_note_optim_alloc → the G3 memory-shrink gate); otherwise it
    /// allocates NOTHING and yields a null (0) placeholder. Returns the value
    /// to push into the moment list — real or null — keeping the list
    /// length==num_params and global-index-addressable. Non-owned nulls are
    /// never dereferenced (m/v are read only inside the owner-gated update
    /// branch; the free loop is null-safe).
    pub(crate) fn emit_owner_gated_moment(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        let owns = self.compile_call_by_name(builder, "nsl_zero_owns_param", &[idx])?;
        let one = builder.ins().iconst(cl_types::I64, 1);
        let owned = builder.ins().icmp(IntCC::Equal, owns, one);
        let alloc_b = builder.create_block();
        let skip_b = builder.create_block();
        let merge_b = builder.create_block();
        builder.append_block_param(merge_b, cl_types::I64);
        builder.ins().brif(owned, alloc_b, &[], skip_b, &[]);

        // Owned: allocate the real moment buffer and record its elements.
        builder.switch_to_block(alloc_b);
        builder.seal_block(alloc_b);
        state.current_block = Some(alloc_b);
        let real = self.emit_moment_zeros_like(builder, param_i, idx, precision_list, offload)?;
        self.compile_call_by_name(builder, "nsl_zero_note_optim_alloc", &[real])?;
        builder.ins().jump(merge_b, &[BlockArg::Value(real)]);

        // Non-owner: allocate nothing — push a null (0) placeholder.
        builder.switch_to_block(skip_b);
        builder.seal_block(skip_b);
        state.current_block = Some(skip_b);
        let null = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().jump(merge_b, &[BlockArg::Value(null)]);

        // Merge — both predecessors connected, safe to seal.
        builder.switch_to_block(merge_b);
        builder.seal_block(merge_b);
        state.current_block = Some(merge_b);
        Ok(builder.block_params(merge_b)[0])
    }

    /// Item C: one slot's moment allocation under the ZeRO-3 deferred fill.
    /// Shared by m and v so the two can never drift into different sharding
    /// decisions for the same parameter.
    fn emit_filled_moment(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        mode: MomentFill,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        Ok(match mode {
            // Elementwise: every rank holds a persistent 1/ws SLICE and
            // steps it, so there is no owner gate — the runtime sizes the
            // buffer from the carved shard and notes its own elements.
            MomentFill::Elementwise => self.compile_call_by_name(
                builder,
                "nsl_zero3_alloc_elem_moment",
                &[param_i, idx],
            )?,
            // Tensor-granular sharded: the SAME owner gate stages 1/2 ship,
            // reused verbatim — one rank allocates the full moment, the
            // rest carry the established null placeholder.
            MomentFill::OwnerGated => {
                self.emit_owner_gated_moment(builder, state, param_i, idx, precision_list, offload)?
            }
            // Replicated (tied / view-rooted / epilogue params): every rank
            // updates these from all-reduced gradients, so every rank needs
            // full m/v. Deliberately NOT noted against `optim_elems` — see
            // that counter's doc for why counting them would break the
            // `r0 + r1 == full` partition identity.
            //
            // It IS noted against the replica counter. Before that counter
            // existed this arm was invisible to every instrument in the tree:
            // the partition assertion the ZeRO-3 gates rest on stayed green
            // for an arbitrarily large replicated remainder, because `Full`
            // dropped out of BOTH sides of the identity. The epilogue set is
            // not a corner case — it is every parameter with no `blocks.N`
            // key, i.e. embedding / final norm / LM head.
            MomentFill::Full => {
                let real =
                    self.emit_moment_zeros_like(builder, param_i, idx, precision_list, offload)?;
                self.compile_call_by_name(
                    builder,
                    "nsl_zero_note_replicated_optim_alloc",
                    &[real],
                )?;
                real
            }
        })
    }

    /// Item C: fill the ZeRO-3 moment lists that section 4 left null.
    ///
    /// Emitted ONCE, inline at the WINDOW register belt — not the
    /// pre-forward belt, which is guarded by `if let Some(ws_fwd_plan)` and
    /// would make this loop silently vacuous when that plan is None (the
    /// PR #482 failure mode). It is a compile-time loop over EVERY parameter
    /// index, not over `wsplan.register_idxs`: replicated and
    /// tensor-granular params are not in the streamed register set but
    /// still need their moments.
    ///
    /// The whole fill is behind ONE runtime latch (`latch[0] == 0`), not a
    /// per-slot `state_list_1[idx] == 0` test: a non-owner's tensor-granular
    /// slot is legitimately null forever, so a per-slot guard would never
    /// latch and would re-run the owner gate for ~(ws-1)/ws of the sharded
    /// set on every optimizer step, inside the hot window region. One pass
    /// covers every slot, so one flag is equally correct and free after the
    /// first window.
    ///
    /// The alloc-surface bracket is re-opened here because setup's bracket
    /// only wraps section 4; without it the moment bytes land on whatever
    /// surface the step body left current (Activations) and
    /// `mem_accounting_gpu_gate`'s attribution is silently wrong. The pool
    /// bracket matters just as much: the step body runs on the TRANSIENT
    /// pool, whose segments are drained at end of step — moments must be
    /// Persistent or they would be handed back under the optimizer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_deferred_moment_fill(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        modes: &[MomentFill],
        param_list: Value,
        state_list_1: Value,
        state_list_2: Value,
        num_state_buffers: usize,
        m_codes: Option<Value>,
        v_codes: Option<Value>,
        muon_v_gate: Option<Value>,
        offload: bool,
        latch: Value,
    ) -> Result<(), CodegenError> {
        if m_codes.is_some() && modes.contains(&MomentFill::Elementwise) {
            // deferral-must-refuse: the elementwise slice allocator sizes
            // from the carve and hands back θ's own dtype — it has no
            // reduced-precision arm. Today this is unreachable (CPDT and
            // muon-state-bf16 are both refused alongside --zero-elementwise),
            // so refuse loudly instead of silently ignoring the plan.
            return Err(CodegenError::new(
                "--zero-elementwise does not compose with a per-parameter \
                 moment-precision plan: the elementwise slice moment is \
                 allocated at the parameter's own dtype. Drop one",
            ));
        }

        let slot0 = builder.ins().iconst(cl_types::I64, 0);
        let done = self.compile_call_by_name(builder, "nsl_list_get", &[latch, slot0])?;
        let first_window = builder.ins().icmp_imm_s(IntCC::Equal, done, 0);
        let fill_b = builder.create_block();
        let after_b = builder.create_block();
        builder.ins().brif(first_window, fill_b, &[], after_b, &[]);
        builder.switch_to_block(fill_b);
        builder.seal_block(fill_b);
        state.current_block = Some(fill_b);

        // The step body runs on the TRANSIENT pool (set at its top, flipped
        // back to Persistent at its bottom) and the allocator DRAINS
        // transient segments at end of step. Optimizer moments must outlive
        // every step, so flip to Persistent for the fill and back after —
        // there is no get/set for the pool, and this site is only ever
        // reached from inside that transient region.
        self.compile_call_by_name(builder, "nsl_gpu_set_persistent_pool", &[])?;
        let surface_prev = self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
        for (i, &mode) in modes.iter().enumerate() {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);
            let param_i = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;

            let s_m = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_M);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[s_m])?;
            let m_buf =
                self.emit_filled_moment(builder, state, mode, param_i, idx, m_codes, offload)?;
            self.compile_call_by_name(builder, "nsl_list_set", &[state_list_1, idx, m_buf])?;

            if num_state_buffers >= 2 {
                let s_v = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_V);
                self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[s_v])?;
                let v_buf = if let Some(route_list) = muon_v_gate {
                    // P1 Muon item 9, carried onto the deferred path: v (the
                    // AdamW second moment) is UNREAD on the Muon route, so
                    // route-gate it here exactly as section 4 does for the
                    // non-zero3 case. The route gate wraps the sharding gate
                    // — a Muon-routed param allocates no v on ANY rank, and
                    // an AdamW-routed one still allocates only its share.
                    let routed =
                        self.emit_muon_route_predicate(builder, route_list, idx, param_i)?;
                    let needs_v = builder.ins().icmp_imm_s(IntCC::Equal, routed, 0);
                    let alloc_b = builder.create_block();
                    let skip_b = builder.create_block();
                    let merge_b = builder.create_block();
                    builder.append_block_param(merge_b, cl_types::I64);
                    builder.ins().brif(needs_v, alloc_b, &[], skip_b, &[]);

                    builder.switch_to_block(alloc_b);
                    builder.seal_block(alloc_b);
                    state.current_block = Some(alloc_b);
                    let real = self.emit_filled_moment(
                        builder, state, mode, param_i, idx, v_codes, offload,
                    )?;
                    builder.ins().jump(merge_b, &[BlockArg::Value(real)]);

                    builder.switch_to_block(skip_b);
                    builder.seal_block(skip_b);
                    state.current_block = Some(skip_b);
                    let null_v = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(merge_b, &[BlockArg::Value(null_v)]);

                    builder.switch_to_block(merge_b);
                    builder.seal_block(merge_b);
                    state.current_block = Some(merge_b);
                    builder.block_params(merge_b)[0]
                } else {
                    self.emit_filled_moment(builder, state, mode, param_i, idx, v_codes, offload)?
                };
                self.compile_call_by_name(builder, "nsl_list_set", &[state_list_2, idx, v_buf])?;
            }
        }
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
        self.compile_call_by_name(builder, "nsl_gpu_set_transient_pool", &[])?;
        let one = builder.ins().iconst(cl_types::I64, 1);
        self.compile_call_by_name(builder, "nsl_list_set", &[latch, slot0, one])?;
        builder.ins().jump(after_b, &[]);

        builder.switch_to_block(after_b);
        builder.seal_block(after_b);
        state.current_block = Some(after_b);
        Ok(())
    }

    /// THE Muon-route predicate: `route_flag == 0 && runtime_rank == 2` —
    /// the one definition of "this parameter is stepped by Muon's matrix
    /// path". Item 8 (dispatcher unification): this rule was previously
    /// hand-spelled at three emission sites (batch-skip, resident-momentum,
    /// v-allocation — the last as its own De Morgan inverse) plus twice in
    /// the runtime, with a comment at each pleading that they stay equal.
    ///
    /// It must mirror, EXACTLY:
    ///   - `nsl_muon_step_batch`'s filter (muon_batch.rs): `route != 0` →
    ///     skip, `ndim != 2` → skip, and every LATER exit (device / dtype /
    ///     contiguity / empty matrix) is an ABORT, never a skip. That
    ///     asymmetry is load-bearing: a graceful `continue` added past the
    ///     first two tests would silently DROP params the batch-skip site
    ///     already jumped over. `muon_route_contract_drift` pins both sides.
    ///   - the stdlib branch (`stdlib/nsl/optim/muon.nsl`:
    ///     `adamw_route > 0.5 or len(s) != 2`).
    ///
    /// Callers needing the negation (v-allocation) invert the returned
    /// value (`icmp_imm == 0`) rather than re-deriving the inverse.
    pub(crate) fn emit_muon_route_predicate(
        &mut self,
        builder: &mut FunctionBuilder,
        route_list: Value,
        idx: Value,
        param: Value,
    ) -> Result<Value, CodegenError> {
        let flag_i = self.compile_call_by_name(builder, "nsl_list_get", &[route_list, idx])?;
        let ndim = self.compile_call_by_name(builder, "nsl_tensor_ndim", &[param])?;
        let zero_c = builder.ins().iconst(cl_types::I64, 0);
        let two_c = builder.ins().iconst(cl_types::I64, 2);
        let is_muon = builder.ins().icmp(IntCC::Equal, flag_i, zero_c);
        let is_r2 = builder.ins().icmp(IntCC::Equal, ndim, two_c);
        Ok(builder.ins().band(is_muon, is_r2))
    }

    /// Item 8: emit ONE batched fused-AdamW launch over the FullBuffer
    /// parameter list — the whole list normally, this rank's OWNER SUBSET
    /// under `--zero-stage 1/2`.
    ///
    /// Both Phase-B forks (clipped and unclipped) call this so the ZeRO
    /// decision cannot drift between them; they differ only in `mp_scale`
    /// (the clip factor, folded into the kernel's `m_partial` read, vs 1.0).
    ///
    /// Under ZeRO the subset is not an optimization, it is the correctness
    /// condition: a non-owned parameter's moment buffers are NULL
    /// placeholders (`emit_owner_gated_moment` allocates nothing for
    /// non-owners), so the batched launcher must never be handed one — and
    /// `nsl_zero_owned_step_indices` also performs the non-owner
    /// `m_partial` zero that the per-param loop's skip arm used to do.
    /// Handing the launcher only owners leaves its null assert intact as a
    /// belt rather than something to relax.
    ///
    /// **`owner_gated_moments` is the STAGES-1/2 predicate, not "ZeRO is
    /// on".** That is the whole invariant: the subset is valid exactly when
    /// every parameter's moments were owner-gated, which is true for stages
    /// 1/2 and for no other configuration. The old `&& !zero_enabled`
    /// exclusion was safe under `>= 1` only because it disabled batching for
    /// EVERY stage; narrowing the exclusion without narrowing the predicate
    /// is what opened the gap.
    ///
    /// **Stage 3 is not a third value of this boolean — it has no correct
    /// value.** Item C made stage-3 moments per-parameter: the deferred fill
    /// (`emit_deferred_moment_fill`, see the note at the allocation site)
    /// picks `MomentFill::Elementwise` (a 1/ws SLICE), `OwnerGated` (NULL on
    /// non-owners) or `Full` (replicated — tied / view-rooted / epilogue
    /// params every rank must step) per entry. `nsl_zero_owned_step_indices`
    /// describes none of those three, so `true` would skip the replicated
    /// params on non-owner ranks — the very freeze this paragraph used to
    /// warn about — while `false` hands the launcher the null placeholders
    /// its own assert exists to reject. The invariant for stage 3 is
    /// therefore "must not reach this launcher at all", not "must be
    /// correctly parameterised".
    ///
    /// That is structural, not incidental: stage 3 is refused unless
    /// `csla_active` (and `--weight-stream`, itself requiring
    /// `--layerwise-accum`), and both call sites sit in the `!csla_active`
    /// sub-arm — the SAME local, so `stage 3 => csla_active => short-circuit`
    /// is one implication, not a coincidence of two guards. The CSLA
    /// in-window batched arm is independently gated on
    /// `zero3.is_none() && zero3_elem.is_none()`. Verified by construction:
    /// a hard refusal planted at the top of this function never fires under
    /// any shipped stage-3 configuration.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_fused_multi_launch(
        &mut self,
        builder: &mut FunctionBuilder,
        owner_gated_moments: bool,
        num_params_val: Value,
        lists: (Value, Value, Value, Value),
        sc: &crate::stmt_fase::FusedAdamwScalars,
        // Live learning rate — see `fase_emit_final_step`. `sc.lr` is the
        // plan's base rate and folding it here discards the schedule.
        lr_runtime: Value,
        bc: (Value, Value),
        groups: (Value, Value),
        mp_scale: Value,
    ) -> Result<(), CodegenError> {
        let (param_list, state_list_1, state_list_2, accum) = lists;
        let lr_v = lr_runtime;
        let b1_v = builder.ins().f64const(sc.beta1);
        let omb1_v = builder.ins().f64const(sc.one_minus_beta1);
        let b2_v = builder.ins().f64const(sc.beta2);
        let omb2_v = builder.ins().f64const(sc.one_minus_beta2);
        let eps_v = builder.ins().f64const(sc.eps);
        let wd_v = builder.ins().f64const(sc.wd);

        if owner_gated_moments {
            let il = self.compile_call_by_name(
                builder,
                "nsl_zero_owned_step_indices",
                &[accum, num_params_val],
            )?;
            self.compile_call_by_name(
                builder,
                "nsl_fase_fused_adamw_step_multi_idx",
                &[
                    param_list,
                    state_list_1,
                    state_list_2,
                    accum,
                    il,
                    lr_v,
                    b1_v,
                    omb1_v,
                    b2_v,
                    omb2_v,
                    eps_v,
                    wd_v,
                    bc.0,
                    bc.1,
                    // Parameter-group arguments stay ORIGINAL-position
                    // indexed; the subset launcher documents that it resolves
                    // λ by the caller's numbering, which is why the exempt
                    // list is passed unfiltered alongside a filtered index
                    // list.
                    groups.0,
                    groups.1,
                    mp_scale,
                ],
            )?;
            self.compile_call_by_name(builder, "nsl_list_free", &[il])?;
        } else {
            self.compile_call_by_name(
                builder,
                "nsl_fase_fused_adamw_step_multi",
                &[
                    param_list,
                    state_list_1,
                    state_list_2,
                    accum,
                    lr_v,
                    b1_v,
                    omb1_v,
                    b2_v,
                    omb2_v,
                    eps_v,
                    wd_v,
                    bc.0,
                    bc.1,
                    groups.0,
                    groups.1,
                    mp_scale,
                ],
            )?;
        }
        Ok(())
    }

    /// Item 12: how a train-loop callback body touches the streamed model θ.
    /// Under `--weight-stream` params are EVICTED (`t.data == null`) when a
    /// callback runs, so a model-field read launches on a null pointer (the
    /// #395 crash). This drives a scoped `upload_all` / `reevict_all` bracket
    /// around any body that references the model, turning the runtime crash
    /// into a compile-time-inserted residency window.
    fn analyze_callback_model_touch(
        &self,
        block: &nsl_ast::stmt::Block,
        model_sym: nsl_ast::Symbol,
    ) -> CallbackModelTouch {
        let mut acc = CallbackModelTouch::default();
        for stmt in &block.stmts {
            self.walk_stmt_model_touch(stmt, model_sym, &mut acc);
        }
        acc
    }

    /// NSL in-place tensor mutators — the only calls that write a param
    /// through a receiver/dest operand (`copy_data(dest, src)`,
    /// `zero_inplace(t)`, the `nsl_tensor_{op}_inplace` family). Everything
    /// else is functional (returns a fresh tensor).
    fn is_inplace_mutator(name: &str) -> bool {
        matches!(name, "copy_data" | "copy_") || name.ends_with("_inplace")
    }

    /// Root identifier of an lvalue/access chain (`model.enc.w[0]` -> `model`).
    fn expr_root_ident(e: &nsl_ast::expr::Expr) -> Option<nsl_ast::Symbol> {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) => Some(*s),
            E::MemberAccess { object, .. } => Self::expr_root_ident(object),
            E::Subscript { object, .. } => Self::expr_root_ident(object),
            E::Paren(inner) => Self::expr_root_ident(inner),
            _ => None,
        }
    }

    /// Dotted path of a model-rooted member chain, for the diagnostic
    /// (`model.encoder.weight`). Best-effort — falls back to the model name.
    fn model_access_path(&self, e: &nsl_ast::expr::Expr) -> Option<String> {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) => Some(self.resolve_sym(*s).to_string()),
            E::MemberAccess { object, member } => {
                let base = self.model_access_path(object)?;
                Some(format!("{base}.{}", self.resolve_sym(*member)))
            }
            E::Subscript { object, .. } => self.model_access_path(object),
            E::Paren(inner) => self.model_access_path(inner),
            _ => None,
        }
    }

    fn walk_stmt_model_touch(
        &self,
        stmt: &nsl_ast::stmt::Stmt,
        model_sym: nsl_ast::Symbol,
        acc: &mut CallbackModelTouch,
    ) {
        use nsl_ast::stmt::StmtKind as S;
        match &stmt.kind {
            S::Assign { target, value, .. } => {
                // A write whose lvalue is rooted at the model mutates θ.
                if Self::expr_root_ident(target) == Some(model_sym) {
                    acc.touches = true;
                    acc.may_write = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(target);
                    }
                }
                self.walk_expr_model_touch(target, model_sym, acc);
                self.walk_expr_model_touch(value, model_sym, acc);
            }
            S::VarDecl { value: Some(v), .. } => {
                // Binding a model-derived value to a local (`let w = m.field`)
                // creates an ALIAS onto the resident streamed buffer; a later
                // `copy_data(w, ..)` would mutate θ through it without the root
                // ever being `model_sym`. We can't cheaply track the alias, so
                // conservatively treat any model-rooted binding as a possible
                // write (writeback=1 is always safe — for an unmutated param
                // device==mirror, so the extra DtoH is byte-identical).
                if Self::expr_root_ident(v) == Some(model_sym) {
                    acc.touches = true;
                    acc.may_write = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(v);
                    }
                }
                self.walk_expr_model_touch(v, model_sym, acc)
            }
            S::Expr(e) | S::Return(Some(e)) | S::Yield(Some(e)) => {
                self.walk_expr_model_touch(e, model_sym, acc)
            }
            S::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.walk_expr_model_touch(condition, model_sym, acc);
                for s in &then_block.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
                for (c, b) in elif_clauses {
                    self.walk_expr_model_touch(c, model_sym, acc);
                    for s in &b.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
                if let Some(b) = else_block {
                    for s in &b.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
            }
            S::For { iterable, body, .. } => {
                self.walk_expr_model_touch(iterable, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            S::While { condition, body } => {
                self.walk_expr_model_touch(condition, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            // A model read inside any of these still dereferences evicted θ —
            // walk them too, or the residency bracket is silently skipped and
            // the #395 crash returns (e.g. `@no_grad: print(m.x.sum())`).
            S::WhileLet { expr, body, .. } => {
                self.walk_expr_model_touch(expr, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            S::Match { subject, arms } => {
                self.walk_expr_model_touch(subject, model_sym, acc);
                for arm in arms {
                    for s in &arm.body.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
            }
            S::Decorated { stmt, .. } => {
                self.walk_stmt_model_touch(stmt, model_sym, acc);
            }
            _ => {}
        }
    }

    fn walk_expr_model_touch(
        &self,
        e: &nsl_ast::expr::Expr,
        model_sym: nsl_ast::Symbol,
        acc: &mut CallbackModelTouch,
    ) {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) if *s == model_sym => {
                acc.touches = true;
                if acc.first_path.is_none() {
                    acc.first_path = Some(self.resolve_sym(*s).to_string());
                }
            }
            E::MemberAccess { object, .. } => {
                if Self::expr_root_ident(e) == Some(model_sym) {
                    acc.touches = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(e);
                    }
                }
                self.walk_expr_model_touch(object, model_sym, acc);
            }
            E::Call { callee, args } => {
                // Only a genuine IN-PLACE mutator (`copy_data(m.x, ..)`,
                // `m.x.add_inplace(..)`, any `*_inplace`) writes θ. Functional
                // methods (`.sum()`, `.transpose()`, `.mean()`) return new
                // tensors and never mutate the receiver, so they are read-only
                // — flagging them would force a needless full-model writeback
                // on every logging callback. The callee name is the free-fn
                // ident or the method member.
                let callee_name = match &callee.kind {
                    E::Ident(s) => Some(self.resolve_sym(*s).to_string()),
                    E::MemberAccess { member, .. } => {
                        Some(self.resolve_sym(*member).to_string())
                    }
                    _ => None,
                };
                let is_mutator = callee_name.as_deref().is_some_and(Self::is_inplace_mutator);
                // A FREE-FN call passing a model-rooted param to a callee that
                // is not a known read-only sink could mutate that param in
                // place (`my_ema(m.field, ..)`), so treat it conservatively as
                // a write. `print`/`model_save` provably don't mutate their
                // tensor args. Method calls are covered by the mutator check
                // above (functional methods return fresh tensors).
                let is_free_fn = matches!(&callee.kind, E::Ident(_));
                let is_readonly_sink = callee_name
                    .as_deref()
                    .is_some_and(|n| matches!(n, "print" | "model_save"));
                let unknown_free_fn_write =
                    is_free_fn && !is_readonly_sink && !Self::is_inplace_mutator(callee_name.as_deref().unwrap_or(""));
                if is_mutator || unknown_free_fn_write {
                    // Dest is the model-rooted operand: the receiver for the
                    // method form (`m.x.add_inplace(..)`), an arg for the
                    // free-fn form (`copy_data(m.x, ..)` / `my_ema(m.x, ..)`).
                    let receiver_model = matches!(&callee.kind, E::MemberAccess { object, .. }
                        if Self::expr_root_ident(object) == Some(model_sym));
                    let arg_model = args
                        .iter()
                        .any(|a| Self::expr_root_ident(&a.value) == Some(model_sym));
                    if receiver_model || arg_model {
                        acc.touches = true;
                        acc.may_write = true;
                        if acc.first_path.is_none() {
                            acc.first_path = args
                                .iter()
                                .find_map(|a| {
                                    (Self::expr_root_ident(&a.value) == Some(model_sym))
                                        .then(|| self.model_access_path(&a.value))
                                        .flatten()
                                })
                                .or_else(|| match &callee.kind {
                                    E::MemberAccess { object, .. } => {
                                        self.model_access_path(object)
                                    }
                                    _ => None,
                                });
                        }
                    }
                }
                self.walk_expr_model_touch(callee, model_sym, acc);
                for a in args {
                    self.walk_expr_model_touch(&a.value, model_sym, acc);
                }
            }
            E::BinaryOp { left, right, .. } => {
                self.walk_expr_model_touch(left, model_sym, acc);
                self.walk_expr_model_touch(right, model_sym, acc);
            }
            E::UnaryOp { operand, .. } | E::Paren(operand) | E::Await(operand) => {
                self.walk_expr_model_touch(operand, model_sym, acc)
            }
            E::Pipe { left, right } => {
                self.walk_expr_model_touch(left, model_sym, acc);
                self.walk_expr_model_touch(right, model_sym, acc);
            }
            E::Subscript { object, .. } => {
                self.walk_expr_model_touch(object, model_sym, acc)
            }
            E::ListLiteral(xs) | E::TupleLiteral(xs) => {
                for x in xs {
                    self.walk_expr_model_touch(x, model_sym, acc);
                }
            }
            E::FString(parts) => {
                for p in parts {
                    if let nsl_ast::expr::FStringPart::Expr(x) = p {
                        self.walk_expr_model_touch(x, model_sym, acc);
                    }
                }
            }
            _ => {}
        }
    }

    /// Item 12 — open bracket: if `--weight-stream` is active and the
    /// callback body references model θ, make every streamed param resident
    /// (`upload_all`) so the body's reads don't launch on evicted (null)
    /// data. Returns `Some(may_write)` when a bracket was opened — the caller
    /// passes it to `emit_callback_residency_close`. Returns `None` (no-op)
    /// when streaming is off or the callback never touches the model, so the
    /// steady-state transfer arithmetic the CSLA gates assert is unchanged.
    pub(crate) fn emit_callback_residency_open(
        &mut self,
        builder: &mut FunctionBuilder,
        body: &nsl_ast::stmt::Block,
        model_sym: nsl_ast::Symbol,
        cb_name: &str,
    ) -> Result<Option<bool>, CodegenError> {
        if !self.compile_options.weight_stream.enabled {
            return Ok(None);
        }
        let touch = self.analyze_callback_model_touch(body, model_sym);
        if !touch.touches {
            return Ok(None);
        }
        nsl_log::nsl_log!(WARN, "weight-stream", 
            "[weight-stream] callback '{}' reads model state ({}); inserting a \
             scoped upload/re-evict bracket ({} writeback) so its reads see \
             resident \u{3b8} instead of crashing on evicted (null) data",
            cb_name,
            touch.first_path.as_deref().unwrap_or("model"),
            if touch.may_write { "with" } else { "without" },
        );
        self.compile_call_by_name(builder, "nsl_weight_stream_upload_all", &[])?;
        Ok(Some(touch.may_write))
    }

    /// Item 12 — close bracket: restore the streamed (evicted) invariant after
    /// a guarded callback body. `writeback=1` when the body might have mutated
    /// θ so the change survives the next window's upload; `0` for a read-only
    /// body (logging, `model_save`).
    pub(crate) fn emit_callback_residency_close(
        &mut self,
        builder: &mut FunctionBuilder,
        guard: Option<bool>,
    ) -> Result<(), CodegenError> {
        if let Some(may_write) = guard {
            let wb = builder
                .ins()
                .iconst(cl_types::I64, if may_write { 1 } else { 0 });
            self.compile_call_by_name(builder, "nsl_weight_stream_reevict_all", &[wb])?;
        }
        Ok(())
    }

    /// Item 10: emit a contiguous layer-pack UPLOAD. Builds an `NslList` of
    /// the pack's param tensor pointers (from their `param_list` indices) and
    /// hands it to the runtime, which stages the whole pack into ONE device
    /// arena transfer. The list build is a few cheap CPU calls; the win is the
    /// single HtoD it replaces N of.
    pub(crate) fn emit_ws_pack_upload(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        self.compile_call_by_name(builder, "nsl_weight_stream_upload_pack", &[pwlist])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    /// Item 10: emit a contiguous layer-pack EVICT (one DtoH when writeback).
    pub(crate) fn emit_ws_pack_evict(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
        writeback: i64,
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        let wb = builder.ins().iconst(cl_types::I64, writeback);
        self.compile_call_by_name(builder, "nsl_weight_stream_evict_pack", &[pwlist, wb])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    /// Item 11: emit an ASYNC pack transfer (`fn_name` is upload_pack's async
    /// sibling `nsl_weight_stream_prefetch_pack`, or the `nsl_weight_stream_
    /// await_pack` consumer). Shares the pw-list build with the sync helpers.
    pub(crate) fn emit_ws_pack_single(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
        fn_name: &str,
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        self.compile_call_by_name(builder, fn_name, &[pwlist])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    /// CFTP §4.3 / Tier A activation (spec 2026-05-17): probe a batch dict
    /// for segment_ids + doc_starts. When the DataLoader has packing=true,
    /// the packer (packing.rs::packed_batch_to_dict) emits both tensors per
    /// batch. Extract device pointers and stash them in the thread-local
    /// packing registry; the model's compiled @flash_attention call sites
    /// read them per launch.
    ///
    /// Probing at runtime (not codegen time) lets a single train block
    /// tolerate mixed-batch workloads or DataLoader implementations that
    /// conditionally emit segment_ids based on actual document structure.
    /// The probe is one CStr lookup — negligible cost vs kernel launches.
    ///
    /// Called once per micro-batch in the train loop, and again per buffered
    /// micro-batch at the head of the CSLA window-backward body (the registry
    /// holds the LAST batch's pointers otherwise, which would mis-mask every
    /// earlier micro-batch's replayed attention backward).
    pub(crate) fn emit_packing_registry_stash(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        batch_val: Value,
    ) -> Result<(), CodegenError> {
        use cranelift_codegen::ir::condcodes::IntCC;
        let k_seg = self.compile_string_literal(builder, "segment_ids")?;
        let has_seg =
            self.compile_call_by_name(builder, "nsl_dict_contains", &[batch_val, k_seg])?;
        let has_seg_block = builder.create_block();
        let no_seg_block = builder.create_block();
        let after_block = builder.create_block();
        let has_seg_cond = builder.ins().icmp_imm_s(IntCC::NotEqual, has_seg, 0);
        builder
            .ins()
            .brif(has_seg_cond, has_seg_block, &[], no_seg_block, &[]);

        // Packing-enabled batch: extract device pointers and set the
        // registry. Both segment_ids and doc_starts must be present
        // together — the packer emits them as a pair.
        builder.switch_to_block(has_seg_block);
        builder.seal_block(has_seg_block);
        let seg_tensor =
            self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_seg])?;
        let k_doc = self.compile_string_literal(builder, "doc_starts")?;
        let doc_tensor =
            self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_doc])?;
        let seg_data_ptr =
            self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[seg_tensor])?;
        let doc_data_ptr =
            self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[doc_tensor])?;
        // Item 17 phase 3a: the fused-AD reads consume these VARIABLES; the
        // thread-local set below stays for the model-METHOD readers
        // (`expr/advanced.rs`), which live in a different Cranelift function
        // until the phase-3b ABI change.
        if let Some((sv, dv)) = self.packing_meta_vars {
            builder.def_var(sv, seg_data_ptr);
            builder.def_var(dv, doc_data_ptr);
        }
        self.compile_call_by_name(
            builder,
            "nsl_packing_metadata_set",
            &[seg_data_ptr, doc_data_ptr],
        )?;
        builder.ins().jump(after_block, &[]);

        // Packing-disabled batch: clear the registry so stale state
        // from a prior step doesn't leak. Setting to (0, 0) is the
        // spec-defined sentinel for "identity path" at the kernel.
        builder.switch_to_block(no_seg_block);
        builder.seal_block(no_seg_block);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        if let Some((sv, dv)) = self.packing_meta_vars {
            builder.def_var(sv, zero);
            builder.def_var(dv, zero);
        }
        self.compile_call_by_name(builder, "nsl_packing_metadata_set", &[zero, zero])?;
        builder.ins().jump(after_block, &[]);

        builder.switch_to_block(after_block);
        builder.seal_block(after_block);
        // The probe's brif TERMINATED the block the caller was in; leaving
        // `state.current_block` pointing at it makes compile_stmt's
        // filled-block guard silently SKIP every subsequent statement.
        // That was the entire tape×DataLoader failure: the step body
        // compiled to nothing, so the 'loss' binding never landed and the
        // tape path refused with "must assign to a variable named 'loss'"
        // (source AD never noticed — it lowers the extracted Wengert list
        // without consulting current_block).
        state.current_block = Some(after_block);

        // PCA Tier A (spec §6.1): when a segment-masked kernel was
        // synthesized for this module, warn once if no segment_ids ever
        // appear in the first N steps (DataLoader-never-packs footgun).
        // Gated on the ACTUAL synthesized config so non-packed training
        // (the common case) never sees this call. has_seg is the
        // nsl_dict_contains("segment_ids") i64 result from above.
        let module_is_masked = self
            .kernels
            .flash_attention_context
            .as_ref()
            .and_then(|c| c.csha_training_config.as_ref())
            .map(|cfg| cfg.segment_masked)
            .unwrap_or(false);
        if module_is_masked {
            self.compile_call_by_name(builder, "nsl_pca_packing_mismatch_check", &[has_seg])?;
        }
        Ok(())
    }

    pub(crate) fn free_wengert_owned_values(
        &mut self,
        builder: &mut FunctionBuilder,
        owned_values: &[(crate::wengert::VarId, Value, crate::wengert::WengertType)],
        retained: &std::collections::HashSet<crate::wengert::VarId>,
    ) -> Result<(), CodegenError> {
        for (var_id, value, value_type) in owned_values {
            if retained.contains(var_id) {
                continue;
            }
            match value_type {
                crate::wengert::WengertType::Tensor => {
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[*value])?;
                }
                crate::wengert::WengertType::List => {
                    self.compile_call_by_name(builder, "nsl_list_free", &[*value])?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// M43b: Emit pipeline-parallel training loop with gradient serialization.
    ///
    /// When a model carries `@pipeline(stages=N)`, the train block emits:
    ///   1. `nsl_pipeline_init(num_stages, schedule_type, num_micro_batches)`
    ///   2. Extract model param_list, optimizer config, and step body.
    ///   3. Forward pass under tape recording — compile step body to produce
    ///      activations and loss.
    ///   4. Activation send — serialize the loss tensor to the next pipeline
    ///      stage via `nsl_pipeline_send`.
    ///   5. Backward pass — `nsl_tape_backward` computes per-parameter
    ///      gradients from the recorded tape.
    ///   6. Gradient send — serialize each parameter gradient to the previous
    ///      pipeline stage via `nsl_pipeline_send_grad`.
    ///   7. Optimizer step — apply optimizer update using the computed
    ///      gradients (same dispatch as the non-pipelined path).
    ///   8. `nsl_pipeline_barrier()` — synchronize all stages.
    ///   9. Cleanup — free gradient tensors, param_list, optimizer buffers,
    ///      and `nsl_pipeline_destroy()`.
    ///
    pub(crate) fn is_trainable_param_name(&self, param_name: &str) -> bool {
        is_trainable_param_leaf_name(param_name)
    }

    pub(crate) fn enumerate_all_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
        let mut paths = Vec::new();
        self.enumerate_tensor_paths_recursive(var_name, type_name, &mut paths, 0, true);
        paths
    }

    /// Enumerate all tensor field paths in a model struct via DFS.
    ///
    /// This mirrors the compiler's view of nested models and fixed arrays, so
    /// the emitted param_list and source-AD parameter resolution stay aligned.
    pub(crate) fn enumerate_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
        let mut paths = Vec::new();
        self.enumerate_tensor_paths_recursive(var_name, type_name, &mut paths, 0, false);
        paths
    }

    fn enumerate_tensor_paths_recursive(
        &self,
        prefix: &str,
        type_name: &str,
        paths: &mut Vec<String>,
        depth: usize,
        include_nontrainable: bool,
    ) {
        if depth > 16 {
            return;
        }

        let layout = match self.types.struct_layouts.get(type_name) {
            Some(layout) => layout.clone(),
            None => return,
        };
        let field_types = self.models.model_field_types.get(type_name).cloned();

        for field in &layout.fields {
            let field_path = format!("{}.{}", prefix, field.name);
            let field_type = field_types
                .as_ref()
                .and_then(|types| types.get(&field.name));

            if let Some(field_type) = field_type {
                if field_type.starts_with('[') && field_type.contains(';') {
                    let inner = field_type.trim_start_matches('[').trim_end_matches(']');
                    let parts: Vec<&str> = inner.split(';').collect();
                    if parts.len() == 2 {
                        let elem_type = parts[0].trim();
                        let count: usize = parts[1].trim().parse().unwrap_or(0);
                        for index in 0..count {
                            let elem_path = format!("{}.{}", field_path, index);
                            self.enumerate_tensor_paths_recursive(
                                &elem_path,
                                elem_type,
                                paths,
                                depth + 1,
                                include_nontrainable,
                            );
                        }
                    }
                    continue;
                }

                self.enumerate_tensor_paths_recursive(
                    &field_path,
                    field_type,
                    paths,
                    depth + 1,
                    include_nontrainable,
                );
                continue;
            }

            if field.cl_type == cl_types::I64
                && (include_nontrainable || self.is_trainable_param_name(&field_path))
            {
                paths.push(field_path);
            }
        }

        // WRGA B.3.2 Option 3: include synthesized adapter-injected fields
        // (lora_A_*, lora_B_*, ia3_scale_*, gate_*) in the ALL-paths
        // enumeration. Source-AD reads this as `trainable_tensor_param_paths`
        // so the gradient-summary diagnostic counts them (B.5 direct probe).
        //
        // Gated on `include_nontrainable` so `enumerate_model_tensor_paths`
        // (used to build the runtime param_list) does NOT return these —
        // runtime load via `load_nested_field` can't traverse the adapter
        // side-table at that point in codegen.
        if include_nontrainable {
            for site in self.bus.adapter_sites() {
                if site.target_model != type_name {
                    continue;
                }
                if site.input_dim == 0 || site.output_dim == 0 {
                    continue;
                }
                for synth in &site.synthesized_fields {
                    let synth_path = format!("{}.{}", prefix, synth);
                    paths.push(synth_path);
                }
            }
        }
    }

    /// Load a nested model field by traversing struct layouts along a compound name path.
    ///
    /// For a compound name like `m.blocks.0.attn.wq`, emits Cranelift IR to:
    /// 1. Start at `base_ptr` (pointer to top-level model struct)
    /// 2. Load `blocks` field from the top-level layout (FixedArray base)
    /// 3. Index element `0` from the array (load pointer at offset 0*8)
    /// 4. Load `attn` field from the TransformerBlock layout (sub-model pointer)
    /// 5. Load `wq` field from the GroupedQueryAttention layout (tensor pointer)
    ///
    /// Returns None if the path cannot be resolved through the struct layouts.
    pub(crate) fn load_nested_field(
        &self,
        builder: &mut FunctionBuilder,
        base_ptr: Value,
        top_layout: &crate::context::StructLayout,
        top_type_name: &str,
        compound_name: &str,
    ) -> Option<Value> {
        let parts: Vec<&str> = compound_name.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        // State: current struct pointer and current type name (for layout/field_type lookup)
        let mut current_ptr = base_ptr;
        let mut current_type_name = top_type_name.to_string();
        let mut current_layout = top_layout.clone();

        // Skip first component (model variable name like "m")
        let path = &parts[1..];

        let mut i = 0;
        while i < path.len() {
            let part = path[i];
            let is_last = i == path.len() - 1;

            // Check if this is a numeric array index (from FixedArray unrolling)
            if let Ok(array_idx) = part.parse::<usize>() {
                // current_ptr is already pointing to the base of the inline array
                // region (set by the preceding FixedArray field handler).
                // Each element is an i64 pointer at offset array_idx * 8.
                let elem_ptr = builder.ins().load(
                    cl_types::I64,
                    cranelift_codegen::ir::MemFlagsData::trusted(),
                    current_ptr,
                    (array_idx * 8) as i32,
                );
                if is_last {
                    return Some(elem_ptr);
                }
                current_ptr = elem_ptr;
                // current_layout and current_type_name were already set to the
                // element type by the preceding array field handler.
                i += 1;
                continue;
            }

            // Named field: look up in current struct layout
            let field = current_layout.fields.iter().find(|f| f.name == part)?;

            // Check if this field is a FixedArray type
            let field_type = self
                .models
                .model_field_types
                .get(&current_type_name)
                .and_then(|ft| ft.get(part))
                .cloned();

            if let Some(ref ft) = field_type
                && ft.starts_with('[') && ft.contains(';')
            {
                // FixedArray field: slots are stored inline in the parent struct.
                // DON'T load the field value — instead compute the address of the
                // array base region within the parent struct.
                let inner = ft.trim_start_matches('[').trim_end_matches(']');
                let elem_type = inner.split(';').next().unwrap_or("").trim();

                // Set current_ptr to address of array base in parent struct
                current_ptr = builder.ins().iadd_imm_s(current_ptr, field.offset as i64);
                current_type_name = elem_type.to_string();
                current_layout = self.types.struct_layouts.get(elem_type)?.clone();
                // Next component should be a numeric index
                i += 1;
                continue;
            }

            // Regular field: load the value
            let field_val = builder.ins().load(
                field.cl_type,
                cranelift_codegen::ir::MemFlagsData::trusted(),
                current_ptr,
                field.offset as i32,
            );

            if is_last {
                return Some(field_val);
            }

            // Navigate into sub-model struct
            current_ptr = field_val;
            if let Some(ref ft) = field_type {
                current_type_name = ft.clone();
                current_layout = self.types.struct_layouts.get(ft)?.clone();
            } else {
                // No type info — can't continue traversal
                return None;
            }

            i += 1;
        }

        None
    }

    /// Emit the runtime in-place-suppression guard around a source-AD FORWARD
    /// primal pass. Raise (`on=true`) before lowering the forward `WengertList`
    /// so FBIP does not overwrite a uniquely-owned input the adjoint still reads
    /// (e.g. `silu(x@W)`'s matmul temp, refcount 1, feeding an input-reading
    /// `SiluBackward`); lower (`on=false`) before the adjoint pass so backward
    /// FBIP still reclaims memory. Tape-AD gets this for free from
    /// `is_recording()`; source-AD builds no tape, so every source-AD forward
    /// site — grad blocks, train blocks, and model calibration — must bracket
    /// its primal lowering with this. Paired inc/dec so nested blocks compose.
    pub(crate) fn emit_inplace_suppress(
        &mut self,
        builder: &mut FunctionBuilder,
        on: bool,
    ) -> Result<(), CodegenError> {
        let v = builder.ins().iconst(cl_types::I64, i64::from(on));
        self.compile_call_by_name(builder, "nsl_set_inplace_suppressed", &[v])?;
        Ok(())
    }

    /// Item 3: bake the derived [`crate::parameter_plan::ParameterPlan`] into
    /// the binary and assert the runtime realized it.
    ///
    /// `nsl_weight_stream_register` chooses a parameter's residency backend
    /// from *global* flags (`zero3_active()` > `srbf16_active()` > host
    /// mirrors) while the plan is per-parameter and compile-time. Nothing
    /// otherwise connects the two, and a mismatch is silent: a parameter that
    /// reached `register` before its mode was enabled lands in the host-mirror
    /// table, trains in f32, and the run exits 0 with a plausible loss curve.
    /// One `declare` per parameter plus one `verify` closes that gap. Both
    /// are emitted inside the per-micro-batch registration region, so the
    /// check re-runs every micro-batch (catching drift, not just the first
    /// step).
    ///
    /// Cost, stated precisely because the two populations differ: the belt
    /// above emits `2s` calls per micro-batch (a `nsl_list_get` + a
    /// `register` for each of the `s` STREAMED parameters); this adds
    /// `2p + 1`, where `p` is ALL parameters — `p >= s`, since residents are
    /// declared too (see below). Measured as no wall-clock change on the
    /// CSLA FFN fixture.
    ///
    /// EVERY parameter is declared, not only the streamed ones. A resident
    /// parameter expects "registered with no backend", which is falsifiable
    /// and worth checking: the streaming schedule deliberately excludes
    /// view-rooted parameters (a buffered `transpose(w)` caches a pointer
    /// into θ's storage, so registering θ would free it under the live view —
    /// the #397 corruption hazard). If such a parameter ever leaks back into
    /// a registration belt, this is what says so.
    pub(crate) fn emit_param_plan_check(
        &mut self,
        builder: &mut FunctionBuilder,
        plan: &crate::parameter_plan::ParameterPlan,
        param_list: Value,
    ) -> Result<(), CodegenError> {
        // Nothing is registered anywhere, so there is no cross-check to make
        // and no call is emitted at all.
        if !plan.has_streamed() {
            return Ok(());
        }
        // Collected first so `plan` is not borrowed across the &mut self calls.
        let declares: Vec<(i64, i64)> = plan
            .entries()
            .iter()
            .map(|e| (e.idx, e.runtime_flags()))
            .collect();
        for (idx, flags) in declares {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let fv = builder.ins().iconst(cl_types::I64, flags);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_param_plan_declare", &[pw, iv, fv])?;
        }
        self.compile_call_by_name(builder, "nsl_param_plan_verify", &[])?;
        Ok(())
    }

    pub(crate) fn register_source_ad_model_instances(
        &self,
        extractor: &mut crate::source_ad::WengertExtractor<'_>,
        state: &FuncState,
    ) {
        for sym in self.variables_in_name_order(state) {
            if let Some(model_type_name) = self.resolve_source_ad_model_type_name(state, sym) {
                extractor.register_model_instance(sym, &model_type_name);
            }
        }
    }

    /// The variables in scope, by name — the order for any walk over
    /// `state.variables` that emits as it goes.
    ///
    /// `state.variables` is a `HashMap`; walking it in iteration order
    /// while calling `use_var` (which numbers a value per call) or
    /// emitting a call per entry laid `main` out differently from one
    /// compile of the same program to the next — same instructions, but
    /// the value numbers and the order of the cleanup frees moved, so no
    /// two `--dump-ir` runs could be compared and the CLIF snapshot tests
    /// (`tests/train_clif_snapshots.rs`) could not exist.
    ///
    /// One interner per compile makes the name injective over symbols
    /// today; the symbol index breaks a tie should that ever change, so
    /// the order never falls back to the map's.
    pub(crate) fn variables_in_name_order(&self, state: &FuncState) -> Vec<nsl_ast::Symbol> {
        let mut syms: Vec<nsl_ast::Symbol> = state.variables.keys().copied().collect();
        syms.sort_by_key(|&sym| (self.resolve_sym(sym), string_interner::Symbol::to_usize(sym.0)));
        syms
    }

    pub(crate) fn resolve_source_ad_model_type_name(
        &self,
        state: &FuncState,
        sym: nsl_ast::Symbol,
    ) -> Option<String> {
        self.models
            .model_var_types
            .get(&sym)
            .cloned()
            .or_else(|| {
                state.variable_types.get(&sym).and_then(|ty| match ty {
                    Type::Model { name, .. } | Type::Struct { name, .. } => {
                        Some(self.resolve_sym(*name).to_string())
                    }
                    _ => None,
                })
            })
            .filter(|name| self.types.struct_layouts.contains_key(name))
    }

    fn resolve_source_ad_expr_name(&self, expr: &nsl_ast::expr::Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Ident(sym) => Some(self.resolve_sym(*sym).to_string()),
            ExprKind::MemberAccess { object, member } => {
                let prefix = self.resolve_source_ad_expr_name(object)?;
                let member_name = self.resolve_sym(*member).to_string();
                Some(format!("{}.{}", prefix, member_name))
            }
            _ => None,
        }
    }

    pub(crate) fn resolve_source_ad_expr_var_id(
        &self,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        expr: &nsl_ast::expr::Expr,
        allow_last_op_fallback: bool,
    ) -> Option<crate::wengert::VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => extractor.symbol_var_map().get(sym).copied(),
            ExprKind::MemberAccess { .. } => {
                let name = self.resolve_source_ad_expr_name(expr)?;
                extractor
                    .named_param_var_ids()
                    .iter()
                    .find_map(|(compound_name, vid)| (compound_name == &name).then_some(*vid))
                    .or_else(|| {
                        extractor.wengert_list().var_names.iter().find_map(
                            |(vid, existing_name)| (existing_name == &name).then_some(*vid),
                        )
                    })
            }
            _ if allow_last_op_fallback => extractor.wengert_list().ops.last().map(|op| op.result),
            _ => None,
        }
    }

    pub(crate) fn load_source_ad_named_param(
        &self,
        builder: &mut FunctionBuilder,
        state: &FuncState,
        compound_name: &str,
    ) -> Option<Value> {
        let root_name = compound_name.split('.').next()?;
        let (&root_sym, &(root_var, _)) = state
            .variables
            .iter()
            .find(|(sym, _)| self.resolve_sym(**sym) == root_name)?;
        let model_type_name = self.resolve_source_ad_model_type_name(state, root_sym)?;
        let layout = self.types.struct_layouts.get(&model_type_name)?;
        let root_ptr = builder.use_var(root_var);
        self.load_nested_field(builder, root_ptr, layout, &model_type_name, compound_name)
    }
}

/// Simple glob matching supporting `*` (any sequence) and `?` (single char) wildcards.
pub(crate) fn glob_match(pattern: &str, text: &str) -> bool {
    let pb = pattern.as_bytes();
    let tb = text.as_bytes();
    let mut pi = 0usize;
    let mut ti = 0usize;
    let mut star_pi = usize::MAX;
    let mut star_ti = 0usize;

    while ti < tb.len() {
        if pi < pb.len() && (pb[pi] == b'?' || pb[pi] == tb[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pb.len() && pb[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pb.len() && pb[pi] == b'*' {
        pi += 1;
    }
    pi == pb.len()
}

/// Derive [`crate::cpdt_optim::AdamWHyperparams`] from a `@train` block's
/// optimizer section — via the SAME resolver the train lowering consumes,
/// so this can no longer drift from the main parse (pre-contract it was an
/// independent third parse: first-section-wins where the lowering was
/// last-wins, int-tolerant where the lowering was float-only, and unknown
/// kwargs silently ignored).
///
/// Returns library defaults when:
/// - `train` is `None`
/// - the sections fail the optimizer contract (the train lowering itself
///   is the refusal site — CPDT output never outlives a refused compile)
/// - the optimizer is not `AdamW`
///
/// `lr` and `weight_decay` are intentionally NOT read here — CPDT's
/// hyperparams cover only the running-moment constants.
///
/// Takes the `Interner` directly so this helper stays free-standing (callable
/// from `invoke_cpdt_if_enabled` via `&compiler.interner`) and is
/// straightforward to unit-test with a local `Interner`. Missing symbols
/// resolve to `"<unknown>"` to mirror `Compiler::resolve_sym`.
pub(crate) fn adamw_from_train_block(
    train: Option<&nsl_ast::block::TrainBlock>,
    interner: &nsl_lexer::Interner,
) -> crate::cpdt_optim::AdamWHyperparams {
    let mut hp = crate::cpdt_optim::AdamWHyperparams::default();

    let Some(train) = train else {
        return hp;
    };

    let Ok(cfg) = nsl_semantic::optim_config::resolve_optim_config(
        &train.sections,
        train.span,
        &|sym| interner.resolve(sym.0).unwrap_or("<unknown>").to_string(),
        nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
    ) else {
        return hp;
    };
    if cfg.optimizer.kind != nsl_semantic::optim_config::OptimizerKind::AdamW {
        return hp;
    }

    hp.beta1 = cfg.optimizer.beta1;
    hp.beta2 = cfg.optimizer.beta2;
    hp.eps = cfg.optimizer.eps;
    hp
}

#[cfg(test)]
mod tests {
    use super::{
        adamw_from_train_block, classify_source_ad_param_name, is_trainable_param_leaf_name,
        SourceAdParamDiagnosticKind,
    };
    use std::collections::HashSet;

    #[test]
    fn source_ad_param_classification_separates_tensor_and_non_tensor_noise() {
        let tensor_paths: HashSet<String> = [
            "m.blocks.0.attn.wq".to_string(),
            "m.blocks.0.attn._dropout_p".to_string(),
            "m.blocks.0.attn.rope.inv_freq".to_string(),
        ]
        .into_iter()
        .collect();

        assert!(is_trainable_param_leaf_name("m.blocks.0.attn.wq"));
        assert!(!is_trainable_param_leaf_name("m.blocks.0.attn._dropout_p"));
        assert!(!is_trainable_param_leaf_name(
            "m.blocks.0.attn.rope.inv_freq"
        ));

        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn.wq", &tensor_paths),
            SourceAdParamDiagnosticKind::Trainable,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn._dropout_p", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredConfig,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn.rope.inv_freq", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredConfig,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn_norm.eps", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredNonTensor,
        );
    }

    // ── Task 2: adamw_from_train_block helper ───────────────────────────
    //
    // Build small TrainBlock fixtures directly (simpler than running the
    // parser + semantic passes just to get AST shape we control).
    use nsl_ast::block::{TrainBlock, TrainSection};
    use nsl_ast::expr::{Arg, Expr, ExprKind};
    use nsl_ast::{NodeId, Span, Symbol};
    use nsl_lexer::Interner;

    fn mk_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span::dummy(),
            id: NodeId::next(),
        }
    }

    fn mk_arg(name: Option<Symbol>, value: Expr) -> Arg {
        Arg {
            name,
            value,
            span: Span::dummy(),
        }
    }

    #[test]
    fn adamw_hyperparams_default_when_no_train_block() {
        let interner: Interner = Interner::new();
        let hp = adamw_from_train_block(None, &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - d.beta1).abs() < 1e-12);
        assert!((hp.beta2 - d.beta2).abs() < 1e-12);
        assert!((hp.eps - d.eps).abs() < 1e-12);
    }

    #[test]
    fn adamw_hyperparams_derived_from_train_block() {
        let mut interner: Interner = Interner::new();
        let adamw_sym = Symbol(interner.get_or_intern("AdamW"));
        let beta1_sym = Symbol(interner.get_or_intern("beta1"));
        let beta2_sym = Symbol(interner.get_or_intern("beta2"));

        // optimizer = AdamW(beta1=0.85, beta2=0.99)
        let callee = Box::new(mk_expr(ExprKind::Ident(adamw_sym)));
        let args = vec![
            mk_arg(Some(beta1_sym), mk_expr(ExprKind::FloatLiteral(0.85))),
            mk_arg(Some(beta2_sym), mk_expr(ExprKind::FloatLiteral(0.99))),
        ];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - 0.85).abs() < 1e-12, "beta1 = {}", hp.beta1);
        assert!((hp.beta2 - 0.99).abs() < 1e-12, "beta2 = {}", hp.beta2);
        // eps was not overridden — should stay at library default.
        assert!((hp.eps - d.eps).abs() < 1e-12, "eps = {}", hp.eps);
    }

    #[test]
    fn adamw_hyperparams_default_when_block_fails_the_contract() {
        // AdamW(beta1=0.85, lrr=0.01): the typo'd kwarg fails
        // resolve_optim_config, so CPDT sees library defaults — never the
        // half-parsed beta1. (The train lowering itself refuses the block,
        // so those defaults cannot train anything; pre-contract this
        // helper would have silently returned beta1=0.85 while the typo
        // trained at the default lr.)
        let mut interner: Interner = Interner::new();
        let adamw_sym = Symbol(interner.get_or_intern("AdamW"));
        let beta1_sym = Symbol(interner.get_or_intern("beta1"));
        let lrr_sym = Symbol(interner.get_or_intern("lrr"));

        let callee = Box::new(mk_expr(ExprKind::Ident(adamw_sym)));
        let args = vec![
            mk_arg(Some(beta1_sym), mk_expr(ExprKind::FloatLiteral(0.85))),
            mk_arg(Some(lrr_sym), mk_expr(ExprKind::FloatLiteral(0.01))),
        ];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!(
            (hp.beta1 - d.beta1).abs() < 1e-12,
            "contract-refused block must not leak half-parsed values"
        );
    }

    #[test]
    fn adamw_hyperparams_falls_back_for_non_adamw_optimizer() {
        // SGD(momentum=0.9) should yield library defaults — no silent β1 override.
        let mut interner: Interner = Interner::new();
        let sgd_sym = Symbol(interner.get_or_intern("SGD"));
        let momentum_sym = Symbol(interner.get_or_intern("momentum"));

        let callee = Box::new(mk_expr(ExprKind::Ident(sgd_sym)));
        let args = vec![mk_arg(
            Some(momentum_sym),
            mk_expr(ExprKind::FloatLiteral(0.9)),
        )];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - d.beta1).abs() < 1e-12);
        assert!((hp.beta2 - d.beta2).abs() < 1e-12);
        assert!((hp.eps - d.eps).abs() < 1e-12);
    }
}
