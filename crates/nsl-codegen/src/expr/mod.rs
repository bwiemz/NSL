pub(crate) mod access;
mod advanced;
mod binary_ops;
mod calls;
mod literals;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::ownership_expr::Ownership;
use crate::types::is_float_type;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::operator::UnaryOp;
use nsl_semantic::types::Type;

impl Compiler<'_> {
    pub fn compile_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        expr: &Expr,
    ) -> Result<Value, CodegenError> {
        // Auto-fusion: check for fusible chains before normal dispatch
        if let Some(fused_result) = self.try_auto_fuse(builder, state, expr)? {
            return Ok(fused_result);
        }

        match &expr.kind {
            ExprKind::IntLiteral(n) => Ok(builder.ins().iconst(cl_types::I64, *n)),
            ExprKind::FloatLiteral(f) => Ok(builder.ins().f64const(*f)),
            ExprKind::BoolLiteral(b) => {
                Ok(builder.ins().iconst(cl_types::I8, if *b { 1 } else { 0 }))
            }
            ExprKind::StringLiteral(s) => self.compile_string_literal(builder, s),
            ExprKind::NoneLiteral => Ok(builder.ins().iconst(cl_types::I64, 0)),

            ExprKind::Ident(sym) => {
                if let Some((var, _)) = state.variables.get(sym) {
                    let var = *var;
                    let val = builder.use_var(var);
                    // ELTLS: register tensor-typed Ident references as
                    // BorrowedFromVar for ownership tracking. Must be done
                    // BEFORE the M38b linear-consume path below because
                    // subsequent consumer commits may inspect the map.
                    let ident_ty = self.node_type(expr.id).clone();
                    if ident_ty.is_tensor() || ident_ty.is_indeterminate() {
                        self.set_ownership(
                            builder,
                            state,
                            val,
                            Ownership::BorrowedFromVar(var),
                        );
                    }
                    // M38b: If ownership lowering says this linear binding should be
                    // freed at consumption, enqueue it for post-statement cleanup.
                    // The actual nsl_tensor_free is emitted in free_linear_consumes()
                    // after the statement finishes using the value.
                    if let Some(ref lowering) = state.ownership.lowering {
                        if lowering.should_free_at_consumption(sym)
                            && !state.param_symbols.contains(sym)
                            && !state.non_owning_symbols.contains(sym)
                        {
                            state.ownership.linear_consume_pending.push(val);
                        }
                    }
                    Ok(val)
                } else {
                    let name = self.resolve_sym(*sym).to_string();
                    // Device constants for .to(device) calls
                    if name == "cuda" {
                        return Ok(builder.ins().iconst(cl_types::I64, 1));
                    }
                    if name == "cpu" {
                        return Ok(builder.ins().iconst(cl_types::I64, 0));
                    }
                    // Check if it's a custom dtype constant (for .to(CustomDtype) etc.)
                    if let Some(dtype_id) = self.resolve_custom_dtype(&name) {
                        return Ok(builder.ins().iconst(cl_types::I64, dtype_id as i64));
                    }
                    // Check if it's an enum variant
                    if let Some(tag) = self.lookup_enum_variant_tag(&name) {
                        Ok(builder.ins().iconst(cl_types::I64, tag))
                    } else {
                        Err(CodegenError::new(format!("undefined variable '{name}'")))
                    }
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                self.compile_binary_op(builder, state, left, *op, right, expr)
            }
            ExprKind::UnaryOp { op, operand } => self.compile_unary_op(builder, state, *op, operand),
            ExprKind::Call { callee, args } => self.compile_call(builder, state, callee, args, expr),
            ExprKind::MemberAccess { object, member } => {
                self.compile_member_access(builder, state, object, *member, expr)
            }
            ExprKind::ListLiteral(elements) => self.compile_list_literal(builder, state, elements),
            ExprKind::Subscript { object, index } => {
                self.compile_subscript(builder, state, object, index)
            }
            ExprKind::ListComp {
                element,
                generators,
            } => self.compile_list_comp(builder, state, element, generators),
            ExprKind::Lambda { params, body } => self.compile_lambda(builder, state, params, body),
            ExprKind::BlockExpr(block) => self.compile_block_expr(builder, state, block),
            ExprKind::TupleLiteral(elements) => {
                self.compile_tuple_literal(builder, state, elements)
            }
            ExprKind::DictLiteral(pairs) => self.compile_dict_literal(builder, state, pairs),
            ExprKind::MatchExpr { subject, arms } => {
                self.compile_match_expr(builder, state, subject, arms, expr)
            }
            ExprKind::Range {
                start,
                end,
                inclusive,
            } => self.compile_range_expr(
                builder,
                state,
                start.as_deref(),
                end.as_deref(),
                *inclusive,
            ),
            ExprKind::FString(parts) => self.compile_fstring(builder, state, parts),
            ExprKind::Paren(inner) => self.compile_expr(builder, state, inner),
            ExprKind::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => self.compile_if_expr(builder, state, condition, then_expr, else_expr, expr),
            ExprKind::Pipe { left, right } => {
                let left_val = self.compile_expr(builder, state, left)?;
                match &right.kind {
                    ExprKind::Ident(_) => {
                        let func_name = self.expr_as_func_name(right);
                        if let Some(name) = func_name {
                            if let Some(result) =
                                self.try_compile_model_tensor_pipe(builder, state, &name, left_val)?
                            {
                                return Ok(result);
                            }
                            // Task 4: emit retention splice BEFORE the matmul call.
                            // No-op when calibration_retention is None.
                            self.try_emit_retention_splice(builder, &name, left_val);
                            self.compile_call_by_name(builder, &name, &[left_val])
                        } else {
                            Err(CodegenError::new("pipe target is not a function"))
                        }
                    }
                    ExprKind::Call { callee, args } => {
                        let func_name = self.expr_as_func_name(callee);
                        if let Some(name) = func_name {
                            // Task 4: emit retention splice BEFORE the matmul call.
                            // No-op when calibration_retention is None.
                            self.try_emit_retention_splice(builder, &name, left_val);
                            let mut arg_vals = vec![left_val];
                            for arg in args {
                                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                            }
                            self.compile_call_by_name(builder, &name, &arg_vals)
                        } else {
                            Err(CodegenError::new("pipe call target is not a function"))
                        }
                    }
                    _ => Err(CodegenError::new("unsupported pipe target")),
                }
            }

            ExprKind::SelfRef => {
                match &state.self_resolution {
                    crate::context::SelfResolution::StructPointer => {
                        // Find the "self" variable in state (set by model method compilation)
                        for (sym, (var, _ty)) in &state.variables {
                            if self.resolve_sym(*sym) == "self" {
                                return Ok(builder.use_var(*var));
                            }
                        }
                        Err(CodegenError::new("`self` used outside of model method"))
                    }
                    crate::context::SelfResolution::WeightPtrsArray { .. } => {
                        Err(CodegenError::new(
                            "@export method: bare `self` has no runtime value — only `self.<weight_field>` access is supported",
                        ))
                    }
                }
            }

            ExprKind::Error => Ok(builder.ins().iconst(cl_types::I64, 0)),
            _ => Err(CodegenError::new(format!(
                "unsupported expression in M3 codegen: {:?}",
                std::mem::discriminant(&expr.kind)
            ))),
        }
    }

    // ── Unary ops ───────────────────────────────────────────────────

    fn compile_unary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        op: UnaryOp,
        operand: &Expr,
    ) -> Result<Value, CodegenError> {
        let val = self.compile_expr(builder, state, operand)?;
        let ty = self.node_type(operand.id).clone();
        match op {
            UnaryOp::Neg
                if ty.is_tensor() || (ty.is_indeterminate() && !state.flags.in_dtype_method) =>
            {
                self.compile_call_by_name(builder, "nsl_tensor_neg", &[val])
            }
            UnaryOp::Neg if is_float_type(&ty) || builder.func.dfg.value_type(val).is_float() => {
                Ok(builder.ins().fneg(val))
            }
            UnaryOp::Neg => Ok(builder.ins().ineg(val)),
            UnaryOp::Not => Ok(builder.ins().icmp_imm(IntCC::Equal, val, 0)),
        }
    }

    pub fn expr_as_func_name(&self, expr: &Expr) -> Option<String> {
        // WRGA B.3 Task 4: fused-adapter rewrite emits synthesized Call
        // nodes whose callee Ident carries a sentinel Symbol; the real
        // FFI name lives in `synth_call_names` keyed by the callee's
        // NodeId.
        if let Some(name) = self.synth_call_names.get(&expr.id) {
            return Some(name.clone());
        }
        match &expr.kind {
            ExprKind::Ident(sym) => Some(self.resolve_sym(*sym).to_string()),
            _ => None,
        }
    }

    fn try_compile_model_tensor_pipe(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        field_name: &str,
        input_val: Value,
    ) -> Result<Option<Value>, CodegenError> {
        let Some(model_name) = self.current_method_model_name.clone() else {
            return Ok(None);
        };
        let is_tensor_field = self
            .models
            .model_tensor_field_shapes
            .get(&model_name)
            .is_some_and(|fields| fields.contains_key(field_name));
        if !is_tensor_field {
            return Ok(None);
        }

        let self_var = state
            .variables
            .iter()
            .find_map(|(sym, (var, _ty))| (self.resolve_sym(*sym) == "self").then_some(*var))
            .ok_or_else(|| CodegenError::new("model tensor pipe missing self binding"))?;
        let field_offset = self
            .types
            .struct_layouts
            .get(&model_name)
            .and_then(|layout| {
                layout
                    .fields
                    .iter()
                    .find(|field| field.name == field_name)
                    .map(|field| field.offset)
            })
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "model '{}' has no tensor field '{}'",
                    model_name, field_name
                ))
            })?;

        let self_ptr = builder.use_var(self_var);
        let field_val = builder.ins().load(
            cl_types::I64,
            MemFlags::trusted(),
            self_ptr,
            field_offset as i32,
        );

        if let Some(ref weight_map) = self.features.weight_map {
            let candidates = [format!("{}.{}", model_name, field_name), field_name.to_string()];
            for key in &candidates {
                if weight_map.get(key).is_some() {
                    state.weight_values.insert(field_val, key.clone());
                    state.weights_by_name.insert(key.clone(), field_val);
                    break;
                }
            }
        }
        if state.weight_values.contains_key(&field_val) {
            self.set_ownership(builder, state, field_val, Ownership::BorrowedWeight);
        }

        self.try_emit_retention_splice(builder, field_name, input_val);
        let flags_zero = builder.ins().iconst(cl_types::I8, 0);
        self.compile_traced_call(builder, "nsl_tensor_matmul", &[input_val, field_val, flags_zero])
            .map(Some)
    }

    // ── M38b: Ownership-aware helpers ──────────────────────────────

    /// Check if a tensor Ident expression should skip refcount ops (retain/release).
    /// Returns true when the binding is linear under ownership lowering, meaning the
    /// compiler has proven single-ownership — no refcount bumps needed.
    ///
    /// ELTLS commit 3: the sole caller (the retain/release dance in binary_ops)
    /// has been deleted. Task 13 will revive this helper to drive the relinquish
    /// flag bytes on tensor binary FFIs.
    #[allow(dead_code)]
    pub(crate) fn should_elide_refcount_for_ident(
        state: &FuncState,
        expr: &nsl_ast::expr::Expr,
    ) -> bool {
        if let Some(ref lowering) = state.ownership.lowering {
            if let nsl_ast::expr::ExprKind::Ident(sym) = &expr.kind {
                return lowering.should_elide_refcount(sym);
            }
        }
        false
    }

    fn track_tensor_temporary(state: &mut FuncState, value: Value) {
        if !state.cleanup.tensor_temporaries.contains(&value) {
            state.cleanup.tensor_temporaries.push(value);
        }
    }

    fn expr_result_is_owned_temporary(&self, expr: &Expr) -> bool {
        match &expr.kind {
            // Only a narrow set of builtin tensor-producing calls need extra
            // registration here. Broad call tracking interferes with training
            // lowering because many calls either alias existing buffers or are
            // already handled by normal statement ownership paths.
            ExprKind::Call { callee, .. } => {
                if let ExprKind::Ident(sym) = &callee.kind {
                    let name = self.resolve_sym(*sym);
                    return matches!(
                        name,
                        "exp"
                            | "log"
                            | "sqrt"
                            | "abs"
                            | "sign"
                            | "neg"
                            | "zeros"
                            | "ones"
                            | "full"
                            | "rand"
                            | "randn"
                            | "arange"
                            | "zeros_like"
                            | "relu"
                            | "gelu"
                            | "silu"
                            | "sigmoid"
                            | "tensor_sin"
                            | "tensor_cos"
                            | "rotate_half"
                            | "tanh"
                            | "softmax"
                            | "log_softmax"
                            | "mean"
                            | "sum"
                    );
                }
                false
            }
            _ => false,
        }
    }

    /// True when `expr` is a call whose result is guaranteed to be an OWNING
    /// reference: either a fresh-tensor builtin (allowlist above) or a call
    /// to a compiled NSL function (`registry.functions` — user code and the
    /// NSL-level stdlib). A compiled function's Return arm always hands back
    /// an owning ref (Owned results transfer; Borrowed/Unknown results are
    /// retained before return), so treating the call result as Unknown and
    /// retaining AGAIN double-owns it — the caller then frees once and one
    /// reference strands. That double-own was the last per-step leak on the
    /// tape path (mse_loss's `return mean(...)`) and a per-call leak for any
    /// user fn returning another user fn's result.
    pub(crate) fn expr_call_returns_owning_ref(&self, expr: &Expr) -> bool {
        if self.expr_result_is_owned_temporary(expr) {
            return true;
        }
        if let ExprKind::Call { callee, .. } = &expr.kind {
            if let ExprKind::Ident(sym) = &callee.kind {
                let name = self.resolve_sym(*sym);
                return self.registry.functions.contains_key(name);
            }
        }
        false
    }

    /// Call-like expression lowering bypasses the per-op temporary registration
    /// used by binary tensor ops. Track those owned tensor results here so
    /// anonymous subexpressions participate in statement cleanup.
    fn track_owned_tensor_expr_result(&self, state: &mut FuncState, expr: &Expr, value: Value) {
        if !self.expr_result_is_owned_temporary(expr) {
            return;
        }
        let ty = self.node_type(expr.id);
        if ty.is_tensor() || matches!(ty, Type::Sparse { .. }) {
            Self::track_tensor_temporary(state, value);
        }
    }

    pub(crate) fn compile_nested_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        expr: &Expr,
    ) -> Result<Value, CodegenError> {
        let value = self.compile_expr(builder, state, expr)?;
        self.track_owned_tensor_expr_result(state, expr, value);
        Ok(value)
    }

    // ── Trace instrumentation (M45) ─────────────────────────────────

    /// FBIP Phase 2: Select inplace or normal variant for a unary tensor op.
    /// Returns `nsl_tensor_{op}_inplace` when the argument is a single-use Ident
    /// binding and we're not in a tape-recording region, otherwise `nsl_tensor_{op}`.
    ///
    /// M38b: Also selects inplace when ownership lowering proves the binding is
    /// linear with no borrows or shared ownership — the compiler guarantees
    /// exclusive access so the runtime refcount check can be bypassed entirely.
    pub(crate) fn fbip_select_variant(
        &self,
        state: &FuncState,
        arg_expr: &nsl_ast::expr::Expr,
        op_name: &str,
    ) -> String {
        // Only optimize when not recording autodiff tape
        if !state.flags.in_tape_region {
            if let nsl_ast::expr::ExprKind::Ident(sym) = &arg_expr.kind {
                // M38b: Ownership lowering can prove exclusive access even when
                // use-count heuristics cannot (e.g., multi-use linear binding
                // where all but this use have already been consumed).
                if let Some(ref lowering) = state.ownership.lowering {
                    if lowering.should_use_inplace(sym) {
                        let inplace = format!("nsl_tensor_{op_name}_inplace");
                        if self.registry.functions.contains_key(&inplace) {
                            return inplace;
                        }
                    }
                }
                // Fall back to use-count heuristic for non-linear bindings
                if let Some(ref uc) = state.use_counts {
                    if uc.is_single_use(sym) {
                        let inplace = format!("nsl_tensor_{op_name}_inplace");
                        // Verify the inplace variant is registered before using it
                        if self.registry.functions.contains_key(&inplace) {
                            return inplace;
                        }
                    }
                }
            }
        }
        format!("nsl_tensor_{op_name}")
    }

    /// Emit a tensor FFI call, optionally followed by trace recording
    /// when `compile_options.trace_ops` is enabled.
    pub(crate) fn compile_traced_call(
        &mut self,
        builder: &mut FunctionBuilder,
        fn_name: &str,
        args: &[Value],
    ) -> Result<Value, CodegenError> {
        let result = self.compile_call_by_name(builder, fn_name, args)?;

        if self.compile_options.trace_ops {
            // Emit: nsl_trace_record_op(op_type_id, input0, input1_or_0, result)
            let op_id = builder
                .ins()
                .iconst(cl_types::I64, self.trace_op_id(fn_name) as i64);
            // Only forward pointer-typed (I64) args — scalar-op FFI calls like
            // `nsl_tensor_mul_scalar(tensor, f64)` would otherwise pass an f64
            // where the trace hook expects an i64, tripping Cranelift's verifier.
            let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
            let in0 = if !args.is_empty()
                && builder.func.dfg.value_type(args[0]) == cl_types::I64
            {
                args[0]
            } else {
                zero_i64
            };
            let in1 = if args.len() > 1
                && builder.func.dfg.value_type(args[1]) == cl_types::I64
            {
                args[1]
            } else {
                zero_i64
            };
            let nan_flag = self.compile_call_by_name(
                builder,
                "nsl_trace_record_op",
                &[op_id, in0, in1, result],
            )?;

            // M45: Pass NaN flag to warning function — it only prints when flag==1
            self.compile_call_by_name(builder, "nsl_trace_nan_warning", &[nan_flag, op_id])?;
        }

        Ok(result)
    }

    /// Task 4 — Calibration retention splice.
    ///
    /// When `calibration_retention` is active and the current pipe target's
    /// qualified path (`<model_name>.<field_name>`) matches an entry in the
    /// retention list, emit a `emit_splice_memcpy` call **before** the matmul
    /// is lowered.  This copies the input activation into the corresponding
    /// slot of `__nsl_calib_retention_arena`.
    ///
    /// # Arguments
    ///
    /// * `builder`    — The Cranelift function builder (mid-lowering of the pipe).
    /// * `field_name` — The bare pipe-target identifier (e.g. `"up_proj"`).
    /// * `input_val`  — The Cranelift `Value` for the tensor pointer flowing
    ///                  into the linear layer (i.e., the pipe LHS).
    ///
    /// # Safety contract
    ///
    /// This is called BEFORE `compile_call_by_name` for the matmul — the
    /// input activation has not been consumed or freed at this point.
    ///
    /// # Zero-IR guarantee
    ///
    /// When `calibration_retention` is `None` (all shipped binaries), this
    /// function returns immediately without emitting any IR.
    pub(crate) fn try_emit_retention_splice(
        &mut self,
        builder: &mut FunctionBuilder,
        field_name: &str,
        input_val: Value,
    ) {
        // Fast path: no retention active.
        let retention = match &self.compile_options.calibration_retention {
            Some(r) => r.clone(),
            None => return,
        };

        // Determine the qualified path for this pipe target.
        let model_name = match &self.current_method_model_name {
            Some(m) => m.clone(),
            None => return, // Not inside a model method — skip.
        };
        let qualified = format!("{}.{}", model_name, field_name);

        // Find a matching retention entry.
        let hit = match retention.iter().find(|dp| dp.projection.0 == qualified) {
            Some(h) => h.clone(),
            None => return,
        };

        // Look up the arena offset for this projection.
        let offset = match self.retention_offsets.get(&hit.projection.0).copied() {
            Some(o) => o,
            None => {
                // Arena not yet declared (e.g., emit_retention_arena hasn't run
                // or the data_id is missing).  Emit a warning and skip.
                eprintln!(
                    "[calibration] retention offset missing for '{}' — splice skipped",
                    hit.projection.0
                );
                return;
            }
        };

        // Load the arena base pointer from the .bss global.
        let arena_data_id = match self.retention_arena_data_id {
            Some(id) => id,
            None => return, // arena not declared yet — skip
        };
        let arena_gv = self.module.declare_data_in_func(arena_data_id, builder.func);
        let arena_ptr = builder.ins().symbol_value(cl_types::I64, arena_gv);

        // Determine the data pointer inside the repr(C) NslTensor layout.
        const DATA_FIELD_OFFSET: i32 = nsl_runtime::tensor::NSL_TENSOR_DATA_OFFSET as i32;
        let data_ptr = builder.ins().load(
            cl_types::I64,
            MemFlags::new(),
            input_val,
            DATA_FIELD_OFFSET,
        );

        // Compute nbytes: batch * seq * in_features * 4.
        // These were baked into the arena layout at emit_retention_arena time.
        // Read from CompileOptions; fall back to (8, 4) for tests that set
        // calibration_retention without providing real calibration data.
        let (batch, seq) = self.compile_options.calibration_batch_seq.unwrap_or((8, 4));
        let in_features = hit.weight_shape[1];
        let nbytes = (batch * seq * in_features * 4) as u64;

        crate::calibration::retention::emit_splice_memcpy(
            builder,
            arena_ptr,
            offset,
            data_ptr,
            nbytes,
        );
        self.retention_splices_emitted = self.retention_splices_emitted.saturating_add(1);
    }

    /// Map FFI function name to a trace op type ID.
    fn trace_op_id(&self, fn_name: &str) -> u16 {
        match fn_name {
            "nsl_tensor_add" | "nsl_tensor_add_scalar" => 0,
            "nsl_tensor_sub" => 1,
            "nsl_tensor_mul" | "nsl_tensor_mul_scalar" => 2,
            "nsl_tensor_div" => 3,
            "nsl_tensor_matmul" => 4,
            "nsl_tensor_relu" => 5,
            "nsl_tensor_sigmoid" => 6,
            "nsl_tensor_softmax" => 7,
            "nsl_tensor_sum" | "nsl_tensor_sum_dim" => 8,
            "nsl_tensor_mean" | "nsl_tensor_mean_dim" => 9,
            _ => 255, // unknown op
        }
    }
}
