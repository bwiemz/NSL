use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::Expr;
use nsl_ast::operator::BinOp;
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::is_float_type;

impl Compiler<'_> {
    /// ELTLS: compute the FBIP-3 flag byte for a tensor binary op from operand
    /// ownership. Returns a Cranelift I8 constant Value. In a future task
    /// only operands with `Ownership::Owned` will have their relinquish bit
    /// set — `BorrowedFromVar`, `BorrowedWeight`, `TapeHeld`, and `Unknown`
    /// will all leave the bit clear.
    ///
    /// Currently returns 0 unconditionally. The reason is that the autodiff
    /// tape is GLOBAL runtime state — when a train/grad block runs, the
    /// tape is live across every function call transitively reachable from
    /// the step body, including model methods (forward_train etc.). Those
    /// methods are compiled as independent Cranelift functions with
    /// `in_tape_region = false`; the codegen has no per-function knowledge
    /// of whether a callee may execute while a tape is active.
    ///
    /// Relinquishing an operand inside such a callee violates the tape's
    /// invariant (the tape holds a raw pointer to input data). Until we
    /// have an interprocedural analysis that propagates "may run under
    /// tape" through the call graph, we conservatively emit flags=0
    /// everywhere. The signature plumbing, consume_ownership bookkeeping,
    /// and phi-merge coordination all land in this task so later commits
    /// can flip the switch with a one-line change.
    fn compute_fbip_flags(
        &self,
        builder: &mut FunctionBuilder,
        _state: &FuncState,
        _lhs: Value,
        _rhs: Value,
    ) -> Value {
        builder.ins().iconst(cl_types::I8, 0)
    }

    pub(crate) fn compile_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        left: &Expr,
        op: BinOp,
        right: &Expr,
        _full_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.compile_short_circuit(builder, state, left, op, right);
        }

        if matches!(op, BinOp::In) {
            let lhs = self.compile_expr(builder, state, left)?;
            let rhs = self.compile_expr(builder, state, right)?;
            let rhs_type = self.node_type(right.id).clone();
            let fn_name = if matches!(rhs_type, Type::Dict(_, _)) {
                "nsl_dict_contains"
            } else {
                "nsl_list_contains"
            };
            let fid = self.registry.runtime_fns[fn_name].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(fref, &[rhs, lhs]);
            return Ok(builder.inst_results(call)[0]);
        }

        let mut lhs = self.compile_nested_expr(builder, state, left)?;
        let mut rhs = self.compile_nested_expr(builder, state, right)?;
        let left_type = self.node_type(left.id).clone();
        let right_type = self.node_type(right.id).clone();

        // String concatenation with +
        if matches!(op, BinOp::Add)
            && (matches!(left_type, Type::Str) || matches!(right_type, Type::Str))
        {
            // Convert non-string operand to string if needed
            if !matches!(left_type, Type::Str) {
                lhs = self.value_to_string(builder, lhs, &left_type)?;
            }
            if !matches!(right_type, Type::Str) {
                rhs = self.value_to_string(builder, rhs, &right_type)?;
            }
            let concat_id = self.registry.runtime_fns["nsl_str_concat"].0;
            let concat_ref = self.module.declare_func_in_func(concat_id, builder.func);
            let call = builder.ins().call(concat_ref, &[lhs, rhs]);
            return Ok(builder.inst_results(call)[0]);
        }

        // String repeat with *
        if matches!(op, BinOp::Mul)
            && (matches!(left_type, Type::Str) || matches!(right_type, Type::Str))
        {
            let (str_val, int_val) = if matches!(left_type, Type::Str) {
                (lhs, rhs)
            } else {
                (rhs, lhs)
            };
            let fid = self.registry.runtime_fns["nsl_str_repeat"].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(fref, &[str_val, int_val]);
            return Ok(builder.inst_results(call)[0]);
        }

        // Tensor operations
        // Unknown (indeterminate) types are treated as tensors when:
        //  - The op is MatMul (@ is tensor-only)
        //  - The other operand is a tensor
        //  - Both operands are indeterminate (likely from .to(device) or other runtime returns)
        // Exception: inside datatype method bodies, indeterminate types are always scalars
        //   (the semantic checker does not descend into method bodies).
        let both_indeterminate = left_type.is_indeterminate()
            && right_type.is_indeterminate()
            && !state.flags.in_dtype_method;
        let left_is_tensor = left_type.is_tensor()
            || (left_type.is_indeterminate()
                && !state.flags.in_dtype_method
                && (matches!(op, BinOp::MatMul) || right_type.is_tensor() || both_indeterminate));
        let right_is_tensor = right_type.is_tensor()
            || (right_type.is_indeterminate()
                && !state.flags.in_dtype_method
                && (matches!(op, BinOp::MatMul) || left_type.is_tensor() || both_indeterminate));
        if left_is_tensor || right_is_tensor {
            // M50: Extract sparse flags for sparse operation dispatch.
            // Check semantic type AND runtime sparse_vars tracking (for values
            // created by sparse_from_dense where the type checker doesn't
            // assign Type::Sparse).
            let left_is_sparse = matches!(left_type, Type::Sparse { .. })
                || matches!(&left.kind, nsl_ast::expr::ExprKind::Ident(sym) if state.ownership.sparse_vars.contains(sym));
            let right_is_sparse = matches!(right_type, Type::Sparse { .. })
                || matches!(&right.kind, nsl_ast::expr::ExprKind::Ident(sym) if state.ownership.sparse_vars.contains(sym));

            // ELTLS commit 5 (Task 13): plumb the FBIP-3 flag byte from
            // `compute_fbip_flags` through `compile_tensor_binary_op` so
            // future tasks can flip on relinquish behavior without
            // touching the dispatch signatures.
            //
            // Capture the PRE-call ownership state: if the flag byte ever
            // becomes non-zero and the runtime consumes an operand, we
            // must call `consume_ownership` to drop tracking before
            // `set_ownership_from_op` promotes the operand to TapeHeld.
            // Reading ownership post-call would see TapeHeld and miss the
            // consume.
            //
            // Currently `compute_fbip_flags` returns 0 unconditionally
            // (see its doc comment for the tape-crossing-function-boundary
            // rationale), so `consume_ownership` below is a dead branch
            // until the switch is flipped.
            use crate::ownership_expr::Ownership;
            let lhs_was_owned =
                matches!(self.get_ownership(state, lhs), Ownership::Owned);
            let rhs_was_owned =
                matches!(self.get_ownership(state, rhs), Ownership::Owned);
            let flags = self.compute_fbip_flags(builder, state, lhs, rhs);
            let result = self.compile_tensor_binary_op(
                builder,
                state,
                lhs,
                rhs,
                flags,
                op,
                left_is_tensor,
                right_is_tensor,
                left_is_sparse,
                right_is_sparse,
            )?;

            // Match `compute_fbip_flags`'s current policy: no operands
            // are relinquished, so no consume_ownership calls fire. When
            // the policy is refined this block activates automatically.
            let _relinquish_active = false;
            if _relinquish_active {
                if lhs_was_owned {
                    self.consume_ownership(state, lhs);
                }
                if rhs_was_owned {
                    self.consume_ownership(state, rhs);
                }
            }

            return Ok(result);
        }

        // When semantic type is Unknown (e.g., inside dtype method bodies), fall back
        // to checking the Cranelift value type to determine float vs int.
        let left_is_float = is_float_type(&left_type)
            || (left_type.is_indeterminate() && builder.func.dfg.value_type(lhs).is_float());
        let right_is_float = is_float_type(&right_type)
            || (right_type.is_indeterminate() && builder.func.dfg.value_type(rhs).is_float());
        let is_float = left_is_float || right_is_float;

        // Promote int -> float if mixed types
        if is_float {
            // Check actual Cranelift value types — the semantic type may claim "Float"
            // (e.g. Int / Int → Float) but the codegen may have emitted sdiv (i64).
            // Always check the actual IR value type to decide whether conversion is needed.
            let lhs_cl_ty = builder.func.dfg.value_type(lhs);
            if !lhs_cl_ty.is_float() {
                if lhs_cl_ty.bits() < 32 {
                    lhs = builder.ins().sextend(cl_types::I64, lhs);
                    lhs = builder.ins().fcvt_from_sint(cl_types::F64, lhs);
                } else {
                    lhs = builder.ins().fcvt_from_sint(cl_types::F64, lhs);
                }
            }
            let rhs_cl_ty = builder.func.dfg.value_type(rhs);
            if !rhs_cl_ty.is_float() {
                if rhs_cl_ty.bits() < 32 {
                    rhs = builder.ins().sextend(cl_types::I64, rhs);
                    rhs = builder.ins().fcvt_from_sint(cl_types::F64, rhs);
                } else {
                    rhs = builder.ins().fcvt_from_sint(cl_types::F64, rhs);
                }
            }
        }

        match op {
            BinOp::Add if is_float => Ok(builder.ins().fadd(lhs, rhs)),
            BinOp::Add => Ok(builder.ins().iadd(lhs, rhs)),
            BinOp::Sub if is_float => Ok(builder.ins().fsub(lhs, rhs)),
            BinOp::Sub => Ok(builder.ins().isub(lhs, rhs)),
            BinOp::Mul if is_float => Ok(builder.ins().fmul(lhs, rhs)),
            BinOp::Mul => Ok(builder.ins().imul(lhs, rhs)),
            BinOp::Div if is_float => Ok(builder.ins().fdiv(lhs, rhs)),
            BinOp::Div => {
                // Guard against division by zero (SIGFPE on x86)
                self.compile_divmod_guard(builder, state, rhs)?;
                Ok(builder.ins().sdiv(lhs, rhs))
            }
            BinOp::FloorDiv if is_float => {
                let div = builder.ins().fdiv(lhs, rhs);
                Ok(builder.ins().floor(div))
            }
            BinOp::FloorDiv => {
                // Floor division: round toward negative infinity, not toward zero.
                // sdiv truncates toward zero, so adjust when remainder is nonzero
                // and operands have different signs.
                self.compile_divmod_guard(builder, state, rhs)?;
                let q = builder.ins().sdiv(lhs, rhs);
                let prod = builder.ins().imul(q, rhs);
                let rem = builder.ins().isub(lhs, prod);
                // Check: remainder != 0 && (lhs ^ rhs) < 0
                let rem_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, rem, 0);
                let sign_bits = builder.ins().bxor(lhs, rhs);
                let diff_sign = builder.ins().icmp_imm(IntCC::SignedLessThan, sign_bits, 0);
                let needs_adjust = builder.ins().band(rem_nonzero, diff_sign);
                let one = builder.ins().iconst(cl_types::I64, 1);
                let zero = builder.ins().iconst(cl_types::I64, 0);
                let adjustment = builder.ins().select(needs_adjust, one, zero);
                Ok(builder.ins().isub(q, adjustment))
            }
            BinOp::Mod if is_float => {
                // a % b = a - floor(a/b) * b
                let div = builder.ins().fdiv(lhs, rhs);
                let floored = builder.ins().floor(div);
                let prod = builder.ins().fmul(floored, rhs);
                Ok(builder.ins().fsub(lhs, prod))
            }
            BinOp::Mod => {
                // Python-style modulo: result has same sign as divisor.
                // srem follows dividend sign, so adjust when result is nonzero
                // and operands have different signs.
                self.compile_divmod_guard(builder, state, rhs)?;
                let r = builder.ins().srem(lhs, rhs);
                let rem_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, r, 0);
                let sign_bits = builder.ins().bxor(lhs, rhs);
                let diff_sign = builder.ins().icmp_imm(IntCC::SignedLessThan, sign_bits, 0);
                let needs_adjust = builder.ins().band(rem_nonzero, diff_sign);
                let adjusted = builder.ins().iadd(r, rhs);
                Ok(builder.ins().select(needs_adjust, adjusted, r))
            }
            BinOp::Pow => self.compile_pow(builder, lhs, rhs, is_float),

            BinOp::Eq if is_float => Ok(builder.ins().fcmp(FloatCC::Equal, lhs, rhs)),
            BinOp::Eq if matches!(left_type, Type::Str) => {
                let fid = self.registry.runtime_fns["nsl_str_eq"].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[lhs, rhs]);
                let result_i64 = builder.inst_results(call)[0];
                // Narrow I64 result to I8 to match icmp return type
                Ok(builder.ins().ireduce(cl_types::I8, result_i64))
            }
            BinOp::Eq => Ok(builder.ins().icmp(IntCC::Equal, lhs, rhs)),
            BinOp::NotEq if is_float => Ok(builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs)),
            BinOp::NotEq if matches!(left_type, Type::Str) => {
                let fid = self.registry.runtime_fns["nsl_str_eq"].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[lhs, rhs]);
                let eq_i64 = builder.inst_results(call)[0];
                // Narrow to I8 and invert: not-equal = 1 - eq
                let eq_i8 = builder.ins().ireduce(cl_types::I8, eq_i64);
                let one = builder.ins().iconst(cl_types::I8, 1);
                Ok(builder.ins().isub(one, eq_i8))
            }
            BinOp::NotEq => Ok(builder.ins().icmp(IntCC::NotEqual, lhs, rhs)),
            BinOp::Lt if is_float => Ok(builder.ins().fcmp(FloatCC::LessThan, lhs, rhs)),
            BinOp::Lt => Ok(builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs)),
            BinOp::Gt if is_float => Ok(builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs)),
            BinOp::Gt => Ok(builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs)),
            BinOp::LtEq if is_float => Ok(builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs)),
            BinOp::LtEq => Ok(builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs)),
            BinOp::GtEq if is_float => {
                Ok(builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs))
            }
            BinOp::GtEq => Ok(builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs)),

            BinOp::BitAnd => Ok(builder.ins().band(lhs, rhs)),
            BinOp::BitOr => Ok(builder.ins().bor(lhs, rhs)),

            BinOp::And | BinOp::Or => unreachable!(),
            _ => Err(CodegenError::new(format!("unsupported binary op: {op:?}"))),
        }
    }

    pub(crate) fn compile_short_circuit(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        left: &Expr,
        op: BinOp,
        right: &Expr,
    ) -> Result<Value, CodegenError> {
        let lhs_raw = self.compile_expr(builder, state, left)?;
        // Normalize to I8 boolean (handles I64 values from comparisons etc.)
        let lhs = builder.ins().icmp_imm(IntCC::NotEqual, lhs_raw, 0);

        let rhs_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, cl_types::I8);

        match op {
            BinOp::And => builder.ins().brif(lhs, rhs_block, &[], merge_block, &[lhs]),
            BinOp::Or => builder.ins().brif(lhs, merge_block, &[lhs], rhs_block, &[]),
            _ => unreachable!(),
        };

        builder.switch_to_block(rhs_block);
        builder.seal_block(rhs_block);
        state.current_block = Some(rhs_block);
        let rhs_raw = self.compile_expr(builder, state, right)?;
        let rhs = builder.ins().icmp_imm(IntCC::NotEqual, rhs_raw, 0);
        builder.ins().jump(merge_block, &[rhs]);

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(builder.block_params(merge_block)[0])
    }

    /// Emit a runtime check that the integer divisor is non-zero.
    /// On x86, `sdiv` / `srem` with divisor=0 causes SIGFPE (hardware fault).
    pub(crate) fn compile_divmod_guard(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        divisor: Value,
    ) -> Result<(), CodegenError> {
        let ok_block = builder.create_block();
        let trap_block = builder.create_block();

        let is_zero = builder.ins().icmp_imm(IntCC::Equal, divisor, 0);
        builder.ins().brif(is_zero, trap_block, &[], ok_block, &[]);

        builder.switch_to_block(trap_block);
        builder.seal_block(trap_block);
        builder
            .ins()
            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));

        builder.switch_to_block(ok_block);
        builder.seal_block(ok_block);
        state.current_block = Some(ok_block);
        Ok(())
    }

    pub(crate) fn compile_pow(
        &mut self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
        is_float: bool,
    ) -> Result<Value, CodegenError> {
        let rt_fn = if is_float {
            "nsl_pow_float"
        } else {
            "nsl_pow_int"
        };
        let func_id = self.registry.runtime_fns[rt_fn].0;
        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, &[lhs, rhs]);
        Ok(builder.inst_results(call)[0])
    }
}
