use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_binary_op(&mut self, left: &Expr, op: BinOp, right: &Expr, span: Span) -> Type {
        let lty = self.check_expr(left);
        let rty = self.check_expr(right);

        if lty.is_indeterminate() || rty.is_indeterminate() {
            return if matches!(lty, Type::Error) || matches!(rty, Type::Error) {
                Type::Error
            } else {
                Type::Unknown
            };
        }

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow => {
                self.check_arithmetic(&lty, &rty, op, span)
            }
            BinOp::MatMul => self.check_matmul_op(&lty, &rty, span),
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                Type::Bool
            }
            BinOp::And | BinOp::Or => Type::Bool,
            BinOp::Is | BinOp::In => Type::Bool,
            BinOp::BitOr | BinOp::BitAnd => lty,
        }
    }

    pub(crate) fn check_arithmetic(&mut self, lty: &Type, rty: &Type, op: BinOp, span: Span) -> Type {
        match (lty, rty) {
            (Type::Int, Type::Int) => {
                if matches!(op, BinOp::Div) {
                    Type::Float
                } else {
                    Type::Int
                }
            }
            (Type::Float, Type::Float)
            | (Type::Int, Type::Float)
            | (Type::Float, Type::Int) => Type::Float,
            // Tensor element-wise ops (includes Param/Buffer)
            (l, r) if l.is_tensor() && r.is_tensor() => {
                let (ls, ld, ldev) = l.as_tensor_parts().unwrap();
                let (rs, rd, rdev) = r.as_tensor_parts().unwrap();
                if ldev != rdev
                    && !matches!(ldev, Device::Unknown)
                    && !matches!(rdev, Device::Unknown)
                {
                    self.diagnostics.push(
                        Diagnostic::error("cannot operate on tensors on different devices")
                            .with_label(span, format!("{} vs {}", display_device(&ldev), display_device(&rdev))),
                    );
                }
                match shapes::check_elementwise(ls, rs, span) {
                    Ok(result_shape) => Type::Tensor {
                        shape: result_shape,
                        dtype: wider_dtype(*ld, *rd),
                        device: ldev,
                    },
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        Type::Error
                    }
                }
            }
            // Scalar + tensor (includes Param/Buffer)
            (l, Type::Int | Type::Float) if l.is_tensor() => lty.clone(),
            (Type::Int | Type::Float, r) if r.is_tensor() => rty.clone(),
            // String concatenation
            (Type::Str, Type::Str) if matches!(op, BinOp::Add) => Type::Str,
            // String repeat
            (Type::Str, Type::Int) if matches!(op, BinOp::Mul) => Type::Str,
            (Type::Int, Type::Str) if matches!(op, BinOp::Mul) => Type::Str,
            // Specific numeric types (e.g. f32 + f32, int32 + int32)
            (l, r) if dtype_rank(l).0 > 0 && dtype_rank(r).0 > 0 => {
                let (lf, lr) = dtype_rank(l);
                let (rf, rr) = dtype_rank(r);
                if lf == rf {
                    // Same family: return the wider type
                    if lr >= rr { l.clone() } else { r.clone() }
                } else {
                    // Mixed int/float: promote to float side
                    if lf == 2 { l.clone() } else { r.clone() }
                }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "unsupported operand types for {:?}: {} and {}",
                        op, display_type(lty), display_type(rty)
                    ))
                    .with_label(span, "type error"),
                );
                Type::Error
            }
        }
    }

    pub(crate) fn check_matmul_op(&mut self, lty: &Type, rty: &Type, span: Span) -> Type {
        // Normalize Param/Buffer to Tensor for type-checking
        match (lty.as_tensor_parts(), rty.as_tensor_parts()) {
            (Some((ls, ld, ldev)), Some((rs, rd, rdev))) => {
                if ldev != rdev
                    && !matches!(ldev, Device::Unknown)
                    && !matches!(rdev, Device::Unknown)
                {
                    self.diagnostics.push(
                        Diagnostic::error("matmul: device mismatch")
                            .with_label(span, format!("{} vs {}", display_device(&ldev), display_device(&rdev))),
                    );
                }
                match shapes::check_matmul(ls, rs, span) {
                    Ok(result_shape) => Type::Tensor {
                        shape: result_shape,
                        dtype: wider_dtype(*ld, *rd),
                        device: ldev,
                    },
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        Type::Error
                    }
                }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error("@ (matmul) requires tensor operands")
                        .with_label(span, "not a tensor"),
                );
                Type::Error
            }
        }
    }

    pub(crate) fn check_unary_op(&mut self, op: UnaryOp, operand: &Expr, _span: Span) -> Type {
        let ty = self.check_expr(operand);
        match op {
            UnaryOp::Neg => match &ty {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                Type::Tensor { .. } => ty,
                _ => ty,
            },
            UnaryOp::Not => Type::Bool,
        }
    }

    pub(crate) fn check_call(&mut self, callee: &Expr, args: &[Arg], span: Span) -> Type {
        let callee_ty = self.check_expr(callee);

        // Check each argument
        let arg_types: Vec<Type> = args.iter().map(|a| self.check_expr(&a.value)).collect();

        // Special type inference for enumerate(list) -> List(Tuple(Int, T))
        if let ExprKind::Ident(sym) = &callee.kind {
            let name = self.interner.resolve(sym.0).unwrap_or("").to_string();
            if name == "enumerate" {
                if let Some(Type::List(elem_ty)) = arg_types.first() {
                    return Type::List(Box::new(Type::Tuple(vec![Type::Int, *elem_ty.clone()])));
                }
            }
            if name == "zip" && arg_types.len() >= 2 {
                let a = match &arg_types[0] {
                    Type::List(t) => *t.clone(),
                    _ => Type::Unknown,
                };
                let b = match &arg_types[1] {
                    Type::List(t) => *t.clone(),
                    _ => Type::Unknown,
                };
                return Type::List(Box::new(Type::Tuple(vec![a, b])));
            }

            // Tensor creation shape inference
            if matches!(name.as_str(), "zeros" | "ones" | "rand" | "randn" | "empty") {
                let shape = self.extract_shape_from_args(args);
                return Type::Tensor {
                    shape,
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }
            if name == "full" {
                let shape = self.extract_shape_from_args(args);
                return Type::Tensor {
                    shape,
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }
            if name == "arange" {
                return Type::Tensor {
                    shape: Shape::unknown(),
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }

            // Math builtins (exp, log, sqrt, sin, cos, abs) — when called with tensor or
            // unknown args, return the arg type (tensor) instead of Float.
            if matches!(name.as_str(), "exp" | "log" | "sqrt" | "sin" | "cos" | "abs" | "neg" | "floor") {
                if let Some(first_arg_ty) = arg_types.first() {
                    if first_arg_ty.is_tensor() || first_arg_ty.is_indeterminate() {
                        return first_arg_ty.clone();
                    }
                }
            }

            // Tensor reduction / manipulation builtins — always return tensor-like
            if matches!(name.as_str(), "mean" | "sum" | "reduce_max" | "gather" | "clamp" | "neg") {
                if let Some(first_arg_ty) = arg_types.first() {
                    if first_arg_ty.is_tensor() || first_arg_ty.is_indeterminate() {
                        return first_arg_ty.clone();
                    }
                }
            }
        }

        // Tensor method shape inference (reshape, transpose)
        if let ExprKind::MemberAccess { object, member } = &callee.kind {
            let obj_ty = self.type_map.get(&object.id).cloned().unwrap_or(Type::Unknown);
            if obj_ty.is_tensor() {
                let method_name = self.interner.resolve(member.0).unwrap_or("").to_string();
                if let Type::Tensor { dtype, device, .. } = &obj_ty {
                    match method_name.as_str() {
                        "reshape" => {
                            let target_shape = self.extract_shape_from_args(args);
                            if let Type::Tensor { shape: source_shape, .. } = &obj_ty {
                                // M49: Prove reshape correctness via shape algebra
                                if !source_shape.dims.is_empty() && !target_shape.dims.is_empty() {
                                    let mut solver = crate::shape_algebra::ShapeAlgebraSolver::new();
                                    let src_expr = Self::shape_to_product(source_shape, &mut solver);
                                    let tgt_expr = Self::shape_to_product(&target_shape, &mut solver);
                                    if let (Some(src), Some(tgt)) = (src_expr, tgt_expr) {
                                        if let Err(pf) = solver.prove_eq_normalized(&src, &tgt) {
                                            // Only warn — runtime check still active as fallback
                                            self.diagnostics.push(
                                                Diagnostic::warning(format!(
                                                    "reshape may be invalid: {}", pf.reason
                                                ))
                                            );
                                        }
                                    }
                                }
                            }
                            return Type::Tensor {
                                shape: target_shape,
                                dtype: *dtype,
                                device: device.clone(),
                            };
                        }
                        "transpose" => {
                            if let Type::Tensor { shape, dtype, device } = &obj_ty {
                                if shape.rank() >= 2 && args.len() >= 2 {
                                    let d0 = match &args[0].value.kind {
                                        ExprKind::IntLiteral(n) => Some(*n as usize),
                                        _ => None,
                                    };
                                    let d1 = match &args[1].value.kind {
                                        ExprKind::IntLiteral(n) => Some(*n as usize),
                                        _ => None,
                                    };
                                    if let (Some(d0), Some(d1)) = (d0, d1) {
                                        if d0 < shape.rank() && d1 < shape.rank() {
                                            let mut new_dims = shape.dims.clone();
                                            new_dims.swap(d0, d1);
                                            return Type::Tensor {
                                                shape: Shape { dims: new_dims },
                                                dtype: *dtype,
                                                device: device.clone(),
                                            };
                                        }
                                    }
                                }
                                // Can't determine statically — return unknown shape
                                return Type::Tensor {
                                    shape: Shape::unknown(),
                                    dtype: *dtype,
                                    device: device.clone(),
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        match &callee_ty {
            Type::Function { params, ret } => {
                // Check arity (allow flexible arity if params has Unknown — variadic builtins)
                let is_variadic = params.iter().any(|p| matches!(p, Type::Unknown));
                if arg_types.len() > params.len() && !is_variadic {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "too many arguments: expected {}, got {}",
                            params.len(),
                            arg_types.len()
                        ))
                        .with_label(span, "extra arguments"),
                    );
                }
                if arg_types.len() < params.len() && !is_variadic {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "too few arguments: expected {}, got {}",
                            params.len(),
                            arg_types.len()
                        ))
                        .with_label(span, "missing arguments"),
                    );
                }
                // Check each arg type against param type
                for (i, (arg_ty, param_ty)) in
                    arg_types.iter().zip(params.iter()).enumerate()
                {
                    if !is_assignable(arg_ty, param_ty) {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "argument {}: expected {}, got {}",
                                i + 1,
                                display_type(param_ty),
                                display_type(arg_ty)
                            ))
                            .with_label(args[i].span, "type mismatch"),
                        );
                    }
                }
                *ret.clone()
            }
            Type::Model { .. } => callee_ty.clone(),
            Type::Struct { .. } => callee_ty.clone(),
            Type::Error => Type::Error,
            Type::Unknown => Type::Unknown,
            _ => {
                self.diagnostics.push(
                    Diagnostic::error("expression is not callable")
                        .with_label(span, format!("type {} is not callable", display_type(&callee_ty))),
                );
                Type::Error
            }
        }
    }

    /// Try to extract a concrete tensor shape from a function call's arguments.
    /// Returns Shape::unknown() if the first arg isn't a list literal or contains non-integer elements.
    /// Convert a Shape's dimensions into a product DimExpr for the solver.
    /// Also feeds bounds from Bounded dims into the solver.
    /// Returns None for empty/wildcard shapes.
    fn shape_to_product(
        shape: &Shape,
        solver: &mut crate::shape_algebra::ShapeAlgebraSolver,
    ) -> Option<DimExpr> {
        let dim_to_expr = |dim: &Dim| -> Option<DimExpr> {
            match dim {
                Dim::Concrete(v) => Some(DimExpr::Lit(*v)),
                Dim::Symbolic(s) => Some(DimExpr::Sym(*s)),
                Dim::Named { size, .. } => match size.as_ref() {
                    Dim::Concrete(v) => Some(DimExpr::Lit(*v)),
                    Dim::Symbolic(s) => Some(DimExpr::Sym(*s)),
                    _ => None,
                },
                Dim::Bounded { name, upper_bound } => {
                    solver.assert_bound(*name, Some(1), Some(*upper_bound));
                    Some(DimExpr::Sym(*name))
                }
                Dim::Computed(expr) => Some(expr.as_ref().clone()),
                Dim::Wildcard => None,
            }
        };

        let exprs: Vec<DimExpr> = shape.dims.iter().filter_map(dim_to_expr).collect();
        if exprs.is_empty() {
            return None;
        }
        let mut product = exprs[0].clone();
        for e in &exprs[1..] {
            product = DimExpr::Mul(Box::new(product), Box::new(e.clone()));
        }
        Some(product)
    }

    pub(crate) fn extract_shape_from_args(&self, args: &[Arg]) -> Shape {
        if args.is_empty() {
            return Shape::unknown();
        }
        match &args[0].value.kind {
            ExprKind::ListLiteral(elems) => {
                let mut dims = Vec::new();
                for elem in elems {
                    match &elem.kind {
                        ExprKind::IntLiteral(n) => dims.push(Dim::Concrete(*n)),
                        ExprKind::Ident(sym) => dims.push(Dim::Symbolic(*sym)),
                        _ => return Shape::unknown(),
                    }
                }
                Shape { dims }
            }
            _ => Shape::unknown(),
        }
    }

    pub(crate) fn check_subscript_kind(&mut self, kind: &nsl_ast::expr::SubscriptKind) {
        match kind {
            nsl_ast::expr::SubscriptKind::Index(expr) => {
                self.check_expr(expr);
            }
            nsl_ast::expr::SubscriptKind::Slice { lower, upper, step } => {
                if let Some(e) = lower { self.check_expr(e); }
                if let Some(e) = upper { self.check_expr(e); }
                if let Some(e) = step { self.check_expr(e); }
            }
            nsl_ast::expr::SubscriptKind::MultiDim(dims) => {
                for d in dims {
                    self.check_subscript_kind(d);
                }
            }
        }
    }

    pub(crate) fn check_member_access(&mut self, object: &Expr, member: Symbol, span: Span) -> Type {
        let obj_ty = self.check_expr(object);

        match &obj_ty {
            Type::Module { exports } => {
                if let Some(ty) = exports.get(&member) {
                    return *ty.clone();
                }
                let name = self.resolve_name(member);
                self.diagnostics.push(
                    Diagnostic::error(format!("module has no export `{name}`"))
                        .with_label(span, "not found in module"),
                );
                Type::Error
            }
            Type::Struct { fields, .. } => {
                if let Some((_, field_ty)) = fields.iter().find(|(name, _)| *name == member) {
                    field_ty.clone()
                } else {
                    let name = self.resolve_name(member);
                    self.diagnostics.push(
                        Diagnostic::error(format!("no field `{name}` on struct"))
                            .with_label(span, "unknown field"),
                    );
                    Type::Error
                }
            }
            Type::Model { fields, methods, .. } => {
                if let Some((_, field_ty)) = fields.iter().find(|(name, _)| *name == member) {
                    return field_ty.clone();
                }
                if let Some((_, method_ty)) = methods.iter().find(|(name, _)| *name == member) {
                    return method_ty.clone();
                }
                // Models have many dynamic attributes, be lenient
                Type::Unknown
            }
            Type::Tensor { .. } | Type::Param { .. } | Type::Buffer { .. } => {
                // Tensors have built-in attributes
                let name = self.resolve_name(member);
                match name.as_str() {
                    "shape" => Type::List(Box::new(Type::Int)),
                    "ndim" => Type::Int,
                    "dtype" | "device" => Type::Unknown,
                    "T" => obj_ty.clone(), // transpose
                    "sum" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Tensor {
                            shape: crate::types::Shape::unknown(),
                            dtype: crate::types::DType::Unknown,
                            device: crate::types::Device::Unknown,
                        }),
                    },
                    "mean" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Tensor {
                            shape: crate::types::Shape::unknown(),
                            dtype: crate::types::DType::Unknown,
                            device: crate::types::Device::Unknown,
                        }),
                    },
                    "reshape" => Type::Function {
                        params: vec![Type::List(Box::new(Type::Int))],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "transpose" => Type::Function {
                        params: vec![Type::Int, Type::Int],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "clone" => Type::Function {
                        params: vec![],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "item" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Float),
                    },
                    "to" => Type::Function {
                        params: vec![Type::Int],
                        ret: Box::new(Type::Tensor {
                            shape: crate::types::Shape::unknown(),
                            dtype: crate::types::DType::Unknown,
                            device: crate::types::Device::Unknown,
                        }),
                    },
                    _ => Type::Unknown,    // tensor methods
                }
            }
            Type::Str => {
                let name = self.resolve_name(member);
                match name.as_str() {
                    // String methods that return strings
                    "upper" | "lower" | "strip" | "replace" | "join" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Str),
                    },
                    // String methods that return lists
                    "split" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::List(Box::new(Type::Str))),
                    },
                    // String methods that return int
                    "find" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Int),
                    },
                    // String methods that return bool
                    "startswith" | "endswith" | "contains" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Bool),
                    },
                    _ => Type::Unknown,
                }
            }
            Type::List(_) => {
                let name = self.resolve_name(member);
                match name.as_str() {
                    "append" | "push" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Void),
                    },
                    "len" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Int),
                    },
                    _ => Type::Unknown,
                }
            }
            Type::Error => Type::Error,
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        }
    }
}
