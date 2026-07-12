//! Typed-AST walker that dispatches recognized ops through the cost model
//! and produces a `ProfileReport`.
//!
//! This is the shared backbone for `nsl profile` and friends. Later tasks
//! (fusion, memory timeline, richer recommendations) layer on top of the
//! `ProfileReport` this walker returns.

use std::collections::{HashMap, HashSet};

use nsl_ast::{
    block::{TrainBlock, TrainSection},
    decl::{FnDef, ModelDef, ModelMember},
    expr::{Arg, Expr, ExprKind},
    operator::BinOp,
    stmt::{Block, Stmt, StmtKind},
    Module, Symbol,
};
use nsl_errors::Span;
use nsl_lexer::Interner;
use nsl_semantic::{
    checker::TypeMap,
    types::{Dim, Shape, Type},
    AnalysisResult,
};

use crate::cost_model::{
    arithmetic_intensity, classify_op, embedding_cost, flash_attention_cost, layernorm_cost,
    matmul_cost, rmsnorm_cost, softmax_cost, OpCost,
};
use crate::gpu_specs::GpuSpec;
use crate::profiling::shape_env::ShapeEnv;
use crate::profiling::types::{EntryKind, ProfileReport, Recommendation};

// ---------------------------------------------------------------------------
// Public entrypoint
// ---------------------------------------------------------------------------

/// Walk a typed NSL AST and produce a roofline-based `ProfileReport`.
///
/// * `module` — the parsed Module.
/// * `analysis` — the output of `nsl_semantic::analyze`; used to look up
///   expression types by `NodeId`.
/// * `interner` — lexer/semantic string interner (for symbol names).
/// * `entry` — which function/block to start walking from.
/// * `env` — user-supplied dim bindings (e.g. `--batch=4 --seq=2048 --dim=D=768`).
/// * `gpu` — target GPU spec, used to compute estimated time.
/// * `dtype` — `"bf16" | "fp16" | "fp8" | "fp32"`.
pub fn walk_ops(
    module: &Module,
    analysis: &AnalysisResult,
    interner: &Interner,
    entry: EntryKind,
    env: &ShapeEnv,
    gpu: &GpuSpec,
    dtype: &str,
) -> Result<ProfileReport, String> {
    let dtype_bytes = dtype_to_bytes(dtype)?;
    let mut ctx = WalkCtx {
        module,
        type_map: &analysis.type_map,
        interner,
        env,
        gpu,
        dtype,
        dtype_bytes,
        ops: Vec::new(),
        visited: HashSet::new(),
    };

    // Resolve entry point and dispatch.
    match &entry {
        EntryKind::Auto => {
            if let Some(tb) = find_train_block(module) {
                ctx.walk_train_block(tb);
            } else if let Some(f) = find_sole_top_fn(module) {
                ctx.walk_block(&f.body);
            } else {
                return Err("entry=auto: no train block or top-level fn found".into());
            }
        }
        EntryKind::Train => {
            let tb = find_train_block(module)
                .ok_or_else(|| "entry=train: no train block in module".to_string())?;
            ctx.walk_train_block(tb);
        }
        EntryKind::Function(name) => {
            let f = find_fn_by_name(module, interner, name)
                .ok_or_else(|| format!("entry=fn:{name}: function not found"))?;
            ctx.walk_block(&f.body);
        }
    }

    Ok(build_report(ctx, entry))
}

// ---------------------------------------------------------------------------
// Walk context
// ---------------------------------------------------------------------------

struct WalkCtx<'a> {
    module: &'a Module,
    type_map: &'a TypeMap,
    interner: &'a Interner,
    env: &'a ShapeEnv,
    gpu: &'a GpuSpec,
    dtype: &'a str,
    dtype_bytes: u64,
    ops: Vec<OpCost>,
    /// Guard against infinite recursion across `(model_name, method_name)`.
    visited: HashSet<(String, String)>,
}

impl<'a> WalkCtx<'a> {
    // --- block / stmt / expr descent -----------------------------------------

    fn walk_train_block(&mut self, tb: &TrainBlock) {
        for section in &tb.sections {
            if let TrainSection::Step { body, .. } = section {
                self.walk_block(body);
            }
        }
    }

    fn walk_block(&mut self, block: &Block) {
        for s in &block.stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::VarDecl { value: Some(e), .. } => self.walk_expr(e),
            StmtKind::Return(Some(e)) | StmtKind::Expr(e) => self.walk_expr(e),
            StmtKind::Assign { value, .. } => self.walk_expr(value),
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.walk_expr(condition);
                self.walk_block(then_block);
                for (c, b) in elif_clauses {
                    self.walk_expr(c);
                    self.walk_block(b);
                }
                if let Some(eb) = else_block {
                    self.walk_block(eb);
                }
            }
            StmtKind::For { iterable, body, .. } => {
                self.walk_expr(iterable);
                self.walk_block(body);
            }
            StmtKind::While { condition, body } => {
                self.walk_expr(condition);
                self.walk_block(body);
            }
            StmtKind::Decorated { stmt: inner, .. } => self.walk_stmt(inner),
            _ => {}
        }
    }

    /// Walk an expression, recursively visiting sub-expressions and, when
    /// the expression itself is a dispatchable op, emitting an `OpCost`.
    fn walk_expr(&mut self, expr: &Expr) {
        // First, recognize the current expression's top-level op.
        self.try_emit_op(expr);

        // Then always descend into sub-expressions so chained/nested calls
        // are captured too.
        match &expr.kind {
            ExprKind::BinaryOp { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            ExprKind::UnaryOp { operand, .. } => self.walk_expr(operand),
            ExprKind::Pipe { left, right } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            ExprKind::MemberAccess { object, .. } => self.walk_expr(object),
            ExprKind::Call { callee, args } => {
                self.walk_expr(callee);
                for a in args {
                    self.walk_expr(&a.value);
                }
            }
            ExprKind::Paren(e) | ExprKind::Await(e) => self.walk_expr(e),
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.walk_expr(condition);
                self.walk_expr(then_expr);
                self.walk_expr(else_expr);
            }
            ExprKind::TupleLiteral(xs) | ExprKind::ListLiteral(xs) => {
                for e in xs { self.walk_expr(e); }
            }
            _ => {}
        }
    }

    // --- op recognition ------------------------------------------------------

    /// If `expr` looks like a recognized tensor op call, push an `OpCost`.
    fn try_emit_op(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::BinaryOp { left, op: BinOp::MatMul, right } => {
                self.emit_matmul(expr, Some(left), Some(right));
            }
            ExprKind::Call { callee, args } => {
                if let Some(name) = self.callee_name(callee) {
                    self.emit_named_call(expr, &name, args, callee);
                } else {
                    // Unresolvable callee — best-effort unknown marker.
                    self.emit_unknown(expr, "<dyn>");
                }
            }
            _ => {}
        }
    }

    fn emit_named_call(&mut self, expr: &Expr, name: &str, args: &[Arg], callee: &Expr) {
        match name {
            "matmul" => {
                let a = args.first().map(|a| &a.value);
                let b = args.get(1).map(|a| &a.value);
                self.emit_matmul(expr, a, b);
            }
            "flash_attn" | "flash_attention" => self.emit_flash_attention(expr),
            "softmax" => self.emit_softmax(expr),
            "layernorm" => self.emit_norm(expr, "layernorm"),
            "rmsnorm" => self.emit_norm(expr, "rmsnorm"),
            "embedding" => self.emit_embedding(expr),
            "forward" | "forward_train" => {
                // `obj.forward(...)` where callee was a MemberAccess.
                self.try_inline_model_method(expr, callee, name);
            }
            other => {
                // Some calls might be model methods accessed via obj.foo — if
                // callee is a MemberAccess, try inlining. Otherwise emit unknown.
                if matches!(&callee.kind, ExprKind::MemberAccess { .. }) {
                    if !self.try_inline_model_method(expr, callee, other) {
                        self.emit_unknown(expr, other);
                    }
                } else {
                    self.emit_unknown(expr, other);
                }
            }
        }
    }

    // --- per-op emitters -----------------------------------------------------

    fn emit_matmul(&mut self, expr: &Expr, a: Option<&Expr>, b: Option<&Expr>) {
        let out_shape = self.expr_shape(expr);
        let a_shape = a.and_then(|e| self.expr_shape(e));
        let b_shape = b.and_then(|e| self.expr_shape(e));

        // M,K from A's last two dims; N from B's last dim (or out's last dim).
        let (m, k, n, resolved) = match (&a_shape, &b_shape, &out_shape) {
            (Some(av), Some(bv), _) if av.len() >= 2 && bv.len() >= 2 => {
                let m = av[av.len() - 2];
                let k = av[av.len() - 1];
                let n = bv[bv.len() - 1];
                let batch: u64 = av[..av.len() - 2].iter().product::<u64>().max(1);
                (batch * m, k, n, true)
            }
            (_, _, Some(ov)) if ov.len() >= 2 => {
                // Fall back on output shape; K is unresolved → flops=0.
                let m = ov[ov.len() - 2];
                let n = ov[ov.len() - 1];
                (m, 0, n, false)
            }
            _ => (0, 0, 0, false),
        };

        let (flops, br, bw) = if resolved && k > 0 {
            matmul_cost(m, k, n, self.dtype_bytes)
        } else {
            (0, 0, 0)
        };
        let loc_extra = if resolved { "" } else { "unresolved shape" };
        self.push_op(
            expr,
            "matmul",
            flops,
            br,
            bw,
            shape_strs(&[a_shape.as_deref(), b_shape.as_deref()]),
            shape_str(out_shape.as_deref()),
            loc_extra,
        );
    }

    fn emit_flash_attention(&mut self, expr: &Expr) {
        let out = self.expr_shape(expr);
        match out.as_deref() {
            Some(d) if d.len() == 4 => {
                let (b, h, s, dd) = (d[0], d[1], d[2], d[3]);
                let (f, br, bw) = flash_attention_cost(b, h, s, dd, self.dtype_bytes);
                self.push_op(expr, "flash_attention", f, br, bw, vec![], shape_str(Some(d)), "");
            }
            _ => self.push_op(expr, "flash_attention", 0, 0, 0, vec![], shape_str(out.as_deref()), "unresolved shape"),
        }
    }

    fn emit_softmax(&mut self, expr: &Expr) {
        let out = self.expr_shape(expr);
        match out.as_deref() {
            Some(d) if !d.is_empty() => {
                let last = *d.last().unwrap();
                let outer: u64 = d[..d.len() - 1].iter().product::<u64>().max(1);
                let (f, br, bw) = softmax_cost(outer, last, self.dtype_bytes);
                self.push_op(expr, "softmax", f, br, bw, vec![], shape_str(Some(d)), "");
            }
            _ => self.push_op(expr, "softmax", 0, 0, 0, vec![], shape_str(out.as_deref()), "unresolved shape"),
        }
    }

    fn emit_norm(&mut self, expr: &Expr, kind: &str) {
        let out = self.expr_shape(expr);
        match out.as_deref() {
            Some(d) if d.len() >= 2 => {
                let dd = *d.last().unwrap();
                let s = if d.len() >= 3 { d[d.len() - 2] } else { 1 };
                let b: u64 = d[..d.len() - 2].iter().product::<u64>().max(1);
                let (f, br, bw) = if kind == "layernorm" {
                    layernorm_cost(b, s, dd, self.dtype_bytes)
                } else {
                    rmsnorm_cost(b, s, dd, self.dtype_bytes)
                };
                self.push_op(expr, kind, f, br, bw, vec![], shape_str(Some(d)), "");
            }
            _ => self.push_op(expr, kind, 0, 0, 0, vec![], shape_str(out.as_deref()), "unresolved shape"),
        }
    }

    fn emit_embedding(&mut self, expr: &Expr) {
        let out = self.expr_shape(expr);
        match out.as_deref() {
            Some(d) if d.len() == 3 => {
                let (b, s, dd) = (d[0], d[1], d[2]);
                let (f, br, bw) = embedding_cost(b, s, dd, self.dtype_bytes);
                self.push_op(expr, "embedding", f, br, bw, vec![], shape_str(Some(d)), "");
            }
            _ => self.push_op(expr, "embedding", 0, 0, 0, vec![], shape_str(out.as_deref()), "unresolved shape"),
        }
    }

    fn emit_unknown(&mut self, expr: &Expr, name: &str) {
        self.push_op(expr, &format!("unknown:{name}"), 0, 0, 0, vec![], shape_str(self.expr_shape(expr).as_deref()), "unknown op");
    }

    fn push_op(
        &mut self,
        expr: &Expr,
        name: &str,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
        input_shapes: Vec<String>,
        output_shape: String,
        note: &str,
    ) {
        let ai = arithmetic_intensity(flops, bytes_read, bytes_written);
        let classification = classify_op(ai, self.gpu.crossover_fp16);
        let loc = format_loc(expr.span, note);
        let estimated_time_us = estimate_time_us(flops, bytes_read, bytes_written, self.gpu, self.dtype);
        self.ops.push(OpCost {
            name: name.to_string(),
            loc,
            input_shapes,
            output_shape,
            flops,
            bytes_read,
            bytes_written,
            arithmetic_intensity: ai,
            classification,
            fused: false,
            estimated_time_us,
            origin_node: Some(expr.id),
        });
    }

    // --- shape resolution ----------------------------------------------------

    fn expr_shape(&self, expr: &Expr) -> Option<Vec<u64>> {
        let ty = self.type_map.get(&expr.id)?;
        let (shape, _dt, _dev) = ty.as_tensor_parts()?;
        resolve_shape(shape, self.interner, self.env)
    }

    // --- callee name / model inlining ---------------------------------------

    fn callee_name(&self, callee: &Expr) -> Option<String> {
        match &callee.kind {
            ExprKind::Ident(sym) => Some(sym_str(self.interner, *sym)),
            ExprKind::MemberAccess { member, .. } => Some(sym_str(self.interner, *member)),
            _ => None,
        }
    }

    /// Try to resolve `obj.method(...)` to a `ModelDef`'s method and walk its
    /// body. Returns true if inlined.
    fn try_inline_model_method(&mut self, _call_expr: &Expr, callee: &Expr, method_name: &str) -> bool {
        // callee should be `obj.method`. Figure out `obj`'s type → Model name.
        let obj = match &callee.kind {
            ExprKind::MemberAccess { object, .. } => object.as_ref(),
            _ => return false,
        };
        let Some(ty) = self.type_map.get(&obj.id) else { return false; };
        let model_name = match ty.strip_borrow() {
            Type::Model { name, .. } => sym_str(self.interner, *name),
            _ => return false,
        };

        let key = (model_name.clone(), method_name.to_string());
        if self.visited.contains(&key) {
            return true; // avoid recursion, silently skip
        }

        let Some(model) = find_model_by_name(self.module, self.interner, &model_name) else {
            return false;
        };
        // Prefer forward_train when we're in train entry mode and user called "forward".
        let Some(method) = find_model_method(model, self.interner, method_name) else {
            return false;
        };

        self.visited.insert(key);
        let body = method.body.clone();
        self.walk_block(&body);
        true
    }
}

// ---------------------------------------------------------------------------
// Small free helpers
// ---------------------------------------------------------------------------

fn dtype_to_bytes(dtype: &str) -> Result<u64, String> {
    match dtype {
        "bf16" | "fp16" => Ok(2),
        "fp8" => Ok(1),
        "fp32" => Ok(4),
        other => Err(format!("unsupported dtype `{other}` (expected bf16|fp16|fp8|fp32)")),
    }
}

fn gpu_peak_tflops(gpu: &GpuSpec, dtype: &str) -> f64 {
    match dtype {
        "bf16" | "fp16" => gpu.peak_fp16_tflops,
        "fp8" => {
            if gpu.peak_fp8_tflops > 0.0 {
                gpu.peak_fp8_tflops
            } else {
                gpu.peak_fp16_tflops
            }
        }
        _ => gpu.peak_fp32_tflops,
    }
}

fn estimate_time_us(flops: u64, br: u64, bw: u64, gpu: &GpuSpec, dtype: &str) -> f64 {
    let peak = gpu_peak_tflops(gpu, dtype).max(1e-9);
    let compute_us = flops as f64 / (peak * 1e6);
    let bw_gbs = gpu.peak_bandwidth_gbs.max(1e-9);
    let mem_us = (br + bw) as f64 / (bw_gbs * 1e3);
    compute_us.max(mem_us)
}

fn resolve_shape(shape: &Shape, interner: &Interner, env: &ShapeEnv) -> Option<Vec<u64>> {
    let mut out = Vec::with_capacity(shape.dims.len());
    for d in &shape.dims {
        match resolve_dim(d, interner, env) {
            Some(v) => out.push(v),
            None => return None,
        }
    }
    Some(out)
}

fn resolve_dim(d: &Dim, interner: &Interner, env: &ShapeEnv) -> Option<u64> {
    match d {
        Dim::Concrete(n) => Some((*n).max(0) as u64),
        Dim::Named { name, size } => {
            // Prefer the attached size when concrete; otherwise try the name.
            if let Some(v) = resolve_dim(size, interner, env) {
                return Some(v);
            }
            env.resolve(&sym_str(interner, *name))
        }
        Dim::Symbolic(name) => env.resolve(&sym_str(interner, *name)),
        Dim::Bounded { upper_bound, .. } => Some((*upper_bound).max(0) as u64),
        Dim::Computed(_) | Dim::Wildcard => None,
    }
}

fn sym_str(interner: &Interner, s: Symbol) -> String {
    interner.resolve(s.0).map(|x| x.to_string()).unwrap_or_default()
}

fn shape_str(v: Option<&[u64]>) -> String {
    match v {
        Some(xs) => format!("[{}]", xs.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")),
        None => "<?>".to_string(),
    }
}

fn shape_strs(inputs: &[Option<&[u64]>]) -> Vec<String> {
    inputs.iter().map(|s| shape_str(*s)).collect()
}

fn format_loc(span: Span, note: &str) -> String {
    let base = format!("{}:{}-{}", span.file_id.0, span.start.0, span.end.0);
    if note.is_empty() {
        base
    } else {
        format!("{base} ({note})")
    }
}

// ---------------------------------------------------------------------------
// Entry-point lookup helpers
// ---------------------------------------------------------------------------

fn find_train_block(module: &Module) -> Option<&TrainBlock> {
    for s in &module.stmts {
        let st = unwrap_decorated_stmt(s);
        if let StmtKind::TrainBlock(tb) = &st.kind {
            return Some(tb);
        }
    }
    None
}

fn find_fn_by_name<'a>(module: &'a Module, interner: &Interner, name: &str) -> Option<&'a FnDef> {
    for s in &module.stmts {
        let st = unwrap_decorated_stmt(s);
        if let StmtKind::FnDef(f) = &st.kind {
            if sym_str(interner, f.name) == name {
                return Some(f);
            }
        }
    }
    None
}

fn find_sole_top_fn(module: &Module) -> Option<&FnDef> {
    let mut found: Option<&FnDef> = None;
    for s in &module.stmts {
        let st = unwrap_decorated_stmt(s);
        if let StmtKind::FnDef(f) = &st.kind {
            if found.is_some() {
                return None;
            }
            found = Some(f);
        }
    }
    found
}

fn find_model_by_name<'a>(module: &'a Module, interner: &Interner, name: &str) -> Option<&'a ModelDef> {
    for s in &module.stmts {
        let st = unwrap_decorated_stmt(s);
        if let StmtKind::ModelDef(m) = &st.kind {
            if sym_str(interner, m.name) == name {
                return Some(m);
            }
        }
    }
    None
}

fn find_model_method<'a>(model: &'a ModelDef, interner: &Interner, name: &str) -> Option<&'a FnDef> {
    let mut forward_train = None;
    let mut forward = None;
    let mut exact = None;
    for m in &model.members {
        if let ModelMember::Method(f, _) = m {
            let n = sym_str(interner, f.name);
            if n == name { exact = Some(f); }
            if n == "forward_train" { forward_train = Some(f); }
            if n == "forward" { forward = Some(f); }
        }
    }
    if name == "forward" {
        // Prefer forward_train when present (training walk), else forward.
        forward_train.or(forward).or(exact)
    } else {
        exact
    }
}

fn unwrap_decorated_stmt(s: &Stmt) -> &Stmt {
    match &s.kind {
        StmtKind::Decorated { stmt: inner, .. } => unwrap_decorated_stmt(inner),
        _ => s,
    }
}

// ---------------------------------------------------------------------------
// Report assembly
// ---------------------------------------------------------------------------

fn build_report(ctx: WalkCtx<'_>, entry: EntryKind) -> ProfileReport {
    let ops = ctx.ops;
    let total_flops: u64 = ops.iter().map(|o| o.flops).sum();
    let total_hbm_bytes: u64 = ops.iter().map(|o| o.bytes_read + o.bytes_written).sum();
    let total_estimated_us: f64 = ops.iter().map(|o| o.estimated_time_us).sum();

    // Minimal recommendations: memory-bound hint on dominant mem-bound ops,
    // and a "dominating_op" note if any op is >10% of total time.
    let mut recommendations = Vec::new();
    if total_estimated_us > 0.0 {
        // Dominating op by estimated_time_us.
        let mut by_time: HashMap<&str, f64> = HashMap::new();
        for o in &ops {
            *by_time.entry(o.name.as_str()).or_insert(0.0) += o.estimated_time_us;
        }
        if let Some((name, us)) = by_time.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            let pct = us / total_estimated_us * 100.0;
            if pct > 10.0 {
                recommendations.push(Recommendation::dominating_op(name, pct));
            }
        }
    }
    for o in &ops {
        if o.flops > 0 && o.arithmetic_intensity > 0.0 && o.arithmetic_intensity < 2.0 {
            recommendations.push(Recommendation::memory_bound_batch_hint(&o.name));
            break; // one hint is enough
        }
    }

    ProfileReport {
        target_gpu: ctx.gpu.name.to_string(),
        dtype: ctx.dtype.to_string(),
        entry,
        ops,
        total_flops,
        total_hbm_bytes,
        total_estimated_us,
        fusion: None,
        memory_timeline: None,
        memory_timeline_approximate: None,
        memory_what_if: None,
        memory_peak_bytes: None,
        memory_unsized_vars: None,
        memory_total_vars: None,
        recommendations,
        wggo_explain: None,
    }
}
