// crates/nsl-codegen/src/kernel_lower.rs
//! AST `KernelDef` → KIR lowering: the one front door for user `kernel`
//! blocks on every GPU target (roadmap A2 step 3).
//!
//! Until step 3 the default CUDA target compiled a `kernel` block straight
//! from the AST to PTX text (`kernel.rs`, `KernelCompiler`) and only
//! `--target rocm|metal|webgpu` came through here, for a straight-line
//! subset. Now every target lowers to `KernelIR`, the verifier checks the
//! result, and the target's printer (`backend_ptx` for CUDA) renders it.
//! Stores, `if`/`elif`/`else`, `for ... in range(...)`, `while`, `break`,
//! `continue`, a bare `return` and assignment to a `let`-declared local are
//! lowered here; a local that is reassigned inside a branch or a loop body
//! is a block parameter at the join or the loop header (roadmap A2 step 2),
//! so the IR stays SSA without a phi.
//!
//! Deferral-must-refuse invariant: any construct this lowering cannot
//! express MUST produce a loud `CodegenError` instead of fabricating
//! placeholder values or silently dropping statements — the lenient
//! behaviour both predecessors started with compiled kernels that produced
//! wrong numbers at runtime (dropped bounds guards, uninitialised registers
//! for unknown calls). Errors carry the span of the innermost statement or
//! expression being lowered (`CodegenError::with_span_if_unset`, as in the
//! Cranelift dispatchers).
//!
//! ## Type inference
//! Parameters carry their AST type annotations through to KIR via a type
//! map; a parameter without one is `Ptr(F32, Global)` (a tensor of f32,
//! which is what the launcher passes). Expressions propagate types from the
//! parameter map through locals, binary ops and subscript dereferences.
//! Mixed-type binary operations promote to the wider type
//! (`promote_types()`) with an explicit `Cast`. An integer literal is `U32`,
//! the index type; a float literal is `F32`.

use std::collections::{BTreeSet, HashMap};

use nsl_ast::block::KernelDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::operator::{AssignOp, BinOp, UnaryOp};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::types::TypeExprKind;
use nsl_lexer::Interner;

use crate::error::CodegenError;
use crate::gpu_target::GpuTarget;
use crate::kernel_ir::*;

/// Shared "what IS supported" hint appended to kernel refusal messages.
const KERNEL_SUPPORTED_HINT: &str = "Supported kernel constructs: `let` bindings, \
assignment to a declared local (`x = v`, `x += v`), arithmetic (+ - * / %), \
comparisons (< <= > >= == !=), element loads (a[i]), element stores \
(out[i] = v, out[i] += v), if/elif/else, `for j in range(...)`, `while`, \
`break`, `continue`, a bare `return`, and the builtins thread_id(), \
thread_id_y(), block_id(), block_id_y(), block_dim(), global_id(), \
sync_threads().";

/// Human-readable name for a statement kind, used in kernel refusal messages.
pub(crate) fn stmt_kind_name(kind: &StmtKind) -> &'static str {
    match kind {
        StmtKind::VarDecl { .. } => "let/const declaration",
        StmtKind::FnDef(_) => "nested fn definition",
        StmtKind::ModelDef(_) => "model definition",
        StmtKind::AgentDef(_) => "agent definition",
        StmtKind::StructDef(_) => "struct definition",
        StmtKind::EnumDef(_) => "enum definition",
        StmtKind::TraitDef(_) => "trait definition",
        StmtKind::If { .. } => "if statement",
        StmtKind::For { .. } => "for loop",
        StmtKind::While { .. } => "while loop",
        StmtKind::WhileLet { .. } => "while-let loop",
        StmtKind::Match { .. } => "match statement",
        StmtKind::Break => "break statement",
        StmtKind::Continue => "continue statement",
        StmtKind::Return(_) => "return statement",
        StmtKind::Yield(_) => "yield statement",
        StmtKind::Assign { .. } => "assignment",
        StmtKind::Import(_) => "import statement",
        StmtKind::FromImport(_) => "from-import statement",
        StmtKind::TrainBlock(_) => "train block",
        StmtKind::DistillBlock(_) => "distill block",
        StmtKind::GradBlock(_) => "grad block",
        StmtKind::QuantBlock(_) => "quant block",
        StmtKind::KernelDef(_) => "nested kernel definition",
        StmtKind::TokenizerDef(_) => "tokenizer definition",
        StmtKind::DatasetDef(_) => "dataset definition",
        StmtKind::DatatypeDef(_) => "datatype definition",
        StmtKind::ServeBlock(_) => "serve block",
        StmtKind::Decorated { .. } => "decorated statement",
        StmtKind::Expr(_) => "expression statement",
    }
}

/// Human-readable name for an expression kind, used in kernel refusal messages.
pub(crate) fn expr_kind_name(kind: &ExprKind) -> &'static str {
    match kind {
        ExprKind::IntLiteral(_) => "integer literal",
        ExprKind::FloatLiteral(_) => "float literal",
        ExprKind::StringLiteral(_) => "string literal",
        ExprKind::FString(_) => "f-string",
        ExprKind::BoolLiteral(_) => "bool literal",
        ExprKind::NoneLiteral => "None literal",
        ExprKind::ListLiteral(_) => "list literal",
        ExprKind::TupleLiteral(_) => "tuple literal",
        ExprKind::DictLiteral(_) => "dict literal",
        ExprKind::Ident(_) => "identifier",
        ExprKind::SelfRef => "self reference",
        ExprKind::BinaryOp { .. } => "binary operation",
        ExprKind::UnaryOp { .. } => "unary operation",
        ExprKind::Pipe { .. } => "pipe expression",
        ExprKind::MemberAccess { .. } => "member access",
        ExprKind::Subscript { .. } => "subscript",
        ExprKind::Call { .. } => "call",
        ExprKind::Lambda { .. } => "lambda",
        ExprKind::BlockExpr(_) => "block expression",
        ExprKind::ListComp { .. } => "list comprehension",
        ExprKind::IfExpr { .. } => "if expression",
        ExprKind::MatchExpr { .. } => "match expression",
        ExprKind::Range { .. } => "range expression",
        ExprKind::Paren(_) => "parenthesized expression",
        ExprKind::Await(_) => "await expression",
        ExprKind::Error => "parse-error expression",
    }
}

/// Surface-syntax token for a binary operator, used in kernel refusal messages.
pub(crate) fn binop_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::FloorDiv => "//",
        BinOp::Mod => "%",
        BinOp::Pow => "**",
        BinOp::MatMul => "@",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Is => "is",
        BinOp::In => "in",
        BinOp::BitOr => "|",
        BinOp::BitAnd => "&",
    }
}

// ---------------------------------------------------------------------------
// Type promotion
// ---------------------------------------------------------------------------

/// Type promotion table for binary operations in KIR.
///
/// Rules:
/// 1. Same type -> same type
/// 2. Float + Float -> wider float
/// 3. Int + Float -> float (or wider float if int is 64-bit to avoid precision loss)
/// 4. Int + Int -> wider int
/// 5. Unsigned + Signed of same width -> signed (to preserve sign information)
pub fn promote_types(a: KirType, b: KirType) -> KirType {
    if a == b {
        return a;
    }

    use KirType::*;
    match (a, b) {
        // Float promotions (wider wins)
        (F16, F32) | (F32, F16) => F32,
        (F16, F64) | (F64, F16) => F64,
        (F32, F64) | (F64, F32) => F64,
        (Bf16, F32) | (F32, Bf16) => F32,
        (Bf16, F64) | (F64, Bf16) => F64,
        (F16, Bf16) | (Bf16, F16) => F32, // both are 16-bit, promote to f32

        // Int promotions (wider wins)
        (I32, I64) | (I64, I32) => I64,
        (U32, U64) | (U64, U32) => U64,
        (U32, I32) | (I32, U32) => I32,
        (U64, I64) | (I64, U64) => I64,
        (U32, I64) | (I64, U32) => I64,
        (U64, I32) | (I32, U64) => I64, // best-effort promotion; no single signed type spans both ranges (matches C99 behavior)

        // Int + Float promotions
        (I32, F32) | (F32, I32) => F32,
        (I32, F64) | (F64, I32) => F64,
        (I64, F32) | (F32, I64) => F64, // i64 + f32 -> f64 (avoid precision loss)
        (I64, F64) | (F64, I64) => F64,
        (U32, F32) | (F32, U32) => F32,
        (U32, F64) | (F64, U32) => F64,
        (U64, F32) | (F32, U64) => F64,
        (U64, F64) | (F64, U64) => F64,
        (I32, F16) | (F16, I32) => F32,
        (I64, F16) | (F16, I64) => F64,
        (U32, F16) | (F16, U32) => F32,
        (U64, F16) | (F16, U64) => F64,
        (I32, Bf16) | (Bf16, I32) => F32,
        (I64, Bf16) | (Bf16, I64) => F64,
        (U32, Bf16) | (Bf16, U32) => F32,
        (U64, Bf16) | (Bf16, U64) => F64,

        // Bool promotes to the other type
        (Bool, other) | (other, Bool) => other,

        // Ternary types do NOT participate in numeric promotion — they must be
        // explicitly cast to/from int8 via `bitnet::pack`/`unpack` before
        // arithmetic. Phase emitters that need int-ternary math should declare
        // their operand types as `TernaryUnpacked` (one trit per i8 slot) and
        // cast to i8/i32 before invoking promote_types via the KirOp path.
        //
        // Catching these here gives a clear error rather than the generic
        // "Cannot promote" message, surfacing the design constraint to whoever
        // is composing the kernel IR.
        (Tq2Packed, _) | (_, Tq2Packed) => panic!(
            "Tq2Packed cannot participate in type promotion; unpack to TernaryUnpacked or cast to i8 first"
        ),
        (TernaryUnpacked, TernaryUnpacked) => TernaryUnpacked,
        (TernaryUnpacked, other) | (other, TernaryUnpacked) => panic!(
            "TernaryUnpacked cannot promote with {:?}; cast TernaryUnpacked → I32 explicitly first",
            other
        ),

        // Pointer types and Vec types -- no arithmetic promotion
        (a, b) => panic!("Cannot promote types {:?} and {:?}", a, b),
    }
}

// ---------------------------------------------------------------------------
// AST dtype -> KIR type mapping
// ---------------------------------------------------------------------------

/// Convert an AST dtype string (resolved from Symbol) to a KIR scalar type.
fn dtype_str_to_kir(dtype: &str) -> KirType {
    match dtype {
        "f16" | "fp16" | "float16" => KirType::F16,
        "bf16" | "bfloat16" => KirType::Bf16,
        "f32" | "fp32" | "float32" | "float" => KirType::F32,
        "f64" | "fp64" | "float64" | "double" => KirType::F64,
        "i32" | "int32" | "int" => KirType::I32,
        "i64" | "int64" | "long" => KirType::I64,
        "u32" | "uint32" => KirType::U32,
        "u64" | "uint64" => KirType::U64,
        "bool" => KirType::Bool,
        "fp8" | "f8" => KirType::F16, // FP8 promoted to F16 for compute
        _ => KirType::F32,            // fallback for unknown types
    }
}

/// Build a type map from kernel parameter names to their KIR types.
///
/// Tensor/Param/Buffer parameters become `Ptr(element_type, Global)`.
/// Named scalar types (int, float, etc.) become their scalar KIR type.
/// Parameters without type annotations default to `Ptr(F32, Global)`.
pub fn build_param_type_map(kernel: &KernelDef, interner: &Interner) -> HashMap<String, KirType> {
    let mut map = HashMap::new();

    for param in &kernel.params {
        let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
        let kir_type = if let Some(type_ann) = &param.type_ann {
            match &type_ann.kind {
                TypeExprKind::Tensor { dtype, .. }
                | TypeExprKind::Param { dtype, .. }
                | TypeExprKind::Buffer { dtype, .. } => {
                    let dtype_str = interner.resolve(dtype.0).unwrap_or("f32");
                    let elem = dtype_str_to_kir(dtype_str);
                    KirType::Ptr(Box::new(elem), AddressSpace::Global)
                }
                TypeExprKind::Named(sym) => {
                    let name = interner.resolve(sym.0).unwrap_or("f32");
                    dtype_str_to_kir(name)
                }
                _ => KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            }
        } else {
            // No type annotation -- default to Ptr(F32, Global) for backward compat
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
        };
        map.insert(pname, kir_type);
    }

    map
}


// ---------------------------------------------------------------------------
// @autotune constant substitution
// ---------------------------------------------------------------------------

/// Clone `kernel` with every `Ident` whose name appears in `constants`
/// replaced by that `IntLiteral`. `@autotune` generates one variant per
/// parameter combination (e.g. BLOCK_SIZE=128, TILE_SIZE=32) this way, and
/// the module-level constant a kernel body names becomes a literal before
/// lowering, so an unsubstituted one is still the "unknown identifier"
/// refusal below.
pub fn substitute_constants(
    kernel: &KernelDef,
    interner: &Interner,
    constants: &HashMap<String, i64>,
) -> KernelDef {
    let mut kernel = kernel.clone();
    if !constants.is_empty() {
        substitute_block_constants(&mut kernel.body, interner, constants);
    }
    kernel
}

fn substitute_block_constants(block: &mut Block, interner: &Interner, constants: &HashMap<String, i64>) {
    for stmt in &mut block.stmts {
        substitute_stmt_constants(stmt, interner, constants);
    }
}

fn substitute_stmt_constants(stmt: &mut Stmt, interner: &Interner, constants: &HashMap<String, i64>) {
    match &mut stmt.kind {
        StmtKind::VarDecl { value: Some(expr), .. } => {
            substitute_expr_constants(expr, interner, constants);
        }
        StmtKind::Expr(expr) => substitute_expr_constants(expr, interner, constants),
        StmtKind::Assign { target, value, .. } => {
            substitute_expr_constants(target, interner, constants);
            substitute_expr_constants(value, interner, constants);
        }
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            substitute_expr_constants(condition, interner, constants);
            substitute_block_constants(then_block, interner, constants);
            for (elif_cond, elif_block) in elif_clauses {
                substitute_expr_constants(elif_cond, interner, constants);
                substitute_block_constants(elif_block, interner, constants);
            }
            if let Some(else_blk) = else_block {
                substitute_block_constants(else_blk, interner, constants);
            }
        }
        StmtKind::For { iterable, body, .. } => {
            substitute_expr_constants(iterable, interner, constants);
            substitute_block_constants(body, interner, constants);
        }
        StmtKind::While { condition, body } => {
            substitute_expr_constants(condition, interner, constants);
            substitute_block_constants(body, interner, constants);
        }
        StmtKind::Return(Some(expr)) => substitute_expr_constants(expr, interner, constants),
        _ => {}
    }
}

fn substitute_expr_constants(expr: &mut Expr, interner: &Interner, constants: &HashMap<String, i64>) {
    match &mut expr.kind {
        ExprKind::Ident(sym) => {
            if let Some(name) = interner.resolve(sym.0)
                && let Some(&value) = constants.get(name)
            {
                expr.kind = ExprKind::IntLiteral(value);
            }
        }
        ExprKind::BinaryOp { left, right, .. } => {
            substitute_expr_constants(left, interner, constants);
            substitute_expr_constants(right, interner, constants);
        }
        ExprKind::UnaryOp { operand, .. } => substitute_expr_constants(operand, interner, constants),
        ExprKind::Paren(inner) => substitute_expr_constants(inner, interner, constants),
        ExprKind::Call { callee, args } => {
            substitute_expr_constants(callee, interner, constants);
            for arg in args {
                substitute_expr_constants(&mut arg.value, interner, constants);
            }
        }
        ExprKind::Subscript { object, index } => {
            substitute_expr_constants(object, interner, constants);
            if let SubscriptKind::Index(idx_expr) = index.as_mut() {
                substitute_expr_constants(idx_expr, interner, constants);
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                substitute_expr_constants(s, interner, constants);
            }
            if let Some(e) = end {
                substitute_expr_constants(e, interner, constants);
            }
        }
        _ => {} // Literals and other leaf nodes: no substitution needed
    }
}

// ---------------------------------------------------------------------------
// Kernel lowering
// ---------------------------------------------------------------------------

/// Lower a KernelDef AST node into a verified KernelIR.
///
/// Parameters:
/// - `kernel`: the parsed kernel AST
/// - `interner`: string interner for resolving symbol names
/// - `_target`: the GPU target (the lowering is target-neutral; the caller
///   checks `required_features` against the target and picks the printer)
///
/// Returns a `KernelIR` ready for backend lowering (e.g.
/// `backend_ptx::lower_kir_to_ptx`), or a `CodegenError` refusal when the
/// kernel body uses a construct outside the supported set.
pub fn lower_kernel_to_ir(
    kernel: &KernelDef,
    interner: &Interner,
    _target: GpuTarget,
) -> Result<KernelIR, CodegenError> {
    let name = interner
        .resolve(kernel.name.0)
        .unwrap_or("__kernel")
        .to_string();

    // Build type map from AST type annotations
    let param_type_map = build_param_type_map(kernel, interner);

    let mut lowerer = KernelLowerer::new(&name);

    // Map kernel parameters to KirParams using their AST types.
    for param in &kernel.params {
        let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
        let kir_type = param_type_map
            .get(&pname)
            .cloned()
            .unwrap_or(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));

        let address_space = match &kir_type {
            KirType::Ptr(_, space) => *space,
            _ => AddressSpace::Local,
        };

        let var_id = lowerer
            .builder
            .add_param(&pname, kir_type.clone(), address_space);
        lowerer.var_map.insert(pname.clone(), var_id);
        lowerer.type_map.insert(pname, kir_type);
    }

    // Create entry block
    let entry = lowerer.builder.new_block();
    lowerer.set_block(entry);

    // Lower body statements
    lower_block(&mut lowerer, &kernel.body, interner)?;

    // Terminate the current block if the body did not (a trailing `return`
    // or `break` already did).
    if !lowerer.terminated {
        lowerer.terminate(KirTerminator::Return);
    }

    lowerer.builder.set_workgroup_size([256, 1, 1]);
    let ir = lowerer.builder.finalize();

    // Roadmap A2 step 2: no kernel leaves the front door unverified. A
    // lowering bug (an operand that never got a definition, a load whose
    // register does not match its pointee, a use ahead of its def) is a
    // compile-time refusal here, not a wrong answer at runtime.
    ir.verify().map_err(|errors| {
        CodegenError::new(format!(
            "kernel `{}`: lowering produced KIR that fails verification ({} violation{}):\n{}",
            name,
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            crate::kir_verify::render_errors(&errors),
        ))
    })?;
    Ok(ir)
}

/// [`lower_kernel_to_ir`] after [`substitute_constants`].
pub fn lower_kernel_to_ir_with_constants(
    kernel: &KernelDef,
    interner: &Interner,
    target: GpuTarget,
    constants: &HashMap<String, i64>,
) -> Result<KernelIR, CodegenError> {
    if constants.is_empty() {
        return lower_kernel_to_ir(kernel, interner, target);
    }
    let kernel = substitute_constants(kernel, interner, constants);
    lower_kernel_to_ir(&kernel, interner, target)
}

/// The CUDA path in one call: AST → KIR → null-terminated PTX. What
/// `KernelCompiler::compile` was before roadmap A2 step 3.
pub fn compile_kernel_ptx(kernel: &KernelDef, interner: &Interner) -> Result<Vec<u8>, CodegenError> {
    let ir = lower_kernel_to_ir(kernel, interner, GpuTarget::Cuda)?;
    Ok(crate::backend_ptx::lower_kir_to_ptx(&ir))
}

/// [`compile_kernel_ptx`] with `@autotune` constants substituted first.
pub fn compile_kernel_ptx_with_constants(
    kernel: &KernelDef,
    interner: &Interner,
    constants: &HashMap<String, i64>,
) -> Result<Vec<u8>, CodegenError> {
    let ir = lower_kernel_to_ir_with_constants(kernel, interner, GpuTarget::Cuda, constants)?;
    Ok(crate::backend_ptx::lower_kir_to_ptx(&ir))
}

/// An enclosing loop, for `break` / `continue`.
struct LoopCtx {
    header: BlockId,
    exit: BlockId,
    /// The locals the loop carries as header (and exit) parameters, in the
    /// order the parameters were added.
    carried: Vec<String>,
    /// `for` loops: the induction variable and its step, so `continue`
    /// advances it like the latch does.
    induction: Option<(String, i64)>,
}

/// Internal lowering state.
struct KernelLowerer {
    builder: KirBuilder,
    /// Map from variable name -> VarId holding its CURRENT value (SSA: an
    /// assignment rebinds the name; a join or loop header rebinds it to a
    /// block parameter).
    var_map: HashMap<String, VarId>,
    /// Map from variable name -> KirType (for type propagation)
    type_map: HashMap<String, KirType>,
    /// Kernel name, used in refusal diagnostics.
    kernel_name: String,
    /// The loops the current statement is nested in, innermost last.
    loops: Vec<LoopCtx>,
    /// Whether the current block already has a terminator (a `break`,
    /// `continue` or `return` was lowered); the next statement in the same
    /// block is then unreachable and refused.
    terminated: bool,
}

impl KernelLowerer {
    fn new(name: &str) -> Self {
        KernelLowerer {
            builder: KirBuilder::new(name),
            var_map: HashMap::new(),
            type_map: HashMap::new(),
            kernel_name: name.to_string(),
            loops: Vec::new(),
            terminated: false,
        }
    }

    fn set_block(&mut self, block: BlockId) {
        self.builder.set_block(block);
        self.terminated = false;
    }

    fn terminate(&mut self, term: KirTerminator) {
        self.builder.terminate(term);
        self.terminated = true;
    }

    /// Look up the KIR type of a variable by name.
    /// Returns F32 as fallback for unknown variables.
    fn var_type(&self, name: &str) -> KirType {
        self.type_map.get(name).cloned().unwrap_or(KirType::F32)
    }

    /// Build a refusal error for an unsupported construct in this kernel.
    fn refuse(&self, construct: &str, hint: &str) -> CodegenError {
        CodegenError::new(format!(
            "kernel '{}': {} is not supported in `kernel` blocks. {}",
            self.kernel_name, construct, hint
        ))
    }

    /// An edge into `target` passing the current values of `names`.
    fn edge(&self, target: BlockId, names: &[String]) -> KirEdge {
        KirEdge::with(target, names.iter().map(|n| self.var_map[n]).collect())
    }

    /// Add one block parameter per name to `block` (typed as the name is
    /// declared) and return them in order.
    fn add_params_for(&mut self, block: BlockId, names: &[String]) -> Vec<VarId> {
        names
            .iter()
            .map(|n| {
                let ty = self.var_type(n);
                self.builder.add_block_param(block, ty)
            })
            .collect()
    }

    /// Rebind `names` to `params` (the values at a join or a loop header).
    fn bind(&mut self, names: &[String], params: &[VarId]) {
        for (n, p) in names.iter().zip(params) {
            self.var_map.insert(n.clone(), *p);
        }
    }

    /// The locals assigned anywhere inside `blocks` that are in scope now —
    /// the values a join or a loop header must carry.
    fn carried_locals(&self, blocks: &[&Block], interner: &Interner) -> Vec<String> {
        let mut assigned = BTreeSet::new();
        for b in blocks {
            assigned_names(b, interner, &mut assigned);
        }
        assigned
            .into_iter()
            .filter(|n| self.var_map.contains_key(n))
            .collect()
    }

    fn emit_const(&mut self, ty: &KirType, value: i64) -> Result<VarId, CodegenError> {
        let konst = match ty {
            KirType::U32 => ConstValue::U32(value as u32),
            KirType::I32 => ConstValue::I32(value as i32),
            KirType::U64 => ConstValue::U64(value as u64),
            KirType::I64 => ConstValue::I64(value),
            KirType::F32 => ConstValue::F32(value as f32),
            KirType::F64 => ConstValue::F64(value as f64),
            other => {
                return Err(self.refuse(
                    &format!("a constant of type {other:?}"),
                    "loop bounds and steps must be integers",
                ))
            }
        };
        let dst = self.builder.new_typed_var(ty.clone());
        self.builder.emit(KirOp::Const(dst, KirConst { ty: ty.clone(), value: konst }));
        Ok(dst)
    }

    /// `src` (of type `from`) as a value of type `to`, inserting a `Cast`
    /// when the types differ. Pointers and predicates do not convert.
    fn cast_to(&mut self, src: VarId, from: &KirType, to: &KirType) -> Result<VarId, CodegenError> {
        if from == to {
            return Ok(src);
        }
        if !is_numeric(from) || !is_numeric(to) {
            return Err(self.refuse(
                &format!("a conversion from {from:?} to {to:?}"),
                "only numeric values convert; a comparison result cannot be stored or \
                 assigned, and a tensor parameter cannot be used as a number",
            ));
        }
        let dst = self.builder.new_typed_var(to.clone());
        self.builder.emit(KirOp::Cast(dst, src, to.clone()));
        Ok(dst)
    }
}

fn is_numeric(ty: &KirType) -> bool {
    matches!(
        ty,
        KirType::U32 | KirType::I32 | KirType::U64 | KirType::I64 | KirType::F16 | KirType::Bf16 | KirType::F32 | KirType::F64
    )
}

fn is_integer(ty: &KirType) -> bool {
    matches!(ty, KirType::U32 | KirType::I32 | KirType::U64 | KirType::I64)
}

/// Every plain-identifier assignment target in `block`, recursing into
/// nested `if` arms and loop bodies.
fn assigned_names(block: &Block, interner: &Interner, out: &mut BTreeSet<String>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Assign { target, .. } => {
                if let ExprKind::Ident(sym) = &target.kind {
                    out.insert(interner.resolve(sym.0).unwrap_or("_").to_string());
                }
            }
            StmtKind::If { then_block, elif_clauses, else_block, .. } => {
                assigned_names(then_block, interner, out);
                for (_, b) in elif_clauses {
                    assigned_names(b, interner, out);
                }
                if let Some(b) = else_block {
                    assigned_names(b, interner, out);
                }
            }
            StmtKind::For { body, .. } | StmtKind::While { body, .. } => {
                assigned_names(body, interner, out);
            }
            _ => {}
        }
    }
}

/// Lower a block of statements.
fn lower_block(
    lowerer: &mut KernelLowerer,
    block: &Block,
    interner: &Interner,
) -> Result<(), CodegenError> {
    for stmt in &block.stmts {
        if lowerer.terminated {
            return Err(lowerer
                .refuse(
                    "a statement after `break`, `continue` or `return`",
                    "it can never run; remove it or move it before the jump",
                )
                .with_span_if_unset(stmt.span));
        }
        lower_stmt(lowerer, stmt, interner)?;
    }
    Ok(())
}

/// Lower one kernel statement; an error raised beneath it leaves here
/// pointing at the innermost statement or expression being lowered
/// (`CodegenError::with_span_if_unset`, as in the Cranelift dispatchers).
fn lower_stmt(lowerer: &mut KernelLowerer, stmt: &Stmt, interner: &Interner) -> Result<(), CodegenError> {
    lower_stmt_dispatch(lowerer, stmt, interner).map_err(|e| e.with_span_if_unset(stmt.span))
}

fn lower_stmt_dispatch(
    lowerer: &mut KernelLowerer,
    stmt: &Stmt,
    interner: &Interner,
) -> Result<(), CodegenError> {
    match &stmt.kind {
        StmtKind::VarDecl { pattern, value, type_ann, .. } => {
            let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind else {
                return Err(lowerer.refuse(
                    "destructuring `let` pattern",
                    "kernel locals must bind a single name, e.g. `let i = thread_id()`",
                ));
            };
            let name = interner.resolve(sym.0).unwrap_or("_v").to_string();
            let Some(expr) = value else {
                return Err(lowerer.refuse(
                    "`let` declaration without an initializer",
                    "kernel locals must be initialized at declaration, e.g. `let i = thread_id()`",
                ));
            };
            let (mut var_id, mut ty) = lower_expr(lowerer, expr, interner)?;
            // `let x: f32 = 0` declares the local's type; the initializer
            // converts to it.
            if let Some(ann) = type_ann
                && let TypeExprKind::Named(tsym) = &ann.kind
            {
                let declared = dtype_str_to_kir(interner.resolve(tsym.0).unwrap_or("f32"));
                var_id = lowerer.cast_to(var_id, &ty, &declared)?;
                ty = declared;
            }
            lowerer.var_map.insert(name.clone(), var_id);
            lowerer.type_map.insert(name, ty);
            Ok(())
        }
        StmtKind::Expr(expr) => {
            // Expression statement (e.g., function call like sync_threads())
            lower_expr(lowerer, expr, interner)?;
            Ok(())
        }
        StmtKind::Assign { target, op, value } => lower_assign(lowerer, target, *op, value, interner),
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            lower_if(lowerer, condition, then_block, elif_clauses, else_block.as_ref(), interner)
        }
        StmtKind::For { pattern, iterable, body } => lower_for(lowerer, pattern, iterable, body, interner),
        StmtKind::While { condition, body } => lower_while(lowerer, condition, body, interner),
        StmtKind::Break => {
            let Some(lp) = lowerer.loops.last() else {
                return Err(lowerer.refuse("`break` outside a loop", KERNEL_SUPPORTED_HINT));
            };
            let edge = lowerer.edge(lp.exit, &lp.carried);
            lowerer.terminate(KirTerminator::Branch(edge));
            Ok(())
        }
        StmtKind::Continue => {
            let Some(lp) = lowerer.loops.last() else {
                return Err(lowerer.refuse("`continue` outside a loop", KERNEL_SUPPORTED_HINT));
            };
            let (header, carried, induction) = (lp.header, lp.carried.clone(), lp.induction.clone());
            let mut args = Vec::new();
            if let Some((var, step)) = induction {
                args.push(advance_induction(lowerer, &var, step)?);
            }
            args.extend(carried.iter().map(|n| lowerer.var_map[n]));
            lowerer.terminate(KirTerminator::Branch(KirEdge::with(header, args)));
            Ok(())
        }
        StmtKind::Return(None) => {
            lowerer.terminate(KirTerminator::Return);
            Ok(())
        }
        StmtKind::Return(Some(_)) => Err(lowerer.refuse(
            "`return` with a value",
            "a kernel returns nothing; write results through a parameter, e.g. `out[i] = v`",
        )),
        other => Err(lowerer.refuse(stmt_kind_name(other), KERNEL_SUPPORTED_HINT)),
    }
}

/// `j + step` for a `for` loop's induction variable, as the latch and
/// `continue` pass it back to the header.
fn advance_induction(lowerer: &mut KernelLowerer, var: &str, step: i64) -> Result<VarId, CodegenError> {
    let ty = lowerer.var_type(var);
    let cur = lowerer.var_map[var];
    let step_v = lowerer.emit_const(&ty, step)?;
    let next = lowerer.builder.new_typed_var(ty);
    lowerer.builder.emit(KirOp::Add(next, cur, step_v));
    Ok(next)
}

/// `target = value` / `target op= value`: a `let`-declared local (SSA
/// rebinding; a join or loop header carries it as a block parameter) or an
/// element store through a pointer parameter.
fn lower_assign(
    lowerer: &mut KernelLowerer,
    target: &Expr,
    op: AssignOp,
    value: &Expr,
    interner: &Interner,
) -> Result<(), CodegenError> {
    let bin = match op {
        AssignOp::Assign => None,
        AssignOp::AddAssign => Some(BinOp::Add),
        AssignOp::SubAssign => Some(BinOp::Sub),
        AssignOp::MulAssign => Some(BinOp::Mul),
        AssignOp::DivAssign => Some(BinOp::Div),
    };
    match &target.kind {
        ExprKind::Ident(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("_").to_string();
            let Some(&cur) = lowerer.var_map.get(&name) else {
                return Err(CodegenError::new(format!(
                    "kernel '{}': assignment to undeclared variable '{}' - declare it first \
                     with `let {} = ...`",
                    lowerer.kernel_name, name, name
                )));
            };
            let declared = lowerer.var_type(&name);
            if matches!(declared, KirType::Ptr(_, _)) {
                return Err(lowerer.refuse(
                    "assignment to a tensor parameter",
                    "kernel stores write through a subscript, e.g. `out[i] = v`",
                ));
            }
            if lowerer
                .loops
                .iter()
                .any(|lp| lp.induction.as_ref().is_some_and(|(v, _)| *v == name))
            {
                return Err(lowerer.refuse(
                    &format!("assignment to the loop variable `{name}` inside its `for` loop"),
                    "the loop advances it; copy it into another local to change it",
                ));
            }
            let (v, vty) = lower_expr(lowerer, value, interner)?;
            let (v, vty) = match bin {
                Some(b) => lower_binop(lowerer, b, cur, declared.clone(), v, vty)?,
                None => (v, vty),
            };
            let v = lowerer.cast_to(v, &vty, &declared)?;
            lowerer.var_map.insert(name, v);
            Ok(())
        }
        ExprKind::Subscript { object, index } => {
            let ExprKind::Ident(osym) = &object.kind else {
                return Err(lowerer.refuse(
                    "a store through a non-identifier subscript base",
                    "only direct stores into a kernel parameter or local, \
                     e.g. `out[i] = v`, are supported",
                ));
            };
            let obj_name = interner.resolve(osym.0).unwrap_or("_").to_string();
            let Some(&base) = lowerer.var_map.get(&obj_name) else {
                return Err(CodegenError::new(format!(
                    "kernel '{}': store into unknown variable '{}' - kernel code can \
                     only store into kernel parameters (or locals derived from them)",
                    lowerer.kernel_name, obj_name
                )));
            };
            let KirType::Ptr(elem, space) = lowerer.var_type(&obj_name) else {
                return Err(lowerer.refuse(
                    &format!("a store into `{obj_name}`, which is not a tensor"),
                    "only tensor parameters can be indexed",
                ));
            };
            let elem_ty = *elem;
            let SubscriptKind::Index(idx_expr) = index.as_ref() else {
                return Err(lowerer.refuse(
                    "slice / multi-dimensional store target",
                    "only single-element stores, e.g. `out[i] = v`, are supported",
                ));
            };
            let addr = lower_element_address(lowerer, base, &elem_ty, space, idx_expr, interner)?;
            let (v, vty) = lower_expr(lowerer, value, interner)?;
            let (v, vty) = match bin {
                Some(b) => {
                    let cur = lowerer.builder.new_typed_var(elem_ty.clone());
                    lowerer.builder.emit(KirOp::Load(cur, addr, space));
                    lower_binop(lowerer, b, cur, elem_ty.clone(), v, vty)?
                }
                None => (v, vty),
            };
            let v = lowerer.cast_to(v, &vty, &elem_ty)?;
            lowerer.builder.emit(KirOp::Store(addr, v, space));
            Ok(())
        }
        _ => Err(lowerer.refuse(
            "assignment to something other than a local or an element",
            "kernel stores write through a subscript, e.g. `out[i] = v`; a local is \
             declared with `let` and reassigned by name",
        )),
    }
}

/// `&base[idx]`: the index lowered to the `U32` offset `PtrOffset` takes.
fn lower_element_address(
    lowerer: &mut KernelLowerer,
    base: VarId,
    elem_ty: &KirType,
    space: AddressSpace,
    idx_expr: &Expr,
    interner: &Interner,
) -> Result<VarId, CodegenError> {
    let (offset, oty) = lower_expr(lowerer, idx_expr, interner)?;
    if !is_integer(&oty) {
        return Err(lowerer
            .refuse(
                &format!("an index of type {oty:?}"),
                "element indices are integers, e.g. `thread_id()` or `i + 1`",
            )
            .with_span_if_unset(idx_expr.span));
    }
    let offset = lowerer.cast_to(offset, &oty, &KirType::U32)?;
    let addr = lowerer
        .builder
        .new_typed_var(KirType::Ptr(Box::new(elem_ty.clone()), space));
    lowerer.builder.emit(KirOp::PtrOffset(addr, base, offset));
    Ok(addr)
}

/// `if`/`elif`/`else` as a chain of `CondBranch`es into one join block. A
/// local assigned in any arm reaches the join as a block parameter; every
/// arm's edge (and the fall-through edge of an `if` without `else`) passes
/// its value of it.
fn lower_if(
    lowerer: &mut KernelLowerer,
    cond: &Expr,
    then_block: &Block,
    elif_clauses: &[(Expr, Block)],
    else_block: Option<&Block>,
    interner: &Interner,
) -> Result<(), CodegenError> {
    let mut arm_blocks: Vec<&Block> = vec![then_block];
    arm_blocks.extend(elif_clauses.iter().map(|(_, b)| b));
    arm_blocks.extend(else_block);
    let carried = lowerer.carried_locals(&arm_blocks, interner);

    let join = lowerer.builder.new_block();
    let join_params = lowerer.add_params_for(join, &carried);
    let before = lowerer.var_map.clone();

    let mut conds: Vec<(&Expr, &Block)> = vec![(cond, then_block)];
    conds.extend(elif_clauses.iter().map(|(c, b)| (c, b)));

    for (c, body) in conds {
        let (cv, cty) = lower_expr(lowerer, c, interner)?;
        if cty != KirType::Bool {
            return Err(lowerer
                .refuse(
                    "an `if` condition that is not a comparison",
                    "kernel `if` / `elif` conditions must be comparisons (< <= > >= == !=), \
                     e.g. `if i < n:`",
                )
                .with_span_if_unset(c.span));
        }
        let arm = lowerer.builder.new_block();
        let next = lowerer.builder.new_block();
        lowerer.terminate(KirTerminator::CondBranch(cv, arm.into(), next.into()));

        lowerer.set_block(arm);
        lowerer.var_map = before.clone();
        lower_block(lowerer, body, interner)?;
        if !lowerer.terminated {
            let e = lowerer.edge(join, &carried);
            lowerer.terminate(KirTerminator::Branch(e));
        }

        lowerer.set_block(next);
        lowerer.var_map = before.clone();
    }

    if let Some(body) = else_block {
        lower_block(lowerer, body, interner)?;
    }
    if !lowerer.terminated {
        let e = lowerer.edge(join, &carried);
        lowerer.terminate(KirTerminator::Branch(e));
    }

    lowerer.set_block(join);
    lowerer.var_map = before;
    lowerer.bind(&carried, &join_params);
    Ok(())
}

/// `while cond:` — the header re-evaluates `cond` each trip; the locals the
/// body assigns are header parameters (the loop-carried values) and exit
/// parameters (so `break` can leave with its own values).
fn lower_while(
    lowerer: &mut KernelLowerer,
    cond: &Expr,
    body: &Block,
    interner: &Interner,
) -> Result<(), CodegenError> {
    let carried = lowerer.carried_locals(&[body], interner);
    let header = lowerer.builder.new_block();
    let header_params = lowerer.add_params_for(header, &carried);
    let entry_edge = lowerer.edge(header, &carried);
    lowerer.terminate(KirTerminator::Branch(entry_edge));

    lowerer.set_block(header);
    lowerer.bind(&carried, &header_params);
    let at_header = lowerer.var_map.clone();
    let (cv, cty) = lower_expr(lowerer, cond, interner)?;
    if cty != KirType::Bool {
        return Err(lowerer
            .refuse(
                "a `while` condition that is not a comparison",
                "kernel `while` conditions must be comparisons (< <= > >= == !=)",
            )
            .with_span_if_unset(cond.span));
    }
    let body_block = lowerer.builder.new_block();
    let exit = lowerer.builder.new_block();
    let exit_params = lowerer.add_params_for(exit, &carried);
    let exit_edge = lowerer.edge(exit, &carried);
    lowerer.terminate(KirTerminator::CondBranch(cv, body_block.into(), exit_edge));

    lowerer.loops.push(LoopCtx { header, exit, carried: carried.clone(), induction: None });
    lowerer.set_block(body_block);
    lowerer.var_map = at_header.clone();
    let lowered = lower_block(lowerer, body, interner);
    lowerer.loops.pop();
    lowered?;
    if !lowerer.terminated {
        let back = lowerer.edge(header, &carried);
        lowerer.terminate(KirTerminator::Branch(back));
    }

    lowerer.set_block(exit);
    lowerer.var_map = at_header;
    lowerer.bind(&carried, &exit_params);
    Ok(())
}

/// `for j in range(...)`: the induction variable is the header's first
/// parameter, advanced by the latch (and by `continue`); the locals the body
/// assigns follow it, as in `while`. `j` is scoped to the loop.
fn lower_for(
    lowerer: &mut KernelLowerer,
    pattern: &nsl_ast::pattern::Pattern,
    iterable: &Expr,
    body: &Block,
    interner: &Interner,
) -> Result<(), CodegenError> {
    let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind else {
        return Err(lowerer.refuse(
            "a `for` loop over a destructuring pattern",
            "kernel loops bind one name, e.g. `for j in range(0, n):`",
        ));
    };
    let var = interner.resolve(sym.0).unwrap_or("_j").to_string();
    let (start, end, step, inclusive) = range_bounds(lowerer, iterable, interner)?;

    // Bounds are evaluated once, before the loop, in the index type.
    let (end_v, end_ty) = lower_expr(lowerer, end, interner)?;
    let (start_v, start_ty) = match start {
        Some(s) => lower_expr(lowerer, s, interner)?,
        None => (lowerer.emit_const(&KirType::U32, 0)?, KirType::U32),
    };
    if !is_integer(&end_ty) || !is_integer(&start_ty) {
        return Err(lowerer
            .refuse(
                "a `range(...)` with non-integer bounds",
                "loop bounds are integers, e.g. `range(0, n)`",
            )
            .with_span_if_unset(iterable.span));
    }
    let idx_ty = promote_types(start_ty.clone(), end_ty.clone());
    let start_v = lowerer.cast_to(start_v, &start_ty, &idx_ty)?;
    let end_v = lowerer.cast_to(end_v, &end_ty, &idx_ty)?;

    let carried = lowerer.carried_locals(&[body], interner);
    if carried.contains(&var) {
        return Err(lowerer.refuse(
            &format!("assignment to the loop variable `{var}` inside its `for` loop"),
            "the loop advances it; copy it into another local to change it",
        ));
    }

    let header = lowerer.builder.new_block();
    let j_param = lowerer.builder.add_block_param(header, idx_ty.clone());
    let header_params = lowerer.add_params_for(header, &carried);
    let mut entry_args = vec![start_v];
    entry_args.extend(carried.iter().map(|n| lowerer.var_map[n]));
    lowerer.terminate(KirTerminator::Branch(KirEdge::with(header, entry_args)));

    // `j` shadows an outer local of the same name for the loop's extent.
    let outer_j = (lowerer.var_map.get(&var).copied(), lowerer.type_map.get(&var).cloned());
    lowerer.set_block(header);
    lowerer.bind(&carried, &header_params);
    lowerer.var_map.insert(var.clone(), j_param);
    lowerer.type_map.insert(var.clone(), idx_ty.clone());
    let at_header = lowerer.var_map.clone();
    let cmp = match (step > 0, inclusive) {
        (true, false) => CmpOp::Lt,
        (true, true) => CmpOp::Le,
        (false, false) => CmpOp::Gt,
        (false, true) => CmpOp::Ge,
    };
    let more = lowerer.builder.new_typed_var(KirType::Bool);
    lowerer.builder.emit(KirOp::Cmp(more, j_param, end_v, cmp));
    let body_block = lowerer.builder.new_block();
    let exit = lowerer.builder.new_block();
    let exit_params = lowerer.add_params_for(exit, &carried);
    let exit_edge = lowerer.edge(exit, &carried);
    lowerer.terminate(KirTerminator::CondBranch(more, body_block.into(), exit_edge));

    lowerer.loops.push(LoopCtx {
        header,
        exit,
        carried: carried.clone(),
        induction: Some((var.clone(), step)),
    });
    lowerer.set_block(body_block);
    lowerer.var_map = at_header.clone();
    let lowered = lower_block(lowerer, body, interner);
    lowerer.loops.pop();
    lowered?;
    if !lowerer.terminated {
        let mut args = vec![advance_induction(lowerer, &var, step)?];
        args.extend(carried.iter().map(|n| lowerer.var_map[n]));
        lowerer.terminate(KirTerminator::Branch(KirEdge::with(header, args)));
    }

    lowerer.set_block(exit);
    lowerer.var_map = at_header;
    lowerer.bind(&carried, &exit_params);
    match outer_j {
        (Some(v), Some(t)) => {
            lowerer.var_map.insert(var.clone(), v);
            lowerer.type_map.insert(var, t);
        }
        _ => {
            lowerer.var_map.remove(&var);
            lowerer.type_map.remove(&var);
        }
    }
    Ok(())
}

/// The bounds of a `for` iterable: `range(end)`, `range(start, end)`,
/// `range(start, end, step)` with a literal non-zero step, or `a..b` /
/// `a..=b`. Returns (start, end, step, inclusive).
fn range_bounds<'a>(
    lowerer: &KernelLowerer,
    iterable: &'a Expr,
    interner: &Interner,
) -> Result<(Option<&'a Expr>, &'a Expr, i64, bool), CodegenError> {
    match &iterable.kind {
        ExprKind::Call { callee, args }
            if matches!(&callee.kind, ExprKind::Ident(s) if interner.resolve(s.0) == Some("range"))
                && args.iter().all(|a| a.name.is_none()) =>
        {
            match args.len() {
                1 => Ok((None, &args[0].value, 1, false)),
                2 => Ok((Some(&args[0].value), &args[1].value, 1, false)),
                3 => {
                    let step = literal_int(&args[2].value);
                    match step {
                        Some(0) | None => Err(lowerer
                            .refuse(
                                "a `range(...)` step that is not a non-zero integer literal",
                                "write the step as a literal, e.g. `range(0, n, 2)` or `range(n, 0, -1)`",
                            )
                            .with_span_if_unset(args[2].value.span)),
                        Some(s) => Ok((Some(&args[0].value), &args[1].value, s, false)),
                    }
                }
                _ => Err(lowerer.refuse(
                    "a `range(...)` with that many arguments",
                    "`range(end)`, `range(start, end)` or `range(start, end, step)`",
                )),
            }
        }
        ExprKind::Range { start, end: Some(end), inclusive } => {
            Ok((start.as_deref(), end, 1, *inclusive))
        }
        _ => Err(lowerer.refuse(
            "a `for` loop over something other than `range(...)`",
            "kernel loops iterate an integer range, e.g. `for j in range(0, n):`",
        )),
    }
}

/// `3`, `-3`, `(3)` as an integer, if `expr` is that literal.
fn literal_int(expr: &Expr) -> Option<i64> {
    match &expr.kind {
        ExprKind::IntLiteral(v) => Some(*v),
        ExprKind::Paren(inner) => literal_int(inner),
        ExprKind::UnaryOp { op: UnaryOp::Neg, operand } => literal_int(operand).map(|v| -v),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Typed expression lowering
// ---------------------------------------------------------------------------

/// Lower an expression; returns the VarId holding the result and its
/// inferred KIR type. Errors leave here carrying the innermost expression's
/// span (see `lower_stmt`).
///
/// Type propagation rules:
/// - Identifiers: look up type from `type_map`
/// - Int literals: U32
/// - Float literals: F32
/// - Subscript (a[idx]): dereferences Ptr(T, _) to T
/// - BinaryOp: promotes operand types via `promote_types()`, inserts Cast ops if needed
/// - Builtins (thread_id, etc.): U32
/// - Unary neg: preserves operand type (an unsigned operand is made signed first)
fn lower_expr(
    lowerer: &mut KernelLowerer,
    expr: &Expr,
    interner: &Interner,
) -> Result<(VarId, KirType), CodegenError> {
    lower_expr_dispatch(lowerer, expr, interner).map_err(|e| e.with_span_if_unset(expr.span))
}

fn lower_expr_dispatch(
    lowerer: &mut KernelLowerer,
    expr: &Expr,
    interner: &Interner,
) -> Result<(VarId, KirType), CodegenError> {
    match &expr.kind {
        ExprKind::IntLiteral(val) => {
            if *val < 0 || *val > u32::MAX as i64 {
                return Err(lowerer.refuse(
                    &format!("the integer literal {val}"),
                    "kernel integer literals are 32-bit indices (0 ..= 4294967295)",
                ));
            }
            lowerer.emit_const(&KirType::U32, *val)
                .map(|v| (v, KirType::U32))
        }
        ExprKind::FloatLiteral(val) => {
            let dst = lowerer.builder.new_typed_var(KirType::F32);
            lowerer.builder.emit(KirOp::Const(
                dst,
                KirConst {
                    ty: KirType::F32,
                    value: ConstValue::F32(*val as f32),
                },
            ));
            Ok((dst, KirType::F32))
        }
        ExprKind::Ident(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("_");
            if let Some(&var_id) = lowerer.var_map.get(name) {
                let ty = lowerer.var_type(name);
                Ok((var_id, ty))
            } else {
                // Unknown variable — used to fabricate an UNINITIALIZED
                // placeholder register. Refuse instead.
                Err(CodegenError::new(format!(
                    "kernel '{}': unknown identifier '{}' in kernel body. Kernel code \
                     can only reference kernel parameters and locals declared with \
                     `let` inside the kernel; module-level constants are not visible \
                     here - pass the value as a kernel parameter or substitute it via \
                     an @autotune constant. {}",
                    lowerer.kernel_name, name, KERNEL_SUPPORTED_HINT
                )))
            }
        }
        ExprKind::BinaryOp { left, op, right } => {
            let (a, a_ty) = lower_expr(lowerer, left, interner)?;
            let (b, b_ty) = lower_expr(lowerer, right, interner)?;
            lower_binop(lowerer, *op, a, a_ty, b, b_ty)
        }
        ExprKind::Call { callee, args } => lower_builtin_call(lowerer, callee, args, interner),
        ExprKind::Subscript { object, index } => {
            // a[idx] -> PtrOffset + Load; the loaded value's type is the
            // pointer's pointee.
            let ExprKind::Ident(osym) = &object.kind else {
                return Err(lowerer.refuse(
                    "subscript of a non-identifier expression",
                    "only direct indexing of a kernel parameter or local, \
                     e.g. `a[i]`, is supported",
                ));
            };
            let obj_name = interner.resolve(osym.0).unwrap_or("_").to_string();
            let Some(&base) = lowerer.var_map.get(&obj_name) else {
                return Err(CodegenError::new(format!(
                    "kernel '{}': subscript of unknown variable '{}' - kernel code \
                     can only index kernel parameters and locals declared with `let`",
                    lowerer.kernel_name, obj_name
                )));
            };
            let KirType::Ptr(elem, space) = lowerer.var_type(&obj_name) else {
                return Err(lowerer.refuse(
                    &format!("indexing `{obj_name}`, which is not a tensor"),
                    "only tensor parameters can be indexed",
                ));
            };
            let elem_ty = *elem;
            let SubscriptKind::Index(idx_expr) = index.as_ref() else {
                return Err(lowerer.refuse(
                    "slice / multi-dimensional subscript",
                    "only single-element indexing, e.g. `a[i]`, is supported",
                ));
            };
            let addr = lower_element_address(lowerer, base, &elem_ty, space, idx_expr, interner)?;
            let val = lowerer.builder.new_typed_var(elem_ty.clone());
            lowerer.builder.emit(KirOp::Load(val, addr, space));
            Ok((val, elem_ty))
        }
        ExprKind::UnaryOp { op, operand } => {
            let (src, src_ty) = lower_expr(lowerer, operand, interner)?;
            match op {
                UnaryOp::Neg => {
                    // PTX has no unsigned `neg`; an unsigned operand is
                    // negated as its signed twin.
                    let ty = match &src_ty {
                        KirType::U32 => KirType::I32,
                        KirType::U64 => KirType::I64,
                        other => other.clone(),
                    };
                    if !is_numeric(&ty) {
                        return Err(lowerer.refuse(
                            &format!("negating a value of type {ty:?}"),
                            "only numeric values negate",
                        ));
                    }
                    let src = lowerer.cast_to(src, &src_ty, &ty)?;
                    let dst = lowerer.builder.new_typed_var(ty.clone());
                    lowerer.builder.emit(KirOp::Neg(dst, src));
                    Ok((dst, ty))
                }
                UnaryOp::Not => {
                    if src_ty != KirType::Bool {
                        return Err(lowerer.refuse(
                            "`not` on something other than a comparison",
                            "e.g. `if not (i < n):`",
                        ));
                    }
                    let dst = lowerer.builder.new_typed_var(KirType::Bool);
                    lowerer.builder.emit(KirOp::Not(dst, src));
                    Ok((dst, KirType::Bool))
                }
            }
        }
        // `(a + b) * 0.5` — parentheses are pure grouping; lower the inner
        // expression directly.
        ExprKind::Paren(inner) => lower_expr(lowerer, inner, interner),
        other => Err(lowerer.refuse(
            &format!("{} expression", expr_kind_name(other)),
            KERNEL_SUPPORTED_HINT,
        )),
    }
}

/// `a op b` with the operands promoted to a common type. Comparisons
/// produce `Bool`; `and` / `or` take two `Bool`s; `%`, `&`, `|` take
/// integers.
fn lower_binop(
    lowerer: &mut KernelLowerer,
    op: BinOp,
    a: VarId,
    a_ty: KirType,
    b: VarId,
    b_ty: KirType,
) -> Result<(VarId, KirType), CodegenError> {
    // Predicates combine only with predicates.
    if matches!(op, BinOp::And | BinOp::Or) {
        if a_ty != KirType::Bool || b_ty != KirType::Bool {
            return Err(lowerer.refuse(
                &format!("`{}` on something other than two comparisons", binop_symbol(op)),
                "e.g. `if i < n and j < m:`",
            ));
        }
        let dst = lowerer.builder.new_typed_var(KirType::Bool);
        lowerer.builder.emit(match op {
            BinOp::And => KirOp::And(dst, a, b),
            _ => KirOp::Or(dst, a, b),
        });
        return Ok((dst, KirType::Bool));
    }
    if !is_numeric(&a_ty) || !is_numeric(&b_ty) {
        return Err(lowerer.refuse(
            &format!("`{}` between {a_ty:?} and {b_ty:?}", binop_symbol(op)),
            "arithmetic and comparisons take numbers; a comparison result is not one, \
             and a tensor parameter must be indexed first",
        ));
    }
    let common = promote_types(a_ty.clone(), b_ty.clone());
    let a = lowerer.cast_to(a, &a_ty, &common)?;
    let b = lowerer.cast_to(b, &b_ty, &common)?;

    let cmp = match op {
        BinOp::Lt => Some(CmpOp::Lt),
        BinOp::LtEq => Some(CmpOp::Le),
        BinOp::Gt => Some(CmpOp::Gt),
        BinOp::GtEq => Some(CmpOp::Ge),
        BinOp::Eq => Some(CmpOp::Eq),
        BinOp::NotEq => Some(CmpOp::Ne),
        _ => None,
    };
    if let Some(cmp) = cmp {
        let dst = lowerer.builder.new_typed_var(KirType::Bool);
        lowerer.builder.emit(KirOp::Cmp(dst, a, b, cmp));
        return Ok((dst, KirType::Bool));
    }

    let dst = lowerer.builder.new_typed_var(common.clone());
    let kir_op = match op {
        BinOp::Add => KirOp::Add(dst, a, b),
        BinOp::Sub => KirOp::Sub(dst, a, b),
        BinOp::Mul => KirOp::Mul(dst, a, b),
        BinOp::Div => KirOp::Div(dst, a, b),
        BinOp::FloorDiv | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr if !is_integer(&common) => {
            return Err(lowerer.refuse(
                &format!("`{}` on non-integer operands", binop_symbol(op)),
                "it is an integer operation; use `/` for float division",
            ));
        }
        BinOp::FloorDiv => KirOp::Div(dst, a, b),
        BinOp::Mod => KirOp::Rem(dst, a, b),
        BinOp::BitAnd => KirOp::And(dst, a, b),
        BinOp::BitOr => KirOp::Or(dst, a, b),
        other => {
            return Err(CodegenError::new(format!(
                "kernel '{}': binary operator '{}' is not supported in kernel \
                 code. Supported arithmetic: + - * / // % & |; supported comparisons: \
                 < <= > >= == !=; `and` / `or` combine comparisons.",
                lowerer.kernel_name,
                binop_symbol(other)
            )));
        }
    };
    lowerer.builder.emit(kir_op);
    Ok((dst, common))
}

/// The kernel builtins. `thread_id()` is the GLOBAL thread index
/// (`blockIdx * blockDim + threadIdx`), as it has always been for CUDA
/// kernel blocks; an optional literal dimension argument selects y / z.
fn lower_builtin_call(
    lowerer: &mut KernelLowerer,
    callee: &Expr,
    args: &[nsl_ast::expr::Arg],
    interner: &Interner,
) -> Result<(VarId, KirType), CodegenError> {
    const BUILTINS: &str = "only direct calls to the kernel builtins thread_id(), \
        thread_id_y(), block_id(), block_id_y(), block_dim(), global_id(), \
        sync_threads() are supported";
    let ExprKind::Ident(sym) = &callee.kind else {
        return Err(lowerer.refuse("an indirect or method call", BUILTINS));
    };
    let name = interner.resolve(sym.0).unwrap_or("");
    let dim = |lowerer: &KernelLowerer| -> Result<u8, CodegenError> {
        match args {
            [] => Ok(0),
            [a] if a.name.is_none() => match literal_int(&a.value) {
                Some(d @ 0..=2) => Ok(d as u8),
                _ => Err(lowerer.refuse(
                    &format!("`{name}` with a dimension that is not 0, 1 or 2"),
                    "e.g. `thread_id(1)` for the y index",
                )),
            },
            _ => Err(lowerer.refuse(
                &format!("`{name}` with those arguments"),
                "the builtins take no argument, or one literal dimension (0, 1 or 2)",
            )),
        }
    };
    let index_op = |lowerer: &mut KernelLowerer, op: KirOp| {
        lowerer.builder.emit(op);
    };
    match name {
        "thread_id" | "global_id" => {
            let d = dim(lowerer)?;
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::GlobalId(dst, d));
            Ok((dst, KirType::U32))
        }
        "thread_id_y" => {
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::GlobalId(dst, 1));
            Ok((dst, KirType::U32))
        }
        "block_id" | "block_idx" => {
            let d = dim(lowerer)?;
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::BlockIdx(dst, d));
            Ok((dst, KirType::U32))
        }
        "block_id_y" => {
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::BlockIdx(dst, 1));
            Ok((dst, KirType::U32))
        }
        "block_dim" => {
            let d = dim(lowerer)?;
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::BlockDim(dst, d));
            Ok((dst, KirType::U32))
        }
        "local_id" => {
            let d = dim(lowerer)?;
            let dst = lowerer.builder.new_typed_var(KirType::U32);
            index_op(lowerer, KirOp::ThreadId(dst, d));
            Ok((dst, KirType::U32))
        }
        "sync_threads" => {
            lowerer.builder.emit(KirOp::Barrier);
            // Barrier has no result value; return a DEFINED zero so a
            // (nonsensical but legal) `let x = sync_threads()` cannot
            // bind an uninitialized KIR value.
            lowerer.emit_const(&KirType::U32, 0).map(|v| (v, KirType::U32))
        }
        _ => {
            // Unrecognized call — used to fabricate a fresh F32 with no
            // defining op (uninitialized value). Refuse instead.
            Err(CodegenError::new(format!(
                "kernel '{}': call to unsupported function '{}' in kernel body. \
                 Supported kernel builtins: thread_id(), thread_id_y(), block_id(), \
                 block_id_y(), block_dim(), global_id(), sync_threads(). Math \
                 intrinsics (sqrt, exp, ...) are not implemented in the kernel \
                 lowering.",
                lowerer.kernel_name, name
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::block::KernelDef;
    use nsl_ast::decl::Param;
    use nsl_ast::expr::{Expr, ExprKind};
    use nsl_ast::pattern::{Pattern, PatternKind};
    use nsl_ast::stmt::{Block, Stmt, StmtKind};
    use nsl_ast::types::{DimExpr, TypeExpr, TypeExprKind};
    use nsl_ast::NodeId;
    use nsl_errors::Span;
    use string_interner::StringInterner;

    fn make_interner() -> Interner {
        StringInterner::new()
    }

    fn dummy_span() -> Span {
        Span {
            file_id: nsl_errors::FileId(0),
            start: nsl_errors::BytePos(0),
            end: nsl_errors::BytePos(0),
        }
    }

    fn make_empty_kernel(interner: &mut Interner) -> KernelDef {
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("test_kernel"));
        KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block {
                stmts: Vec::new(),
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        }
    }

    fn make_kernel_with_params(interner: &mut Interner) -> KernelDef {
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("add_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_b = nsl_ast::Symbol(interner.get_or_intern("b"));
        KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_a,
                    type_ann: None,
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_b,
                    type_ann: None,
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block {
                stmts: Vec::new(),
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        }
    }

    /// Helper to create a tensor type annotation with the given dtype.
    fn make_tensor_type_ann(interner: &mut Interner, dtype_str: &str) -> TypeExpr {
        let dtype_sym = nsl_ast::Symbol(interner.get_or_intern(dtype_str));
        TypeExpr {
            kind: TypeExprKind::Tensor {
                shape: vec![DimExpr::Concrete(1024)],
                dtype: dtype_sym,
                device: None,
            },
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    /// Helper to create a named (scalar) type annotation.
    fn make_named_type_ann(interner: &mut Interner, name: &str) -> TypeExpr {
        let sym = nsl_ast::Symbol(interner.get_or_intern(name));
        TypeExpr {
            kind: TypeExprKind::Named(sym),
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    // -----------------------------------------------------------------------
    // Task 1: Type promotion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_promotion_same_types() {
        assert_eq!(promote_types(KirType::F32, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I32, KirType::I32), KirType::I32);
        assert_eq!(promote_types(KirType::I64, KirType::I64), KirType::I64);
        assert_eq!(promote_types(KirType::U32, KirType::U32), KirType::U32);
        assert_eq!(promote_types(KirType::F16, KirType::F16), KirType::F16);
    }

    #[test]
    fn test_type_promotion_mixed_float() {
        assert_eq!(promote_types(KirType::F16, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::F32, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::F16, KirType::F64), KirType::F64);
        // Symmetric
        assert_eq!(promote_types(KirType::F32, KirType::F16), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::F32), KirType::F64);
    }

    #[test]
    fn test_type_promotion_int_float() {
        assert_eq!(promote_types(KirType::I32, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::I32, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I64, KirType::F32), KirType::F64); // precision safety
        assert_eq!(promote_types(KirType::I64, KirType::F64), KirType::F64);
        assert_eq!(promote_types(KirType::I32, KirType::F16), KirType::F32);
        // Symmetric
        assert_eq!(promote_types(KirType::F32, KirType::I32), KirType::F32);
        assert_eq!(promote_types(KirType::F64, KirType::I64), KirType::F64);
    }

    #[test]
    fn test_type_promotion_mixed_int() {
        assert_eq!(promote_types(KirType::I32, KirType::I64), KirType::I64);
        assert_eq!(promote_types(KirType::U32, KirType::U64), KirType::U64);
        assert_eq!(promote_types(KirType::U32, KirType::I32), KirType::I32);
    }

    #[test]
    fn test_type_promotion_bool() {
        assert_eq!(promote_types(KirType::Bool, KirType::F32), KirType::F32);
        assert_eq!(promote_types(KirType::Bool, KirType::I32), KirType::I32);
        assert_eq!(promote_types(KirType::F64, KirType::Bool), KirType::F64);
    }

    #[test]
    #[should_panic(expected = "Cannot promote types")]
    fn test_type_promotion_ptr_panics() {
        let ptr_a = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let ptr_b = KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global);
        promote_types(ptr_a, ptr_b);
    }

    // -----------------------------------------------------------------------
    // Task 2: Parameter type map tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_type_map_tensor_f64() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_b = nsl_ast::Symbol(interner.get_or_intern("b"));
        let p_out = nsl_ast::Symbol(interner.get_or_intern("out"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_a,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_b,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "i32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_out,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block {
                stmts: Vec::new(),
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let type_map = build_param_type_map(&kernel, &interner);

        assert_eq!(
            type_map["a"],
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global)
        );
        assert_eq!(
            type_map["b"],
            KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global)
        );
        assert_eq!(
            type_map["out"],
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global)
        );
    }

    #[test]
    fn test_param_type_map_scalar() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("scalar_kernel"));
        let p_n = nsl_ast::Symbol(interner.get_or_intern("n"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![Param {
                name: p_n,
                type_ann: Some(make_named_type_ann(&mut interner, "int")),
                default: None,
                is_variadic: false,
                span: dummy_span(),
            }],
            return_type: None,
            body: Block {
                stmts: Vec::new(),
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let type_map = build_param_type_map(&kernel, &interner);
        assert_eq!(type_map["n"], KirType::I32);
    }

    #[test]
    fn test_param_type_map_no_annotation_defaults_f32() {
        let mut interner = make_interner();
        let kernel = make_kernel_with_params(&mut interner);
        let type_map = build_param_type_map(&kernel, &interner);

        // No type annotation -> Ptr(F32, Global) default
        assert_eq!(
            type_map["a"],
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
        );
        assert_eq!(
            type_map["b"],
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
        );
    }

    // -----------------------------------------------------------------------
    // Task 3: Type propagation in expressions
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_propagation_f64_subscript() {
        // Kernel with f64 tensors: a[idx] + b[idx] should produce f64
        let mut interner = make_interner();
        let _name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_add"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        // Build: a[idx] + b[idx]
        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let mut lowerer = KernelLowerer::new("test");
        // Set up type map: a and b are Ptr(F64, Global)
        let a_var = lowerer
            .builder
            .new_typed_var(KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
        lowerer.var_map.insert("a".to_string(), a_var);
        lowerer.type_map.insert(
            "a".to_string(),
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        let b_var = lowerer
            .builder
            .new_typed_var(KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global));
        lowerer.var_map.insert("b".to_string(), b_var);
        lowerer.type_map.insert(
            "b".to_string(),
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global),
        );
        // idx is U32 (thread index)
        let idx_var = lowerer.builder.new_typed_var(KirType::U32);
        lowerer.var_map.insert("idx".to_string(), idx_var);
        lowerer.type_map.insert("idx".to_string(), KirType::U32);

        // Need a block to emit into
        let entry = lowerer.builder.new_block();
        lowerer.builder.set_block(entry);

        let (_, result_ty) = lower_expr(&mut lowerer, &add_expr, &interner)
            .expect("typed add expression must lower");
        assert_eq!(
            result_ty,
            KirType::F64,
            "a[idx]+b[idx] with F64 ptrs should produce F64"
        );
    }

    #[test]
    fn test_mixed_type_promotion_in_expr() {
        // a is Ptr(I32), b is Ptr(F32)
        // a[idx] + b[idx] should promote to F32
        let mut interner = make_interner();
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let mut lowerer = KernelLowerer::new("test");
        let a_var = lowerer
            .builder
            .new_typed_var(KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global));
        lowerer.var_map.insert("a".to_string(), a_var);
        lowerer.type_map.insert(
            "a".to_string(),
            KirType::Ptr(Box::new(KirType::I32), AddressSpace::Global),
        );
        let b_var = lowerer
            .builder
            .new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        lowerer.var_map.insert("b".to_string(), b_var);
        lowerer.type_map.insert(
            "b".to_string(),
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        );
        let idx_var = lowerer.builder.new_typed_var(KirType::U32);
        lowerer.var_map.insert("idx".to_string(), idx_var);
        lowerer.type_map.insert("idx".to_string(), KirType::U32);

        let entry = lowerer.builder.new_block();
        lowerer.builder.set_block(entry);

        let (_, result_ty) = lower_expr(&mut lowerer, &add_expr, &interner)
            .expect("typed add expression must lower");
        assert_eq!(result_ty, KirType::F32, "I32 + F32 should promote to F32");
    }

    // -----------------------------------------------------------------------
    // Task 4: Full kernel lowering with AST types
    // -----------------------------------------------------------------------

    #[test]
    fn test_lower_empty_kernel() {
        let mut interner = make_interner();
        let kernel = make_empty_kernel(&mut interner);
        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        assert_eq!(ir.name, "test_kernel");
        assert_eq!(ir.params.len(), 0);
        assert!(ir.is_well_formed());
        assert_eq!(ir.blocks.len(), 1); // entry block
    }

    #[test]
    fn test_lower_params() {
        let mut interner = make_interner();
        let kernel = make_kernel_with_params(&mut interner);
        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        assert_eq!(ir.name, "add_kernel");
        assert_eq!(ir.params.len(), 2);
        assert_eq!(ir.params[0].name, "a");
        assert_eq!(ir.params[1].name, "b");
        // No type annotation -> default Ptr(F32, Global)
        assert_eq!(
            ir.params[0].ty,
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
        );
    }

    #[test]
    fn test_lower_f64_params() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("f64_kernel"));
        let p_a = nsl_ast::Symbol(interner.get_or_intern("a"));
        let p_out = nsl_ast::Symbol(interner.get_or_intern("out"));

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: p_a,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: p_out,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block {
                stmts: Vec::new(),
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        assert_eq!(
            ir.params[0].ty,
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global)
        );
        assert_eq!(
            ir.params[1].ty,
            KirType::Ptr(Box::new(KirType::F64), AddressSpace::Global)
        );
    }

    #[test]
    fn test_lower_basic_ops() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("arith_kernel"));
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        // Build: let x = 1.0 + 2.0
        let left = Expr {
            kind: ExprKind::FloatLiteral(1.0),
            span: dummy_span(),
            id: NodeId::next(),
        };
        let right = Expr {
            kind: ExprKind::FloatLiteral(2.0),
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(left),
                op: BinOp::Add,
                right: Box::new(right),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let var_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(x_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block {
                stmts: vec![var_decl],
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        // Should have ops: Const(1.0), Const(2.0), Add
        assert!(
            ir.op_count() >= 3,
            "expected at least 3 ops, got {}",
            ir.op_count()
        );
    }

    #[test]
    fn test_feature_tracking() {
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("sync_kernel"));
        let sync_sym = nsl_ast::Symbol(interner.get_or_intern("sync_threads"));

        // Build: sync_threads()
        let call_expr = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(sync_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let expr_stmt = Stmt {
            kind: StmtKind::Expr(call_expr),
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: Vec::new(),
            return_type: None,
            body: Block {
                stmts: vec![expr_stmt],
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        // sync_threads() should set SHARED_MEMORY feature
        assert!(
            ir.required_features
                .contains(crate::gpu_target::FeatureSet::SHARED_MEMORY),
            "sync_threads() should require SHARED_MEMORY feature"
        );
    }

    // -----------------------------------------------------------------------
    // Task 5: Local variable type inference
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_variable_type_inference() {
        // let idx = global_id(); let x = a[idx]  -> x should be F64 if a is Ptr(F64)
        //
        // NOTE: this test originally referenced `idx` WITHOUT declaring it,
        // relying on the old fabrication path that minted an uninitialized
        // placeholder for unknown identifiers. Unknown identifiers now refuse
        // (deferral-must-refuse), so the test declares `idx` properly.
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("local_infer_kernel"));
        let a_sym_param = nsl_ast::Symbol(interner.get_or_intern("a"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        // Build: let idx = global_id()
        let global_id_call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(nsl_ast::Symbol(interner.get_or_intern("global_id"))),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let idx_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(global_id_call),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        // Build: let x = a[idx]
        let a_subscript = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym_param),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let var_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(x_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(a_subscript),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![Param {
                name: a_sym_param,
                type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                default: None,
                is_variadic: false,
                span: dummy_span(),
            }],
            return_type: None,
            body: Block {
                stmts: vec![idx_decl, var_decl],
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");

        // The load from a[idx] should produce an F64 value.
        // Check that the loaded variable has F64 type in var_types.
        let f64_vars: Vec<_> = ir
            .var_types
            .iter()
            .filter(|(_, ty)| **ty == KirType::F64)
            .collect();
        assert!(
            !f64_vars.is_empty(),
            "Expected at least one F64 variable from `let x = a[idx]` with F64 tensor 'a'. \
             var_types: {:?}",
            ir.var_types,
        );
    }

    // -----------------------------------------------------------------------
    // Task 6: E2E PTX tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_f64_kernel_ptx_uses_f64_ops() {
        // Build an F64 kernel: a[idx] + b[idx] -> out[idx]
        // Verify the PTX output uses .f64 operations.
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("add_f64"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let out_sym = nsl_ast::Symbol(interner.get_or_intern("out"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        // Build body:
        //   let idx = global_id()
        //   let val = a[idx] + b[idx]
        // (We don't implement store via statement yet, so test just the typed expressions)

        let global_id_call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(nsl_ast::Symbol(interner.get_or_intern("global_id"))),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let idx_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(global_id_call),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let val_sym = nsl_ast::Symbol(interner.get_or_intern("val"));
        let val_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(val_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: a_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: b_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: out_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f64")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block {
                stmts: vec![idx_decl, val_decl],
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");
        let ptx_bytes = crate::backend_ptx::lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]).to_string();

        // F64 kernel: should use .f64 for the add operation
        assert!(
            ptx.contains("add.f64"),
            "F64 kernel should produce add.f64 in PTX. Got:\n{ptx}"
        );
        // The add should NOT be f32
        assert!(
            !ptx.contains("add.f32"),
            "F64 kernel should NOT produce add.f32. Got:\n{ptx}"
        );
        // Load should be f64
        assert!(
            ptx.contains("ld.global.f64"),
            "F64 kernel should load .f64 from global. Got:\n{ptx}"
        );
    }

    #[test]
    fn test_mixed_precision_kernel_ptx() {
        // a is Tensor<[1024], i32>, b is Tensor<[1024], f32>
        // a[idx] + b[idx] should produce cvt (cast) + add.f32
        let mut interner = make_interner();
        let name_sym = nsl_ast::Symbol(interner.get_or_intern("mixed_add"));
        let a_sym = nsl_ast::Symbol(interner.get_or_intern("a"));
        let b_sym = nsl_ast::Symbol(interner.get_or_intern("b"));
        let idx_sym = nsl_ast::Symbol(interner.get_or_intern("idx"));

        let global_id_call = Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident(nsl_ast::Symbol(interner.get_or_intern("global_id"))),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                args: Vec::new(),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let idx_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(global_id_call),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let a_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(a_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let b_idx = Expr {
            kind: ExprKind::Subscript {
                object: Box::new(Expr {
                    kind: ExprKind::Ident(b_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                }),
                index: Box::new(SubscriptKind::Index(Expr {
                    kind: ExprKind::Ident(idx_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                })),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };
        let add_expr = Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(a_idx),
                op: BinOp::Add,
                right: Box::new(b_idx),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let val_sym = nsl_ast::Symbol(interner.get_or_intern("val"));
        let val_decl = Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: Pattern {
                    kind: PatternKind::Ident(val_sym),
                    span: dummy_span(),
                    id: NodeId::next(),
                },
                type_ann: None,
                value: Some(add_expr),
            },
            span: dummy_span(),
            id: NodeId::next(),
        };

        let kernel = KernelDef {
            name: name_sym,
            params: vec![
                Param {
                    name: a_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "i32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
                Param {
                    name: b_sym,
                    type_ann: Some(make_tensor_type_ann(&mut interner, "f32")),
                    default: None,
                    is_variadic: false,
                    span: dummy_span(),
                },
            ],
            return_type: None,
            body: Block {
                stmts: vec![idx_decl, val_decl],
                span: dummy_span(),
            },
            decorators: Vec::new(),
            span: dummy_span(),
        };

        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .expect("portable-subset kernel must lower");
        let ptx_bytes = crate::backend_ptx::lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]).to_string();

        // I32 + F32 should promote to F32
        assert!(
            ptx.contains("add.f32"),
            "Mixed I32+F32 kernel should produce add.f32 in PTX. Got:\n{ptx}"
        );
        // Should have a cvt (cast from i32 to f32). PTX spells the signed
        // 32-bit type `.s32`, and an int → float conversion carries `.rn`
        // (roadmap A2 step 4; this test pinned `cvt.f32.i32`, which ptxas
        // rejects, until the printer went through the assembler).
        assert!(
            ptx.contains("cvt.rn.f32.s32"),
            "Mixed I32+F32 kernel should produce cvt.rn.f32.s32 in PTX. Got:\n{ptx}"
        );
        // Load from a should be s32
        assert!(
            ptx.contains("ld.global.s32"),
            "I32 tensor should load as .s32. Got:\n{ptx}"
        );
        // Load from b should be f32
        assert!(
            ptx.contains("ld.global.f32"),
            "F32 tensor should load as .f32. Got:\n{ptx}"
        );
    }

    // -----------------------------------------------------------------------
    // dtype_str_to_kir coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_str_to_kir() {
        assert_eq!(dtype_str_to_kir("f16"), KirType::F16);
        assert_eq!(dtype_str_to_kir("fp16"), KirType::F16);
        assert_eq!(dtype_str_to_kir("bf16"), KirType::Bf16);
        assert_eq!(dtype_str_to_kir("f32"), KirType::F32);
        assert_eq!(dtype_str_to_kir("fp32"), KirType::F32);
        assert_eq!(dtype_str_to_kir("float"), KirType::F32);
        assert_eq!(dtype_str_to_kir("f64"), KirType::F64);
        assert_eq!(dtype_str_to_kir("float64"), KirType::F64);
        assert_eq!(dtype_str_to_kir("double"), KirType::F64);
        assert_eq!(dtype_str_to_kir("i32"), KirType::I32);
        assert_eq!(dtype_str_to_kir("int"), KirType::I32);
        assert_eq!(dtype_str_to_kir("i64"), KirType::I64);
        assert_eq!(dtype_str_to_kir("u32"), KirType::U32);
        assert_eq!(dtype_str_to_kir("u64"), KirType::U64);
        assert_eq!(dtype_str_to_kir("bool"), KirType::Bool);
        assert_eq!(dtype_str_to_kir("fp8"), KirType::F16);
        assert_eq!(dtype_str_to_kir("unknown"), KirType::F32); // fallback
    }

    // -----------------------------------------------------------------------
    // Source-level coverage: the constructs a `kernel` block may use, and the
    // refusals for those it may not (deferral-must-refuse: a construct
    // outside the set is a loud CodegenError, never a fabricated value or a
    // silently dropped statement).
    // -----------------------------------------------------------------------

    /// Parse NSL source and return the first `kernel` definition found
    /// (bare or decorated), together with the interner.
    fn parse_first_kernel(src: &str) -> (KernelDef, Interner) {
        let mut interner = make_interner();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let errs: Vec<_> = lex_diags
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        let errs: Vec<_> = parsed
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(errs.is_empty(), "parse errors: {errs:?}");
        for stmt in &parsed.module.stmts {
            match &stmt.kind {
                StmtKind::KernelDef(k) => return (k.clone(), interner),
                StmtKind::Decorated { stmt: inner, .. } => {
                    if let StmtKind::KernelDef(k) = &inner.kind {
                        return (k.clone(), interner);
                    }
                }
                _ => {}
            }
        }
        panic!("no kernel definition in test source");
    }

    /// Lower the first kernel in `src`, expecting a refusal; returns the message.
    fn lower_err(src: &str) -> String {
        let (kernel, interner) = parse_first_kernel(src);
        match lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda) {
            Ok(_) => panic!("expected refusal, but kernel lowered successfully"),
            Err(e) => e.to_string(),
        }
    }

    /// Lower the first kernel in `src`, expecting success; returns the
    /// verified IR.
    fn lower_ok(src: &str) -> KernelIR {
        let (kernel, interner) = parse_first_kernel(src);
        lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda)
            .unwrap_or_else(|e| panic!("valid kernel must lower: {e}"))
    }

    /// Compile the first kernel in `src` to PTX text, expecting success.
    fn ptx_ok(src: &str) -> String {
        let (kernel, interner) = parse_first_kernel(src);
        let bytes = compile_kernel_ptx(&kernel, &interner)
            .unwrap_or_else(|e| panic!("valid kernel must compile: {e}"));
        assert_eq!(bytes.last(), Some(&0), "PTX must be null-terminated");
        String::from_utf8_lossy(&bytes[..bytes.len() - 1]).to_string()
    }

    // ── the kernels the e2e fixtures run ────────────────────────────────

    #[test]
    fn test_valid_vec_add_compiles() {
        // Mirrors tests/m17_kernel_test.nsl
        let ptx = ptx_ok(
            "kernel vec_add(a, b, c):\n    let i = thread_id()\n    c[i] = a[i] + b[i]\n",
        );
        assert!(ptx.contains(".visible .entry vec_add("), "PTX:\n{ptx}");
        assert!(ptx.contains("ld.global.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("add.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX:\n{ptx}");
        // `thread_id()` is the GLOBAL index, as it has always been for CUDA
        // kernel blocks.
        assert!(ptx.contains("mov.u32 %gid0, %ctaid.x;"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_valid_if_comparison_compiles() {
        let ptx = ptx_ok(
            "kernel guarded(a, out):\n    let i = thread_id()\n    if i < 4:\n        out[i] = a[i]\n",
        );
        assert!(ptx.contains("setp.lt.u32"), "PTX:\n{ptx}");
        assert!(ptx.contains("bra BB"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_autotune_constant_substitution_still_compiles() {
        // A module-level constant substituted by @autotune must compile...
        let src = "kernel scaled(a, out):\n    let i = thread_id()\n    out[i] = a[i] * BLOCK\n";
        let (kernel, interner) = parse_first_kernel(src);
        let mut constants = HashMap::new();
        constants.insert("BLOCK".to_string(), 4i64);
        let bytes = compile_kernel_ptx_with_constants(&kernel, &interner, &constants)
            .expect("constant-substituted kernel must compile");
        assert!(!bytes.is_empty());

        // ...and the same kernel WITHOUT the substitution must refuse loudly
        // (this is exactly the unknown-module-const fabrication the old code
        // papered over with a dummy register).
        let err = compile_kernel_ptx(&kernel, &interner)
            .expect_err("unknown ident must refuse")
            .to_string();
        assert!(err.contains("BLOCK"), "err: {err}");
        assert!(err.contains("@autotune"), "err: {err}");
    }

    #[test]
    fn test_autotune_substitution_reaches_loop_bounds() {
        let src = "kernel tiled(a, out):\n    let i = thread_id()\n    let acc = 0.0\n    for j in range(0, TILE):\n        acc = acc + a[i * TILE + j]\n    out[i] = acc\n";
        let (kernel, interner) = parse_first_kernel(src);
        let mut constants = HashMap::new();
        constants.insert("TILE".to_string(), 8i64);
        compile_kernel_ptx_with_constants(&kernel, &interner, &constants)
            .expect("TILE is substituted in the range bound and the index");
    }

    #[test]
    fn test_paren_expression_compiles_as_grouping() {
        let ptx = ptx_ok(
            "kernel avg(a, b, out):\n    let i = thread_id()\n    out[i] = (a[i] + b[i]) * 0.5\n",
        );
        assert!(ptx.contains("add.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("mul.f32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_float_division_carries_a_rounding_mode() {
        // `div.f32` is not PTX; the printer spells `/` as `div.rn.f32`.
        let ptx = ptx_ok(
            "kernel halve(a, out):\n    let i = thread_id()\n    out[i] = a[i] / 2.0\n",
        );
        assert!(ptx.contains("div.rn.f32"), "PTX:\n{ptx}");
    }

    // ── control flow and reassigned locals (roadmap A2 step 3) ─────────

    #[test]
    fn test_if_elif_else_assigning_a_local_joins_through_a_block_parameter() {
        let ir = lower_ok(
            "kernel bucket(a, out):\n    let i = thread_id()\n    let x = 0.0\n    if a[i] < 1.0:\n        x = 1.0\n    elif a[i] < 2.0:\n        x = 2.0\n    else:\n        x = 3.0\n    out[i] = x\n",
        );
        // Exactly one block has a parameter: the join, carrying `x`.
        let joins: Vec<_> = ir.blocks.iter().filter(|b| !b.params.is_empty()).collect();
        assert_eq!(joins.len(), 1, "{:?}", ir.blocks);
        assert_eq!(joins[0].params.len(), 1);
        assert_eq!(joins[0].params[0].ty, KirType::F32);
        // Three arms plus the chain blocks all branch into it with one argument.
        let into_join = ir
            .blocks
            .iter()
            .filter_map(|b| b.terminator.as_ref())
            .flat_map(|t| t.edges())
            .filter(|e| e.target == joins[0].id)
            .count();
        assert_eq!(into_join, 3);
    }

    #[test]
    fn test_if_without_else_passes_the_old_value_on_the_fallthrough_edge() {
        let ir = lower_ok(
            "kernel clamp(a, out):\n    let i = thread_id()\n    let x = a[i]\n    if x > 1.0:\n        x = 1.0\n    out[i] = x\n",
        );
        let join = ir.blocks.iter().find(|b| !b.params.is_empty()).expect("a join");
        let args: Vec<_> = ir
            .blocks
            .iter()
            .filter_map(|b| b.terminator.as_ref())
            .flat_map(|t| t.edges())
            .filter(|e| e.target == join.id)
            .map(|e| e.args[0])
            .collect();
        assert_eq!(args.len(), 2);
        assert_ne!(args[0], args[1], "the two edges carry different values of x");
    }

    #[test]
    fn test_a_let_inside_an_arm_is_scoped_to_the_arm() {
        let err = lower_err(
            "kernel scoped(a, out):\n    let i = thread_id()\n    if i < 4:\n        let y = a[i]\n    out[i] = y\n",
        );
        assert!(err.contains("unknown identifier 'y'"), "err: {err}");
    }

    #[test]
    fn test_for_range_loop_with_accumulator() {
        let ptx = ptx_ok(
            "kernel rowsum(a, out):\n    let i = thread_id()\n    let acc = 0.0\n    for j in range(0, 4):\n        acc = acc + a[i * 4 + j]\n    out[i] = acc\n",
        );
        // The header takes `j` and `acc`; the back edge is a parallel copy.
        assert!(ptx.contains("setp.lt.u32"), "PTX:\n{ptx}");
        // Entry edge and back edge both target the header.
        assert_eq!(ptx.matches("bra BB1;").count(), 2, "PTX:\n{ptx}");
        let (kernel, interner) = parse_first_kernel(
            "kernel rowsum(a, out):\n    let i = thread_id()\n    let acc = 0.0\n    for j in range(0, 4):\n        acc = acc + a[i * 4 + j]\n    out[i] = acc\n",
        );
        let ir = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda).unwrap();
        let header = ir.blocks.iter().find(|b| b.params.len() == 2).expect("header j, acc");
        assert_eq!(header.params[0].ty, KirType::U32);
        assert_eq!(header.params[1].ty, KirType::F32);
    }

    #[test]
    fn test_for_range_forms() {
        // range(n), range(a, b), range(a, b, step), a..b, a..=b
        lower_ok("kernel r1(a, out):\n    let i = thread_id()\n    for j in range(4):\n        out[i * 4 + j] = a[j]\n");
        lower_ok("kernel r3(a, out):\n    let i = thread_id()\n    for j in range(0, 8, 2):\n        out[i * 8 + j] = a[j]\n");
        lower_ok("kernel r4(a, out):\n    let i = thread_id()\n    for j in 0..4:\n        out[i * 4 + j] = a[j]\n");
        lower_ok("kernel r5(a, out):\n    let i = thread_id()\n    for j in 0..=3:\n        out[i * 4 + j] = a[j]\n");
        // A negative step counts down (`setp.gt`).
        let ptx = ptx_ok("kernel down(a, out):\n    let i = thread_id()\n    for j in range(3, 0, -1):\n        out[i * 4 + j] = a[j]\n");
        assert!(ptx.contains("setp.gt.u32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_loop_variable_is_scoped_to_the_loop() {
        let err = lower_err(
            "kernel leak(a, out):\n    let i = thread_id()\n    for j in range(0, 4):\n        out[j] = a[j]\n    out[i] = a[j]\n",
        );
        assert!(err.contains("unknown identifier 'j'"), "err: {err}");
    }

    #[test]
    fn test_while_loop_with_break_and_continue() {
        let ir = lower_ok(
            "kernel scan(a, out):\n    let i = thread_id()\n    let j = 0\n    let hits = 0\n    while j < 16:\n        j = j + 1\n        if a[i * 16 + j] < 0.0:\n            continue\n        if a[i * 16 + j] > 100.0:\n            break\n        hits = hits + 1\n    out[i] = hits\n",
        );
        // Header and exit both carry `hits` and `j`.
        let carriers: Vec<_> = ir.blocks.iter().filter(|b| b.params.len() == 2).collect();
        assert_eq!(carriers.len(), 2, "{:?}", ir.blocks);
    }

    #[test]
    fn test_compound_assignment_on_locals_and_elements() {
        let ptx = ptx_ok(
            "kernel acc(a, out):\n    let i = thread_id()\n    let s = 1.0\n    s += a[i]\n    s *= 2.0\n    out[i] += s\n",
        );
        assert!(ptx.contains("ld.global.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_assignment_converts_to_the_declared_type() {
        // `x` is declared f32; assigning the u32 literal converts.
        let ptx = ptx_ok(
            "kernel conv(a, out):\n    let i = thread_id()\n    let x: f32 = 0\n    if i < 4:\n        x = 2\n    out[i] = x\n",
        );
        assert!(ptx.contains("cvt.rn.f32.u32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_bare_return_and_a_guard_at_the_top() {
        let ir = lower_ok(
            "kernel guard(a, out, n):\n    let i = thread_id()\n    if i >= n[0]:\n        return\n    out[i] = a[i]\n",
        );
        let returns = ir
            .blocks
            .iter()
            .filter(|b| matches!(b.terminator, Some(KirTerminator::Return)))
            .count();
        assert_eq!(returns, 2);
    }

    #[test]
    fn test_index_builtins_and_dimension_arguments() {
        let ptx = ptx_ok(
            "kernel idx(out):\n    let x = thread_id()\n    let y = thread_id_y()\n    let b = block_id()\n    let d = block_dim()\n    let z = thread_id(2)\n    out[x + y + b + d + z] = 1.0\n",
        );
        assert!(ptx.contains("%ctaid.y"), "PTX:\n{ptx}");
        assert!(ptx.contains("%ctaid.z"), "PTX:\n{ptx}");
        assert!(ptx.contains("%ntid.x"), "PTX:\n{ptx}");
        let err = lower_err("kernel idx3(out):\n    let x = thread_id(3)\n    out[x] = 1.0\n");
        assert!(err.contains("dimension"), "err: {err}");
    }

    #[test]
    fn test_negating_an_unsigned_literal_goes_signed() {
        let ptx = ptx_ok("kernel neg(out):\n    let i = thread_id()\n    let m = -1\n    out[i] = m\n");
        assert!(ptx.contains("neg.s32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_eq_comparison_is_supported() {
        let ir = lower_ok("kernel k_eq(a):\n    let x = 1 == 2\n");
        assert!(ir.is_well_formed());
    }

    #[test]
    fn test_and_or_combine_comparisons() {
        lower_ok("kernel band(a, out):\n    let i = thread_id()\n    if i > 2 and i < 8 or i == 0:\n        out[i] = a[i]\n");
    }

    // ── refusals: statements ────────────────────────────────────────────

    #[test]
    fn test_refuses_return_with_a_value() {
        let err = lower_err("kernel bad_ret(a, out):\n    let i = thread_id()\n    return a[i]\n");
        assert!(err.contains("`return` with a value"), "err: {err}");
    }

    #[test]
    fn test_refuses_statement_after_break() {
        let err = lower_err(
            "kernel dead(a, out):\n    let i = thread_id()\n    let j = 0\n    while j < 4:\n        j = j + 1\n        break\n        out[i] = a[i]\n",
        );
        assert!(err.contains("after `break`"), "err: {err}");
    }

    #[test]
    fn test_refuses_break_outside_a_loop() {
        let err = lower_err("kernel stray(a, out):\n    let i = thread_id()\n    break\n");
        assert!(err.contains("`break` outside a loop"), "err: {err}");
    }

    #[test]
    fn test_refuses_for_over_something_other_than_range() {
        let err = lower_err("kernel k_for(a):\n    for x in a:\n        let y = x\n");
        assert!(err.contains("other than `range(...)`"), "err: {err}");
    }

    #[test]
    fn test_refuses_assigning_the_loop_variable() {
        let err = lower_err(
            "kernel k_j(a, out):\n    for j in range(0, 4):\n        j = 0\n        out[j] = a[j]\n",
        );
        assert!(err.contains("loop variable `j`"), "err: {err}");
    }

    #[test]
    fn test_refuses_assignment_to_an_undeclared_local() {
        let err = lower_err("kernel bad_assign(a, out):\n    x = 2.0\n");
        assert!(err.contains("undeclared variable 'x'"), "err: {err}");
    }

    #[test]
    fn test_refuses_destructuring_let() {
        let err = lower_err("kernel bad_let(a, out):\n    let (x, y) = (1, 2)\n");
        assert!(err.contains("destructuring"), "err: {err}");
    }

    #[test]
    fn test_refuses_store_to_unknown_base() {
        let err = lower_err(
            "kernel bad_base(a, out):\n    let i = thread_id()\n    bogus[i] = a[i]\n",
        );
        assert!(err.contains("bogus"), "err: {err}");
    }

    // ── refusals: expressions ───────────────────────────────────────────

    #[test]
    fn test_refuses_unknown_identifier() {
        let err = lower_err("kernel k_ident(a):\n    let x = MISSING\n");
        assert!(err.contains("MISSING"), "err: {err}");
        assert!(err.contains("unknown identifier"), "err: {err}");
        assert!(err.contains("@autotune"), "err: {err}");
    }

    #[test]
    fn test_refuses_unknown_call() {
        let err = lower_err("kernel k_call(a):\n    let x = sqrt(2.0)\n");
        assert!(err.contains("sqrt"), "err: {err}");
        assert!(err.contains("thread_id"), "err: {err}");
    }

    #[test]
    fn test_refuses_pow_binop() {
        let err = lower_err("kernel bad_pow(a, out):\n    let i = thread_id()\n    out[i] = a[i] ** 2.0\n");
        assert!(err.contains("'**'"), "err: {err}");
    }

    #[test]
    fn test_refuses_mod_on_floats() {
        let err = lower_err("kernel k_mod(a):\n    let x = 5.0 % 2.0\n");
        assert!(err.contains("`%`"), "err: {err}");
        // ...but integer `%` is `rem`.
        let ptx = ptx_ok("kernel k_rem(a, out):\n    let i = thread_id()\n    out[i] = a[i % 2]\n");
        assert!(ptx.contains("rem.u32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_refuses_unary_not_on_a_number() {
        let err = lower_err("kernel k_not(a):\n    let x = not 1\n");
        assert!(err.contains("not"), "err: {err}");
    }

    #[test]
    fn test_refuses_non_comparison_if_condition() {
        let err = lower_err(
            "kernel bad_cond(flag, out):\n    let i = thread_id()\n    if flag:\n        out[i] = 1.0\n",
        );
        assert!(err.contains("comparison"), "err: {err}");
    }

    #[test]
    fn test_refuses_a_float_index() {
        let err = lower_err("kernel fidx(a, out):\n    let i = thread_id()\n    out[i] = a[1.5]\n");
        assert!(err.contains("index of type F32"), "err: {err}");
    }

    #[test]
    fn test_refusal_carries_the_innermost_span() {
        // `**` on line 3 col 14 of the source; the error points at the
        // operator expression, not at the whole assignment.
        let src = "kernel bad_pow(a, out):\n    let i = thread_id()\n    out[i] = a[i] ** 2.0\n";
        let (kernel, interner) = parse_first_kernel(src);
        let err = lower_kernel_to_ir(&kernel, &interner, GpuTarget::Cuda).unwrap_err();
        let span = err.span.expect("spanned");
        let line_start = src.rfind("out[i] = ").unwrap();
        let col = span.start.0 as usize - line_start;
        assert_eq!(&src[line_start + col..line_start + col + 4], "a[i]", "span at {span:?}");
    }
}
