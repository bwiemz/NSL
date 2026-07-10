//! KernelCompiler: translates KernelDef AST nodes into PTX strings.
//!
//! Deferral-must-refuse invariant: any construct this compiler cannot lower
//! MUST produce a loud `CodegenError` instead of fabricating placeholder
//! registers or silently dropping statements — the old lenient behavior
//! compiled kernels that produced wrong numbers at runtime (dropped
//! bounds-guards, uninitialized registers for unknown calls, etc.).

use std::collections::HashMap;

use nsl_ast::block::KernelDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::operator::{AssignOp, BinOp};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_lexer::Interner;

use crate::error::CodegenError;

/// Shared "what IS supported" hint appended to kernel refusal messages.
const KERNEL_SUPPORTED_HINT: &str = "Supported kernel constructs: `let` bindings, \
arithmetic (+ - * /), comparisons (< <= > >=), element loads (a[i]), element \
stores (out[i] = v), if/elif/else with comparison conditions, and the builtins \
thread_id(), thread_id_y(), block_id(), block_dim(), sync_threads().";

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

#[derive(Clone, Copy, PartialEq)]
enum RegKind {
    U64, // integer / pointer
    F32, // float
}

pub struct KernelCompiler {
    ptx: String,
    next_u32: u32,
    next_u64: u32,
    next_f32: u32,
    next_pred: u32,
    next_label: u32,
    /// Variable → (register_name, RegKind)
    var_regs: HashMap<String, (String, RegKind)>,
    /// Kernel name, used in refusal diagnostics.
    kernel_name: String,
}

impl Default for KernelCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelCompiler {
    pub fn new() -> Self {
        KernelCompiler {
            ptx: String::new(),
            next_u32: 0,
            next_u64: 0,
            next_f32: 0,
            next_pred: 0,
            next_label: 0,
            var_regs: HashMap::new(),
            kernel_name: String::new(),
        }
    }

    /// Build a refusal error for an unsupported construct in this kernel.
    fn refuse(&self, construct: &str, hint: &str) -> CodegenError {
        CodegenError::new(format!(
            "kernel '{}': {} is not supported in `kernel` blocks. {}",
            self.kernel_name, construct, hint
        ))
    }

    /// Compile a KernelDef with constant substitutions applied.
    ///
    /// Clones the kernel AST, walks all expressions replacing `Ident` nodes
    /// whose names appear in `constants` with `IntLiteral` values, then compiles
    /// the modified AST to PTX. Used by @autotune to generate one PTX variant
    /// per parameter combination (e.g., BLOCK_SIZE=128, TILE_SIZE=32).
    pub fn compile_with_constants(
        kernel: &KernelDef,
        interner: &Interner,
        constants: &HashMap<String, i64>,
    ) -> Result<Vec<u8>, CodegenError> {
        if constants.is_empty() {
            return Self::compile(kernel, interner);
        }

        // Clone the kernel and substitute constants throughout the body
        let mut kernel_clone = kernel.clone();
        Self::substitute_block_constants(&mut kernel_clone.body, interner, constants);
        Self::compile(&kernel_clone, interner)
    }

    /// Walk a Block, substituting Ident nodes matching constant names with IntLiteral.
    fn substitute_block_constants(
        block: &mut Block,
        interner: &Interner,
        constants: &HashMap<String, i64>,
    ) {
        for stmt in &mut block.stmts {
            Self::substitute_stmt_constants(stmt, interner, constants);
        }
    }

    /// Walk a Stmt, substituting constants in all nested expressions.
    fn substitute_stmt_constants(
        stmt: &mut Stmt,
        interner: &Interner,
        constants: &HashMap<String, i64>,
    ) {
        match &mut stmt.kind {
            StmtKind::VarDecl {
                value: Some(expr), ..
            } => {
                Self::substitute_expr_constants(expr, interner, constants);
            }
            StmtKind::Expr(expr) => {
                Self::substitute_expr_constants(expr, interner, constants);
            }
            StmtKind::Assign { target, value, .. } => {
                Self::substitute_expr_constants(target, interner, constants);
                Self::substitute_expr_constants(value, interner, constants);
            }
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                Self::substitute_expr_constants(condition, interner, constants);
                Self::substitute_block_constants(then_block, interner, constants);
                for (elif_cond, elif_block) in elif_clauses {
                    Self::substitute_expr_constants(elif_cond, interner, constants);
                    Self::substitute_block_constants(elif_block, interner, constants);
                }
                if let Some(else_blk) = else_block {
                    Self::substitute_block_constants(else_blk, interner, constants);
                }
            }
            StmtKind::Return(Some(expr)) => {
                Self::substitute_expr_constants(expr, interner, constants);
            }
            _ => {}
        }
    }

    /// Recursively substitute Ident nodes matching constant names with IntLiteral.
    fn substitute_expr_constants(
        expr: &mut Expr,
        interner: &Interner,
        constants: &HashMap<String, i64>,
    ) {
        match &mut expr.kind {
            ExprKind::Ident(sym) => {
                if let Some(name) = interner.resolve(sym.0) {
                    if let Some(&value) = constants.get(name) {
                        expr.kind = ExprKind::IntLiteral(value);
                    }
                }
            }
            ExprKind::BinaryOp { left, right, .. } => {
                Self::substitute_expr_constants(left, interner, constants);
                Self::substitute_expr_constants(right, interner, constants);
            }
            ExprKind::UnaryOp { operand, .. } => {
                Self::substitute_expr_constants(operand, interner, constants);
            }
            ExprKind::Call { callee, args } => {
                Self::substitute_expr_constants(callee, interner, constants);
                for arg in args {
                    Self::substitute_expr_constants(&mut arg.value, interner, constants);
                }
            }
            ExprKind::Subscript { object, index } => {
                Self::substitute_expr_constants(object, interner, constants);
                if let SubscriptKind::Index(idx_expr) = index.as_mut() {
                    Self::substitute_expr_constants(idx_expr, interner, constants);
                }
            }
            _ => {} // Literals and other leaf nodes: no substitution needed
        }
    }

    /// Compile a KernelDef into a null-terminated PTX byte string.
    ///
    /// Refuses (returns `Err`) on any construct outside the supported subset;
    /// see `KERNEL_SUPPORTED_HINT` for the supported set.
    pub fn compile(kernel: &KernelDef, interner: &Interner) -> Result<Vec<u8>, CodegenError> {
        let mut kc = KernelCompiler::new();
        let name = interner.resolve(kernel.name.0).unwrap_or("__kernel");
        kc.kernel_name = name.to_string();

        // PTX header
        kc.emit(".version 7.0\n");
        kc.emit(".target sm_70\n");
        kc.emit(".address_size 64\n\n");

        // Entry point signature
        kc.emit(&format!(".visible .entry {}(\n", name));
        for (i, param) in kernel.params.iter().enumerate() {
            let pname = interner.resolve(param.name.0).unwrap_or("_p");
            kc.emit(&format!("    .param .u64 {}", pname));
            if i < kernel.params.len() - 1 {
                kc.emit(",");
            }
            kc.emit("\n");
        }
        kc.emit(") {\n");

        // Placeholder for register declarations (filled after body compilation)
        kc.emit("    // REGISTERS\n");

        // Load parameters into registers
        for param in &kernel.params {
            let pname = interner.resolve(param.name.0).unwrap_or("_p").to_string();
            let rd = kc.alloc_u64();
            kc.emit(&format!("    ld.param.u64 {}, [{}];\n", rd, pname));
            kc.var_regs.insert(pname, (rd, RegKind::U64));
        }

        // Compile body
        kc.compile_block(&kernel.body, interner)?;

        kc.emit("DONE:\n    ret;\n}\n");

        // Replace register placeholder with actual declarations
        let reg_decls = kc.gen_reg_decls();
        kc.ptx = kc.ptx.replacen("    // REGISTERS\n", &reg_decls, 1);

        let mut bytes = kc.ptx.into_bytes();
        bytes.push(0); // null-terminate for C string compatibility
        Ok(bytes)
    }

    fn emit(&mut self, s: &str) {
        self.ptx.push_str(s);
    }

    fn alloc_u64(&mut self) -> String {
        let r = format!("%rd{}", self.next_u64);
        self.next_u64 += 1;
        r
    }

    fn alloc_f32(&mut self) -> String {
        let r = format!("%fs{}", self.next_f32);
        self.next_f32 += 1;
        r
    }

    fn alloc_u32(&mut self) -> String {
        let r = format!("%r{}", self.next_u32);
        self.next_u32 += 1;
        r
    }

    fn alloc_pred(&mut self) -> String {
        let r = format!("%p{}", self.next_pred);
        self.next_pred += 1;
        r
    }

    fn gen_reg_decls(&self) -> String {
        let mut s = String::new();
        if self.next_u32 > 0 {
            s.push_str(&format!("    .reg .u32 %r<{}>;\n", self.next_u32));
        }
        if self.next_u64 > 0 {
            s.push_str(&format!("    .reg .u64 %rd<{}>;\n", self.next_u64));
        }
        if self.next_f32 > 0 {
            s.push_str(&format!("    .reg .f32 %fs<{}>;\n", self.next_f32));
        }
        if self.next_pred > 0 {
            s.push_str(&format!("    .reg .pred %p<{}>;\n", self.next_pred));
        }
        s
    }

    fn compile_block(&mut self, block: &Block, interner: &Interner) -> Result<(), CodegenError> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt, interner)?;
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt, interner: &Interner) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                use nsl_ast::pattern::PatternKind;
                let PatternKind::Ident(sym) = &pattern.kind else {
                    return Err(self.refuse(
                        "destructuring `let` pattern",
                        "kernel locals must bind a single name, e.g. `let i = thread_id()`",
                    ));
                };
                let var_name = interner.resolve(sym.0).unwrap_or("_v").to_string();
                let Some(val_expr) = value else {
                    return Err(self.refuse(
                        "`let` declaration without an initializer",
                        "kernel locals must be initialized at declaration, e.g. `let i = thread_id()`",
                    ));
                };
                let (reg, kind) = self.compile_expr(val_expr, interner)?;
                self.var_regs.insert(var_name, (reg, kind));
                Ok(())
            }
            StmtKind::Expr(expr) => {
                self.compile_expr(expr, interner)?;
                Ok(())
            }
            StmtKind::Assign { target, op, value } => {
                self.compile_assign(target, *op, value, interner)
            }
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => self.compile_if(
                condition,
                then_block,
                elif_clauses,
                else_block.as_ref(),
                interner,
            ),
            other => Err(self.refuse(stmt_kind_name(other), KERNEL_SUPPORTED_HINT)),
        }
    }

    /// Compile an expression; returns (register_name, RegKind).
    fn compile_expr(
        &mut self,
        expr: &Expr,
        interner: &Interner,
    ) -> Result<(String, RegKind), CodegenError> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let rd = self.alloc_u64();
                self.emit(&format!("    mov.u64 {}, {};\n", rd, n));
                Ok((rd, RegKind::U64))
            }
            ExprKind::FloatLiteral(f) => {
                let fs = self.alloc_f32();
                let f32_val = *f as f32;
                let bits = f32_val.to_bits();
                self.emit(&format!("    mov.b32 {}, 0x{:08X};\n", fs, bits));
                Ok((fs, RegKind::F32))
            }
            ExprKind::Ident(sym) => {
                let name = interner.resolve(sym.0).unwrap_or("_").to_string();
                if let Some((reg, kind)) = self.var_regs.get(&name) {
                    Ok((reg.clone(), *kind))
                } else {
                    Err(CodegenError::new(format!(
                        "kernel '{}': unknown identifier '{}' in kernel body. Kernel code \
                         can only reference kernel parameters and locals declared with \
                         `let` inside the kernel; module-level constants are not visible \
                         here - pass the value as a kernel parameter or substitute it via \
                         an @autotune constant. {}",
                        self.kernel_name, name, KERNEL_SUPPORTED_HINT
                    )))
                }
            }
            ExprKind::Call { callee, args: _ } => {
                let ExprKind::Ident(sym) = &callee.kind else {
                    return Err(self.refuse(
                        "an indirect or method call",
                        "only direct calls to the kernel builtins thread_id(), \
                         thread_id_y(), block_id(), block_dim(), sync_threads() are supported",
                    ));
                };
                let name = interner.resolve(sym.0).unwrap_or("");
                match name {
                    "thread_id" => {
                        // Global thread index: blockIdx.x * blockDim.x + threadIdx.x
                        let r_bix = self.alloc_u32();
                        let r_bdx = self.alloc_u32();
                        let r_tid = self.alloc_u32();
                        let r_gid = self.alloc_u32();
                        self.emit(&format!("    mov.u32 {}, %ctaid.x;\n", r_bix));
                        self.emit(&format!("    mov.u32 {}, %ntid.x;\n", r_bdx));
                        self.emit(&format!(
                            "    mul.lo.u32 {}, {}, {};\n",
                            r_gid, r_bix, r_bdx
                        ));
                        self.emit(&format!("    mov.u32 {}, %tid.x;\n", r_tid));
                        self.emit(&format!("    add.u32 {}, {}, {};\n", r_gid, r_gid, r_tid));
                        let rd = self.alloc_u64();
                        self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r_gid));
                        Ok((rd, RegKind::U64))
                    }
                    "thread_id_y" => {
                        let r_biy = self.alloc_u32();
                        let r_bdy = self.alloc_u32();
                        let r_tidy = self.alloc_u32();
                        let r_gidy = self.alloc_u32();
                        self.emit(&format!("    mov.u32 {}, %ctaid.y;\n", r_biy));
                        self.emit(&format!("    mov.u32 {}, %ntid.y;\n", r_bdy));
                        self.emit(&format!(
                            "    mul.lo.u32 {}, {}, {};\n",
                            r_gidy, r_biy, r_bdy
                        ));
                        self.emit(&format!("    mov.u32 {}, %tid.y;\n", r_tidy));
                        self.emit(&format!(
                            "    add.u32 {}, {}, {};\n",
                            r_gidy, r_gidy, r_tidy
                        ));
                        let rd = self.alloc_u64();
                        self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r_gidy));
                        Ok((rd, RegKind::U64))
                    }
                    "block_id" => {
                        let r = self.alloc_u32();
                        self.emit(&format!("    mov.u32 {}, %ctaid.x;\n", r));
                        let rd = self.alloc_u64();
                        self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r));
                        Ok((rd, RegKind::U64))
                    }
                    "block_dim" => {
                        let r = self.alloc_u32();
                        self.emit(&format!("    mov.u32 {}, %ntid.x;\n", r));
                        let rd = self.alloc_u64();
                        self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r));
                        Ok((rd, RegKind::U64))
                    }
                    "sync_threads" => {
                        self.emit("    bar.sync 0;\n");
                        let rd = self.alloc_u64();
                        self.emit(&format!("    mov.u64 {}, 0;\n", rd));
                        Ok((rd, RegKind::U64))
                    }
                    _ => Err(CodegenError::new(format!(
                        "kernel '{}': call to unsupported function '{}' in kernel body. \
                         Supported kernel builtins: thread_id(), thread_id_y(), \
                         block_id(), block_dim(), sync_threads(). Math intrinsics \
                         (sqrt, exp, ...) are not implemented in the kernel PTX compiler.",
                        self.kernel_name, name
                    ))),
                }
            }
            ExprKind::Subscript { object, index } => {
                // a[idx] → compute address, load f32
                let ExprKind::Ident(sym) = &object.kind else {
                    return Err(self.refuse(
                        "subscript of a non-identifier expression",
                        "only direct indexing of a kernel parameter or local, \
                         e.g. `a[i]`, is supported",
                    ));
                };
                let obj_name = interner.resolve(sym.0).unwrap_or("_").to_string();
                let Some((base_reg, _)) = self.var_regs.get(&obj_name).cloned() else {
                    return Err(CodegenError::new(format!(
                        "kernel '{}': subscript of unknown variable '{}' - kernel code \
                         can only index kernel parameters and locals declared with `let`",
                        self.kernel_name, obj_name
                    )));
                };
                let SubscriptKind::Index(idx_expr) = index.as_ref() else {
                    return Err(self.refuse(
                        "slice / multi-dimensional subscript",
                        "only single-element indexing, e.g. `a[i]`, is supported",
                    ));
                };
                let (idx_reg, _) = self.compile_expr(idx_expr, interner)?;
                let offset = self.alloc_u64();
                self.emit(&format!("    shl.b64 {}, {}, 2;\n", offset, idx_reg));
                let addr = self.alloc_u64();
                self.emit(&format!(
                    "    add.u64 {}, {}, {};\n",
                    addr, base_reg, offset
                ));
                let fs = self.alloc_f32();
                self.emit(&format!("    ld.global.f32 {}, [{}];\n", fs, addr));
                Ok((fs, RegKind::F32))
            }
            ExprKind::BinaryOp { left, op, right } => {
                let (l_reg, l_kind) = self.compile_expr(left, interner)?;
                let (r_reg, r_kind) = self.compile_expr(right, interner)?;

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if l_kind == RegKind::F32 || r_kind == RegKind::F32 {
                            let l_f32 = if l_kind == RegKind::F32 {
                                l_reg.clone()
                            } else {
                                let fs = self.alloc_f32();
                                self.emit(&format!("    cvt.rn.f32.u64 {}, {};\n", fs, l_reg));
                                fs
                            };
                            let r_f32 = if r_kind == RegKind::F32 {
                                r_reg.clone()
                            } else {
                                let fs = self.alloc_f32();
                                self.emit(&format!("    cvt.rn.f32.u64 {}, {};\n", fs, r_reg));
                                fs
                            };
                            let result = self.alloc_f32();
                            let op_str = match op {
                                BinOp::Add => "add.f32",
                                BinOp::Sub => "sub.f32",
                                BinOp::Mul => "mul.f32",
                                BinOp::Div => "div.approx.f32",
                                _ => unreachable!(),
                            };
                            self.emit(&format!(
                                "    {} {}, {}, {};\n",
                                op_str, result, l_f32, r_f32
                            ));
                            Ok((result, RegKind::F32))
                        } else {
                            let result = self.alloc_u64();
                            let op_str = match op {
                                BinOp::Add => "add.u64",
                                BinOp::Sub => "sub.u64",
                                BinOp::Mul => "mul.lo.u64",
                                BinOp::Div => "div.s64",
                                _ => unreachable!(),
                            };
                            self.emit(&format!(
                                "    {} {}, {}, {};\n",
                                op_str, result, l_reg, r_reg
                            ));
                            Ok((result, RegKind::U64))
                        }
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                        let pred = self.alloc_pred();
                        let cmp_str = match op {
                            BinOp::Lt => "lt",
                            BinOp::Gt => "gt",
                            BinOp::LtEq => "le",
                            BinOp::GtEq => "ge",
                            _ => unreachable!(),
                        };
                        if l_kind == RegKind::F32 || r_kind == RegKind::F32 {
                            let l_f32 = if l_kind == RegKind::F32 {
                                l_reg.clone()
                            } else {
                                let fs = self.alloc_f32();
                                self.emit(&format!("    cvt.rn.f32.u64 {}, {};\n", fs, l_reg));
                                fs
                            };
                            let r_f32 = if r_kind == RegKind::F32 {
                                r_reg.clone()
                            } else {
                                let fs = self.alloc_f32();
                                self.emit(&format!("    cvt.rn.f32.u64 {}, {};\n", fs, r_reg));
                                fs
                            };
                            self.emit(&format!(
                                "    setp.{}.f32 {}, {}, {};\n",
                                cmp_str, pred, l_f32, r_f32
                            ));
                        } else {
                            self.emit(&format!(
                                "    setp.{}.u64 {}, {}, {};\n",
                                cmp_str, pred, l_reg, r_reg
                            ));
                        }
                        // Return the pred register name with U64 kind (caller checks %p prefix)
                        Ok((pred, RegKind::U64))
                    }
                    other => Err(CodegenError::new(format!(
                        "kernel '{}': binary operator '{}' is not supported in kernel \
                         code. Supported arithmetic: + - * /; supported comparisons: \
                         < <= > >=.",
                        self.kernel_name,
                        binop_symbol(*other)
                    ))),
                }
            }
            // `(a + b) * 0.5` — parentheses are pure grouping; compile the
            // inner expression directly.
            ExprKind::Paren(inner) => self.compile_expr(inner, interner),
            other => Err(self.refuse(
                &format!("{} expression", expr_kind_name(other)),
                KERNEL_SUPPORTED_HINT,
            )),
        }
    }

    /// Compile `target[idx] = value` (store to global memory).
    fn compile_assign(
        &mut self,
        target: &Expr,
        op: AssignOp,
        value: &Expr,
        interner: &Interner,
    ) -> Result<(), CodegenError> {
        if op != AssignOp::Assign {
            return Err(self.refuse(
                "compound assignment (+=, -=, *=, /=)",
                "expand it to a plain store, e.g. `out[i] = out[i] + v`",
            ));
        }
        let ExprKind::Subscript { object, index } = &target.kind else {
            return Err(self.refuse(
                "assignment to a plain variable",
                "kernel stores must write through a subscript, e.g. `out[i] = v`; \
                 to introduce a new value use a fresh `let` binding",
            ));
        };
        let ExprKind::Ident(sym) = &object.kind else {
            return Err(self.refuse(
                "a store through a non-identifier subscript base",
                "only direct stores into a kernel parameter or local, \
                 e.g. `out[i] = v`, are supported",
            ));
        };
        let obj_name = interner.resolve(sym.0).unwrap_or("_").to_string();
        let Some((base_reg, _)) = self.var_regs.get(&obj_name).cloned() else {
            return Err(CodegenError::new(format!(
                "kernel '{}': store into unknown variable '{}' - kernel code can \
                 only store into kernel parameters (or locals derived from them)",
                self.kernel_name, obj_name
            )));
        };
        let SubscriptKind::Index(idx_expr) = index.as_ref() else {
            return Err(self.refuse(
                "slice / multi-dimensional store target",
                "only single-element stores, e.g. `out[i] = v`, are supported",
            ));
        };
        let (idx_reg, _) = self.compile_expr(idx_expr, interner)?;
        let (val_reg, val_kind) = self.compile_expr(value, interner)?;
        let val_f32 = if val_kind == RegKind::F32 {
            val_reg
        } else {
            let fs = self.alloc_f32();
            self.emit(&format!("    cvt.rn.f32.u64 {}, {};\n", fs, val_reg));
            fs
        };
        let offset = self.alloc_u64();
        self.emit(&format!("    shl.b64 {}, {}, 2;\n", offset, idx_reg));
        let addr = self.alloc_u64();
        self.emit(&format!(
            "    add.u64 {}, {}, {};\n",
            addr, base_reg, offset
        ));
        self.emit(&format!("    st.global.f32 [{}], {};\n", addr, val_f32));
        Ok(())
    }

    /// Compile an if/elif/else statement with predicate branching.
    fn compile_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        elif_clauses: &[(Expr, Block)],
        else_block: Option<&Block>,
        interner: &Interner,
    ) -> Result<(), CodegenError> {
        let (pred_reg, _) = self.compile_expr(cond, interner)?;

        // The condition must have compiled to a predicate register (comparison
        // result). Anything else used to be silently dropped — the whole
        // if-body vanished from the PTX. Refuse instead.
        if !pred_reg.starts_with("%p") {
            return Err(self.refuse(
                "an `if` condition that is not a comparison",
                "kernel `if` conditions must be comparisons (< <= > >=), \
                 e.g. `if i < n:`",
            ));
        }

        let label_id = self.next_label;
        self.next_label += 1;
        let end_label = format!("ENDIF_{}", label_id);

        // Determine where to branch if the initial condition is false
        let first_false_label = if !elif_clauses.is_empty() {
            format!("ELIF_{}_{}", label_id, 0)
        } else if else_block.is_some() {
            format!("ELSE_{}", label_id)
        } else {
            end_label.clone()
        };

        self.emit(&format!("    @!{} bra {};\n", pred_reg, first_false_label));
        self.compile_block(then_block, interner)?;
        self.emit(&format!("    bra {};\n", end_label));

        // Emit elif clauses
        for (i, (elif_cond, elif_block)) in elif_clauses.iter().enumerate() {
            self.emit(&format!("ELIF_{}_{}:\n", label_id, i));

            let (elif_pred, _) = self.compile_expr(elif_cond, interner)?;
            if !elif_pred.starts_with("%p") {
                // Previously the elif body was compiled UNCONDITIONALLY here
                // (no branch) — silently wrong control flow. Refuse instead.
                return Err(self.refuse(
                    "an `elif` condition that is not a comparison",
                    "kernel `elif` conditions must be comparisons (< <= > >=)",
                ));
            }
            // Determine where to branch if this elif condition is false
            let next_false_label = if i + 1 < elif_clauses.len() {
                format!("ELIF_{}_{}", label_id, i + 1)
            } else if else_block.is_some() {
                format!("ELSE_{}", label_id)
            } else {
                end_label.clone()
            };

            self.emit(&format!("    @!{} bra {};\n", elif_pred, next_false_label));
            self.compile_block(elif_block, interner)?;
            self.emit(&format!("    bra {};\n", end_label));
        }

        // Emit else block
        if let Some(else_blk) = else_block {
            self.emit(&format!("ELSE_{}:\n", label_id));
            self.compile_block(else_blk, interner)?;
        }

        self.emit(&format!("{}:\n", end_label));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::stmt::StmtKind;

    /// Parse NSL source and return the first `kernel` definition found
    /// (bare or decorated), together with the interner.
    fn parse_first_kernel(src: &str) -> (KernelDef, Interner) {
        let mut interner = Interner::new();
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

    /// Compile the first kernel in `src`, expecting a refusal; returns the
    /// error message.
    fn compile_err(src: &str) -> String {
        let (kernel, interner) = parse_first_kernel(src);
        match KernelCompiler::compile(&kernel, &interner) {
            Ok(_) => panic!("expected refusal, but kernel compiled successfully"),
            Err(e) => e.to_string(),
        }
    }

    /// Compile the first kernel in `src`, expecting success; returns PTX text.
    fn compile_ok(src: &str) -> String {
        let (kernel, interner) = parse_first_kernel(src);
        let bytes = KernelCompiler::compile(&kernel, &interner)
            .expect("valid kernel must compile");
        assert_eq!(bytes.last(), Some(&0), "PTX must be null-terminated");
        String::from_utf8_lossy(&bytes[..bytes.len() - 1]).to_string()
    }

    // ── valid kernels still compile (emission unchanged) ───────────────

    #[test]
    fn test_valid_vec_add_compiles() {
        // Mirrors tests/m17_kernel_test.nsl
        let ptx = compile_ok(
            "kernel vec_add(a, b, c):\n    let i = thread_id()\n    c[i] = a[i] + b[i]\n",
        );
        assert!(ptx.contains(".visible .entry vec_add("), "PTX:\n{ptx}");
        assert!(ptx.contains("ld.global.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("add.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_valid_if_comparison_compiles() {
        let ptx = compile_ok(
            "kernel guarded(a, out):\n    let i = thread_id()\n    if i < 4:\n        out[i] = a[i]\n",
        );
        assert!(ptx.contains("setp.lt.u64"), "PTX:\n{ptx}");
        assert!(ptx.contains("ENDIF_0:"), "PTX:\n{ptx}");
    }

    #[test]
    fn test_autotune_constant_substitution_still_compiles() {
        // A module-level constant substituted by @autotune must compile...
        let src = "kernel scaled(a, out):\n    let i = thread_id()\n    out[i] = a[i] * BLOCK\n";
        let (kernel, interner) = parse_first_kernel(src);
        let mut constants = HashMap::new();
        constants.insert("BLOCK".to_string(), 4i64);
        let bytes = KernelCompiler::compile_with_constants(&kernel, &interner, &constants)
            .expect("constant-substituted kernel must compile");
        assert!(!bytes.is_empty());

        // ...and the same kernel WITHOUT the substitution must refuse loudly
        // (this is exactly the unknown-module-const fabrication the old code
        // papered over with a dummy register).
        let err = KernelCompiler::compile(&kernel, &interner)
            .expect_err("unknown ident must refuse")
            .to_string();
        assert!(err.contains("BLOCK"), "err: {err}");
        assert!(err.contains("@autotune"), "err: {err}");
    }

    #[test]
    fn test_paren_expression_compiles_as_grouping() {
        // `(a[i] + b[i]) * 0.5` — parentheses are pure grouping and must
        // compile via the inner expression (the old code fabricated a garbage
        // register for Paren; the first refusal cut then refused it, which
        // contradicted KERNEL_SUPPORTED_HINT's claim that arithmetic works).
        let ptx = compile_ok(
            "kernel avg(a, b, out):\n    let i = thread_id()\n    out[i] = (a[i] + b[i]) * 0.5\n",
        );
        assert!(ptx.contains("add.f32"), "PTX:\n{ptx}");
        assert!(ptx.contains("mul.f32"), "PTX:\n{ptx}");
    }

    // ── refusals: statements ────────────────────────────────────────────

    #[test]
    fn test_refuses_for_loop() {
        let err = compile_err(
            "kernel bad_loop(a, out):\n    let i = thread_id()\n    for j in range(0, 4):\n        out[i] = a[i]\n",
        );
        assert!(err.contains("bad_loop"), "err: {err}");
        assert!(err.contains("for loop"), "err: {err}");
    }

    #[test]
    fn test_refuses_while_loop() {
        let err = compile_err(
            "kernel bad_while(a, out):\n    let i = thread_id()\n    while i < 4:\n        out[i] = a[i]\n",
        );
        assert!(err.contains("while loop"), "err: {err}");
    }

    #[test]
    fn test_refuses_return_stmt() {
        let err = compile_err(
            "kernel bad_ret(a, out):\n    let i = thread_id()\n    return\n",
        );
        assert!(err.contains("return statement"), "err: {err}");
    }

    // ── refusals: expressions ───────────────────────────────────────────

    #[test]
    fn test_refuses_unknown_identifier() {
        let err = compile_err(
            "kernel bad_ident(a, out):\n    let i = thread_id()\n    out[i] = a[i] * SOME_CONST\n",
        );
        assert!(err.contains("SOME_CONST"), "err: {err}");
        assert!(err.contains("@autotune"), "err: {err}");
    }

    #[test]
    fn test_refuses_unknown_call() {
        let err = compile_err(
            "kernel bad_call(a, out):\n    let i = thread_id()\n    out[i] = sqrt(a[i])\n",
        );
        assert!(err.contains("sqrt"), "err: {err}");
        assert!(err.contains("thread_id"), "err: {err}");
    }

    #[test]
    fn test_refuses_eq_binop() {
        let err = compile_err(
            "kernel bad_eq(a, out):\n    let i = thread_id()\n    if i == 0:\n        out[i] = a[i]\n",
        );
        assert!(err.contains("'=='"), "err: {err}");
    }

    #[test]
    fn test_refuses_mod_binop() {
        let err = compile_err(
            "kernel bad_mod(a, out):\n    let i = thread_id()\n    out[i] = a[i % 2]\n",
        );
        assert!(err.contains("'%'"), "err: {err}");
    }

    #[test]
    fn test_refuses_pow_binop() {
        let err = compile_err(
            "kernel bad_pow(a, out):\n    let i = thread_id()\n    out[i] = a[i] ** 2.0\n",
        );
        assert!(err.contains("'**'"), "err: {err}");
    }

    #[test]
    fn test_refuses_non_comparison_if_condition() {
        let err = compile_err(
            "kernel bad_cond(flag, out):\n    let i = thread_id()\n    if flag:\n        out[i] = 1.0\n",
        );
        assert!(err.contains("comparison"), "err: {err}");
    }

    // ── refusals: assignment forms ──────────────────────────────────────

    #[test]
    fn test_refuses_plain_variable_reassignment() {
        let err = compile_err(
            "kernel bad_reassign(a, out):\n    let x = 1.0\n    x = 2.0\n",
        );
        assert!(err.contains("plain variable"), "err: {err}");
    }

    #[test]
    fn test_refuses_compound_assignment() {
        let err = compile_err(
            "kernel bad_compound(a, out):\n    let i = thread_id()\n    out[i] += a[i]\n",
        );
        assert!(err.contains("compound assignment"), "err: {err}");
    }

    #[test]
    fn test_refuses_store_to_unknown_base() {
        let err = compile_err(
            "kernel bad_base(a, out):\n    let i = thread_id()\n    bogus[i] = a[i]\n",
        );
        assert!(err.contains("bogus"), "err: {err}");
    }
}
