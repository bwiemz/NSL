//! KernelCompiler: translates KernelDef AST nodes into PTX strings.

use std::collections::HashMap;

use nsl_ast::block::KernelDef;
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::operator::BinOp;
use nsl_lexer::Interner;

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
    /// Variable → (register_name, RegKind)
    var_regs: HashMap<String, (String, RegKind)>,
}

impl KernelCompiler {
    pub fn new() -> Self {
        KernelCompiler {
            ptx: String::new(),
            next_u32: 0,
            next_u64: 0,
            next_f32: 0,
            next_pred: 0,
            var_regs: HashMap::new(),
        }
    }

    /// Compile a KernelDef into a null-terminated PTX byte string.
    pub fn compile(kernel: &KernelDef, interner: &Interner) -> Vec<u8> {
        let mut kc = KernelCompiler::new();
        let name = interner.resolve(kernel.name.0).unwrap_or("__kernel");

        // PTX header
        kc.emit(".version 7.0\n");
        kc.emit(".target sm_52\n");
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
        kc.compile_block(&kernel.body, interner);

        kc.emit("DONE:\n    ret;\n}\n");

        // Replace register placeholder with actual declarations
        let reg_decls = kc.gen_reg_decls();
        kc.ptx = kc.ptx.replacen("    // REGISTERS\n", &reg_decls, 1);

        let mut bytes = kc.ptx.into_bytes();
        bytes.push(0); // null-terminate for C string compatibility
        bytes
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

    fn compile_block(&mut self, block: &Block, interner: &Interner) {
        for stmt in &block.stmts {
            self.compile_stmt(stmt, interner);
        }
    }

    fn compile_stmt(&mut self, stmt: &Stmt, interner: &Interner) {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                use nsl_ast::pattern::PatternKind;
                if let PatternKind::Ident(sym) = &pattern.kind {
                    let var_name = interner.resolve(sym.0).unwrap_or("_v").to_string();
                    if let Some(val_expr) = value {
                        let (reg, kind) = self.compile_expr(val_expr, interner);
                        self.var_regs.insert(var_name, (reg, kind));
                    }
                }
            }
            StmtKind::Expr(expr) => {
                self.compile_expr(expr, interner);
            }
            StmtKind::Assign { target, value, .. } => {
                self.compile_assign(target, value, interner);
            }
            StmtKind::If { condition, then_block, else_block, .. } => {
                self.compile_if(condition, then_block, else_block.as_ref(), interner);
            }
            _ => {} // Skip unsupported statements silently
        }
    }

    /// Compile an expression; returns (register_name, RegKind).
    fn compile_expr(&mut self, expr: &Expr, interner: &Interner) -> (String, RegKind) {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let rd = self.alloc_u64();
                self.emit(&format!("    mov.u64 {}, {};\n", rd, n));
                (rd, RegKind::U64)
            }
            ExprKind::FloatLiteral(f) => {
                let fs = self.alloc_f32();
                let f32_val = *f as f32;
                let bits = f32_val.to_bits();
                self.emit(&format!("    mov.b32 {}, 0x{:08X};\n", fs, bits));
                (fs, RegKind::F32)
            }
            ExprKind::Ident(sym) => {
                let name = interner.resolve(sym.0).unwrap_or("_").to_string();
                if let Some((reg, kind)) = self.var_regs.get(&name) {
                    (reg.clone(), *kind)
                } else {
                    // Unknown variable — return a fresh dummy register
                    let rd = self.alloc_u64();
                    (rd, RegKind::U64)
                }
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(sym) = &callee.kind {
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
                            self.emit(&format!("    mul.lo.u32 {}, {}, {};\n", r_gid, r_bix, r_bdx));
                            self.emit(&format!("    mov.u32 {}, %tid.x;\n", r_tid));
                            self.emit(&format!("    add.u32 {}, {}, {};\n", r_gid, r_gid, r_tid));
                            let rd = self.alloc_u64();
                            self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r_gid));
                            (rd, RegKind::U64)
                        }
                        "thread_id_y" => {
                            let r_biy = self.alloc_u32();
                            let r_bdy = self.alloc_u32();
                            let r_tidy = self.alloc_u32();
                            let r_gidy = self.alloc_u32();
                            self.emit(&format!("    mov.u32 {}, %ctaid.y;\n", r_biy));
                            self.emit(&format!("    mov.u32 {}, %ntid.y;\n", r_bdy));
                            self.emit(&format!("    mul.lo.u32 {}, {}, {};\n", r_gidy, r_biy, r_bdy));
                            self.emit(&format!("    mov.u32 {}, %tid.y;\n", r_tidy));
                            self.emit(&format!("    add.u32 {}, {}, {};\n", r_gidy, r_gidy, r_tidy));
                            let rd = self.alloc_u64();
                            self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r_gidy));
                            (rd, RegKind::U64)
                        }
                        "block_id" => {
                            let r = self.alloc_u32();
                            self.emit(&format!("    mov.u32 {}, %ctaid.x;\n", r));
                            let rd = self.alloc_u64();
                            self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r));
                            (rd, RegKind::U64)
                        }
                        "block_dim" => {
                            let r = self.alloc_u32();
                            self.emit(&format!("    mov.u32 {}, %ntid.x;\n", r));
                            let rd = self.alloc_u64();
                            self.emit(&format!("    cvt.u64.u32 {}, {};\n", rd, r));
                            (rd, RegKind::U64)
                        }
                        "sync_threads" => {
                            self.emit("    bar.sync 0;\n");
                            let rd = self.alloc_u64();
                            (rd, RegKind::U64)
                        }
                        _ => {
                            // Compile args for side effects, return dummy
                            for arg in args {
                                self.compile_expr(&arg.value, interner);
                            }
                            let rd = self.alloc_u64();
                            (rd, RegKind::U64)
                        }
                    }
                } else {
                    for arg in args {
                        self.compile_expr(&arg.value, interner);
                    }
                    let rd = self.alloc_u64();
                    (rd, RegKind::U64)
                }
            }
            ExprKind::Subscript { object, index } => {
                // a[idx] → compute address, load f32
                if let ExprKind::Ident(sym) = &object.kind {
                    let obj_name = interner.resolve(sym.0).unwrap_or("_").to_string();
                    if let Some((base_reg, _)) = self.var_regs.get(&obj_name).cloned() {
                        if let SubscriptKind::Index(idx_expr) = index.as_ref() {
                            let (idx_reg, _) = self.compile_expr(idx_expr, interner);
                            let offset = self.alloc_u64();
                            self.emit(&format!("    shl.b64 {}, {}, 2;\n", offset, idx_reg));
                            let addr = self.alloc_u64();
                            self.emit(&format!("    add.u64 {}, {}, {};\n", addr, base_reg, offset));
                            let fs = self.alloc_f32();
                            self.emit(&format!("    ld.global.f32 {}, [{}];\n", fs, addr));
                            return (fs, RegKind::F32);
                        }
                    }
                }
                let fs = self.alloc_f32();
                (fs, RegKind::F32)
            }
            ExprKind::BinaryOp { left, op, right } => {
                let (l_reg, l_kind) = self.compile_expr(left, interner);
                let (r_reg, r_kind) = self.compile_expr(right, interner);

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
                            self.emit(&format!("    {} {}, {}, {};\n", op_str, result, l_f32, r_f32));
                            (result, RegKind::F32)
                        } else {
                            let result = self.alloc_u64();
                            let op_str = match op {
                                BinOp::Add => "add.u64",
                                BinOp::Sub => "sub.u64",
                                BinOp::Mul => "mul.lo.u64",
                                BinOp::Div => "div.s64",
                                _ => unreachable!(),
                            };
                            self.emit(&format!("    {} {}, {}, {};\n", op_str, result, l_reg, r_reg));
                            (result, RegKind::U64)
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
                        (pred, RegKind::U64)
                    }
                    _ => {
                        let rd = self.alloc_u64();
                        (rd, RegKind::U64)
                    }
                }
            }
            _ => {
                let rd = self.alloc_u64();
                (rd, RegKind::U64)
            }
        }
    }

    /// Compile `target[idx] = value` (store to global memory).
    fn compile_assign(&mut self, target: &Expr, value: &Expr, interner: &Interner) {
        if let ExprKind::Subscript { object, index } = &target.kind {
            if let ExprKind::Ident(sym) = &object.kind {
                let obj_name = interner.resolve(sym.0).unwrap_or("_").to_string();
                if let Some((base_reg, _)) = self.var_regs.get(&obj_name).cloned() {
                    if let SubscriptKind::Index(idx_expr) = index.as_ref() {
                        let (idx_reg, _) = self.compile_expr(idx_expr, interner);
                        let (val_reg, val_kind) = self.compile_expr(value, interner);
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
                        self.emit(&format!("    add.u64 {}, {}, {};\n", addr, base_reg, offset));
                        self.emit(&format!("    st.global.f32 [{}], {};\n", addr, val_f32));
                    }
                }
            }
        }
    }

    /// Compile an if statement with predicate branching.
    fn compile_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_block: Option<&Block>,
        interner: &Interner,
    ) {
        let (pred_reg, _) = self.compile_expr(cond, interner);

        // Use pred register directly if it starts with %p (comparison result)
        if pred_reg.starts_with("%p") {
            let label_id = self.next_pred;
            self.next_pred += 1;
            let else_label = format!("ELSE_{}", label_id);
            let end_label = format!("ENDIF_{}", label_id);

            if else_block.is_some() {
                self.emit(&format!("    @!{} bra {};\n", pred_reg, else_label));
            } else {
                self.emit(&format!("    @!{} bra {};\n", pred_reg, end_label));
            }

            self.compile_block(then_block, interner);

            if let Some(else_blk) = else_block {
                self.emit(&format!("    bra {};\n", end_label));
                self.emit(&format!("{}:\n", else_label));
                self.compile_block(else_blk, interner);
            }

            self.emit(&format!("{}:\n", end_label));
        }
        // If pred_reg doesn't start with %p, the condition wasn't a comparison —
        // silently skip (handles unknown/unsupported conditions gracefully).
    }
}
