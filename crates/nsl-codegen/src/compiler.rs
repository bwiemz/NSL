use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectModule, ObjectBuilder};

use nsl_ast::decl::ModelMember;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::TypeExprKind;
use nsl_ast::{NodeId, Symbol};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;
use nsl_semantic::types::Type;

use crate::builtins;
use crate::context::{FuncState, StructField, StructLayout};
use crate::error::CodegenError;
use crate::types::{nsl_type_to_cl, pointer_type};

/// A lambda body to be compiled after the current function.
pub struct PendingLambda {
    pub name: String,
    pub func_id: FuncId,
    pub sig: Signature,
    pub params: Vec<(Symbol, cl_types::Type)>,
    pub body: Expr,
    /// Captured variables from outer scope (name, cranelift type). These become extra params
    /// after the normal params in the lambda's Cranelift function signature.
    pub captures: Vec<(Symbol, cl_types::Type)>,
}

pub struct Compiler<'a> {
    pub module: ObjectModule,
    pub interner: &'a Interner,
    pub type_map: &'a TypeMap,
    pub functions: HashMap<String, (FuncId, Signature)>,
    pub runtime_fns: HashMap<String, (FuncId, Signature)>,
    pub string_pool: HashMap<String, DataId>,
    pub struct_layouts: HashMap<String, StructLayout>,
    /// Enum variant name → integer tag. Flat map: "ReLU" → 0, "Sigmoid" → 1, etc.
    pub enum_variants: HashMap<String, i64>,
    /// Enum name → list of (variant_name, tag)
    pub enum_defs: HashMap<String, Vec<(String, i64)>>,
    /// Model name → { method_name → mangled_fn_name } for method dispatch.
    pub model_methods: HashMap<String, HashMap<String, String>>,
    /// Lambda bodies to compile after the current function is finished.
    pub pending_lambdas: Vec<PendingLambda>,
    /// Maps variable Symbol to its closure capture count. If present, the variable holds a
    /// heap-allocated closure struct `{ fn_ptr, num_captures, captures[] }` instead of a bare fn ptr.
    pub closure_info: HashMap<Symbol, usize>,
    /// Set by compile_lambda when it creates a closure; consumed by VarDecl to register in closure_info.
    pub last_lambda_capture_count: Option<usize>,
    pub call_conv: CallConv,
    pub dump_ir: bool,
    /// Functions decorated with `@no_grad` — tape is paused during their execution.
    pub no_grad_fns: HashSet<String>,
    /// Functions decorated with `@test` — discovered by `nsl test` runner.
    pub test_fns: Vec<String>,
    /// Module prefix for name mangling. Empty means no mangling (entry module / single-file).
    pub module_prefix: String,
    func_index: u32,
}

/// Mangle a function name with a module prefix for unique Cranelift symbols.
/// If prefix is empty, returns the name unchanged.
fn mangle_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}

impl<'a> Compiler<'a> {
    pub fn new(interner: &'a Interner, type_map: &'a TypeMap) -> Result<Self, CodegenError> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| CodegenError::new(e.to_string()))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| CodegenError::new(format!("failed to create ISA builder: {e}")))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| CodegenError::new(format!("failed to build ISA: {e}")))?;

        let call_conv = isa.default_call_conv();

        let builder = ObjectBuilder::new(
            isa,
            "nsl_module",
            cranelift_module::default_libcall_names(),
        )
        .map_err(|e| CodegenError::new(format!("failed to create object builder: {e}")))?;

        let module = ObjectModule::new(builder);

        Ok(Compiler {
            module,
            interner,
            type_map,
            functions: HashMap::new(),
            runtime_fns: HashMap::new(),
            string_pool: HashMap::new(),
            struct_layouts: HashMap::new(),
            model_methods: HashMap::new(),
            enum_variants: HashMap::new(),
            enum_defs: HashMap::new(),
            pending_lambdas: Vec::new(),
            closure_info: HashMap::new(),
            last_lambda_capture_count: None,
            call_conv,
            dump_ir: false,
            no_grad_fns: HashSet::new(),
            test_fns: Vec::new(),
            module_prefix: String::new(),
            func_index: 0,
        })
    }

    /// Resolve a Symbol to its interned string.
    pub fn resolve_sym(&self, sym: Symbol) -> &str {
        self.interner.resolve(sym.0).unwrap_or("<unknown>")
    }

    /// Get the inferred type of an AST node by NodeId.
    pub fn node_type(&self, id: NodeId) -> &Type {
        self.type_map.get(&id).unwrap_or(&Type::Unknown)
    }

    /// Allocate a unique function index for Cranelift.
    pub fn next_func_index(&mut self) -> u32 {
        let idx = self.func_index;
        self.func_index += 1;
        idx
    }

    /// Resolve a type name (from AST annotation) to a Cranelift type.
    fn resolve_type_name_to_cl(&self, sym: Symbol) -> cl_types::Type {
        let name = self.resolve_sym(sym);
        match name {
            "int" => cl_types::I64,
            "float" => cl_types::F64,
            "f32" => cl_types::F32,
            "f64" => cl_types::F64,
            "bool" => cl_types::I8,
            "str" => cl_types::I64,
            "i8" | "int8" => cl_types::I8,
            "i16" | "int16" => cl_types::I16,
            "i32" | "int32" => cl_types::I32,
            "i64" | "int64" => cl_types::I64,
            "bf16" | "fp16" => cl_types::F32,
            _ => cl_types::I64,
        }
    }

    // ── Pass 0: Collect string literals ─────────────────────────────

    pub fn collect_strings(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            self.collect_strings_in_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_strings_in_stmt(&mut self, stmt: &Stmt) -> Result<(), CodegenError> {
        match &stmt.kind {
            StmtKind::VarDecl { value: Some(expr), .. } => self.collect_strings_in_expr(expr),
            StmtKind::FnDef(fn_def) => {
                for s in &fn_def.body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.collect_strings_in_expr(condition)?;
                for s in &then_block.stmts { self.collect_strings_in_stmt(s)?; }
                for (cond, block) in elif_clauses {
                    self.collect_strings_in_expr(cond)?;
                    for s in &block.stmts { self.collect_strings_in_stmt(s)?; }
                }
                if let Some(block) = else_block {
                    for s in &block.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            StmtKind::For { iterable, body, .. } => {
                self.collect_strings_in_expr(iterable)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::While { condition, body } => {
                self.collect_strings_in_expr(condition)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::WhileLet { expr, body, .. } => {
                self.collect_strings_in_expr(expr)?;
                for s in &body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            StmtKind::Match { subject, arms } => {
                self.collect_strings_in_expr(subject)?;
                for arm in arms {
                    if let nsl_ast::pattern::PatternKind::Literal(lit) = &arm.pattern.kind {
                        self.collect_strings_in_expr(lit)?;
                    }
                    for s in &arm.body.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            StmtKind::ModelDef(md) => {
                for member in &md.members {
                    match member {
                        ModelMember::LayerDecl { init: Some(expr), .. } => {
                            self.collect_strings_in_expr(expr)?;
                        }
                        ModelMember::Method(fn_def) => {
                            for s in &fn_def.body.stmts { self.collect_strings_in_stmt(s)?; }
                        }
                        _ => {}
                    }
                }
                Ok(())
            }
            StmtKind::Decorated { stmt, .. } => self.collect_strings_in_stmt(stmt),
            StmtKind::Return(Some(expr)) => self.collect_strings_in_expr(expr),
            StmtKind::Assign { value, .. } => self.collect_strings_in_expr(value),
            StmtKind::Expr(expr) => self.collect_strings_in_expr(expr),
            StmtKind::GradBlock(grad) => {
                self.collect_strings_in_expr(&grad.targets)?;
                for s in &grad.body.stmts { self.collect_strings_in_stmt(s)?; }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn collect_strings_in_expr(&mut self, expr: &Expr) -> Result<(), CodegenError> {
        match &expr.kind {
            ExprKind::StringLiteral(s) => { self.intern_string(s)?; Ok(()) }
            ExprKind::FString(parts) => {
                for part in parts {
                    match part {
                        nsl_ast::expr::FStringPart::Text(s) => { self.intern_string(s)?; }
                        nsl_ast::expr::FStringPart::Expr(e) => { self.collect_strings_in_expr(e)?; }
                    }
                }
                Ok(())
            }
            ExprKind::BinaryOp { left, right, .. } | ExprKind::Pipe { left, right } => {
                self.collect_strings_in_expr(left)?;
                self.collect_strings_in_expr(right)
            }
            ExprKind::UnaryOp { operand, .. } => self.collect_strings_in_expr(operand),
            ExprKind::Call { callee, args } => {
                self.collect_strings_in_expr(callee)?;
                for arg in args { self.collect_strings_in_expr(&arg.value)?; }
                Ok(())
            }
            ExprKind::ListLiteral(elems) => {
                for e in elems { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            ExprKind::MemberAccess { object, .. } | ExprKind::Paren(object) => {
                self.collect_strings_in_expr(object)
            }
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.collect_strings_in_expr(condition)?;
                self.collect_strings_in_expr(then_expr)?;
                self.collect_strings_in_expr(else_expr)
            }
            ExprKind::Subscript { object, index } => {
                self.collect_strings_in_expr(object)?;
                if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                    self.collect_strings_in_expr(idx_expr)?;
                }
                Ok(())
            }
            ExprKind::ListComp { element, generators } => {
                self.collect_strings_in_expr(element)?;
                for gen in generators {
                    self.collect_strings_in_expr(&gen.iterable)?;
                    for cond in &gen.conditions {
                        self.collect_strings_in_expr(cond)?;
                    }
                }
                Ok(())
            }
            ExprKind::DictLiteral(pairs) => {
                for (k, v) in pairs {
                    match &k.kind {
                        ExprKind::Ident(sym) => {
                            let name = self.resolve_sym(*sym).to_string();
                            self.intern_string(&name)?;
                        }
                        _ => { self.collect_strings_in_expr(k)?; }
                    }
                    self.collect_strings_in_expr(v)?;
                }
                Ok(())
            }
            ExprKind::TupleLiteral(elems) => {
                for e in elems { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            ExprKind::Lambda { body, .. } => {
                self.collect_strings_in_expr(body)
            }
            ExprKind::MatchExpr { subject, arms } => {
                self.collect_strings_in_expr(subject)?;
                for arm in arms {
                    // Collect strings in pattern literals
                    if let nsl_ast::pattern::PatternKind::Literal(lit) = &arm.pattern.kind {
                        self.collect_strings_in_expr(lit)?;
                    }
                    for s in &arm.body.stmts { self.collect_strings_in_stmt(s)?; }
                }
                Ok(())
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.collect_strings_in_expr(s)?; }
                if let Some(e) = end { self.collect_strings_in_expr(e)?; }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Intern a string literal into the module's data section.
    pub fn intern_string(&mut self, s: &str) -> Result<DataId, CodegenError> {
        if let Some(&data_id) = self.string_pool.get(s) {
            return Ok(data_id);
        }
        let name = format!(".str.{}", self.string_pool.len());
        let data_id = self.module
            .declare_data(&name, Linkage::Local, false, false)
            .map_err(|e| CodegenError::new(format!("failed to declare string data: {e}")))?;

        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);

        let mut desc = DataDescription::new();
        desc.define(bytes.into_boxed_slice());
        self.module.define_data(data_id, &desc)
            .map_err(|e| CodegenError::new(format!("failed to define string data: {e}")))?;

        self.string_pool.insert(s.to_string(), data_id);
        Ok(data_id)
    }

    // ── Pass 0.5a: Collect enum definitions ─────────────────────────

    pub fn collect_enums(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::EnumDef(ed) = &stmt.kind {
                let enum_name = self.resolve_sym(ed.name).to_string();
                let mut variants = Vec::new();
                for (i, variant) in ed.variants.iter().enumerate() {
                    let variant_name = self.resolve_sym(variant.name).to_string();
                    let tag = i as i64;
                    // Store qualified name (always unique)
                    let qualified = format!("{}.{}", enum_name, variant_name);
                    self.enum_variants.insert(qualified, tag);
                    // Store bare name (may collide, last-write-wins for unqualified use)
                    self.enum_variants.insert(variant_name.clone(), tag);
                    variants.push((variant_name, tag));
                }
                self.enum_defs.insert(enum_name, variants);
            }
        }
        Ok(())
    }

    /// Look up an enum variant by name and return its integer tag.
    /// Accepts both bare names ("Red") and qualified names ("Color.Red").
    pub fn lookup_enum_variant_tag(&self, name: &str) -> Option<i64> {
        self.enum_variants.get(name).copied()
    }

    /// Look up with qualified enum name (EnumName.VariantName).
    pub fn lookup_qualified_variant(&self, enum_name: &str, variant_name: &str) -> Option<i64> {
        let qualified = format!("{}.{}", enum_name, variant_name);
        self.enum_variants.get(&qualified).copied()
    }

    // ── Pass 0.5b: Collect struct definitions ─────────────────────────

    pub fn collect_structs(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::StructDef(sd) = &stmt.kind {
                let name = self.resolve_sym(sd.name).to_string();
                let mut fields = Vec::new();
                let mut offset = 0usize;

                let struct_type = self.node_type(stmt.id).clone();
                if let Type::Struct { fields: type_fields, .. } = &struct_type {
                    for (field_sym, field_type) in type_fields {
                        let field_name = self.resolve_sym(*field_sym).to_string();
                        let cl_type = nsl_type_to_cl(field_type);
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name, cl_type, offset });
                        offset += size;
                    }
                } else {
                    for field in &sd.fields {
                        let field_name = self.resolve_sym(field.name).to_string();
                        let cl_type = match &field.type_ann.kind {
                            TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                            _ => cl_types::I64,
                        };
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name, cl_type, offset });
                        offset += size;
                    }
                }

                self.struct_layouts.insert(name.clone(), StructLayout { name, fields, total_size: offset });
            }
        }
        Ok(())
    }

    // ── Pass 0.5c: Collect model definitions ─────────────────────────

    pub fn collect_models(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::ModelDef(md) = &stmt.kind {
                let name = self.resolve_sym(md.name).to_string();
                let mut fields = Vec::new();
                let mut offset = 0usize;

                // Extract fields from AST LayerDecl members (type_map may not have stmt types)
                for member in &md.members {
                    if let nsl_ast::decl::ModelMember::LayerDecl { name: field_sym, type_ann, .. } = member {
                        let field_name = self.resolve_sym(*field_sym).to_string();
                        let cl_type = match &type_ann.kind {
                            nsl_ast::types::TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                            _ => cl_types::I64,
                        };
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name, cl_type, offset });
                        offset += size;
                    }
                }

                self.struct_layouts.insert(name.clone(), StructLayout { name, fields, total_size: offset });
            }
        }
        Ok(())
    }

    // ── Pass 1: Declare functions ───────────────────────────────────

    pub fn declare_runtime_functions(&mut self) -> Result<(), CodegenError> {
        self.runtime_fns = builtins::declare_runtime_functions(&mut self.module, self.call_conv)?;
        Ok(())
    }

    pub fn declare_user_functions(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        self.declare_user_functions_with_linkage(stmts, Linkage::Local)
    }

    pub fn declare_user_functions_with_linkage(
        &mut self,
        stmts: &[Stmt],
        linkage: Linkage,
    ) -> Result<(), CodegenError> {
        for stmt in stmts {
            let (fn_def, decorators) = match &stmt.kind {
                StmtKind::FnDef(fd) => (fd, None),
                StmtKind::Decorated { decorators, stmt } => {
                    if let StmtKind::FnDef(fd) = &stmt.kind {
                        (fd, Some(decorators))
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let raw_name = self.resolve_sym(fn_def.name).to_string();
            let cranelift_name = mangle_name(&self.module_prefix, &raw_name);
            let sig = self.build_fn_signature(fn_def);
            let func_id = self.module
                .declare_function(&cranelift_name, linkage, &sig)
                .map_err(|e| CodegenError::new(format!("failed to declare fn '{raw_name}': {e}")))?;
            self.functions.insert(raw_name.clone(), (func_id, sig));

            // Track decorated functions
            if let Some(decos) = decorators {
                for d in decos {
                    if d.name.len() == 1 {
                        let dname = self.resolve_sym(d.name[0]);
                        if dname == "no_grad" {
                            self.no_grad_fns.insert(raw_name.clone());
                        } else if dname == "test" {
                            self.test_fns.push(raw_name.clone());
                        }
                    }
                }
            }
        }

        // Collect model names so we skip them in the struct constructor loop
        let model_name_set: std::collections::HashSet<String> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind {
                    Some(self.resolve_sym(md.name).to_string())
                } else {
                    None
                }
            })
            .collect();

        let struct_names: Vec<String> = self.struct_layouts.keys()
            .filter(|k| !model_name_set.contains(*k))
            .cloned()
            .collect();
        for sname in struct_names {
            let layout = &self.struct_layouts[&sname];
            let mut sig = self.module.make_signature();
            sig.call_conv = self.call_conv;
            for field in &layout.fields {
                sig.params.push(AbiParam::new(field.cl_type));
            }
            sig.returns.push(AbiParam::new(pointer_type()));
            let func_id = self.module
                .declare_function(&format!("__nsl_struct_{sname}"), linkage, &sig)
                .map_err(|e| CodegenError::new(format!("failed to declare struct ctor '{sname}': {e}")))?;
            self.functions.insert(sname, (func_id, sig));
        }

        // Declare model constructors and methods
        let model_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
            })
            .collect();

        for md in &model_defs {
            let model_name = self.resolve_sym(md.name).to_string();

            // Constructor: takes constructor params, returns pointer
            let mut ctor_sig = self.module.make_signature();
            ctor_sig.call_conv = self.call_conv;
            for param in &md.params {
                let cl_type = if let Some(ref type_ann) = param.type_ann {
                    match &type_ann.kind {
                        TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                        _ => cl_types::I64,
                    }
                } else {
                    cl_types::I64
                };
                ctor_sig.params.push(AbiParam::new(cl_type));
            }
            ctor_sig.returns.push(AbiParam::new(pointer_type()));
            let ctor_name = format!("__nsl_model_{model_name}");
            let ctor_id = self.module
                .declare_function(&ctor_name, linkage, &ctor_sig)
                .map_err(|e| CodegenError::new(format!("failed to declare model ctor '{model_name}': {e}")))?;
            self.functions.insert(model_name.clone(), (ctor_id, ctor_sig));

            // Methods
            let mut method_map = HashMap::new();
            for member in &md.members {
                if let ModelMember::Method(fn_def) = member {
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mangled = format!("__nsl_model_{model_name}_{method_name}");

                    let mut method_sig = self.module.make_signature();
                    method_sig.call_conv = self.call_conv;
                    // First param: self pointer
                    method_sig.params.push(AbiParam::new(pointer_type()));
                    // Remaining params (skip "self" in AST)
                    for param in &fn_def.params {
                        let pname = self.resolve_sym(param.name).to_string();
                        if pname == "self" {
                            continue;
                        }
                        let cl_type = if let Some(ref type_ann) = param.type_ann {
                            match &type_ann.kind {
                                TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                                _ => cl_types::I64,
                            }
                        } else {
                            cl_types::I64
                        };
                        method_sig.params.push(AbiParam::new(cl_type));
                    }
                    // Return type
                    if let Some(ref ret_type) = fn_def.return_type {
                        match &ret_type.kind {
                            TypeExprKind::Named(sym) => {
                                let name = self.resolve_sym(*sym);
                                if name != "void" {
                                    method_sig.returns.push(AbiParam::new(self.resolve_type_name_to_cl(*sym)));
                                }
                            }
                            _ => { method_sig.returns.push(AbiParam::new(cl_types::I64)); }
                        }
                    } else {
                        method_sig.returns.push(AbiParam::new(cl_types::I64));
                    }

                    let method_id = self.module
                        .declare_function(&mangled, linkage, &method_sig)
                        .map_err(|e| CodegenError::new(format!("failed to declare model method '{mangled}': {e}")))?;
                    self.functions.insert(mangled.clone(), (method_id, method_sig));
                    method_map.insert(method_name, mangled);
                }
            }
            self.model_methods.insert(model_name, method_map);
        }

        Ok(())
    }

    /// Declare functions imported from other modules (Linkage::Import).
    /// Declare imported functions from other modules.
    /// Each entry is (raw_name, mangled_name, signature).
    /// The mangled name is used for the Cranelift symbol (must match the export),
    /// while the raw name is used as the lookup key in self.functions.
    pub fn declare_imported_functions(
        &mut self,
        imports: &[(String, String, Signature)],
    ) -> Result<(), CodegenError> {
        for (raw_name, mangled_name, sig) in imports {
            let func_id = self.module
                .declare_function(mangled_name, Linkage::Import, sig)
                .map_err(|e| CodegenError::new(format!("failed to declare imported fn '{raw_name}': {e}")))?;
            self.functions.insert(raw_name.clone(), (func_id, sig.clone()));
        }
        Ok(())
    }

    pub fn build_fn_signature(&self, fn_def: &nsl_ast::decl::FnDef) -> Signature {
        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;

        for param in &fn_def.params {
            let cl_type = if let Some(ref type_ann) = param.type_ann {
                match &type_ann.kind {
                    TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                    _ => cl_types::I64,
                }
            } else {
                cl_types::I64
            };
            sig.params.push(AbiParam::new(cl_type));
        }

        if let Some(ref ret_type) = fn_def.return_type {
            match &ret_type.kind {
                TypeExprKind::Named(sym) => {
                    let name = self.resolve_sym(*sym);
                    if name != "void" {
                        sig.returns.push(AbiParam::new(self.resolve_type_name_to_cl(*sym)));
                    }
                }
                _ => { sig.returns.push(AbiParam::new(cl_types::I64)); }
            }
        }
        sig
    }

    // ── Pass 2: Compile function bodies ─────────────────────────────

    pub fn compile_user_functions(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let fn_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::FnDef(fn_def) => Some(fn_def.clone()),
                StmtKind::Decorated { stmt, .. } => {
                    if let StmtKind::FnDef(fn_def) = &stmt.kind { Some(fn_def.clone()) } else { None }
                }
                _ => None,
            })
            .collect();

        for fn_def in &fn_defs {
            self.compile_fn_def(fn_def)?;
        }

        let layouts: Vec<(String, Vec<(cl_types::Type, usize)>, usize)> = self
            .struct_layouts.iter()
            .filter(|(name, _)| !self.model_methods.contains_key(*name))
            .map(|(name, layout)| {
                let fields: Vec<_> = layout.fields.iter().map(|f| (f.cl_type, f.offset)).collect();
                (name.clone(), fields, layout.total_size)
            })
            .collect();

        for (name, fields, total_size) in &layouts {
            self.compile_struct_constructor(name, fields, *total_size)?;
        }

        // Compile model constructors and methods
        let model_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
            })
            .collect();
        for md in &model_defs {
            self.compile_model_constructor(&md)?;
        }
        self.compile_model_methods(stmts)?;

        self.compile_pending_lambdas()?;
        Ok(())
    }

    /// Compile all pending lambda bodies. Drains in a loop to handle nested lambdas.
    pub fn compile_pending_lambdas(&mut self) -> Result<(), CodegenError> {
        while !self.pending_lambdas.is_empty() {
            let lambdas: Vec<PendingLambda> = self.pending_lambdas.drain(..).collect();
            for lambda in lambdas {
                self.compile_lambda_body(&lambda)?;
            }
        }
        Ok(())
    }

    fn compile_lambda_body(&mut self, lambda: &PendingLambda) -> Result<(), CodegenError> {
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            lambda.sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Bind normal parameters
            for (i, (sym, cl_type)) in lambda.params.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                let var = state.new_variable();
                builder.declare_var(var, *cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(*sym, (var, *cl_type));
            }

            // Bind captured variables (come after normal params in the signature)
            let capture_offset = lambda.params.len();
            for (i, (sym, cl_type)) in lambda.captures.iter().enumerate() {
                let param_val = builder.block_params(entry)[capture_offset + i];
                let var = state.new_variable();
                builder.declare_var(var, *cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(*sym, (var, *cl_type));
            }

            // Compile body expression
            let result = self.compile_expr(&mut builder, &mut state, &lambda.body)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: lambda '{}' ---\n{}", lambda.name, ctx.func.display());
        }

        self.module
            .define_function(lambda.func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define lambda '{}': {e}", lambda.name)))?;
        Ok(())
    }

    fn compile_struct_constructor(
        &mut self,
        name: &str,
        fields: &[(cl_types::Type, usize)],
        total_size: usize,
    ) -> Result<(), CodegenError> {
        let (func_id, sig) = self.functions[name].clone();
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let alloc_id = self.runtime_fns["nsl_alloc"].0;
            let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
            let size_val = builder.ins().iconst(cl_types::I64, total_size as i64);
            let call = builder.ins().call(alloc_ref, &[size_val]);
            let ptr = builder.inst_results(call)[0];

            for (i, (_cl_type, offset)) in fields.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                builder.ins().store(MemFlags::trusted(), param_val, ptr, *offset as i32);
            }

            builder.ins().return_(&[ptr]);
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: struct ctor '{name}' ---\n{}", ctx.func.display());
        }

        self.module.define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define struct ctor '{name}': {e}")))?;
        Ok(())
    }

    fn compile_model_constructor(
        &mut self,
        md: &nsl_ast::decl::ModelDef,
    ) -> Result<(), CodegenError> {
        let model_name = self.resolve_sym(md.name).to_string();
        let (func_id, sig) = self.functions[&model_name].clone();

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Bind constructor params as variables
            for (i, param) in md.params.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                let cl_type = if i < sig.params.len() {
                    sig.params[i].value_type
                } else {
                    cl_types::I64
                };
                let var = state.new_variable();
                builder.declare_var(var, cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(param.name, (var, cl_type));
            }

            // Allocate model memory
            let layout = self.struct_layouts.get(&model_name).cloned();
            let total_size = layout.as_ref().map(|l| l.total_size).unwrap_or(0);
            let alloc_size = total_size.max(8) as i64;

            let alloc_id = self.runtime_fns["nsl_alloc"].0;
            let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
            let size_val = builder.ins().iconst(cl_types::I64, alloc_size);
            let call = builder.ins().call(alloc_ref, &[size_val]);
            let ptr = builder.inst_results(call)[0];

            // Initialize fields
            if let Some(ref layout) = layout {
                let field_info: Vec<_> = layout.fields.iter().map(|f| (f.name.clone(), f.cl_type, f.offset)).collect();
                for member in &md.members {
                    if let ModelMember::LayerDecl { name, init, .. } = member {
                        let field_name = self.resolve_sym(*name).to_string();
                        if let Some((_fname, cl_type, offset)) = field_info.iter().find(|(n, _, _)| *n == field_name) {
                            if let Some(init_expr) = init {
                                let val = self.compile_expr(&mut builder, &mut state, init_expr)?;
                                builder.ins().store(MemFlags::trusted(), val, ptr, *offset as i32);
                            } else {
                                // Store zero for uninitialized fields
                                let zero = if cl_type.is_float() {
                                    builder.ins().f64const(0.0)
                                } else {
                                    builder.ins().iconst(*cl_type, 0)
                                };
                                builder.ins().store(MemFlags::trusted(), zero, ptr, *offset as i32);
                            }
                        }
                    }
                }
            }

            builder.ins().return_(&[ptr]);
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: model ctor '{model_name}' ---\n{}", ctx.func.display());
        }

        self.module.define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define model ctor '{model_name}': {e}")))?;
        Ok(())
    }

    fn compile_model_methods(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let model_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
            })
            .collect();

        for md in &model_defs {
            let model_name = self.resolve_sym(md.name).to_string();
            for member in &md.members {
                if let ModelMember::Method(fn_def) = member {
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mangled = format!("__nsl_model_{model_name}_{method_name}");
                    let (func_id, sig) = self.functions[&mangled].clone();

                    let mut ctx = Context::for_function(Function::with_name_signature(
                        UserFuncName::user(0, self.next_func_index()),
                        sig.clone(),
                    ));
                    let mut fn_builder_ctx = FunctionBuilderContext::new();

                    {
                        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
                        let mut state = FuncState::new();

                        let entry = builder.create_block();
                        builder.append_block_params_for_function_params(entry);
                        builder.switch_to_block(entry);
                        builder.seal_block(entry);
                        state.current_block = Some(entry);

                        // First Cranelift param is self (pointer)
                        let self_val = builder.block_params(entry)[0];
                        // Find the "self" symbol from the AST params
                        let self_sym = fn_def.params.iter()
                            .find(|p| self.resolve_sym(p.name) == "self")
                            .map(|p| p.name)
                            .unwrap_or_else(|| fn_def.params[0].name);
                        let self_var = state.new_variable();
                        builder.declare_var(self_var, pointer_type());
                        builder.def_var(self_var, self_val);
                        state.variables.insert(self_sym, (self_var, pointer_type()));

                        // Bind remaining params (skip "self" in AST)
                        let mut cl_param_idx = 1usize;
                        for param in &fn_def.params {
                            let pname = self.resolve_sym(param.name).to_string();
                            if pname == "self" {
                                continue;
                            }
                            let param_val = builder.block_params(entry)[cl_param_idx];
                            let cl_type = if cl_param_idx < sig.params.len() {
                                sig.params[cl_param_idx].value_type
                            } else {
                                cl_types::I64
                            };
                            let var = state.new_variable();
                            builder.declare_var(var, cl_type);
                            builder.def_var(var, param_val);
                            state.variables.insert(param.name, (var, cl_type));
                            cl_param_idx += 1;
                        }

                        // Compile method body
                        for stmt in &fn_def.body.stmts {
                            self.compile_stmt(&mut builder, &mut state, stmt)?;
                        }

                        // Add implicit return if body doesn't end with one
                        let current = state.current_block.unwrap_or(entry);
                        if !crate::types::is_block_filled(&builder, current) {
                            if sig.returns.is_empty() {
                                builder.ins().return_(&[]);
                            } else {
                                let ret_type = sig.returns[0].value_type;
                                let zero = if ret_type.is_float() {
                                    builder.ins().f64const(0.0)
                                } else {
                                    builder.ins().iconst(ret_type, 0)
                                };
                                builder.ins().return_(&[zero]);
                            }
                        }

                        builder.finalize();
                    }

                    if self.dump_ir {
                        eprintln!("--- IR: model method '{mangled}' ---\n{}", ctx.func.display());
                    }

                    self.module.define_function(func_id, &mut ctx)
                        .map_err(|e| CodegenError::new(format!("failed to define model method '{mangled}': {e}")))?;
                }
            }
        }
        Ok(())
    }

    // ── Pass 3: Compile top-level stmts into main() ─────────────────

    pub fn compile_main(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let top_level: Vec<_> = stmts
            .iter()
            .filter(|s| {
                // Filter out decorated function definitions (compiled as top-level fns)
                if let StmtKind::Decorated { stmt, .. } = &s.kind {
                    if matches!(stmt.kind, StmtKind::FnDef(_)) {
                        return false;
                    }
                }
                !matches!(
                    s.kind,
                    StmtKind::FnDef(_) | StmtKind::StructDef(_) | StmtKind::ModelDef(_)
                        | StmtKind::EnumDef(_) | StmtKind::TraitDef(_)
                        | StmtKind::Import(_) | StmtKind::FromImport(_)
                        | StmtKind::DatasetDef(_) | StmtKind::TokenizerDef(_)
                        | StmtKind::QuantBlock(_)
                        | StmtKind::KernelDef(_)
                )
            })
            .cloned()
            .collect();

        if top_level.is_empty() {
            return Ok(());
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        sig.params.push(AbiParam::new(cl_types::I32));  // argc
        sig.params.push(AbiParam::new(cl_types::I64));  // argv
        sig.returns.push(AbiParam::new(cl_types::I32));

        let main_id = self.module
            .declare_function("main", Linkage::Export, &sig)
            .map_err(|e| CodegenError::new(format!("failed to declare main: {e}")))?;

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig,
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Initialize command-line args for args() support
            let argc_val = builder.block_params(entry)[0];
            let argv_val = builder.block_params(entry)[1];
            let init_id = self.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            for stmt in &top_level {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                let zero = builder.ins().iconst(cl_types::I32, 0);
                builder.ins().return_(&[zero]);
            }

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: main ---\n{}", ctx.func.display());
        }

        self.module.define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define main: {e}")))?;
        Ok(())
    }

    pub fn finalize(self) -> Result<Vec<u8>, CodegenError> {
        let product = self.module.finish();
        product.emit().map_err(|e| CodegenError::new(format!("failed to emit object: {e}")))
    }
}

/// Main entry point (single-file, backward compatible).
pub fn compile(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}

/// Compile a library module (non-entry). Functions use Linkage::Export, no main().
pub fn compile_module(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;
    compiler.module_prefix = module_prefix.to_string();
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}

/// Compile the entry module with imported functions from other modules.
/// Own functions use Linkage::Export, imported functions use Linkage::Import.
/// imported_fns entries are (raw_name, mangled_name, signature).
#[allow(clippy::too_many_arguments)]
pub fn compile_entry(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_enum_variants: HashMap<String, i64>,
    imported_enum_defs: HashMap<String, Vec<(String, i64)>>,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;

    // Register imported structs/enums so the entry module can reference them
    for (name, layout) in imported_struct_layouts {
        compiler.struct_layouts.insert(name, layout);
    }
    for (name, tag) in imported_enum_variants {
        compiler.enum_variants.insert(name, tag);
    }
    for (name, variants) in imported_enum_defs {
        compiler.enum_defs.insert(name, variants);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}
