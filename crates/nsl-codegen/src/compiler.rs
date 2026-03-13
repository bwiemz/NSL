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

use nsl_ast::block::DatatypeMethodKind;
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
    /// Kernel name → (ptx_data_id, name_data_id) for compiled GPU kernels.
    pub kernel_ptx_data: HashMap<String, (DataId, DataId)>,
    /// model_name → { field_name → type_marker }
    /// type_marker is "[ElemModel;N]" for fixed model arrays, or the nested model name.
    pub model_field_types: HashMap<String, HashMap<String, String>>,
    /// Variable Symbol → model name (for model array loop variables)
    pub model_var_types: HashMap<Symbol, String>,
    /// Names of models imported from other modules (not defined locally).
    /// These have struct layouts but should NOT generate struct ctors.
    pub imported_model_names: HashSet<String>,
    /// Custom dtype name → numeric id (starting at 256). Populated by compile_datatype_defs.
    pub custom_dtype_ids: HashMap<String, u16>,
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
            kernel_ptx_data: HashMap::new(),
            model_field_types: HashMap::new(),
            model_var_types: HashMap::new(),
            imported_model_names: HashSet::new(),
            custom_dtype_ids: HashMap::new(),
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
                let mut field_type_map: HashMap<String, String> = HashMap::new();

                // Extract fields from AST LayerDecl members (type_map may not have stmt types)
                for member in &md.members {
                    if let nsl_ast::decl::ModelMember::LayerDecl { name: field_sym, type_ann, .. } = member {
                        let field_name = self.resolve_sym(*field_sym).to_string();

                        // Check if this is a FixedArray type
                        if let nsl_ast::types::TypeExprKind::FixedArray { element_type, size } = &type_ann.kind {
                            let align = 8usize;
                            offset = (offset + align - 1) & !(align - 1);
                            // Store a single StructField pointing to the base of the array
                            fields.push(StructField { name: field_name.clone(), cl_type: cl_types::I64, offset });
                            offset += (*size as usize) * 8;

                            // Record array marker in field_type_map
                            if let nsl_ast::types::TypeExprKind::Named(elem_sym) = &element_type.kind {
                                let elem_name = self.resolve_sym(*elem_sym).to_string();
                                field_type_map.insert(field_name, format!("[{};{}]", elem_name, size));
                            }
                            continue;
                        }

                        let cl_type = match &type_ann.kind {
                            nsl_ast::types::TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                            _ => cl_types::I64,
                        };
                        let size = cl_type.bytes() as usize;
                        let align = size.max(1);
                        offset = (offset + align - 1) & !(align - 1);
                        fields.push(StructField { name: field_name.clone(), cl_type, offset });
                        offset += size;

                        // Check if this is a nested model field
                        if let nsl_ast::types::TypeExprKind::Named(type_sym) = &type_ann.kind {
                            let type_name = self.resolve_sym(*type_sym).to_string();
                            if self.struct_layouts.contains_key(&type_name) {
                                field_type_map.insert(field_name, type_name);
                            }
                        }
                    }
                }

                if !field_type_map.is_empty() {
                    self.model_field_types.insert(name.clone(), field_type_map);
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
            .filter(|k| !model_name_set.contains(*k) && !self.imported_model_names.contains(*k))
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

    /// Build signatures for all model ctors and methods in a module's AST.
    /// Returns (raw_name, mangled_name, signature) tuples for use as imported_fns.
    pub fn build_model_signatures(&mut self, stmts: &[Stmt]) -> Vec<(String, String, Signature)> {
        let mut result = Vec::new();

        for stmt in stmts {
            if let StmtKind::ModelDef(md) = &stmt.kind {
                let model_name = self.resolve_sym(md.name).to_string();

                // Constructor signature: (params...) -> ptr
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
                let ctor_mangled = format!("__nsl_model_{model_name}");
                result.push((model_name.clone(), ctor_mangled, ctor_sig));

                // Method signatures
                for member in &md.members {
                    if let nsl_ast::decl::ModelMember::Method(fn_def) = member {
                        let method_name = self.resolve_sym(fn_def.name).to_string();
                        let mangled = format!("__nsl_model_{model_name}_{method_name}");

                        let mut method_sig = self.module.make_signature();
                        method_sig.call_conv = self.call_conv;
                        // First param: self pointer
                        method_sig.params.push(AbiParam::new(pointer_type()));
                        // Remaining params (skip "self")
                        for param in &fn_def.params {
                            let pname = self.resolve_sym(param.name).to_string();
                            if pname == "self" { continue; }
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

                        result.push((mangled.clone(), mangled, method_sig));
                    }
                }
            }
        }

        result
    }

    // ── Compile custom datatype definitions (before kernels) ──

    pub fn compile_datatype_defs(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::DatatypeDef(def) = &stmt.kind {
                let dtype_name = self.resolve_sym(def.name).to_string();
                let id = 256u16 + self.custom_dtype_ids.len() as u16;
                self.custom_dtype_ids.insert(dtype_name.clone(), id);

                // Compile each method as a standalone Cranelift function
                for method in &def.methods {
                    let (fn_suffix, compile_it) = match method.kind {
                        DatatypeMethodKind::Pack => ("pack", true),
                        DatatypeMethodKind::Unpack => ("unpack", true),
                        DatatypeMethodKind::BackwardPack => ("backward_pack", true),
                        DatatypeMethodKind::Arithmetic => ("arithmetic", true),
                    };
                    if !compile_it {
                        continue;
                    }

                    let fn_name = format!("__nsl_dtype_{}_{}", dtype_name, fn_suffix);

                    // Build signature from method params and return type
                    let mut sig = self.module.make_signature();
                    sig.call_conv = self.call_conv;
                    for param in &method.params {
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
                    if let Some(ref ret_type) = method.return_type {
                        match &ret_type.kind {
                            TypeExprKind::Named(sym) => {
                                let rname = self.resolve_sym(*sym);
                                if rname != "void" {
                                    sig.returns.push(AbiParam::new(self.resolve_type_name_to_cl(*sym)));
                                }
                            }
                            _ => { sig.returns.push(AbiParam::new(cl_types::I64)); }
                        }
                    }

                    let func_id = self.module
                        .declare_function(&fn_name, Linkage::Local, &sig)
                        .map_err(|e| CodegenError::new(format!(
                            "failed to declare dtype method '{fn_name}': {e}"
                        )))?;
                    self.functions.insert(fn_name.clone(), (func_id, sig.clone()));

                    // Compile the method body
                    let mut ctx = Context::for_function(Function::with_name_signature(
                        UserFuncName::user(0, self.next_func_index()),
                        sig.clone(),
                    ));
                    let mut fn_builder_ctx = FunctionBuilderContext::new();

                    {
                        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
                        let mut state = crate::context::FuncState::new();

                        let entry = builder.create_block();
                        builder.append_block_params_for_function_params(entry);
                        builder.switch_to_block(entry);
                        builder.seal_block(entry);
                        state.current_block = Some(entry);

                        // Bind parameters
                        for (i, param) in method.params.iter().enumerate() {
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

                        // Compile body statements
                        for body_stmt in &method.body {
                            self.compile_stmt(&mut builder, &mut state, body_stmt)?;
                        }

                        // Implicit return if not already terminated
                        let current = state.current_block.unwrap_or(entry);
                        if !crate::types::is_block_filled(&builder, current) {
                            if sig.returns.is_empty() {
                                builder.ins().return_(&[]);
                            } else {
                                let ret_type = sig.returns[0].value_type;
                                if ret_type == cl_types::F64 {
                                    let zero = builder.ins().f64const(0.0);
                                    builder.ins().return_(&[zero]);
                                } else if ret_type == cl_types::F32 {
                                    let zero = builder.ins().f32const(0.0);
                                    builder.ins().return_(&[zero]);
                                } else {
                                    let zero = builder.ins().iconst(ret_type, 0);
                                    builder.ins().return_(&[zero]);
                                }
                            }
                        }

                        builder.finalize();
                    }

                    if self.dump_ir {
                        eprintln!("--- IR: dtype method '{fn_name}' ---\n{}", ctx.func.display());
                    }

                    self.module
                        .define_function(func_id, &mut ctx)
                        .map_err(|e| CodegenError::new(format!(
                            "failed to define dtype method '{fn_name}': {e}"
                        )))?;
                }

                // Intern the dtype name string for registration calls
                self.intern_string(&dtype_name)?;
            }
        }
        Ok(())
    }

    /// Emit registration calls for all custom dtypes into a function being built.
    /// Must be called inside a builder context (e.g., in compile_main).
    pub fn emit_dtype_registration(
        &mut self,
        builder: &mut FunctionBuilder,
        stmts: &[Stmt],
    ) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::DatatypeDef(def) = &stmt.kind {
                let dtype_name = self.resolve_sym(def.name).to_string();
                let id = match self.custom_dtype_ids.get(&dtype_name) {
                    Some(&id) => id,
                    None => continue,
                };
                let bit_width = def.bits.unwrap_or(0) as i64;
                let block_size = def.block_size.unwrap_or(0) as i64;

                // Compute packed_block_size from block_size and bit_width
                let packed_block_size = if block_size > 0 {
                    (block_size * bit_width + 7) / 8
                } else {
                    0i64
                };

                // Get function pointers for pack/unpack
                let pack_fn_name = format!("__nsl_dtype_{}_pack", dtype_name);
                let unpack_fn_name = format!("__nsl_dtype_{}_unpack", dtype_name);

                let pack_addr = if let Some((fid, _)) = self.functions.get(&pack_fn_name) {
                    let fref = self.module.declare_func_in_func(*fid, builder.func);
                    builder.ins().func_addr(cl_types::I64, fref)
                } else {
                    builder.ins().iconst(cl_types::I64, 0)
                };
                let unpack_addr = if let Some((fid, _)) = self.functions.get(&unpack_fn_name) {
                    let fref = self.module.declare_func_in_func(*fid, builder.func);
                    builder.ins().func_addr(cl_types::I64, fref)
                } else {
                    builder.ins().iconst(cl_types::I64, 0)
                };

                // Get name data pointer and length
                let name_data_id = self.string_pool[&dtype_name];
                let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
                let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
                let name_len = builder.ins().iconst(cl_types::I64, dtype_name.len() as i64);

                let id_val = builder.ins().iconst(cl_types::I64, id as i64);
                let bw_val = builder.ins().iconst(cl_types::I64, bit_width);
                let bs_val = builder.ins().iconst(cl_types::I64, block_size);
                let pbs_val = builder.ins().iconst(cl_types::I64, packed_block_size);

                // Call nsl_register_custom_dtype(id, name_ptr, name_len, bit_width, block_size, packed_block_size, pack_fn, unpack_fn)
                let reg_id = self.runtime_fns["nsl_register_custom_dtype"].0;
                let reg_ref = self.module.declare_func_in_func(reg_id, builder.func);
                builder.ins().call(reg_ref, &[id_val, name_ptr, name_len, bw_val, bs_val, pbs_val, pack_addr, unpack_addr]);
            }
        }

        // Call nsl_finalize_dtype_registry() after all registrations
        if self.custom_dtype_ids.is_empty() {
            return Ok(());
        }
        let fin_id = self.runtime_fns["nsl_finalize_dtype_registry"].0;
        let fin_ref = self.module.declare_func_in_func(fin_id, builder.func);
        builder.ins().call(fin_ref, &[]);

        Ok(())
    }

    // ── Compile kernel definitions (PTX → .rodata, before functions) ──

    pub fn compile_kernels(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            if let StmtKind::KernelDef(kernel) = &stmt.kind {
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
                    .map_err(|e| CodegenError::new(format!("failed to declare PTX data for kernel '{}': {e}", kernel_name)))?;
                let mut data_desc = cranelift_module::DataDescription::new();
                data_desc.define(ptx_bytes.into_boxed_slice());
                self.module
                    .define_data(ptx_data_id, &data_desc)
                    .map_err(|e| CodegenError::new(format!("failed to define PTX data for kernel '{}': {e}", kernel_name)))?;

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
                    .map_err(|e| CodegenError::new(format!("failed to declare name data for kernel '{}': {e}", kernel_name)))?;
                let mut name_desc = cranelift_module::DataDescription::new();
                name_desc.define(name_bytes.into_boxed_slice());
                self.module
                    .define_data(name_data_id, &name_desc)
                    .map_err(|e| CodegenError::new(format!("failed to define name data for kernel '{}': {e}", kernel_name)))?;

                self.kernel_ptx_data.insert(kernel_name, (ptx_data_id, name_data_id));
            }
        }
        Ok(())
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

        #[allow(clippy::type_complexity)]
        let layouts: Vec<(String, Vec<(cl_types::Type, usize)>, usize)> = self
            .struct_layouts.iter()
            .filter(|(name, _)| !self.model_methods.contains_key(*name) && !self.imported_model_names.contains(*name))
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
            self.compile_model_constructor(md)?;
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
            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                builder.ins().return_(&[result]);
            }
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
                let members_clone: Vec<_> = md.members.clone();
                for member in &members_clone {
                    if let ModelMember::LayerDecl { name, init, type_ann, .. } = member {
                        let field_name = self.resolve_sym(*name).to_string();

                        // Check if this is a FixedArray field → generate init loop
                        if let nsl_ast::types::TypeExprKind::FixedArray { size, .. } = &type_ann.kind {
                            let arr_size = *size;
                            if let Some((_fname, _cl_type, field_offset)) = field_info.iter().find(|(n, _, _)| *n == field_name) {
                                let field_offset = *field_offset;
                                if let Some(init_expr) = init {
                                    let init_expr_clone = init_expr.clone();
                                    // Generate loop: for i in 0..arr_size, compile init, store at ptr+field_offset+i*8
                                    let counter_var = state.new_variable();
                                    builder.declare_var(counter_var, cl_types::I64);
                                    let zero = builder.ins().iconst(cl_types::I64, 0);
                                    builder.def_var(counter_var, zero);

                                    let loop_header = builder.create_block();
                                    let loop_body = builder.create_block();
                                    let loop_exit = builder.create_block();

                                    builder.ins().jump(loop_header, &[]);

                                    builder.switch_to_block(loop_header);
                                    state.current_block = Some(loop_header);
                                    let counter = builder.use_var(counter_var);
                                    let limit = builder.ins().iconst(cl_types::I64, arr_size);
                                    let cond = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, counter, limit);
                                    builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

                                    builder.switch_to_block(loop_body);
                                    builder.seal_block(loop_body);
                                    state.current_block = Some(loop_body);

                                    let init_val = self.compile_expr(&mut builder, &mut state, &init_expr_clone)?;
                                    let counter = builder.use_var(counter_var);
                                    let eight = builder.ins().iconst(cl_types::I64, 8);
                                    let elem_byte_offset = builder.ins().imul(counter, eight);
                                    let base_off = builder.ins().iconst(cl_types::I64, field_offset as i64);
                                    let total_off = builder.ins().iadd(base_off, elem_byte_offset);
                                    let addr = builder.ins().iadd(ptr, total_off);
                                    builder.ins().store(MemFlags::trusted(), init_val, addr, 0);

                                    let next_counter = builder.use_var(counter_var);
                                    let one = builder.ins().iconst(cl_types::I64, 1);
                                    let next = builder.ins().iadd(next_counter, one);
                                    builder.def_var(counter_var, next);
                                    builder.ins().jump(loop_header, &[]);

                                    builder.seal_block(loop_header);
                                    builder.switch_to_block(loop_exit);
                                    builder.seal_block(loop_exit);
                                    state.current_block = Some(loop_exit);
                                }
                            }
                            continue;
                        }

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
                    let (func_id, sig) = self.functions.get(&mangled)
                        .ok_or_else(|| CodegenError::new(format!("model method '{}' not registered", mangled)))?
                        .clone();

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
                                let zero = if ret_type == cl_types::F64 {
                                    builder.ins().f64const(0.0)
                                } else if ret_type == cl_types::F32 {
                                    builder.ins().f32const(0.0)
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
                        | StmtKind::KernelDef(_) | StmtKind::DatatypeDef(_)
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

            // Register custom dtypes before user code runs
            self.emit_dtype_registration(&mut builder, stmts)?;

            for stmt in &top_level {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                // Stop and free any DataLoaders before implicit main() return
                self.teardown_dataloaders(&mut builder, &mut state);
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

    /// Compile a test-dispatch main() that reads `--run <name>` from argv and
    /// calls the corresponding @test function. Used by `nsl test`.
    pub fn compile_test_main(&mut self) -> Result<(), CodegenError> {
        let test_fns = self.test_fns.clone();
        if test_fns.is_empty() {
            return Err(CodegenError::new("no @test functions found".to_string()));
        }

        // Ensure test function name strings are in the string pool
        let run_flag = "--run".to_string();
        self.intern_string(&run_flag)?;
        for name in &test_fns {
            self.intern_string(name)?;
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        sig.params.push(AbiParam::new(cl_types::I32)); // argc
        sig.params.push(AbiParam::new(cl_types::I64)); // argv
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

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            // Initialize args
            let argc_val = builder.block_params(entry)[0];
            let argv_val = builder.block_params(entry)[1];
            let init_id = self.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // Get args list
            let args_id = self.runtime_fns["nsl_args"].0;
            let args_ref = self.module.declare_func_in_func(args_id, builder.func);
            let args_call = builder.ins().call(args_ref, &[]);
            let args_list = builder.inst_results(args_call)[0];

            // Get list length
            let len_id = self.runtime_fns["nsl_list_len"].0;
            let len_ref = self.module.declare_func_in_func(len_id, builder.func);
            let len_call = builder.ins().call(len_ref, &[args_list]);
            let args_len = builder.inst_results(len_call)[0];

            // Check argc >= 3 (program, --run, test_name)
            let three = builder.ins().iconst(cl_types::I64, 3);
            let has_args = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual, args_len, three);

            let dispatch_block = builder.create_block();
            let exit_one_block = builder.create_block();

            builder.ins().brif(has_args, dispatch_block, &[], exit_one_block, &[]);

            // exit(1) block — no --run flag
            builder.switch_to_block(exit_one_block);
            builder.seal_block(exit_one_block);
            let one = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().return_(&[one]);

            // Dispatch block: check argv[1] == "--run"
            builder.switch_to_block(dispatch_block);
            builder.seal_block(dispatch_block);

            let get_id = self.runtime_fns["nsl_list_get"].0;
            let get_ref = self.module.declare_func_in_func(get_id, builder.func);
            let eq_id = self.runtime_fns["nsl_str_eq"].0;
            let eq_ref = self.module.declare_func_in_func(eq_id, builder.func);

            // argv[1]
            let idx1 = builder.ins().iconst(cl_types::I64, 1);
            let get_call1 = builder.ins().call(get_ref, &[args_list, idx1]);
            let arg1 = builder.inst_results(get_call1)[0];

            // "--run" string constant
            let run_data_id = self.string_pool[&run_flag];
            let run_gv = self.module.declare_data_in_func(run_data_id, builder.func);
            let run_ptr = builder.ins().symbol_value(cl_types::I64, run_gv);

            let eq_call = builder.ins().call(eq_ref, &[arg1, run_ptr]);
            let is_run = builder.inst_results(eq_call)[0];
            let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
            let run_check = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, is_run, zero_i64);

            let name_check_block = builder.create_block();
            let exit_one_block2 = builder.create_block();

            builder.ins().brif(run_check, name_check_block, &[], exit_one_block2, &[]);

            builder.switch_to_block(exit_one_block2);
            builder.seal_block(exit_one_block2);
            let one2 = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().return_(&[one2]);

            // Name check block: get argv[2] and match against test names
            builder.switch_to_block(name_check_block);
            builder.seal_block(name_check_block);

            let idx2 = builder.ins().iconst(cl_types::I64, 2);
            let get_call2 = builder.ins().call(get_ref, &[args_list, idx2]);
            let arg2 = builder.inst_results(get_call2)[0];

            // Build chain: if arg2 == "test_name" { call test_name(); return 0; }
            let exit_fail_block = builder.create_block();

            for (i, test_name) in test_fns.iter().enumerate() {
                let match_block = builder.create_block();
                let next_block = if i + 1 < test_fns.len() {
                    builder.create_block()
                } else {
                    exit_fail_block
                };

                // Compare arg2 with test_name string constant
                let name_data_id = self.string_pool[test_name];
                let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
                let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
                let eq_ref2 = self.module.declare_func_in_func(eq_id, builder.func);
                let eq_call2 = builder.ins().call(eq_ref2, &[arg2, name_ptr]);
                let is_match = builder.inst_results(eq_call2)[0];
                let zero2 = builder.ins().iconst(cl_types::I64, 0);
                let match_check = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, is_match, zero2);

                builder.ins().brif(match_check, match_block, &[], next_block, &[]);

                // Match block: call the test function and return 0
                builder.switch_to_block(match_block);
                builder.seal_block(match_block);

                let (func_id, _) = &self.functions[test_name];
                let fn_ref = self.module.declare_func_in_func(*func_id, builder.func);
                builder.ins().call(fn_ref, &[]);
                let zero_ret = builder.ins().iconst(cl_types::I32, 0);
                builder.ins().return_(&[zero_ret]);

                if i + 1 < test_fns.len() {
                    builder.switch_to_block(next_block);
                    builder.seal_block(next_block);
                }
            }

            // No match — exit with code 2
            builder.switch_to_block(exit_fail_block);
            builder.seal_block(exit_fail_block);
            let two = builder.ins().iconst(cl_types::I32, 2);
            builder.ins().return_(&[two]);

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: main (test dispatch) ---\n{}", ctx.func.display());
        }

        self.module.define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define test main: {e}")))?;
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
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}

/// Compile in test mode: functions are compiled normally but main() dispatches
/// to @test functions based on `--run <name>` argv. Returns (object_bytes, test_fn_names).
pub fn compile_test(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
) -> Result<(Vec<u8>, Vec<String>), CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    let test_fns = compiler.test_fns.clone();
    if test_fns.is_empty() {
        return Err(CodegenError::new("no @test functions found".to_string()));
    }
    compiler.compile_test_main()?;
    let bytes = compiler.finalize()?;
    Ok((bytes, test_fns))
}

/// Compile a library module (non-entry). Functions use Linkage::Export, no main().
pub fn compile_module(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    compile_module_with_imports(ast, interner, type_map, module_prefix, &[], HashMap::new(), HashSet::new(), dump_ir)
}

/// Compile a library module with imported symbols from its own dependencies.
#[allow(clippy::too_many_arguments)]
pub fn compile_module_with_imports(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;
    compiler.module_prefix = module_prefix.to_string();

    // Register imported structs/models from dependencies
    for (name, layout) in imported_struct_layouts {
        compiler.struct_layouts.insert(name, layout);
    }
    for name in imported_model_names {
        compiler.imported_model_names.insert(name);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
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
    imported_model_names: HashSet<String>,
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
    // Mark imported model names so we don't generate struct ctors for them
    for name in imported_model_names {
        compiler.imported_model_names.insert(name);
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
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}
