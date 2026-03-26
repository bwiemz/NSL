use std::collections::HashMap;

use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};

use nsl_ast::block::DatatypeMethodKind;
use nsl_ast::decl::ModelMember;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::TypeExprKind;

use super::{mangle_name, Compiler};
use crate::error::CodegenError;
use crate::types::pointer_type;

impl Compiler<'_> {
    // ── Pass 1: Declare functions ───────────────────────────────────

    pub fn declare_runtime_functions(&mut self) -> Result<(), CodegenError> {
        self.runtime_fns = crate::builtins::declare_runtime_functions(&mut self.module, self.call_conv)?;
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
                        } else if dname == "fp8_compute" {
                            self.features.fp8_compute_fns.insert(raw_name.clone());
                        }
                    }
                }
                // M53: Extract @real_time and @wcet_budget constraints
                if let Some(rt) = crate::wcet::extract_real_time_decorator(decos, &|sym| self.resolve_sym(sym)) {
                    self.features.real_time_fns.insert(raw_name.clone(), rt);
                }
                if let Some(wb) = crate::wcet::extract_wcet_budget_decorator(decos, &|sym| self.resolve_sym(sym)) {
                    self.features.wcet_budget_fns.insert(raw_name.clone(), wb);
                }
                // M55: Extract @zk_proof from bare function decorators
                if let Some(mode) = crate::zk::extract_zk_proof_decorator(decos, &|sym| self.resolve_sym(sym)) {
                    self.features.zk_proof_fns.insert(raw_name.clone(), mode);
                }
                // M55: Extract @zk_lookup from function decorators
                if let Some((ib, ob)) = crate::zk::extract_zk_lookup_decorator(decos, &|sym| self.resolve_sym(sym)) {
                    self.features.zk_lookup_fns.insert(raw_name.clone(), (ib, ob));
                }
                // M39: Extract @vmap from function decorators
                if let Some(vmap_config) = crate::vmap::extract_vmap_decorator(decos, &|sym| self.resolve_sym(sym)) {
                    self.features.vmap_configs.insert(raw_name.clone(), vmap_config);
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
                if let ModelMember::Method(fn_def, decos) = member {
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

                    // M53: Extract @real_time and @wcet_budget from model method decorators
                    if !decos.is_empty() {
                        if let Some(rt) = crate::wcet::extract_real_time_decorator(decos, &|sym| self.resolve_sym(sym)) {
                            self.features.real_time_fns.insert(mangled.clone(), rt);
                        }
                        if let Some(wb) = crate::wcet::extract_wcet_budget_decorator(decos, &|sym| self.resolve_sym(sym)) {
                            self.features.wcet_budget_fns.insert(mangled.clone(), wb);
                        }
                        // M55: Extract @zk_proof from model method decorators
                        if let Some(mode) = crate::zk::extract_zk_proof_decorator(decos, &|sym| self.resolve_sym(sym)) {
                            self.features.zk_proof_fns.insert(mangled.clone(), mode);
                        }
                    }

                    method_map.insert(method_name, mangled);
                }
            }
            self.model_methods.insert(model_name, method_map);
        }

        // M55: Handle @zk_proof on whole model blocks.
        // When a `model` statement is wrapped in `@zk_proof(...)`, register all
        // its methods in `zk_proof_fns` using the same mangled names used above.
        for stmt in stmts {
            if let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind {
                if let StmtKind::ModelDef(md) = &inner.kind {
                    if let Some(mode) = crate::zk::extract_zk_proof_decorator(
                        decorators,
                        &|sym| self.resolve_sym(sym),
                    ) {
                        let model_name = self.resolve_sym(md.name).to_string();
                        for member in &md.members {
                            if let ModelMember::Method(fn_def, _) = member {
                                let method_name = self.resolve_sym(fn_def.name).to_string();
                                let mangled = format!("__nsl_model_{model_name}_{method_name}");
                                self.features.zk_proof_fns.insert(mangled, mode);
                            }
                        }
                    }
                }
            }
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
                    if let nsl_ast::decl::ModelMember::Method(fn_def, _decos) = member {
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
                    let fn_suffix = match method.kind {
                        DatatypeMethodKind::Pack => "pack",
                        DatatypeMethodKind::Unpack => "unpack",
                        DatatypeMethodKind::BackwardPack => "backward_pack",
                        DatatypeMethodKind::Arithmetic => "arithmetic",
                    };

                    let fn_name = format!("__nsl_dtype_{}_{}", dtype_name, fn_suffix);
                    let is_block_mode = def.block_size.unwrap_or(0) > 0;

                    // Build signature using the runtime's calling convention, NOT the
                    // NSL-level type annotations.  The runtime transmutes function
                    // pointers to a fixed ABI:
                    //   element-wise  pack:   extern "C" fn(f64) -> i64
                    //   element-wise  unpack: extern "C" fn(i64) -> f64
                    //   block-wise    pack:   extern "C" fn(*const f64, i64, *mut u8)
                    //   block-wise    unpack: extern "C" fn(*const u8, i64, *mut f64)
                    let mut sig = self.module.make_signature();
                    sig.call_conv = self.call_conv;

                    match (method.kind, is_block_mode) {
                        (DatatypeMethodKind::Pack, false) => {
                            // element-wise pack: fn(i64) -> i64
                            // Runtime passes f64 bits as i64 and expects packed i64 back.
                            // The body bitcasts i64→f64 so the user-visible param is float.
                            sig.params.push(AbiParam::new(cl_types::I64));
                            sig.returns.push(AbiParam::new(cl_types::I64));
                        }
                        (DatatypeMethodKind::Unpack, false) => {
                            // element-wise unpack: fn(i64) -> i64
                            // Runtime passes packed i64, expects f64 bits as i64 back.
                            sig.params.push(AbiParam::new(cl_types::I64));
                            sig.returns.push(AbiParam::new(cl_types::I64));
                        }
                        (DatatypeMethodKind::Pack, true) => {
                            // block-wise pack: fn(*const f64, i64, *mut u8) -> void
                            sig.params.push(AbiParam::new(cl_types::I64)); // ptr to f64 block
                            sig.params.push(AbiParam::new(cl_types::I64)); // block size
                            sig.params.push(AbiParam::new(cl_types::I64)); // ptr to output
                        }
                        (DatatypeMethodKind::Unpack, true) => {
                            // block-wise unpack: fn(*const u8, i64, *mut f64) -> void
                            sig.params.push(AbiParam::new(cl_types::I64)); // ptr to packed
                            sig.params.push(AbiParam::new(cl_types::I64)); // block size
                            sig.params.push(AbiParam::new(cl_types::I64)); // ptr to output
                        }
                        _ => {
                            // BackwardPack, Arithmetic — use AST annotations as before
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
                        }
                    }

                    let func_id = self.module
                        .declare_function(&fn_name, Linkage::Export, &sig)
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
                        state.in_dtype_method = true;

                        let entry = builder.create_block();
                        builder.append_block_params_for_function_params(entry);
                        builder.switch_to_block(entry);
                        builder.seal_block(entry);
                        state.current_block = Some(entry);

                        // Bind parameters.
                        // For element-wise pack: CL param is i64 (f64 bits), user expects f64.
                        //   Bitcast i64→f64 and declare the variable as F64.
                        // For element-wise unpack: CL param is i64, user expects i64. Direct bind.
                        // For block-wise: all params are i64 pointers/sizes. Direct bind.
                        let is_elem_pack = !is_block_mode && method.kind == DatatypeMethodKind::Pack;
                        for (i, param) in method.params.iter().enumerate() {
                            if i >= sig.params.len() { break; }
                            let param_val = builder.block_params(entry)[i];
                            if is_elem_pack && i == 0 {
                                // Bitcast i64 → f64 for the value parameter
                                let f64_val = builder.ins().bitcast(cl_types::F64, MemFlags::new(), param_val);
                                let var = state.new_variable();
                                builder.declare_var(var, cl_types::F64);
                                builder.def_var(var, f64_val);
                                state.variables.insert(param.name, (var, cl_types::F64));
                            } else {
                                let cl_type = sig.params[i].value_type;
                                let var = state.new_variable();
                                builder.declare_var(var, cl_type);
                                builder.def_var(var, param_val);
                                state.variables.insert(param.name, (var, cl_type));
                            }
                        }

                        // Mark unpack methods so return statements bitcast f64→i64
                        let is_elem_unpack = !is_block_mode && method.kind == DatatypeMethodKind::Unpack;
                        state.dtype_unpack_ret_bitcast = is_elem_unpack;

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
}
