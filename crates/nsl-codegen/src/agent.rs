//! M56 Task 17 — Agent struct layout and method codegen.
//! M56 Task 18 — `@pipeline_agent` function lowering (Option A: direct dispatch).
//!
//! Spec §4.1 constraints:
//! - Agents compile to C-layout structs.
//! - Cache-line aligned: `alignment = 64`.
//! - Methods are Cranelift functions with `*mut AgentInstance` (`I64`) as
//!   their first argument.
//! - Field accesses inside methods (`self.field`) compile to
//!   `load(state_ptr + field_offset)` using the same `struct_layouts` map
//!   that models and structs use.
//!
//! Task 18 scope (Option A — spec §5.2):
//! - Detect `@pipeline_agent`-decorated `fn` at `compile_user_functions` time.
//! - Stack-allocate one agent-state buffer per listed agent at pipeline fn entry.
//! - Bind the lowercase receiver name (e.g. `drafter` for agent `Drafter`) to
//!   the stack-buffer address in `FuncState.variables`.
//! - Register the binding in `Compiler.models.agent_var_types` so that the
//!   Unknown-type dispatch path in `calls.rs` routes `drafter.draft(...)` to
//!   `__nsl_agent_Drafter_draft` without full type-checker support.
//!
//! Out of scope for Task 18 (v1):
//! - No `nsl_agent_scheduler_step` calls (deferred to v2 serve-block lowering).
//! - No `nsl_agent_pool_*` calls.
//! - No `init`/`reset` lifecycle; state buffers are zero-initialised by the OS.
//! - No `@auto_device_transfer` insertion (Task 19).
//! - No serve-block integration.

use std::collections::HashMap;

use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, StackSlotData, StackSlotKind,
    UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};

use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::{DeviceExpr, TypeExprKind};
use nsl_ast::Symbol;

use crate::compiler::Compiler;
use crate::context::{FuncState, StructField, StructLayout};
use crate::error::CodegenError;
use crate::types::pointer_type;

/// Cache-line alignment for agent state structs, per spec §4.1.
pub const AGENT_CACHE_LINE_ALIGNMENT: usize = 64;

/// Layout information for a compiled agent struct.
///
/// Returned by [`compute_agent_layout`] and also queryable after
/// `collect_agents` writes agent layouts into the compiler's
/// `struct_layouts` map under the agent name.
#[derive(Debug, Clone)]
pub struct AgentLayout {
    /// Byte offset of each field within the struct, keyed by field name.
    pub field_offsets: HashMap<String, usize>,
    /// Total allocated byte size, rounded up to cache-line alignment (64).
    pub total_size: usize,
    /// Struct alignment in bytes (always 64 per spec §4.1).
    pub alignment: usize,
}

/// Compute the layout of a single agent definition without needing a
/// `Compiler` context.  Useful in unit tests and for standalone analysis.
///
/// `resolve_field` maps each field's (`Symbol`, `TypeExpr`) pair to a
/// `(field_name: String, cl_type: cl_types::Type)` pair.  In the full
/// compiler this is `(self.resolve_sym(sym), self.resolve_type_name_to_cl(sym))`;
/// in tests a closure over an `Interner` suffices.
///
/// Field sizes and offsets mirror `collect_models` in
/// `compiler/collection.rs`, except the total size is padded to
/// `AGENT_CACHE_LINE_ALIGNMENT` (64 bytes) per spec §4.1.
pub fn compute_agent_layout<F>(agent: &AgentDef, resolve_field: F) -> AgentLayout
where
    F: Fn(nsl_ast::Symbol, &nsl_ast::types::TypeExpr) -> (String, cl_types::Type),
{
    let mut field_offsets = HashMap::new();
    let mut offset = 0usize;

    for member in &agent.members {
        if let AgentMember::FieldDecl {
            name, type_ann, ..
        } = member
        {
            let (field_name, cl_type) = resolve_field(*name, type_ann);
            let size = cl_type.bytes() as usize;
            let align = size.max(1);
            offset = (offset + align - 1) & !(align - 1);
            field_offsets.insert(field_name, offset);
            offset += size;
        }
    }

    // Pad total size to cache-line boundary (spec §4.1).
    let total_size = if offset == 0 {
        AGENT_CACHE_LINE_ALIGNMENT
    } else {
        (offset + AGENT_CACHE_LINE_ALIGNMENT - 1) & !(AGENT_CACHE_LINE_ALIGNMENT - 1)
    };

    AgentLayout {
        field_offsets,
        total_size,
        alignment: AGENT_CACHE_LINE_ALIGNMENT,
    }
}

// ── Compiler impl block ───────────────────────────────────────────────────────

impl Compiler<'_> {
    // ── Pass 0.5d: Collect agent definitions (layout) ────────────────

    /// M56 Task 17: compute C-layout structs for all `agent` declarations and
    /// register them in `self.types.struct_layouts` under the agent name.
    ///
    /// Spec §4.1: agents are cache-line aligned (64 bytes).  Field offset
    /// computation mirrors `collect_models` (`compiler/collection.rs`).
    pub fn collect_agents(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        for stmt in stmts {
            let agent = match &stmt.kind {
                StmtKind::AgentDef(a) => a,
                _ => continue,
            };

            let name = self.resolve_sym(agent.name).to_string();
            let mut fields: Vec<StructField> = Vec::new();
            let mut offset = 0usize;

            for member in &agent.members {
                if let AgentMember::FieldDecl {
                    name: field_sym,
                    type_ann,
                    ..
                } = member
                {
                    let field_name = self.resolve_sym(*field_sym).to_string();
                    let cl_type = match &type_ann.kind {
                        TypeExprKind::Named(sym) => self.resolve_type_name_to_cl(*sym),
                        _ => cl_types::I64,
                    };
                    let size = cl_type.bytes() as usize;
                    let align = size.max(1);
                    offset = (offset + align - 1) & !(align - 1);
                    fields.push(StructField {
                        name: field_name,
                        cl_type,
                        offset,
                    });
                    offset += size;
                }
            }

            // Spec §4.1: pad total size to cache-line boundary.
            let total_size = if offset == 0 {
                AGENT_CACHE_LINE_ALIGNMENT
            } else {
                (offset + AGENT_CACHE_LINE_ALIGNMENT - 1) & !(AGENT_CACHE_LINE_ALIGNMENT - 1)
            };

            self.types.struct_layouts.insert(
                name.clone(),
                StructLayout {
                    name: name.clone(),
                    fields,
                    total_size,
                    adapter_sidetable_offset: None,
                },
            );

            // M56 Task 19: scan @auto_device_transfer methods to populate
            // `agent_auto_device_params` so calls.rs can insert transfers.
            for member in &agent.members {
                if let AgentMember::Method(fn_def, decorators) = member {
                    let has_adt = decorators.iter().any(|d| {
                        d.name.len() == 1
                            && self.resolve_sym(d.name[0]) == "auto_device_transfer"
                    });
                    if !has_adt {
                        continue;
                    }
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mut device_transfers: Vec<(usize, i64)> = Vec::new();
                    let mut call_arg_idx = 0usize;
                    for param in &fn_def.params {
                        let pname = self.resolve_sym(param.name).to_string();
                        if pname == "self" {
                            continue;
                        }
                        if let Some(ref type_ann) = param.type_ann {
                            if let TypeExprKind::Tensor {
                                device: Some(dev), ..
                            } = &type_ann.kind
                            {
                                let target_device: i64 = match dev {
                                    DeviceExpr::Cpu => 0,
                                    DeviceExpr::Cuda(_) => 1,
                                    // Metal/ROCm/NPU not yet modelled in Task 19 v1.
                                    _ => {
                                        call_arg_idx += 1;
                                        continue;
                                    }
                                };
                                device_transfers.push((call_arg_idx, target_device));
                            }
                        }
                        call_arg_idx += 1;
                    }
                    if !device_transfers.is_empty() {
                        self.models
                            .agent_auto_device_params
                            .insert((name.clone(), method_name), device_transfers);
                    }
                }
            }
        }
        Ok(())
    }

    // ── Pass 1.5: Declare agent methods ─────────────────────────────

    /// M56 Task 17: declare Cranelift function IDs for every agent method
    /// found in top-level `agent` declarations.
    ///
    /// Mangling: `__nsl_agent_{AgentName}_{MethodName}`.
    /// Signature: `(state_ptr: I64, ...params) -> return_type`.
    ///
    /// Must be called after `collect_agents` (so `struct_layouts` is populated)
    /// and after `declare_runtime_functions` (so the function registry exists).
    pub fn declare_agent_methods(
        &mut self,
        stmts: &[Stmt],
        linkage: Linkage,
    ) -> Result<(), CodegenError> {
        let agent_defs: Vec<AgentDef> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::AgentDef(a) = &s.kind {
                    Some(a.clone())
                } else {
                    None
                }
            })
            .collect();

        for agent in &agent_defs {
            let agent_name = self.resolve_sym(agent.name).to_string();

            for member in &agent.members {
                if let AgentMember::Method(fn_def, _decos) = member {
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mangled = format!("__nsl_agent_{}_{}", agent_name, method_name);

                    let mut sig = self.module.make_signature();
                    sig.call_conv = self.call_conv;

                    // First Cranelift param: state pointer (I64, spec §4.1).
                    sig.params.push(AbiParam::new(pointer_type()));

                    // Remaining params: skip "self" in the AST param list.
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
                        sig.params.push(AbiParam::new(cl_type));
                    }

                    // Return type.
                    if let Some(ref ret_type) = fn_def.return_type {
                        match &ret_type.kind {
                            TypeExprKind::Named(sym) => {
                                let name = self.resolve_sym(*sym);
                                if name != "void" {
                                    sig.returns.push(AbiParam::new(
                                        self.resolve_type_name_to_cl(*sym),
                                    ));
                                }
                            }
                            _ => {
                                sig.returns.push(AbiParam::new(cl_types::I64));
                            }
                        }
                    }
                    // Implicit void return: no entry in sig.returns.

                    let func_id = self
                        .module
                        .declare_function(&mangled, linkage, &sig)
                        .map_err(|e| {
                            CodegenError::new(format!(
                                "failed to declare agent method '{mangled}': {e}"
                            ))
                        })?;
                    self.registry.functions.insert(mangled, (func_id, sig));
                }
            }
        }
        Ok(())
    }

    // ── Pass 2.5: Compile agent method bodies ─────────────────────────

    /// M56 Task 17: compile Cranelift function bodies for all agent methods.
    ///
    /// For each method:
    /// - Block param 0 is `state_ptr: I64` (the `*mut AgentInstance`).
    /// - `self` in the AST is bound to `state_ptr`.
    /// - `self.field` compiles to `load(state_ptr + field_offset)` via
    ///   the existing `compile_member_access` path, which checks
    ///   `struct_layouts` for any type that resolves to the agent name.
    ///
    /// This uses the same `current_method_model_name` mechanism that model
    /// methods use for synthesized SelfRef resolution.  The agent name is
    /// stored there during body compilation so `access.rs` can resolve
    /// `self.field` through `struct_layouts[agent_name]`.
    pub fn compile_agent_methods(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let agent_defs: Vec<AgentDef> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::AgentDef(a) = &s.kind {
                    Some(a.clone())
                } else {
                    None
                }
            })
            .collect();

        for agent in &agent_defs {
            let agent_name = self.resolve_sym(agent.name).to_string();

            for member in &agent.members {
                if let AgentMember::Method(fn_def, _decos) = member {
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mangled = format!("__nsl_agent_{}_{}", agent_name, method_name);

                    let (func_id, sig) = self
                        .registry
                        .functions
                        .get(&mangled)
                        .ok_or_else(|| {
                            CodegenError::new(format!(
                                "agent method '{}' not registered (declare_agent_methods must \
                                 run before compile_agent_methods)",
                                mangled
                            ))
                        })?
                        .clone();

                    let mut ctx = Context::for_function(Function::with_name_signature(
                        UserFuncName::user(0, self.next_func_index()),
                        sig.clone(),
                    ));
                    let mut fn_builder_ctx = FunctionBuilderContext::new();

                    {
                        let mut builder =
                            FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
                        let mut state = FuncState::new();

                        let entry = builder.create_block();
                        builder.append_block_params_for_function_params(entry);
                        builder.switch_to_block(entry);
                        builder.seal_block(entry);
                        state.current_block = Some(entry);

                        // Block param 0: state_ptr (spec §4.1 — *mut AgentInstance).
                        let state_ptr_val = builder.block_params(entry)[0];

                        // Find the "self" symbol from the AST params and bind it to
                        // state_ptr so `self.field` accesses resolve through
                        // `compile_member_access`.
                        let self_sym = fn_def
                            .params
                            .iter()
                            .find(|p| self.resolve_sym(p.name) == "self")
                            .map(|p| p.name)
                            .or_else(|| fn_def.params.first().map(|p| p.name))
                            .expect("agent method missing both explicit 'self' param and any params at all");
                        let self_var = state.new_variable();
                        builder.declare_var(self_var, pointer_type());
                        builder.def_var(self_var, state_ptr_val);
                        state.variables.insert(self_sym, (self_var, pointer_type()));

                        // Bind remaining params (skip "self" in the AST).
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

                        // Store agent name so `compile_member_access` can resolve
                        // synthesized SelfRef nodes via `struct_layouts[agent_name]`.
                        // This reuses the same `current_method_model_name` field that
                        // model methods use — both write the type name here so that
                        // `access.rs`'s Unknown-type fallback fires on bare `self`.
                        self.current_method_model_name = Some(agent_name.clone());

                        // Compile the method body.
                        let mut body_err: Option<CodegenError> = None;
                        for stmt in &fn_def.body.stmts {
                            if let Err(e) =
                                self.compile_stmt(&mut builder, &mut state, stmt)
                            {
                                body_err = Some(e);
                                break;
                            }
                        }
                        self.current_method_model_name = None;

                        if let Some(e) = body_err {
                            return Err(e);
                        }

                        // Add implicit return if body doesn't already end with one.
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
                        eprintln!(
                            "--- IR: agent method '{mangled}' ---\n{}",
                            ctx.func.display()
                        );
                    }

                    self.module
                        .define_function(func_id, &mut ctx)
                        .map_err(|e| {
                            CodegenError::new(format!(
                                "failed to define agent method '{mangled}': {e}"
                            ))
                        })?;
                }
            }
        }
        Ok(())
    }

    // ── Pass 2.6: Compile @pipeline_agent function bodies ────────────

    /// M56 Task 18 — spec §5.2, Option A (direct dispatch, no scheduler call).
    ///
    /// Compiles a `@pipeline_agent`-decorated function into a Cranelift function
    /// that calls agent method symbols (`__nsl_agent_{Name}_{method}`) directly,
    /// bypassing the runtime scheduler.  This is the v1 path; serve-block
    /// integration (Task 21) may use `nsl_agent_scheduler_step` in v2.
    ///
    /// At pipeline fn entry the codegen:
    /// 1. Stack-allocates one cache-line-aligned buffer per agent listed in the
    ///    `agents=[...]` decorator argument (buffer size = agent layout total_size).
    /// 2. Binds the lowercase agent name (e.g. `drafter` for `Drafter`) to the
    ///    buffer address in `FuncState.variables` and in
    ///    `Compiler.models.agent_var_types`.
    /// 3. Compiles the function body normally; `calls.rs` Unknown-type dispatch
    ///    picks up the `agent_var_types` entry and routes `drafter.draft(...)`
    ///    to `__nsl_agent_Drafter_draft(state_ptr, ...)`.
    pub fn compile_pipeline_agent_fn(
        &mut self,
        all_stmts: &[Stmt],
        fn_def: &nsl_ast::decl::FnDef,
    ) -> Result<(), CodegenError> {
        let fn_name = self.resolve_sym(fn_def.name).to_string();
        let (func_id, sig) = self
            .registry
            .functions
            .get(&fn_name)
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "@pipeline_agent fn '{fn_name}' not registered (declare_user_functions must \
                     run before compile_pipeline_agent_fn)"
                ))
            })?
            .clone();

        // Collect agent symbols from the @pipeline_agent decorator.
        let agent_syms: Vec<Symbol> = find_pipeline_agent_decorator_args(self, all_stmts, fn_def);

        // Build (agent_name: String, lowercase_binding: String, total_size: usize) triples.
        let agent_info: Vec<(String, String, usize)> = agent_syms
            .iter()
            .map(|sym| {
                let agent_name = self.resolve_sym(*sym).to_string();
                let total_size = self
                    .types
                    .struct_layouts
                    .get(&agent_name)
                    .map(|l| l.total_size)
                    .unwrap_or(AGENT_CACHE_LINE_ALIGNMENT);
                let lowercase = lowercase_first(&agent_name);
                (agent_name, lowercase, total_size)
            })
            .collect();

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

            // ── Step 1: bind fn params (mirrors compile_fn_def_named) ──────────
            for (i, param) in fn_def.params.iter().enumerate() {
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
                state.param_symbols.insert(param.name);
            }

            // ── Step 2: stack-allocate one buffer per agent ──────────────────
            for (agent_name, lowercase, total_size) in &agent_info {
                // Stack slot sized to agent layout total_size (always ≥ 64, cache-aligned).
                let slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    *total_size as u32,
                    // 6-bit log2 of 64-byte alignment: 2^6 = 64.
                    6,
                ));
                // Address of the slot — used as `state_ptr` for agent method calls.
                let state_ptr = builder.ins().stack_addr(pointer_type(), slot, 0);

                // Bind the lowercase agent name to the state pointer.
                // `interner.get(s)` does a read-only lookup; if the string was
                // parsed in the pipeline fn body (e.g. `drafter.draft(...)`), it
                // is already interned and `get` returns `Some`.
                let binding_sym_opt = self
                    .interner
                    .get(lowercase.as_str())
                    .map(Symbol);

                if let Some(binding_sym) = binding_sym_opt {
                    let var = state.new_variable();
                    builder.declare_var(var, pointer_type());
                    builder.def_var(var, state_ptr);
                    state.variables.insert(binding_sym, (var, pointer_type()));
                    // Register for Unknown-type dispatch in calls.rs.
                    self.models
                        .agent_var_types
                        .insert(binding_sym, agent_name.clone());
                } else {
                    // The lowercase binding never appears in the function body — this
                    // agent slot is still allocated but nothing dispatches through it.
                    // Log so it's diagnosable without a hard error (the slot doesn't hurt).
                    eprintln!(
                        "[nsl-codegen] @pipeline_agent fn '{fn_name}': agent '{agent_name}' \
                         binding '{lowercase}' not found in parsed source; \
                         agent method calls on this receiver will not dispatch."
                    );
                }
            }

            // ── Step 3: compile the pipeline body ────────────────────────────
            for stmt in &fn_def.body.stmts {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            // Implicit return (mirrors compile_fn_def_named's fallback).
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
            eprintln!(
                "--- IR: @pipeline_agent fn '{fn_name}' ---\n{}",
                ctx.func.display()
            );
        }

        // `agent_var_types` entries must be removed regardless of whether
        // `define_function` succeeds — entries are pipeline-fn-scoped, not
        // module-scoped. Cleaning up before propagating the error prevents stale
        // entries from leaking into subsequent pipeline fn compilations that use
        // different agent bindings.
        let define_result = self.module.define_function(func_id, &mut ctx).map_err(|e| {
            CodegenError::new(format!(
                "failed to define @pipeline_agent fn '{fn_name}': {e}"
            ))
        });
        for (_, lowercase, _) in &agent_info {
            if let Some(sym) = self.interner.get(lowercase.as_str()).map(Symbol) {
                self.models.agent_var_types.remove(&sym);
            }
        }
        define_result?;

        Ok(())
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Lowercase the first ASCII character of a string slice (agent name → binding).
///
/// `"Drafter"` → `"drafter"`, `"MyAgent"` → `"myAgent"`.
/// For non-ASCII first characters the string is returned unchanged.
fn lowercase_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Scan `all_stmts` for the `@pipeline_agent(agents=[...])` decorator on `fn_def`
/// and return the list of agent `Symbol`s from the `agents` keyword argument.
fn find_pipeline_agent_decorator_args(
    compiler: &Compiler<'_>,
    all_stmts: &[Stmt],
    fn_def: &nsl_ast::decl::FnDef,
) -> Vec<Symbol> {
    for stmt in all_stmts {
        if let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind {
            if let StmtKind::FnDef(fd) = &inner.kind {
                if fd.name == fn_def.name {
                    return extract_agents_from_decorators(compiler, decorators);
                }
            }
        }
    }
    Vec::new()
}

fn extract_agents_from_decorators(
    compiler: &Compiler<'_>,
    decorators: &[Decorator],
) -> Vec<Symbol> {
    for d in decorators {
        if d.name.len() != 1 {
            continue;
        }
        if compiler.resolve_sym(d.name[0]) != "pipeline_agent" {
            continue;
        }
        let Some(args) = &d.args else { continue };
        for arg in args {
            let is_agents = arg
                .name
                .map(|s| compiler.resolve_sym(s) == "agents")
                .unwrap_or(false);
            if !is_agents {
                continue;
            }
            if let ExprKind::ListLiteral(items) = &arg.value.kind {
                return items
                    .iter()
                    .filter_map(|item| match &item.kind {
                        ExprKind::Ident(s) => Some(*s),
                        _ => None,
                    })
                    .collect();
            }
        }
    }
    Vec::new()
}
