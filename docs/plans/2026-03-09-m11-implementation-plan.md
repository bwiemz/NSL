# M11: Model System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make model definitions compile and run — instantiation, field access, method dispatch, and forward calls.

**Architecture:** Models follow the same memory layout as structs (heap-allocated, fields at computed offsets). Model constructors are compiled functions that evaluate field initializer expressions. Model methods are standalone functions with an implicit `self` pointer as the first parameter. Method dispatch prepends `self` to the argument list. Forward dispatch allows calling a model instance as a function.

**Tech Stack:** Rust (nsl-semantic, nsl-codegen), Cranelift 0.116, existing nsl-runtime (no new runtime functions needed)

---

## Current State

**Already implemented (M1-M10):**
- **AST**: `ModelDef`, `ModelMember` (LayerDecl, Method) in `crates/nsl-ast/src/decl.rs`
- **Parser**: `parse_model_def_stmt()` in `crates/nsl-parser/src/decl.rs` — fully implemented
- **Semantic**: `check_model_def()` in `crates/nsl-semantic/src/checker.rs` — types model fields, methods, registers `Type::Model`
- **Semantic**: `check_member_access()` resolves model fields and methods (line 1147-1155)
- **Semantic**: Model instantiation returns model type (line 1095)
- **Codegen**: Models are **silently skipped** — `StmtKind::ModelDef(_)` produces no code (line 170-175 in stmt.rs)

**What M11 adds:**
1. Fix `self` typing in semantic checker (currently `Type::Unknown`, needs to be `Type::Model`)
2. Model memory layout collection (parallel to `collect_structs()`)
3. Model constructor compilation (evaluate field init expressions, allocate, store, return pointer)
4. Model method compilation (standalone functions with `self` as first param)
5. Model field access in codegen (`self.field` → load from offset)
6. Model field assignment in codegen (`self.field = val` → store at offset)
7. Model method dispatch (`model.method(args)` → call mangled function with self prepended)
8. Forward dispatch (`model(x)` → `model.forward(x)`)

**What's deferred (future milestones):**
- Param/Buffer as distinct runtime types (M12 — they need gradient tracking)
- Layer types written in NSL (requires model system to be working first — post-M11 or late M11)
- `@tie_weights`, `@checkpoint` decorators (M12+)
- Serialization/deserialization (M13+)
- `nsl fmt` formatter (separate task, not blocking)
- Weight sharing, lazy init, device placement (future)

---

## Key Files Reference

| File | Role |
|------|------|
| `crates/nsl-ast/src/decl.rs` | `ModelDef`, `ModelMember` AST types |
| `crates/nsl-parser/src/decl.rs` | `parse_model_def_stmt()` |
| `crates/nsl-semantic/src/checker.rs` | `check_model_def()`, `check_member_access()`, `check_call()` |
| `crates/nsl-semantic/src/types.rs` | `Type::Model { name, fields, methods }` |
| `crates/nsl-codegen/src/compiler.rs` | `collect_structs()`, `declare_user_functions()`, `compile_user_functions()`, `compile()` entry point |
| `crates/nsl-codegen/src/context.rs` | `StructLayout`, `StructField`, `FuncState` |
| `crates/nsl-codegen/src/expr.rs` | `compile_call()`, `compile_member_access()` |
| `crates/nsl-codegen/src/stmt.rs` | `compile_stmt()`, `compile_assign()` |
| `crates/nsl-codegen/src/types.rs` | `nsl_type_to_cl()` — type mapping |
| `crates/nsl-codegen/src/builtins.rs` | Runtime function declarations |

---

## How Structs Work (Pattern to Follow)

Models will follow the struct pattern exactly. Here's the struct lifecycle for reference:

1. **Layout** (`collect_structs()` in compiler.rs:352-389): Iterate StructDef stmts, compute field types/offsets/sizes, store in `self.struct_layouts` HashMap
2. **Declare** (`declare_user_functions()` in compiler.rs:418-431): Create `__nsl_struct_{name}` function signature (params = field types, returns pointer), register under struct name in `self.functions`
3. **Compile ctor** (`compile_struct_constructor()` in compiler.rs:571-613): Build Cranelift function: alloc memory via `nsl_alloc`, store each param at its field offset, return pointer
4. **Field access** (`compile_member_access()` in expr.rs:717-735): Match `Type::Struct`, look up layout, load from field offset
5. **Field assign** (`compile_assign()` in stmt.rs:275-310): Match `Type::Struct`, look up layout, store at field offset
6. **Call** (expr.rs:566-571): Struct name is in `self.functions`, generic call path handles it

**Key difference for models**: Struct constructors take field values directly as args. Model constructors take constructor params and evaluate field init expressions in that context. Model methods need to be compiled as separate functions.

---

### Task 1: Fix `self` Typing in Semantic Checker

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` (~line 427-429)
- Test: `cargo test -p nsl-semantic`

**Context:** Currently `check_model_def()` declares `self` with `Type::Unknown` (line 429). This prevents codegen from resolving `self.field` access because the model name is unknown. We need `self` typed as the model type so codegen can look up the correct layout.

**Step 1: Write a unit test**

Add to `crates/nsl-semantic/src/checker.rs` (in the `#[cfg(test)]` module at the bottom, or create one if needed):

```rust
#[test]
fn test_model_self_type() {
    // This test verifies that `self` in model methods is typed as the model type,
    // not Type::Unknown. We check this by verifying that `self.x` resolves correctly
    // and doesn't produce an error for a model with field `x`.
    let src = r#"
model Foo():
    x: int = 42

    fn get(self) -> int:
        return self.x
"#;
    let (_, diagnostics) = check_source(src);
    assert!(diagnostics.is_empty(), "Expected no errors, got: {:?}", diagnostics);
}
```

Note: If there's no `check_source` test helper, look at existing tests to see how they set up the checker. The pattern is: lex → parse → check → inspect diagnostics.

**Step 2: Run test to verify current behavior**

```
cargo test -p nsl-semantic test_model_self_type
```

Expected: The test should PASS even now (the checker is lenient about model member access — returns `Type::Unknown` on miss, line 1155). So this test alone isn't sufficient to verify the fix. The real test is in codegen (Task 5). However, we can add a stronger test:

```rust
#[test]
fn test_model_self_is_model_type() {
    let src = r#"
model Foo():
    x: int = 42

    fn get(self) -> int:
        return self.x
"#;
    let (type_map, diagnostics) = check_source(src);
    assert!(diagnostics.is_empty());
    // Find the `self.x` member access node and verify its type is Int (resolved from model fields)
    // This confirms self is typed as Model, not Unknown
}
```

Since we can't easily inspect node types in a unit test without knowing NodeIds, the primary verification will be the end-to-end test in Task 5.

**Step 3: Implement the fix**

In `crates/nsl-semantic/src/checker.rs`, in `check_model_def()`, find the self declaration (around line 427-429):

```rust
// BEFORE (line 427-429):
let self_sym = Symbol(self.interner.get_or_intern_static("self"));
self.declare_symbol(self_sym, Type::Unknown, fn_def.span, false, true);
```

Change to:

```rust
// AFTER:
let self_sym = Symbol(self.interner.get_or_intern_static("self"));
let self_type = Type::Model {
    name: model_def.name,
    fields: fields.clone(),
    methods: Vec::new(), // methods not fully collected yet, but fields are enough for self.field access
};
self.declare_symbol(self_sym, self_type, fn_def.span, false, true);
```

**Important:** This line is inside the `ModelMember::Method` arm of the member loop (line 416-461). At this point, `fields` has been partially populated (only fields declared before this method). This is fine for most models where fields come before methods. If methods reference fields declared after them, we'd need a two-pass approach — but that's a future concern.

**Step 4: Run all semantic tests**

```
cargo test -p nsl-semantic
```

Expected: All tests pass.

**Step 5: Verify with nsl check**

Create `examples/m11_model_basic.nsl`:

```nsl
model Counter(start: int):
    value: int = start

    fn get(self) -> int:
        return self.value

    fn increment(self):
        self.value = self.value + 1
```

Run: `cargo run -p nsl-cli -- check examples/m11_model_basic.nsl`

Expected: No errors. With `--dump-types`, `self` should show as `Model` type, not `Unknown`.

---

### Task 2: Model Layout Collection

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs` (add `collect_models()`, wire into compile pipeline)
- Test: `cargo test -p nsl-codegen`

**Context:** Models need memory layouts computed before constructors can be compiled. This mirrors `collect_structs()` (compiler.rs:352-389). We reuse the existing `struct_layouts` HashMap and `StructLayout`/`StructField` types — models have the same memory representation as structs.

**Step 1: Add `collect_models()` method**

In `crates/nsl-codegen/src/compiler.rs`, add after `collect_structs()` (~line 389):

```rust
// ── Pass 0.5c: Collect model definitions ─────────────────────────

pub fn collect_models(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
    for stmt in stmts {
        if let StmtKind::ModelDef(md) = &stmt.kind {
            let name = self.resolve_sym(md.name).to_string();
            let mut fields = Vec::new();
            let mut offset = 0usize;

            let model_type = self.node_type(stmt.id).clone();
            if let Type::Model { fields: type_fields, .. } = &model_type {
                for (field_sym, field_type) in type_fields {
                    let field_name = self.resolve_sym(*field_sym).to_string();
                    let cl_type = nsl_type_to_cl(field_type);
                    let size = cl_type.bytes() as usize;
                    let align = size.max(1);
                    offset = (offset + align - 1) & !(align - 1);
                    fields.push(StructField { name: field_name, cl_type, offset });
                    offset += size;
                }
            }

            self.struct_layouts.insert(
                name.clone(),
                StructLayout { name, fields, total_size: offset },
            );
        }
    }
    Ok(())
}
```

**Step 2: Wire into compile pipeline**

In `compile()` (line 709), `compile_module()` (line 730), and `compile_entry_module()` (line 768), add after `collect_structs()`:

```rust
compiler.collect_models(&ast.stmts)?;
```

There are 3 call sites — add to all of them.

**Step 3: Run tests**

```
cargo test --workspace
```

Expected: All existing tests pass. Models now have layouts but no constructors yet — no visible behavior change.

---

### Task 3: Model Constructor Declaration and Compilation

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs` (modify `declare_user_functions()`, add `compile_model_constructors()`)
- Test: end-to-end in Task 5

**Context:** Model constructors differ from struct constructors. Struct constructors take field values directly. Model constructors take constructor params and evaluate field init expressions. The constructor:
1. Accepts constructor params as function args
2. Evaluates each field's init expression (which may reference params)
3. Allocates memory for the model instance
4. Stores field values at computed offsets
5. Returns the pointer

**Step 1: Declare model constructor functions**

In `declare_user_functions_with_linkage()` (compiler.rs, after the struct constructor declaration block ~line 418-431), add model constructor declarations:

```rust
// Declare model constructors
for stmt in stmts {
    if let StmtKind::ModelDef(md) = &stmt.kind {
        let model_name = self.resolve_sym(md.name).to_string();
        // Skip if already registered (e.g., from struct with same name)
        if self.functions.contains_key(&model_name) {
            continue;
        }
        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        // Constructor params (from the model definition, NOT fields)
        for param in &md.params {
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
        sig.returns.push(AbiParam::new(pointer_type()));
        let func_id = self.module
            .declare_function(&format!("__nsl_model_{model_name}"), linkage, &sig)
            .map_err(|e| CodegenError::new(format!("failed to declare model ctor '{model_name}': {e}")))?;
        self.functions.insert(model_name, (func_id, sig));
    }
}
```

**Step 2: Compile model constructors**

Add a new method in `compiler.rs`:

```rust
fn compile_model_constructor(
    &mut self,
    model_def: &nsl_ast::decl::ModelDef,
) -> Result<(), CodegenError> {
    let model_name = self.resolve_sym(model_def.name).to_string();
    let (func_id, sig) = self.functions[&model_name].clone();
    let layout = self.struct_layouts[&model_name].clone();

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

        let mut state = FuncState::new();
        state.current_block = Some(entry);

        // Bind constructor params as variables
        for (i, param) in model_def.params.iter().enumerate() {
            let var = state.new_variable();
            let cl_type = sig.params[i].value_type;
            builder.declare_var(var, cl_type);
            builder.def_var(var, builder.block_params(entry)[i]);
            state.variables.insert(param.name, (var, cl_type));
        }

        // Allocate model instance
        let alloc_id = self.runtime_fns["nsl_alloc"].0;
        let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
        let size_val = builder.ins().iconst(cl_types::I64, layout.total_size.max(8) as i64);
        let call = builder.ins().call(alloc_ref, &[size_val]);
        let ptr = builder.inst_results(call)[0];

        // Compile field initializers and store at offsets
        let mut field_idx = 0;
        for member in &model_def.members {
            if let nsl_ast::decl::ModelMember::LayerDecl { init: Some(init_expr), .. } = member {
                if field_idx < layout.fields.len() {
                    let val = self.compile_expr(&mut builder, &mut state, init_expr)?;
                    builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        val,
                        ptr,
                        layout.fields[field_idx].offset as i32,
                    );
                }
                field_idx += 1;
            } else if let nsl_ast::decl::ModelMember::LayerDecl { init: None, .. } = member {
                // No initializer — store zero
                if field_idx < layout.fields.len() {
                    let zero = builder.ins().iconst(layout.fields[field_idx].cl_type, 0);
                    builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        zero,
                        ptr,
                        layout.fields[field_idx].offset as i32,
                    );
                }
                field_idx += 1;
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
```

**Step 3: Call model constructor compilation from `compile_user_functions()`**

In `compile_user_functions()` (~line 481-506), after struct constructors are compiled (line 502), add:

```rust
// Compile model constructors
let model_defs: Vec<_> = stmts
    .iter()
    .filter_map(|s| {
        if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
    })
    .collect();
for md in &model_defs {
    self.compile_model_constructor(md)?;
}
```

**Step 4: Run tests**

```
cargo test --workspace
```

Expected: All tests pass. Model constructors are now compiled but we can't test them yet (need field access and method dispatch first).

---

### Task 4: Model Method Declaration and Compilation

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs` (add to `declare_user_functions()`, add `compile_model_methods()`)
- Add: `model_methods` field to `Compiler` struct

**Context:** Model methods are compiled as standalone Cranelift functions with `self` (pointer to model instance) as the first parameter. We need:
1. A mapping from `(model_name, method_name)` to the mangled function name
2. Declaration of these functions in the module
3. Compilation of method bodies

**Step 1: Add `model_methods` field to Compiler**

In `crates/nsl-codegen/src/compiler.rs`, add to the `Compiler` struct (around line 38-60):

```rust
/// Maps model name → { method_name → mangled_fn_name }
pub model_methods: HashMap<String, HashMap<String, String>>,
```

Initialize in `Compiler::new()` (around line 86-101):

```rust
model_methods: HashMap::new(),
```

**Step 2: Declare model methods in `declare_user_functions_with_linkage()`**

After the model constructor declaration block (added in Task 3), add:

```rust
// Declare model methods
for stmt in stmts {
    if let StmtKind::ModelDef(md) = &stmt.kind {
        let model_name = self.resolve_sym(md.name).to_string();
        let mut method_map = HashMap::new();

        for member in &md.members {
            if let nsl_ast::decl::ModelMember::Method(fn_def) = member {
                let method_name = self.resolve_sym(fn_def.name).to_string();
                let mangled = format!("__nsl_model_{model_name}_{method_name}");

                let mut sig = self.module.make_signature();
                sig.call_conv = self.call_conv;
                // First param: self (pointer)
                sig.params.push(AbiParam::new(pointer_type()));
                // Remaining params (skip `self` in the AST param list)
                for param in &fn_def.params {
                    let name_str = self.interner.resolve(param.name.0).unwrap_or("");
                    if name_str == "self" {
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
                // Return type
                if let Some(ref ret_ann) = fn_def.return_type {
                    match &ret_ann.kind {
                        TypeExprKind::Named(sym) => {
                            let ret_type = self.resolve_type_name_to_cl(*sym);
                            if ret_type != cl_types::I8 || self.resolve_sym(*sym) != "void" {
                                sig.returns.push(AbiParam::new(ret_type));
                            }
                        }
                        _ => { sig.returns.push(AbiParam::new(cl_types::I64)); }
                    }
                } else {
                    // Check if method body returns something via type_map
                    // Default: assume returns I64 (pointer for tensors, etc.)
                    // Methods without explicit return type that don't return values
                    // will work because Cranelift allows missing returns for void functions.
                    // For now, add I64 return to match general convention.
                    sig.returns.push(AbiParam::new(cl_types::I64));
                }

                let func_id = self.module
                    .declare_function(&mangled, linkage, &sig)
                    .map_err(|e| CodegenError::new(format!("failed to declare model method '{mangled}': {e}")))?;
                self.functions.insert(mangled.clone(), (func_id, sig));
                method_map.insert(method_name, mangled);
            }
        }

        if !method_map.is_empty() {
            self.model_methods.insert(model_name, method_map);
        }
    }
}
```

**Step 3: Compile model methods**

Add to `crates/nsl-codegen/src/compiler.rs`:

```rust
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
            if let nsl_ast::decl::ModelMember::Method(fn_def) = member {
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
                    let entry = builder.create_block();
                    builder.append_block_params_for_function_params(entry);
                    builder.switch_to_block(entry);
                    builder.seal_block(entry);

                    let mut state = FuncState::new();
                    state.current_block = Some(entry);

                    // Bind parameters: first is `self`, rest are method params
                    let mut param_idx = 0;
                    for param in &fn_def.params {
                        let name_str = self.interner.resolve(param.name.0).unwrap_or("");
                        let var = state.new_variable();
                        let cl_type = sig.params[param_idx].value_type;
                        builder.declare_var(var, cl_type);
                        builder.def_var(var, builder.block_params(entry)[param_idx]);
                        state.variables.insert(param.name, (var, cl_type));
                        param_idx += 1;
                    }

                    // Compile method body
                    for s in &fn_def.body.stmts {
                        self.compile_stmt(&mut builder, &mut state, s)?;
                    }

                    // If function doesn't end with a return, add one
                    if let Some(block) = state.current_block {
                        if !crate::types::is_block_filled(&builder, block) {
                            if sig.returns.is_empty() {
                                builder.ins().return_(&[]);
                            } else {
                                let zero = builder.ins().iconst(cl_types::I64, 0);
                                builder.ins().return_(&[zero]);
                            }
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
```

**Step 4: Call from `compile_user_functions()`**

In `compile_user_functions()`, after model constructor compilation (added in Task 3):

```rust
self.compile_model_methods(stmts)?;
```

**Step 5: Run tests**

```
cargo test --workspace
```

Expected: All tests pass. Methods are now compiled but can't be called yet (dispatch comes in Task 6).

---

### Task 5: Model Field Access and Assignment in Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs` (~line 717, in `compile_member_access()`)
- Modify: `crates/nsl-codegen/src/stmt.rs` (~line 275, in `compile_assign()`)

**Context:** When codegen encounters `self.field` or `model_instance.field`, it needs to load/store from the model's memory layout. The pattern is identical to struct field access — match `Type::Model`, look up layout, use field offset. With Task 1's fix, `self` is now typed as `Type::Model` so the model name is available.

**Step 1: Add model field read access**

In `crates/nsl-codegen/src/expr.rs`, in `compile_member_access()`, after the struct field access block (after line 735 `}`), add:

```rust
if let Type::Model { name, .. } = &obj_type {
    let model_name = self.resolve_sym(*name).to_string();
    if let Some(layout) = self.struct_layouts.get(&model_name) {
        for field in &layout.fields {
            if field.name == member_name {
                let val = builder.ins().load(
                    field.cl_type,
                    MemFlags::trusted(),
                    obj_val,
                    field.offset as i32,
                );
                return Ok(val);
            }
        }
        return Err(CodegenError::new(format!(
            "model '{model_name}' has no field '{member_name}'"
        )));
    }
}
```

**Step 2: Add model field write access**

In `crates/nsl-codegen/src/stmt.rs`, in `compile_assign()`, in the `MemberAccess` arm (after the struct block ending at line 310), add a parallel block for models:

```rust
if let nsl_semantic::types::Type::Model { name, .. } = &obj_type {
    let model_name = self.resolve_sym(*name).to_string();
    if let Some(layout) = self.struct_layouts.get(&model_name) {
        for field in &layout.fields {
            if field.name == member_name {
                let final_val = if matches!(op, AssignOp::Assign) {
                    new_val
                } else {
                    let old_val = builder.ins().load(
                        field.cl_type,
                        cranelift_codegen::ir::MemFlags::trusted(),
                        obj_val,
                        field.offset as i32,
                    );
                    match op {
                        AssignOp::AddAssign => builder.ins().iadd(old_val, new_val),
                        AssignOp::SubAssign => builder.ins().isub(old_val, new_val),
                        AssignOp::MulAssign => builder.ins().imul(old_val, new_val),
                        AssignOp::DivAssign => builder.ins().sdiv(old_val, new_val),
                        _ => unreachable!(),
                    }
                };
                builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    final_val,
                    obj_val,
                    field.offset as i32,
                );
                return Ok(());
            }
        }
        return Err(CodegenError::new(format!(
            "model '{model_name}' has no field '{member_name}'"
        )));
    }
}
```

**Step 3: Run tests**

```
cargo test --workspace
```

Expected: All tests pass.

---

### Task 6: Model Method Dispatch

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs` (in `compile_call()`, around line 360-372)

**Context:** When `model_instance.method(args)` is called, codegen needs to:
1. Recognize the callee is a method call on a model type
2. Compile the object expression to get the self pointer
3. Look up the mangled method name from `self.model_methods`
4. Call the mangled function with `[self_ptr, ...compiled_args]`

**Step 1: Add model method dispatch**

In `crates/nsl-codegen/src/expr.rs`, in `compile_call()`, after the tensor method dispatch (line 369-371), add:

```rust
if let Type::Model { name, .. } = &obj_type {
    let model_name = self.resolve_sym(*name).to_string();
    return self.compile_model_method_call(
        builder, state, object, &model_name, &member_name, args,
    );
}
```

**Step 2: Implement `compile_model_method_call()`**

Add to `crates/nsl-codegen/src/expr.rs`:

```rust
fn compile_model_method_call(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    object: &Expr,
    model_name: &str,
    method_name: &str,
    args: &[nsl_ast::expr::Arg],
) -> Result<Value, CodegenError> {
    let self_val = self.compile_expr(builder, state, object)?;

    // Look up mangled method name
    let mangled = self.model_methods
        .get(model_name)
        .and_then(|methods| methods.get(method_name))
        .ok_or_else(|| CodegenError::new(format!(
            "model '{model_name}' has no method '{method_name}'"
        )))?
        .clone();

    // Compile arguments
    let mut arg_vals = vec![self_val];
    for arg in args {
        arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
    }

    self.compile_call_by_name(builder, &mangled, &arg_vals)
}
```

**Step 3: Run tests**

```
cargo test --workspace
```

Expected: All tests pass.

---

### Task 7: End-to-End Test — Simple Model with Int Fields

**Files:**
- Create: `examples/m11_model_basic.nsl`
- Create: `tests/expected/m11_model_basic.txt`

**Context:** This is the first end-to-end verification that models compile and run. We use a simple model with int fields and methods — no tensors yet. This tests: instantiation, field init from constructor params, method dispatch, self.field read, self.field write.

**Step 1: Write the test program**

Create `examples/m11_model_basic.nsl`:

```nsl
# M11 Basic Model Test

model Counter(start: int):
    value: int = start

    fn get(self) -> int:
        return self.value

    fn increment(self):
        self.value = self.value + 1

let c = Counter(10)
print(c.get())
c.increment()
c.increment()
c.increment()
print(c.get())
```

**Step 2: Run with nsl check first**

```
cargo run -p nsl-cli -- check examples/m11_model_basic.nsl
```

Expected: No errors.

**Step 3: Compile and run**

```
cargo run -p nsl-cli -- run examples/m11_model_basic.nsl
```

Expected output:
```
10
13
```

**Step 4: Debug if needed**

If compilation fails, use `--dump-ir` to inspect generated Cranelift IR:
```
cargo run -p nsl-cli -- build examples/m11_model_basic.nsl --dump-ir
```

Common issues to check:
- Model layout not found → Task 2 wiring issue
- Constructor param types wrong → Task 3 signature issue
- `self.value` access fails → Task 5 field access issue, or Task 1 self typing issue
- Method call fails → Task 6 dispatch issue, or method not declared

**Step 5: Save expected output**

Create `tests/expected/m11_model_basic.txt`:
```
10
13
```

---

### Task 8: End-to-End Test — Model with Tensor Fields

**Files:**
- Create: `examples/m11_model_tensor.nsl`
- Create: `tests/expected/m11_model_tensor.txt`

**Context:** This tests models with tensor fields, demonstrating the model system works for ML use cases. The model has a tensor weight matrix, and the forward method performs a matmul.

**Step 1: Write the test program**

Create `examples/m11_model_tensor.nsl`:

```nsl
# M11 Tensor Model Test

model SimpleLinear():
    weight: Tensor = ones([3, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

let m = SimpleLinear()
let x = ones([2, 3])
let y = m.forward(x)
print(y)
```

**Step 2: Run**

```
cargo run -p nsl-cli -- run examples/m11_model_tensor.nsl
```

Expected output: A 2×4 tensor where every element is 3.0 (since `ones([2,3]) @ ones([3,4])` = matrix of 3s).

```
[[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]
```

(Exact format depends on `nsl_tensor_print` — check existing tensor output format.)

**Step 3: Save expected output**

Create `tests/expected/m11_model_tensor.txt` with the actual output.

---

### Task 9: Forward Dispatch (Calling Model as Function)

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs` (in `compile_call()`)
- Create: `examples/m11_forward_dispatch.nsl`

**Context:** In the NSL spec, calling a model instance as a function should invoke its `forward` method: `model(x)` is equivalent to `model.forward(x)`. This is syntactic sugar that makes model composition natural: `let y = layer2(layer1(x))`.

**Step 1: Implement forward dispatch**

In `crates/nsl-codegen/src/expr.rs`, in `compile_call()`, after the function name extraction (around line 374-380), add a check for model-typed callees:

```rust
// Before the func_name extraction, check if callee is a model instance
// (direct Ident that resolves to a Model type)
if let ExprKind::Ident(sym) = &callee.kind {
    let callee_type = self.node_type(callee.id).clone();
    if let Type::Model { name, .. } = &callee_type {
        let model_name = self.resolve_sym(*name).to_string();
        // Forward dispatch: calling model instance invokes forward()
        return self.compile_model_method_call(
            builder, state, callee, &model_name, "forward", args,
        );
    }
}
```

**Important placement:** This must come BEFORE the `self.expr_as_func_name(callee)` call (line 374) because model instances are Ident expressions that would be misinterpreted as function names.

Actually, this is tricky. The `expr_as_func_name` for a model instance would return the variable name (e.g., "m"), and then the generic call path would fail because "m" isn't a declared function. So the check needs to happen before the generic path.

**Better approach:** In the generic call path fallback, after checking `self.functions.contains_key(&func_name)` (line 566), add:

```rust
// Forward dispatch: if callee is a variable holding a model instance, call its forward method
let callee_type = self.node_type(callee.id).clone();
if let Type::Model { name, .. } = &callee_type {
    let model_name = self.resolve_sym(*name).to_string();
    return self.compile_model_method_call(
        builder, state, callee, &model_name, "forward", args,
    );
}
```

Place this right before the `self.compile_indirect_call()` fallthrough (line 575).

**Step 2: Write test**

Create `examples/m11_forward_dispatch.nsl`:

```nsl
# M11 Forward Dispatch Test

model Doubler():
    factor: int = 2

    fn forward(self, x: int) -> int:
        return x * self.factor

let d = Doubler()

# Explicit method call
print(d.forward(5))

# Forward dispatch: calling model as function
print(d(5))
```

**Step 3: Run**

```
cargo run -p nsl-cli -- run examples/m11_forward_dispatch.nsl
```

Expected output:
```
10
10
```

---

### Task 10: Multi-Model Composition Test

**Files:**
- Create: `examples/m11_model_compose.nsl`

**Context:** This tests model composition — passing one model's output as input to another. This is the fundamental pattern for building neural networks from layers.

**Step 1: Write the test**

Create `examples/m11_model_compose.nsl`:

```nsl
# M11 Model Composition Test

model ScaleUp():
    weight: Tensor = ones([3, 5])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

model ScaleDown():
    weight: Tensor = ones([5, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

let up = ScaleUp()
let down = ScaleDown()

let x = ones([2, 3])
let h = up.forward(x)
let y = down.forward(h)
print(y)
```

**Step 2: Run**

```
cargo run -p nsl-cli -- run examples/m11_model_compose.nsl
```

Expected: A 2×2 tensor where each element is 15.0 (`ones[2,3] @ ones[3,5]` = 3s matrix `[2,5]`, then `[2,5] @ ones[5,2]` = 15s matrix `[2,2]`).

---

### Task 11: Regression Tests and Cleanup

**Files:**
- Run all existing examples
- Update memory files

**Step 1: Run all existing tests**

```
cargo test --workspace
```

Expected: All tests pass.

**Step 2: Run M1-M10 example programs**

Run each of these and verify output matches expected:

```
cargo run -p nsl-cli -- run examples/hello.nsl
cargo run -p nsl-cli -- run examples/functions.nsl
cargo run -p nsl-cli -- run examples/control_flow.nsl
cargo run -p nsl-cli -- run examples/structs.nsl
cargo run -p nsl-cli -- run examples/lists.nsl
cargo run -p nsl-cli -- run examples/m10_shape_check.nsl
```

Compare each output against `tests/expected/*.txt` baselines.

**Step 3: Run M11 examples**

```
cargo run -p nsl-cli -- run examples/m11_model_basic.nsl
cargo run -p nsl-cli -- run examples/m11_model_tensor.nsl
cargo run -p nsl-cli -- run examples/m11_forward_dispatch.nsl
cargo run -p nsl-cli -- run examples/m11_model_compose.nsl
```

All should produce correct output.

**Step 4: Verify shape checking still works with model methods**

```
cargo run -p nsl-cli -- check examples/m10_shape_errors.nsl
```

Should still report shape errors.

**Step 5: Update memory**

Update `MEMORY.md` to mark M11 as complete and note key implementation details.

---

## Verification Checklist

- [ ] `cargo test --workspace` — all tests pass
- [ ] `nsl run examples/m11_model_basic.nsl` — model with int fields works (instantiation, get, increment)
- [ ] `nsl run examples/m11_model_tensor.nsl` — model with tensor fields works (forward with matmul)
- [ ] `nsl run examples/m11_forward_dispatch.nsl` — calling model as function invokes forward
- [ ] `nsl run examples/m11_model_compose.nsl` — multi-model pipeline works
- [ ] `nsl check examples/m10_shape_errors.nsl` — M10 shape errors still caught
- [ ] All M1-M9 example programs produce correct output (regression test)
- [ ] `--dump-types` shows model types correctly
- [ ] `--dump-ir` shows model constructor and method IR

## File Summary

| Action | File | Description |
|--------|------|-------------|
| Modify | `crates/nsl-semantic/src/checker.rs` | Type `self` as Model in model methods |
| Modify | `crates/nsl-codegen/src/compiler.rs` | `collect_models()`, model ctor declaration/compilation, model method compilation, `model_methods` field |
| Modify | `crates/nsl-codegen/src/expr.rs` | Model field access, model method dispatch, forward dispatch |
| Modify | `crates/nsl-codegen/src/stmt.rs` | Model field assignment |
| Create | `examples/m11_model_basic.nsl` | Basic model test (int fields, methods) |
| Create | `examples/m11_model_tensor.nsl` | Tensor model test (matmul in forward) |
| Create | `examples/m11_forward_dispatch.nsl` | Forward dispatch test |
| Create | `examples/m11_model_compose.nsl` | Model composition test |
| Create | `tests/expected/m11_model_basic.txt` | Expected output baseline |
| Create | `tests/expected/m11_model_tensor.txt` | Expected output baseline |
