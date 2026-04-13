# WRGA Milestone B.2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `@adapter(type=lora|ia3|gatedlora, target=[...], rank=r, alpha=a)` produce runtime effect. After B.2.1, a compiled NSL model decorated with `@adapter` allocates real LoRA/IA³/GatedLoRA tensors, initializes them with strict init (LoRA B = zeros, A = Kaiming-normal), and applies the adapter in its forward pass using an unfused `y = x @ W + ((x @ A) @ B) * (alpha/rank)` rewrite.

**Architecture:** Six bite-sized tasks. Dim extraction walks `compiler.models` ModelDef registry (never the codegen type map). Constructor init uses a single-pointer side-table (one extra pointer slot on the model struct, indexed by adapter-tensor index) — zero callers outside adapter code understand the layout change. Forward rewrite is a full AST pass over model-method bodies; recognizes `BinaryOp { op: BinOp::MatMul, lhs, rhs: MemberAccess(self, W) }` where W is a raw `Tensor<...>` field. Four-build equivalence test with Build 4 (A=ones, B=ones, expected `y_base + k_in*r*scale`) as the load-bearing proof.

**Tech Stack:** Rust 2021, Cranelift IR, `cargo test -p nsl-codegen` / `-p nsl-cli`. Reference: `NSL-WRGA-Research.PDF` at repo root. Build on `main` at commit post-merge (has WGGO + WRGA B.2 + stack-overflow fix).

---

## Pre-flight

- [ ] **Confirm baseline on `main`**:
  Run: `git log -1 --oneline`
  Expected: message references B.2.1 spec (`docs(wrga): B.2.1 spec revisions ...`) or a later commit.

- [ ] **Record baseline test counts**:
  ```bash
  cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
  ```
  Floors: semantic ≥ 260, codegen lib ≥ 1102, codegen tests (all wrga integration + unit) green, nsl-cli e2e = 80, wrga_report_cli = 6.

- [ ] **Create the B.2.1 worktree**:
  ```bash
  cd c:/Users/bwiem/projects/NSL
  git worktree add ../NSL-wrga-b21 -b feat/wrga-milestone-b21
  cd ../NSL-wrga-b21
  cargo build --features cuda 2>&1 | tail -3
  ```
  Expected: clean build. All subsequent work happens in `c:/Users/bwiem/projects/NSL-wrga-b21`.

---

## File Structure

**Modify:**
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — Task 1 real dim extraction (`dims_for` replaces the `(0,0)` stub); Task 2 hook for emitting adapter init; add `resolve_target_to_model_field` helper.
- `crates/nsl-codegen/src/compiler/functions.rs` — Task 2 constructor hook between line 356 and line 395 (after user-field init, before `builder.ins().return_(&[ptr])`); Task 3/4 method-body AST rewrite pre-pass in `compile_model_methods`.
- `crates/nsl-codegen/src/expr/access.rs` — Task 2 field-access extension for synthesized adapter fields (route `self.lora_A_<site>` etc. through the side-table rather than the struct offset map).
- `crates/nsl-codegen/src/compiler/mod.rs` — Task 2 add `adapter_sidetable_slot_offset: Option<u32>` to `StructLayout` (or equivalent single-field extension) so constructor + access know where the side-table pointer lives.

**Create:**
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` — Task 3/4 AST rewrite pattern matcher + expression synthesis. Separate file for unit-testability in isolation.
- `crates/nsl-codegen/tests/wrga_adapter_runtime_equivalence.rs` — Task 5 runtime equivalence tests (Build 1/2/3/4).

**Test (extend):**
- `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` — Task 2 compile-time shape + init strategy tests extended to cover IA³ and GatedLoRA.
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — Task 1 inline unit test for dim resolution.
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` — Task 3/4 inline unit tests for pattern matcher.

---

## Task 1: Real dim extraction on `AdapterPlacement`

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_adapter_inject.rs`

**Goal:** Replace the `(0, 0)` stub in `dims_for` with real dims pulled from the target weight's declared type. Source is `compiler.models` (ModelDef registry), accessed via a reference passed into `run`. Never silently ship `(0, 0)` — emit a diagnostic and skip when resolution fails.

**Recon findings to use:**
- `AdapterPlacement.name` is the stable site name (e.g., `"blocks.6.wq"`). Target parameter string (e.g., `"m.w"`) comes from the decorator config — currently not stored directly on placement; must be threaded in or recovered from `name`.
- `compiler.models.model_field_types: HashMap<String, HashMap<String, String>>` maps model name → field name → type-annotation string (e.g., `"Tensor<[32, 16], f32>"`). This is the simplest lookup surface.
- Direct AST access via `compiler.models.model_method_bodies` + walking `ModelDef.members` gives structured `TypeExpr` but is more complex.
- `ExprKind::MemberAccess { object, member: Symbol }` — reference shape.
- `BinOp::MatMul` — matmul variant.

- [ ] **Step 1: Write the failing unit test**

Append at the bottom of `crates/nsl-codegen/src/wrga_adapter_inject.rs` (inside the existing `#[cfg(test)] mod tests`, or add one if missing):

```rust
#[cfg(test)]
mod dim_resolution_tests {
    use super::*;
    use std::collections::HashMap;

    /// Minimal field-types registry that mirrors `compiler.models.model_field_types`.
    fn mk_field_types() -> HashMap<String, HashMap<String, String>> {
        let mut inner = HashMap::new();
        inner.insert("w".to_string(), "Tensor<[32, 16], f32>".to_string());
        let mut outer = HashMap::new();
        outer.insert("Toy".to_string(), inner);
        outer
    }

    #[test]
    fn dims_for_resolves_from_field_types_registry() {
        let field_types = mk_field_types();
        // target_param "m.w" on a model named "Toy" should resolve to the
        // inner 2D tensor shape [out=32, in=16].
        let (input_dim, output_dim) = resolve_dims_for_target(
            "Toy",
            "w",
            &field_types,
        ).expect("dims must resolve");
        assert_eq!(input_dim, 16, "input_dim = k_in (second dim of Tensor<[out, in]>)");
        assert_eq!(output_dim, 32, "output_dim = d_out (first dim)");
    }

    #[test]
    fn dims_for_returns_none_on_missing_model() {
        let field_types = mk_field_types();
        assert!(
            resolve_dims_for_target("Unknown", "w", &field_types).is_none(),
            "unknown model must return None (caller emits diagnostic)",
        );
    }

    #[test]
    fn dims_for_returns_none_on_missing_field() {
        let field_types = mk_field_types();
        assert!(
            resolve_dims_for_target("Toy", "nonexistent", &field_types).is_none(),
        );
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen wrga_adapter_inject::dim_resolution_tests 2>&1 | tail -15`
Expected: compile error — `resolve_dims_for_target` does not exist.

- [ ] **Step 3: Add the `resolve_dims_for_target` helper**

In `crates/nsl-codegen/src/wrga_adapter_inject.rs`, add a free function near the top:

```rust
/// Parse a tensor type string like `"Tensor<[32, 16], f32>"` into its
/// `(output_dim, input_dim)` shape pair.  Returns None if the string is
/// not a 2-D tensor type annotation (1-D or higher-rank handled later).
///
/// Convention: NSL weights are declared as `Tensor<[out, in], dtype>` so
/// the first dim is the output channel count and the second is the input.
pub(crate) fn parse_tensor_2d_shape(type_str: &str) -> Option<(u32, u32)> {
    // Expected shape:  "Tensor<[<out>, <in>], <dtype>>"
    // Minimal parser — find the first "[" ... "]" slice and split on comma.
    let open = type_str.find('[')?;
    let close = type_str.find(']')?;
    if close <= open { return None; }
    let inner = &type_str[open + 1 .. close];
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 { return None; }
    let out: u32 = parts[0].parse().ok()?;
    let inp: u32 = parts[1].parse().ok()?;
    Some((out, inp))
}

/// Resolve `(input_dim, output_dim)` for an adapter target weight.
///
/// Inputs:
///   - `model_name`: the owning model (e.g., `"Toy"`).
///   - `field_name`: the weight field (e.g., `"w"`).
///   - `field_types`: `compiler.models.model_field_types` — the codegen's
///      declared-type registry populated during model collection.
///
/// Returns `Some((input_dim, output_dim))` when resolution succeeds;
/// `None` when the model/field is absent or the type isn't a 2-D tensor.
/// On `None`, the caller must emit a diagnostic — never ship `(0, 0)`
/// silently onto an `AdapterSite`.
pub(crate) fn resolve_dims_for_target(
    model_name: &str,
    field_name: &str,
    field_types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> Option<(u32, u32)> {
    let field_map = field_types.get(model_name)?;
    let type_str = field_map.get(field_name)?;
    let (out, inp) = parse_tensor_2d_shape(type_str)?;
    Some((inp, out))  // Return as (input_dim, output_dim)
}
```

- [ ] **Step 4: Run the test — expect pass**

Run: `cargo test -p nsl-codegen wrga_adapter_inject::dim_resolution_tests 2>&1 | tail -10`
Expected: all three tests pass.

- [ ] **Step 5: Wire `resolve_dims_for_target` into `run`**

The existing `run(plan: &mut WrgaPlan) -> AdapterInjectResult` signature in `wrga_adapter_inject.rs` does not take a compiler reference. Add a new variant that does:

```rust
/// B.2.1: Like `run`, but accepts the compiler's `model_field_types` so
/// synthesized AdapterSite dims are real (not `(0, 0)`).
pub fn run_with_model_types(
    plan: &mut crate::wrga::WrgaPlan,
    model_field_types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> AdapterInjectResult {
    let mut result = run(plan);
    // For each site, replace `input_dim`/`output_dim` using the registry.
    for site in result.sites.iter_mut() {
        // `site.target_param` is the decorator target (e.g., "m.w").
        // Split at '.' to get (model_var_name, field_name).  Note: the
        // "model_var_name" here is the let-binding name, not the model
        // TYPE name.  Resolve via plan context or the compiler — see
        // caller-side wiring below.
        let (model_var, field) = match site.target_param.split_once('.') {
            Some((a, b)) => (a, b),
            None => continue,  // malformed target; dims stay (0, 0) — caller diagnostic
        };
        // Heuristic: the model var and the model-type name are often the
        // same under type inference; for the B.2.1 test source where
        // `let m = Toy()` declares `m`, look up "Toy" in model_field_types
        // by iterating.  Full name resolution requires the compiler's
        // let-binding type map — addressed in Step 6.
        for (_model_name, fields) in model_field_types.iter() {
            if let Some((inp, out)) = resolve_dims_for_target(_model_name, field, model_field_types) {
                site.input_dim = inp;
                site.output_dim = out;
                break;
            }
            let _ = model_var;  // reserved; full resolution in Step 6
        }
    }
    result
}
```

> **Note:** The above is a minimal Step-5 wiring that relies on "one model type per compile with this field name." It is correct for the B.2.1 test source but not for multi-model compiles. Step 6 tightens resolution by passing the full compiler context.

- [ ] **Step 6: Tighten resolution using `compiler.models` full registry + let-binding type map**

Extend the signature of `run_with_model_types` (or add another wrapper) to accept enough context to resolve `target_param` precisely. The cleanest form:

```rust
pub fn run_with_compiler(
    plan: &mut crate::wrga::WrgaPlan,
    compiler: &crate::compiler::Compiler,
) -> AdapterInjectResult {
    let mut result = run(plan);
    for site in result.sites.iter_mut() {
        let target = &site.target_param;
        // Split "m.w" into ("m", "w").
        let (var_name, field_name) = match target.split_once('.') {
            Some(p) => p,
            None => {
                eprintln!("[wrga] @adapter target '{}' is not a `model.field` form; skipping",
                          target);
                continue;
            }
        };

        // Resolve the model type for `var_name`.  Two paths:
        //   (a) compiler.models.let_binding_types: HashMap<String, String>
        //       maps `"m"` → `"Toy"` (if such a map exists today).
        //   (b) Iterate model_field_types and pick the type whose field
        //       list contains `field_name`.  Ambiguous when multiple types
        //       declare the same field name — emit a diagnostic in that case.
        let field_types = &compiler.models.model_field_types;
        let mut candidates: Vec<&String> = field_types
            .iter()
            .filter(|(_, fields)| fields.contains_key(field_name))
            .map(|(name, _)| name)
            .collect();
        if candidates.is_empty() {
            eprintln!("[wrga] @adapter target '{}': field '{}' not found in any known model; \
                       skipping adapter materialisation (dims remain 0)",
                      target, field_name);
            continue;
        }
        if candidates.len() > 1 {
            eprintln!("[wrga] @adapter target '{}': field '{}' ambiguous across models {:?}; \
                       using first; follow-up: thread let-binding type map",
                      target, field_name, candidates);
        }
        let model_name = candidates.remove(0);
        match resolve_dims_for_target(model_name, field_name, field_types) {
            Some((inp, out)) => {
                site.input_dim = inp;
                site.output_dim = out;
            }
            None => {
                eprintln!("[wrga] @adapter target '{}': type string for '{}.{}' \
                           isn't a 2-D tensor; skipping",
                          target, model_name, field_name);
            }
        }
        let _ = var_name;  // used for later improvements
    }
    result
}
```

- [ ] **Step 7: Update the call site in `stmt.rs`**

In `crates/nsl-codegen/src/stmt.rs`, find the existing `crate::wrga_adapter_inject::run(&mut plan)` call (around line 158) and replace with:

```rust
let _inject = crate::wrga_adapter_inject::run_with_compiler(&mut plan, self);
```

`self` here is the `&Compiler` in scope inside `invoke_wrga_if_enabled`.

- [ ] **Step 8: Verify existing tests still pass**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime 2>&1 | tail -10
cargo test -p nsl-codegen --lib wrga_adapter_inject 2>&1 | tail -5
cargo build --features cuda 2>&1 | tail -3
```

The existing `adapter_inject_emits_lora_a_b_fields_with_expected_shapes` test asserts synthesized field names; it should still pass (this task changed dim values, not field names).

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-codegen/src/wrga_adapter_inject.rs crates/nsl-codegen/src/stmt.rs
git commit -m "feat(wrga): resolve real adapter dims from compiler.models registry"
```

---

## Task 2: Constructor init emission via single-slot side-table

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (extend `StructLayout` with optional side-table slot offset)
- Modify: `crates/nsl-codegen/src/compiler/functions.rs` (constructor hook post-user-init)
- Modify: `crates/nsl-codegen/src/expr/access.rs` (field-access extension for synthesized names)
- Modify: `crates/nsl-codegen/src/wrga_adapter_inject.rs` (helper to list sites for a given model)
- Modify: `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` (extend with LoRA / IA³ / GatedLoRA shape + init tests)

**Goal:** At model construction time, allocate the adapter tensors decided by `wrga_adapter_inject` and thread them through a single side-table pointer slot on the model struct. Field access for synthesized names (`lora_A_<site>`, `lora_B_<site>`, `ia3_scale_<site>`, `gate_<site>`) loads from the side-table instead of the struct-offset map.

- [ ] **Step 1: Write the failing integration test**

Append to `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`:

```rust
#[test]
fn lora_constructor_allocates_a_and_b_with_expected_shapes() {
    const SRC: &str = r#"
model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let m = Toy()
    let x: Tensor<[4, 8], f32> = zeros([4, 8])
    let _y = m.forward(x)
"#;
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    // Observational: after Task 1, synthesized fields on the placement
    // carry real dims; the constructor-emission side of Task 2 must
    // assert via compile success + Build 2 of the equivalence test.
    let lora_a_entry = plan.placements.iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .find(|n| n.starts_with("lora_A_"))
        .expect("lora_A field synthesized");
    assert!(lora_a_entry.contains("m_w"), "site id must reflect target");
}

#[test]
fn ia3_constructor_emits_scale_field() {
    const SRC: &str = r#"
model ToyIa3:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let m = ToyIa3()
    let x: Tensor<[4, 8], f32> = zeros([4, 8])
    let _y = m.forward(x)
"#;
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Ia3,
                targets: vec!["m.w".into()],
                rank: None,
                alpha: None,
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    assert!(
        plan.placements.iter()
            .flat_map(|p| p.synthesized_fields.iter())
            .any(|n| n.starts_with("ia3_scale_")),
        "ia3_scale field must be synthesized",
    );
}

#[test]
fn gatedlora_constructor_emits_a_b_and_gate() {
    const SRC: &str = r#"
model ToyGated:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let m = ToyGated()
    let x: Tensor<[4, 8], f32> = zeros([4, 8])
    let _y = m.forward(x)
"#;
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::GatedLora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    let fields: Vec<_> = plan.placements.iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .collect();
    assert!(fields.iter().any(|n| n.starts_with("lora_A_")));
    assert!(fields.iter().any(|n| n.starts_with("lora_B_")));
    assert!(fields.iter().any(|n| n.starts_with("gate_")));
}
```

- [ ] **Step 2: Run — expect pass for the existing LoRA observation assertions**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime 2>&1 | tail -15
```

These tests assert observational plan state that B.2 already produces — they should pass immediately. The actual constructor-emission work is proven by Task 5 Build 4; these are prerequisite sanity.

> If IA³ or GatedLoRA tests fail because the decorator kinds aren't yet wired through the inject pass for those variants, that's a B.2 gap — fix here by adding IA³/GatedLoRA arms to `wrga_adapter_inject::run`'s match on kind.

- [ ] **Step 3: Extend `StructLayout` with optional side-table slot offset**

In `crates/nsl-codegen/src/compiler/mod.rs`, find the `StructLayout` type (used by `compile_model_constructor`). Add:

```rust
pub struct StructLayout {
    pub fields: Vec<FieldLayout>,
    pub total_size: u32,
    // B.2.1: offset of the single adapter-sidetable pointer slot.
    // `None` when the model has no `@adapter` placements.  When `Some(o)`,
    // the slot at byte offset `o` holds an `i64` pointer to a heap-allocated
    // Vec<i64> of adapter-tensor pointers indexed by `AdapterSite.index`.
    pub adapter_sidetable_offset: Option<u32>,
}
```

Default-init the new field to `None` at every existing construction site (mechanical: the Rust compiler will flag each).

- [ ] **Step 4: Compute the side-table offset when models have adapters**

In the same file (or wherever `StructLayout::total_size` is computed for a `ModelDef`), extend layout computation. Pseudocode:

```rust
// After computing fields + total_size for ModelDef members:
let adapter_sidetable_offset = if compiler_has_adapters_for_model(model_name) {
    let off = align_to_8(layout.total_size);
    layout.total_size = off + 8;  // reserve 8 bytes for i64 pointer
    Some(off)
} else {
    None
};
layout.adapter_sidetable_offset = adapter_sidetable_offset;
```

The `compiler_has_adapters_for_model` check may be done later (at constructor codegen) — simpler for this step: ALWAYS allocate the slot when the compile has ANY `@adapter` decorators. One wasted 8 bytes per non-adapted model is negligible.

- [ ] **Step 5: Emit side-table allocation + init in `compile_model_constructor`**

In `crates/nsl-codegen/src/compiler/functions.rs` at `compile_model_constructor` (recon: line 201-396), insert adapter-init emission between line 356 (end of user-field loop) and line 395 (return).

Add a new helper below `compile_model_constructor`:

```rust
/// B.2.1: emit side-table allocation + per-site adapter tensor allocation
/// + init.  Called from compile_model_constructor when `layout.adapter_sidetable_offset`
/// is Some.
fn emit_adapter_sidetable(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    model_name: &str,
    struct_ptr: Value,
    sidetable_offset: u32,
) -> Result<(), CodegenError> {
    use cranelift_codegen::ir::InstBuilder;
    use cranelift_codegen::ir::MemFlags;

    // 1. Collect adapter sites matching this model (via the let-binding
    //    name `m` → model type `model_name` mapping).  For B.2.1's first
    //    cut, iterate all sites in compiler.wrga_adapter_sites and include
    //    those whose target_param's field portion is declared on this model.
    let sites: Vec<crate::wrga_adapter_inject::AdapterSite> = self
        .wrga_adapter_sites
        .iter()
        .filter(|s| site_targets_model(s, model_name, &self.models.model_field_types))
        .cloned()
        .collect();

    if sites.is_empty() {
        // Still write a null pointer into the slot so downstream field
        // access can rely on a defined value.
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().store(
            MemFlags::trusted(), zero, struct_ptr, sidetable_offset as i32,
        );
        return Ok(());
    }

    // 2. Allocate the side-table array: size = num_tensors * 8 bytes.
    //    One i64 pointer per synthesized field.
    let total_tensors: usize = sites.iter().map(|s| s.synthesized_fields.len()).sum();
    let table_bytes = (total_tensors * 8) as i64;
    let alloc_ref = self.runtime_fn_ref(builder, "nsl_alloc")?;
    let size_val = builder.ins().iconst(cl_types::I64, table_bytes);
    let alloc_call = builder.ins().call(alloc_ref, &[size_val]);
    let table_ptr = builder.inst_results(alloc_call)[0];

    // 3. For each site, emit init per InitKind and store pointer into the
    //    table at the correct index.
    let mut index: usize = 0;
    for site in &sites {
        for field_name in &site.synthesized_fields {
            let init_kind = site.init_strategies.iter()
                .find(|s| &s.field_name == field_name)
                .map(|s| s.kind)
                .unwrap_or(crate::wrga::InitKind::Zeros);
            let (dim_a, dim_b) = shape_for_field(site, field_name);

            let tensor_ptr = emit_init_tensor(
                self, builder, state, init_kind, dim_a, dim_b, site,
            )?;

            // Store at table_ptr + index*8.
            let byte_offset = (index * 8) as i32;
            builder.ins().store(
                MemFlags::trusted(), tensor_ptr, table_ptr, byte_offset,
            );
            index += 1;
        }
    }

    // 4. Store the table pointer into the struct's sidetable slot.
    builder.ins().store(
        MemFlags::trusted(), table_ptr, struct_ptr, sidetable_offset as i32,
    );
    Ok(())
}
```

Supporting helpers (same file or in `wrga_adapter_inject.rs`):

```rust
/// Decide the 2-D shape of a synthesized adapter field.
/// Returns (first_dim, second_dim).  For 1-D fields (ia3_scale, gate),
/// second_dim = 0 and the caller uses a 1-D shape list instead.
fn shape_for_field(site: &crate::wrga_adapter_inject::AdapterSite, field: &str) -> (u32, u32) {
    if field.starts_with("lora_A_") {
        // Shape [r, k_in]
        (site.rank as u32, site.input_dim)
    } else if field.starts_with("lora_B_") {
        // Shape [d_out, r]
        (site.output_dim, site.rank as u32)
    } else if field.starts_with("ia3_scale_") || field.starts_with("gate_") {
        // Shape [d_out] — 1-D.
        (site.output_dim, 0)
    } else {
        (0, 0)
    }
}

/// True when the site's target_param field exists on `model_name`.
/// Recon: compiler.models.model_field_types is HashMap<String, HashMap<String, String>>.
fn site_targets_model(
    site: &crate::wrga_adapter_inject::AdapterSite,
    model_name: &str,
    field_types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> bool {
    let field = site.target_param.rsplit('.').next().unwrap_or("");
    field_types.get(model_name).map(|m| m.contains_key(field)).unwrap_or(false)
}

/// Emit an init tensor via FFI calls.  Returns the Cranelift Value
/// holding the tensor pointer (i64).
fn emit_init_tensor(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    kind: crate::wrga::InitKind,
    dim_a: u32,
    dim_b: u32,
    site: &crate::wrga_adapter_inject::AdapterSite,
) -> Result<Value, CodegenError> {
    use cranelift_codegen::ir::InstBuilder;
    // Build a shape list: for 2-D, [dim_a, dim_b]; for 1-D, [dim_a].
    let shape_list = if dim_b == 0 {
        emit_shape_list_1d(compiler, builder, state, dim_a)?
    } else {
        emit_shape_list_2d(compiler, builder, state, dim_a, dim_b)?
    };

    match kind {
        crate::wrga::InitKind::Zeros => {
            let fn_ref = compiler.runtime_fn_ref(builder, "nsl_tensor_zeros")?;
            let call = builder.ins().call(fn_ref, &[shape_list]);
            Ok(builder.inst_results(call)[0])
        }
        crate::wrga::InitKind::Ones => {
            let fn_ref = compiler.runtime_fn_ref(builder, "nsl_tensor_ones")?;
            let call = builder.ins().call(fn_ref, &[shape_list]);
            Ok(builder.inst_results(call)[0])
        }
        crate::wrga::InitKind::KaimingUniform => {
            // Per spec: use randn * (1 / sqrt(fan_in)) — LeCun/Kaiming-normal.
            // For lora_A, fan_in = k_in = site.input_dim.
            let fan_in = site.input_dim.max(1) as f64;
            let scale = 1.0 / fan_in.sqrt();

            let randn_ref = compiler.runtime_fn_ref(builder, "nsl_tensor_randn")?;
            let randn_call = builder.ins().call(randn_ref, &[shape_list]);
            let base = builder.inst_results(randn_call)[0];

            // No `nsl_tensor_scalar_mul` exists.  Workaround: create a
            // [1]-shape tensor filled with `scale`, elementwise-multiply.
            let full_ref = compiler.runtime_fn_ref(builder, "nsl_tensor_full")?;
            let scalar_shape = emit_shape_list_1d(compiler, builder, state, 1)?;
            let scale_val = builder.ins().f64const(scale);
            let full_call = builder.ins().call(full_ref, &[scalar_shape, scale_val]);
            let scalar_tensor = builder.inst_results(full_call)[0];

            let mul_ref = compiler.runtime_fn_ref(builder, "nsl_tensor_mul")?;
            let mul_call = builder.ins().call(mul_ref, &[base, scalar_tensor]);
            Ok(builder.inst_results(mul_call)[0])
        }
    }
}

/// Shape list helpers are small FFI emitters for nsl_list_i64_push-style
/// list construction.  Implementation follows existing codegen patterns
/// in crates/nsl-codegen/src/expr/ for shape-list building.
fn emit_shape_list_1d(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    dim: u32,
) -> Result<Value, CodegenError> {
    // Existing codegen emits shape lists via `nsl_list_i64_new` +
    // `nsl_list_i64_push`.  If that pattern exists, mirror it.  If not,
    // mirror whatever `compile_tensor_literal`-like path uses.
    todo!("reuse the existing shape-list emission pattern in the codebase")
}

fn emit_shape_list_2d(
    compiler: &mut crate::compiler::Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    dim_a: u32,
    dim_b: u32,
) -> Result<Value, CodegenError> {
    todo!("reuse the existing shape-list emission pattern in the codebase")
}
```

> **Implementer note on `todo!()`:** The `emit_shape_list_*` helpers MUST be filled in before commit — no shipping `todo!()` in committed code. Grep for `nsl_list_i64_push` or similar in the existing codegen (e.g., `crates/nsl-codegen/src/expr/call.rs`) to find the canonical pattern. If the pattern is tied to a specific call-site (e.g., the existing `zeros([...])` call lowering), extract the list-building logic into a helper.

Now hook the emission into `compile_model_constructor`. Between line 356 and line 395, add:

```rust
if let Some(off) = layout.adapter_sidetable_offset {
    self.emit_adapter_sidetable(&mut builder, &mut state, &model_name, ptr, off)?;
}
```

- [ ] **Step 6: Extend field-access codegen for synthesized names**

In `crates/nsl-codegen/src/expr/access.rs` near line 99-106, BEFORE the existing field-offset lookup loop, add a branch that recognizes synthesized adapter names and routes them through the side-table:

```rust
if let Type::Model { name, .. } = &obj_type {
    let model_name = self.resolve_name(*name);
    let member_name = self.resolve_name(member);

    // B.2.1: Route synthesized adapter field accesses through the
    // side-table.  Names: "lora_A_*", "lora_B_*", "ia3_scale_*", "gate_*".
    if is_synthesized_adapter_field_name(&member_name) {
        if let Some(layout) = self.types.struct_layouts.get(&model_name) {
            if let Some(sidetable_off) = layout.adapter_sidetable_offset {
                // Compute the site's index in the per-model
                // synthesized-fields sequence.
                let index = self.adapter_field_index(&model_name, &member_name)
                    .ok_or_else(|| CodegenError::from(format!(
                        "synthesized adapter field {member_name} not found on {model_name}"
                    )))?;
                // Load side-table pointer from struct.
                let table_ptr = builder.ins().load(
                    cl_types::I64, MemFlags::trusted(), obj_val, sidetable_off as i32,
                );
                // Load tensor pointer from table[index].
                let byte_off = (index * 8) as i32;
                let tensor_ptr = builder.ins().load(
                    cl_types::I64, MemFlags::trusted(), table_ptr, byte_off,
                );
                return Ok(tensor_ptr);
            }
        }
    }

    // Existing user-field loop continues here.
    for field in &layout.fields {
        ...
    }
}

fn is_synthesized_adapter_field_name(name: &str) -> bool {
    name.starts_with("lora_A_")
        || name.starts_with("lora_B_")
        || name.starts_with("ia3_scale_")
        || name.starts_with("gate_")
}
```

Add `Compiler::adapter_field_index(model_name, field_name) -> Option<usize>` returning the linear index of that field across all `AdapterSite`s targeting this model (matches the insertion order used by `emit_adapter_sidetable`). Implement by iterating `self.wrga_adapter_sites` filtered by the same `site_targets_model` predicate, then flattening `synthesized_fields` and finding the match.

- [ ] **Step 7: Ensure IA³ and GatedLoRA arms exist in `wrga_adapter_inject::run`**

Check the existing `run` match on adapter kind. If only LoRA has a case filled in (B.2 may have scoped down to LoRA), extend:

```rust
match kind {
    AdapterKind::Lora => { /* lora_A, lora_B synthesized */ }
    AdapterKind::Ia3 => {
        let f = format!("ia3_scale_{}", site_id);
        placement.synthesized_fields.push(f.clone());
        placement.init_strategies.push(InitStrategy { field_name: f, kind: InitKind::Ones });
    }
    AdapterKind::GatedLora => {
        let a = format!("lora_A_{}", site_id);
        let b = format!("lora_B_{}", site_id);
        let g = format!("gate_{}", site_id);
        placement.synthesized_fields.extend([a.clone(), b.clone(), g.clone()]);
        placement.init_strategies.push(InitStrategy { field_name: a, kind: InitKind::KaimingUniform });
        placement.init_strategies.push(InitStrategy { field_name: b, kind: InitKind::Zeros });
        placement.init_strategies.push(InitStrategy { field_name: g, kind: InitKind::Zeros });
    }
}
```

- [ ] **Step 8: Run the three integration tests**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime 2>&1 | tail -20
cargo build --features cuda 2>&1 | tail -3
```

Expected: LoRA test passes (pre-existing); IA³ and GatedLoRA tests pass (new).

- [ ] **Step 9: Run the full codegen + nsl-cli e2e suite**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
```

No regressions.

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/compiler crates/nsl-codegen/src/expr crates/nsl-codegen/src/wrga_adapter_inject.rs crates/nsl-codegen/tests/wrga_adapter_runtime.rs
git commit -m "feat(wrga): materialise adapter tensors via model-struct side-table"
```

---

## Task 3: Forward-pass AST rewrite — LoRA

**Files:**
- Create: `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod wrga_adapter_rewrite;`)
- Modify: `crates/nsl-codegen/src/compiler/functions.rs::compile_model_methods` (invoke rewrite pre-pass)

**Goal:** Before lowering each model method body, walk the body's AST and rewrite every `BinaryOp { op: BinOp::MatMul, lhs: x, rhs: MemberAccess(self, W) }` where `W` is a raw `Tensor<...>` field targeted by an active LoRA adapter into the scaled expression `x @ self.W + ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale`.

- [ ] **Step 1: Write the failing unit test for the pattern matcher**

Create `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`:

```rust
//! WRGA Milestone B.2.1 Task 3/4: AST rewrite pass that applies adapters
//! to matmul sites in model method bodies.
//!
//! Triggers on:
//!   `BinaryOp { op: BinOp::MatMul, lhs: x, rhs: MemberAccess(SelfRef, W) }`
//! where `W` is a raw `Tensor<...>` field targeted by an active adapter.
//!
//! Does NOT rewrite:
//!   - `x |> self.W` (pipe syntax) — fallthrough
//!   - `self.submodel(x)` (call-site pattern) — explicit error with guidance
//!   - `otherModel.W @ x` (non-self receiver) — fallthrough
//!
//! Task 5's Build 4 is the load-bearing assertion that the rewrite fired
//! correctly.  See docs/superpowers/specs/2026-04-12-wrga-milestone-b21-design.md.

use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::operator::BinOp;
use nsl_ast::Symbol;

use crate::wrga_adapter_inject::AdapterSite;

pub struct RewriteContext<'a> {
    pub sites_for_model: Vec<&'a AdapterSite>,
    pub resolve_symbol: Box<dyn Fn(Symbol) -> String + 'a>,
}

/// Walk `expr`, rewriting any matmul sites that match an active adapter's
/// target.  Returns a new Expr tree; the input is NOT mutated in place
/// so the caller can fall back on the original if rewrite produces an
/// invalid tree.
pub fn rewrite_expr(expr: &Expr, ctx: &RewriteContext) -> Expr {
    // Recursive walk. For each sub-expression, descend; if the current
    // node matches the matmul-on-self-field pattern, substitute.
    // Leaf expressions are cloned unchanged.
    todo!("implement recursive walk + pattern match + substitution")
}

/// True when `expr` is `BinaryOp { op: MatMul, lhs: _, rhs: MemberAccess(self, W) }`
/// where `W` matches some active adapter's target field name.  Returns the
/// matching site and the lhs expression for use in the substitution.
pub fn match_lora_site<'a>(
    expr: &'a Expr,
    ctx: &'a RewriteContext,
) -> Option<(&'a AdapterSite, &'a Expr)> {
    let ExprKind::BinaryOp { left, op, right } = &expr.kind else { return None; };
    if !matches!(op, BinOp::MatMul) { return None; }
    let ExprKind::MemberAccess { object, member } = &right.kind else { return None; };
    // `object` must be `self`.
    let ExprKind::Ident(sym) = &object.kind else { return None; };
    let obj_name = (ctx.resolve_symbol)(*sym);
    if obj_name != "self" { return None; }
    let field_name = (ctx.resolve_symbol)(*member);
    // Match against the field portion of any site's target_param.
    for site in &ctx.sites_for_model {
        let Some(site_field) = site.target_param.rsplit('.').next() else { continue; };
        if site_field == field_name && matches!(site.kind, crate::AdapterKind::Lora) {
            return Some((*site, left));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Span;

    fn mk_span() -> Span { Span::default() }

    // Minimal site factory.
    fn mk_site(target_param: &str, rank: i64, alpha: i64) -> AdapterSite {
        AdapterSite {
            site_id: format!("{}__lora", target_param.replace('.', "_")),
            kind: crate::AdapterKind::Lora,
            target_param: target_param.to_string(),
            rank,
            alpha,
            synthesized_fields: vec![],
            input_dim: 0,
            output_dim: 0,
        }
    }

    #[test]
    fn matches_x_matmul_self_w() {
        // Construct an Expr for `x @ self.w` and assert match_lora_site fires.
        // (Full test body depends on Expr constructor surface; the structure
        //  is the same as the pattern docstring above.)
        todo!("construct Expr from AST constructor helpers and assert match succeeds")
    }

    #[test]
    fn does_not_match_other_model_w() {
        // `otherModel.w @ x` — no match (not self on the field side).
        todo!("assert match_lora_site returns None for non-self receiver")
    }

    #[test]
    fn does_not_match_when_op_is_not_matmul() {
        // `x + self.w` — no match (not MatMul).
        todo!("assert match_lora_site returns None for Add")
    }
}
```

**Note:** The three `todo!()` occurrences in the test bodies and `rewrite_expr` MUST be filled in before commit. The sketch above lays out the contract.

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-codegen wrga_adapter_rewrite::tests 2>&1 | tail -10
```

Expected: `todo!()` panics.

- [ ] **Step 3: Implement `match_lora_site` + the three unit tests**

Fill in `rewrite_expr`:

```rust
pub fn rewrite_expr(expr: &Expr, ctx: &RewriteContext) -> Expr {
    // Post-order walk: rewrite children first, then check current node.
    let new_kind = match &expr.kind {
        ExprKind::BinaryOp { left, op, right } => {
            // Recurse into children first.
            let new_left = rewrite_expr(left, ctx);
            let new_right = rewrite_expr(right, ctx);
            ExprKind::BinaryOp {
                left: Box::new(new_left),
                op: op.clone(),
                right: Box::new(new_right),
            }
        }
        ExprKind::MemberAccess { object, member } => ExprKind::MemberAccess {
            object: Box::new(rewrite_expr(object, ctx)),
            member: *member,
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(rewrite_expr(callee, ctx)),
            args: args.iter().map(|a| rewrite_expr(a, ctx)).collect(),
        },
        // All other kinds: structural clone.  Copy-paste every variant.
        // For the initial cut, start with the three above and fall back
        // to `expr.kind.clone()` on everything else.
        _ => expr.kind.clone(),
    };

    let rebuilt = Expr {
        kind: new_kind,
        span: expr.span,
        id: expr.id,
    };

    // Now check if the CURRENT node matches the adapter pattern.
    if let Some((site, lhs_expr)) = match_lora_site(&rebuilt, ctx) {
        return synthesize_lora_adapted(&rebuilt, lhs_expr, site, ctx);
    }
    rebuilt
}

/// Given the original `x @ self.w` expression + matched site, construct
/// `x @ self.w + ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale`.
fn synthesize_lora_adapted(
    original: &Expr,
    lhs: &Expr,
    site: &AdapterSite,
    _ctx: &RewriteContext,
) -> Expr {
    // Construct:
    //   let adapted = (x @ self.lora_A_<site>) @ self.lora_B_<site>;
    //   let scaled = adapted * scale;
    //   original + scaled
    let lora_a_name = format!("lora_A_{}", site.site_id);
    let lora_b_name = format!("lora_B_{}", site.site_id);
    let scale = (site.alpha as f64) / (site.rank as f64);

    // NOTE: Constructing Exprs requires access to an Interner to create
    // Symbols for field names.  The RewriteContext must expose an Interner
    // or a Symbol-creating closure.  Refine by threading `&mut Interner`.
    todo!("synthesize adapted expression using Interner for field-name Symbols")
}
```

Fill in the three unit tests by constructing AST fragments. If the required `Expr` constructor helpers don't exist, add a thin test-only helper module in the same file.

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p nsl-codegen wrga_adapter_rewrite::tests 2>&1 | tail -10
```

All three pattern-matcher tests pass.

- [ ] **Step 5: Wire the rewrite into `compile_model_methods`**

In `crates/nsl-codegen/src/compiler/functions.rs::compile_model_methods` around line 511 (where `self.compile_stmt(&mut builder, &mut state, stmt)` is called for each body statement), insert a pre-pass:

```rust
// B.2.1 Task 3: rewrite adapter matmul sites in method body AST.
let rewrite_ctx = crate::wrga_adapter_rewrite::RewriteContext {
    sites_for_model: self.wrga_adapter_sites.iter()
        .filter(|s| /* site targets this model */)
        .collect(),
    resolve_symbol: Box::new(|sym| {
        self.interner.resolve(sym.0).unwrap_or("").to_string()
    }),
};

for stmt in &fn_def.body.stmts {
    let rewritten = crate::wrga_adapter_rewrite::rewrite_stmt(stmt, &rewrite_ctx);
    self.compile_stmt(&mut builder, &mut state, &rewritten)?;
}
```

Add a `rewrite_stmt` helper to `wrga_adapter_rewrite.rs` that walks a `Stmt`'s expressions and calls `rewrite_expr` on each.

- [ ] **Step 6: Add submodel-target error in the inject pass**

In `wrga_adapter_inject::run` (or `run_with_compiler`), when a placement's `target_param` doesn't resolve to a raw `Tensor<...>` field, emit a clear error. Detect the case by checking whether `field_types[model_name][field_name]` starts with `"Tensor<"`. If not (e.g., starts with `"Linear"` or similar submodel type), emit:

```rust
eprintln!(
    "[wrga] @adapter(target=[\"{}\"]): targets a submodel, not a weight tensor; \
     adapt the submodel's inner weight directly (e.g., \"{}.weight\") \
     or use a submodel-level decorator (not yet supported)",
    site.target_param, site.target_param,
);
// Mark site as unresolved so downstream consumers skip it.
site.input_dim = 0;
site.output_dim = 0;
```

- [ ] **Step 7: Regression**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

No regressions.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/wrga_adapter_rewrite.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/compiler/functions.rs crates/nsl-codegen/src/wrga_adapter_inject.rs
git commit -m "feat(wrga): LoRA forward-pass AST rewrite (adapter applied at matmul sites)"
```

---

## Task 4: Forward-pass rewrites for IA³ and GatedLoRA

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` (extend with IA³ + GatedLoRA rewriters)
- Modify: inline unit tests in same file

**Goal:** Extend Task 3's AST rewrite to cover IA³ (`(x @ self.w) * self.ia3_scale_<site>`) and GatedLoRA (`x @ self.w + sigmoid(self.gate_<site>) * ((x @ self.lora_A) @ self.lora_B) * scale`). Include the step-0 analysis comment called out in the spec.

- [ ] **Step 1: Write the failing unit test for IA³ + GatedLoRA match + synthesis**

Append to the `#[cfg(test)] mod tests` in `wrga_adapter_rewrite.rs`:

```rust
#[test]
fn ia3_rewrite_multiplies_matmul_result_by_scale() {
    // Construct x @ self.w for an IA³-decorated model, run rewrite,
    // assert the result is (x @ self.w) * self.ia3_scale_<site>.
    todo!("construct IA3 site + AST, assert rewrite produces scaled form")
}

#[test]
fn gatedlora_rewrite_adds_sigmoid_gate_times_lora_contrib() {
    // Expected:
    //   x @ self.w + sigmoid(self.gate_<site>) * ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale
    //
    // STEP-0 INVARIANT NOTE (do not delete):
    //   `gate_<site>` is initialized to zeros.  `sigmoid(0) == 0.5`, NOT 0.
    //   The gate is HALF-OPEN at step 0.  Base-model equivalence at step 0
    //   depends ENTIRELY on `lora_B = 0`.  A refactor that changes B's init
    //   without simultaneously changing gate init will silently break the
    //   equivalence invariant.  Task 5 Build 2 is a sanity check; Task 5
    //   Build 4 is the load-bearing assertion.
    todo!("construct GatedLoRA site + AST, assert rewrite produces gated form")
}
```

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-codegen wrga_adapter_rewrite::tests 2>&1 | tail -10
```

- [ ] **Step 3: Extend `match_lora_site` → `match_adapter_site` for all kinds**

Rename for clarity and return `(&AdapterSite, &Expr)` across all kinds:

```rust
pub fn match_adapter_site<'a>(
    expr: &'a Expr,
    ctx: &'a RewriteContext,
) -> Option<(&'a AdapterSite, &'a Expr)> {
    // Same matcher as before — the `kind` check moves out of the matcher
    // and into the synthesizer (each kind has its own synthesis fn).
    let ExprKind::BinaryOp { left, op, right } = &expr.kind else { return None; };
    if !matches!(op, BinOp::MatMul) { return None; }
    let ExprKind::MemberAccess { object, member } = &right.kind else { return None; };
    let ExprKind::Ident(sym) = &object.kind else { return None; };
    let obj_name = (ctx.resolve_symbol)(*sym);
    if obj_name != "self" { return None; }
    let field_name = (ctx.resolve_symbol)(*member);
    for site in &ctx.sites_for_model {
        let Some(site_field) = site.target_param.rsplit('.').next() else { continue; };
        if site_field == field_name {
            return Some((*site, left));
        }
    }
    None
}
```

In `rewrite_expr`, dispatch on `site.kind`:

```rust
if let Some((site, lhs_expr)) = match_adapter_site(&rebuilt, ctx) {
    return match site.kind {
        crate::AdapterKind::Lora => synthesize_lora_adapted(&rebuilt, lhs_expr, site, ctx),
        crate::AdapterKind::Ia3 => synthesize_ia3_adapted(&rebuilt, site, ctx),
        crate::AdapterKind::GatedLora => synthesize_gatedlora_adapted(&rebuilt, lhs_expr, site, ctx),
    };
}
```

Add:

```rust
fn synthesize_ia3_adapted(
    original: &Expr,  // Already the `x @ self.w` node.
    site: &AdapterSite,
    ctx: &RewriteContext,
) -> Expr {
    // Result: (original) * self.ia3_scale_<site>
    //
    // Implementation: construct a BinaryOp { op: Mul, lhs: original,
    //   rhs: MemberAccess(self, ia3_scale_<site>) }.
    todo!("construct Expr: (x @ self.w) * self.ia3_scale_<site>")
}

fn synthesize_gatedlora_adapted(
    original: &Expr,  // Already the `x @ self.w` node.
    lhs: &Expr,       // `x` from the matched matmul.
    site: &AdapterSite,
    ctx: &RewriteContext,
) -> Expr {
    // Step-0 invariant: sigmoid(gate=0) == 0.5, NOT 0.  Equivalence with
    // base model at step 0 depends on lora_B = 0 zeroing the entire
    // adapter contribution.  If lora_B's init is changed away from zero,
    // this rewrite will break base-model equivalence silently — GatedLoRA
    // step-0 output will diverge from base by 0.5 * (x @ A) @ B * scale.
    //
    // Result: original + sigmoid(self.gate_<site>) * ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * scale
    todo!("construct the full gated-adapter expression tree")
}
```

Fill the `todo!()`s using the same AST-construction patterns as `synthesize_lora_adapted`.

- [ ] **Step 4: Run — expect pass**

```bash
cargo test -p nsl-codegen wrga_adapter_rewrite::tests 2>&1 | tail -10
```

- [ ] **Step 5: Regression**

```bash
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wrga_adapter_rewrite.rs
git commit -m "feat(wrga): IA3 + GatedLoRA forward rewrites (with step-0 invariant comment)"
```

---

## Task 5: Runtime equivalence test — four builds

**Files:**
- Create: `crates/nsl-codegen/tests/wrga_adapter_runtime_equivalence.rs`
- Possibly add: small test helpers to programmatically seed adapter tensors — see Step 3.

**Goal:** Build 1 reference (no adapter). Build 2 sanity (B=0 matches base). Build 3 divergence sanity (B nonzero diverges). **Build 4 load-bearing:** A=ones, B=ones, expected output = `y_base + k_in * r * scale` per element. Build 4 fails if forward rewrite never fired.

The test harness uses `nsl run` via `Command::cargo_bin("nsl")`, captures stdout, parses tensor-literal output format (recon: `tensor([[val, ...], ...])`).

- [ ] **Step 1: Write the test skeleton**

Create `crates/nsl-codegen/tests/wrga_adapter_runtime_equivalence.rs`:

```rust
//! WRGA B.2.1 Task 5: runtime equivalence tests for LoRA adapter rewrite.
//!
//! Build 1 = reference (no adapter).
//! Build 2 = adapter present, B=0 init (SANITY: matches Build 1).
//!   NOT LOAD-BEARING — if the rewrite never fires, this still passes.
//! Build 3 = adapter present, B seeded to nonzero (SANITY: output diverges).
//! Build 4 = adapter present, A=ones, B=ones, x=ones, W=zeros.
//!   LOAD-BEARING PROOF.  If rewrite fired correctly with scale=alpha/rank=1.0,
//!   every output element = y_base + k_in*r*scale = 0 + 8*2*1 = 16.0.
//!   Tolerance: 1e-5 f32.

use assert_cmd::prelude::*;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

const REFERENCE_SRC: &str = r#"
model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let m = Toy()
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let y = m.forward(x)
    print(y)
"#;

fn run_nsl(src_path: &std::path::Path, extra_args: &[&str]) -> String {
    let stdlib = std::env::var("NSL_STDLIB_PATH")
        .unwrap_or_else(|_| "c:/Users/bwiem/projects/NSL-wrga-b21/stdlib".to_string());
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .arg("run").arg("--source-ad")
        .args(extra_args)
        .arg(src_path);
    let output = cmd.output().expect("nsl run failed");
    assert!(
        output.status.success(),
        "nsl run exit={}\nstdout:{}\nstderr:{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Parse `tensor([[1.0, 2.0], [3.0, 4.0]])` → Vec<Vec<f32>>.
fn parse_tensor_2d(s: &str) -> Vec<Vec<f32>> {
    // Strip prefix `tensor(` and suffix `)`, then parse the nested list.
    let s = s.trim();
    let start = s.find('[').expect("no opening bracket");
    let end = s.rfind(']').expect("no closing bracket");
    let body = &s[start..=end];
    // body = "[[1.0, 2.0], [3.0, 4.0]]"
    let rows: Vec<&str> = body
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split("],")
        .map(|r| r.trim().trim_start_matches('[').trim_end_matches(']'))
        .collect();
    rows.into_iter()
        .map(|r| r.split(',').map(|v| v.trim().parse::<f32>().unwrap()).collect())
        .collect()
}

#[test]
fn build_1_reference_no_adapter_runs_cleanly() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, REFERENCE_SRC).unwrap();
    let stdout = run_nsl(&src_path, &[]);
    let tensor = parse_tensor_2d(&stdout);
    // y_base = ones @ zeros = zeros.
    for row in &tensor {
        for v in row {
            assert!(v.abs() < 1e-5, "reference output must be zero; got {v}");
        }
    }
}
```

> **Note on Build 2/3/4:** programmatically seeding `A`/`B` to specific values requires a mechanism to override the constructor's random init. Three options:
> - (a) Add a runtime helper `nsl_adapter_seed(model_ptr, field_name, pattern)` that overwrites adapter tensors after construction. Add the FFI, call from NSL test fixtures.
> - (b) Compile with a test-only `@adapter(init=...)` decorator extension that accepts preset patterns. Big change.
> - (c) Write a tiny NSL program that constructs the model, then manually overwrites adapter fields via member access (once Task 2's side-table field access is wired, `self.lora_A_m_w__lora = ones([2, 8])` should be a valid assignment).
>
> **Recommended: (c).** It composes with the existing NSL surface and requires no new FFIs. The test source becomes:
>
> ```nsl
> let m = Toy()
> // After constructor, override A and B with known patterns.
> m.lora_A_m_w__lora = ones([2, 8])   // Shape [r, k_in]
> m.lora_B_m_w__lora = ones([8, 2])   // Shape [d_out, r]
> let x: Tensor<[4, 8], f32> = ones([4, 8])
> let y = m.forward(x)
> print(y)
> ```
>
> This works ONLY if NSL permits writing to model fields via `model.field = expr`. Verify during implementation. If not, fall back to (a) — one new FFI is small.

- [ ] **Step 2: Implement Build 4 (the load-bearing test)**

Add to the same test file:

```rust
/// BUILD 4 — LOAD-BEARING PROOF.
///
/// Seeds A = ones([r, k_in]), B = ones([d_out, r]).  With x = ones([batch, k_in])
/// and W = zeros, the adapter contribution is:
///   (x @ A) @ B = ones([batch, r]) * k_in @ ones([d_out, r])
///               = ones([batch, d_out]) * k_in * r
/// Scaled by alpha/rank = 1.0, every output element should equal
///   y_base[i,j] + k_in * r * 1.0 = 0 + 8 * 2 = 16.0
///
/// IF THIS TEST FAILS by exactly 16.0 per element, the forward rewrite
/// did not fire (output = y_base = zeros).  If it fails by a different
/// amount, the scaling factor is wrong (check alpha/rank math).
#[test]
fn build_4_lora_rewrite_load_bearing_proof() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    // Seed A and B to known patterns.
    m.lora_A_m_w__lora = ones([2, 8])
    m.lora_B_m_w__lora = ones([8, 2])
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let y = m.forward(x)
    print(y)
"#;
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4.nsl");
    fs::write(&src_path, SRC).unwrap();

    let stdout = run_nsl(&src_path, &[]);
    let tensor = parse_tensor_2d(&stdout);
    // Expected: every element = 16.0 (= 8 * 2 * 1.0).
    let expected = 16.0_f32;
    for (i, row) in tensor.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let err = (v - expected).abs();
            assert!(
                err < 1e-5,
                "Build 4 failure at [{i},{j}]: expected {expected}, got {v}, diff {err}.\n\
                 Interpretation: if diff is exactly 16.0, the forward rewrite DID NOT FIRE.\n\
                 Output is y_base = zeros, not y_base + adapter.\n\
                 If diff is a different value, the scaling (alpha/rank) is wrong.",
                i = i, j = j, expected = expected, v = v, err = err,
            );
        }
    }
}
```

- [ ] **Step 3: Implement Build 2 and Build 3**

```rust
#[test]
fn build_2_sanity_b_zero_matches_base() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    // Don't seed A/B — use constructor defaults.  B is zeros by strict init,
    // so (x @ A) @ B = 0 regardless of A's random init.
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let y = m.forward(x)
    print(y)
"#;
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build2.nsl");
    fs::write(&src_path, SRC).unwrap();
    let stdout = run_nsl(&src_path, &[]);
    let tensor = parse_tensor_2d(&stdout);
    // SANITY: every element ≈ 0 (y_base + 0).
    // NOTE: This passes even if the rewrite didn't fire (base forward also
    // produces zeros).  Build 4 is the actual proof.
    for row in &tensor {
        for v in row {
            assert!(v.abs() < 1e-5, "Build 2 sanity failed; got {v}");
        }
    }
}

#[test]
fn build_3_sanity_b_nonzero_diverges() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    m.lora_B_m_w__lora = ones([8, 2])  // Only B; A stays Kaiming-normal.
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let y = m.forward(x)
    print(y)
"#;
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build3.nsl");
    fs::write(&src_path, SRC).unwrap();
    let stdout = run_nsl(&src_path, &[]);
    let tensor = parse_tensor_2d(&stdout);
    // Expect output is NOT identically zero.  A is Kaiming-normal ≈ N(0, 1/k_in),
    // so (x @ A) @ B won't be exactly zero with high probability.
    let any_nonzero = tensor.iter().flatten().any(|v| v.abs() > 1e-4);
    assert!(any_nonzero, "Build 3 expected nonzero output; got all zeros");
}
```

- [ ] **Step 4: Add `assert_cmd`, `predicates`, `tempfile` dev-deps if not already present**

Check `crates/nsl-codegen/Cargo.toml`'s `[dev-dependencies]`. If missing, add:

```toml
[dev-dependencies]
# ...existing...
assert_cmd = "2"
tempfile = "3"
```

- [ ] **Step 5: Run the full equivalence suite**

```bash
cargo build --bin nsl 2>&1 | tail -3
cargo test -p nsl-codegen --test wrga_adapter_runtime_equivalence 2>&1 | tail -15
```

Expected: all four tests pass. If Build 4 fails by exactly 16.0 per element, the forward rewrite didn't fire; investigate Task 3's `rewrite_expr` wiring into `compile_model_methods`.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/wrga_adapter_runtime_equivalence.rs crates/nsl-codegen/Cargo.toml
git commit -m "test(wrga): runtime equivalence test — Build 4 load-bearing LoRA rewrite proof"
```

---

## Task 6: Close-out

**Files:**
- Create (outside worktree): `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b21.md`
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md` (append pointer)
- **No code changes.**

- [ ] **Step 1: Run the full regression**

```bash
cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen flash_attention 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
cargo build --features cuda 2>&1 | tail -3
cargo build --release --features cuda 2>&1 | tail -3
cargo clippy -p nsl-codegen -p nsl-semantic -p nsl-cli --features cuda --all-targets 2>&1 | grep -E "^warning:" | grep -v "nsl-runtime" | head -20
```

Expected:
- Every existing test count ≥ baseline from Pre-flight.
- New test files: `wrga_adapter_runtime_equivalence` (4 tests) + Task 1 unit tests + Task 3/4 rewrite unit tests.
- Release build clean.
- No new clippy warnings in files this plan touched.

- [ ] **Step 2: Write the memory file**

Create `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b21.md`:

```markdown
---
name: WRGA Milestone B.2.1 — runtime LoRA/IA3/GatedLoRA complete
description: Constructor init via side-table, forward-pass AST rewrite, Build 4 load-bearing proof; B.3 (MMA epilogue) unblocked
type: project
---

## WRGA Milestone B.2.1 (landed 2026-04-12 on branch feat/wrga-milestone-b21)

**Per-task commits on branch (pre-merge):**
- <Task 1 SHA> feat(wrga): resolve real adapter dims from compiler.models registry
- <Task 2 SHA> feat(wrga): materialise adapter tensors via model-struct side-table
- <Task 3 SHA> feat(wrga): LoRA forward-pass AST rewrite (adapter applied at matmul sites)
- <Task 4 SHA> feat(wrga): IA3 + GatedLoRA forward rewrites (with step-0 invariant comment)
- <Task 5 SHA> test(wrga): runtime equivalence test — Build 4 load-bearing LoRA rewrite proof

**Where the runtime infrastructure lives:**
- `wrga_adapter_inject::resolve_dims_for_target` — parses `"Tensor<[out, in], dtype>"` from `compiler.models.model_field_types`. Never ships `(0, 0)` silently.
- Model struct: single `adapter_sidetable_offset: Option<u32>` field on `StructLayout`; when present, the struct's tail has an 8-byte pointer to a heap `Vec<i64>` of adapter-tensor pointers.
- `emit_adapter_sidetable` in `compile_model_constructor` allocates + inits the side-table after user-field init; LoRA A = randn * (1/sqrt(fan_in)), B = strict zeros, IA³ = ones, GatedLoRA gate = zeros.
- `expr/access.rs` routes synthesized adapter field names through `adapter_field_index` → side-table load chain; non-adapter fields untouched.
- `wrga_adapter_rewrite::rewrite_expr` is the AST pre-pass over model method bodies. Matches `BinaryOp { op: MatMul, lhs: x, rhs: MemberAccess(self, W) }` where `W` targets an active site; dispatches to LoRA/IA³/GatedLoRA synthesizers.
- Build 4 runtime equivalence test in `wrga_adapter_runtime_equivalence.rs` — the gating proof.

**What B.3 can now validate against:**
The MMA epilogue PTX kernel will fuse `x @ W + (x @ A) @ B * scale` into one kernel. B.2.1's unfused path is the reference: compile the same source with B.2.1 (3 kernels) and B.3 (1 fused kernel), assert outputs match within f32 tolerance.

**Known limitations (documented, deferred):**
- Submodel call-site patterns (`self.c_attn(h)` desugaring to `h @ self.c_attn.weight`) not handled. B.2.1 emits an explicit error on submodel targets.
- Pipe syntax (`x |> self.w`) not rewritten. Fallthrough with no adapter applied.
- Multi-model compiles with shared field names resolve to the first match (warning emitted).
```

Append to `MEMORY.md`:

```
## WRGA Milestone B.2.1 (2026-04-12)
- See [project_wrga_milestone_b21.md](project_wrga_milestone_b21.md) — runtime LoRA/IA3/GatedLoRA materialisation + forward rewrite complete on branch feat/wrga-milestone-b21; B.3 (MMA epilogue) unblocked
```

- [ ] **Step 3: Do NOT merge**

The controlling session performs the merge after subagent review. Report the branch status (per-task SHAs, test counts, known limitations) and stop.

---

## Final verification (do not skip before reporting DONE)

- [ ] `cargo test --workspace --features cuda` — accept Windows parallel-test flake; single-thread nsl-cli e2e re-run must pass.
- [ ] `cargo clippy --features cuda --all-targets -- -D warnings` on touched crates; no new warnings in B.2.1 code.
- [ ] `cargo build --release --features cuda` clean.
- [ ] Build 4 passes with the expected `16.0` per element. If it fails by exactly `16.0`, the forward rewrite regressed; investigate before declaring DONE.

---

## Out of scope (explicit)

- **FusionPlan MMA epilogue PTX in `backend_ptx.rs`** — this is B.3's core work.
- **Configurable adapter init via `@adapter(init=...)`** — follow-up if demand appears.
- **`@wrga_adapters` serialization section** — follow-up when cross-compile checkpoint rename is needed.
- **Submodel call-site rewriting (behavior 2 from the spec)** — B.2.2 or later.
- **Pipe-syntax and method-chained matmul rewriting** — follow-up.
- **Multi-model let-binding type resolution** — currently first-match; follow-up to thread the full let-binding type map.

---

## Risk log

1. **Side-table field-access routing coverage.** `expr/access.rs` must recognize synthesized names BEFORE the existing struct-offset lookup; otherwise the name falls through to "field not found" codegen error. Task 2 Step 6 wires this; if Build 4 fails during implementation, check this first.

2. **Constructor-emission shape-list helpers.** `emit_shape_list_1d` / `emit_shape_list_2d` are flagged as `todo!()` in the sketch; they MUST be filled using the existing NSL shape-list FFI pattern before commit. Grep for `nsl_list_i64_push` or look at how `zeros([...])` lowers today.

3. **Build 4 false pass from tiny numerics.** `k_in = 8, r = 2` gives a cleanly-integer expected value of 16.0. If the test accidentally uses different dims, the expected value changes — keep the dims aligned between source and assertion.

4. **NSL model-field assignment syntax.** Build 4's seeding uses `m.lora_A_m_w__lora = ones([...])`. If NSL doesn't permit this (model fields may be immutable post-construction), fall back to a runtime FFI `nsl_adapter_seed` — spec notes this alternative. Verify syntax support during Step 1.

5. **Task 3 AST rewrite completeness.** The walk must cover every `ExprKind` variant transparently or the tree is lossy. The sketch starts with three common variants and a catchall; before commit, extend the match to every variant used in real model methods (Conditional, Block, Call, etc.). Missing variants silently lose the rewrite on expressions that contain matmuls nested inside them.

6. **Step-0 GatedLoRA invariant.** If Task 4's synthesis changes gate init without updating LoRA B's init, Build 4 equivalent tests for GatedLoRA will silently regress. The step-0 comment in the synthesizer code and in Task 5's future GatedLoRA equivalence test is the only line of defense.

7. **Windows stack budget.** The AST rewrite adds recursion. The existing 16MB main-thread bootstrap handles today's workload; if Build 4 fails with a stack overflow, bump to 32MB.

8. **`nsl_tensor_mul` signature.** The Kaiming-normal init uses `nsl_tensor_mul(base, scalar_tensor)` where `scalar_tensor` is `[1]`-shape. Verify the FFI broadcasts correctly over a 2-D operand; if not, use `nsl_tensor_full([r, k_in], scale)` + `nsl_tensor_mul(base, filled)` instead.
