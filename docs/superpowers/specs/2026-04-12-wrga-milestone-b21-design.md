# WRGA Milestone B.2.1 — Runtime LoRA / IA³ / GatedLoRA

**Status:** Design approved 2026-04-12.

**Goal:** Make `@adapter(type=lora|ia3|gatedlora, target=[...], rank=r, alpha=a)` produce runtime effect. After B.2.1, a compiled NSL model decorated with `@adapter` actually allocates the adapter tensors, initializes them correctly, and applies the adapter in its forward pass. B.3's MMA epilogue can then validate the fused kernel against this unfused reference.

## Background

Milestone B.2 shipped the adapter observation surface only (commit `da600f5`). `wrga_adapter_inject::run` walks `WrgaPlan.placements`, picks site IDs + field names + init strategies, and writes them back onto the plan. But:

- `AdapterSite.input_dim` / `AdapterSite.output_dim` are stubbed at `(0, 0)`.
- The synthesized fields (`lora_A_<site>`, `lora_B_<site>`, `ia3_scale_<site>`, `gate_<site>`) are not allocated at model-construction time.
- The forward pass compiles as if no adapter were present.

Result: `@adapter` is a no-op at runtime. B.2.1 closes this gap.

## Scope decision — one milestone, all three adapter kinds

Single cohesive B.2.1 covering LoRA + IA³ + GatedLoRA end-to-end. Splitting into "init only" vs "init + rewrite" would ship a useless intermediate state (fields that exist but are never consumed). Splitting into "LoRA only" vs "IA³/GatedLoRA" was considered and rejected because decorator-parity keeps the test surface coherent and the IA³/GatedLoRA rewrites are short once the LoRA pattern is proven.

**Out of scope (future work):**
- FusionPlan MMA epilogue PTX (B.3).
- Configurable `@adapter(init=...)`.
- `@wrga_adapters` serialization section.
- Flipping `--wrga-fold-allocations` default on.
- Any rank > 1 heuristic adjustments beyond the decorator's explicit `rank=` / `alpha=` values.

## Architectural choices (locked during brainstorm)

1. **Constructor init emission: codegen-level field synthesis.** The synthesized fields (`lora_A_<site>` etc.) are NOT added to the `ModelDef` AST. Instead, `compile_model_constructor` appends extra allocations + init calls directly to its Cranelift IR output after the user-declared fields are initialized. The model struct's runtime layout is extended accordingly. This avoids rerunning the semantic checker on a mutated AST and keeps the ModelDef AST matching what the user wrote.

2. **Forward-pass rewrite: full AST-level rewrite.** `compile_model_methods` hosts a new sub-pass that walks each method body's AST, finds matmul sites (`Expr::MatMul { rhs: FieldAccess(self, W) }` or equivalent) where `W` matches a target param of an active adapter, and substitutes the scaled adapter expression. The rewrite produces `self.<synthesized_field>` access expressions which subsequent codegen resolves through the codegen-extended model layout (1). Canonical matmul forms are covered; unusual shapes (pipe syntax `x |> self.W`, method-chained) fall through unmodified in B.2.1 — documented limitation, follow-up if encountered.

3. **Init primitives (from research paper + NSL's existing FFIs):** The WRGA paper does not specify A's distribution; deferring to standard LoRA practice gives two equivalent options (peft's Kaiming-uniform vs original LoRA's scaled Normal). NSL has `nsl_tensor_randn` (N(0,1)) and `nsl_tensor_zeros`/`ones`/`full` but no bounded uniform. Cleanest path using existing primitives:
   - **LoRA A:** `nsl_tensor_randn([r, k_in]) * (1.0 / sqrt(k_in))` — two ops, LeCun/Kaiming-normal behavior, convergence-equivalent to peft's Kaiming-uniform.
   - **LoRA B:** `nsl_tensor_zeros([d_out, r])` — strict zeros, NEVER default init. Guarantees `(x @ A) @ B = 0` at step 0, so adapted model matches base model exactly on the first forward pass.
   - **IA³ scale:** `nsl_tensor_ones([d_out])` — identity scaling at step 0.
   - **GatedLoRA:** A/B as LoRA; `gate = nsl_tensor_zeros([d_out])` — gate closed at step 0.

4. **Scaling factor:** For LoRA forward, `scale = alpha as f32 / rank as f32`. The `alpha` field is already captured from `@adapter(alpha=N)` per B.2 Task 2a; default is `alpha = rank` → `scale = 1.0`.

## Task breakdown

### Task 1 — Real dim extraction on `AdapterSite`

Replace the `(0, 0)` stub in `wrga_adapter_inject::run::dims_for`. For each placement, look up the target weight's tensor type in the compiler's `type_map` (weights are declared as `Tensor<[out, in], f32>` on model fields) and set `input_dim = in`, `output_dim = out`. If the target isn't found (shouldn't happen for a placement that WRGA built), emit a diagnostic and skip that placement — don't ship `(0, 0)` into constructor/rewrite.

**Acceptance:** unit test in `wrga_adapter_inject.rs` proves a placement targeting `Tensor<[32, 16], f32>` yields `(input_dim=16, output_dim=32)`. Existing `adapter_inject_emits_lora_a_b_fields_with_expected_shapes` test stays green.

### Task 2 — Constructor init emission (all three adapter kinds)

In `crates/nsl-codegen/src/compiler/functions.rs::compile_model_constructor` (around line 356 per B.2 recon), after the existing user-field init loop and before the constructor returns, walk `compiler.wrga_adapter_sites` for sites matching the current model. For each matching site, emit:

**LoRA:**
```
// lora_A_<site>: shape [r, k_in], init = randn * (1.0 / sqrt(k_in))
let tmp_a = nsl_tensor_randn([r, k_in]);
let scale_a = 1.0 / sqrt(k_in as f64);
let lora_A = nsl_tensor_scalar_mul(tmp_a, scale_a);
// Store into model struct at the synthesized offset for lora_A_<site>.

// lora_B_<site>: shape [d_out, r], init = zeros
let lora_B = nsl_tensor_zeros([d_out, r]);
// Store.
```

**IA³:**
```
// ia3_scale_<site>: shape [d_out], init = ones
let ia3 = nsl_tensor_ones([d_out]);
// Store.
```

**GatedLoRA:** LoRA A/B as above plus `gate_<site> = nsl_tensor_zeros([d_out])`.

The model struct's runtime layout must be extended to reserve slots for the synthesized fields. The canonical way today is that `compile_model_constructor` calls `nsl_alloc(struct_size)` where `struct_size` is computed from `ModelDef.members`; extend this to include the synthesized fields' pointer slots (one `i64` per tensor). Field-access codegen (`self.<synthesized_field>`) must resolve against this extended layout.

**Acceptance:** integration test compiles a LoRA-decorated model, calls the constructor under `nsl run`, inspects the resulting model struct, and verifies the synthesized fields contain tensors of the expected shapes. `lora_B` is all-zeros; `ia3_scale` is all-ones.

### Task 3 — Forward-pass AST rewrite: LoRA

New sub-pass in `crates/nsl-codegen/src/compiler/functions.rs::compile_model_methods` (around line 511). For each method body in a model with at least one active LoRA `AdapterSite`, walk the method body's AST before lowering. For each `ExprKind::BinaryOp { op: MatMul, lhs: x, rhs: FieldAccess { receiver: self, field: w } }` where `w` matches a site's `target_param`, substitute:

```
x @ self.w  →  x @ self.w + ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * (alpha / rank)
```

The rewrite happens on a per-statement basis, mutating the AST's `Expr` tree in place (or constructing a fresh expr with equivalent span/NodeId). Scaling factor is emitted as a literal `f64` computed at compile time from the `AdapterSite.alpha` and `.rank`. The resulting expressions use ordinary `self.<field>` access which codegen resolves through Task 2's extended layout.

**Sharp edges:**
- Rewrite must ONLY trigger for the `self.<W>` form where `W` is the target. Rewriting `otherModel.w` or local-variable `w` is wrong. Gate on: receiver is `self`, field name matches `target_param` exactly after the dotted normalization used by site IDs.
- If a method body contains the same `self.w` matmul multiple times (e.g., inside a loop or used twice), each occurrence gets rewritten independently. That duplicates the adapter-application cost but is semantically correct.
- Pipe syntax (`x |> self.w`) and method-chained forms (`x.matmul(self.w)`) are not rewritten in B.2.1. Document this as a known limitation; if a user reports hitting it, follow-up with targeted extensions.

**Acceptance:** integration test's forward pass compiles with the rewrite applied; observable via debugger or via the LoRA-B=0 equivalence test (Task 5) — if rewrite didn't happen, B=0 is irrelevant and output differs.

### Task 4 — Forward-pass AST rewrite: IA³ and GatedLoRA

Extend Task 3's rewrite to handle the other two kinds.

**IA³** (element-wise per-output scaling after the matmul):
```
x @ self.w  →  (x @ self.w) * self.ia3_scale_<site>
```
(Broadcast over batch; `ia3_scale_<site>` has shape `[d_out]`.)

**GatedLoRA** (LoRA with a learned gate that modulates the adapter contribution):
```
x @ self.w  →  x @ self.w + sigmoid(self.gate_<site>) * ((x @ self.lora_A_<site>) @ self.lora_B_<site>) * (alpha / rank)
```
`sigmoid` applied element-wise to the gate; broadcast over batch; at step 0, gate is zeros, so `sigmoid(0) = 0.5`, but since B is also zeros, the adapter contribution stays zero. (Equivalence with base model still holds at step 0 because the `B=0` factor dominates.)

**Acceptance:** the existing `adapter_inject_emits_*_fields_with_expected_shapes` tests extended to cover IA³ and GatedLoRA compile + init; a runtime-output equivalence test for each (IA³ with scale=1 matches base; GatedLoRA with B=0 matches base).

### Task 5 — Runtime equivalence test

Load-bearing test. Compile an NSL program with a LoRA-decorated model, run it via `nsl run`, read the output, compare against the same model compiled without `@adapter`.

```
model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

# Build 1: no adapter — reference output.
# Build 2: same source + WrgaInputs with @adapter(type=lora, target=["m.w"], rank=2, alpha=2)
#   Expect: output == reference (within 1e-5 f32 tolerance) because B = 0 at step 0.
# Build 3: same as Build 2 but after programmatically seeding B to a nonzero pattern.
#   Expect: output diverges from reference by the analytically computed ((x @ A) @ B) * scale term.
```

The test harness can be either a new end-to-end CLI test (spawn `nsl run` twice, diff stdout) OR an in-process test that compiles + loads the shared library and invokes the forward directly. The CLI form is simpler and matches existing `nsl-cli/tests/e2e.rs` patterns.

**Tolerance:** 1e-5 for f32; 1e-9 for f64. Use element-wise absolute + relative combined.

**Acceptance:** both checks pass. If B=0 equivalence fails, the forward rewrite is wrong (most likely bug). If B=nonzero-pattern divergence doesn't match analytics, the scaling factor is wrong.

### Task 6 — Close-out

1. Create `project_wrga_milestone_b21.md` in the memory dir documenting the per-task commits + what B.3 can now validate against.
2. Append a pointer line to `MEMORY.md`.
3. Run full regression: semantic, codegen lib + tests, flash_attention, nsl-cli e2e (single-threaded), wrga_report_cli.
4. Do NOT merge — controlling session merges after subagent review.

## File structure

**Modify:**
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — Task 1 real dims + optionally helpers shared by the constructor + rewrite passes.
- `crates/nsl-codegen/src/compiler/functions.rs` — Task 2 constructor init hook (post-user-field init); Task 3 + 4 forward-pass AST rewrite entry point.
- `crates/nsl-codegen/src/compiler/model.rs` (or wherever model struct layout is computed) — extend layout to include synthesized field slots.

**Create:**
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` — AST rewrite helpers (pattern matcher + expression synthesis). Separate file because the rewrite logic is nontrivial and benefits from unit tests in isolation.
- `crates/nsl-codegen/tests/wrga_adapter_runtime_equivalence.rs` — Task 5 test.

**Test:**
- `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` (extend with Task 2 shape + init tests).
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` (inline unit test for Task 1 dim extraction).
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` (inline unit tests for pattern matcher and expression synthesis).

## Risk register

1. **Model struct layout extension invasiveness.** Extending `ModelDef`-derived layouts to include codegen-synthesized fields may touch multiple call sites (field offset resolution, reflection/serialization paths, source-AD model observation). If layout touching spreads beyond a contained hook, the constructor-init task balloons. Mitigation: scope the layout extension to a single private helper that Task 2 owns; if more than three callers need updating, pause and reassess.

2. **AST rewrite pattern matching correctness.** Recognizing `self.<target_param>` at matmul sites reliably requires understanding NSL's ExprKind variants around dot access + binary ops. Mitigation: pattern match against a small set of canonical shapes in B.2.1; anything non-matching falls through unmodified, and the LoRA-B=0 equivalence test catches missed rewrites (because without rewrite, B=0 doesn't help — the adapter was never applied but B wasn't consulted either, so the test may actually pass for the wrong reason).

3. **LoRA-B=0 equivalence test false pass.** If forward rewrite never fires, output equals base trivially (adapter isn't applied at all). Mitigation: add a secondary assertion — the synthesized fields MUST exist in the runtime model struct (observable via a debug helper or via poking the model's memory layout). If fields exist AND B=0 equivalence holds, rewrite is working. If fields exist AND B=nonzero-pattern divergence matches analytics, scaling is correct.

4. **Source-AD interaction.** Once LoRA A/B become real fields, the source-AD extractor will see them as trainable params. This is the intended behavior but may cause surprises if existing tests assume a fixed set of model params. Mitigation: run the full nsl-codegen test suite after Task 2; any test that enumerates "model fields" needs to account for adapter-site synthesized fields.

5. **`nsl_tensor_scalar_mul` availability.** The init emission uses `randn * (1/sqrt(fan_in))`. If no `scalar_mul` FFI exists with f64 scalar, use `nsl_tensor_full` to materialize a filled tensor and elementwise-multiply, or add the scalar_mul FFI. Recon-during-implementation decision.

6. **Windows stack budget after Task 3.** The post-merge stack-overflow fix (16MB main thread) holds today, but the AST rewrite adds additional recursion depth. If tests start failing again, bump the stack size or move the rewrite to iterative form.
