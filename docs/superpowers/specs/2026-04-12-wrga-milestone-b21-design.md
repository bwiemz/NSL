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

1. **Constructor init emission: codegen-level field synthesis via single-slot side table.** The synthesized fields (`lora_A_<site>` etc.) are NOT added to the `ModelDef` AST. The model struct layout is NOT extended per-adapter-field. Instead, a single extra pointer slot (`adapter_tensors: *mut i64`) is appended to the model struct. At construction time, `compile_model_constructor` allocates a `Vec<i64>` (or equivalent boxed pointer array) sized to `num_adapter_tensors_for_this_model`, stores pointers to each initialized adapter tensor at deterministic indices, and stores the array's base pointer into the side-table slot. Field access (`self.lora_A_<site>`) compiles to: load base pointer from struct, index by the site's adapter-tensor index, deref. This adds one pointer-chase per adapter access at forward time — negligible, and B.3's fused kernel eliminates the chase anyway. Critically, this approach means **zero callers outside the new adapter code need to understand the layout change**; layout computation for `ModelDef` members is untouched, reflection/serialization paths see one new opaque pointer slot, and source-AD can treat the side-table as a GPU-opaque blob.

2. **Forward-pass rewrite: full AST-level rewrite.** `compile_model_methods` hosts a new sub-pass that walks each method body's AST, finds matmul sites (`Expr::MatMul { rhs: FieldAccess(self, W) }` or equivalent) where `W` matches a target param of an active adapter, and substitutes the scaled adapter expression. The rewrite produces `self.<synthesized_field>` access expressions which subsequent codegen resolves through the codegen-extended model layout (1). Canonical matmul forms are covered; unusual shapes (pipe syntax `x |> self.W`, method-chained) fall through unmodified in B.2.1 — documented limitation, follow-up if encountered.

3. **Init primitives (from research paper + NSL's existing FFIs):** The WRGA paper does not specify A's distribution; deferring to standard LoRA practice gives two equivalent options (peft's Kaiming-uniform vs original LoRA's scaled Normal). NSL has `nsl_tensor_randn` (N(0,1)) and `nsl_tensor_zeros`/`ones`/`full` but no bounded uniform. Cleanest path using existing primitives:
   - **LoRA A:** `nsl_tensor_randn([r, k_in]) * (1.0 / sqrt(k_in))` — two ops, LeCun/Kaiming-normal behavior, convergence-equivalent to peft's Kaiming-uniform.
   - **LoRA B:** `nsl_tensor_zeros([d_out, r])` — strict zeros, NEVER default init. Guarantees `(x @ A) @ B = 0` at step 0, so adapted model matches base model exactly on the first forward pass.
   - **IA³ scale:** `nsl_tensor_ones([d_out])` — identity scaling at step 0.
   - **GatedLoRA:** A/B as LoRA; `gate = nsl_tensor_zeros([d_out])` — gate closed at step 0.

4. **Scaling factor:** For LoRA forward, `scale = alpha as f32 / rank as f32`. The `alpha` field is already captured from `@adapter(alpha=N)` per B.2 Task 2a; default is `alpha = rank` → `scale = 1.0`.

## Task breakdown

### Task 1 — Real dim extraction on `AdapterSite`

Replace the `(0, 0)` stub in `wrga_adapter_inject::run::dims_for`.

**Source of truth for dims:** Model field types are resolved during semantic analysis and stored on `ModelDef.members[i].type_ann` (for `LayerDecl` variants) as a `TypeExpr` that encodes `Tensor<[out, in], dtype>`. The dims come from there, NOT from a general `type_map` keyed by AST NodeId. During `wrga_adapter_inject::run`, each placement's `target_param` (e.g. `"m.w"` or `"blocks.0.c_attn.weight"`) must be resolved to the specific `ModelMember::LayerDecl` whose declared shape provides the dims.

**Ordering concern (caller must verify during implementation):** `wrga_adapter_inject::run` fires inside `invoke_wrga_if_enabled` in `stmt.rs`, which runs during @train compile time — by which point all model definitions have been collected by the compiler (`compile_models` runs earlier in the pipeline). If the resolution path walks `compiler.models` (the collected ModelDef registry), dims are always available. If it walks a codegen-maintained type map, verify the map is populated by the time the inject pass runs; if not, fall back to the ModelDef walk. **Never silently emit `(0, 0)`** — if dims can't be resolved, emit a diagnostic and skip that placement (the compile proceeds without that adapter materialised; user sees a clear error).

**Acceptance:** unit test in `wrga_adapter_inject.rs` constructs a mock placement targeting a 2-field model with `Tensor<[32, 16], f32>` layer and asserts `(input_dim=16, output_dim=32)`. Existing `adapter_inject_emits_lora_a_b_fields_with_expected_shapes` test stays green (but its synthesized field shapes now reflect real dims instead of zeros).

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

**Submodel call-site edge (Linear and friends — deliberate B.2.1 limitation):** NSL models frequently wrap weights in submodels — e.g., `self.c_attn(h)` on a `Linear` submodel desugars to `h @ self.c_attn.weight` inside `Linear::forward`. A user writing `@adapter(target=["blocks.0.c_attn.weight"])` expects the adaptation to land on that inner matmul. But the parent model's forward-method AST only contains the outer call `self.c_attn(h)`, not the inner matmul — it lives inside `Linear::forward`, a separate method body. B.2.1's rewrite walks **only the current method body** and won't see submodel-internal matmuls.

Two reasonable user-facing behaviors:

1. **Target only raw weights:** `@adapter(target=["m.w"])` where `m.w` is a `Tensor<...>` field directly on the model. This is what the B.2.1 rewrite handles. The user-facing decorator convention is "target the weight tensor, not the submodel wrapping it."

2. **Rewrite at submodel call sites:** `@adapter(target=["blocks.0.c_attn"])` would recognize `self.c_attn(h)` as the call site. Rewriting at call sites rather than matmul sites is a different pattern: `self.c_attn(h) → self.c_attn(h) + ((h @ self.lora_A_c_attn__lora) @ self.lora_B_c_attn__lora) * scale`. This doesn't require walking into `Linear::forward` because the adaptation is added OUTSIDE the submodel call.

**B.2.1 ships behavior (1) only.** The rewrite pattern strictly matches `MatMul { lhs: x, rhs: FieldAccess(self, W) }` where `W` is a `Tensor<...>` field. If the user's `target_param` matches a submodel field (not a raw tensor), the inject pass emits a clear error: `"@adapter(target=[\"<path>\"]): targets a submodel, not a weight tensor; adapt the submodel's inner weight directly (e.g., \"blocks.0.c_attn.weight\") or use a submodel-level decorator (not yet supported)"`.

Behavior (2) — submodel call-site rewriting — becomes B.2.2 or a follow-up. The B.2.1 test source decorates a raw `Tensor` field (`m.w`) to exercise behavior (1) unambiguously.

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
`sigmoid` applied element-wise to the gate; broadcast over batch.

**Step-0 analysis (important, explicit — contradicts intuition):** At step 0, `gate_<site>` is initialized to zeros. `sigmoid(0) = 0.5`, NOT zero — **the gate is half-open at step 0, not closed.** Equivalence with the base model holds at step 0 solely because `lora_B = 0`, which zeroes the entire adapter contribution regardless of gate state. A future contributor who changes LoRA B's init without changing the gate init will silently break base-model equivalence at step 0 — the gate doesn't save you. The test comments in Task 5 MUST make this explicit so the load-bearing invariant stays visible under refactor.

**Acceptance:** the existing `adapter_inject_emits_*_fields_with_expected_shapes` tests extended to cover IA³ and GatedLoRA compile + init; a runtime-output equivalence test for each (IA³ with scale=1 matches base; GatedLoRA with B=0 matches base).

### Task 5 — Runtime equivalence test (four builds; Build 4 is the load-bearing assertion)

Compile an NSL program with a LoRA-decorated model, run it via `nsl run`, capture output, compare against analytical expectations.

```
model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w
```

**Build 1 — reference:** no adapter. Capture output `y_base = x @ W = 0` (W is zeros; use a nonzero W if seeding needed). This is the baseline.

**Build 2 — sanity, NOT proof:** compile with `@adapter(type=lora, target=["m.w"], rank=2, alpha=2)`. At step 0, `B = zeros`, so `(x @ A) @ B = 0`, so adapted output should equal `y_base`. **Equivalence within 1e-5 f32 tolerance.** This is a SANITY check — if it fails, something is very wrong. But **passing is NOT proof the rewrite fired**, because if the rewrite never runs at all, the base forward executes and output also equals `y_base`. A passing Build 2 is necessary but not sufficient.

**Build 3 — divergence sanity:** compile with the same `@adapter(...)`, then programmatically seed `B` to a known nonzero pattern (e.g., `B = full([d_out, r], 0.1)`). Expect output to diverge from `y_base` by the analytically computed `((x @ A) @ B) * scale` term. This confirms the adapter contribution is being applied and scaled.

**Build 4 — LOAD-BEARING PROOF:** compile with `@adapter(...)`, then programmatically seed BOTH `A = nsl_tensor_ones([r, k_in])` and `B = nsl_tensor_ones([d_out, r])`. For `x = nsl_tensor_ones([batch, k_in])`, the adapter contribution is analytically:
```
(x @ A) @ B = ones([batch, r]) * k_in @ ones([d_out, r])   # Each entry of x@A is k_in.
            = ones([batch, d_out]) * k_in * r
```
Scaled: `* scale = alpha/rank = 1.0` (with alpha = rank = 2). Adapted output = `y_base + k_in * r * 1.0 * ones`. With k_in=8, r=2, every element of the adapted output is exactly `y_base[i,j] + 16.0`. Assert this element-by-element within 1e-5 tolerance.

**If the forward rewrite never fired**, Build 4 produces `y_base` (no adapter contribution), which differs from `y_base + 16` everywhere by exactly 16.0 — the test fails loudly. Build 4 is the actual proof the rewrite did its job; Build 2 is just a sanity check that B=0 is honored.

**Harness:** CLI form — spawn `nsl run` for each build via `Command::cargo_bin("nsl")`, capture stdout, parse numeric tensor output. Matches the existing `nsl-cli/tests/e2e.rs` pattern. Programmatic seeding of A/B uses a small helper that writes test values into the model's adapter side-table via the FFI surface (readable from Rust test code via `unsafe` tensor-pointer dereference).

**Tolerance:** 1e-5 absolute + relative combined for f32; 1e-9 for f64.

**Acceptance:** Builds 1, 2, 3, and 4 all pass. Build 4 is the gating assertion; the others corroborate correctness of the zero-init and scaling paths.

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

1. **Model struct layout extension — neutralised via side-table.** Per the architectural choice in §Architectural choices (1), the layout extension is exactly **one pointer slot wide** regardless of adapter count. `ModelDef`-derived layout computation is untouched. Zero callers outside the adapter code need to understand the new slot. The only interface the side-table exposes is `load_ptr → index → deref` at forward sites, which is mechanical codegen. This reclassifies the risk from "highest probability" to "mechanical implementation detail."

2. **AST rewrite pattern matching correctness.** Recognizing `self.<target_param>` at matmul sites reliably requires understanding NSL's ExprKind variants around dot access + binary ops. Mitigation: pattern match against a small set of canonical shapes in B.2.1; anything non-matching falls through unmodified. Task 5 Build 4 is the load-bearing assertion — if the rewrite didn't fire, Build 4 fails by a known analytical amount.

3. **LoRA-B=0 equivalence test false pass — neutralised by Build 4.** Task 5's original Build 2 (B=0 matches base) is a SANITY check; it passes trivially if the rewrite never runs. Build 4 (A=ones, B=ones, expect known-amount divergence) is the actual proof the rewrite fired and applied scaling correctly. Build 4 fails with a clear numeric mismatch if the rewrite is a no-op, so this risk is now a test-design-level mitigation rather than a hope.

4. **Submodel call-site adaptation is out of scope.** Users writing `@adapter(target=["blocks.0.c_attn"])` where `c_attn` is a Linear submodel will NOT get adaptation in B.2.1 — the inject pass emits an explicit error pointing at the raw-weight pattern. Mitigation: test source uses a raw-tensor target (`m.w`). B.2.2 or follow-up adds submodel call-site rewriting.

5. **Source-AD interaction.** Once LoRA A/B become real fields (even behind the side-table pointer), the source-AD extractor must see them as trainable params — otherwise the adapter gradients are zero and fine-tuning breaks. Mitigation: verify the source-AD `param_symbols` enumeration includes the side-table's tensors, not just the ModelDef-declared fields. If it doesn't, extend the enumeration to walk the side-table at extraction time. Task 5 Build 4 exercises forward-only; a post-B.2.1 check that `nsl run` with a `train(...)` block + `@adapter` actually updates A/B across steps is worth doing as part of Task 5 or Task 6.

6. **GatedLoRA step-0 invariant is subtle.** `sigmoid(0) = 0.5`, not zero — the gate is half-open at step 0, so base-model equivalence depends entirely on `B = 0`. A refactor that changes LoRA B's init without simultaneously changing gate init (e.g., to `sigmoid^(-1)(0) = -inf`, effectively `gate = -10`) will silently break equivalence. Mitigation: Task 5's test comments MUST make this explicit (see §Architectural choices (3) note). Task 4 doc comments in the rewrite code call it out.

7. **Task 1 dim-resolution ordering.** Dims come from `ModelDef.members[i].type_ann` — resolved during semantic analysis. `wrga_adapter_inject::run` fires inside `invoke_wrga_if_enabled` at @train-block compile time, after all models are collected. If dim resolution walks `compiler.models`, timing is fine. If it walks a codegen-internal type map populated during method-body lowering, it may run too early. Mitigation: the Task 1 implementation uses `compiler.models` (the ModelDef registry) explicitly; any fallback to a codegen type map must verify population order.

8. **`nsl_tensor_scalar_mul` availability.** The init emission uses `randn * (1/sqrt(fan_in))`. If no `scalar_mul` FFI exists with f64 scalar, use `nsl_tensor_full` to materialize a filled tensor and elementwise-multiply, or add the scalar_mul FFI. Recon-during-implementation decision.

9. **Windows stack budget after Task 3.** The post-merge stack-overflow fix (16MB main thread) holds today, but the AST rewrite adds additional recursion depth. If tests start failing again, bump the stack size or move the rewrite to iterative form.
