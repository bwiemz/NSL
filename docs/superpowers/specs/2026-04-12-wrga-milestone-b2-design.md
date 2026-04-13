# WRGA Milestone B.2 — Runtime Adapters + Allocator Plumbing + Test/Report Polish

**Status:** Design approved 2026-04-12.

**Goal:** Close four of the five gaps deferred from Milestone B.1. Make adapter A/B matrices exist at runtime, plumb the real allocator merge behind an opt-in flag, let the `debug_compile_and_return_plan` test helper handle real training sources, and wire `--wrga-report` for the remaining two build paths. The MMA epilogue PTX (item 1 from the B.1 deferral list) is explicitly out of scope and becomes its own Milestone B.3.

## Scope decision

Milestone B.2 covers items 2–5 from B.1's deferral list:

- **Task 1 — stdlib loading** in `debug_compile_and_return_plan` (B.1 deferred #4).
- **Task 2 — Adapter A/B materialisation** as synthesized model fields (B.1 deferred #2).
- **Task 3 — `LivenessAnalyzer` VarId→slot plumbing** + `--wrga-fold-allocations` flag (B.1 deferred #3, half-step variant).
- **Task 4 — ZK/standalone `--wrga-report`** emission (B.1 deferred #5).
- **Task 5 — Close-out**: memory file, regression, no merge.

Explicitly **out of scope** (becomes Milestone B.3, own brainstorm + plan):
- FusionPlan MMA epilogue PTX in `backend_ptx.rs` (the dominant item, ~1 week of PTX work).

## Execution order

Tasks run in this sequence because each downstream task benefits from the previous:

1. **Task 1 first.** Every later task's test suite benefits from being able to compile real `train(...): optimizer: sgd/adamw(...)` sources. Without it, tests keep using workaround sources that dilute coverage.
2. **Task 2 second.** Biggest item; establishes the adapter runtime surface that Milestone B.3's MMA epilogue will later fuse. Depends on Task 1 so its integration test can use a real training source.
3. **Task 3 third.** Stands alone but sequenced after Task 2 so Task 2's integration test runs first — defends against regressions from allocator plumbing.
4. **Task 4 fourth.** Pure plumbing; lowest risk; additive.
5. **Task 5 close-out.** Memory file + full regression run. No merge — controlling session performs the merge after review.

## Task 1 — stdlib loading in `debug_compile_and_return_plan`

**What it does.** The helper in `crates/nsl-codegen/src/lib.rs` currently parses and analyses only the caller's single source. When that source includes `train(...): optimizer: sgd(lr=...)`, codegen hits `undefined function 'nsl__optim__sgd__sgd_step'` because the stdlib module graph wasn't loaded. Fix so the helper loads stdlib the way `nsl-cli`'s `frontend()` does.

**Primary approach.** Factor the module-graph-loading logic from `crates/nsl-cli/src/loader.rs` into a reusable free function (or small struct) that both `nsl-cli` and `debug_compile_and_return_plan` call. `nsl-cli` stays the canonical compile flow; `debug_compile_and_return_plan` becomes a thin wrapper.

**Fallback.** If the existing loader is too entangled with CLI state to extract cleanly, inline a minimal stdlib loader in `debug_compile_and_return_plan` that resolves `NSL_STDLIB_PATH` and loads `nsl/optim/*.nsl` plus the minimal set of imports needed by the failing test. Document the limitation; a fuller refactor lands as a follow-up.

**Acceptance.**
- The B.1 Task 5 non-training workaround test (`no_wrga_inputs_skips_wrga_run_on_simple_compile`) grows a sibling test using a real `train(...): optimizer: sgd(lr=1e-3)` source. Both pass.
- No regression on `cargo test -p nsl-codegen --tests`, `nsl-cli` e2e, or CLI build paths.

**Files touched.**
- Modify: `crates/nsl-codegen/src/lib.rs` (helper rework).
- Modify (possibly): `crates/nsl-cli/src/loader.rs` (extraction) or a new `crates/nsl-codegen/src/stdlib_loader.rs` (inline path).
- Create: new test in `crates/nsl-codegen/tests/wrga_freeze_end_to_end.rs` (append).

## Task 2 — Adapter A/B matrix materialisation

**What it does.** For every `@adapter(type=lora|ia3|gatedlora, target=[...], rank=r)` site, inject synthesized model fields holding the adapter's weight tensors, initialize them correctly at model construction, and thread them into the forward pass so the adapter actually applies.

**Architecture.** Synthesized model fields (not a separate registry). At compile time, after `wrga::run` produces `WrgaPlan`, a new codegen pass walks `plan.placements` and emits per-site synthesized fields on the model struct:

- **LoRA:** `lora_A_<site>: Tensor<[r, k_in], f32>`, `lora_B_<site>: Tensor<[d_out, r], f32>`.
- **IA³:** `ia3_scale_<site>: Tensor<[d_out], f32>`.
- **GatedLoRA:** A/B plus `gate_<site>: Tensor<[d_out], f32>`.

**Site identifier.** A stable `<layer_name>__<kind>` string (e.g. `blocks_0_attn_q__lora`) derived from the decorator's `target` glob + adapter kind. Deterministic across compiles so checkpoints round-trip.

**Initialization.** Emitted by the codegen pass as part of the model's construction path. LoRA `A` = Kaiming-uniform, `B` = zeros; IA³ = ones; GatedLoRA `gate` = zeros. Hardcoded constants for B.2 — configurable init via `@adapter(init=...)` becomes a follow-up if needed.

**Forward-pass threading.** The same codegen pass rewrites forward-path matmul calls at adapter sites to apply the adapter. For LoRA, the unfused rewrite is:
```
y = x @ W + (x @ A) @ B
```
This is slow — two extra matmuls per site — but it is correct, produces real numerical output, and is what the B.3 MMA epilogue will later collapse into one fused kernel.

**Serialization.** The synthesized fields are real model fields, so the existing `save_model` / `load_model` paths discover them automatically. No serialization-format changes in B.2; a follow-up can wrap them in an `@wrga_adapters` section if cross-compile checkpoint rename becomes necessary.

**Testing approach.**
- **Compile-time test:** compile a model with `@adapter(type=lora, target=["m.w"], rank=4)` and inspect the resulting model struct's synthesized fields via the debug helper. Assert `lora_A_m_w__lora` and `lora_B_m_w__lora` exist with shapes `[4, k_in]` and `[d_out, 4]` respectively.
- **Runtime test:** compile and run a model with one LoRA site. Compute forward output. Because `B` is initialized to zeros, the initial output must equal the base-model output within float tolerance. Then hand-seed `B` to a non-zero value and assert the output diverges by the expected `(x @ A) @ B` amount.

**Sharp edge.** The adapter forward-threading must survive subsequent compiler passes (Wengert extraction, source-AD, memory planning). The synthesized fields exist as real model fields, so they participate in source-AD for free. The forward rewrite happens as part of the existing model-method codegen, not as a separate IR pass.

**Acceptance.**
- Integration test compiles a model with LoRA adapter, runs under `nsl run`, observes initial forward output matching base model (B = 0) and post-seed output diverging as expected.
- `cargo test --workspace --features cuda` stays green.

**Files touched.**
- Create: `crates/nsl-codegen/src/wrga_adapter_inject.rs` (new codegen pass).
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs` (invoke the pass after `wrga::run`).
- Modify: whatever module owns model forward-method codegen (probably `crates/nsl-codegen/src/compiler/model.rs` or similar — recon confirms in plan).
- Create: `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`.

## Task 3 — `LivenessAnalyzer` VarId→slot plumbing + `--wrga-fold-allocations` flag

**What it does.** Extend `LivenessAnalyzer` with a parallel VarId-keyed slot map. Implement `consume_hints(&WrgaPlan) -> usize` that actually folds allocations using the existing size+liveness safety gates. Gate real folding behind a new CLI flag (default off). Existing observational `apply_wrga_hints` behavior is unchanged.

**Architecture.** Dual-keyspace allocator.

- Add `allocated_slots: HashMap<VarId, RealSlotId>` to `LivenessAnalyzer`, with new type `RealSlotId(u32)`.
- Populated ONLY by WRGA-aware codegen paths (i.e., the `@train` block under source-AD). The existing String-keyed allocation path is unchanged — most of nsl-codegen continues to work as it does today.
- `real_slot_for(var: VarId) -> Option<RealSlotId>` and `try_merge_into_slot(var: VarId, target: RealSlotId) -> bool` become real implementations. Safety gates are the same ones B.1 proved: size equality, liveness disjointness. Refuse on any failure.
- `consume_hints(&WrgaPlan) -> usize`: walks `plan.memory.assignments`, groups entries by WRGA slot, calls `try_merge_into_slot` pairwise within each group. Returns post-merge real slot count.
- `apply_wrga_hints` (from B.1) stays as-is for the default path. `stmt.rs` chooses `apply_wrga_hints` (observational) or `consume_hints` (folding) based on the new flag.

**CLI flag.** `--wrga-fold-allocations: bool` on the `Build` subcommand in `crates/nsl-cli/src/main.rs`, plumbed through `CompileOptions` as `wrga_fold_allocations: bool`. Default **off**. B.3 or later flips the default after the MMA epilogue gives us real workloads to validate against.

**Testing approach.**
- **Unit test:** in `memory_planner.rs`, construct a synthetic `LivenessAnalyzer`, insert two VarIds into the same WRGA slot with non-overlapping liveness and equal size, call `consume_hints(&plan_with_those_assignments)`, assert post-merge slot count = 1.
- **Integration test:** compile a source with multiple frozen weights and `--wrga-fold-allocations`. Assert `debug_last_allocator_slot_count_post_hint() < pre_hint`. Without the flag, B.1's observational test (`post ≤ pre`) continues to pass.

**Sharp edge.** Dual-keyspace risk — the VarId-keyed slot number may differ from the String-keyed slot number for the same logical tensor. Mitigation: no cross-keyspace comparisons. The VarId map is populated only by WRGA-aware codegen; `consume_hints` operates only on VarId-keyed entries; String-keyed entries are untouched.

**Acceptance.**
- Unit test for `consume_hints` passes.
- Integration test with the flag set shows measurable slot-count reduction.
- All B.1 tests (including the observational-path test) still pass.
- Default compile behavior unchanged (flag is off by default).

**Files touched.**
- Modify: `crates/nsl-codegen/src/memory_planner.rs` (`allocated_slots`, `real_slot_for`, `try_merge_into_slot`, `consume_hints`).
- Modify: `crates/nsl-codegen/src/lib.rs` (extend `CompileOptions` with `wrga_fold_allocations: bool`).
- Modify: `crates/nsl-codegen/src/stmt.rs` (branch on the flag).
- Modify: `crates/nsl-cli/src/main.rs` (clap arg + pass-through).

## Task 4 — ZK/standalone `--wrga-report` emission

**What it does.** Replace the "not yet supported" stderr on `--zk-circuit` and `--standalone` + `--wrga-report` combinations with real plan-returning variants.

**Architecture.**
- Add `compile_with_zk_info_returning_plan(...) -> (ZkArtifact, Option<WrgaPlan>)` and `compile_standalone_returning_plan(...) -> (StandaloneArtifact, Option<WrgaPlan>)` in `crates/nsl-codegen/src/compiler/entry_points.rs`. Same body as existing functions plus the plan clone from `compiler.last_wrga_plan`.
- Re-export both at the nsl-codegen crate root, matching the B.1 `compile_returning_plan` pattern.
- Remove the two "not yet supported" branches in `run_build_zk` and `run_build_standalone` in `crates/nsl-cli/src/main.rs`. Wire to the new `_returning_plan` variants and call the existing `emit_wrga_report` helper.
- `--source-ad` precondition from B.1 Task 3 continues to apply.

**Testing approach.** Two new tests in `crates/nsl-cli/tests/wrga_report_cli.rs`:
- `wrga_report_works_on_zk_build_path` — `nsl build src.nsl --zk-circuit --source-ad --wrga-report`, asserts stdout contains `=== WRGA Compilation Report ===`.
- `wrga_report_works_on_standalone_build_path` — same with `--standalone`.

**Risk + fallback.** If the ZK or standalone pipelines have non-trivial structural differences from the normal path (distinct entry points, separate compile contexts), the `_returning_plan` variants may need more than mechanical duplication. If recon during implementation shows this, scope adjusts: emit a more informative stderr explaining exactly what's missing, and defer full support. Recon so far suggests these paths share the `Compiler` context and WRGA fires during the same source-AD lowering.

**Acceptance.** Both new tests pass. The two "not yet supported" stderrs are gone from main.rs.

**Files touched.**
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs` (two new variants).
- Modify: `crates/nsl-codegen/src/lib.rs` (re-exports).
- Modify: `crates/nsl-cli/src/main.rs` (remove "not yet supported" branches, wire new variants).
- Modify: `crates/nsl-cli/tests/wrga_report_cli.rs` (append two tests).

## Task 5 — Close-out

1. Create `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b2.md` documenting what B.2 shipped, per-task commit SHAs, and what remains for B.3 (just the MMA epilogue + any fallout uncovered during B.2).
2. Append one line to `MEMORY.md` under the existing Milestone B.1 entry, pointing at the new B.2 file.
3. Run full regression:
   ```
   cargo test -p nsl-semantic
   cargo test -p nsl-codegen --lib
   cargo test -p nsl-codegen --tests
   cargo test -p nsl-codegen flash_attention
   cargo test -p nsl-cli --test e2e -- --test-threads=1
   cargo test -p nsl-cli --test wrga_report_cli
   cargo build --features cuda
   cargo build --release --features cuda
   cargo clippy -p nsl-codegen -p nsl-semantic -p nsl-cli --features cuda --all-targets
   ```
4. **Do NOT merge.** Controlling session performs the merge after subagent reviews, same as B.1.

## Out-of-scope for B.2 (explicit list)

- FusionPlan MMA epilogue PTX in `backend_ptx.rs`. Becomes Milestone B.3 (own brainstorm → plan → execution).
- Configurable adapter init (`@adapter(init=...)`). Follow-up if demand appears.
- `@wrga_adapters` serialization section. Follow-up if checkpoint-rename compatibility becomes necessary.
- Flipping `--wrga-fold-allocations` to on-by-default. Depends on B.3 + real-workload validation.
- `nsl-cli` loader refactor beyond the minimum needed by Task 1.

## Risk register

1. **Task 1 loader extraction fragility.** If nsl-cli's loader is too entangled to extract cleanly, fall back to the inline minimal loader in `debug_compile_and_return_plan`. Acceptance bar is "test with real optimizer source passes", not "loader is perfectly factored."
2. **Task 2 forward-pass rewrite correctness.** The unfused `y = x @ W + (x @ A) @ B` rewrite must preserve the base-model output when `B = 0`. The integration test pins this down; if it drifts, the rewrite is wrong, not the baseline.
3. **Task 3 dual-keyspace drift.** The String-keyed and VarId-keyed maps are independent. The mitigation (no cross-comparisons) relies on convention, not type enforcement. Risk of a future contributor introducing a bridge that assumes equality. Document the invariant in `memory_planner.rs` docstring.
4. **Task 4 pipeline asymmetry.** ZK/standalone may differ structurally from the normal path. If so, ship the plan-returning variants but document gaps in `emit_wrga_report` output for those paths.
5. **B.3 dependency on B.2 Task 2.** The MMA epilogue kernel needs real A/B matrices to validate against. If Task 2 ships with a known correctness issue, B.3 is blocked. Task 2's integration test is the gate.
