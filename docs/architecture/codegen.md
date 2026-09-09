# nsl-codegen architecture

`nsl-codegen` is the back half of the NSL compiler: it takes the type-checked
AST (`nsl-ast`) plus the semantic `TypeMap` (`nsl-semantic`) and produces a
native object file through Cranelift, embedding any GPU kernels it synthesized
as PTX text in the object's data section. Everything a compiled NSL program
does at run time is a call into `nsl-runtime`'s `extern "C"` surface, so the
crate is, in the end, a large emitter of calls into a fixed C ABI plus the
analysis passes that decide *which* calls to emit.

This document extends `crates/nsl-codegen/ARCHITECTURE.md` (the facade map:
`core`, `gpu`, `training`, `quantization`, `distributed`, `analysis`,
`experimental` in `src/lib.rs`). It does not repeat the facade table. Read the
facade map first, then this, then:

- `docs/architecture/compiler-state.md` — where mutable compile state lives
  (`Compiler` fields, `CompileOptions`, the pass bus, the thread-local
  inventory and its drift gate).
- `docs/architecture/2026-08-15-milestone-c-trainir-reassessment.md` — why
  the train block is driven by `stmt.rs` with a `PassManager` judging order,
  rather than by a separate TrainIR.
- `docs/wiki/Compiler-Pipeline.md` and `docs/wiki/Optimization-Passes.md` —
  the user-facing pipeline and per-pass descriptions (the pass registry drift
  gate checks the wiki's headings against `pass_registry.rs`).
- `docs/summaries/01-language-and-compiler.md` and
  `docs/summaries/02-gpu-kernels-and-optimization.md` — feature-level
  summaries (the crate table in 01 is stale on line counts; the M31 fusion
  section in 02 records that `epilogue_fusion.rs` / `reduction_fusion.rs` /
  `fusion_graph.rs` were deleted — `ARCHITECTURE.md` still names the first
  two in its `analysis` row, which is a doc bug, not a hidden module).

Scale, for orientation: `src/lib.rs` is ~2.5k lines, `src/stmt.rs` ~3.2k
(plus `src/stmt_train/driver.rs`, ~2.7k, `src/stmt_control.rs` and
`src/stmt_assign.rs`, ~1.2k each, `src/stmt_pass_bridges.rs`, ~0.7k, and
`src/stmt_grad.rs`, `src/stmt_quant.rs`, `src/stmt_inspect.rs` and
`src/stmt_distill.rs`, 0.2–0.4k each),
`src/compiler/` ~32k across eight files, `src/source_ad.rs` ~8.7k,
`src/flash_attention.rs` ~8.5k. There are 301 integration-test files under
`tests/` and ~200 modules at the crate root.

## Overview: the pipeline as code

There is no IR between the AST and Cranelift for CPU code. `Compiler` (in
`src/compiler/mod.rs`) walks the AST once per phase and emits CLIF directly
through `cranelift_frontend::FunctionBuilder`. The only intermediate
representations are (a) the `WengertList` that source-to-source autodiff
builds for a train block's forward and backward, and (b) `KernelIR` for the
portable GPU path. Analysis passes are Rust functions that read the AST, the
Wengert list, or the compiler's collected metadata, and publish plans that
later emission consults.

```
                    nsl_codegen::compile / compile_returning_plan / compile_entry /
                    compile_module(_with_imports) / compile_test / compile_standalone /
                    compile_with_options (test convenience: lexes+parses+analyzes itself)
                                        │   src/compiler/entry_points.rs
                                        ▼
  Compiler::new(interner, type_map, &CompileOptions)          src/compiler/mod.rs
  install_per_compile_program_facts · load weights (--weights) · profile pre-pass
                                        │
   ┌── COLLECT ─────────────────────────┼──────────────────── src/compiler/collection.rs
   │  intern_string · collect_strings · collect_enums · collect_structs
   │  collect_models · collect_agents
   │  cpdt_decorator / cpdt_expert_prune / cpdt_moe_capacity (metadata-only passes)
   │  populate_calibration_retention_from_ast_if_unset (AWQ + WGGO pre-scan)
   │  emit_retention_arena · emit_grad_retention_arena
   ├── DECLARE ─────────────────────────┼──────────────────── src/compiler/declaration.rs
   │  declare_runtime_functions   ← the nsl-abi table (nsl_abi::for_each_runtime_fn!)
   │  declare_user_functions · declare_agent_methods
   │  apply_vmap_transforms / register_batched_functions   (src/vmap.rs)
   ├── COMPILE ─────────────────────────┼─────────────────────
   │  compile_datatype_defs
   │  compile_kernels                 ← src/compiler/kernel.rs  (`kernel` blocks → PTX/KIR)
   │  wrga_prescan (adapter sites)
   │  compile_flash_attention_kernels ← phase KernelPrepass: WGGO prepass, PCA detection
   │  compile_user_functions          ← src/compiler/functions.rs → func.rs → stmt.rs / expr/
   │  compile_agent_methods · compile_batched_functions
   │  MemoryPlanner (whole-program slab plan, scheduled through PassManager)
   │  compile_main                    ← src/compiler/main_entry.rs, phase TrainBlock
   │      └─ top-level stmts → compile_stmt → compile_train_block (see below)
   │  compile_pending_lambdas · run_wcet_analysis · fusion report
   │  embed_weight_hash · emit_export_wrappers (c_wrapper.rs) · write profile manifest
   └── FINALIZE ────────────────────────┼─────────────────────
      Compiler::finalize → ObjectModule::finish → object bytes (Vec<u8>)
                                        │
                     nsl-cli: linker::link / link_multi / link_shared_with_exports
                              c_header::emit for --shared-lib     src/linker.rs, src/c_header.rs
                                        ▼
                     executable · shared library (+ .h) · standalone · unikernel image
```

The phases are literal method calls in `compile_returning_plan_impl`
(`src/compiler/entry_points.rs`); the other entry points
(`compile_module_with_imports`, `compile_entry`, `compile_test`,
`compile_standalone`) run the same sequence with different linkage and
`main` policies. `compile_returning_splice_count_for_tests` in the same file
is the shortest readable copy of the sequence.

### Statement and expression lowering

- `Compiler::compile_stmt` (`src/stmt.rs`) dispatches on `StmtKind`; the
  control-flow lowerings it dispatches to — `if`, `while`, `while let`,
  `for` (ranges and lists, model arrays, a DataLoader) and `match`, with
  the non-owning-alias materialization before a branch or loop — live in
  `src/stmt_control.rs`; the assignment lowering (`compile_assign`, the
  destructuring patterns, the slab-tensor fast path) and the binding facts
  the ownership sweep reads live in `src/stmt_assign.rs`; the `grad` block
  drivers (the source-AD arm's compile-time backward and the tape-AD arm's
  runtime backward, which the train block's tape-AD path shares) live in
  `src/stmt_grad.rs`; the pass bridges the train-block driver calls into
  (`invoke_wrga_if_enabled`, `invoke_cpdt_if_enabled`,
  `invoke_csha_if_enabled` — each builds the pass input from the state
  stashed on the `Compiler`, runs the pass, records its disposition and
  publishes the product on the bus) live in `src/stmt_pass_bridges.rs`;
  the `quant` block lowering and the AWQ projection discovery in
  `src/stmt_quant.rs`, the `distill` block (a synthetic train block with
  the teacher frozen) in `src/stmt_distill.rs`, and the `@inspect` hook
  emission in `src/stmt_inspect.rs`. `stmt.rs` is the file that also owns
  the train block (below), the `serve` lowering, and most feature-specific
  refusals. `FuncState`
  (`src/context.rs`) is the per-function state: variables, types, loop
  context, tensor cleanup bookkeeping, ownership state.
- `Compiler::compile_expr` (`src/expr/mod.rs`) dispatches to `expr/access.rs`
  (field/subscript), `expr/binary_ops.rs`, `expr/calls.rs` (function, method,
  indirect and runtime calls — including `compile_call_by_name`),
  `expr/literals.rs`, and `expr/advanced.rs` (tensor methods, model-method
  calls, packing metadata reads).
- `src/func.rs` builds a Cranelift `Function` for each `FnDef`
  (`compile_fn_def`, `compile_fn_def_named`); `src/types.rs` maps NSL types
  to Cranelift types (`nsl_type_to_cl`, `pointer_type`).
- Ownership/lifetime analysis feeding free-emission: `src/ownership.rs`,
  `src/ownership_expr.rs`, `src/escape.rs`, `src/dict_lifetime.rs`,
  `src/use_count.rs`, `src/ffi_ownership.rs` (the FFI ownership table —
  `crates/nsl-codegen/tests/ffi_ownership_drift.rs` binds the three ownership authorities).

### The train-block compiler

`compile_train_block` (`src/stmt_train/driver.rs`) is the callee-side phase scope: it
installs `CompilePhase::TrainBlock` via `pass_trace::enter_phase`, refuses the
`@pipeline` + `--layerwise-accum` / `--zero-stage` compositions, offers CPDT
at the wrapper (`schedule("CPDT", …)`) and then calls
`compile_train_block_inner` (same file), a ~2.3k-line driver. Its shape, in the order the
driver runs it:

1. Config extraction from `train(...)` arguments — one resolver in
   `nsl-semantic` owns the key set; this call is the backstop.
2. `fase::plan` under `schedule("FASE", …)` → `FasePlan` / `FaseMode`.
3. Admission (`src/stmt_admission.rs`, `csla_and_zero_admission`) — decides
   whether CSLA Stage 2 is active and refuses the non-validated combinations
   of `--layerwise-accum`, `--weight-stream`, `--zero-stage`.
4. Parameter lists (`src/stmt_train/param_lists.rs`: `muon_route_flags`,
   `decay_exempt_flags`, `alloc_grad_accum_buffers`). After the optimizer
   state is allocated the driver builds the block's `TrainPlan`
   (`src/stmt_train/plan.rs`: the resolved contract and hyper-parameters,
   the FASE plan and the admissions as `TrainSpec`, the parameter paths and
   state-buffer count as `ParamPlan`, accumulation and checkpointing as
   `TrainSchedule` — plain data, no Cranelift handle), which the
   checkpoint-identity record and the late emitters below read as `&plan`
   (roadmap A1, TrainPlan design step 1).
5. Epoch/batch loops; forward extraction into a `WengertList` by
   `WengertExtractor` (`src/source_ad.rs`), then the initial primal
   `VarMap` (`src/stmt_train/primal_vars.rs`: `emit_primal_vars` — named
   inputs / parameters to their Cranelift values, the input device guards,
   the nested parameter and frozen teacher loads, the CPKD report facts);
   the in-pipeline passes, each under
   `PassScheduler::schedule`: CPKD, WGGO (with the tape;
   `src/stmt_train/plan_wggo.rs`: `plan_wggo`, which also folds the
   wrapper's pre-plan and publishes the `WggoOverrides`), CSHA and the
   WGGO prune (`src/stmt_train/plan_csha_prune.rs`:
   `run_csha_and_wggo_prune`), WRGA (with
   the tape) and CPDT (both in `src/stmt_train/plan_wrga_cpdt.rs`:
   `run_wrga_and_plan_cpdt`; the plan's adapter sites — the override
   diagnostics, the adapter init side-table and the adapter-tensor loads
   into the VarMap — are then emitted by `src/stmt_train/adapter_sites.rs`:
   `emit_wrga_adapter_sites`), CCR (the WRGA fork into the effective primal and the
   CCR plan — blocks, stride, budget, compression — in
   `src/stmt_train/plan_ccr.rs`: `fork_wrga_and_plan_ccr`), and the
   transient-arena `MemoryPlanner` (with the
   adjoint). PCA and WGGO's prepass run earlier, in
   `compile_flash_attention_kernels` (`src/compiler/kernel.rs`).
6. Adjoint generation (`AdjointGenerator::generate`, `ad_rules::apply_ad_rule`)
   and lowering (`wengert_lower::compile_wengert_ops` /
   `compile_wengert_ops_range`; under the FASE hook through
   `src/stmt_train/fase_hook_lowering.rs`: `emit_fase_hook_adjoint_lowering`,
   the per-parameter accumulate callback inside the grad-integrity
   bracket). Between the two, sections 6a–6b.5
   (`src/stmt_train/adjoint_tape_opt.rs`: `optimize_adjoint_tape`) rewrite
   the tape — the WRGA backward-live filter, dead-gradient elimination, the
   bit-exact backward folds and the CSLA schedule report — and return the
   parameter-gradient adjoint set; then, under CCR, section 6d
   (`src/stmt_train/ccr_adjoint_frees.rs`: `insert_ccr_adjoint_frees`)
   inserts the adjoint-region last-use frees on the tape, protecting the
   parameter-gradient adjoints and the planned wgrad fusion chains; section
   6e (`src/stmt_train/transient_arena_projection.rs`:
   `emit_transient_arena_projection`) then projects the transient-memory
   arena over the final tape — the element hints, the `--memory-report`
   arena report and the `--transient-arena` placement with its runtime slot
   declarations; the CSLA schedule precompute
   (`src/stmt_train/csla_precompute.rs`: `precompute_csla_schedule`) then
   derives, on that final adjoint, the layerwise plan the save phase and the
   window backward share and the `--weight-stream` sliced-forward plan; the
   forward lowering (`src/stmt_train/forward_lowering.rs`:
   `emit_forward_lowering`) — the tape-unchanged assertion, the per-segment
   early-free plan and the monolithic or segment-streamed primal lowering —
   follows. Under
   `--layerwise-accum` the adjoint is
   instead buffered per micro-batch (`src/stmt_train/csla_window.rs`:
   `emit_csla_window_save`, the `csla_active` arm of that site, which
   pushes every adjoint-read primal value into the window's slot list and
   returns the `CslaPending` carrier) and replayed on the accumulation
   boundary by the CSLA window backward (same file:
   `emit_csla_window_backward` — the D1b layer-major schedule, per-b
   seeding, per-range lowering with the fused per-layer update, the
   weight-stream prefetch belt, the window cleanup; the `CslaPre` /
   `CslaPending` / `CslaSchedule` carriers live there too).
   After the adjoint is lowered, section 8 (`src/stmt_train/source_ad_grads.rs`:
   `emit_source_ad_grads`) builds the parameter-gradient list (a null
   sentinel under the FASE hook) and sweeps the lowering's intermediates.
7. Per-step diagnostics (`src/stmt_train/health_hooks.rs`:
   `emit_train_health_hooks` — the `--debug-training` gradient checksum,
   the grad-integrity scan, the health-monitor loss / gradient-norm /
   weight-norm records and snapshot flush), then gradient clipping and
   accumulation, then the optimizer step (`src/stmt_train/optimizer_step.rs`: `emit_optimizer_step`
   — the accumulation gate, the mode-table / FASE-deferred / stdlib step
   arms, the ZeRO reduce and sync, the post-optimizer cleanup), calling the
   CSLA window emitters (`src/stmt_csla.rs`: `emit_csla_accum_alloc`,
   `emit_csla_group_update`) or the FASE Deferred emitters
   (`src/stmt_fase.rs`: `fase_emit_accumulate`, `fase_emit_final_step`,
   `match_adamw_program`); then the scheduler call, the step-count
   increment and the periodic full-state checkpoint
   (`src/stmt_train/scheduler_step.rs`: `emit_scheduler_step`).
8. Teardown after the epoch loop (`src/stmt_train/teardown.rs`:
   `emit_train_teardown`).

`src/stmt_train/mod.rs` documents the rule for these peels: each is a
byte-for-byte move under the train-block CLIF snapshots
(`crates/nsl-codegen/tests/train_clif_snapshots.rs`). `compile_train_block_pipelined_inner`
(`src/stmt_train/pipelined.rs`, with its `compile_train_block_pipelined`
entry point) is the separate `@pipeline` path; it shares none of the
scheduled passes.

### Pass scheduling: `PassManager`, `PassScheduler`, `PassBus`

- `src/pass_registry.rs` — `PASSES: &[PassDescriptor]`, the eleven declared
  passes (CCR, FASE, WGGO, CSHA, WRGA, CPDT, PCA, CPKD, CEP, CFIE,
  MemoryPlanner) with `source_files`, `cli_flags`, `stage: PipelineStage`,
  `phases: &[CompilePhase]`, `decorator_triggers`, `wiki`, `tape: TapeAccess`.
  `CompilePhase` is `{KernelPrepass, TrainBlock, Lowering, Analysis,
  OutOfBand}`; `PipelineStage` is `{PreExtraction, ModuleScan, OnWengert,
  OnAdjoint, Lowering, OutOfBand}` and is descriptive, not a scheduling key.
- `src/pass_trace.rs` — `enter_phase` (RAII `PhaseGuard`), `record`,
  `record_disposition` (`PassDisposition`, `DeclineReason`), and the
  `NSL_PASS_TRACE=1` report. Every pass records at its own entry
  (callee-side), which is what covers drivers nobody wrapped.
- `src/pass_bus.rs` — `PassBus` (reached as `compiler.bus`) with one private
  field per `Channel` (`CshaBridge`, `CshaClaimedOps`, `CshaBackwardClaims`,
  `WrgaPlan`, `AdapterPrescanPlan`, `CpkdPlan`, `CpdtPlan`, `CfiePlan`,
  `WggoOverrides`, `WggoPreplans`, `AdapterSites`, `CfieServeGen`), each
  described by a `ChannelDescriptor` in `CHANNELS` (producer, consumers,
  `consumed_by_passes`, `dead_output` / `applied_implies_published` /
  `read_before_publish` invariants, `OrderClaim`). Accessors count traffic;
  `findings()` and `dependency_order_violations()` read it back.
- `src/pass_manager.rs` — `PassManager` (one per `Compiler`, field
  `compiler.passes`; anchors the compile epoch; `!Send` by a
  `PhantomData<*const ()>`), `PassScheduler::schedule(name, tape, body)`
  returning `Scheduled<R>`, which must be settled with `finish(&bus)` (or an
  explicit `defer_postconditions(reason)`); `assert_tape_unchanged_since` and
  `rescan_tape` implement the positional-reference rule for tape scans;
  `enforce_dependency_order` is run at the train block's exit.

The gates that make these declarations true: `crates/nsl-codegen/tests/pass_registry_drift.rs`,
`crates/nsl-codegen/tests/pass_bus_drift.rs`, `crates/nsl-codegen/tests/pass_manager_drift.rs`,
`crates/nsl-codegen/tests/tape_access_drift.rs`, `crates/nsl-codegen/tests/pass_scheduler.rs`,
`crates/nsl-codegen/tests/pass_scheduler_coverage.rs`, `crates/nsl-codegen/tests/tape_reference_discipline.rs`,
`crates/nsl-codegen/tests/pass_trace_unit.rs`, `crates/nsl-codegen/tests/pass_bus_unit.rs`, `crates/nsl-codegen/tests/pass_manager_unit.rs`.

### Object emission, linking, C export

- `Compiler::finalize` publishes the `@export` list into
  `CompileOptions::export.functions_out` (an `Arc<Mutex<…>>` slot the CLI
  owns) and calls `ObjectModule::finish().emit()`.
- `src/linker.rs` — `link`, `link_multi`, `link_shared`,
  `link_shared_with_exports`, `default_output_path`,
  `default_shared_lib_path`; finds `libnsl_runtime*.a` in the toolchain dir
  (`find_runtime_lib`) and drives the system C compiler (`find_c_compiler`).
  Called from `crates/nsl-cli/src/commands/build/{normal,shared_lib,standalone,zk}.rs`
  and `commands/test.rs`, never from inside codegen.
- `src/c_wrapper.rs` — per-`@export` C-ABI wrapper emission
  (`emit_c_abi_wrapper`, `build_c_abi_wrapper_signature`,
  `emit_c_abi_dispatch_wrapper`); `src/c_export_table.rs` — the export table
  `nsl_model_create` reads; `src/c_header.rs` — `ExportInfo`, `lower_type_expr`,
  `emit(exports, module_name)`, stamping `NSL_ABI_VERSION_MAJOR/MINOR` from
  `nsl_abi::wire::version`; `src/c_wrapper.rs` steps through descriptor
  arrays by `sizeof` of `nsl_abi::wire::tensor_desc::NslTensorDesc`. Gates: `crates/nsl-codegen/tests/c_header_agreement.rs` (header vs runtime,
  through `nsl_abi`), `crates/nsl-codegen/tests/c_header_compiles.rs` (real C compiler +
  `_Static_assert` on `NslTensorDesc`), `crates/nsl-codegen/tests/c_header_snapshot.rs`,
  `crates/nsl-codegen/tests/exported_symbols_are_dlsym_findable.rs`,
  `crates/nsl-codegen/tests/export_table_runtime_ffis.rs`.
- `src/standalone.rs` (`create_weight_object`) and `StandaloneConfig`
  (`src/compiler/mod.rs`) back `nsl build --standalone`; `src/unikernel.rs` /
  `src/unikernel_boot.rs` back `--unikernel`.

## Entry points and CompileOptions

The curated public surface is re-exported at the crate root from
`src/compiler/entry_points.rs`: `compile`, `compile_returning_plan`,
`compile_entry`, `compile_entry_returning_plan`, `compile_entry_capturing_ir`,
`compile_module`, `compile_module_with_imports` (+ `_returning_plan`,
`_best_effort_plan(s)`), `compile_test`, `compile_standalone`
(+ `_returning_plan`), `compile_with_profile_captures`, `compile_with_zk_info`
(+ `_returning_plan`), and `compile_returning_splice_count_for_tests`. The
`_returning_plan` variants also hand back the `WrgaPlan` the train block
published (for `nsl build --wrga-report`). `compile_with_options` and
`compile_and_calibrate` live in `src/lib.rs`; the former is the one-call
"source string → object bytes" wrapper (it runs `nsl_lexer`, `nsl_parser`,
`nsl_semantic::analyze` itself) and is what most integration tests use.

`CompileOptions` (`src/lib.rs`, doc comment "Compiler configuration flags
passed from CLI") is the configuration half of the session; `Compiler` copies
it into `compile_options` at `Compiler::new`. Where it comes from:

- `crates/nsl-cli/src/commands/build/options.rs` (`dispatch(BuildArgs)`) and
  `crates/nsl-cli/src/commands/run.rs` construct it from clap args; the
  matmul flag group is shared through `crates/nsl-cli/src/matmul_args.rs`.
- Analysis subcommands (`commands/check.rs`, `commands/autotune.rs`,
  `commands/build/wrga_check.rs`, `profile.rs`, `meta_flags.rs`) build their
  own instances.
- `debug_resolve_pre_scan_opts` (`src/lib.rs`) exposes the pre-scan phase
  (`run_pre_scan_phase` in `entry_points.rs`) that fills still-`None`
  calibration/WGGO fields from the AST.

The struct has 29 `pub` fields today. The decomposition into cohesive
sub-structs that already exists (grep `Options {` in `src/lib.rs`):
`WggoOptions` (`opts.wggo`), `CfieOptions` (`opts.cfie`), `WcetOptions`
(`opts.wcet`), `ZkOptions` (`opts.zk`), `CshaOptions` (`opts.csha`),
`CpdtOptions` (`opts.cpdt`), `CalibrationOptions` (`opts.calibration`:
data path, mode, sample/batch/timeout budgets, the AWQ `retention` and
WGGO `grad_retention` plans, `batch_seq`, the subprocess `compile_bundle`
and the `sidecar` the harness writes back), `DevToolsOptions`
(`opts.dev_tools`: the kernel profiler's `profile_kernels` /
`manifest_output_path` / `profile_source_text` / `profile_source_file_name`,
the health monitor's `health_monitor` / `health_flush_interval`, and
`inspect_enabled`), `CheckpointOptions` (`opts.checkpoint`: the CCR flags
`blocks` / `selective` / `budget_mib` / `stride` / `compress`, plus the
decorator-derived per-function `policies` map the CLI publishes after
semantic analysis), `WeightStreamOptions` (`opts.weight_stream`: the
`--weight-stream` ladder — `enabled` / `arena` / `prefetch` /
`async_writeback`), `MuonOptions` (`opts.muon`: `batch_ns` /
`resident_momentum` / `state_bf16`; `Features` keeps its own
`muon_state_bf16` copy for the emission paths), `ImportedModelOptions`
(`opts.imported_model`: the multi-file build's `field_dims` / `field_ranks`
/ `tensor_fields_without_dims` / `field_values` channel for model fields
declared in imported modules, which `ctor_fold` also merges into and
`entry_points` merges under the entry module's own collection),
`ZeroOptions` (`opts.zero`: `stage` / `elementwise` for `--zero-stage` /
`--zero-elementwise`; `Features` and the parameter plan's `PlanFeatures`
keep their own copies), `AutotuneOptions` (`opts.autotune`: `disabled` /
`fresh` for `--no-autotune` / `--autotune-fresh`), `WeightsOptions`
(`opts.weights`: the `--weights` `file`, the M52 weight-aware `config`,
the `nsl check --weight-analysis` report flag `analysis`, and the `@export`
`index_map` the CLI fills from `AnalysisResult.weight_index_map`),
`FusionOptions` (`opts.fusion`: the `disabled` kill switch, the
`--fusion-report` `report` flag, and the opt-in source-AD fusions
`rmsnorm_backward` / `wgrad_accum` / `wgrad_accum_from_bundle`; the `@fuse`
kernel path reads the first two through `FusionState`), `DiagnosticsOptions`
(`opts.diagnostics`: the observation-only `trace_ops` / `nan_analysis` /
`grad_integrity` gates and the lowering-changing `debug_training` /
`training_reference` modes, field names unchanged), `MemoryOptions`
(`opts.memory`: the M36 `vram_budget` and plan `report`, and the
`transient_arena` placement switch), `WrgaOptions` (`opts.wrga`: the
decorator-config `inputs` the CLI bridge forwards from nsl-semantic, the
Milestone B.2 `fold_allocations` switch, and the `nsl check` override
`check` — a `WrgaCheckContext`, which retired the CLI's WRGA thread-locals;
see compiler-state.md Phase 2), `AnalysisOptions` (`opts.analysis`: the
facts the CLI bridge copies out of `nsl_semantic::AnalysisResult` —
`ownership_info` and the `csha_configs` / `fused_ce_configs` /
`fused_kl_ce_configs` / `pca_user_strategies` decorator configs; not user
flags), `ExportOptions` (`opts.export`: the `--shared-lib` PIC switch
`shared_lib`, the export-table emitter decision `emit_table`, and the
`@export` header slot `functions_out`), `TrainOptions` (`opts.train`: the
training-execution knobs `optim_state_offload`, `layerwise_accum`,
`param_dtype_bf16sr` and `cuda_graphs`, all execution-fingerprint keys),
`DeterminismOptions` (`opts.determinism`: the M46 `enabled` switch for
`--deterministic` and the program-start RNG `seed` for `--seed`), plus
`MatmulConfig`
(`opts.matmul`). Everything else is still a
flat field (`source_ad`, `target`, `world_size`,
`target_gpu`, `dtype`, …). New options
belong in a sub-struct when they share a subsystem; otherwise a flat field
is acceptable but should carry a doc comment naming the flag.
`HarnessConfig` (`src/calibration/mod.rs`) keeps its own
`calibration_data: PathBuf` — it is the harness's input record, not an
alias of `opts.calibration.data`. `target_gpu` / `dtype` stay flat on
purpose: the profile walker shares them with `serve`, the GPU-spec lookups
and the execution fingerprint.

**`MatmulConfig`** (`src/lib.rs`): `mode: MatmulMode`, `bf16_rounding`,
`bf16_min_ratio`, `bf16_cast_cache`, `bf16_lt`, `bf16_lt_workspace_mib`,
`bf16_lt_tune`. `with_env_fallback` applies the deprecated `NSL_MATMUL_BF16*`
environment variables to fields still at their default (an explicit flag
always wins) and warns once per variable; `clamped()` mirrors the runtime's
clamping so the fingerprint records effective values.

**`exec_fingerprint`** (`CompileOptions::exec_fingerprint`, `src/lib.rs`):
a fixed-order `k=v,…` string of every option that changes *what arithmetic
runs* or *where bytes are placed* (`ad`, `det`, `dtype`, `fusion`,
`fuse_rms`, `fuse_wgrad`, `zero`, `zero_elem`, `ws`, `muon_*`, `lmhead`,
the `mm*` matmul keys, `arena`, `graphs`, `ckpt`, `offload`). Three rules its
doc comment states and the unit tests in `exec_fingerprint_tests` pin: fixed
order, closed vocabularies (never `{:?}`), and no key omitted when false.
`compile_main` (`src/compiler/main_entry.rs`) interns the string and emits a
call to `nsl_set_exec_fingerprint` (a `diagnostics` row of the nsl-abi
table, implemented in `crates/nsl-runtime/src/exec_fingerprint.rs`), so a checkpoint written later
carries it and a resume can refuse an arithmetic mismatch.

`--dump-ir` (`dump_ir` on the entry points) prints each function's CLIF just
before `define_function`; `src/ir_capture.rs` (`Compiler::record_ir`,
`IrDump`) keeps the same text as data for `compile_entry_capturing_ir` and
the CLIF snapshot suite.

## The runtime ABI boundary

Every runtime call the codegen can emit is declared once, in the typed
ABI table `crates/nsl-abi/src/table.rs` (roadmap A3): one row
`[group] name(params) -> ret = runtime::path;` per function, 682 of them,
exposed as the X-macro `nsl_abi::for_each_runtime_fn!` and as data
(`nsl_abi::RUNTIME_ABI`). The groups are the split PR #600 made along
"what the language exposes vs what the runtime implements": `memory`,
`io`, `scalar`, `collections`, `tensor` are the surface a program writes
directly (adding one widens the language); `abi_tensor`, `training`,
`optimizer`, `distributed`, `inference`, `quantization`, `diagnostics`,
`abi_memory`, `interop` are the implementation surface behind it (adding
one changes how a program runs, not what it can say). The lowering code for
the first five groups lives in `src/builtins/{memory,io,scalar,collections,
tensor}.rs`.

`src/builtins/mod.rs` renders the table into `RUNTIME_FUNCTIONS`
(`render_runtime_functions!`, one `(name, &[Cranelift types], ret)` per row);
`all_runtime_functions()` iterates it and `declare_runtime_functions`
declares each as `Linkage::Import`, returning the
`HashMap<String, (FuncId, Signature)>` stored in
`compiler.registry.runtime_fns` (`FunctionRegistry`, `src/compiler/mod.rs`).
Declaration order is *not* load-bearing for CLIF (funcrefs are numbered per
function by first use; verified by the train CLIF snapshots when `nsl_alloc`
moved tables), so a row may be moved between groups freely. It does reach
the `.o` symbol table, on which nothing depends.

A call is emitted by `Compiler::compile_call_by_name(builder, name, args)`
(`src/expr/calls.rs`): it first tries `registry.functions` (user functions,
after vmap dispatch), then `registry.runtime_fns`, then a
`tensor_unary_runtime_alias`; it calls `module.declare_func_in_func` and
`builder.ins().call`, returns `iconst 0` for void callees, and records the
emission in `last_ffi_emission` for the FFI-ownership classifier. Direct
`registry.runtime_fns.get(...)` lookups exist for a few hand-built signatures
(e.g. the health hooks in `src/stmt_train/health_hooks.rs`) but the by-name path is the norm.

**Drift gates.** The table and the runtime are linked by symbol name, so
the runtime's own build checks the agreement: `crates/nsl-runtime/src/abi_check.rs`
renders every row into a `const` that casts the named implementation to
`unsafe extern "C" fn(_, …) -> _` and compares its inferred signature with
the row slot by slot (`nsl_abi::typed::assert_sig`; register class and width,
so `u64`, `usize` and raw pointers are `i64` slots). A row whose arity,
types or path disagree fails `cargo build -p nsl-runtime` with the
function's name. As belt-and-braces, `crates/nsl-abi` (dependency-free)
parses every `#[unsafe(no_mangle)] extern "C" fn` in `nsl-runtime` and
cross-checks it against the typed table (`nsl_abi::check_workspace`,
`cross_check`, `MismatchKind::DuplicateDecl`);
`crates/nsl-abi/tests/signature_agreement.rs` is that CI gate
(`runtime_function_signatures_agree_with_extern_impls`, with a truncation
floor of 682 rows recorded 2026-09-02, and `nsl_abi::table::tests` pins the
row count exactly). Inside the codegen, `builtins/mod.rs` unit tests
`no_runtime_function_is_declared_twice` and `registry_is_the_abi_table`
guard the rendering. `crates/nsl-codegen/tests/c_header_agreement.rs` reuses
the text parser for the generated C header.

**Shared constants imported from `nsl_runtime`** (grep `nsl_runtime::` in
`src`, non-comment uses): `nsl_runtime::param_plan::{PLAN_BF16_SR,
PLAN_ELEMENTWISE, PLAN_SHARDED, PLAN_STREAMED}` (re-exported by
`src/parameter_plan.rs`), `nsl_runtime::pca_tier_b_runtime::{TIER_B_MAX_BAKED_SEQ_LEN,
TIER_B_SEQ_LEN_FLOOR}` (re-exported by `src/pca_tier_b.rs`),
`nsl_runtime::tensor::NSL_TENSOR_DATA_OFFSET` (`src/expr/mod.rs`,
`src/calibration/binary_codegen.rs`), `nsl_runtime::tensor::DTYPE_*`
(compile-time asserts in `src/cpdt_precision_exec.rs`),
`nsl_runtime::c_api::{NSL_ABI_VERSION_MAJOR, NSL_ABI_VERSION_MINOR,
nsl_abi_version}` (`src/c_header.rs`), `nsl_runtime::awq::AwqScales`
(`src/stmt.rs`), `nsl_runtime::calibration_data::peek_batch_seq`
(`src/lib.rs`, calibration), and `nsl_runtime::CudaDeviceIdentity` /
`cuda_device_name` (`src/gpu_specs.rs`, `src/autotune.rs`). The rule these
follow: a layout or plan constant the emitted code must agree with is
imported from the runtime, never retyped in codegen.

## GPU codegen

Two ways exist to produce a GPU kernel, and the freeze decides which one new
code may use.

**KernelIR path (the sanctioned one).** The IR lives in the leaf crate
`crates/nsl-kir` (roadmap A2 step 1: no workspace dependencies, pinned by
`crates/nsl-kir/tests/leaf.rs`, so the runtime can build kernels on it
without depending on the compiler); `nsl_codegen::{kernel_ir, kir_verify,
backend_ptx}` and `nsl_codegen::gpu_target::FeatureSet` re-export it at the
historical paths. `crates/nsl-kir/src/kernel_ir.rs` defines
`KernelIR { name, params: Vec<KirParam>, blocks: Vec<KirBlock>, var_types,
shared_mem_bytes, workgroup_size, required_features: FeatureSet }`, `KirOp`,
`KirTerminator`, `KirType`, `AddressSpace`, and `KirBuilder` (`new`,
`add_param`, `new_typed_var`, `new_block`, `set_block`, `emit`, `terminate`,
`finalize() -> KernelIR`). `crates/nsl-kir/src/kir_verify.rs::verify` is the KIR verifier
(roadmap A2 step 2): block shape and branch targets, SSA (one definition per
`VarId`), def-before-use under dominance over the block CFG, and operand
typing wherever `var_types` records both sides; `KernelIR::verify` and
`KernelIR::is_well_formed` are the same check, and `lower_kernel_to_ir`
refuses a kernel that fails it with every violation listed. A loop-carried
value is a block parameter (roadmap A2 step 2): `KirBlock::params`,
`KirBuilder::add_block_param`, and edges (`KirEdge { target, args }`, the
payload of `KirTerminator::Branch` / `CondBranch`; `b.into()` is an edge
with no arguments) pass a value per parameter — verifier rule 7 holds the
argument count and types to the target's parameters and refuses parameters
on the entry block, and the PTX printer implements an edge as a parallel
copy into the parameter registers before the jump (a per-class
`%edge_*` scratch register breaks a swap; a `CondBranch` whose edges carry
arguments gets a `BB<n>_else` label). `crates/nsl-codegen/tests/kir_block_params_ptxas.rs`
assembles a grid-stride loop with `ptxas`. The scalar ISA the hand estate
is made of is first-class too (roadmap A2 step 4): `And`/`Or`/`Xor`/`Not`,
`Shl`/`Shr` (arithmetic for signed types), `Rem`, `Min`/`Max`,
`Rcp`/`Rsqrt`, `WarpShuffle { mode: Down | Up | Xor | Idx, width }`,
`Vote { Any | All | Ballot }`, `LaneId`/`WarpId`, `LoadVec`/`StoreVec`
(2 or 4 pointee-typed values), `CastRounded { mode }` beside `Cast`
(which now prints the rounding modifier PTX requires: `.rn` for a float
result that can round, `.rzi` for float → int), and `Predicated { pred,
negate, op }` around an op with no destination (a predicated definition is
refused as partial SSA). The 16-bit class is `.reg .b16 %h<N>` and 16-bit
loads and stores are `.b16`; a kernel that requires
`FeatureSet::BF16_ARITHMETIC` (any bf16 value or conversion) prints
`.version 7.8` / `sm_80`. `crates/nsl-codegen/tests/kir_scalar_isa_ptxas.rs`
assembles one kernel using every family. Registers are allocated
(roadmap A2 step 5, `crates/nsl-kir/src/regalloc.rs`): each value gets a
class from its type (`RegClass::of`: `%r`/`%rd`/`%f`/`%fd`/`%h`/`%p`/`%v`)
and a dense index by linear scan over live intervals on the block-order
linearisation — a kernel parameter is live from before block 0, a block
parameter from its block's entry and at every incoming edge's terminator,
and a value to its last use and the end of every block it is live-out of,
so a loop-carried value keeps its register through the loop; the printer
emits `%<class><VarId>` names and renames them through the `Allocation`
as a last pass, declares each class at its allocated count (the
`dst + 1000` scratch idiom is gone: `GlobalId` uses `%gid0`/`%gid1`), and
prints `.maxntid` / `.minnctapersm` / `.maxnreg` from
`KirBuilder::set_launch_bounds` / `set_max_registers`.
`KernelIR::register_pressure()` is the per-class count. The async-copy
group is first-class KIR (`SharedBase`, `CpAsync { bytes: 4 | 8 | 16 }`,
`CpAsyncCommit`, `CpAsyncWait { pending }`, `FeatureSet::ASYNC_COPY`): the
verifier checks the global → shared state spaces and the commit/wait
discipline per block, and the PTX backend lowers them to `cp.async` under an
`sm_80` target (every other kernel keeps `sm_70`).
The tensor-core ops are first-class too (`LdMatrixX4`, `MmaF16M16N8K16`,
`FeatureSet::TENSOR_CORES`): fragments are `Vec(F16, 2)` (one packed `.b32`
register each) and `F32` accumulators, held to those types by the verifier,
and the PTX backend declares the `.reg .b32 %v<N>` class for them.
`crates/nsl-codegen/tests/kir_async_copy_ptxas.rs` and
`crates/nsl-codegen/tests/kir_mma_ptxas.rs` assemble those lowerings with
`ptxas` where the toolkit is present (CI's cuda-feature lane).
`crates/nsl-kir/src/backend_ptx.rs::lower_kir_to_ptx` prints PTX
(ISA 7.0, `sm_70`) from it; `src/backend_amdgpu.rs::lower_kir_to_amdgpu`,
`src/backend_metal.rs::lower_kir_to_msl`, `src/backend_wgsl.rs::lower_kir_to_wgsl`
are the other printers. `src/kernel_lower.rs::lower_kernel_to_ir` lowers a
user `kernel` block's AST to KIR for the portable subset and refuses
everything else. `src/gpu_target.rs` (`GpuTarget::{Cuda, Rocm, Metal, WebGpu,
Fpga}`, re-exporting `FeatureSet`) selects the backend; `Compiler::compile_kernels`
(`src/compiler/kernel.rs`) dispatches: CUDA still goes to the AST→PTX
`KernelCompiler` (`src/kernel.rs`), ROCm/Metal/WebGPU go through KIR, and
`Fpga` returns `FPGA_TARGET_REDIRECT_MSG` (use `nsl fpga-compile`). PTX bytes
are embedded via `declare_data` / `define_data` in the same file.
`crates/nsl-codegen/tests/common/kir_builder.rs` is the shared test helper for building KIR;
`crates/nsl-codegen/tests/snapshot_tests.rs` pins KIR-generated PTX.

**Hand-written PTX emitters (frozen).** `src/flash_attention.rs`
(`synthesize_flash_attention_ptx`, `synthesize_flash_attention_backward_ptx`),
`src/flash_attention_v2/` (`synthesize_flash_attention_ptx_v2`, `phases/`,
`tier_b1/`, `tier_b2/`, `mma_forward.rs`, `per_doc_cta.rs`, `sinks.rs`),
`src/flash_attention_selector.rs`, `src/fused_linear_ce.rs`
(`synthesize_fused_linear_ce_ptx` and the large-vocab v2 pair),
`src/matmul_mma.rs` (MMA fragment primitives), `src/moe_kernels.rs`,
`src/precision_cast_ptx.rs`, `src/wrga_fused_ptx.rs`,
`src/cpkd_fused_loss.rs`, `src/bitnet/`, `src/pca_rope.rs`,
`src/pca_tilerange.rs`, `src/cfie_*_ptx.rs`, `src/cfie_decode_attention.rs`,
`src/fusion.rs` (elementwise chains), `src/kernel.rs`, and the shared preludes
in `src/kernel_skeleton/` (`header.rs`, `indexing.rs`, `pad.rs`, `params.rs`,
`smem.rs`) all `push_str` PTX text with hand-numbered registers.

**The freeze (roadmap A2).** `ci/hand-ptx-manifest.txt` lists every file that
writes PTX into a string (71 members at the 2026-09-02 freeze: the codegen
files above plus seven under `crates/nsl-runtime/src/cuda/` and
`crates/nsl-runtime/src/flash_attention.rs`; `backend_ptx.rs` is the one
member that belongs by construction). `scripts/hand-ptx-freeze.sh --check`
(membership decided by `scripts/hand-ptx-scan.awk`; `--list`, `--explain`,
`--write-manifest`, `--self-test`) fails CI (`hand-ptx-freeze` job in
`.github/workflows/ci.yml`) if a file joins the set or a listed file no
longer emits. Existing members may still be edited — it is a gate on the file
set, not a line count. **A new kernel must therefore be built as `KernelIR`
and lowered through `backend_ptx.rs`; a new `.rs` file that formats PTX text
will not merge.** If a kernel needs something KIR cannot express, extend
`KirOp` and the printers rather than adding a hand emitter. What KIR still
cannot express, and the order the frozen files migrate in, is the design
spec `docs/superpowers/specs/2026-09-09-a2-kir-v2-design.md`.

**Supporting pieces.** `src/gpu_specs.rs` — `GpuSpec` (`sm_version`, peak
TFLOPs, bandwidth, VRAM, L2, crossover points, launch overhead),
`GPU_DATABASE`, `find_gpu`, `default_gpu`, `resolve_local_gpu`
(via `nsl_runtime::CudaDeviceIdentity`), plus `FPGA_DATABASE` / `CPU_DATABASE`.
`src/ptxas_validation.rs::validate_ptx` assembles PTX through `cudarc`
`cuModuleLoadData` when a context is current, else `nvcc --cubin`; it is the
basis of every `*_ptxas*.rs` test. `src/ptx_metadata.rs` extracts static
kernel metadata; `src/autotune.rs` benchmarks kernel variants at build time.
SASS baselines: `crates/nsl-codegen/tests/sass_baseline_helpers.rs` (`Baseline {variant_name, sm,
instruction_count, spill_bytes, tolerance}`) reads the files under the
workspace-level `tests/sass_baselines/` for `crates/nsl-codegen/tests/pca_tier_b_sass_baselines.rs`;
`crates/nsl-codegen/tests/bitnet_sass_discipline.rs`, `crates/nsl-codegen/tests/tier_b1_kernel_sass_no_spill.rs`,
`crates/nsl-codegen/tests/tier_b2_dq_kernel_sass_no_spill.rs` and `crates/nsl-codegen/tests/pca_sass_byte_identity.rs`
are the other SASS-level gates. All need a device and run in the local GPU
lane, not hosted CI.

## Autodiff

NSL has two autodiff modes, selected by `CompileOptions::source_ad`
(`--source-ad` on `nsl run`/`nsl build`; `exec_fingerprint` records
`ad=source|tape`):

- **Tape AD** (default): the runtime records operations on a thread-local
  `TAPE` (`crates/nsl-runtime/src/autodiff/`), and the train block emits
  `backward`/optimizer calls against it. Codegen's role is small — it emits
  the calls and the parameter lists.
- **Source-to-source AD** (`--source-ad`): `WengertExtractor`
  (`src/source_ad.rs`, `extract_stmts`) turns the forward body into a
  `WengertList` of `WengertOp { … PrimalOp … }` (`src/wengert.rs`, which also
  owns the primal/adjoint `IdSpace` split — `adjoint_op_id`,
  `renumber_adjoint_ops`, `assert_ids_in_space`; `crates/nsl-codegen/tests/adjoint_id_space_gate.rs`
  pins that renumbering happens only there). `AdjointGenerator::generate`
  (`src/source_ad.rs`) applies `ad_rules::apply_ad_rule` per op
  (`AdjointExpr`, `InputAdjoint`, `saved_for_backward`), runs dead-gradient
  elimination (`eliminate_dead_gradients`, `eliminate_by_backward_live`) and
  the rmsnorm/SwiGLU adjoint fusions, and `wengert_lower::compile_wengert_ops`
  lowers every op to runtime FFI calls, returning `LoweredWengert`
  (a `VarId → Value` map plus `ParamGradSource`s). Every in-pipeline pass in
  the train block operates on this list.

**Refusal discipline.** Source AD does not fall back silently: a callee or
construct the extractor cannot represent is an `Err(CodegenError)` naming the
construct (`"[source-ad] unsupported callee expression"` and the
`wengert_lower.rs` `Err(CodegenError::new(...))` sites are the pattern), and
feature compositions that were never validated under it refuse at admission
(`crates/nsl-codegen/tests/fp8_source_ad_refusal.rs`, `crates/nsl-codegen/tests/source_ad_diagnostics.rs`,
`crates/nsl-codegen/tests/csha_r13_n1p0_refusal.rs`, `crates/nsl-codegen/tests/sinks_v1_backward_refusal.rs`). The
CLI side registers each such refusal in
`crates/nsl-cli/src/feature_rules.rs` (`FeatureRule { flag, kind: RuleKind,
other, enforcement: Enforcement::{Clap, Source { file, fragment }} }`), and
`crates/nsl-cli/tests/feature_composition_gate.rs` cross-checks the registry
against clap attributes and the refusal text in the named file.

**FASE** (Fused Accumulation + Step + Epilogue): `src/fase.rs::plan(&FaseConfig)
-> FasePlan` (with `FaseMode`, `FaseOptimizer`, `UpdateRecipe`, `BackwardPhase`)
decides per train block whether gradient accumulation and the optimizer step
fuse into per-parameter recipes (`src/fase_optimizer.rs`, `src/fase_clip.rs`,
`src/fase_memory.rs`, `src/fase_codegen_table.rs`). `src/stmt_fase.rs` emits
the Deferred-mode arms inside the train block (the `m_partial += (1/N)·g`
accumulate and the post-loop fused step); `src/wgrad_fusion.rs` (Item 7,
`--fuse-wgrad-accum`) hooks the same path to collapse weight-gradient chains
into an accumulating GEMM. `src/ew_chain_fusion.rs::run_backward_ew_fusion`
fuses elementwise runs on the adjoint tape. `src/training_report.rs`
(`nsl check --training-report`) runs the FASE and PCA planners without
emitting code. Tests: `tests/fase_*.rs` (eight files), `crates/nsl-codegen/tests/train_clif_snapshots.rs`
fixtures `tests/train_clif/*.nsl`.

## Analysis and fusion passes

- **Elementwise fusion** — `src/fusion.rs` (`FusedKernel`, `is_fusible_op`,
  `is_fusion_barrier`, `synthesize_fused_ptx`; `let`-binding is the fusion
  barrier), `src/fusion_report.rs` (`--fusion-report`, `FusionEvent`,
  `BarrierReason`, collected in `compiler.fusion: FusionState`).
  `src/ew_chain_fusion.rs` (adjoint tape) and `src/wgrad_fusion.rs` are
  training-side fusions; `src/wrga_fusion.rs` is WRGA's epilogue-fusion plan.
  The M31 modules `epilogue_fusion.rs` / `reduction_fusion.rs` no longer
  exist (see summary 02).
- **Cost model** — `src/cost_model.rs` (M37 roofline: `OpCost`,
  `BoundClassification`, `matmul_cost`, `softmax_cost`, …), consumed by
  autotune's cost-model selection, WRGA's roofline (`src/wrga_roofline.rs`),
  WGGO's cost (`src/wggo_cost.rs`), CFIE's (`src/cfie_cost.rs`) and
  `src/wcet.rs`.
- **Memory planner** — `src/memory_planner.rs`: `analyze_ast_liveness`
  → `TensorAlloc`s, `InterferenceGraph::build`, `plan_slab` → `SlabPlan`,
  `format_memory_report` (`--memory-report`), `check_vram_budget`
  (`--vram-budget`), and `apply_wrga_hints` / `consume_hints` for the WRGA
  channel. Scheduled as `"MemoryPlanner"` twice: the whole-program plan in
  `entry_points.rs` and the transient arena inside the train block
  (`src/transient_arena.rs`, `--transient-arena`).
- **Autotune** — `src/autotune.rs` (`@autotune`: `cartesian_product`,
  `measure_variants`, `find_best_variant`, `find_best_variant_cost_model`,
  `AutotuneCacheRecord` keyed by `DeviceIdentity` under `.nsl-cache/autotune/`);
  `crates/nsl-codegen/tests/autotune_cache_identity.rs`, `crates/nsl-codegen/tests/autotune_frozen_db.rs`.
- **Calibration** — `src/calibration/` (compile-time AWQ / WGGO-gradient
  calibration: `discovery.rs`, `hooks.rs`, `awq_hook.rs`,
  `wggo_gradient_hook.rs`, `retention_pass.rs`, `binary_codegen.rs` emits a
  `calibration_main()` binary, `subprocess.rs` runs it, `sidecar.rs` /
  `awq_sidecar.rs` carry results back). Entry: `compile_and_calibrate`
  (`src/lib.rs`); tests `tests/calibration_*.rs`, `tests/awq_*.rs`.
- **Weight-aware compilation** — `src/weight_aware.rs` (M52 constant folding
  from `--weights`), `src/ctor_fold.rs`, `src/lm_head_inference.rs`
  (`--fuse-lm-head`), `src/param_roles.rs`, `src/parameter_plan.rs`.
- **Profiling / inspection** — `src/profiling/` (`captures.rs`,
  `instrument.rs`, `walker.rs`, `memory_timeline.rs`), `src/inspect/`,
  `src/wcet.rs`.

**Pass registry drift gate.** `crates/nsl-codegen/tests/pass_registry_drift.rs` checks
`PASSES` against the tree in both directions: every `source_files` entry
exists, every full name appears in its own source, every registered CLI flag
is on exactly its declared subcommands, every `docs/wiki/Optimization-Passes.md`
section is registered and vice versa, and — the direction that matters for
new modules — `every_codegen_pass_module_belongs_to_a_registered_pass`. A
module under `src/` whose name is not a registered pass's prefix must be
classified in that test's `NOT_A_PASS` table with a reason (`ad_rules`,
`autotune`, `backend_`, `builtins`, `c_header`, … each carry one), and
`not_a_pass_exclusions_are_justified_and_live` fails when an exclusion no
longer matches anything. So adding a module means either registering a pass
or explaining why it is not one.

## Error model

`CodegenError` (`src/error.rs`, `#[non_exhaustive]`): `message: String`,
`span: Option<Span>`, `notes: Vec<String>`. Constructors and combinators:
`new`, `with_span` (override), `with_span_if_unset` (attach only if none and
the span is not `Span::DUMMY`), `with_note`, `to_diagnostic()` (→
`nsl_errors::Diagnostic` with the span as primary label and notes carried
over; the CLI renders it through the source map), `missing_scales`. `Display`
is the bare message — the CLI adds the `codegen error:` prefix.

Spans are attached by exactly four dispatchers, each wrapping its
`*_dispatch` twin with `.map_err(|e| e.with_span_if_unset(node.span))`:
`Compiler::compile_stmt` (`src/stmt.rs`), `Compiler::compile_expr`
(`src/expr/mod.rs`), and `KernelCompiler::compile_stmt` /
`KernelCompiler::compile_expr` (`src/kernel.rs`). Because the innermost node
runs first, a helper deep in a lowering can raise `CodegenError::new(msg)`
with no span and still be reported at the right expression. Errors raised
outside statement compilation (model collection, kernel synthesis, the WGGO
prepass, `main` assembly) have no span unless the site attaches one —
`src/wggo_prepass.rs` does so with `model_arg_span`.

**No `process::exit` in the library** (PR #560, "typed, spanned CodegenError;
no process::exit in the codegen library", roadmap C1). Every refusal is a
returned `Err` the CLI renders; the only `process::exit` calls under `src/`
are in `src/bin/` binaries. `crates/nsl-cli/tests/codegen_error_rendering.rs`
pins the behaviour end to end (a spanned error renders file:line:col plus an
excerpt; an infeasible WGGO budget and a wrong CPDT checkpoint are rendered
refusals that write no artifact), and the WGGO unit test
`infeasible_budget_is_a_returned_compile_error_not_an_exit` (`src/wggo.rs`)
pins the site that used to exit. Diagnostic *messages* on stderr are a
different thing: they are the execution markers (below) and are fine.

**Diagnostics go through `nsl_log::nsl_log!`** (roadmap C3; the front
door and its byte-identical stderr subscriber are the `nsl-log` crate,
described in runtime.md, "Logging"). A compile-time warning, note or
marker line is `nsl_log::nsl_log!(LEVEL, "target", "…")` rather than
`eprintln!`: the
`warning:` / `error:` / `note:` lines use target `codegen`, a line that
starts with its own `[marker]` uses that marker (`autotune`, `ccr`,
`source-ad`, `wggo`, `cpdt`, `arena`, `weight-stream`, …), and the levels
follow the runtime's rule (`ERROR` for a lost result, `WARN` for a refusal
or fallback, `INFO` for reports and traces). `nsl-log` depends on
`tracing` alone, so the diagnostics are no longer a reason for this crate
to depend on the runtime (roadmap A3). The multi-line report dumps that `eprint!` a pre-rendered string
(`plan.render_report()`, the linker's tool output) and the dev-tool
binaries under `src/bin/` are the only raw prints left.

## Experimental subsystems

The `experimental` facade (`src/lib.rs`) and STATUS.md's Experimental tier
hold the research subsystems; each has a driver module and a family of
`<prefix>_*.rs` helpers, is registered in `pass_registry.rs` when it is a
pass, and is described by a paper or note under `docs/research/`. Entry
modules: **WGGO** `src/wggo.rs` (+ `wggo_prepass.rs`, `wggo_apply.rs`,
`wggo_overrides.rs`, …), **WRGA** `src/wrga.rs` (+ `wrga_prescan.rs`,
`wrga_adapter_*.rs`, `wrga_fused_ptx.rs`, …; gated by the `experimental-wrga`
Cargo feature at its `stmt.rs` entry), **CEP** `src/cep.rs`, **CFIE**
`src/cfie.rs` (+ `cfie_serve.rs`, `src/serve.rs`), **CSHA** `src/csha.rs`
(+ `csha_apply.rs`, `csha_pipeline.rs`, …), **CPDT** `src/cpdt.rs`
(+ `cpdt_decorator.rs`, `cpdt_zero.rs`, …; `experimental-cpdt` feature),
**CPKD** `src/cpkd.rs`, **CCR** `src/ccr.rs`, **CSLA** `src/layerwise.rs` +
`src/stmt_csla.rs` (`docs/research/CSLA-compiler-scheduled-layerwise-accumulation.md`),
**PCA** `src/pca_detect.rs` (+ `pca_tier_b.rs`, `pca_per_doc.rs`, …),
**FASE** `src/fase.rs`, **ZK** `src/zk/` (`nsl zk`, Plonky3 and folding
backends), **FPGA/HIR** `src/hir/` + `src/backend_verilog/` +
`src/kernel_lower_fpga.rs` + `src/fpga_error.rs` (`nsl fpga-compile`),
**WCET** `src/wcet.rs`, **unikernel** `src/unikernel.rs`, **sparse**
`src/sparse.rs`, **speculative** `src/speculative.rs`, **multimodal**
`src/multimodal.rs`, **BitNet** `src/bitnet/`. Their APIs, flags and on-disk
formats are not stable; see `STATUS.md` ("Experimental" and "Opting out of
experimental subsystems") for the tier contract and `docs/wiki/Optimization-Passes.md`
("Per-pass descriptions") for what each pass does. This document does not
describe their internals.

## Invariants

Each of these is enforced by a named test or gate; a change that breaks one
should fail before review.

- **Emission is deterministic.** Struct constructors are defined in name
  order (`Compiler::struct_layouts` is a `BTreeMap`; PR #570 "define struct
  constructors in name order, not hash order") and `FuncState::variables` is
  walked in name order in `stmt.rs` / `ccr.rs` (commit "walk
  FuncState::variables in name order, not HashMap order"). The gate is
  `crates/nsl-codegen/tests/train_clif_snapshots.rs`, which compiles every fixture twice and
  reports a difference between the two dumps as nondeterminism before it
  compares against the snapshot. New emission must not iterate a `HashMap`
  into IR.
- **Train-block CLIF is snapshot-stable.** 26 `insta` snapshots under
  `tests/snapshots/train_clif_snapshots__*` (fixtures `tests/train_clif/*.nsl`:
  `mlp_adamw`, `sgd`, `lion`, `muon`, `csla_ffn`, `csla_ffn_muon`,
  `dataloader_checkpoint`; the call-conv token is normalised to `<callconv>`).
  Every phase peeled out of `compile_train_block_inner` is a byte-for-byte
  move under them; a moved snapshot under a refactor that meant to change
  nothing is the finding. Review with `cargo insta review -p nsl-codegen`.
  (`src/stmt_admission.rs` says "28 snapshots"; the directory holds 26.)
- **stderr markers are an API.** Subsystems announce themselves with
  bracketed tags (`[csla]`, `[zero3]`, `[wggo]`, `[pass-trace]`, …) that
  integration tests assert on. The vocabulary is
  `crates/nsl-cli/src/exec_markers.rs` (`EXEC_MARKERS` with `emitted_by`
  files, `NEGATIVE_NEEDLES`, `EVENT_SCHEMAS`); the gate
  `every_exec_marker_is_still_emitted_by_its_source` in
  `crates/nsl-cli/tests/feature_composition_gate.rs` fails when a listed file
  stops printing its token. Renaming a marker means updating the registry and
  every assertion, not just the `eprintln!`.
- **Deferral must refuse.** An unsupported composition, an unlowerable
  construct, or a flag that did nothing produces a loud `Err`, never a silent
  fallback — `src/kernel.rs` header, `src/kernel_lower.rs`,
  `src/stmt_admission.rs`, the `wgrad_hook_blocks` refusal in
  `compile_main`, and eighteen `src/` files invoke the rule by name. Refusals
  are pinned by message text in `feature_rules.rs` +
  `feature_composition_gate.rs` (CLI) and the `*_refusal*.rs` /
  `*_gate.rs` tests here.
- **Registry entry ↔ runtime `extern "C"` ↔ emission are in lockstep.** A
  runtime function appears exactly once in the typed table
  (`no_runtime_function_is_declared_twice`, `nsl_abi::table::tests`), the
  codegen's rendering is the table row for row (`registry_is_the_abi_table`),
  and every row's signature agrees with the runtime's implementation — checked
  by `rustc` in the runtime's build (`abi_check.rs`) and by text in
  `crates/nsl-abi/tests/signature_agreement.rs`. The emitted argument order
  must match the table's parameter order — nothing checks that except the
  snapshot and numerical tests, which is why a new call site should be
  covered by one. `crates/nsl-codegen/tests/muon_route_contract_drift.rs` and
  `crates/nsl-codegen/tests/ffi_ownership_drift.rs` pin two contracts that straddle the boundary.
- **Every in-pipeline pass runs through the scheduler.** A pass with a
  non-empty `phases` declaration is invoked only inside
  `PassScheduler::schedule` and settles with `finish(&bus)`
  (`crates/nsl-codegen/tests/pass_scheduler_coverage.rs`); the value order the bus declares is
  enforced at the train block's exit (`enforce_dependency_order`); a
  positional tape reference is consumed only after
  `assert_tape_unchanged_since`; `applied ⇒ published` holds for every
  `Enforced` channel. New readers of `begin_epoch` / `current_epoch` outside
  `pass_manager.rs` fail the coverage gate.
- **Inter-pass values travel on the bus, not on ad-hoc `Compiler` fields.**
  Every `PassBus` field is a declared `Channel` and every channel reader is a
  declared consumer (`crates/nsl-codegen/tests/pass_bus_drift.rs`).
- **No new thread-locals.** State goes on `Compiler`, `CompileOptions`, or
  the bus; a new `thread_local!` anywhere in the workspace must be added to
  the inventory in `docs/architecture/compiler-state.md` with a class and
  reason (`thread_local_inventory_drift.rs` in `nsl-runtime`'s tests).
- **No new hand-PTX files.** `ci/hand-ptx-manifest.txt` only shrinks
  (`scripts/hand-ptx-freeze.sh --check`).
- **The C header describes the real ABI.** `crates/nsl-codegen/tests/c_header_agreement.rs`,
  `crates/nsl-codegen/tests/c_header_compiles.rs`, and the `NSL_ABI_VERSION_*` constants come
  from `nsl_runtime::c_api`.

## Tests and gates

`crates/nsl-codegen/tests/` holds 301 `.rs` integration-test files (plus
`common/`, `data/`, `fixtures/`, `snapshots/`, `train_clif/`). By filename
prefix the largest families are `tier_*` (45, FA-v2 Tier B1/B2 kernels),
`csha_*` (34), `pca_*` (29), `fused_*` (25, fused linear-CE and LM head),
`wggo_*` (14), `cpdt_*` (13), `bitnet_*` (11), `wrga_*` (10), `cfie_*` (10),
`pass_*` (8), `fase_*` (8), `fa_*` (6), `sinks_*` (5), `awq_*` (5),
`profiling_*`/`profile_*` (8), `bench_*` (4), plus the ABI/export, HIR/Verilog,
calibration, autotune and drift-gate singles. Functionally they fall into:

- **Snapshot suites** (`insta`, `cargo insta review -p nsl-codegen`):
  `train_clif_snapshots.rs` (CLIF, 26 snapshots), `fa_v2_snapshots.rs`
  (per-phase FA-v2 PTX, 25 tests), `verilog_emission_snapshots.rs` (8),
  `hir_pass_snapshots.rs` (7), `snapshot_tests.rs` (KIR-generated PTX/KIR,
  12), `bitnet_ptx_snapshots.rs`, `pca_*_kernel_snapshot.rs`,
  `tier_b1_*_snapshot.rs`, `csha_pipeline_cost_model_snapshot.rs`,
  `cpdt_sensitivity_snapshot.rs`, `c_header_snapshot.rs`, and the byte-identity
  pins (`fused_linear_ce_v1_byte_identity.rs`, `sinks_v1a_byte_identity.rs`,
  `pca_sass_byte_identity.rs`). 102 snapshot files under `crates/nsl-codegen/tests/snapshots/`.
- **Static drift gates** (read the tree, no compile): `pass_registry_drift`,
  `pass_bus_drift`, `pass_manager_drift`, `tape_access_drift`,
  `pass_scheduler_coverage`, `ffi_ownership_drift`,
  `activation_contract_static_gate`, `adjoint_id_space_gate`,
  `muon_route_contract_drift`, `c_header_agreement`.
- **Compile-and-inspect** (CPU, `compile_with_options` / `compile_entry_capturing_ir`):
  refusal pins, decorator plumbing, `*_lowering_end_to_end`, `*_ffi_decls`,
  `*_emission`, `health_codegen`, `training_report_test`.
- **ptxas gates** (`ptxas_validation::validate_ptx`; need `nvcc` or a
  device): `*_ptxas*.rs`, `*_ptxas_validation.rs`, `*_ptxas_probe.rs`.
- **GPU numerical / parity / SASS** (need a device; `#[ignore]` with a
  reason): `*_gpu_parity.rs`, `*_numerical.rs`, `*_gpu_e2e.rs`,
  `*_sass_*.rs`, `flash_attention_*_gpu.rs`. 72 files carry `#[ignore]`
  (251 attributes); `scripts/gpu-cert.sh --check-reasons` refuses a bare one.
- **Hardware-adjacent toolchains**: `yosys_gate.rs` (FPGA, skipped without
  `yosys`), `c_header_compiles.rs` (a C compiler),
  `awq_real_subprocess_link.rs` / `exported_symbols_are_dlsym_findable.rs`
  (the system linker).

**Bench.** `benches/compile.rs` is the criterion compile-latency bench
(`cargo bench -p nsl-codegen --bench compile`; CPU-only, discovers the
import-free examples and measures only codegen). `.github/workflows/bench.yml`
runs `scripts/bench.sh` on PRs carrying the `bench` label. The `bench`
*binary* (`src/bin/bench.rs`, `required-features = ["cuda",
"debug_kernel_instrumentation"]`) is the PCA Tier B GPU harness — a
different thing.

**What runs where.**

| Gate | Where |
|------|-------|
| `cargo test --workspace -- --skip e2e_` (all non-ignored codegen tests, unit tests, the static drift gates, `signature_agreement`) | `ci.yml` `build-and-test` |
| `verilog_emission_snapshots`, `hir_pass_snapshots`, `yosys_gate` | `ci.yml` `fpga` |
| `csha_ptx_ptxas_validation`, `fused_linear_ce_{bf16,fp16,large_vocab}_ptxas`, `bitnet_gpu_correctness` under `--features cuda` against cudart stubs (assembles PTX, executes nothing) | `ci.yml` `cuda-feature` |
| `scripts/hand-ptx-freeze.sh --self-test` / `--check` | `ci.yml` `hand-ptx-freeze` |
| `scripts/gpu-cert.sh --check-inventory` / `--check-reasons` / `--check-long-arms` (manifest `ci/gpu-cert-manifest.tsv`: 471 gates, 198 in this crate) | `ci.yml` `gpu-gate-inventory` |
| `scripts/check-doc-agreement.sh`, version agreement | `ci.yml` `doc-agreement`, `version-agreement` |
| `scripts/gpu-cert.sh --run [--tier gpu\|toolchain\|multiproc\|isolate\|all]` — every `#[ignore]`d device test, under `scripts/gpu-guard.sh`; known-red list `ci/gpu-cert-known-red.txt` | `.github/workflows/gpu-cert.yml`, nightly + `workflow_dispatch` on the self-hosted sm_120 box; or locally |
| `scripts/gpu-tier.sh smoke\|certify\|endurance` | local only |

Hosted runners have no GPU: anything that executes a kernel is certified by
the local lane, and a kernel-heavy PR should dispatch `gpu-cert.yml` after
review. See `docs/wiki/GPU-Test-Harness.md` and `docs/wiki/Testing-Strategy.md`.

## Where to add a new X

### A new runtime function call

1. Implement the `#[unsafe(no_mangle)] extern "C" fn` in `nsl-runtime`
   (C-ABI scalars only: `i64`/`f64`/pointers-as-`i64`).
2. Add one row `[group] nsl_…(i64, …) -> i64 = module::path::nsl_…;` to
   `crates/nsl-abi/src/table.rs`, in the group its subject belongs to
   (`memory`/`io`/`scalar`/`collections`/`tensor` if a program calls it by
   name, else `abi_tensor`/`training`/`optimizer`/`distributed`/`inference`/
   `quantization`/`diagnostics`/`abi_memory`/`interop`; `[interop]` after
   the path if the implementation is behind the runtime's `interop`
   feature). Position within the group is free. Nothing else is edited: the
   codegen declares it and the runtime's build checks it.
3. Emit the call with `self.compile_call_by_name(builder, "nsl_…", &args)`
   from the lowering site; argument order must match the row.
4. Build `nsl-runtime` (the row is checked against the implementation at
   compile time) and run `cargo test -p nsl-abi --test signature_agreement`
   and `cargo test -p nsl-codegen --lib builtins` (the duplicate/rendering
   tests). If the symbol is exported to C hosts, `crates/nsl-codegen/tests/c_header_agreement.rs`
   and `crates/nsl-codegen/tests/c_header_compiles.rs` cover the header; add the prototype in
   `src/c_header.rs` if the header must expose it.
5. If the function takes or returns tensor ownership, classify it in
   `src/ffi_ownership.rs` (`crates/nsl-codegen/tests/ffi_ownership_drift.rs`).
6. Cover the emission with a compile-and-inspect test (`*_ffi_decls.rs` /
   `*_emission.rs` style) or a CLIF snapshot; if the call prints a marker,
   register it in `crates/nsl-cli/src/exec_markers.rs`.

### A new compile option or flag

1. Add the field to `CompileOptions` in `src/lib.rs` — inside the matching
   sub-struct (`WggoOptions`, `CfieOptions`, `WcetOptions`, `ZkOptions`,
   `CshaOptions`, `CpdtOptions`, `CalibrationOptions`, `DevToolsOptions`, `CheckpointOptions`,
   `WeightStreamOptions`, `MuonOptions`, `ImportedModelOptions`, `ZeroOptions`,
   `AutotuneOptions`, `WeightsOptions`, `FusionOptions`, `DiagnosticsOptions`,
   `MemoryOptions`, `WrgaOptions`, `AnalysisOptions`, `ExportOptions`,
   `TrainOptions`, `DeterminismOptions`, `MatmulConfig`) when one exists — with its
   default in `impl Default for CompileOptions` (or the sub-struct's).
2. Declare the clap flag in `crates/nsl-cli/src/args.rs`. Shared flags are
   declared twice (`BuildArgs`, `RunArgs`) and must be identical; a flag
   group shared by both belongs in a flattened struct like
   `crates/nsl-cli/src/matmul_args.rs`. Thread it in
   `crates/nsl-cli/src/commands/build/options.rs` and `commands/run.rs`.
3. If the flag changes arithmetic or placement, add a key to
   `CompileOptions::exec_fingerprint` (fixed position, closed vocabulary,
   emitted even when false) and its class to
   `crates/nsl-runtime/src/exec_fingerprint.rs`; extend
   `exec_fingerprint_tests` in `src/lib.rs`.
4. If it composes with other features only in validated combinations, add
   the refusal in code (an `Err(CodegenError)` at admission, e.g.
   `src/stmt_admission.rs`) and register it as a `FeatureRule` in
   `crates/nsl-cli/src/feature_rules.rs` (`Enforcement::Source { file,
   fragment }` or `Enforcement::Clap`) so `feature_composition_gate.rs` pins
   it.
5. If the flag drives a registered pass, add it to that pass's `cli_flags`
   in `src/pass_registry.rs` (`every_registered_cli_flag_is_on_exactly_its_declared_subcommands`).
6. Document it in `docs/wiki/CLI-Reference.md` / `Environment-Variables.md`
   (the `doc-agreement` job checks docs against the tree).

### A new train-block phase (or peeling one out of `compile_train_block_inner`)

1. Run `cargo test -p nsl-codegen --test train_clif_snapshots` first so the
   baseline is green; if the phase has no fixture under `crates/nsl-codegen/tests/train_clif/`,
   add one and accept its snapshots before touching the driver.
2. Create `src/stmt_train/<phase>.rs` (declare it in `src/stmt_train/mod.rs`
   and add the module to its doc list in driver order) exposing one
   `pub(crate) fn` on `impl Compiler<'_>` (or a free function taking
   `&mut Compiler`, `&mut FunctionBuilder`, `&mut FuncState`) whose
   signature names the locals the block actually reads — that naming is the
   point of the peel. Emit-only helpers with their own subject go beside it
   (`src/stmt_csla.rs`, `src/stmt_fase.rs`); read-only admission goes in
   `src/stmt_admission.rs`.
3. Move the code byte-for-byte; the CLIF snapshots must not move. If the
   phase is a pass (it plans and publishes), invoke its planner under
   `self.passes.scheduler().schedule("NAME", tape, || …)` and settle with
   `.finish(&self.bus)`; register the pass (below) first.
4. Re-run `train_clif_snapshots`, `pass_scheduler_coverage`, and the CLI
   composition gates (`feature_composition_gate`, `zero3_gate`,
   `zero_spmd_gate`, `muon_state_gate` in `crates/nsl-cli/tests/`) — they
   pin refusal text by file, so a moved refusal must have its `file` updated
   in `feature_rules.rs`.

### A new GPU kernel (KIR path only)

1. Build it with `KirBuilder` (`crates/nsl-kir/src/kernel_ir.rs`): `add_param` for each
   argument with its `AddressSpace`, `new_typed_var`/`emit(KirOp::…)` for the
   body, `terminate`, `finalize()`. Missing operations are added as `KirOp`
   variants with lowering in **every** printer (`backend_ptx.rs`,
   `backend_amdgpu.rs`, `backend_metal.rs`, `backend_wgsl.rs`) and, if
   needed, `FeatureSet` bits in `src/gpu_target.rs`.
2. Lower with `backend_ptx::lower_kir_to_ptx` at the launch site and embed
   the bytes the way `Compiler::compile_kernels` does (`declare_data` /
   `define_data`, `src/compiler/kernel.rs`); launch through the existing
   runtime FFIs. Do **not** add a module that formats PTX text —
   `scripts/hand-ptx-freeze.sh --check` refuses it; if you must touch a
   frozen emitter, edit the existing member file.
3. Pin the PTX with a snapshot in `crates/nsl-codegen/tests/snapshot_tests.rs` (using
   `crates/nsl-codegen/tests/common/kir_builder.rs`) and assemble it in a `*_ptxas.rs` test
   through `ptxas_validation::validate_ptx`.
4. Add the device test (`#[ignore = "<reason>"]`, GPU parity against a CPU
   reference) and refresh `ci/gpu-cert-manifest.tsv` with
   `scripts/gpu-cert.sh --write-manifest`; run it in the local lane. If
   register pressure matters, add a SASS baseline under
   `tests/sass_baselines/` and a `*_sass_no_spill.rs` gate.
5. If the kernel is selected by a decision (dtype, shape, flag), the decline
   path must be a refusal or a recorded `PassDisposition`, not a silent
   fallback.

### A new optimization pass

1. Write the pass as a module with a `plan`-style entry that takes
   `&mut Compiler` (or the `WengertList` it scans) and returns a plan; put
   its family under one prefix (`src/foo.rs`, `src/foo_*.rs`).
2. Register a `PassDescriptor` in `src/pass_registry.rs::PASSES` —
   `source_files` (every file), `cli_flags`, `stage`, `phases` (non-empty for
   an in-pipeline pass; empty means out-of-band and `schedule()` will refuse
   it), `decorator_triggers`, `wiki`, `tape: TapeAccess` (declare positional
   references honestly — `crates/nsl-codegen/tests/tape_access_drift.rs` checks the code).
3. If the pass hands a value to another pass, add a `Channel` variant, a
   private `PassBus` field with `publish_*`/accessor methods, and a
   `ChannelDescriptor` in `CHANNELS` (`src/pass_bus.rs`) naming producer,
   consumers, `consumed_by_passes` (with an `OrderClaim`), and the
   invariants. `crates/nsl-codegen/tests/pass_bus_drift.rs` enforces all of it.
4. Invoke it only through `PassScheduler::schedule("NAME", tape, body)`
   inside the phase its descriptor declares, record the disposition
   (`pass_trace::record_disposition`) at the callee, and settle with
   `finish(&bus)`. If the plan is consumed positionally later, call
   `assert_tape_unchanged_since` at that fork (`rescan_tape` if the pass
   mutated the list it scanned).
5. Add a section under "Per-pass descriptions" in
   `docs/wiki/Optimization-Passes.md` with the registered `full_name` as the
   heading (or `WikiCoverage::Undocumented(reason)`), then run
   `pass_registry_drift`, `pass_bus_drift`, `pass_scheduler_coverage`,
   `pass_scheduler`, and `NSL_PASS_TRACE=1 nsl run …` on a fixture to see the
   pass attributed to its phase.
6. If the pass is research-grade, gate its entry behind a Cargo feature the
   way `experimental-wrga` does in `src/stmt.rs`, list it under
   `experimental` in `src/lib.rs`, and add it to STATUS.md's Experimental
   tier.

### A new refusal

1. Raise it as `Err(CodegenError::new(msg).with_note(remedy))` at the
   narrowest site that can decide it — admission in `src/stmt_admission.rs`
   for train-block compositions, the dispatcher-covered lowering site for a
   construct (the span is attached on the way out), or the end of
   `compile_main` for "this flag did nothing anywhere" (see
   `wgrad_hook_blocks`). Never `eprintln!` + `process::exit`, never a silent
   fallback.
2. Say what was refused, why, and what to do; keep a distinctive fragment
   stable, because it becomes the gate's needle.
3. Register the composition in `crates/nsl-cli/src/feature_rules.rs`
   (`Enforcement::Source { file, fragment }`; also as a clap
   `conflicts_with`/`requires` in both `BuildArgs` and `RunArgs` when it can
   be decided at parse time) and add or extend the tier in
   `crates/nsl-cli/tests/feature_composition_gate.rs`.
4. Pin it with a `*_refusal.rs` test here (`crates/nsl-codegen/tests/fp8_source_ad_refusal.rs`
   is the template) and, if it renders with a span, extend
   `crates/nsl-cli/tests/codegen_error_rendering.rs`.
5. If the refusal replaces a former silent path, record the decline as a
   `PassDisposition::Declined` where a pass owns the decision so
   `NSL_PASS_TRACE` shows it.
