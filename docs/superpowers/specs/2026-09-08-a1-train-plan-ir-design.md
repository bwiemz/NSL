# Roadmap A1 — `TrainPlan` IR and the `TrainPass` pipeline

**Roadmap criterion:** *Introduce a TrainPlan mid-level IR and a pass pipeline:
`lower_train_block(ast) -> TrainPlan`; each technique a `TrainPass`; ordering
declared once in a `PASS_ORDER` table; `emit_train_plan(plan, builder)` does the
Cranelift emission with no technique branches.* (A1, steps 1–3; step 4 — the
mechanical peels — is what shipped first.)

This is a design spec, not a plan: it says what the data structure and the
trait are, how today's code maps onto them, and the order of mechanical steps
that get there under the same proof the peels used. Nothing here changes the
emitted CLIF; every step below is gated on the 28 train-block CLIF snapshots
(`crates/nsl-codegen/tests/train_clif_snapshots.rs`) staying byte-identical.

## Where the driver stands

`compile_train_block_inner` (`crates/nsl-codegen/src/stmt.rs`) was 11,564
lines at the audit. After the peels it is ~2.3k lines, and 24 phases live as
their own modules under `crates/nsl-codegen/src/stmt_train/`, each behind an
`<Phase>Inputs<'a>` struct that names the driver bindings the phase reads
(see `stmt_train/mod.rs` for the list in driver order). What is left in the
driver is orchestration: the epoch and batch loops, the bindings that flow
between the phases, and the branches that decide which phase runs.

The peels were the survey this design needed. Naming every phase's inputs in a
struct made the driver's implicit state explicit, and it sorts into five
kinds:

| Kind | Examples (driver names) | Where it belongs |
|---|---|---|
| **Configuration** — resolved once from the `train(...)` header and the CLI | `lr_value`, `grad_accumulation_steps`, `grad_clip`, `optimizer_name`, `weight_decay_value`, `scheduler`, `checkpoint_every`, `checkpoint_save_path`, `no_decay_scope` | `TrainPlan::spec` |
| **Parameter facts** — compile-time knowledge of the model | `param_paths`, `model_type_name`, `layout`, the Muon route / decay-exempt / dtype-code lists' *sources*, `num_state_buffers` | `TrainPlan::params` |
| **Tapes** — the Wengert lists and their generators | `extractor` (primal list), `effective_primal`, `adjoint`, `generator`, `loss_var_id`, `param_adj_set`, `adj_vid_to_hook_entry` | `TrainPlan::tapes` |
| **Technique plans** — what each pass decided | `fase_plan`, `wrga_plan`, `wggo_applied`, `ccr_plan` + `ccr_compress_map` + `ccr_fresh`, `csla_pre`, `ws_fwd_plan`, `cpdt_precision_dtypes`' source, `elem_hints`, the arena slots | `TrainPlan::techniques` |
| **Emission handles** — Cranelift values and variables | `param_list`, `grads_list`, `num_params_val`, `state_list_1/2`, `accum_list`, `muon_route_list`, `decay_exempt_list`, `mode_table_base`, `step_count_var`, `lr_var`, `should_step_var`, `epoch_counter_var`, `step_param_var`, `has_dataloader`, `csla_buffers`, `csla_pending` | **not** in the plan: `EmitState` |

The flags the driver branches on (`csla_active`, `fase_deferred`,
`fase_hook_active`) are derived facts — each is a function of the plan
(`techniques.csla.is_some()`, `techniques.fase.mode`, and so on) and becomes a
method rather than a binding.

## The IR: `TrainPlan`

A plain data structure. It holds no `cranelift` type, no `FunctionBuilder`
borrow and no `Compiler` reference, so it can be constructed, inspected,
serialized for a fingerprint, and unit-tested without lowering anything.

```rust
pub struct TrainPlan<'ast> {
    pub spec: TrainSpec<'ast>,          // config + contract, resolved once
    pub params: ParamPlan,              // the model's parameters, as facts
    pub tapes: Tapes,                   // primal / effective primal / adjoint
    pub techniques: TechniquePlans,     // one Option per technique
    pub schedule: TrainSchedule,        // accumulation, checkpointing, LR
}

pub struct TrainSpec<'ast> {
    // today: stmt_train::config::TrainConfigSection + contract::TrainContract
    pub optimizer: ResolvedOptimizer,   // name + hyper-parameters
    pub scheduler: Option<ResolvedScheduler>,
    pub callbacks: Vec<&'ast Callback>,
    pub data: DataSection<'ast>,        // the `data:` section, DataLoader or not
    pub no_decay_scope: NoDecayScope,
    pub grad_clip: f64,
    pub distill: Option<DistillOverrides>,
}

pub struct ParamPlan {
    // today: stmt_train::model_params::ModelParams + param_lists.rs' inputs
    pub model_type_name: String,
    pub paths: Vec<String>,             // list order == runtime list index
    pub roles: Vec<ParamRole>,          // Muon / AdamW routing, per path
    pub decay_exempt: Vec<bool>,
    pub dtype_codes: Vec<u16>,          // CPDT moment precision, per path
    pub num_state_buffers: usize,
}

pub struct Tapes {
    pub primal: WengertList,            // the extractor's list, post-prune
    pub effective_primal: WengertList,  // the WRGA fork (== primal without a plan)
    pub adjoint: Option<WengertList>,   // None until AdjointGen has run
    pub loss_var: VarId,
    pub param_adjoints: HashSet<VarId>,
    pub hook_entries: HashMap<VarId, ParamHookEntry>,
}

pub struct TechniquePlans {
    pub fase: FasePlan,                 // always present; `mode` says which arm
    pub wggo: Option<AppliedPlan>,
    pub wrga: Option<WrgaPlan>,
    pub cpdt: Option<CpdtPlan>,
    pub ccr: Option<CcrPlanned>,        // plan + compress map + fresh-id base
    pub csla: Option<CslaPre>,          // Some ⇔ today's `csla_active`
    pub weight_stream: Option<WsForwardPlan>,
    pub arena: Option<ArenaProjection>, // elem hints + slot declarations
    pub early_free: Option<CcrSegmentFree>,
}

pub struct TrainSchedule {
    pub grad_accumulation_steps: i64,
    pub checkpoint: Option<CheckpointSchedule>,   // path + every-N
    pub resume: Option<ResumeState>,              // Item 8 identity
}
```

Two rules keep this honest:

1. **Plans are data, handles are not.** Anything that is a `Value` or a
   `Variable` today stays out of the plan. `EmitState` (below) owns them, and
   the emitters take `(&TrainPlan, &mut EmitState)`.
2. **A technique is present iff its `Option` is `Some`.** The driver's
   `if self.compile_options.<technique>` / `if self.features.<x>` branches
   move into the passes' `applies` predicates; the emitters read
   `plan.techniques.*` and nothing else. That is the "no technique branches
   in emission" criterion made checkable: a grep for `compile_options` in the
   emission modules is the drift gate.

`EmitState` is the per-lowering side of the same split: the runtime handles
(`param_list`, `grads_list`, the state lists, the accumulation buffers, the
Muon route and decay-exempt lists, the mode-table base) and the loop
variables (`step_count`, `lr`, `should_step`, `epoch_counter`, the step
parameter, the DataLoader handle). Today these are the ~15 `Value` /
`Variable` inputs every late phase repeats; tomorrow they are one struct
populated by the setup emitters in order.

## The pipeline: `TrainPass` and `PASS_ORDER`

```rust
pub trait TrainPass {
    /// The registry name (`pass_registry::PASSES`), so the existing
    /// dependency-order, tape-access and phase gates apply unchanged.
    fn name(&self) -> &'static str;
    /// Whether this pass runs for this plan under these options — the
    /// driver's `if` moves here, once.
    fn applies(&self, plan: &TrainPlan, ctx: &PassCtx) -> bool;
    /// Read the plan, decide, write the decision back into `plan.techniques`
    /// (and, for the tape-rewriting passes, into `plan.tapes`).
    fn apply(&self, plan: &mut TrainPlan, ctx: &mut PassCtx) -> Result<(), CodegenError>;
}

pub struct PassCtx<'c> {
    pub options: &'c CompileOptions,
    pub scheduler: PassScheduler,        // pass_manager: schedule / digests
    pub bus: &'c PassBus,                // the channels the passes publish on
    pub interner: &'c Interner,
    pub train: &'c TrainBlock,           // the AST, for the passes that read it
    pub diagnostics: &'c mut Vec<Diagnostic>,
}
```

`PASS_ORDER` is one table, in the order the driver runs today. Every entry is
an existing pass or an existing peeled module; the table is the composition
matrix that CLASP / CADENCE / CADRE / CADENZA assume exists.

| # | Pass | Today | Stage | Reads | Writes |
|---|---|---|---|---|---|
| 1 | `ConfigPass` | `stmt_train/config.rs`, `contract.rs` | pre-tape | AST, options | `spec`, `schedule` |
| 2 | `FasePlanPass` (FASE) | `fase::plan` under `schedule("FASE")` | pre-tape | `spec` | `techniques.fase` |
| 3 | `AdmissionPass` | `stmt_admission.rs` | pre-tape | `spec`, `techniques.fase` | `techniques.csla` (presence), refusals |
| 4 | `ParamPlanPass` | `stmt_train/model_params.rs`, `param_lists.rs` (the facts half) | pre-tape | layout, `spec` | `params` |
| 5 | `ExtractPass` | `WengertExtractor` (`source_ad.rs`) | tape | AST | `tapes.primal`, `tapes.loss_var` |
| 6 | `CpkdPass` (CPKD) | `cpkd_plan` channel | tape | `tapes.primal` | bus |
| 7 | `WggoPlanPass` (WGGO) | `stmt_train/plan_wggo.rs` | tape | `tapes.primal`, pre-plans | `techniques.wggo`, `wggo_overrides` |
| 8 | `CshaPass` (CSHA) | `stmt_train/plan_csha_prune.rs` (first half) | tape (reads) | `tapes.primal`, `wggo_overrides` | bus |
| 9 | `WggoPrunePass` (WGGO) | `stmt_train/plan_csha_prune.rs` (second half) | tape (mutates) | `techniques.wggo` | `tapes.primal` |
| 10 | `WrgaPass` (WRGA) | `stmt_train/plan_wrga_cpdt.rs` (first half) | tape | `tapes.primal`, `params` | `techniques.wrga`, `wrga_plan` |
| 11 | `CpdtPass` (CPDT) | `stmt_train/plan_wrga_cpdt.rs` (second half) | tape | `techniques.wggo`, `params` | `techniques.cpdt`, `params.dtype_codes` |
| 12 | `CcrPlanPass` (CCR) | `stmt_train/plan_ccr.rs` | tape | `tapes.primal`, `techniques.wrga` | `tapes.effective_primal`, `techniques.ccr` |
| 13 | `AdjointGenPass` | `AdjointGenerator::generate` | adjoint | `tapes.effective_primal` | `tapes.adjoint`, `param_adjoints` |
| 14 | `AdjointTapeOptPass` | `stmt_train/adjoint_tape_opt.rs` | adjoint (mutates) | `techniques.wrga`, `tapes` | `tapes.adjoint` |
| 15 | `CcrAdjointFreesPass` (CCR) | `stmt_train/ccr_adjoint_frees.rs` | adjoint (mutates) | `techniques.ccr`, `tapes` | `tapes.adjoint` |
| 16 | `ArenaProjectionPass` (MemoryPlanner) | `stmt_train/transient_arena_projection.rs` | adjoint | `tapes` | `techniques.arena` |
| 17 | `CslaPrecomputePass` | `stmt_train/csla_precompute.rs` | adjoint | `tapes`, `techniques.ccr`, `techniques.arena` | `techniques.csla`, `techniques.weight_stream` |

Passes 6–12 and 14–17 are exactly the sites that already run under
`PassScheduler::schedule` or are declared in `pass_registry::PASSES`, so the
existing `dependency_order_violations` check, the tape digests, the
`pass_bus` consumer inventory and the `pass_bus_drift` gate keep judging them.
The table adds nothing they do not already enforce; it makes the order a
value instead of the reading order of a 2.3k-line function.

Two places do not fit the table and are called out on purpose:

- **The Muon mode table.** `plan_wggo` takes `mode_table_base: Option<Value>`
  today — an emission handle read during planning. The planner needs the
  *fact* (which parameters route to Muon), which `params.roles` already
  carries; the handle is only needed by the optimizer-step emitter. Step 2
  below breaks that dependency before the pass is wrapped.
- **The pipelined path.** `compile_train_block_pipelined_inner`
  (`stmt_train/pipelined.rs`) shares none of the scheduled passes and stays a
  separate driver; `TrainPlan` is for the scheduled path only.

## Emission: `emit_train_plan`

```rust
pub fn emit_train_plan(
    c: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    plan: &TrainPlan<'_>,
) -> Result<(), CodegenError>
```

runs the existing emitters in today's order, each taking `(&TrainPlan, &mut
EmitState)` instead of a hand-listed `Inputs` struct:

| Emitter | Today | Populates / reads `EmitState` |
|---|---|---|
| setup: parameter lists | `stmt_train/model_params.rs` (emission half), `param_lists.rs` | writes `param_list`, `num_params`, route / exempt / accum lists |
| setup: optimizer state | `stmt_train/optimizer_state.rs` | writes `state_list_1/2`, moment latch |
| setup: identity + resume | `stmt_train/identity.rs` | writes `has_dataloader`, `checkpoint_dl_handle` |
| epoch / batch loop headers | driver | writes the loop variables |
| primal `VarMap` | `stmt_train/primal_vars.rs` | reads `param_list`, `step_param` |
| adapter sites | `stmt_train/adapter_sites.rs` | reads `techniques.wrga` |
| forward | `stmt_train/forward_lowering.rs` | reads `techniques.{ccr,weight_stream,early_free}` |
| adjoint | `wengert_lower`, `stmt_train/fase_hook_lowering.rs`, `csla_window.rs` (save) | reads `techniques.{fase,csla}` |
| gradients | `stmt_train/source_ad_grads.rs` | writes `grads_list` |
| per-step diagnostics | `stmt_train/health_hooks.rs` | reads `grads_list`, `step_count` |
| clip + accumulate | driver | reads `schedule`, `techniques.fase` |
| optimizer step | `stmt_train/optimizer_step.rs`, `stmt_csla.rs`, `stmt_fase.rs`, `csla_window.rs` (backward) | reads everything above |
| scheduler + checkpoint | `stmt_train/scheduler_step.rs` | reads `spec.scheduler`, `schedule.checkpoint` |
| callbacks | driver | reads `spec.callbacks` |
| epoch close, teardown | `stmt_train/epoch_close.rs`, `teardown.rs` | frees what setup allocated |

The invariant is the one stated above: an emitter branches on `plan` fields
only. `if self.compile_options.diagnostics.debug_training` in the health
hooks becomes `if plan.spec.diagnostics.debug_training`; `if csla_active`
becomes `if let Some(csla) = &plan.techniques.csla`.

## Mechanical steps, each a mergeable PR under the snapshots

1. **`TrainPlan` as a carrier, no behavior change.** Add the struct, build it
   in the driver from the bindings that already exist (the `TrainConfigSection`,
   `TrainContract`, `ModelParams`, `PreForwardPlans`, `WggoPlanning`,
   `CslaPre` … values the peels already return), and hand `&plan` to each
   `Inputs` struct in place of the fields it now copies out of the plan. The
   `Inputs` structs shrink to their emission handles. Snapshots unchanged;
   this is the same proof as every peel.
2. **Break the two planning-time handle reads.** `plan_wggo`'s
   `mode_table_base` becomes a `params.roles` read (the emitter keeps the
   handle); `plan_ccr`'s `sched` moves into `PassCtx`. After this, no
   planning module names a `Value`, which the `grep` gate enforces.
3. **`EmitState`.** Collect the runtime handles and loop variables the setup
   emitters produce into one struct; the late emitters take it by `&mut`.
   The 30-field `OptimizerStepInputs` / `CslaWindowInputs` become
   `(&TrainPlan, &mut EmitState)`.
4. **`TrainPass` + `PASS_ORDER`.** Wrap each planning module in the trait;
   replace the driver's planning stretch with `for pass in PASS_ORDER { if
   pass.applies(..) { pass.apply(..)? } }`. Add a drift gate that the table's
   order agrees with `pass_registry` / `PassManager::dependency_order_violations`,
   the way `pass_scheduler_coverage.rs` pins the scheduled set today.
5. **Split the driver.** `lower_train_block(&self, train) -> TrainPlan` (steps
   1–17, no `builder`) and `emit_train_plan(plan, builder, state)`; the loop
   emission stays in the emitter. `compile_train_block_inner` becomes the
   two calls plus the `@inspect` / CSHA-cache epilogue.

Each step keeps `scripts/gated-tests.sh`, the ten drift gates, the composition
gate and `clippy -D warnings` green; steps 1 and 3 are the ones the CLIF
snapshots pin hardest, since they touch every emitter's argument list.

## What this unblocks (roadmap §5)

- **CLASP** — the budget-solver certificate needs the resolved plan as a
  hashable value: `TrainPlan` is serializable by construction, and
  `exec_fingerprint` can hash it whole instead of the per-option fields.
- **CADENCE MUSE** — a fused LOMO backward is a `TrainPass` that rewrites
  `tapes.adjoint` and sets a technique plan the adjoint emitter reads, not a
  branch inside the driver.
- **CADENZA** — growth operators transform a `TrainPlan` between stages; the
  per-stage fingerprint is the plan's hash.

## Non-goals

- A general HIR for the CPU path (A6) stays deferred; `TrainPlan` is the
  training loop's IR only.
- The pipelined train block keeps its own driver.
- No pass changes its decisions: this is the same compiler with its decisions
  written down before they are emitted.
