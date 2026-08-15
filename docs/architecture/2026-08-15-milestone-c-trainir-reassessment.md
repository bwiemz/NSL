# Milestone C — PassManager owns training optimization scheduling, and the TrainIR verdict

Date: 2026-08-15. Branch `feat/milestone-c-pass-scheduler`, off `origin/main`
@ `2a8a5931` (#503). Predecessors: item 2 steps 1–7 (#453, #456, #458, #462,
#466, #499).

## What shipped, per exit criterion

**All nine in-pipeline passes are scheduled.** WRGA (already, #499), and now
WGGO (kernel prepass + in-place replan), CSHA, CCR, CPDT (three offer
sites), CPKD, FASE, PCA (segment-masked decision + per-doc admission),
MemoryPlanner (whole-program plan + transient arena) — 14 `schedule("…")`
sites, every one settling postconditions with `finish(&bus)` except one
explicit, reasoned `defer_postconditions` (WGGO's prepass, where
`wggo_overrides` is structurally not yet publishable). CEP and CFIE stay
out by design: their `phases` declarations are empty (out-of-band drivers)
and `schedule()` refuses empty-phase passes structurally.
`pass_scheduler_coverage.rs` pins the site table in both directions.

**1. No production scheduled pass has phase=None.** The fix is one
callee-side scope: `compile_train_block` installs `TrainBlock` at its own
entry, so every arrival path — `compile_user_functions` (train nested in
any fn, including `@test` fns), lambdas, model/agent methods, and module
compiles, where fn-nested blocks are the ONLY train path — is covered by
construction. Witnessed end to end by `pass_scheduler_gate.rs` on a
nested-train fixture: the `unscoped(phase check skipped)` token is gone and
FASE attributes `phase=TrainBlock` with the check live. `None` remains
admitted for direct-drive unit tests and out-of-band CLI drivers — the #499
lesson (refusing unscoped invocations broke every direct-drive test) held.

**2. Ordering is derived from declarations.** The enforced order is the
VALUE order declared on the bus (`consumed_by_passes` + `OrderClaim`),
checked per compile epoch at the train-block wrapper's single exit; the
scheduler traces each pass's declared `InvocationOrdered` predecessors at
invocation. A new gate proves the declared edge set forms a DAG. What was
deliberately NOT added: stage-rank enforcement. `PipelineStage` is not a
scheduling key (step-1 finding), and the tree contains a standing
counterexample — CPDT plans in the wrapper BEFORE FASE plans in the inner
function (the moment consult needs `cpdt_plan` before the moments
allocate), inverting their stage ranks on every pre-plan compile, by
design. Any rank-based refusal refuses correct builds.

**3. applied⇒published checked.** `finish()` now judges the boundary for
every pass with an `Enforced` channel: `wrga_plan`, `csha_bridge`,
`cpdt_plan`, `wggo_overrides`. The conversions forced the windows to be
honest: CSHA's and WGGO's scheduled bodies WIDEN through their driver-side
publishes (finish directly after the planner call would refuse every
applying compile — Applied is recorded inside the pass, the publish is
driver code up to 221 lines later). CCR has no channel; its real
postcondition gap — `Applied` recorded at plan construction, plan dropped
when the owned-restriction empties it — is now corrected at the drop site
with a `Declined` re-record (last-wins, the FASE precedent).

**4. Tape reference lifetimes checked.** Five
`assert_tape_unchanged_since` sites, each at the ENTRY of a positional
consumption fork: WRGA's `effective_primal` fork (pre-existing), WGGO's
prune fork, CCR's `apply_to_adjoint` fork and the CSLA `seg_bounds` fork
(~1,200 lines from planning), and the arena's lowering fork. One API bend
was forced: a `MutatesInPlace` pass whose body mutates the list it scanned
(CCR's compressed-save append) supersedes its own entry digest —
`Scheduled::rescan_tape` re-digests the state the pass LEFT, which is what
the forks must see intact. CSHA is the one declared-positional pass with
NO assert, exempted with its reason in the coverage gate: it converts
positions to OpIds at the scan boundary inside its own window, and an
assert would refuse every prune+CSHA composition for a mutation that
invalidates nothing it retained.

**5. stmt.rs stops being the actual pass manager.** Precisely: the driver
keeps the CALL ORDER (the bodies are interleaved with lowering-local data
flow), and the AUTHORITY moved. Whether a pass may run where it is
invoked, whether its product reached its consumers, whether its plan's
tape is still the tape it scanned, and whether the compile's invocation
order honored the declared edges are all judged by the manager, from the
registry and bus declarations. CPKD — the one registered pass with no
module entry, its plan built and recorded inline in stmt.rs — gained
`cpkd::build_plan` (the tape scan stays in the driver so the
`TapeAccess::None` declaration stays true of the cpkd* family). The
coverage gate pins that no invocation site exists outside a schedule call.

**6. CURRENT_EPOCH can begin disappearing.** Its ambient-reader surface is
now exactly three: `record` and `record_disposition` (callee-side
attribution) and the manager's own LIFO drop-check; every enforcement
decision already reads the epoch BY VALUE through `PassScheduler`. The
coverage gate seals the surface — a new caller of
`begin_epoch`/`restore_epoch`/`current_epoch` outside `pass_manager.rs`
fails CI. What full removal requires is threading an explicit context
through every pass entry point (11 entries, plus every exit that records a
disposition) — which is not an epoch question at all. It is the TrainIR
question.

## The TrainIR verdict: not justified now

The suspicion that prompted this reassessment holds. Reasoning:

**What a separate TrainIR would buy, concretely.**
1. *By-construction ordering* — the manager invoking pass bodies from the
   declarations instead of checking the driver's order after the fact.
2. *An explicit context* — killing the `CURRENT_EPOCH`/`CURRENT_PHASE`
   thread-locals and the callee-side ambient recording.
3. *Reified inter-pass state* — the lowering-local values pass regions
   exchange today (`extractor`, `effective_primal`, `adjoint`,
   `mode_table_base`, `wggo_preplan_offered`…) becoming named IR instead of
   ~10k lines of shared locals in `compile_train_block_inner`.

**Why each is worth less than it looks after Milestone C.**
1. Ordering: the enforceable invariant is VALUE order, and it is already
   enforced per compile from declarations, at a single exit, with refusals
   that fire (gate-pinned). By-construction ordering adds soundness only
   for defects the current check would anyway catch one step later — and
   the two attempts at stronger schedule-time refusals both had to be
   reverted as unsound (#499's predecessor pre-check; stage ranks above).
2. The context: real, but it purchases the deletion of two thread-locals
   whose caller surface is now sealed by a gate at three readers. The
   ambient design is load-bearing for exactly the property the step-2
   lesson names: recording at the callee covers drivers nobody wrapped.
   A threaded context re-opens that hole unless every driver is perfect.
3. Reification is the true cost center. The pass regions are not black
   boxes in sequence — they are interleaved with the lowering's own
   decisions (CPDT's wrapper offer exists BECAUSE the moment consult sits
   2.2k lines before the in-place site; FASE's mode is rewritten by the
   driver after planning; CCR's plan is arbitrated against ownership the
   lowering computes). A TrainIR that reifies this faithfully reproduces
   today's data flow under new names — a large, parity-risky restructure
   of the compiler's most defect-dense file — and one that simplifies it
   instead is a redesign of the passes themselves, not an IR refactor.

**What would reopen the question.** (a) A tenth in-pipeline pass whose
product needs a consumption-window discipline the scheduler cannot express
— the current API bends (`rescan_tape`, `defer_postconditions`) were
absorbed cleanly, but a third bend of that kind is a signal. (b) A hard
requirement to delete the thread-locals (e.g. a multi-threaded compile,
which today is structurally excluded — `PassManager` is `!Send` by
design). (c) Evidence that the post-hoc order enforcement missed a real
defect a by-construction schedule would have prevented — none observed
across the campaign's sweeps.

Until one of those arrives, the declarations + manager + gates carry the
milestone's guarantees at a fraction of TrainIR's cost, and the next
marginal effort is better spent where the audits keep finding real
defects: feature composition.
