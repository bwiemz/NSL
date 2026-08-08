# Changelog

All notable changes to NeuralScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed — item 17 phase 3a: packing metadata becomes dataflow on the fused-AD path

- **The `@flash_attention` reads inside a train block no longer consult the
  `PACKING_METADATA` thread-local.** The stash `def_var`s the
  `(segment_ids, doc_starts)` device pointers into function-local Cranelift
  variables per micro-batch, and the wengert-lowered forward claim and csha
  backward read those — so the CSLA window-backward's re-stash now feeds
  per-iteration dataflow instead of compensating for a global that survives
  loop iterations, and a read can no longer see a stale batch's (freed)
  pointers on that path. The variables are cleared on every train-block
  exit path (they belong to that function; a later block or grad block gets
  the getter fallback). The thread-local, its setter, and both getters stay
  for the model-METHOD readers (`expr/advanced.rs`) — a different Cranelift
  function that needs phase 3b's ABI decision (hidden param vs opaque
  handle) documented in `docs/architecture/compiler-state.md`.

- **An ~8.5% measured step-time win no longer depends on hand-editing model
  source.** The CFTP fused linear-CE kernel required a `@fused_lm_ce`
  decorator carrying four numbers the compiler already knows; no shipped
  pretraining script had one. `--fuse-lm-head {off,auto,require}` infers the
  chain and the numbers: `vocab/hidden` from the head weight's derivable dims,
  `batch/seq` from a **unanimous-or-nothing DataLoader scan** that refuses
  `drop_last=false`, non-literal shapes, or disagreeing loaders — the row
  count is baked into PTX, so a short batch would read out of bounds and the
  scan treats "cannot prove" as "do not fuse". `require` turns the decline
  into a compile error; `--pretrain-optimized` selects `auto`;
  `--training-reference` forces `off`; an explicit decorator (including
  `enabled=false`) always wins.
- **The house `logits.shape` idiom is folded to the proven `[B, S, V]`** so
  the shape read stops keeping the logits alive — without this the flag fires
  on zero shipped programs. Validated: inferred vs explicit decorator
  bit-identical over 195 deterministic steps; composite-vs-fused divergence
  3.7e-2 max-rel (the kernel's known numerics), fused arm bit-reproducible.
- Fixed en route: the multi-file build dropped `model_field_dims`/`ranks` on
  the floor for every imported model, which had silently disarmed the CFTP
  v10 rank guard for every real model since it landed.

### Added — `--transient-arena` Stage 2B/2C: placed backward temporaries

- **Compile-time shape propagation that can actually cross a matmul.** Dims
  flow through the `[forward ; adjoint]` tape from semantic types,
  model-field initializers, constructor-argument folding (`ctor_fold.rs`:
  `randn([d_model, d_ff])` under `TransformerBlock(512, …)` — previously
  structurally underivable, which was ALL block weights), and the
  `--fuse-lm-head` loader proof for batch fields. A constant lattice folds
  `x.shape` reads, subscripts, integer arithmetic, and
  `int(self._n_heads.item())` config reads, so the attention stack's
  runtime-value reshapes resolve statically. coder50m: 1456 of 2159 tape
  values sized, where the numel-only Stage-2A bridge sized 1.
- **Admission models the runtime it actually runs against.** Beyond the
  Stage-1 rules (backward-only, sized, unsaved, non-escaping, non-aliasing,
  single-allocation elementwise), two rules exist because the bind/placement
  reconciliation caught the plan lying: an op whose input dies at it is
  refused (the FBIP arm mutates a dying uniquely-owned input IN PLACE — every
  admitted Sqrt/Neg leaked its bind through exactly that), and a binary
  without provably-equal shapes or with a view operand is refused (broadcast
  and non-contiguous routes allocate off the straight-line path).
- **Placement is sequential disjoint regions, deliberately not BFD
  time-sharing.** The shared-offset plan aliased payload ranges under the
  runtime layout (`payload = base + offset + REDZONE·(idx+1)`) — found by
  planned-vs-unplanned byte-bisection (`NSL_ARENA_SLOT_LIMIT`), losses
  diverging at step 1 from slot 8 on. Stable addresses are the point; the
  sequential cost on coder50m is 308 MiB.
- **The pin is size-exact, single-shot, and audited.** Armed before each
  admitted op, disarmed after; consumed only by an allocation of exactly the
  planned bytes — and the consult now sits in BOTH allocator entries
  (`try_alloc_managed` is the one elementwise outputs use; with the consult
  only in `alloc_managed`, all 7240 binds leaked). Teardown reconciles binds
  vs placements; `NSL_ARENA_CHECK=1` sweeps every 0xA5 red zone per step.
- **Validation (coder50m, 40 steps, `--deterministic --seed 4242`):** planned
  vs unplanned loss streams and 199 MB `.nslm` checkpoints byte-identical;
  7240/7240 binds placed; 0 guard failures; canary run clean. CUDA-graph
  offset feeding is deliberately NOT wired — that lands separately now that
  the byte-identity gate is green.

### Added — SR-BF16 certification instrumentation (item 7)

- `NSL_SR_HIST=1`: every fused SR step samples each mirrored parameter's
  bf16 mirror before/after the update and reports a `|Δθ|`
  binary-exponent histogram plus the **stalled rate** (bf16 bits unchanged —
  the empirical underflow measure under stochastic rounding) at teardown.
- `models/coder{500m,1b}/pretrain_srbf16_{cert,continue}.nsl` and a `1b`
  campaign scale; `srbf16_campaign.py` gains `--corpus` (slice a real token
  stream instead of the synthetic tiled block), `--tag` (parallel log
  namespaces), `--sr-hist`; `tools/build_byte_corpus.py` builds a 23M-token
  real corpus from the repo's own sources (byte-level, labeled as such).

### Fixed — the item-6 matrix silently dropped its layerwise-family arms

- The generated program hard-coded `grad_clip=1.0` (refused by
  `--layerwise-accum`) and never emitted `grad_accumulation>=2` (required by
  FASE-Deferred), so the 500M `srbf16` and both 1B `layerwise` arms failed to
  BUILD and vanished from the results without a trace. Arms now carry their
  own `accum`/`grad_clip`; tokens-per-step scales with accumulation;
  loss-divergence only compares config-matching arms. Also fixed: the
  admissions/dispositions capture scanned the RUNTIME stderr for
  COMPILE-time markers (every optimized-arm admission read false), and
  `REGION_RE` lacked `re.M` so region attribution parsed nothing.

- **The trace could say a pass was REACHED; it could not say whether anything
  happened.** "CSHA did not run" and "CSHA ran and found no attention chain it
  could fuse" were the same line, and only one of them is a bug. A flag that
  enables a pass which then declines produces a clean, plausible, wrong build
  exactly like a flag that enables a pass which never runs.

  `PassDisposition` — `Applied { rewrites }` / `Declined { reason }` /
  `AdvisoryOnly` — with a `DeclineReason` of five structural categories
  (`ModeOff`, `NoCandidates`, `PreconditionViolated`, `FeatureDisabled`,
  `BudgetInfeasible`) carrying `&'static str` detail. All eleven registered
  passes report at every exit, 36 sites. Under `NSL_PASS_TRACE=1`:

  ```text
  [pass-trace] 3 pass(es) ran: WGGO(OnWengert)@KernelPrepass -> FASE(PreExtraction)@TrainBlock -> CSHA(OnWengert)@TrainBlock
  [pass-trace] WGGO: applied, 3 rewrite(s)
  [pass-trace] FASE: applied, 2 rewrite(s)
  [pass-trace] CSHA: declined, no candidates - no attention boundary chain admitted a fused kernel
  ```

- **Disposition is recorded through a SECOND function that asserts a prior
  entry record.** `record` stays exactly as it was; `record_disposition`
  panics unless the pass is already in the trace. That is what makes
  disposition structurally presuppose invocation, so it can never quietly
  become a replacement for the entry instrumentation — the property the
  original module was built to protect.

- **`Applied { rewrites: 0 }` is not how "nothing happened" is spelled.** Zero
  is exactly what a hard-coded stub produces, so a pass that changed nothing
  reports a `Declined` with a reason instead. There is deliberately **no
  `Failed` variant**: no registered pass returns an error — the driver raises
  `CodegenError` — so it would have zero producers, which is the
  decorative-metadata defect `pass_registry.rs` deleted its `status` field to
  avoid.

- **FASE reports from the driver, not from inside the pass.** `stmt.rs`
  rewrites FASE's mode after `plan` returns (muon, `--layerwise-accum`), so a
  disposition recorded inside would be accurate about the pass and wrong about
  the build.

- `pass_registry_drift`'s static scan matched the literal
  `pass_trace::record("`, which `record_disposition(` does not contain — the
  new sites would have had strictly WEAKER coverage than the ones they extend.
  Extended to both spellings, with a subset assertion (every disposition name
  must be an entry name) and raised floors. `wgrad_fusion` stays out of the
  registry: it is invoked from the Cranelift lowerer rather than
  `compile_train_block` and already prints its own banner; its `NOT_A_PASS`
  reason now states that boundary instead of merely asserting it.

### Added — GPU kernel paths refuse a non-f32 tensor instead of misreading it

- **~45 wrappers in `cuda/mod.rs` size buffers at `n * 4`, publish dtype 1, and
  cast `t.data as *const f32` — while the dispatchers reaching them branched on
  `device > 0` alone.** Feed a bf16 or fp16 device tensor in and the kernel
  reads twice its allocation: not a fault, a plausible wrong answer, published
  as f32 and believed. Device-resident non-f32 tensors are not hypothetical —
  `nsl_tensor_to_device` carries any dtype outside {f64, f32} to the GPU
  verbatim, `nsl_tensor_zeros_like_dtype` allocates 2-byte device buffers,
  `nsl_tensor_zeros_f16_on` is emitted six times per CSHA fused backward, and
  `--checkpoint-compress bf16` publishes bf16 halves. Each is kept away from
  these kernels today only by a widen the compiler happens to emit.

- `assert_gpu_f32` at 68 call sites across 47 functions in `cuda/mod.rs`,
  plus 3 more in its callers. The two highest-reach
  are in `shape_ops`: `nsl_tensor_contiguous` and `nsl_tensor_slice` have
  dtype-correct CPU arms that were simply unreachable for a device tensor.
  Falling through to them is not an option — that arm allocates HOST memory, so
  a per-element copy would read a device address from the host.

- **A static drift gate keeps it from rotting.** It reads `cuda/mod.rs` as
  text, finds every function that sizes a buffer at 4 bytes per element or
  launches an `*_f32` kernel, and requires either a guard or an allowlist entry
  with a stated reason — checked in both directions, so a stale exemption fails
  too. It runs under `cargo test --lib` with no GPU, which means a new
  unguarded path reddens the ordinary CPU lane on the commit that introduces
  it rather than a day later in the nightly.

- `assert!`, never `debug_assert!` — CI ships release and there is no
  `[profile.release]` override. One consequence worth recording: an assert
  reached through a `pub extern "C"` frame **aborts** rather than panicking
  (`panic_cannot_unwind`, Rust >= 1.81), so `#[should_panic]` cannot observe
  any of these. The new `gpu_dtype_refusal` gate re-execs itself and checks the
  child's exit status and stderr, which also pins the operand name and the
  offending dtype tag rather than substring-matching a payload.

### Fixed — three GPU-cert lane defects that only running the lane could find

- **`tf32_is_faster_and_less_accurate_than_full_f32` took ONE timing sample per
  arm.** Run ~340 targets deep on a GPU saturated for the better part of an
  hour, both arms dilated ~5x (2.79 / 2.48 ms against the 0.583 / 0.365 ms
  reference) and the ratio fell under the 1.35x floor — while the error ratio
  stayed at 402x, which is proof TF32 *was* reaching cuBLAS. The same binary
  passed in isolation. Now best-of-3 INTERLEAVED: a stall can only make a
  sample slower, so the fastest sample measures the throughput ceiling the
  assertion is actually about, and alternating keeps a drifting clock off one
  arm. It does not weaken the gate — if the mode never reached cuBLAS the two
  arms would be the same code and no round would be fast. Verified green idle
  and under deliberate contention. The failure message now says to read the
  error ratio first, since that is what separates "timing" from "dispatch".

- **The self-hosted runner was never registered, and the script said it was.**
  `setup-gpu-runner.sh` aborted at `config.sh` under `set -e` and then printed
  "registered and running" unconditionally — which is why the box sat
  unregistered while the script reported success. It now verifies against the
  GitHub API before claiming anything, treats a linger failure as fatal
  (without it the 09:00 UTC schedule silently never fires), passes `--replace`
  and `curl --fail`, and stops truncating a hand-edited systemd unit on every
  run. The unit gained an explicit `PATH`: Actions steps are non-login shells
  that never source `~/.cargo/env`, and a missing `ptxas` makes `gpu-cert.sh`
  refuse outright rather than merely run slower.

- **A failing gate left no durable diagnostic.** Every log lived in a `mktemp`
  dir the EXIT trap deleted, so a red gate left three grep'd lines in an
  Actions log that expires long before the 90-day artifact — and a target whose
  `--ignored --list` failed to build produced silent mass-`NOTFOUND` with no
  error text anywhere. The lane now banks per-gate logs, the batch log, the
  listing stderr and a `run-metadata.tsv` recording tier, features and the
  timeout actually in force; the workflow copies the manifest, known-red list
  and long-arms table beside them, and tees the preflight.

- `ci/gpu-cert-long-arms.txt` raises the per-gate budget to 1800s for the three
  full-training targets, as `max(NSL_CERT_TIMEOUT, entry)` so a deliberately
  raised global is never shortened back. It is a separate file rather than a
  fifth manifest column because `--write-manifest` regenerates the manifest
  verbatim from a four-field scanner and `--check-inventory` diffs it
  byte-exactly — a fifth column would go permanently red and then be erased.
  The new GPU-free `--check-long-arms` validates the format and cross-checks
  every path against the manifest, so a test-file rename fails CI instead of
  silently reverting the gate to the default.

### Fixed — module-level fns now win every builtin dispatch arm; stale closure info cleared

- **~50 builtin dispatch arms carried no registry guard**, so a
  module-level user fn sharing the name silently misdispatched to the
  builtin: `fn sum(x: Tensor) -> float: return 42.0` printed the
  BUILTIN's sum (4.0), exit 0; `fn abs` likewise; `fn reduce_max` hit
  the builtin's arity check as a compile error (the gap documented in
  PR #435). One hoisted `registry.functions` check at the top of
  `compile_call` now routes to the registered fn before every arm —
  placed after the local-binding route (#435), the vmap rewrite, kernel
  dispatch, and the @fuse arm (whose fns live in the registry and must
  keep fusion precedence). The 37 per-arm guards remain (redundant but
  each arm stays correct in isolation).
- **The checker had the same by-name claiming bug internally**: its
  special-case typing for `sum`/`mean`/`reduce_max`/`gather`/`clamp`/
  `neg`/math builtins fired regardless of user declarations, so after
  the hoist a shadowed `sum(t)` was typed as the builtin (Tensor) while
  codegen called the user fn (Float) — a verifier error. The whole
  by-name special-case block in `check_call` now yields when the name
  resolves to a USER declaration (real `def_span`; builtins carry DUMMY
  spans). Checker and codegen now agree in both directions.
- The one stdlib name collision is `generate`: unimported, the CFIE
  serve arm still owns the name (pinned); imported, the user asked for
  the stdlib loop and it now actually runs (previously the arm
  intercepted and errored even with the import).
- **`closure_info` staleness (#435 review LOW)**: the map is
  compiler-global and symbol-keyed, so rebinding a captured-lambda name
  to a NON-capturing lambda left the stale capture count — the call
  site then read the bare function pointer as a closure struct (probed:
  silent death mid-program). VarDecl now clears the entry on any
  non-capturing rebind.
- Review fixes on the first cut, all gated: the closure_info CLEAR was
  itself a regression — the map is compiler-global and a nested fn body
  compiles mid-way through the outer body, so a nested non-capturing
  `let f = ...` deleted the OUTER function's live closure entry (the
  outer call then executed the closure struct as code). The FnDef arm
  now snapshots/restores closure_info around the nested compile, which
  also stops nested capture-counts leaking outward (pre-existing).
  Plain-Assign rebinds (`f = <lambda>`, no let) get the same
  transfer/clear as VarDecl (previously stale in both directions,
  pre-existing death). The checker's user-vs-builtin distinguisher is
  now an explicit `is_builtin` flag on SymbolInfo (only
  register_builtins sets it; user redeclaration sheds it) — the interim
  DUMMY-span heuristic mislabeled synthesized glob imports (train-block
  auto-imports carry DUMMY spans). The `cumsum` arm arity-checks
  instead of indexing args[1] (compiler panic → diagnosed refusal).
- **Deliberate behavior change**: shadowing a builtin name with a
  NON-Function value (`let sum = 5`) now makes calls of that name a
  compile error ("not callable") — Python semantics — instead of the
  silent fallback to the builtin. Pinned by a gate.
- Second review pass found the clear was still BLOCK-scope-blind: a
  dead if-arm's non-capturing rebind deleted the outer closure's entry
  (the post-arm call executed the closure struct as code — a
  regression over pre-audit behavior). `closure_info` now lives on
  FuncState (per-function by construction — the nested-fn snapshot is
  retired) with a per-scope undo log rolled back by
  `pop_fn_binding_scope`, mirroring the checker's block scoping the
  same way `live_fn_bindings` does.
- Third review pass: the FuncState move itself broke closure
  COMPOSITION — a lambda capturing a capturing closure and calling it
  died (deferred lambda bodies compile with a fresh FuncState, losing
  the definer's closure metadata). PendingLambda now records captured
  closures' capture counts and compile_lambda_body seeds its state
  from them.
- 12 new CPU gates in `builtin_shadow_dispatch.rs` (23 total).
  Mutation-proven in three directions: hoist off → exactly the 3
  module-fn shadow gates red; checker gating off → exactly the 2
  tensor-arg shadow gates red (verifier disagreement); closure clear
  off → exactly the rebind gate red.


### Fixed — container/field assignments no longer strand or dangle tensor temporaries

- **`d["k"] = t * 2.0` left the stored tensor in the statement-
  temporaries list** (PR #433 review LOW-2): the next statement's sweep
  freed it while the container still held the raw handle — dict reads
  cloned a freed tensor (bad-magic abort), list reads handed out the
  dangling pointer itself, struct/model field loads read freed memory,
  and the WRGA adapter side-table's free-on-overwrite became a double
  free. `compile_assign`'s Ident arm has always ended with the
  ownership-transfer + drain pair; the Subscript arms (dict/list set,
  tensor multi-dim set) and every MemberAccess store path had neither.
- All five store paths now share one exit
  (`assign_container_store_tail`): `free_tensor_temporaries` with the
  just-stored value as `keep` — the drain removes the handle from the
  sweep's reach without freeing it, which IS the ownership transfer
  into the container (per the dict_lifetime.rs borrow-store convention
  nothing retains: no machinery ever releases a container-stored
  borrow). Sub-expression temporaries (`d["k"] = t * 2.0 + 1.0` leaves
  the inner product) are freed at the statement — previously they
  straddled into any region that frees the list without draining it,
  and the train-block step loop freed such a straddler once per step:
  double free at step 2, the review's original scenario.
- The review's residual pass found the SAME class in two adjacent
  arms, both fixed here: **dict/list LITERALS stored owned-temp
  elements raw** while the VarDecl statement-end sweep freed them —
  `let d = {"a": t * 2.0}` then reading `d["a"]` SILENTLY returned the
  reused header of the next allocation (printed 2 instead of 8; list
  reads aborted on bad magic). Owned elements now transfer into the
  container (consume-only — no retain for borrows, since nothing ever
  frees container-held values, unlike the tuple literal's convention).
  And **destructuring `let (a, b) = ...` arms had no statement-end
  drain** — a sub-expression temp straddled into the train step loop
  exactly like the assign case (double free at step 2, runtime-
  confirmed); both destructure arms now drain with the destructured
  value kept.
- 10 CPU gates in `assign_temp_drain_gate.rs`: dict, list, and
  struct-field stores of owned temps surviving later sweeps; the
  sub-expression straddler crossing a real 2-epoch train block; a
  borrow-store + source-variable-liveness pin; an int-list pin.
  Mutation-proven in both directions: drain removed → exactly the 4
  defect gates red (delayed-sweep use-after-free); keep dropped →
  exactly the 4 defect gates red (immediate-free direction), pins
  green both times.

### Fixed — indirect calls no longer miscompile float arguments, returns, or captures

- **`(|v: float| v * 2.0)(3.0)` returned 4** — silent wrong values,
  exit 0 (bugs.md 2026-07-29). `compile_lambda` hardcoded every
  parameter to I64 in the lambda's compiled signature while
  `compile_indirect_call` builds the call-site signature from the
  checker's `Type::Function` (F64 for float): the caller passed the
  argument in an XMM register, the callee read an integer register, and
  the value-type-keyed int→float promotion in `compile_binary_op`
  laundered the garbage into a plausible float. Lambda signatures are
  now derived from the checker's Function type — the same source every
  call site lowers from. Int lambdas agreed by accident and are
  unchanged.
- **Float and bool captures had the mirror bug**: the lambda's
  signature declared capture params with the captured variable's real
  cl_type while the closure call site declares and loads every capture
  slot as I64. Capture slots are now raw I64 bit-patterns on BOTH
  sides — normalized at closure construction (floats bitcast, narrow
  ints widened) and restored at lambda entry.
- **`map()`/`filter()` refuse float-typed functions at compile time.**
  The runtime invokes the passed pointer as `extern "C" fn(i64) -> i64`
  (hof.rs) — a float parameter or return cannot survive that ABI and
  produced silent garbage (the lambda wrote XMM0; the runtime read
  RAX). Narrow-int returns (int8/16/32) are refused too (undefined
  upper register bits). Int-typed functions work unchanged.
- **`filter()` with a bool predicate kept garbage-dependent elements**
  (pre-existing): a bool return lowers to I8 — only the low byte of the
  return register is defined — but `nsl_filter` compared the full i64
  against 0, seeing stale upper bits. `nsl_map`/`nsl_filter` now take a
  codegen-supplied `ret_is_bool` flag and mask to the defined byte.
- Two review follow-ups folded in: indirect call sites now COERCE
  checker-legal numeric widenings to the declared param type (`f(3)` on
  `(float) -> float` previously reached the Cranelift verifier as an
  i64-vs-f64 arg mismatch — an ICE with a raw dump), and map()/filter()
  refuse UNANNOTATED lambda params (Unknown-typed bodies lowered through
  tensor ops and aborted at runtime; refusing is strictly an upgrade).
- 16 CPU gates in `lambda_float_abi.rs`: the two bugs.md repros, float
  through a fn-typed parameter, mixed float/int params, float + bool +
  int captures, the nested-fn float route, int map/filter regression
  pins, int→float widening (direct + via fn-typed param), and four
  compile-time refusals. Mutation-proven in six independent directions:
  param typing reverted → exactly the 5 float-param gates red; capture
  convention reverted → exactly the 2 non-I64 capture gates red;
  refusal guard off → exactly the 3 float-refusal gates red;
  ret_is_bool flag off → exactly the filter gate red; arg coercion
  off → exactly the 2 widening gates red; Unknown-param refusal off →
  exactly the unannotated-map gate red.

### Fixed — nested fns, lambdas, and fn-typed params now win builtin-name dispatch

- **A local binding holding a function value lost dispatch to any
  builtin arm sharing its name.** The registry guards from #429 cover
  module-level user fns, but nested `fn`s are removed from
  `registry.functions` right after their body compiles and lambdas
  never enter it — so a nested `fn topk` hit the BUILTIN arm while the
  checker had typed the user fn: a runtime magic-abort or a cranelift
  ICE depending on arity (the bugs.md 2026-07-28 `fn topk` shadow ICE).
  `compile_call` now routes any local binding whose checker type is
  `Type::Function` straight to `compile_indirect_call` before every
  builtin arm; non-Function locals (a tensor named `sum`) keep builtin
  dispatch unchanged.
- The guard's binding lookup is the new scope-aware `live_fn_bindings`
  map, NOT the flat `state.variables`: the flat map never unbinds, but
  the checker is block-scoped — the first cut trusted the flat map and
  the independent review runtime-confirmed four crash shapes where a
  call AFTER the shadowing arm exited (which the checker resolves to
  the builtin or module fn) rerouted into a dead function pointer.
  Every arm/body/match/block-expression lowering now pushes and pops a
  binding scope mirroring the checker.
- The review's residual-gap pass added grad-block bodies (runtime-
  confirmed crash: a post-grad-block builtin call rerouted into the
  grad body's dead nested fn) and both serve endpoint-body sites to the
  scoped lowerings; the pop-direction safety comment was also inverted
  and is now stated correctly (a MISSED pop is the dangerous direction —
  balance is load-bearing).
- Eight new CPU gates: nested fn, let-bound lambda, fn-typed parameter
  (each value-pinned; mutation-proven red with the route disabled) plus
  the reviewer's four post-scope repros — dead-arm nested fn, LIVE-arm
  nested fn, dead-arm lambda, dead-arm module-fn shadow (mutation-proven
  red with the scope pop disabled) — and the post-grad-block shape
  (mutation-proven red with the grad scope removed). Module-level arms
  WITHOUT a registry guard (e.g. `reduce_max`) remain a documented gap
  for the dispatch-arm audit.
- Discovered en route, filed in bugs.md, NOT fixed here: indirect calls
  MISCOMPILE float arguments (`(|v: float| v * 2.0)(3.0)` returns 4,
  silently) — pre-existing on the parent tree, int-typed indirect calls
  are correct; the new gates use int lambdas deliberately and say why.

### Fixed — while-condition temporaries no longer strand per evaluation

- **A while loop's condition stranded its tensor temporaries once per
  evaluation** (all but the final one): condition temps register below
  the loop-scope mark, and the only free site was the While statement's
  end-of-statement cleanup in the exit block — which frees exactly one
  evaluation's values (the final one, whose SSA results dominate the
  exit). `while sum(cumsum(g, -1)).item() * n > 0.0:` leaked 2 blocks
  per iteration. This was the deliberately-pinned exact-6 residual in
  `nested_arg_temporaries_gate` (documented as "needs a header/exit free
  placement", PR #430).
- Fix: `free_condition_temporaries` — the loop HEADER frees each
  evaluation's temps right after the branch scalar is computed and
  before the `brif`, and drains them from `tensor_temporaries` so the
  exit-block cleanup cannot double-free. The header re-executes before
  every body entry AND before the exit branch, so the final evaluation
  is freed exactly once too; entries below the snapshot (an outer
  statement's temps) stay listed and untouched. `while-let` routes
  through the same helper with the bound expression value excluded (it
  is read throughout the body — the exclusion is defensive today since
  top-level expression results are not tracked, and stated as such).
- Gate updated: the exact-6 pin is retired — zero strands at 4 AND 7
  evaluations (two counts so per-evaluation frees can't be faked by a
  constant offset), plus a new while-let shape (int-typed binding whose
  expression carries nested tensor temps; 4 stranded pre-fix) with loop
  value correctness pinned throughout.
### Added — AdamW parameter groups: `no_decay=[...]`

- `AdamW(weight_decay=λ)` in a `train` block applied λ to EVERY trainable
  parameter — RMSNorm gains and the tied embedding included — because the
  optimizer DSL had no parameter groups, so the conventional
  decay-the-matrices-only recipe could not be expressed. `no_decay=[...]` now
  names parameter ROLES to exempt, reusing the vocabulary `@param_role` and the
  mixed Muon/AdamW router already share: `"vector"`, `"embedding"`, `"head"`,
  `"hidden"`. `no_decay=["vector"]` is the usual norms-and-biases convention.
- **`"vector"` is resolved at STEP time from the tensor's real rank, not at
  compile time, and that is the load-bearing decision.** A model field only
  gets a statically-known rank when its initializer is a direct
  `zeros/ones/randn/rand/full/arange` call whose shape list is all integer
  literals (`extract_shape_from_tensor_init`). Real models fail that on both
  counts: `RMSNorm.weight = ones([dim])` passes an identifier, and
  `wq = randn([...]) * sqrt(...)` is a multiply. Measured on `models/coder50m`
  — 74 parameters classify as 1 `embedding` + 73 `hidden`, **zero** `vector`,
  with both RMSNorm gains in that `hidden` bucket. A static-only "exempt
  rank < 2" would have compiled, printed a plausible exemption table, and
  decayed every norm in the model anyway.
- Exemption is expressed as λ = 0, so it reuses each optimizer arm's EXISTING
  no-decay branch rather than adding arithmetic: the stdlib `adamw_step` /
  `muon_step` already guard `if weight_decay > 0.0`, the FASE update program
  already elides its `wd·θ` term when `wd == 0.0`, and the fused kernels
  already take `has_wd` as a launch parameter. A decayed parameter's numerics
  are therefore bit-identical to a run without the feature.
- The rule lives in ONE place, `nsl-runtime/src/optim_groups.rs`
  (`nsl_optim_param_wd`). Codegen calls it per parameter inside the optimizer
  loop and `nsl_fase_fused_adamw_step_multi` calls it while bucketing launches
  — two copies of an eligibility predicate that had to agree exactly is the bug
  pattern that module exists to avoid.
- The flat-grid fused AdamW kernel takes `neg_lr_wd`/`has_wd` as LAUNCH
  parameters, one value per grid, so λ is now part of `UpdateKey` alongside
  device and dtype. λ takes at most two distinct values, so this costs at most
  one extra launch and each parameter still gets bit-identical arithmetic to a
  scalar-λ launch at its own λ.
- Threaded through the FullBuffer per-param loop, both sub-arms of the unified
  WGGO mode-table dispatch, and the two Deferred `fase_emit_final_step` sites.
  The Deferred sites branch between two COMPILE-TIME recipes (the plan's, and a
  λ = 0 clone) rather than handing the emitter a runtime λ: each arm is then
  byte-identical to its pre-feature emission, and the exempt arm contains no
  decay arithmetic at all. Multiplying θ by a runtime zero would have been the
  smaller diff and is deliberately not used — the same file already documents
  that mul-by-0.0 is not a zeroing idiom here because it keeps NaN/Inf alive.
- Refused loudly, not silently ignored, where a single λ is hoisted out of the
  per-parameter loop: `--muon-batch-ns`, `--layerwise-accum`,
  `--optim-state-offload`, and the `@pipeline` train path.
- Everything about this is loud. Every run prints a `[wd-groups]` table naming
  the parameters it exempted; an unknown role name is a compile error; a scope
  that matches no parameter is a compile error that points at
  `no_decay=["vector"]`; and exempting every role a parameter can have refuses
  in favour of writing `weight_decay=0.0`.
- `crates/nsl-codegen/src/muon_roles.rs` is renamed `param_roles.rs`
  (`classify_muon_param_roles` -> `classify_param_roles`) — it now has two
  consumers, and the Muon-specific name would have implied weight-decay groups
  only work under Muon.
- Gates: `crates/nsl-cli/tests/weight_decay_groups_gate.rs` (12 tests). The two
  load-bearing ones are exact equivalences rather than "the numbers moved" — an
  all-rank-1 model under `no_decay=["vector"]` must be BIT-IDENTICAL to
  `weight_decay=0.0`, and an all-rank-2 model must be BIT-IDENTICAL to plain
  `weight_decay=λ` — each with an anti-vacuity assertion that λ affects the
  fixture at all. Plus `optim_groups` unit tests and two `UpdateKey` bucketing
  tests (λ splits launches; uniform λ stays one launch).

### Fixed — dispatch results classify from the ffi_ownership table without allowlist entries (ELTLS v2a)

- **A dispatch arm whose NSL-level name was missing from the
  hand-maintained owning-ref allowlist stranded its fresh result** in
  nested, receiver, and bare-statement position — the recurring leak
  class behind three separate hand-patch cycles (#423 sdpa, #424
  rmsnorm/dropout, #426 lt_scalar/multinomial). `reduce_max` and `clamp`
  were live instances: `reduce_max(x, 0, 0).item()` stranded one block
  per call.
- Root closure at the dispatch boundary: `compile_call_by_name` records
  its last non-void emission (the EFFECTIVE extern name — alias-resolved,
  so alias-path calls are covered — plus the result Value), and the
  `compile_expr` Call arm registers the dispatch result as Owned when it
  is provably that emission's fresh output: the emission counter must
  have advanced DURING the dispatch (stale records from earlier
  statements or function bodies cannot match) and the recorded Value
  must be identical to the value the arm returned. The nested tracker
  accepts the per-statement `dispatch_fresh` set as an owning signal, so
  any arm composing table-listed FFIs is tracked automatically — the
  allowlist is no longer the single point of drift.
- Deliberate refusals, documented in
  `register_dispatch_result_ownership`: nothing registers inside tape
  regions (the promote_to_tape_held retain balance stays byte-identical
  to pre-v2a); an arm's own ownership claim always wins; and the
  "thread `state` through all ~867 `compile_call_by_name` sites" v2
  sketch was REJECTED as unsound — machinery-managed emissions (clone,
  to_device in stmt.rs choreography) registered Owned would double-free
  inside train regions.
- Five new table classifications, each from a per-runtime freshness
  read: `nsl_tensor_reduce_to_shape` and `nsl_tensor_gelu_backward`
  (closing the drift-gate's STILL-UNCLASSIFIED note; the raw-call strand
  that note predicted turned out to be unreachable — the semantic
  checker refuses raw extern names, so these matter for machinery and
  future arms), plus `nsl_tensor_clamp`, `nsl_tensor_conv2d`,
  `nsl_tensor_maxpool2d` (live dispatch-arm terminals that stranded).
- The method table's `.to()` comment claimed "no refcount bump" — stale:
  `nsl_tensor_to_device` retains before returning the receiver on the
  same-device path (a counted reference). The static entry stays false
  (dtype-arg `.to()` forms route to unverified custom-dtype FFIs and
  model `.to()` returns the model); the dynamic path classifies the
  device-transfer form from the FFI actually emitted, fixing the
  pre-existing `.to()` receiver-position strand as a side effect.
- Gates (`dispatch_ownership_gate.rs`, both GPU): unallowlisted
  dispatch results leave live_blocks == 0 at two round counts with
  exact bound-vs-nested value parity (mutation-proven red at exactly
  (6, 18) — 3 strands/round — with the wrapper disabled, and again with
  only the tracking half disabled); an identity-shaped dispatch fixture
  pins that `copy_data` never steals a live binding (defensive — no
  current arm has the full dangerous shape, stated plainly in the gate
  header).

### Fixed — the three ownership authorities are now drift-gated against each other

- The ownership campaign's recurring bug class was silent divergence
  between the owning-ref Ident ALLOWLIST, the FFI ownership TABLE
  (`ffi_ownership.rs`), and the runtime's actual extern inventory: three
  leak cycles came from allowlist omissions, the allowlist carried a dead
  `"slice"` entry, and the table carried a documented-wrong
  `nsl_tensor_slice` borrow plus TWELVE entries naming FFIs that do not
  exist (`nsl_tensor_concat` vs the real `nsl_tensor_cat`,
  `nsl_tensor_tanh` vs `nsl_tensor_tanh_act`, `nsl_tensor_log_softmax`
  vs `nsl_tensor_logsoftmax`, `nsl_tensor_max` vs
  `nsl_tensor_reduce_max`, and eight keys with no counterpart at all —
  min/argmin/permute/scatter/view/broadcast_to/sum_to_scalar/
  mean_to_scalar).
- New `ffi_ownership_drift.rs` (3 CPU gates, each mutation-proven red):
  the allowlist parsed from source must equal the gate's classified map
  exactly; every mapped FFI must be in the table as OwnedNewResult; and
  every table key must name a real `pub extern "C"` fn.
- Table corrections: dead keys renamed to the real externs or removed
  (removal is behavior-neutral — absence takes the instrumented
  fallback); `nsl_tensor_slice` flipped to OwnedNewResult (the runtime
  allocates fresh on both paths — triple-re-verified across the
  #423/#426/#427 cycles, and the owning classification has been live on
  the method/free-fn paths with leak gates green throughout; the borrow
  misfile governed only the two `register_ffi_result_ownership` sites,
  where it under-freed); the Ident-allowlist family added with the
  verifications those PRs recorded; `nsl_tensor_topk` classified
  NotATensor (dict handle — `dict_lifetime.rs` owns its lifecycle).
- The generic-dispatch `tensor_unary_runtime_alias` fallback mapped
  `tanh` to the nonexistent bare spelling — an "undefined function"
  error instead of a fallback; corrected to `nsl_tensor_tanh_act`.
- Full emission-site registration (threading `state` through
  `compile_call_by_name` so the table becomes the single runtime
  authority) is the queued v2; these bindings close the drift class
  first.

### Fixed — nested builtin arguments no longer strand their results

- **A builtin call nested as another builtin's argument stranded its
  fresh output** — `sum(cumsum(g, -1))` leaked the inner cumsum result 1
  block per call (device-independent) while the bound spelling was
  clean. 184 argument-compilation sites across `compile_call`'s dispatch
  arms used `compile_expr`, which never registers a nested call's result
  as a statement temporary; all now use `compile_nested_expr`, whose
  tracking predicate registers exactly fresh-owning tensor results
  (idents, literals, and non-tensor arguments are untouched — the
  tracking fires only for `expr_call_returns_owning_ref` + tensor-typed
  results, so string/path/int arms are no-ops by construction).
- Gate `nested_arg_temporaries_gate.rs` pins the sampling + norm +
  reduction families incl. a double-nested shape at zero per-round
  strand (red-proven at exactly 6/round pre-fix) with bound-vs-nested
  VALUE parity folded in. Method-form receiver chains were already
  tracked (PR #423); method-form ARGUMENTS stay as-is (lists/ints
  today).
- **A pre-existing loop-scope ICE became trivially reachable and is
  fixed** (review HIGH): temps registered while compiling a WHILE
  condition / FOR iterable predate the loop-scope mark, and
  `free_tensor_temporaries`' whole-list drain stole them — leaving the
  exit cleanup's `[scope_start..]` slice out of bounds, a COMPILE-TIME
  panic (`while sum(cumsum(g,-1)).item() ...` — also reproducible on
  main via method-form conditions). The drain now takes only entries
  above the innermost scope mark, and both loop-exit slices clamp
  defensively. Residuals pinned by the new loop-condition gate: a
  for-iterable evaluates once and sweeps clean; a while condition's
  per-evaluation temps still strand until condition temps get a
  header/exit free placement (queued).
- Sites deliberately NOT converted (differently-spelled or unsound to
  convert): module-alias/agent/`@fuse`/kv-cache/kernel-launch argument
  paths, DataLoader data/labels, the escape-gated generic-call path
  (converting it unconditionally would be unsound), and indirect/lambda
  calls (no escape info — a lambda can store the pointer). These keep
  their status-quo strands.

### Fixed — dict-lifetime follow-ups: loop-local dicts, load_safetensors, builtin shadowing

- **Loop-local dict bindings no longer strand per iteration** (the eltls
  clear twin for dicts). A dict `let` in the direct body of a top-level
  `for`/`while` joins the sweep plan's loop-rebind pool: the loop
  lowering zero-predeclares its slot, `compile_var_decl` frees the
  previous iteration's dict before each rebind, and the return sweep
  frees the last one. Measured 2 blocks/iteration → 0 across for/while/
  break/zero-iteration/script-scope shapes
  (`loop_local_dicts_are_cleared_per_rebind`); admission-off and
  clear-off mutations proven red (the latter at exactly the predicted
  4 = stranded-iterations count). A subscript read after the loop is a
  checker error, so the scan's outside-the-loop mention veto is
  unreachable belt-and-braces.
- **`load_safetensors` joins `FRESH_DICT_BUILTINS`** — a weights dict
  that is only subscript-read is now swept at exit (was 2 stranded GPU
  blocks in the new gate's shape, red-proven). The #427 review had
  verified the runtime side (fresh, solely-owned stores; reads clone);
  the entry waited for its gate per the fail-closed rule.
- **A user `fn` shadowing a sampling builtin no longer miscompiles.**
  The `topk`/`multinomial`/`manual_seed`/`argmax`/`cumsum`/`lt_scalar`
  dispatch arms claimed calls by name while the checker had resolved the
  user fn — a runtime magic-probe abort or a cranelift-frontend panic
  depending on signature shape (found by the #427 review). The arms now
  carry the `!registry.functions.contains_key` guard ~30 sibling arms
  already had, so the user definition wins, matching the checker
  (`builtin_shadow_dispatch.rs`, CPU-runnable).

### Fixed — dict locals no longer strand their stored tensors (aggregate-lifetime gap)

- **A `Dict<Str, Tensor>` local was never freed outside DataLoader
  lowerings** — `let r = topk(scaled, k)` stranded the dict plus both
  stored tensors on every call: 2 VRAM blocks per call once the sampling
  FFIs gained GPU support, and CPU heap before that (CPU sweep proven by
  trace: exactly +2 tensor destroys per call on the minimal fixture; the
  large `topk(x, 1M)`-loop fixture's remaining RSS growth is
  allocator-level retention of transients — identical with the fix
  reverted, no handle strand: destroys ≥ creations per iteration). The dict OWNS its
  stored tensors outright (`nsl_dict_set_str` stores the raw handle;
  every subscript read CLONES), so nothing but the dict itself ever
  releases them.
- The return-local sweep now carries a `nsl_dict_free_tensor_values` pass
  armed by a conservative usage scan (`dict_lifetime.rs`): a dict local
  qualifies only when its single binding is a top-level `let` from a
  fresh-dict BUILTIN call (whitelist; today `topk`) and every other
  appearance is a subscript READ. Aliases (`let r2 = r`), subscript
  writes, ANY pattern re-binding of the name (loop patterns,
  comprehension generators, match arms), a locally-bound callee name, and
  opaque blocks all veto — the failure direction of a wrong admit is a
  double-free, so the scan only allows shapes it fully understands.
  Review proved that direction twice with runtime crashes on the first
  cut (a `[0 for r in ...]` generator shadowing the candidate's slot; a
  lambda returning its captured dict minting two "owning" candidates) —
  both are vetoed and pinned as gates now
  (`comprehension_generator_shadowing_vetoes_the_sweep`,
  `lambda_returned_dict_aliases_do_not_double_free`).
  `load_safetensors` also returns a fresh solely-owned tensor dict
  (reviewer-verified) but stays OFF the whitelist until it has its own
  gate. Probed: returning a dict
  from a user fn is structurally impossible today (checker "wrong type"),
  and passing one to a user fn dies in codegen — the two big escape shapes
  cannot occur.
- The whole stdlib sampling chain is now leak-free on GPU logits:
  `sample_top_k` + `sample_top_p` went from 4 stranded blocks per round to
  ZERO (`topk_dicts_no_longer_strand` ratchets the old 2-per-topk pin).
  New gate file `dict_local_lifetime_gate.rs` (5 gates) pins the allowed
  shape, the veto shapes (run-success asserts catch the double-free
  direction), and the loop-local status quo. Scan-disable and veto-removal
  mutations proven red.
- Loop-local dict bindings (`for ...: let r = topk(...)`) keep the
  pre-existing per-iteration strand — the eltls predeclare/clear twin for
  dicts is queued follow-up work; the real generation pattern (sampling
  helpers called per token) is function-wrapped and fully covered.

### Fixed — sampling builtins no longer segfault on GPU tensors

- **`topk` / `multinomial` / `argmax` / `cumsum` / `lt_scalar` accepted GPU
  tensors and dereferenced their device pointers on the host** — an
  immediate SIGSEGV. This is the entire tensor-input surface of
  `sampling.rs`, and it is exactly what `stdlib/nsl/inference/sampling.nsl`
  (`sample_greedy` / `sample_top_k` / `sample_top_p`) feeds with
  GPU-resident logits, so *every* sampling strategy crashed on a model
  living on the GPU. Reproduced on all five ops (RTX 5070 Ti, CUDA 13.3).
- Fixed with the established CPU-redirect idiom (`nsl_tensor_gather`'s
  non-dim-0 arm): GPU inputs are staged through the host and results are
  handed back on the input's device, tape paused across the redirect. The
  f32→f64→f32 round trip is value-exact; sampling tensors are vocab-sized,
  so the transfer cost is noise next to a forward pass.
- **`return multinomial(...)` / `return lt_scalar(...)` double-owned** —
  both were missing from the owning-ref allowlist (the `return rmsnorm`
  class), stranding one block per call from a user fn. Added with
  per-runtime verification; `topk` deliberately stays out (it returns a
  dict handle, not a tensor).
- New gate `sampling_device_gate.rs` (5 tests): per-primitive CPU/GPU exact
  parity, the stdlib chain end-to-end on GPU logits, redirect
  leak-cleanliness, the return-position double-own, and a pin on the
  PRE-EXISTING `topk` dict strand (below). Crash + double-own gates
  mutation-proven red against their reverted fixes.
- **Surfaced, pre-existing, now measured and pinned (not fixed here):**
  (a) `topk`'s result dict owns its `values`/`indices` tensors and no
  codegen path ever frees non-DataLoader dict locals — ~79 MB/call resident
  growth in a CPU `topk(x, 1M)` loop, 2 VRAM blocks/call on GPU (the
  aggregate-lifetime gap); (b) most builtin dispatch arms compile arguments
  with `compile_expr`, so a NESTED call's fresh result (`sum(cumsum(g,-1))`)
  never registers as a statement temporary and strands 1 block/call,
  device-independent (76 arms affected). Both are queued as their own
  items.

### Fixed — CPU tape-mode dropout no longer strands its mask; ref-trace completed

- The CPU tape-recording arm of `nsl_tensor_dropout` published its mask and
  then bumped it again before recording. The mask is TAPE-ONLY — no caller
  ever receives it — so the tape takes the publish reference itself and
  `release_tape_op_refs` frees it exactly once, the accounting the GPU arm
  has always used. The extra bump stranded one mask tensor per
  training-mode CPU dropout call (independent-review finding on 1def2b9f).
  Pinned by `cpu_tape_mode_dropout_mask_refcount_is_exactly_the_tapes`,
  which asserts the recorded mask's refcount is exactly the tape's one —
  red before the fix (rc=2), and rc=0 would flag the opposite
  (use-after-free) direction.
- `nsl_tensor_release` (decrement-without-free) now emits a
  `[tensor-trace] release` event under `NSL_DEBUG_MEM_TRACE=1`, closing the
  last unlogged refcount path (retain/release pairs were half-visible).
- Investigated, deliberately NOT changed: the review's declared-vs-inferred
  return-type divergence (a `-> Tensor` method whose return expression is
  indeterminate skips the callee-side retain). Two candidate reproducers
  produced correct values and balanced refcounts — the skip is symmetric
  (the same static type gates both the retain and the sweep), so no
  demonstrable defect exists to gate a change against, and adding
  retains/frees without a reproducer is how UAFs get shipped. Recorded as a
  watch item alongside the ffi_ownership dual-authority refactor.

### Fixed — Coder-50M pure-inference forward no longer leaks AT ALL (roadmap item 1 complete)

- **The remaining +33 blocks / +132 MB per forward is closed: per-call growth
  is now ZERO** (154 blocks / 216 MB at N=1 and N=3, RTX 5070 Ti). At the
  campaign's start a Coder-50M forward stranded +292 MB per call.
- **Root cause A — the Return-arm double-own, recurring:** a compiled
  function's Return arm conservatively retains any result it cannot prove
  owning. `return rmsnorm(...)` (RMSNorm) and `return dropout(...)` (GQA,
  SwiGLUFFN) hit that arm because the free-function builtins were missing
  from the owning-ref allowlist — the same bug previously fixed for
  `return scaled_dot_product_attention(...)`, one allowlist gap later. Each
  nested call stranded its fresh output: 4 blocks per TransformerBlock call
  (two RMSNorm outputs + two eval-`dropout` clones — eval dropout returns
  `nsl_tensor_clone`, which ALWAYS allocates), ×8 layers = 32 of the 33.
  Fixed by adding `rmsnorm`/`dropout`/`layernorm`/`bias_add`/`gather`/
  `embedding_lookup` to the allowlist, each verified against its runtime
  implementation on every path (the stdlib's complete set of
  return-position tensor builtins).
- **Root cause B — the Assign-arm double-own:** the Return handler had an
  Unknown→Owned upgrade via `expr_call_returns_owning_ref`; the ASSIGN
  handler did not, so `x = self.norm.forward(x)` (model-method results carry
  no ELTLS registration) conservatively retained and stranded one block per
  reassignment — the final-norm strand on every Coder-50M forward
  (`forward_core`'s post-loop `x = self.norm.forward(x)`). The Assign
  handler now carries the identical upgrade.
- **The GQA gate's former "1 block/call floor" was a misdiagnosis** — that
  block was GQA's double-owned `dropout` clone, not the caller-bound live
  result (top-level `let`s are swept at main's return). The ceiling is now
  0.0 and `the_gqa_residual_is_closed_exactly` pins exactly zero.
- **New composition gate**
  (`nested_model_composition_and_reassignment_do_not_strand_per_call`):
  the exact Coder-50M TransformerBlock/forward_core shape — nested
  model-method args and operands, builtin-returning callees, assign-in-loop
  and post-loop reassignment — asserts zero per-call growth. Mutation-proven
  red against both root causes independently (M6: Assign upgrade reverted;
  M7: rmsnorm/dropout entries removed — which also reddens the GQA gate).
- Diagnostics: `NSL_DEBUG_MEM_TRACE=1` now also prints `retain` and
  `deref-owner` events, completing the per-handle ref-history needed to find
  double-owns (this is how both root causes were isolated).

### Fixed — the GQA view-chain residual: Unknown-typed chain links no longer strand (roadmap item 1)

- **Root cause, two layers.** The semantic member table
  (`check_member_access`, nsl-semantic/src/checker/ops.rs) typed the results
  of `expand`, `contiguous`, `unsqueeze`, `select`, `slice` and `cumsum` as
  `Unknown` — only `reshape`/`transpose` were covered — so every LATER link
  of a method chain was Unknown-typed. Codegen ownership tracking
  (`track_owned_tensor_expr_result`) filters on the result type and the
  owned-temporary classifier required a proven-Tensor receiver, so each such
  anonymous link silently stranded its handle: per
  `expand(..).contiguous().reshape(..)` chain, the expand view pinned the
  source block (the RoPE output) and the contiguous materialisation stranded
  outright. `GroupedQueryAttention::forward` leaked 4 blocks (12 MB) per
  pure-inference call.
- **Fix layer 1 (root):** the six methods are now typed in the member table —
  `contiguous`/`cumsum` return the receiver's type (shape-preserving),
  `expand`/`unsqueeze`/`select`/`slice` return tensor-typed unknown-shape
  results, and `expand` gets its exact target shape at the call site the same
  way `reshape` does. Drift gate:
  `nsl-semantic/tests/tensor_method_result_typing.rs` (fails naming the
  method if any of them regresses to Unknown).
- **Fix layer 2 (backstop):** the codegen ownership filters accept
  indeterminate types exactly where the dispatcher already defaults to tensor
  dispatch — an Unknown-receiver method call classified by the same ownership
  table the dispatch uses, and an indeterminate RESULT type accepted only for
  table-owning tensor-method calls (never Ident-callee builtins or user fns,
  where an Unknown-typed value can be a scalar or list and freeing it as a
  tensor is memory corruption; a Cranelift I64 pointer-type guard backs this
  up). With layer 1 deliberately reverted, this layer alone still holds GQA
  at the 1-block floor — mutation-verified both ways.
- **Fix layer 3:** the free-function spellings (`contiguous(t)`,
  `unsqueeze(t,d)`, `tensor_slice(t,..)`, `stack(l,d)`, `tensor_cat(l,d)`,
  `cumsum(t,d)`, `argmax(t)`, `causal_mask(n)`) joined the Ident allowlist in
  `expr_result_is_owned_temporary`, each entry verified against its runtime
  implementation. (The entry first shipped as `"slice"`, which is dead — no
  free function of that name exists; review caught it and it is now the real
  spelling `tensor_slice`.) Measured before:
  `contiguous(x.transpose(0,1)).sum()` retained 2 blocks/call while the
  method spelling retained 1 — same runtime call, different books.
- **Review hardening (independent review of 734c548e):** the
  indeterminate-receiver arms now mirror the dispatcher's precedence — an
  indeterminate-typed Ident registered as a model-array or agent variable
  routes to compiled model/agent methods, so it is never classified by the
  tensor ownership table (`indeterminate_receiver_takes_tensor_dispatch`).
  Unguarded, a method sharing a table name (`fn mean(self) -> int`) would
  have its plain-I64 return freed as a tensor — memory corruption, not a
  leak. Mutation-tested reachability: today's checker types model-array loop
  vars concretely, so the model-array shape does not currently reach the
  unguarded arm; the live exposure is the M56 @pipeline_agent path (agent
  vars are Error-typed by design) and any future inference regression. CPU
  regression gate:
  `model_array_methods_with_tensor_table_names_are_not_freed_as_tensors`
  (doubles as the crash reproducer if inference ever de-types those
  receivers).
  The free-function gate now exercises every allowlist entry that yields a
  fresh GPU block (contiguous/tensor_slice/stack/tensor_cat) and documents
  why the other four cannot be observed by a GPU block gate (views pin
  already-live roots; cumsum/argmax/causal_mask are host-side — and
  device-blind, a pre-existing hazard noted for follow-up).
- **Measured (RTX 5070 Ti, CUDA 13.3):** GQA `[2,1024,512]` forward
  5 → **1 block/call** (the caller-bound result — the floor), 16 → 4 MB/call.
  Coder-50M `[2,1024]` pure-inference forward +65 → **+33 blocks/call**,
  +228 → **+132 MB/call**; N=3 reserved 2.10 → 1.99 GB. The remaining +33 is
  the separate nested-model-method-argument class (escape.rs's sound refusal
  for un-analysed callees), unchanged by this fix.
- **Gates:** `view_chain_leak_gate.rs` ceiling lowered 5.0 → 1.0;
  `the_residual_is_still_present_and_bounded` (which deliberately failed on
  improvement) replaced by `the_gqa_residual_is_closed_exactly`, pinning
  exactly one retained block per call in both directions — fewer than one
  means the bound result itself was freed, the use-after-free direction. New:
  `free_function_chain_links_do_not_strand_per_call` and
  `unknown_typed_receiver_chains_do_not_strand_per_call` (anti-vacuity via
  the compiler's "defaulting to tensor dispatch" warning and driver
  allocation counters). Every gate proven RED against its own layer's
  reverted fix before shipping.
- **Diagnostics:** `NSL_DEBUG_MEM_TRACE=1` now prints a `[tensor-trace] new`
  line at every tensor-handle publish (pointer, data, len, ndim, device,
  data-owner), pairing with the existing free trace so leaked handles can be
  diffed generation-accurately instead of by data-pointer heuristics.

### Fixed — RoPE configuration was dead config; attention output projection was not depth-scaled

- **`ROPE_THETA` never reached the model.** `models/coder1b` and
  `models/coder7b` both declared `ROPE_THETA = 500000.0` (and
  `MAX_SEQ_LEN = 2048`), but `stdlib/nsl/nn/gqa.nsl` hardcoded
  `RotaryEmbedding(head_dim, 1024, 10000.0)`. Every model in `models/`
  therefore trained at theta=10000 with a 1024-row rotary table regardless
  of what its config said — the "Llama-3 long-context readiness" the model
  comments claimed was not what ran. `GroupedQueryAttention` now takes
  `max_seq_len` and `rope_theta` explicitly and threads them through; the
  models pass their own constants. `coder500m` / `coder50m` / `coder-rl` /
  the muon benchmarks pass `(1024, 10000.0)`, which is what they were
  effectively using, so their numerics are unchanged.
- **Asymmetric residual depth-scaling.** `SwiGLUFFN.w_down` carried the
  GPT-2 `1/sqrt(2 * n_layers)` residual-stream scale but the attention
  output projection `wo` did not, so only half the contributions into the
  residual stream were damped. `GroupedQueryAttention` now takes `n_layers`
  and applies the same factor to `wo`. The models' FFN scale is now derived
  from `n_layers` too instead of a hand-written `sqrt(32.0)` / `sqrt(64.0)`
  literal per model.
- **RoPE cos/sin are cached instead of rebuilt per call.**
  `RotaryEmbedding` now precomputes cos/sin over the full
  `[max_seq_len, dim]` position grid at construction (`max_seq_len` was
  previously accepted and ignored) and `forward` gathers the first
  `seq_len` rows. The old path rebuilt `arange -> outer product -> cat ->
  cos/sin` on every call, twice per attention layer (Q and K) — 32 rebuilds
  per forward for a 16-layer model. Bit-exact with the old path (sum of
  squared differences == 0 over the shared prefix). The tables use `_`
  prefixes so they are excluded from the trainable parameter set — no
  gradients, no AdamW moments, no weight decay.
  - The gather is deliberate, not a slice: `PrimalOp::Slice` bakes its
    bounds in as compile-time constants, so a runtime `seq_len` cannot go
    through it, and neither `.slice()` nor `tensor_slice()` is reachable
    from the source-AD extractor at all (both fall back to tape AD).
    `embedding_lookup` takes runtime indices and is already a first-class
    source-AD op.
  - `forward_with_positions` (the packed path) still builds its table per
    call: its positions are per-token, and its op sequence is what
    `pca_rope` / the segment-masked flash kernel selection match on.
- **Packed entry points wired into the coder models.**
  `NSLCoder.forward_train_packed(input_ids, segment_ids, position_ids,
  training)` and `TransformerBlock.forward_packed(...)` route through the
  stdlib GQA's `forward_packed`, so a `DataLoader(..., packing=true)`
  stream gets document-masked attention and per-document RoPE position
  reset. The `pretrain_fase.nsl` demos stay on the unpacked path on
  purpose — their corpus is a single repeated token with no document
  boundaries — and the READMEs now say so.
- **New drift gate** `crates/nsl-cli/tests/model_config_drift.rs`: NSL
  `const`s are not in scope inside model field initializers and
  `[TransformerBlock; N_LAYERS]` does not parse, so every model restates
  its architecture as literals. The gate parses `config.nsl`, the
  `model.nsl` const block, and the literals actually passed to
  `TransformerBlock(...)` / `randn([V, D])`, and fails on any disagreement.
  It is anti-vacuous: a stdlib-GQA model missing from its coverage list
  fails the gate.
- **Documented, not fixed:** `AdamW(weight_decay=...)` in a `train` block
  is one scalar over all trainable parameters, RMSNorm gains and the tied
  embedding included — NSL's train-block optimizer DSL has no parameter
  groups, so the conventional decay-2-D-weights-only split cannot be
  expressed. The per-parameter classification a fix would need already
  exists in `crates/nsl-codegen/src/muon_roles.rs`. The `coder1b` /
  `coder7b` READMEs also now state the pretrain token budget against a
  compute-optimal reference (~15% and ~1.7% respectively) so their cert
  curves are not read as quality claims.

### Fixed — flip hardening: one shared math-mode resolution, and the two gates the flip broke

- The cuBLAS handle and the dispatch coupling each did their own lazy
  `resolve_math_mode()` env read at different first-use times, so a process
  mutating `NSL_MATMUL_TF32` between them silently landed in a mixed cell
  (e.g. TF32 handle + copy dispatch — the measured-slower configuration)
  with no numeric signature. Both now read one `OnceLock`; first reader
  wins, both consumers agree forever (review finding).
- The full lane caught two e2e gates broken by the flip, both the same
  shape: they compare two paths whose transposes sit at different call
  sites, so the coupled default put the compared paths on different kernel
  arms and the reduction-order delta crossed their tolerances
  (`muon_p1_gate` at 1.09e-6 vs 1e-6; the FASE-vs-plain checkpoint gate
  past 5e-3 after 32 AdamW steps). Both gates' subjects are not dispatch
  policy — the Muon primitive and optimizer windowing semantics — so both
  pin `NSL_MATMUL_TRANSPOSE_VIEWS=0` in their child runs, the same
  isolation move as the existing `NSL_FLASH_BWD_CPU` pin. Dispatch arms
  keep their own value gates.
- OP_T-under-Pedantic — the one env-reachable cell no value gate had ever
  touched — is now gated (`op_t_override_is_correct_under_pedantic`); the
  vacuity probe's expected math-mode signature is parent-declared so the
  pedantic parent can require the ABSENCE of TF32 drift. Stale
  "OFF BY DEFAULT" comments inside the shared dispatch predicate corrected.

### Changed — transposed views now reach cuBLAS as `OP_T` under TF32 (math-mode-coupled default)

- `NSL_MATMUL_TRANSPOSE_VIEWS` grew a per-math-mode default: **OP_T under
  TF32** (the shipped default), the materialising copy under FP32 cores and
  Pedantic. The literal `"1"`/`"0"` always win; any other value falls
  through to the coupling (same tri-state discipline as `NSL_MATMUL_TF32`).
- Flipped on two levels of measurement plus gates, per the process the
  reproducer commit demanded: the per-call grid (OP_T 0.65x/0.72x/0.46x
  under TF32) and a Coder-50M 20-forward loop — **63.4 → 56.6 ms
  end-to-end (1.12x)**, three paired runs, sgemm at parity, the whole win
  the vanished 96 MiB LM-head copy; plus the ~90 MB of peak memory that
  copy always cost. OP_T values under TF32 are gated by
  `matmul_dispatch_under_tf32` (path-witnessed).
- The coupling is a decision per **measured cell**, not a shape heuristic:
  FP32 cores keep the copy because OP_T measured 1.40x slower on the LM
  head in that cell; Pedantic keeps the copy because that cell is
  unmeasured and unmeasured cells keep the conservative arm.
- New behavior gates spawn children with all three matmul variables
  controlled: default-under-TF32 takes OP_T, default-under-f32/Pedantic
  materialises, and the env var beats the coupling in both directions.
  The under-TF32 suite gained an explicit copy-arm configuration, and its
  path witness now takes its expectation from the parent instead of
  re-deriving the resolution it is testing.

### Fixed — gpu-cert lane: `cpu-stub` class ends the permanent 2-NOTFOUND noise

- The gate-inventory scanner classified `#[cfg(not(feature = "cuda"))]`
  placeholder tests into RUN classes, but the lane builds WITH the cuda
  feature, which compiles those tests out of every binary it can run — a
  guaranteed, permanent NOTFOUND on every sweep
  (`wrga_b32_trigger_measurement_requires_cuda`,
  `fp8_dispatcher::cuda_feature_required`). The scanner now tracks the cfg
  through the attribute stack and files them as `cpu-stub`, a never-run
  class; the flag is cleared at the first non-attribute line so a cfg-gated
  item elsewhere in a file cannot demote a later, genuinely runnable gate.
- Regenerating the manifest also heals a red CI drift gate: the TF32 commit
  renamed two `matmul_tf32_mode` gates after its manifest regeneration, and
  the fp8 device-guard commit added five gates without one.

### Fixed — inventory-scanner review findings, pinned by committed probes

- The batch review probed `gpu-gate-inventory.awk` with synthetic files and
  found holes the in-tree stubs don't exercise: a one-line
  `#[ignore] #[cfg(not(feature = "cuda"))]` classified `gpu` (a permanent
  NOTFOUND back from the dead), the mirror order vanished from the
  inventory entirely, a block-comment line inside the attribute stack
  cleared the cfg flag, and spacing variants of the cfg escaped the
  anchored regex. All fixed; a block-comment state machine also stops the
  scanner from parsing `#[ignore]` prose inside `/* ... */` as gates.
- The probes are committed as `.rs-probe` fixtures with a 7-gate test
  (`gpu_gate_inventory_scanner`) — not inline strings, which the
  line-oriented scanner itself inventoried as phantom gates; not `.rs`
  files, which the inventory's `find` would walk.
- Known limit, documented in the scanner: `#[cfg(not(feature = "cuda"))]`
  on a `mod` of stubs is still defeated by the boundary rule (surfaces as a
  visible NOTFOUND, never silent coverage loss). No in-tree stub is
  mod-scoped; brace tracking waits until one is.
- The OP_T measurement children now strip `NSL_MATMUL_PEDANTIC`, which
  outranks `NSL_MATMUL_TF32` and would have silently measured pedantic math
  in all four grid cells under a pedantic-pinned shell.

### Added — dispatch-path correctness gates UNDER TF32

- Every pinned matmul suite documents the same gap: dispatch paths were
  gated at full f32 only, while TF32 — the shipped default — selects a
  different cuBLAS kernel family that could carry its own operand-mapping
  bug. `matmul_dispatch_under_tf32` closes it from fresh child processes
  (the math mode resolves once per process): transposed left/right/both,
  batch collapse, the tied-LM-head composite, and fused wgrad accumulation,
  each against an f64 CPU reference at 5e-3 relative (~5x TF32 drift,
  three orders below a mapping bug).
- Two anti-vacuity devices: a probe that requires a 256^3 product to drift
  MORE than full f32 ever would (so a math-mode resolver regression fails
  loudly instead of making the gates trivially green), and a path witness
  that counts `nsl_strided_copy_f32` launches — at small K the OP_T and
  copy arms produce bit-identical values, so values alone cannot prove
  which arm ran.
- `op_t_exemption_is_correct_under_tf32` is the named precondition for
  flipping the `NSL_MATMUL_TRANSPOSE_VIEWS` default, now that the
  reproducer below shows OP_T is faster under TF32.

### Added — committed reproducer for the OP_T-vs-copy measurement, which now flips under TF32

- `matmul_transposed_operand::the_op_t_tradeoff_is_remeasurable` (class
  `diagnostic`, manual-run) re-measures the grid behind the
  `NSL_MATMUL_TRANSPOSE_VIEWS` default from four fresh child processes —
  {copy+OP_N, OP_T} × {full f32, TF32} over the three documented shapes —
  with the clock-ramp discipline (1 s busy warmup per shape, drained
  per-call timing). Until now the default rode on a one-off hand
  measurement reconstructible only from prose.
- First run reproduced the documented f32 table (OP_T 1.40x slower on the
  LM head vs the recorded 1.51x) **and showed the premise inverts under
  the new TF32 default**: OP_T measured faster on every shape (0.65x /
  0.72x / 0.46x). The default stays OFF — OP_T has no correctness gates
  under TF32 yet, and a dispatch default should not move on one grid — but
  both doc sites now record that the "copy wins" table is a property of
  FP32-core math, and the flip is an open decision rather than an
  oversight.

### Changed — `@fp8_compute × --source-ad` joins the feature-composition registry

- The item-9 refusal is now a registered item-20 rule
  (`@fp8_compute` Conflicts `--source-ad`, enforced in `stmt.rs`), so the
  deleted-refusal and fragment gates defend it like every other composition.
  The refusal message now names `--source-ad` explicitly — required by the
  fragment-distinctiveness gate, and clearer for the user. The subprocess
  sweep cannot drive a source-level decorator from the command line, so the
  rule is covered by the source tier only.

### Fixed — fp8 host-path FFIs: GPU segfault and wrong tape gradients

- Every fp8 FFI (`nsl_fp8_cast`, `nsl_fp8_compute_scale`,
  `calibrate_gradient_scale`, `nsl_fp8_update_calibration`,
  `nsl_mxfp8_quantize`, `nsl_nvfp4_quantize`) walked `t.data` with
  `from_raw_parts` as host memory. Two live consequences:
  - **GPU segfault:** the tape-AD backward for `@fp8_compute` hands
    GPU-resident gradients straight into `calibrate_gradient_scale` and
    `nsl_fp8_cast`, so `@fp8_compute` + `--tape-ad` + GPU crashed at the
    first backward step — and since source AD refuses `@fp8_compute`, that
    was the only path. Reproduced as SIGSEGV by the new
    `fp8_gpu_device_guard` gates against the pre-fix code.
  - **Wrong CPU gradients:** the backward transposes `saved_a`/`saved_b` as
    zero-copy views, and the stride-blind cast read the shared storage flat
    while stamping fresh row-major strides onto the view's shape — turning
    the transpose into a reshape. `grad_A = G @ reshape(B)` instead of
    `G @ B^T`: misplaced operands, not reduced precision. Confirmed by
    mutation control (the new rectangular-shape tape gate fails against the
    old code with errors of order 1e0).
- Fix: a `stage_for_host_read` guard on every host-path FFI — D2H transfer
  for device tensors, `nsl_tensor_contiguous` for CPU views, loud refusal
  (abort with a named FFI) for dtypes outside {f32, f64}, which would read
  at the wrong element stride. Quantizer outputs are built on host and moved
  to the input's device, replacing the old behavior of stamping the input's
  device label onto a host allocation.
- `nsl_fp8_update_calibration` refuses a device-resident or non-{f32,f64}
  running-max state tensor instead of corrupting it through a host
  read-modify-write (no in-tree caller creates one).
- New gates: 3 CPU unit gates (view ordering for cast + block quantizer,
  end-to-end tape gradients on rectangular shapes) and 5 GPU gates
  (`fp8_gpu_device_guard`: cast/scale/backward/tape/block-quantizers vs the
  CPU run on identical values). The two load-bearing GPU gates joined the
  canary list (26 entries).
- Post-review hardening: the guard no longer trusts `is_contiguous()` for
  rank-1 tensors (it returns true regardless of strides, so a 1-D stride-0
  expand view would have flat-read past its buffer — gated by a new unit
  test with a mutation control); the tape backward now quantizes each
  operand ONCE instead of staging the gradient three times per matmul
  (five→three synchronous PCIe round-trips per step on GPU); the dtype
  refusal now fires for empty tensors on every FFI.

### Changed — TF32 is now the matmul default (`NSL_MATMUL_TF32=0` opts out)

- Every `cublasSgemm` NSL issues now runs on tensor cores with a 10-bit
  mantissa instead of FP32 CUDA cores. This is a numerics change for the whole
  stack, taken deliberately.
- **Measured at model scale, not extrapolated from a microbenchmark.** On a
  Coder-50M forward (RTX 5070 Ti, sm_120), steady state over the second half
  of a 20-forward loop:

  | | FP32 cores | TF32 | |
  |---|---|---|---|
  | sgemm | 33.3 ms | 21.4 ms | **1.55x** |
  | all kernels | 76.5 ms | 64.8 ms | **1.18x** |

  Three paired runs, spread under 5%. **The end-to-end win is ~15%, not 55%** —
  GEMM is only ~44% of kernel time at this size. The N=2048 microbenchmark says
  1.60x and agrees with the sgemm column; it says nothing about the model.
- A first attempt measuring ONE forward per configuration gave
  `{9.9, 9.7, 10.1, 25.3, 25.5}` against `{3.3, 24.1, 35.5, 3.6, 10.3}` — the
  same configuration swinging 10x, because the SM clock idles at 1515 MHz
  against a ~2900 boost and a single forward samples a different point on the
  ramp each run. Numbers from unlooped GPU runs are not evidence.
- Cost: ~13 bits of mantissa per product. Measured at N=2048, pedantic f32
  drifts 1.36e-6 and TF32 9.48e-4 — **698x**, bit-deterministic across runs.
- `NSL_MATMUL_TF32` is a tri-state: only the literal `"1"` and `"0"` are
  honoured and anything else falls through to the default, so a typo'd
  `NSL_MATMUL_TF32=true` cannot quietly change the arithmetic.
  `NSL_MATMUL_PEDANTIC=1` and the `strict-matmul` feature still win.
- **Known property of the dispatcher, now default-visible:** the naive PTX
  matmul kernels run on FP32 cores regardless of the cuBLAS math mode, so a
  product that falls off the cuBLAS path is computed at a different precision
  than the same product on it — `matmul_batch_collapse` measures 1.03e-3
  versus 9.75e-7 for the two arms. In practice nothing falls off: a kernel
  histogram of a 20-forward Coder-50M loop shows 758 `sgemm_cublas` + 212
  `sgemm_cublas_batched` and **zero** naive-matmul launches, because
  `nsl_tensor_matmul` materialises every non-expressible operand first.
- Gates: the operand-mapping suites (`matmul_batch_collapse`,
  `matmul_transposed_operand`) pin `NSL_MATMUL_TF32=0` rather than widening
  their tolerances. They exist to catch a crossed `lda`/`ldb` or an `OP_T` on
  the wrong operand, which produce errors of order 1e0 — full f32 keeps three
  orders of magnitude of headroom for free. `matmul_tf32_mode` asserts both
  halves from fresh child processes: the default IS TF32, and the opt-out
  restores f32.

### Removed — the M31 fusion-graph subsystem (4,200 lines that never ran)

- `epilogue_fusion.rs` (1237), `reduction_fusion.rs` (1573) and
  `fusion_graph.rs` (496) implemented matmul-epilogue fusion, map-into-reduction
  fusion, and DAG-level fusion planning. A `FusionGraph` was never constructed
  outside a `#[cfg(test)]` block or a test file, so **none of the three passes
  was reachable from any compilation**. Their only non-test references were to
  each other.
- `epilogue_fusion`'s PTX synthesiser was also wrong, not merely unused: it
  computed an indexed load (`add.u64 %rd6, %rd2, %rd5`) against an
  **un-indexed store** (`st.global.f32 [%rd4]`), so every thread would have
  written element 0. Nothing noticed because the kernel had never been
  assembled by ptxas, let alone launched; all ten of its tests were
  `.contains()` checks on the emitted string.
- Went with them: `csha_apply::apply_marks_to_graph` / `is_csha_fused` (CSHA's
  interface to the graph — the marks it applied had no reader), the two test
  files that existed solely to exercise the dead passes, and one snapshot.
  `FusionMark` and its producers stay: the fused-backward emitter consumes
  them directly.
- Also removed `fp8.rs`'s `compile_fp8_matmul` and both
  `emit_fp8_matmul_ptx{,_wgmma}` emitters (~365 lines). Both were already
  `#[allow(dead_code)]`, `compile_fp8_matmul` had no caller outside its own
  file, and the PTX targets `sm_90` — unloadable on this repo's sm_120
  hardware. Every test was a string `.contains()`.
- Docs corrected rather than deleted: `docs/wiki/Glossary.md` claimed `@fuse`
  works "via the epilogue fusion pass in `epilogue_fusion.rs`". It does not —
  `stmt.rs` validates the body and extracts its op chain for a fused launch at
  each call site. `docs/summaries/02-gpu-kernels-and-optimization.md` described
  all three passes as shipping features.

### Fixed — cuda-graph eager repair replayed a weight-grad gemm as a plain one

- `GpuOp::Sgemm` recorded no operand transposition and no wrapper identity.
  `sgemm_wgrad_accum` and the plain row-major path both fed one shared hook,
  while handing cuBLAS different mappings — `OP_N/OP_T` for the
  weight-gradient contraction versus `OP_N/OP_N` with the row-major swap.
- The unconditional half was **replay**: the dispatch branched on
  `batch == 1` alone, so a recorded wgrad op was reconstructed as a
  row-major gemm regardless of what else it carried. Wrong contraction,
  silently wrong gradients, no error.
- Equality and the digest could also collide, but only for two gemms sharing
  all three device pointers as well as `(m, n, k, alpha, beta)` — a much
  narrower precondition than the replay bug, and one an earlier draft of this
  entry overstated as "the same eight values compare equal". Corrected here.
- `GpuOp::Sgemm` gains `kind` (`RowMajor` / `WgradAccum`) plus `transa`/`transb`,
  in equality, in the digest, and in replay dispatch. Four unit gates in
  `graph_capture.rs` now pin that those three agree — the failure class being
  fixed is precisely "three places that must agree, and one didn't", and
  nothing had pinned it.

### Fixed — `--profile-kernels` reported every kernel as 0.0 ms

- `kernel_profile.json` gave `"dur": 0.0` and `"ts": 0.0` for every launch and
  `total_kernel_time_ms: 0.0`, on runs doing tens of milliseconds of real GPU
  work. Long-standing enough that a previous campaign attributed kernel cost
  by launch count and bytes moved instead.
- Root cause is `atexit` ordering. Handlers run LIFO. The profiler registered
  its flush from `nsl_kernel_profiler_start`, inside `nsl_args_init` — before
  the first kernel launch, so before the CUDA context exists, so before the
  driver registers its own teardown. LIFO ran the driver's teardown FIRST and
  the flush met a deinitialised driver. Reproduced deliberately: the flush
  then reports `744 of 744 timing queries FAILED (CUDA_ERROR_DEINITIALIZED)`.
  (The same hazard is already documented for the cuBLAS handle, leaked on
  purpose because destroying it post-teardown "produces spurious driver
  errors".) Registration moved to `ensure_event_pool_initialized`, which runs
  from `kernel_launch` after `ensure_context()`.
- Second defect, independent: `cu_event_elapsed_time` discarded its
  `CUresult`, and `cuEventElapsedTime_v2` leaves its out-param UNTOUCHED on
  failure — so a failed query became a 0.0 indistinguishable from a
  measurement. It now returns the status; the flush counts failures, prints a
  refusal naming the driver error, and stamps `timing_valid` /
  `timing_query_failures` into the JSON so a consumer can reject the file.
- Measured after the fix on the cuda-graph fixture: 412/412 durations
  non-zero, `total_kernel_time_ms: 52.26`, `timing_valid: true`.

### Fixed — `@fp8_compute` was silently ignored under source-to-source AD

- The decorator routes matmuls to `nsl_fp8_matmul_training` so the backward
  records a `TapeOp::Fp8MatMul` and applies the E5M2 round-trip. That routing
  lives in `expr/advanced.rs`, which source AD does not use: it inlines the
  method body into a Wengert list, and `wengert_lower.rs` lowers
  `PrimalOp::Matmul` to an unconditional `nsl_tensor_matmul`. Neither
  `source_ad.rs` nor `wengert_lower.rs` contains a single occurrence of "fp8".
- So a user who wrote the decorator and trained with `--source-ad` — the
  default, and the path every production model in this repo uses — got plain
  f32 with no error and no warning.
- The train/grad lowering now refuses, naming each offending
  `Model::method` and pointing at `--tape-ad`, where the decorator does take
  effect. Sub-model methods reached through `self.field` are covered, which is
  how every real model here is structured.

### Changed — dispatch narrowing in `gpu_matmul_f32` (documenting an item 9 side effect)

- The 2-D cuBLAS arm's condition changed from `total_batch == 1` to also
  requiring each operand be contiguous or an expressible transpose. A strided
  2-D operand that is neither now falls through to the naive `nsl_bmm_f32`
  kernel rather than cuBLAS. Not a live behaviour change — `nsl_tensor_matmul`
  is the sole caller and materialises every non-exempt operand first, and both
  kernels are stride-blind — but it removes a defence-in-depth property and
  went undocumented when it landed.

### Fixed — GPU matmul read non-f32 tensors as f32, two bytes for one

- `gpu_matmul_f32` never inspected `.dtype`: it sizes the output `* 4`,
  hardcodes the output dtype to f32, and casts `a.data as *const f32`. A bf16
  or f16 operand carries the same element COUNT in half the bytes, so it was
  read past the end of its allocation — returning plausible numbers rather
  than faulting. It now refuses, naming the dtypes. Mixed-precision GEMM
  dispatch is roadmap item 9 and remains unimplemented; saying so is the only
  honest option.
- Relatedly, `gpu_matmul_f32` returns 0 on a cuBLAS error with a comment
  saying that is "so callers can detect the failure". No caller ever did — the
  null propagated and the process died wherever it was next dereferenced.
  `nsl_tensor_matmul` now checks it, with the shapes still in scope.

### Added — `NSL_MATMUL_TRANSPOSE_VIEWS=1`: CUBLAS_OP_T for transposed operands (opt-in, and the default is faster)

- A 2-D transposed view can go to cuBLAS as `OP_T` instead of being copied.
  For the weight-tied LM head (`x @ embed.transpose(0,1)`,
  models/coder50m/model.nsl:80) the copy is 25,165,824 elements — 96 MiB per
  forward, through a kernel doing a software 64-bit div/rem per element. On a
  Coder-50M forward, `nsl_strided_copy_f32` was the most-launched kernel of
  all: 105 of 308 launches, 453 MB moved.
- **The obvious conclusion is wrong, and only measurement caught it.** That
  copy is not waste; it buys a much faster GEMM:

  | shape | `OP_T` | copy + `OP_N` | verdict |
  |---|---|---|---|
  | `[2048,512] @ [512,49152]` (LM head) | 5.92 ms | 1.04 + 2.89 = **3.93 ms** | OP_T **1.51x slower** |
  | `[2048,512] @ [512,4096]` | 0.344 ms | 0.351 ms | a wash |
  | `[512,512] @ [512,512]` | **0.023 ms** | 0.043 ms | OP_T 1.9x faster |

  End to end: 29 ms/forward with the copy, 33 ms without; peak GPU 2.10 GB
  versus 2.01 GB. So it trades ~90 MB of peak memory for ~4 ms per forward —
  useful when memory binds, a regression otherwise. Hence opt-in, and hence a
  gate asserting the default still materialises.
- Not defaulted on behind a shape heuristic: three data points is not a cost
  model, and this repo already shipped one of those (item 10's autotune DB
  ranked variants by a roofline estimate with no measurement behind it).

### Fixed — every transformer projection ran on a naive scalar PTX kernel, not cuBLAS

- `gpu_matmul_f32` dispatched to cuBLAS **only when `total_batch == 1`**;
  everything else launched `nsl_bmm_f32`, a 16x16 scalar-`fma.rn.f32` kernel.
  `total_batch` counts the OUTPUT batch dims, so `[B,S,C] @ [C,O]` — the shape
  of every transformer projection — had `total_batch = B` and took the naive
  arm, and nothing in codegen flattened it first. A `GroupedQueryAttention`
  forward profiled six `nsl_bmm_f32` launches and ZERO cuBLAS calls.
- Two dispatch paths added. `[..,m,k] @ [k,n]` with a batch-free B collapses to
  a single sgemm — a reinterpretation of A's existing buffer as
  `(batch*m) x k`, summing the same products in the same order. Genuinely
  batched products (in a transformer, QK^T and PV) go to
  `cublasGemmStridedBatchedEx` via a new row-major `sgemm_batched_row_major`.
  `NSL_MATMUL_NO_BATCH_COLLAPSE=1` restores the old dispatch.
- Measured on an RTX 5070 Ti / CUDA 13.3: `[8,1024,512] @ [512,512]`
  **0.67 -> 29.5 TFLOP/s**; a Coder-50M `[2,1024]` forward
  **147.1 -> 33.3 ms (4.42x)**; GQA forward 6 naive launches -> 0.
- `GpuOp::Sgemm` gains batch/stride fields. Batched products used to reach a
  captured CUDA-graph region as `GpuOp::Kernel`, whose params blob
  distinguished them for free; as cuBLAS pseudo-ops they would otherwise have
  compared equal on `(a,b,c,m,n,k)` alone and replayed a graph built for the
  wrong shape.

### Fixed — `NSL_MATMUL_TF32=1` did nothing, and the startup banner said the opposite

- The 2026-04-21 cuBLAS-swap work recorded `CUBLAS_DEFAULT_MATH` as "TF32
  tensor cores on sm_80+" and shipped a banner telling users pedantic mode was
  "~5-10x slower than TF32 default". Both false: that mode does not enable
  TF32 for `cublasSgemm`. Measured at 4096^3 — default 32.9 TFLOP/s, pedantic
  **33.2** (marginally *faster*). `NSL_MATMUL_TF32=1` selected the same
  do-nothing variant, so there was no way to reach tensor cores at all outside
  the muon batch engine. (The P6 muon notes below had already recorded the
  discrepancy; nothing acted on it.)
- `CublasMathMode` is now `{Fp32Cores, Pedantic, Tf32}`, named for what each
  actually does, and `Tf32` sets `CUBLAS_TF32_TENSOR_OP_MATH`. Measured:
  N=2048 **4.12x** faster / 698x less accurate, N=4096 **1.51x** / 190x.
- **The default is unchanged** — it is what NSL has always really done, only
  now described accurately. Flipping it to TF32 buys 1.5-4x on every matmul
  but costs ~13 bits of mantissa per product, which needs a tolerance audit
  across every numerical gate first. Deliberately left as a separate decision.
- `matmul_cublas_tf32_default_sanity.rs` could not have caught any of this:
  its 5e-3 tolerance is satisfied by real f32 and real TF32 alike. The new
  `matmul_tf32_mode.rs` asserts TF32 is both measurably faster AND measurably
  less accurate, because either alone passes for the wrong reasons.

### Fixed — FlashAttention forward output and fused-backward operands crossed the f16/f32 boundary as raw bits

- **Every** FlashAttention forward kernel (classic and v2) stores its
  output as f16, but `nsl_flash_attention`, `nsl_flash_attention_csha`,
  and `nsl_flash_attention_csha_with_saves` handed the caller's f32
  tensor straight to the kernel and never widened: f16 bit pairs
  reinterpreted as f32 are ~0, so every `@flash_attention` inference
  call returned an input-independent noise-scale output (identical
  1.7e-8 L1 across different inputs) with exit 0, and the fused
  training forward fed the same garbage to the loss.
  `nsl_sdpa_fused_forward` has carried the correct stage-and-widen
  contract since PCA Stage C; the `nsl_flash_attention*` entries never
  did — hidden until the launch pipeline was repaired. The three
  entries now launch into transient f16 staging and widen on-device
  into the caller's tensor — but only for f32 `NslTensor` targets: the
  kernel-harness ABI (raw device f16 buffers) keeps the direct f16
  store. Inference certified against exact numpy ground truth at seq 32
  (rel 2.8e-5) and seq 64 (rel 3.7e-6).
- The fused CSHA **backward** had three mirror-image breaks: it reads
  dO and O as f16 (the `D = rowsum(dO . O)` preprocess) but received
  f32 tensors — dO showed the classic reinterpret signature (powers of
  two, -512.0) and amplified into dwq = -inf; the dRMSNorm phase reads
  Wq/Wk/Wv as f16 but got the f32 model weights — dx_norm came out
  2/3 NaN, poisoning dgamma and dx; and the six f16 gradient outputs
  were handed directly to the f32-reading optimizer — weight updates
  collapsed to ~1e-39 denormals ("frozen" wq/wk with exit 0). The FFI
  now narrows f32-boxed f16-read operands into transient staging, and
  the lowering widens the f16 gradients on-device before the extract
  cache.
- Certification: new `csha_gap_gpu_e2e_fused_vs_baseline_parity` gate
  trains the toy twice (per-op AD baseline vs `--csha auto`) and
  compares per-parameter SGD movement — one step makes
  `delta_l1 = lr * |grad|_1`, so this certifies gradient-L1 parity for
  dQ/dK/dV/dgamma. Measured agreement: ≤3e-4 relative on all four
  params (tolerance bar 5e-2). The old e2e fixture was doubly
  degenerate (all-ones x AND weights → dQ/dK mathematically zero, the
  baseline "passed" on one f32 ulp of noise; softmax saturation;
  never `m.to(cuda)`) — rewritten sin-varied on GPU with
  `sum(abs(w - w0))` movement probes, and the historical
  graceful-skip ladders are now hard asserts.

### Fixed — FBIP refcount-based in-place elision silently mutated live variables and MODEL WEIGHTS

- `let s = abs(x)` left `x` overwritten with `|x|`; `print(sum(abs(m.wq)))`
  rewrote the model weight in place — silently, exit 0. The runtime's
  `can_mutate_inplace{,_gpu}` treated refcount==1 as "exclusively owned",
  but ident/member reads hand out pointers WITHOUT retaining, so a live
  `let` binding, a model field, or a dict entry is indistinguishable from
  an owned temporary at the FFI boundary. Training was shielded only by
  accident (tape recording gates the elision under tape-AD; the source-AD
  forward raises the suppression scope) — inference and script-scope code
  corrupted freely. Both predicates are now permanently disabled; sound
  in-place mutation is unaffected (the codegen-proven `nsl_tensor_*_inplace`
  variants and the binary relinquish-flag paths carry compile-time
  ownership proof). Runtime unit tests now pin the input-preserving
  contract.
- **`@flash_attention` inference never worked from compiled NSL code** —
  three stacked breaks, all fixed: the Cranelift call site passes tensor
  BOX pointers which `nsl_flash_attention` forwarded raw as device
  addresses (inputs now resolve through the CSHA data-pointer path with
  CPU auto-promote; CPU-resident write targets hard-abort with an
  actionable message); the launch passed 21 then 30 args against a v2
  kernel that declares 36 parameters (a short kernel-params array is not
  an error the driver reports — cuLaunchKernel SEGFAULTS); and a failed
  launch printed "Refusing to continue silently" while continuing with a
  zeros output (now a real abort, `NSL_FLASH_ALLOW_FAILED=1` escape).
  The "v2 kernel produces ~zero output" residual noted here at the time
  was the missing f16→f32 output widen — fixed above.

### Added — `--grad-integrity` now covers the CSLA (`--layerwise-accum`) windowed backward

- The gradient-integrity gate previously reported `checks=0` under
  `--layerwise-accum` (with a loud not-wired warning): the CSLA windowed
  replay consumed parameter gradients through its own hook, which never
  fed the accumulator. The replay hook now notes each parameter's
  gradient before accumulating it, bracketed once per accumulation
  window (`step_begin` before the range loop, `step_end` after the
  epilogue update) — so `checks` counts optimizer steps and the report
  proves every trainable parameter received a finite, nonzero gradient
  on EVERY window. Repeat notes for the same parameter (one per
  micro-batch under the window bracket) MERGE — finite ANDs, nonzero
  ORs — so a NaN in any micro-batch's partial gradient is caught, not
  just micro-batch 0's (review finding; the accumulator previously kept
  only the first classification). Trailing partial windows don't check,
  exactly as they don't step. Observe-only: the CSLA loss stream AND
  saved model bytes are bit-identical with the flag on. Gate:
  `csla_grad_integrity_layerwise_cpu` (window-count pin, full-coverage
  pin, anti-vacuity via the window-phase counter, stale-warning absence,
  observe-only loss + model-bytes comparison).

### Fixed — inference-loop tensor leak: the loop-let free machinery never fired

- **The ELTLS loop-let predeclare zero-def was emitted INSIDE the loop
  body**, re-zeroing the variable slot at the top of every iteration — so
  the rebind free (`eltls_clear_old_slot`) only ever saw `0` and the
  previous iteration's tensor stranded, at ALL FIVE loop lowerings
  (for/while/while-let/model-array/dataloader — including the one labeled
  "THE PRIMARY FIX FOR THE TRAINING-LOOP LEAK", which had therefore never
  worked at runtime). A script-scope `for i in range(N): let y =
  m.forward(x)` leaked one activation set per iteration forever — the
  "2.3GB/forward" inference-leak class. The predeclare now lands in the
  pre-loop block, so each rebind frees the loop-carried value.
- **Ownership veto on the newly-activated frees:** a symbol is only
  predeclared (and its loop-top free armed) when EVERY binding of it in
  the loop body yields an owning, tensor-typed reference. Ident copies,
  member reads (`let w = m.w_in`), and non-dict subscripts (lists,
  tuples, Unknown all share elements via `nsl_list_get` with no retain)
  hand out the referent's own pointer — freeing a loop-carried borrow
  would free a model weight or a shared tuple element. Int-typed
  bindings are vetoed too: an armed clear on an INT slot hands the
  integer to `free_if_valid`, whose magic probe dereferences any
  8-aligned value >= 0x10000. Loop/while-let/match PATTERN bindings of
  the same name and opaque block constructs (train/grad/distill/quant/
  serve) also veto — their bindings write the same slot. Vetoed symbols
  keep the old declare-in-body behavior.
- **`main()` now runs the return-local sweep**: top-level `let`-bound
  tensors (script inputs, the final iteration's activations) were never
  freed — `compile_main` had no epilogue sweep at all.
- **The return-local sweep (fn returns AND the new main epilogue) now
  filters by semantic type** (tensor/indeterminate locals only).
  `nsl_tensor_free_if_valid`'s pointer probes are not sufficient for
  plain integers: a large 8-aligned int local (e.g. a byte count from
  `gpu_peak_bytes()`) passes the null/low/alignment checks and the magic
  probe dereferences it — a latent segfault in the #415 fn sweeps, caught
  live by the mem-accounting gate once main() gained the sweep.
- **Method-form calls participate in temporary cleanup.** `y.sum()`
  parses as `Call{callee: MemberAccess}`, which the owned-temporary
  predicate never matched — statement-position method temps
  (`print(y.sum())`) stranded one block per call. The predicate now
  covers fresh-result tensor methods (never `.clone()`/`.to()`, which can
  hand back the receiver) and model methods (compiled NSL fns return
  owning refs by contract; `.to(device)` excepted), and the safe
  consumer sites (print args, expression statements, method receivers)
  compile through the tracking path. Call ARGS deliberately stay
  untracked: a callee may ESCAPE a param (member-assign stores the raw
  pointer with no retain — `self.buf = t`), so freeing a tracked owning
  arg at statement end would be a use-after-free on the stored field;
  owning call-result args strand (status quo) until member stores
  retain.
- Gate: `inference_loop_leak_gate.rs` — exit live_blocks must be
  IDENTICAL across iteration counts (3 vs 9) for the method-call and
  method-temp loops, pinned to the exact weight count; plus a borrow
  fixture asserting the veto (identical per-iteration sums prove the
  rebound weight survived).

### Fixed — #415 follow-ups (previously unreleased-note gaps)

- `TapeOp::Reshape`: reshape views carried tape_id 0 and silently
  disconnected the tape at every reshape; the masked-SDPA tape parity
  gate had passed vacuously since inception. Backward reshapes the
  gradient back via a zero-copy view.
- `csla_parity_fused_lmce_gpu` ground truth re-derived: the pinned
  composite loss predated P0's `--seed` fix (which changed deterministic
  init); the fused kernel bit-matches a freshly derived composite twin —
  the "parity failure" was a stale pin, not a kernel bug.
- CSHA g14_b/g14_c gates set `rope_q=false` — checkpoint(full)+rope_q now
  refuses up front by design.
- `cuda_clock` checkout_event calls `ensure_context()` before
  `cuEventCreate` (silent dead-handle class).

### Fixed — tape-AD on GPU computed ZERO gradients (silent), plus the per-step tensor leaks

- **GPU ops now record on the autodiff tape.** Every GPU arm in the tensor
  FFIs (add/sub/mul/div/neg/scalar ops/matmul, exp/log/sqrt/abs/clamp/
  relu/gelu/silu/sigmoid/tanh, sin/cos, stack/slice/cat, gather/softmax,
  embedding_lookup/bias_add) returned before the CPU path's `maybe_record`,
  so a train block WITHOUT `--source-ad` on GPU taped almost nothing,
  every parameter gradient fell back to `zeros_like`, and "training"
  moved on weight decay alone (~0.02%/step loss drift). Records now mirror
  the CPU conventions exactly (saved-ref bumps ordered before relinquish
  frees); CPU-redirect wrappers (stack/cat/gather/softmax/log_softmax)
  pause the tape across the redirect (`TapePause`) and record the real op
  against the caller's tensors; FBIP in-place arms are defensively gated
  on `!is_recording()`. The tape backward is now device-safe: the
  SumReduce/MeanReduce global arms keep grads on the graph's device,
  `broadcast_grad_along_dim`/`scatter_grad_to_argmax` bounce through the
  CPU, and the raw activation-backward helpers reconcile mixed
  grad/saved devices.
- **Disconnection backstop:** `nsl_tape_backward` (train entry) now aborts
  loudly when the backward produces a gradient for NONE of the parameters
  — the silent zero-grad failure mode is a hard error with an actionable
  message (`NSL_TAPE_ALLOW_DISCONNECTED=1` escape hatch; grad blocks keep
  the permissive zeros answer for unused inputs).
- **Explicit-`return` paths now free function locals.** Only the implicit
  fall-off-the-end path swept I64 locals; every `let`-bound tensor local
  in a function ending with `return <expr>` leaked its final reference
  (`mse_loss`'s `diff`, residual-block `f`s, loop-carried `h`s) — the
  tape-mode per-step leak AND a large slice of the `@no_grad` inference
  leak. The sweep is shared (`emit_return_local_sweep`) and
  `state.param_symbols` is now populated on EVERY function-body compile
  path (model methods/constructors/export methods, dtype methods, agent
  fns, train callbacks) so the sweep can never free caller-owned params.
- **`return <call>(...)` no longer double-owns the result.** Calls to
  compiled NSL functions (and fresh-tensor builtins, now incl. mean/sum)
  contractually return an owning reference; the Return arm's conservative
  Unknown-retain added a second owner and stranded one tensor per call.
- **Tape-path train epilogue owns the loss** (both the pure-tape arm and
  the source-AD extraction-failure fallback), closing the last +1/step.
- Gates: `mse_leak_gate.rs` now runs BOTH paths — source-AD and tape-AD
  exact-flat live_blocks (post optimizer-state warmup) plus a loss-descent
  assert on the tape path (live_blocks stays plausible under zero-grad;
  only the descent assert catches it). Debug tooling added (permanent,
  env-gated): `NSL_DEBUG_MEM_TRACE=1` traces allocator handout/free with
  pointer+context, tensor frees with pre-decrement refcount, and tape
  record/release events.

### Added — Fusion queue: multi-tensor fused AdamW + GPU-native cross-entropy backward

- **Multi-tensor fused AdamW** (always on for eligible configs;
  `NSL_FASE_MULTI_STEP=0` kill-switch): the non-clip FASE-Deferred
  optimizer loop (one fused-step launch + one zero per param per step)
  collapses into ONE `nsl_fase_fused_adamw_step_multi` pointer-table
  launch over the param/m/v/m_partial lists — grid.y = param index,
  per-param length table with early-exit blocks, tables staged through
  persistent pinned memory on the compute stream. The kernel body is
  byte-for-byte the single-step kernel's sequence, so results are
  **BIT-IDENTICAL** to N launches; the shared tail's m_partial zeroing is
  folded into the same kernel. CPU/non-uniform tensors fall back
  per-param (also bit-identical). Admission mirrors the fused-step
  conditions exactly and excludes ZeRO / offload / bf16-sr / CPDT
  precision; the fused-step counter keeps per-param semantics. Gates:
  GPU and CPU bit-identity (multi on vs off) + marker/kill-switch
  anti-vacuity.
- **GPU-native cross-entropy backward** (default on;
  `NSL_GPU_CE_BACKWARD=0` restores the bounce): `nsl_cross_entropy_backward`
  no longer copies logits `[N, vocab]`, targets, and grad_output to the
  host — the whole softmax-minus-onehot gradient runs on device (existing
  softmax kernel + a valid-count reduction + an in-place finish pass),
  with grad_output read in-kernel when device-resident. Kills CE
  training's largest per-step PCIe round trip and removes one of the two
  taints keeping loss-epilogue cuda-graph regions eager (embedding
  backward still bounces). Target semantics mirror the CPU arm exactly
  (truncate-then-compare, `target < 0` rows zeroed, `max(valid, 1)`
  denominator); f32 kernel vs the old f64 host math differs ~1e-6
  relative. Gates: GPU-vs-CPU elementwise parity, ignored-row exact-zero,
  device-scalar grad_output; ptxas covers all new kernels.

### Deferred — wgrad GEMM-accumulate into CSLA buffers (fusion item 3, design banked)
Writing weight-gradient GEMMs straight into `m_partial` with
`alpha=accum_scale, beta=1` (eliminating the fresh dW alloc + the
separate axpy launch) requires a wengert-lowering pattern pass that fuses
across Transpose -> Matmul -> reduce_to_shape AND re-plumbs the FASE
hook's ownership bookkeeping (owned_values / explicit_freed /
still_needed) — grad-integrity-sensitive surgery deliberately not
attempted as a session add-on. Admission notes for the future slice:
identity-reduce shapes only, non-still_needed adjoints only, Deferred
hook path only; two-phase clip composes (it scales m_partial after
accumulation).

### Added — Muon perf campaign: batched Newton-Schulz + resident momentum + internal profiler

- **`--muon-batch-ns`** (default off, GPU-only): every Muon-routed rank-2
  param is updated by ONE batched runtime call per optimizer step
  (`nsl_muon_step_batch`): momentum update + Frobenius normalization +
  quintic Newton-Schulz + parameter update, shape-grouped and executed as
  strided-batched tensor-core **TF32** GEMMs (`cublasGemmStridedBatchedEx`,
  `CUBLAS_COMPUTE_32F_FAST_TF32`) over persistent per-shape workspaces with
  device pointer-table addressing. Design wins folded in: no physical
  transposes anywhere (tall matrices transpose on the fly in the pack/update
  kernels; the Gram product runs as a `transa=T` GEMM), the polynomial
  combine folds `ns_a*x + b@x` into one GEMM via a diagonal add
  (`(ns_a*I + B) @ x`), and every square NS intermediate is symmetric so the
  row/column-major operand swaps cancel. Workspace bounded by
  `NSL_MUON_BATCH_MB` (default 256 MiB) with budget chunking;
  `NSL_MUON_BATCH_TF32=0` forces strict FP32. Measured on the coder500m
  Muon-routed load (168 matrices, ns=5): **341 ms vs 795 ms sequential
  (2.3x)**; ~2,520 GEMM launches/step collapse to ~110 batched calls.
  The AdamW-routed arm keeps the stdlib path bit-for-bit; the momentum
  update is bit-exact vs the stdlib arm; NS output is tolerance-equivalent
  (TF32 + batched reduction order), pinned by differential gates. Refuses
  `--layerwise-accum`, `--optim-state-offload`, `--zero-stage`,
  `--muon-state-dtype bf16`, and non-muon optimizers loudly.
- **`--muon-resident-momentum`** (default off): under `--optim-state-offload`,
  Muon-routed rank-2 params keep their first moment DEVICE-resident and skip
  the per-step PCIe stage-in/writeback envelope entirely (plus the pointless
  v round-trip on that route). AdamW state (embeddings/head/vectors) stays
  offloaded. Bit-identical to plain offload (gated). This removes the mixed
  recipe's only per-step optimizer-state round trip — the 500M campaign's
  main non-GEMM pathology.
- **`NSL_MUON_PROF`** internal profiler (perf item 2): 13 timestamp regions
  across the whole Muon path (momentum stage-in/update, Frobenius reduce,
  normalize scale, entry/exit transpose, Gram/Gram-square/poly GEMMs, param
  update, momentum writeback) with two modes — `1` synced attribution, `2`
  enqueue-only (the difference isolates launch/dispatch overhead from
  execution). `[muon-prof]` table at exit; `nsl_muon_prof_report()` on
  demand. First profile findings: the sequential NS load is ~0.8 s/step at
  500M scale (NOT the 66.8 s/micro pathology — that was staging/offload);
  within NS, the Gram GEMM dominates (~40%) and the per-matrix
  single-block Frobenius reduction costs ~1 ms/call.
- Batched-path building blocks: 5 new PTX kernels (ptxas-gated) —
  pointer-table momentum update, deterministic per-matrix sum-square
  reduction (same order as the sequential stats kernel — bit-identical
  sums), transposing pack/scale, diagonal-folded polynomial combine, fused
  transposing unpack + parameter update; `sgemm_strided_batched_raw` cuBLAS
  wrapper with per-call compute-type control.
- Fixed en route: `nsl_cpdt_allgather_add` relinquish-flag orientation (see
  the FBIP leak fix); NOTE `[nsl-matmul]`'s "TF32 (default)" banner —
  `CUBLAS_DEFAULT_MATH` does NOT enable TF32 tensor cores for f32 GEMMs on
  cuBLAS >= 11, so the runtime's standard matmuls run FP32 CUDA cores; the
  batch engine requests TF32 per call. A global re-evaluation is a separate
  (numerics-affecting) decision, deliberately not made here.
- Deferred (documented): CSLA x batch composition (per-layer groups /
  deferred-epilogue batching), WGGO-driven workspace budget, cublasLt
  autotuned plans, batched SYRK (no strided-batched SYRK exists in cuBLAS —
  symmetry is exploited for mapping, not half-flop Grams).

### Added — P5 item 19: opportunistic per-region CUDA graph capture (`--cuda-graphs`)

- Every source-AD Wengert lowering (forward CCR slice, CSLA backward layer
  range, recompute segment) becomes a capture REGION; weight-stream transfers
  and optimizer updates stay outside. Per (region, accumulation-phase) state
  machine: record the full pseudo-op sequence (kernel function/dims/param
  bytes via `cuFuncGetParamInfo`, cuBLAS sgemm, memset, HtoD/DtoD) until two
  passes prove it static, capture it as a CUDA graph on the compute stream
  (the capture step still executes via instantiate+launch), then replay:
  verify-and-skip each issued op, one `cuGraphLaunch` per region end.
- Self-healing everywhere: mismatches eager-repair the verified prefix from
  the stored records (bit-identical by construction); sync readbacks and
  transfer ops taint regions to permanent-eager; deferred frees queue
  in-region and record their events only after the region's work is on the
  stream; the allocator pool drain taints first so a pending graph never
  references unmapped memory. HtoD uploads flow through graph-owned pinned
  staging buffers refreshed per step (token-id batches stream fresh data
  through a stable memcpy node).
- Refuses tape AD, ZeRO, `@pipeline`, `--cuda-sync`, `--profile-kernels`,
  the legacy NULL stream. `NSL_CUDA_GRAPHS=0` kill-switch,
  `NSL_CUDA_GRAPH_LOG=1` decision tracing. Gates: capture+replay
  bit-identical to eager (plain + CSLA shapes, anti-vacuity counters),
  self-heal bit-identical on a fixture with real per-step host bounces.

### Fixed — root causes surfaced by the capture digests

- **`model.to(device)` missed every field declared after an inline model
  array** (`[Blk; N]` spans N slots but the walker got `fields.len()`):
  such params silently stayed CPU-resident — the P4 "`ones([64])` norm param
  still on CPU" gotcha — costing a hidden per-step PCIe upload + host
  bounces through `reconcile_device` on every use. Now passes
  `total_size / 8`, the convention `nsl_collect_model_params` always used.
- **`nsl_l1_backward` crashed on GPU tensors**: it captured the dtype before
  the CPU transfer widens f32→f64, then read f64 buffers through
  `data_f32()`.
- Known issue (pre-existing, tracked separately): `mse_loss` leaks its
  `(pred-target)` intermediate every step on the source-AD path; the leak
  set is expression-structure-dependent (an identity-mul variant leaks six
  blocks/step, l1 is leak-free). Documented in the cuda-graph gate fixture.

### Added — P5 item 20: more fused backward regions

- **Fused RMSNorm gamma backward** under `--fuse-rmsnorm-backward` (which
  previously fused only dx): the 7-op decomposition with three
  `[rows, cols]` temporaries becomes one `nsl_rmsnorm_dgamma_backward` op —
  a per-row 1/rms kernel into a tiny `[rows]` scratch, then a per-column
  sequential row-loop kernel. Fixed summation order → bit-deterministic
  run-to-run; f64 CPU reference. Gates: fused == decomposition == tape-AD on
  trained w AND gamma (CPU), fused-GPU == CPU tape reference +
  bit-deterministic reruns.
- **SwiGLU gate-backward peephole** (always-on, BIT-EXACT): for
  `f = silu(g) * u` the adjoint pair Mul(dy, up) → silu_backward fuses into
  one `swiglu_gate_backward` launch when the product has exactly one reader;
  the kernel rounds `t = mul.rn(dy, up)` exactly like the standalone Mul then
  runs the identical silu-backward sequence — proven `to_bits`-equal on CPU
  f64 and GPU f32. Mismatched shapes fall back to the decomposed pair.

### Added — P5 item 21: non-uniform checkpoint partition DP (`--checkpoint-stride dp`)

- A Pareto-frontier dynamic program over NON-uniform block-anchor partitions:
  per-block escape/force-saved/recompute bytes come from the exact stride-1
  plan; cost is the GpuSpec-calibrated recompute estimate (launches ×
  worst-case launch overhead floored by 2×bytes/HBM bandwidth, × the CSLA
  window), lowered when `--fuse-rmsnorm-backward` is active and credited
  10 µs/segment under `--stream-prefetch`. Cheapest partition whose projected
  peak fits `--checkpoint-budget-mib` wins (min-peak otherwise); the DP's
  projection is re-verified against the true plan before committing, with
  fallback to the uniform `auto` search. `plan_with_kept_anchors` generalizes
  the plan machinery to arbitrary anchor subsets — bit-exact for the same
  reason any stride is. Gates: DP units (incl. dropping a single huge
  boundary non-uniformly — inexpressible as a uniform stride), dp/dp+budget
  e2e bit-exact vs stride 1, GPU peak-not-worse.

### Changed — P4 item 16: dtype ABI migration (both tag collisions removed)

- **`DTYPE_I32 = 9`**: i32 token tensors carry their own canonical tag.
  Tag 4 means `DTYPE_INT8` alone, with its true 1-byte width in
  `dtype_element_size`. Every producer/consumer migrated in lockstep:
  DataLoader batches, the CPU tensor factory, the `*i32` runtime readers,
  the GPU i32-index kernel dispatch (embedding fwd/bwd, gather), the
  fused-CE label decoder, the elementwise promote paths, and the
  collective byte-width table (which gains `I32=4`, closing a latent
  1-byte mis-size for i32 tensors through collectives). No on-disk format
  stores numeric tags, so the migration is disk-safe.
- **The C API speaks the canonical tag space verbatim** (`NslTensorDesc.
  dtype`: 0=f64, 1=f32, 2=f16, 3=bf16, 4=int8, ..., 9=i32). The
  historical inverted 0=f32/1=f64 convention is GONE; `capi_dtype_to_nsl`
  / `nsl_dtype_to_capi` remain only as validating identity chokepoints
  that abort on unknown tags instead of the old silent fall-back-to-f64
  (which mislabeled the never-mapped C-API int64/uint8 slots). Updated in
  lockstep: calibration-wrapper codegen immediates, the ONNX-RT element
  map (int64/uint8 now refused — they never reached a real compute path),
  the dispatch element-size table, the generated C header, and the Python
  ctypes mirrors. **Breaking for external C/ctypes callers** that baked
  the old convention; the `dtype_abi_lock` golden test pins the new one.

### Added — P4 item 17: SR-BF16 authoritative weights (`--param-dtype bf16-sr`)

- Every STREAMED parameter's authoritative copy is a device-resident
  BF16 buffer — 2 bytes/param, **no FP32 master copy, no host mirror**.
  Rides the weight-stream residency schedule: upload = device-side
  bf16→f32 widen into a transient working view (no PCIe), evict = free
  (no writeback — the fused SR optimizer step IS the persistence),
  teardown re-materializes plain f32 tensors for model_save/eval.
  Un-streamed params (view-rooted/tied — the set ZeRO-3 keeps
  Replicated) keep f32 authority through the plain fused step.
- The update is a fused AdamW PTX kernel with the f32 kernel's exact
  rounding sequence plus a stochastic-rounding tail: `splitmix64(seed ^
  step·SALT, param_idx≪40 + elem)` → 16-bit dither added to the f32
  result bits before bf16 truncation. **Compiler-owned counters**: the
  stream is a pure function of (`--seed`, step, param, element) —
  deterministic across reruns, ranks, and launch order. Explicit edge
  policy: rounding-induced overflow saturates to ±max-normal; arithmetic
  Inf propagates; NaN forced to quiet NaN; underflow gradual.
- FP32 gradients/reductions everywhere (m/v/m_partial stay f32).
  Refusals: requires `--weight-stream` + the FASE-Deferred fused
  AdamW/Adam shape; refuses Muon, ZeRO, offload, reduced-precision
  moments, WGGO per-layer overrides, `--training-reference`.
- Gates: exhaustive-dither exact-unbiasedness unit tests; GPU rounding
  tail bit-identical to the CPU reference over 65k adversarial values;
  e2e training with bit-identical same-seed reruns, seed-sensitive
  streams, and per-step tracking of the f32 baseline.

### Added — P4 item 18 rung 2: compressed Muon state (`--muon-state-dtype bf16`)

- Ladder order: f32 (default) → **bf16 momentum + f32 working buffer**
  (this rung) → blockwise 8-bit → 4-bit structural (later rungs refuse
  loudly with the ladder order in the message).
- `bf16` halves Muon first-moment memory: m allocates BF16; each CSLA
  group update dequants to f32, runs the unchanged stdlib `muon_step`,
  and quant-stores back with counter-based SR (salted separately from
  the item-17 weight stream). SR — not RTE — because the momentum EMA's
  `(1−β)·g` increment routinely falls below a bf16 ulp. `v` stays f32.
  Muon-only; requires `--layerwise-accum`; refuses ZeRO and offload.

### Added — P3: ZeRO-3 tensor-granular parameter sharding (items 12-14)

- **`--zero-stage 3` is lowered** on the layerwise residency schedule
  (requires `--layerwise-accum --weight-stream --checkpoint-blocks
  --source-ad`; anything else refuses with the flag list). Each parameter
  tensor is OWNED by one rank (the stage-1/2 byte-balanced partition);
  at rest the owner keeps it device-resident and every other rank holds
  NOTHING — per-rank at-rest parameter memory is ~1/ws. This is the
  per-parameter FSDP granularity (owner = singleton group per tensor):
  it reuses the existing owner maps, composes with Muon (whole matrices
  stay whole), and needs no gather in the optimizer. Elementwise 1/ws
  sharding via `all_gather` is the documented follow-up.
- **Item 12 — `ParameterResidency`** (`Replicated | ShardedResident |
  GatheredTemporary | Evicted`) tracked per replica in the runtime.
  Callbacks that touch θ ride the existing residency bracket
  (`upload_all`/`reevict_all` redirect to gather/release), `model_save`
  reads full replicas via the teardown restore, and tied/view-rooted
  params stay `Replicated` through the same view-rooted exclusion the
  weight streamer uses (updated identically on every rank from
  all-reduced gradients).
- **Item 13 — JIT gather per layer**: the weight-stream upload/evict
  sites (per-segment forward brackets, window range heads, packs,
  prefetch, async evict) redirect to a collective broadcast-fill /
  free-release backend when zero3 is active — GPU-only, no host mirrors
  involved. Registration evicts non-owner replicas on the first window.
- **Item 14 — comm ordering from source-AD readiness**: each layer
  group's gradient slots all-reduce (sum ÷ ws, the stage-1/2 averaging
  convention) at its group update — the exact point the layerwise
  schedule knows that layer's backward completed — then the owner
  updates and non-owners release; the next range head (or the prefetch
  edge, when `--stream-prefetch` is on) gathers the following layer.
  Under the sim/sim-gpu backends collectives are synchronous, so the
  ordering is validated bit-exactly; true comm/compute overlap on a
  dedicated stream is NCCL-gated follow-up (not reachable on a 1-GPU
  box).
- Constraints (loud refusals): stage 3 × `--optim-state-offload`,
  stage 3 × reduced-precision moments, stage ≥ 4. Optimizer state stays
  REPLICATED in v1 (the enable note says so).
- Gates (`zero3_gate.rs`): 2-rank sim-gpu training is BIT-IDENTICAL to
  the single-rank baseline (rank-0 loss stream + saved model bytes),
  with gather/release counters asserted non-vacuous; the same parity
  holds for Muon × zero3 × arena/prefetch/async-writeback × a callback
  that reads a sharded param mid-training; refusal coverage for the
  unsupported combos.

### Changed — P1: Muon validation + performance (items 5-11)

- **Parameter-ROLE routing replaces the name-substring exclusion list**
  (item 6). Mixed Muon/AdamW routing is now decided per parameter:
  explicit `@param_role("embedding"|"head"|"hidden")` field decorator
  (invalid values are compile errors) > structural inference (the table
  argument of `embedding_lookup(self.field, ids)` is role `embedding`;
  weight-tied heads are the same tensor) > declared rank != 2 (`vector`)
  > default `hidden`. A hidden weight named `embed_proj` now correctly
  takes Muon; an embedding named `tok_table` now correctly takes AdamW —
  both silently misrouted before. The routing table prints every param's
  role + provenance; untied-head models get a loud annotate-me note.
- **`adamw_lr` knob on `Muon(...)`** (item 5 prerequisite): the AdamW arm
  (embeddings/head/vectors) gets its own learning rate, threaded as a
  fixed ratio of `lr` so schedulers modulate both arms proportionally.
  Unset, it follows `lr` exactly (bit-exact with the old single-lr step).
- **`ns_steps` is validated** (item 7): floats, zero, and negatives are
  compile errors instead of silently degrading the NS iteration.
- **`.item()` removed from the Muon step / planned NS primitive**
  (items 8+10): `muon_step` now calls `muon_orthogonalize_fast`, a
  runtime primitive (`nsl_tensor_muon_orthogonalize`) that runs the
  quintic Newton-Schulz chain from Rust — on GPU the Frobenius
  pre-normalization is computed and consumed entirely on-device (stats
  kernel into a persistent 16-byte scratch + a scale kernel that reads
  it; no DtoH sync per rank-2 param per step), tall/wide handled by
  materialized transposes at entry/exit, intermediates recycled through
  the caching allocator. The NSL-level `muon_orthogonalize` stays in
  `nsl.optim.muon` as the pinned reference (CPU f64 gate: sum-sq diff
  < 1e-18; GPU gate bounds the f32 norm-path drift). Deferred:
  batched small-matrix NS, stochastic-rounded direct updates.
- **v (AdamW second moment) allocates only where the AdamW arm reads it**
  (item 9): Muon-routed params carry a null slot (exactly the runtime
  condition `muon_step` routes on); ZeRO owner-gating composes. Under
  `--optim-state-offload` / CPDT moment-precision plans v stays fully
  allocated (their stage-in envelopes touch both moments) with a loud
  note; reduced-precision moments + muon refuse outright.
- **Muon composes with `--layerwise-accum`** (item 11, the
  separate-accumulator full-quality mode): the window backward
  accumulates RAW per-layer gradient sums (Deferred-shaped plan,
  accum_scale forced to 1.0 — the FullBuffer convention) and the
  per-layer group updates dispatch the stdlib `muon_step` (bias
  correction from the micro-batch counter, as the non-layerwise path
  does). Gated BIT-IDENTICAL to the non-layerwise Muon run on CPU and
  GPU (loss stream + saved model bytes). The exact one-buffer
  "classical Muon" accumulation is intentionally NOT this mode and
  would ship separately named.

### Known issues — found by the P1 Muon campaign (not fixed here)

- The plain-call inference path (`m.forward(...)` from functions,
  callbacks, or top-level code — NOT the source-AD-extracted train step)
  LEAKS its device intermediates: a [2, 1024] coder50m forward leaves
  ~2.3 GB of un-freed GPU tensors per call (the non-flash GQA attention
  scores dominate), and `@no_grad` does not change it — the temps are
  simply never freed. Repeated in-training validation forwards OOM'd
  after 3 calls until the campaign switched to [2, 256] eval windows.
  Needs a bounded-lifetime fix for function-body tensor temps.
- Returning from a user fn whose body mixes tensor ops with several
  scalar (`.item()`-derived) `let` locals SIGSEGVs the emitted program
  shortly after the call returns (JIT frames, masked as exit 1 by
  `execute_temp_build`; repro: `shape_probe`'s original fn-based form).
- A float `let` after tensor `let`s inside a top-level `while` body
  fails Cranelift verification: "declared type of variable varN doesn't
  match type of value vM". Both worked around by inlining + avoiding
  scalar locals in `models/benchmarks/muon50m/shape_probe.nsl`.
- Subscripting a model fixed-array field from a CALLBACK body
  (`m.blocks[0].w_up` in `on_step`) compiles but SIGSEGVs the emitted
  program; the `for pb in m.blocks:` iteration form works (used by the
  zero3 callback gate).

### Added — P5: full-precision mixed Muon/AdamW optimizer

- `Muon(...)` in train blocks is now the REAL Muon (Jordan et al., 2024):
  rank-2 hidden weight matrices take the quintic Newton-Schulz
  orthogonalized-momentum update (`ns_steps`, default 5) scaled by
  `sqrt(max(1, rows/cols))`; embeddings and the LM head (name-routed:
  embed/lm_head/unembed/wte/wpe/vocab in the param path) plus all
  non-rank-2 params (biases, norms) take a standard AdamW step
  (`beta1`/`beta2`/`eps`/`weight_decay`). Previously `muon` was plain
  momentum-SGD with decoupled weight decay — out of spec.
- Muon is now two-state (m = Muon momentum / AdamW first moment; v = AdamW
  second moment). The routing table prints at train start (`[muon]` line).
- The Newton-Schulz chain is pinned against an independent f64 reference
  (wide + tall branches), the all-AdamW-routed case is bit-exact vs
  `AdamW(...)`, and GPU training is gated (device f32 end-to-end).
- `@pipeline` train blocks refuse `Muon` loudly (not wired there yet).

### Fixed — ZeRO deferred hardening (post-#404)

- Stage-2 reduce_scatter groups now split into padded sub-groups capped at
  `NSL_ZERO_BUCKET_MB` (fractional MB accepted), and tensors larger than
  cap/ws are chunked by byte range — a 1B-scale gradient group (or a single
  64MB embedding grad) no longer hits the 64MB CPU-shm slot wall.
- Cross-rank partition-plan verification: every rank hashes
  (stage, ws, param sizes, owners) and compares against rank 0 at partition
  time — a divergent plan refuses loudly instead of training a torn model.
- Non-owner `m_partial` cleanup uses a true zero-fill instead of `x *= 0.0`
  (which preserved NaN/Inf forever after a diverging window).
- `cuda_device_name()` probes the device the process will actually bind
  (NSL_CUDA_DEVICE / spawner striping), not unconditionally ordinal 0.
- `NcclBackend` tracks `ncclCommAbort` and skips `ncclCommDestroy` in Drop
  after an abort (destroying an aborted communicator is undefined).

### Added — Architecture hardening (stable/experimental boundaries, ABI versioning)

- `STATUS.md` — single source of truth tiering every subsystem as Stable / Beta /
  Experimental, with the per-tier test expectations.
- `docs/hardware/` — tested-on matrix plus `cuda_status.md` and `fpga_status.md`,
  making GPU/FPGA claims traceable (Validated / Built / Analysis-only) to actual
  evidence instead of aspirational support.
- `docs/abi/README.md` — runtime C-ABI contract: versioning policy and the
  per-symbol FFI safety checklist.
- Runtime C-ABI versioning: `nsl_runtime::c_api::{NSL_ABI_VERSION_MAJOR,
  NSL_ABI_VERSION_MINOR}` constants and the `nsl_abi_version()` exported fn
  (packed `(major<<16)|minor`). Generated C headers now emit matching
  `NSL_ABI_VERSION_*` macros (pinned to the runtime constants) and the
  `nsl_abi_version()` prototype, so hosts can detect runtime/header skew.
- Golden ABI-layout test pinning `NslTensorDesc` to 48 bytes / 8-byte align /
  fixed field offsets (`nsl_tensor_desc_abi_layout_is_pinned`).
- `CONTRIBUTING.md` — "Review gates" section formalizing the stable-vs-
  experimental, FFI, hardware-claim, config-sprawl, and clippy-suppression
  boundaries; test requirements split into required-on-PR / nightly / research.

### Added — Architecture hardening (cont.: config decomposition, PTX metadata, FFI/state)

- `CompileOptions` decomposition continued: extracted the `csha_*` cluster into
  `CshaOptions` and the `cpdt_*` cluster into `CpdtOptions` (joining the existing
  `WcetOptions`/`ZkOptions`/`WggoOptions`). `CompileOptions.{csha,cpdt}` replace
  six flat fields; behavior-preserving. The `calibration_*` and dev-tools
  clusters are left flat deliberately (already prefix-cohesive; their field names
  collide with identically-named fields on other structs, so a blind rename is
  unsafe) — rationale recorded in `STATUS.md`.
- `nsl_codegen::ptx_metadata` — static, dependency-free, GPU-free parser that
  extracts per-kernel declared register counts, static shared-memory bytes, and
  target SM from synthesized PTX text, plus a report formatter that flags kernels
  exceeding the 255-register per-thread cap. Covered by unit tests + a public-API
  integration test.
- `nsl ptx-metadata <file.ptx>` — CLI subcommand surfacing the per-kernel PTX
  resource report (registers / shared memory / target SM). Pure text analysis;
  no CUDA toolkit required.
- FFI safety tests: `grad_context::abi_layout_tests::magic_is_first_field_for_ffi_validation`
  pins `GradContext.magic` at offset 0 (the only field the C side reads through
  the opaque handle); `c_header_abi_version_matches_runtime_constants` asserts the
  generated header's `NSL_ABI_VERSION_*` macros equal the live runtime constants
  (catches codegen/runtime version skew; no C compiler needed).
- Experimental subsystem feature flags: `experimental-wrga` / `experimental-cpdt`
  (both in `default`) gate the WRGA and CPDT pass entry points in `stmt.rs`, so a
  `--no-default-features` build can turn those research passes into no-ops. The
  default build is byte-identical (gates compile out). Phase-1 behavioral gating;
  see `docs/architecture/compiler-state.md`.
- `docs/architecture/compiler-state.md` — audit of compiler/runtime thread-local
  globals (classified test-only / FFI-OK / migrate), establishing `Compiler` as
  the session object and a staged plan to retire the WRGA build-side globals into
  explicit context (review item: "replace hidden thread-local state").
- `docs/hardware/cuda_status.md` — "Golden CPU-reference test coverage" section
  making the GPU-vs-CPU-oracle validation pattern traceable to specific tests.

### Changed

- `SECURITY.md` — corrected the supported-versions table (now `main` + 0.9.x)
  and enumerated the highest-risk areas (C ABI, dlopen, model/weight parsers,
  path handling, CUDA launch, generated-code execution, compiler DoS).
- `README.md` — replaced the inaccurate "no runtime" claim with "no Python
  interpreter / no GIL; programs link a small native runtime", and added a
  pre-1.0 maturity pointer to `STATUS.md`.

### Fixed

- `test-onnx-rt` CI job: install the `rustfmt` toolchain component, which
  `bindgen` requires when generating bindings in `tools/verify-ort-vendoring.sh`
  (the pinned 1.95.0 toolchain ships without it, failing the job at the
  vendoring step independently of ONNX itself).

### Added — CSHA Tier B.2 backward Phase 2 (foundation + dQ-kernel emitter)

- `flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass` — D pre-pass kernel emitter (row-per-lane schedule: 32 lanes × 1 row each, sequential over `head_dim`; no inter-lane reduction; no SMEM; sm_80+; computes `D[b,h,q] = rowsum(dO * O)`). Spec §3.3's original butterfly-reduction schedule was replaced after the first GPU launch revealed a row/col conflation bug — both schedules are HBM-bandwidth-bound at canonical sizes, and the row-per-lane schedule avoids the bug class entirely.
- `flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel` — dQ-kernel emitter (~700 LOC), producer-consumer warp specialization, register-resident dQ accumulator across kv-inner loop, **no atomicAdd**. Inner-loop MAC chain: S = QK^T → P recompute → dP = dO·V^T → dS = P·(dP - D) → col-major K re-stage band (Path A) → dQ_acc += dS @ K.
- `flash_attention_v2::tier_b2::register_budget_backward` — `BackwardKernel` enum (DPrePass | DQ | DKDV), `count_registers_backward`, `predict_fallback` planner helper covering BOTH SMEM-pressure (hd=128) and register-pressure (hd=256) cases.
- `flash_attention_v2::smem_layout::tier_b2_dq_*_offset` accessors including the new `tier_b2_dq_k_colmajor_offset` Path A re-stage band, plus `tier_b2_effective_bq`/`tier_b2_effective_bkv` per-hd fallback schedule and `tier_b2_dkdv_*` stubs for Phase 3.
- `matmul_mma::emit_load_b_fragment_smem` parameter renamed `row_stride_bytes` → `col_stride_bytes` (was misnamed for B.1's actual use as the column stride between adjacent n-axis positions). The Task 2 `load_transposed: bool` extension was reverted after V-B2-5 verification found it architecturally unsound (commit `275d849d`).
- `nsl-test` crate (new workspace member) with `nsl_test::diagnostic_mode::{DSource, compute_d_for_test}` — permanent test utility for backward-kernel localizability (swap CPU-D in for B.2-pre-pass-D to bisect failures). Phase 3 dK/dV-kernel and future-milestone backward work inherit the primitive.
- `crates/nsl-codegen/tests/tier_b2_no_atomic_in_dq.rs` — Rust-level PTX-parse invariant test (CPU-only, runs every commit; asserts dQ-kernel emits zero `atom.*` instructions per spec §7.2).
- `crates/nsl-codegen/tests/tier_b2_dq_k_colmajor_lane_mapping.rs` — Spec §5.5 institutional pin: lane-mapping byte-pattern test for the col-major K re-stage band.
- `crates/nsl-codegen/tests/tier_b2_dq_kernel_cpu_reference.rs` — Layer-1 dQ tests (Test 1: D pre-pass standalone; Test 2: dQ smoke at canonical; Test 3: dQ head_dim sweep across {32, 64, 128}). All `#[ignore]` + `feature="cuda"` — manual GPU validation gates Phase 2 closure.
- `crates/nsl-test/tests/diagnostic_mode_localizes_d_bug.rs` — Spec §7.3 sharpened FAIL→PASS exit criterion: injects corrupted D and proves the swap localizes correctly.
- Phase 1 `synthesize_tier_b2_backward → Err(NotImplemented)` stub removed; selector wrapper now routes through the real emitter.
- `crates/nsl-codegen/tests/tier_b2_ascii_only_ptx.rs` — ASCII-only invariant guardrail: every byte of emitted PTX must be 7-bit ASCII. Catches Unicode characters in `//` comments that cause cudarc's ptxas JIT to abort with `CUDA_ERROR_INVALID_PTX`. First-incident origin: 2026-05-20 D pre-pass launch failed because of em-dash + multiplication-sign characters in section comments.

### Validated on GPU (RTX 5070 Ti sm_120, 2026-05-20)

- D pre-pass GPU validation: **max_abs = 0.0** (bit-exact vs CPU reference) at all 4 tested configurations: canonical `(b=1, h=1, s=32, hd=32)` plus sweep cases `(1,1,64,32)` / `(1,2,96,64)` / `(2,1,128,128)`. Tolerances 5e-3 / 2e-2 / 4e-2 not even relevant — match is exact.
- `run_d_prepass_on_gpu` cudarc launcher wired (via `nsl_kernel_launch` + `nsl_test_cuda_*` primitives).
- `run_b1_forward_for_test` + `run_dq_kernel_on_gpu` remain `unimplemented!()`, **explicitly gated on Phase 2.5**. The dQ-kernel emitter is currently a structural scaffold (sections + register decls + MMA chain + labels — verified by ~20 ptxas/structural tests) but is **not data-mobile**: cp.async loads, HBM address derivation, dS SMEM scatter, col-major K re-stage scatter, tile_skip predicate computation, MMA fragment row/col setup, and loop back-edges all ship as PTX comments rather than emitted instructions. A launched dQ-kernel would read uninitialized SMEM. Phase 2.5 fills the data-mobility gap and is the gate to dQ GPU validation (Tests 2 + 3 in `tier_b2_dq_kernel_cpu_reference.rs`).

### Changed
- Moved the root-level research PDFs into `docs/research/` so research artifacts live with the rest of the repository's research material.
- Refreshed `README.md` to reflect the current documentation layout and the current local validation snapshot instead of stale passing-test counts.
- Refreshed `SPECIFICATION.md` to match the workspace version in `Cargo.toml` (`0.9.0`) and point readers at the current docs/research layout and validation status.

## [0.9.1] - 2026-03-26

### M41b: NVLink/RDMA/TCP KV Transfer Backends
- **TcpBackend**: TCP socket-based KV transfer for multi-node disaggregated inference (per-rank listener, retry logic, Nagle disabled)
- **NvlinkBackend**: CUDA IPC GPU-direct transfer for same-node multi-GPU (cuIpcGetMemHandle/cuIpcOpenMemHandle, falls back to staged CPU transfer)
- **RdmaBackend**: RDMA verbs-based zero-copy transport for HPC clusters (ibverbs memory registration, InfiniBand/RoCE hardware probe, TCP fallback)
- **Auto-detection**: `auto_select_backend()` probes NVLink > RDMA > TCP > SharedMem based on available hardware
- **Serve block wiring**: `kv_transfer` config string flows through codegen, workers emit `nsl_kv_transfer_init`/`destroy`

### M35b: GPTQ Full OBQ Algorithm
- **Optimal Brain Quantizer**: Column-wise quantization with Hessian-based error compensation (replaces RTN stub)
- **Hessian computation**: `HessianAccumulator` for X^T X calibration data accumulation
- **Cholesky factorization**: Damped Hessian inverse via Cholesky decomposition for numerical stability
- **Act-order**: Columns quantized in descending Hessian diagonal order for better quality
- **Blocked updates**: Lazy batch error propagation for memory efficiency on large matrices
- **Calibration FFI**: `nsl_gptq_hessian_init`, `nsl_gptq_hessian_add_batch`, `nsl_gptq_hessian_finalize`

### M54b: Bare-Metal Unikernel Boot Stub, Runtime & GPU Init
- **x86_64 boot stub generator**: Multiboot2 header, GDT (64-bit code/data segments), PML4/PDPT page tables (identity-map 4GB), SSE/AVX enable, long mode transition
- **Unikernel runtime**: Bump allocator (lock-free atomic), serial console (COM1 115200 8N1), boot config JSON parser
- **GPU init framework**: PCI bus scan (CF8h/CFCh), NVIDIA device discovery, VFIO passthrough path (cuInit), direct register path (BAR0 MMIO)
- **ELF image builder**: Combines boot stub + compiled code + weights + linker script into single binary

### Documentation
- Updated README.md with new CLI commands (unikernel, ZK), test count (1,558)
- Updated implementation status: 34 production milestones (was 30), 131,800 LOC across 282 files
- Updated CHANGELOG and SPECIFICATION

## [0.8.0] - 2026-03-18

### Consolidation & Code Quality
- **CLI flag wiring**: all CompileOptions (--no-autotune, --deterministic, --disable-fusion, --tape-ad, --trace-ops, --nan-analysis, --target) now flow from CLI to compiler
- **Refactored hotspot files**: tensor.rs (5K→6 files), expr.rs (3.5K→6), compiler.rs (2.8K→7), checker.rs (2.6K→8), autodiff.rs (1.8K→3)
- **Error handling**: replaced 14 panics in process spawning and FFI with graceful error codes
- **Parser**: generic trait bounds now parsed (not enforced yet); if-expression limitations documented
- **Deterministic scatter_add**: changed from silent null return to explicit abort with message
- **E2E precision**: float comparison tightened from 4 to 6 decimal places
- **Version**: workspace version aligned to release tags

### Phase 8–9 Infrastructure (analysis + FFI complete, codegen wiring in progress)
- **M45**: Tensor debugger — trace recording, NaN analysis, trace diffing, Chrome export
- **M46**: Reproducibility — determinism checker, kernel variant selection, RNG tracking
- **M48**: Multimodal — PatchEmbed, MelSpectrogram, cross_attention, modality classification
- **M49**: Shape algebra — symbolic dimension solver (equality, divisibility, range proofs)
- **M50**: Sparse tensors — NslSparseTensor, COO/CSR/CSC/BSR format dispatch

## [0.7.0] - 2026-03-18

### Phase 7: Distributed Training
- **M38b**: Linear types codegen — ownership decisions for tensor lifetime
- **M40b**: Source AD extraction — Wengert extraction from AST, backward context
- **M43**: Pipeline parallelism — 1F1B/GPipe scheduling, 3D rank mapping, ZeRO sharding

## [0.6.0] - 2026-03-18

### Phase 6: Deployment & Portability
- **M41**: Disaggregated inference — prefill/decode worker separation, KV transfer
- **M47**: Multi-backend KIR — Kernel IR, PTX backend, GpuTarget, GpuBackend trait
- **M39b**: vmap AST transform — VmapTransformer FnDef→FnDef rewriting
- Snapshot testing (insta) and differential testing infrastructure

## [0.5.0] - 2026-03-18

### Phase 5: Inference Optimization
- **M42**: KV-cache compression — INT8/INT4/FP8, sliding window, H2O eviction
- **M44**: Constrained decoding — compiled FSM, token-level DFA, logit masking

## [0.4.0] - 2026-03-18

### Phase 4 continued
- **M41**: Disaggregated inference (moved to Phase 6 delivery)

## [0.3.0] - 2026-03-17

### Phase 4: Scaling & Optimization (M32-M40)
- **M32**: Mixture of Experts — @moe, top-k gating, capacity routing, aux loss
- **M33**: Speculative Decoding — @speculative, tree attention, rejection sampling
- **M34**: Ring Attention — @context_parallel, cross-GPU sequence parallelism
- **M35**: FP8/AWQ/GPTQ quantization
- **M36**: Memory planning — compile-time liveness analysis, slab allocation
- **M37**: Roofline cost model — per-op FLOP/byte analysis
- **M38a**: Linear types semantics — ownership checker, @shared
- **M39a**: vmap analysis — batch tracking, shape rewriting, matmul classification
- **M40a**: Source AD analysis — Wengert list, adjoint rules, dead gradient elimination

## [0.2.0] - 2026-03-15

### Production Inference & Optimization (M23-M31)

#### M23: Custom Datatypes (BYOD)
- `datatype` block with `@pack`/`@unpack` methods for user-defined numeric formats
- Custom dtype registration with element-wise pack/unpack dispatch
- NslTensor.dtype expanded from u8 to u16 for custom dtype IDs

#### M24: Standalone Export
- `nsl build --standalone` produces zero-dependency native executables
- Embedded weights (bundled in binary) and sidecar weights (.nslweights file)
- WeightProvider abstraction with embedded and mmap backends
- Build-time safetensors reading for weight bundling

#### M25: PagedAttention & Memory Profiling
- Paged KV-cache with BlockAllocator, PageTable, and KvCacheManager
- `@paged_kv` decorator for automatic KV-cache management
- Memory watermark profiler with `--profile-memory` flag
- Chrome tracing JSON output for memory analysis

#### M26: @autotune, Fusion & Kernel Profiling
- `@autotune` decorator with Cartesian product search and build-time caching
- `@fuse` decorator for elementwise fusion chain detection
- Fused PTX synthesis for elementwise op chains
- Kernel profiler with Chrome tracing JSON (`--profile-kernels`)

#### M27: FlashAttention-2
- FlashAttention-2 PTX template synthesis with 5 kernel variants
- `scaled_dot_product_attention` lowering with naive and flash paths
- RoPE cache write kernels and GQA replication
- `@flash_attention`, `@rope`, `@gqa` decorator validation
- Shared memory parameter support in kernel_launch

#### M28: Dynamic Shapes & Bounded Dimensions
- Symbolic dimension tracking with `SymbolicDimTracker`
- Bounded dimension syntax (`SeqLen < 4096`) with parse/semantic/codegen support
- Runtime dimension assertions (`nsl_tensor_assert_dim`, `assert_dim_bound`)
- Dimension unification for Bounded and Computed dimensions

#### M29: Continuous Batching & Serving
- `serve` block language frontend (lexer, AST, parser, semantic, codegen)
- `BatchScheduler` with chunked prefill and `RaggedBatchBuilder`
- `PreemptionManager` with swap/recompute policies
- `InferenceRequest` lifecycle management

#### M30: Tensor Parallelism
- `@shard` decorator for weight distribution across GPUs
- `CollectiveBackend` trait with simulated backend for testing
- SPMD process launcher with `--devices` flag
- Tensor parallel FFI: init, rank, collectives (all-reduce, all-gather, broadcast), destroy
- Weight sharding with `compute_shard_slice` and `copy_shard`

#### M31: Graph-Level Operator Fusion
- `FusionGraph` DAG with ANF node model and consumer counting
- Epilogue fusion: matmul+bias+activation chain detection and PTX synthesis
- Reduction fusion: softmax, layernorm, rmsnorm pattern matching and PTX synthesis
- `@fuse_graph` and `@no_fuse` decorator validation
- `--fusion-report` CLI flag for fusion event logging

### Bug Fixes
- Fix use-after-free in autodiff backward for SumReduce/MeanReduce global reductions
- Add `in_tape_region` guard to suppress tensor temporary cleanup during tape recording
- Fix macOS platform version for Cranelift objects
- Fix macOS linker flags and E2E baselines
- Make interop non-default feature to avoid OpenSSL link dependency
- Numerous clippy warning fixes across all modules

## [0.1.0] - 2026-03-12

### Language Features
- Indentation-based syntax with Python-familiar keywords
- Pipe operator (`|>`) for model op chaining
- `let`/`const` variable declarations with type inference
- `fn` functions with named/default parameters
- `model` keyword for neural network definitions
- `grad` keyword for tape-based automatic differentiation
- `train` block DSL with declarative data/optimizer/scheduler
- `quant` block for INT4/INT8 weight quantization
- `kernel` keyword for custom GPU kernels (PTX codegen)
- Compile-time tensor shape checking with named dimensions
- `@no_grad`, `@checkpoint`, `@backward`, `@test` decorators
- Import system with multi-file compilation

### Standard Library
- **nsl.nn**: Linear, Embedding, Conv2d, MaxPool2d, LayerNorm, RMSNorm, Dropout, Attention, TransformerBlock
- **nsl.nn.activations**: relu, gelu, silu, sigmoid, tanh, softmax, elu
- **nsl.nn.losses**: mse_loss, l1_loss, cross_entropy, bce_loss
- **nsl.optim**: SGD, Adam, AdamW, Lion, Muon, SOAP
- **nsl.optim.schedulers**: constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
- **nsl.tokenize**: byte_tokenizer, BPE encode/decode
- **nsl.data**: JSONL/CSV/mmap DataLoader with batching, shuffling, sequence packing
- **nsl.inference**: topk, multinomial, argmax, autoregressive generation
- **nsl.quant**: quantize/dequantize (INT4/INT8)
- **nsl.compat**: safetensors load/save, HuggingFace model loading, ONNX export

### Tooling
- `nsl run` -- compile and execute NSL programs
- `nsl build` -- compile to native executable
- `nsl check` -- type checking and semantic analysis
- `nsl test` -- run `@test` annotated functions
- `nsl export` -- ONNX model export
- `nsl fmt` -- code formatter
- `nsl init` -- project scaffolding

### GPU Support
- CUDA backend with 15 PTX kernels (elementwise ops + matmul)
- `kernel` keyword for custom GPU ops
- Device transfer (`.to(cuda)`, `.to(cpu)`)
- Unified memory via cuMemAllocManaged

### Interop
- Safetensors read/write
- HuggingFace Hub model loading (single + sharded)
- ONNX export

### Known Limitations
- No package manager or dependency resolution
- No PyTorch FFI (`to_torch`/`from_torch`)
- No distributed multi-GPU training (DDP)
- No REPL
- CUDA required for GPU features (no ROCm/Metal)
- Windows requires Visual Studio Build Tools for linking
