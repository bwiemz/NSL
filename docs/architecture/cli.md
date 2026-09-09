# nsl-cli architecture

`crates/nsl-cli` builds the `nsl` binary and a small `nsl_cli` library. It is
the driver: it owns argument parsing, the frontend pipeline, the build and
run flavours, the flag contract that keeps advertised features honest, the
generated reference pages, and the largest gate suite in the workspace. This
document is the contributor map; the runtime it links is described in
`crates/nsl-runtime/ARCHITECTURE.md` and the companion `runtime.md`, and
compile-time state in `docs/architecture/compiler-state.md`.

## Overview

`Cargo.toml` declares two targets: the `nsl` binary (`src/main.rs`) and the
`nsl_cli` lib (`src/lib.rs`). The split exists because tests and the profiler
need pieces of the CLI as a library: `lib.rs` exports `analysis_bridges`,
`exec_markers`, `feature_rules`, `health_monitor`, `monitor`, `profile`,
`profile_render`, `shape_debug`, `wggo_explain`, `loader`, `mangling`,
`resolver`, and `formatter`. Everything else is `mod`-declared in `main.rs`
and `pub(crate)`. The crate depends on the runtime with `features =
["interop"]`, and its own features (`cuda`, `nccl`, `test-hooks`,
`onnx-rt-op`, `csha_cycle19_probe`) forward to `nsl-codegen` / `nsl-runtime`.

**`main.rs`.** `main` spawns a thread named `nsl-main` with a 16 MiB stack
and joins it, exiting 101 if the driver panicked; `main_inner` does
`Cli::parse()` and one `match` arm per subcommand. Two helpers live there
because several commands need them: `has_train_block` (a module contains a
`TrainBlock` or `DistillBlock`) and the `nsl doc stdlib` root lookup that
reports "no stdlib directory found (looked at $NSL_STDLIB_PATH, ...)".

**`args.rs`** is the clap tree, ~1,900 lines, all `pub(crate)`. `enum Cli`
has the variants `Check(CheckArgs)`, `Run(RunArgs)`, `Build(BuildArgs)`,
`Test`, `Export`, `Convert`, `Init`, `Fmt`, `Debug`, `Zk { cmd: ZkCmd }`
(`Stats`, `Prove`, `Verify`), `Profile`, `Autotune`, `Tokenize`,
`FpgaCompile`, `PtxMetadata`, `Env { cmd: EnvCmd }` (`List`, `Current`),
and `Doc { cmd: DocCmd }` (`Cli`, `Stdlib`). `BuildArgs` and `RunArgs`
declare every shared compile flag **twice** — a `--source-ad` field exists in
both — which is the drift the flag contract below exists to police.

**`commands/`** holds one file per subcommand (25 files):

| File | Command |
|---|---|
| `check.rs` | `nsl check` — lex/parse/semantic without codegen; `--linear-types`, `--cpkd-design-student` (via `cpkd_design.rs`) |
| `build/` | `nsl build`, `nsl run`'s build half, `nsl zk`, WRGA analysis: `mod.rs`, `normal.rs` (single/multi-file object emit + link), `run.rs` (`build_to_temp` / `execute_temp_build`), `shared_lib.rs` (`--shared-lib`, C header emission), `standalone.rs`, `zk.rs`, `wrga_check.rs`, `reports.rs`, `options.rs` (`CompileOptions` assembly) |
| `run.rs` | `nsl run` dispatcher — wraps the build with the monitor/profiler/multi-process spawners |
| `test.rs` | `nsl test` — compile-and-run NSL test files with an optional filter |
| `export.rs` | `nsl export` — ONNX / safetensors export |
| `convert.rs` | `nsl convert` — NSLM ↔ safetensors |
| `fmt.rs` | `nsl fmt [--check]` — the formatter driver over `src/formatter.rs` |
| `init.rs` | `nsl init` — project scaffolding |
| `autotune.rs` | `nsl autotune` — measure `@autotune` kernel variants on this GPU |
| `tokenize.rs` | `nsl tokenize` — train a BPE tokenizer over source directories |
| `fpga.rs` | `nsl fpga-compile` — NSL → KIR → HIR → Verilog (experimental) |
| `ptx_metadata.rs` | `nsl ptx-metadata` — static per-kernel resource report for a `.ptx` |
| `env.rs` | `nsl env list` / `nsl env current` |
| `cep.rs`, `cpkd_design.rs`, `profile_merge.rs`, `mod.rs` | CEP pruning frontend, CPKD design-student check, trace merging for `nsl run --profile`, shared helpers |

`nsl debug`, `nsl profile`, and `nsl doc` are dispatched straight from
`main.rs` into `src/debug.rs`, `src/profile.rs`, `src/cli_reference.rs` and
`src/stdlib_reference.rs`.

**`pipeline.rs`: frontend → semantic → codegen.** `frontend_with_source_map`
reads the file, `nsl_lexer::tokenize`s it, `nsl_parser::parse`s the tokens,
runs `nsl_semantic::analyze_with_imports` (threading `linear_types` so E0610
fires), emits every diagnostic through the `SourceMap`, and `exit(1)`s if any
is an error — so by the time a command holds the `AnalysisResult` the
frontend is known-clean. `exit_on_codegen_error` renders a `CodegenError`
through the same source map when it carries a span. The rest of the file is
bridges from semantic decorator configs into codegen newtypes
(`module_data_to_wrga_inputs`, `module_data_to_fused_ce_configs`, ...); the
single-file twins live in `src/analysis_bridges.rs` so `profile` can reuse
them. Multi-file programs go through `src/loader.rs` (`ModuleData`) and the
import resolver in `src/resolver.rs`.

**How `nsl run` executes.** There is no JIT. `commands/build/run.rs`
`build_to_temp` compiles the program exactly as `nsl build` would — Cranelift
emits a `.o` (`commands/build/normal.rs`; `--emit-obj` stops here), then
`nsl_codegen::linker::link` locates `libnsl_runtime.a` (`find_runtime_lib`)
and invokes the system C compiler (`find_c_compiler` tries `gcc`, `cc`,
`clang`; MSVC `link` on Windows) with the object, the static runtime, and
the platform libraries — into a scratch directory. `execute_temp_build` then
`std::process::Command::new(exe_path)`s the result with the program's
arguments, forwards its exit code, and deletes the whole scratch directory.
The split into two functions exists so `nsl run --monitor` can start its
health poller between the build and the child. Multi-device runs
(`commands/run.rs`) spawn one child per rank from the same executable.

**`NSL_STDLIB_PATH`.** `resolver::stdlib_roots` resolves the standard library
in order: `$NSL_STDLIB_PATH`, `<exe_dir>/stdlib/`, `<exe_dir>/../lib/stdlib/`
(the distribution layout), then `./stdlib/` for `cargo run` from the
workspace root. `nsl.math` maps to `<root>/nsl/math.nsl`.

## The flag contract

Three registries make the CLI's promises checkable instead of trusted.

**`feature_rules.rs` — composition rules** (roadmap item 20). `FEATURE_RULES`
is a `const` table of `FeatureRule`s built with `clap_rule(flag, RuleKind,
other)` (17 entries) and `src_rule(...)` (71 entries). Each rule carries an
`Enforcement`:

- `Enforcement::Clap` — a `requires` / `conflicts_with` attribute that must be
  present and identical in **both** `BuildArgs` and `RunArgs`;
- `Enforcement::Source { file, fragment }` — a hand-written refusal in
  compiler or CLI code, pinned by repo-relative file and a distinctive,
  whitespace-normalised slice of its user-facing message.

`clap_rules()` / `source_rules()` iterate the table; `flag_to_field` maps a
`--flag` to its clap field. `crates/nsl-cli/tests/feature_composition_gate.rs` runs three
tiers: A parses `args.rs` and checks the clap attributes agree across both
arg blocks; B greps each `Source` fragment out of its file, so a deleted or
reworded refusal fails; C drives the binary for the compositions clap can
actually reject end to end. Nothing in the table is transcribed on trust — a
row whose enforcement is gone is a failing test.

**`exec_markers.rs` — the stderr vocabulary.** `EXEC_MARKERS` lists every
bracketed tag a subsystem prints to announce that it engaged — `[wggo]`,
`[csla]`, `[zero3]`, `[fase-multi]`, `[cuda-graph]`, `[arena]`,
`[sr-bf16]`, `[param-plan]`, `[nsl-gpu-launch-count]`, `[wgrad-fusion]`, ...
— as `ExecMarker { token, emitted_by, means }`, with the constants under
`exec_markers::tokens` (`tokens::CSLA`, `tokens::ZERO3`, ...). The
convention: a marker is `[subsystem]` at the start of a stderr line followed
by a space-separated status (`[wgrad-fusion] declined: train block #N — ...`,
`[fase-fused] ...`), and gates compare those lines byte for byte. Two kinds
of assertion use them: **positive needles** (`stderr.contains("[csla]")`
proves the feature ran) and **negative needles** (`!stderr.contains("[csha]
csha[")` proves it did *not*). The registry exists because a renamed tag
turns a negative needle into a permanent false pass; the `emitted_by` list
lets a gate fail when the emitting call site disappears. `EVENT_SCHEMAS`, in
the same file, is the registry of `NSL_EVENTS` JSONL kinds and their fields —
the machine-readable twins of the markers.

**`meta_flags.rs` — bundles.** `--pretrain-optimized` expands
(`expand_pretrain_optimized`, returning a `PretrainBundle`) into
`--source-ad`, `--wggo greedy`, `--csha auto`, `--fuse-rmsnorm-backward` and
`--fuse-wgrad-accum`, filling only flags the user left unset. Because the
bundle is applied after clap has validated, clap can no longer enforce
`--fuse-wgrad-accum`'s conflicts; `WgradFusionBlockers` re-checks them, with
a matching hard error in the codegen's `stmt.rs` as backstop.
`apply_training_reference` (`--training-reference`) strips the
arithmetic-changing fusions back out; `parse_checkpoint_stride` parses the
`--checkpoint-blocks` stride syntax.

**Activation contracts (Milestone A).** "No advertised feature can be
silently inert." The pure half — the contract table mapping each request
surface (a flag, a decorator) to the owner that must record activity, and
the reconciler — is `nsl_codegen::activation`. The policy half is
`src/activation_enforce.rs`: `requested_long_flags` reads argv (a flag the
user typed is a request; a clap default is not; tokens after `--` are program
arguments), `enforce_from_argv(Subcommand)` (the `Subcommand` enum comes from
`nsl_codegen::pass_registry`) runs the reconciliation after codegen and, on `build` and `run`, turns an unsatisfied contract into a hard
error (exit 1); `--allow-inert-requests` demotes it to a warning; `check` is
report-only because it runs no codegen. `refuse_unimplemented_distribute`
and `apply_allow_unknown_decorators` handle the two request surfaces that
need special-casing. `crates/nsl-cli/tests/activation_contract_gate.rs` runs each case's
control (same program minus the request) first so a broken harness cannot
pass as enforcement; `cpdt_decorator_activation_gate.rs` and
`fase_decorator_activation_gate.rs` cover decorators.

## Generated references

Three wiki pages are renderings of code and are gated as such — nothing on
them is written by hand:

- **`nsl doc cli`** (`src/cli_reference.rs`, PR #576, roadmap Doc2):
  `render_markdown(&Cli::command())` walks the clap tree and produces
  `docs/wiki/CLI-Reference.md`; the unit test
  `wiki_page_is_the_command_tree_rendering` compares the checked-in page with
  the rendering, so a flag added to `args.rs` without regenerating the page
  fails `cargo test -p nsl-cli`.
- **`nsl doc stdlib`** (`src/stdlib_reference.rs`, PR #590, roadmap Doc3):
  `render_markdown(stdlib_root)` parses the stdlib sources with the real
  parser — a rendered signature is one the compiler accepted — and pairs
  each with its doc comment; `docs/wiki/Stdlib-Reference.md` is gated by
  `wiki_page_is_the_stdlib_rendering`. The stdlib root comes from
  `resolver::stdlib_roots`, hence the `$NSL_STDLIB_PATH` message in
  `main.rs`.
- **`nsl env list [--markdown]` / `nsl env current`**
  (`src/commands/env.rs`, roadmap A5): renders `nsl_env::REGISTRY` from
  `crates/nsl-env/src/registry.rs` — name, `Kind`, accepted values, default,
  `Tier` (`Behavior`, `Perf`, `Safety`, `Platform`, `Diagnostic`, `Test`),
  `ReadAt` (`Compile`, `Runtime`, `Both`, `Test`), doc — into
  `docs/wiki/Environment-Variables.md`. `nsl env current` reports which
  registered variables are set in the calling environment and flags any
  `NSL_*` that is set but unknown; run it before recording a measurement,
  because a `Behavior`-tier variable changes what the run computed.
  `crates/nsl-env/tests/registry_agreement.rs` scans every workspace crate
  (`crates/nsl-env/src/scan.rs`) for `NSL_*` reads and fails on a read
  missing from the registry or a row nothing reads any more.

`crates/nsl-cli/tests/env_cli.rs` drives the `env` subcommand end to end.

## The gate suite

`crates/nsl-cli/tests/` is 172 test files (~51K lines) plus `crates/nsl-cli/tests/fixtures/` and `crates/nsl-cli/tests/differential_scripts/`;
the e2e stdout baselines live at the workspace root in `tests/expected/`. The dominant shape is a **gate**: build or run a small
`.nsl` program through the real binary and assert on exit status and stderr
markers. By filename suffix: 78 `*_gate.rs`, 18 `*_e2e.rs`, 7
`*_gpu_gate.rs`, and the rest named for the subsystem under test (16
`cpdt_*`, 7 `wrga_*`, 6 `wggo_*`, 6 `cep_*`, 5 `ccr_*`, ...). Categories:

- **Composition and flag contract** — `feature_composition_gate.rs`,
  `pretrain_prod_agreement_gate.rs`, `training_reference_gate.rs`.
- **Activation** — `activation_contract_gate.rs`,
  `cpdt_decorator_activation_gate.rs`, `fase_decorator_activation_gate.rs`,
  `run_pca_per_doc_gate.rs`.
- **Checkpoint and resume** — `train_checkpoint_gate.rs`,
  `train_config_resume_gate.rs`, `train_resume_dataloader_gate.rs`,
  `exec_fingerprint_resume_gate.rs`, `ccr_checkpoint_parity.rs`,
  `model_config_drift.rs`.
- **End-to-end examples** — `e2e.rs` compiles and runs the workspace's
  `examples/*.nsl` and diffs stdout against the root `tests/expected/*.txt`; `m56_e2e_examples.rs` runs
  `nsl check --linear-types` over the M56 examples; the `*_e2e.rs` files
  (`csha_checkpoint_decorator_cli_e2e.rs`, `fused_lm_ce_e2e_nsl_source.rs`,
  `pretrain_loss_decrease_gpu_e2e.rs`, ...) run one feature through the CLI.
- **GPU-certified gates** — the `*_gpu_gate.rs` files
  (`deferred_free_gpu_gate.rs`, `fase_fused_step_gpu_gate.rs`,
  `long_run_drift_gpu_gate.rs`, `mem_accounting_gpu_gate.rs`,
  `stream_ordering_gpu_gate.rs`, `stream_migration_gpu_gate.rs`, ...) and
  every other arm carrying `#[ignore = "requires CUDA GPU"]` (64 files, 117
  arms with that exact reason).
- **Doc, version, and inventory agreement** — `gpu_gate_inventory_scanner.rs`
  (edge cases for `scripts/gpu-gate-inventory.awk`), the reference-page
  tests inside `src/cli_reference.rs` and `src/stdlib_reference.rs`, and
  the shell scripts under `scripts/` described below.

**How the binary is reached.** 70 test files use
`env!("CARGO_BIN_EXE_nsl")` — Cargo's path to the freshly built `nsl` — and
drive it with `std::process::Command` or `assert_cmd`. They never call the
compiler in-process, so what they certify is the real user path including
argument parsing, the linker search, and the child's exit code.

**Serial gates.** Gates that spawn several `nsl` processes, hold the GPU, or
depend on process-global runtime state document `-- --test-threads=1` in
their header (`param_plan_gate.rs`, `stream_ordering_gpu_gate.rs`, and the
GPU gates above); CI runs the e2e suite and `zero_spmd_gate` that way.

**CI (`.github/workflows/ci.yml`).**

| Job | What it runs |
|---|---|
| `build-and-test` | `cargo test --workspace --no-fail-fast -- --skip e2e_` (every unit and gate test that is not an e2e example), then `cargo test -p nsl-cli --test e2e -- --test-threads=1`, then `--test zero_spmd_gate -- --include-ignored --test-threads=1`; clippy on crates and benches |
| `cuda-feature` | Compiles the workspace with `--features cuda` against cudart stubs; runs the PTX assembly gate, the `cuda::caching_allocator` tests, and the `csha_ptx_ptxas_validation` emitter check — no device |
| `version-agreement` | `scripts/check-version-agreement.sh`: every hand-duplicated version string equals `[workspace.package].version`; also asserts workflow-invoked scripts are committed executable |
| `doc-agreement` | `scripts/check-doc-agreement.sh` (roadmap item 21): claims in docs about the tree (paths, counts, names) still hold |
| `gpu-gate-inventory` | `scripts/gpu-cert.sh --check-inventory` / `--check-reasons` / `--check-long-arms` against `ci/gpu-cert-manifest.tsv`, with anti-vacuity steps that delete gates and expect the check to fail |
| `hand-ptx-freeze` | `scripts/hand-ptx-freeze.sh`: no new hand-written PTX kernels |
| `test-onnx-rt`, `python-interop`, `fpga` | The ORT custom-op build and Python E2E; `python/tests` against a CPU torch; the Verilator/Yosys FPGA layers |

The design-only enforcement gate is a **separate workflow**,
`.github/workflows/design_only_enforcement.yml` (M35.2): it checks that
`BLOCKED_ON_V_P1_D.md` exists and that the backward emitter stays a stub
while the design is blocked. Real GPU execution is not in CI at all; it is
the `certify` tier of `scripts/gpu-tier.sh` / `scripts/gpu-cert.sh` run on a
machine with a card, and the manifest is how CI proves that lane's inventory
has not drifted.

## Invariants

- **`nsl run` is `nsl build` plus exec.** Never add a run-only compile path;
  both must produce the same object and link line, or the fingerprint and
  resume guarantees in the runtime break.
- **Every shared flag is declared in both `BuildArgs` and `RunArgs` with
  identical clap rules** (`feature_composition_gate.rs` tier A).
- **Every refusal in the table exists in its source file**
  (`Enforcement::Source` fragments, tier B).
- **A request the compiler does not honour is an error**, not a silent
  identical binary (`activation_contract_gate.rs`).
- **Marker tokens are constants**, both positive and negative needles use
  `exec_markers::tokens`, and an emitter that disappears fails
  `EXEC_MARKERS`'s gate.
- **Generated pages equal their renderings**: `CLI-Reference.md`,
  `Stdlib-Reference.md`, `Environment-Variables.md`, and `python/nslpy/_abi.py`
  (`nsl abi python`, pinned by `cargo test -p nsl-abi`).
- **Every `NSL_*` read is registered and tiered** (`registry_agreement`).
- **Every `#[ignore]` carries a reason and is in the manifest**
  (`gpu-cert.sh --check-reasons` / `--check-inventory`).
- **Gates drive the real binary** (`CARGO_BIN_EXE_nsl`), and e2e stdout
  baselines are byte-exact (CRLF is normalised on Windows, nothing else).
- **The frontend exits before codegen on any error**; commands may assume a
  clean `AnalysisResult`.
- **Diagnostics go through `nsl_runtime::nsl_log!`** (roadmap C3): the
  `error:` / `warning:` / `note:` lines and the `[nsl] …` launcher lines
  are `tracing` events (target `cli`, `nsl`, or the line's own marker)
  rendered byte-identically to stderr by the runtime's subscriber
  (runtime.md, "Logging"). Command output on stdout stays `println!`, and
  the `nsl` process opts itself out of `NSL_EVENTS`
  (`events::opt_out_this_process`, first thing in `main`) because that
  stream belongs to the program it spawns.
- **Version strings and doc claims agree with the tree** (the two
  `scripts/check-*-agreement.sh` gates).

## Where to add a new X

**A new subcommand.**
1. Add the variant to `enum Cli` in `src/args.rs` with its clap doc comment
   (that text becomes the reference page).
2. Create `src/commands/<name>.rs` with a `run_*` or `dispatch` entry, add
   `mod <name>;` to `src/commands/mod.rs`, and add the `match` arm in
   `main_inner` (`src/main.rs`). Use `pipeline::frontend_with_source_map` for
   anything that needs the AST, and `exit_on_codegen_error` for codegen
   failures.
3. If the command compiles code, add a `nsl_codegen::pass_registry::Subcommand`
   variant and call `activation_enforce::enforce_from_argv` with it so
   requests cannot go inert through the new path.
4. Regenerate `docs/wiki/CLI-Reference.md` with `nsl doc cli` and commit it;
   `wiki_page_is_the_command_tree_rendering` fails otherwise.
5. Add a gate under `tests/` that drives `CARGO_BIN_EXE_nsl`.

**A new flag.**
1. Add the field to **both** `BuildArgs` and `RunArgs` (and `CheckArgs` if
   `check` should accept it) in `src/args.rs`, with identical `requires` /
   `conflicts_with` attributes; thread it into `CompileOptions` in
   `src/commands/build/options.rs`.
2. Register every composition rule in `FEATURE_RULES`
   (`src/feature_rules.rs`): `clap_rule` for attributes, `src_rule` with the
   refusal's file and message fragment for anything enforced in code.
3. If the flag changes training arithmetic, add it to the execution
   fingerprint (`crates/nsl-runtime/src/exec_fingerprint.rs`) so a resume
   refuses drift; if it only moves bytes, add it to the placement class.
4. If it is an optimisation request, give it an owner in
   `nsl_codegen::activation`'s contract table so an inert request is refused.
5. If `--pretrain-optimized` should imply it, extend `PretrainBundle`.
6. Regenerate `docs/wiki/CLI-Reference.md` (`nsl doc cli`) and, if the flag
   has an environment-variable twin, the registry and
   `docs/wiki/Environment-Variables.md` (`nsl env list --markdown`).

**A new stderr marker.**
1. Emit `[tag] ...` as the start of a stderr line from the runtime or codegen,
   and if it has a JSON twin, `events::emit` from the same counter snapshot.
2. Add `m("[tag]", &["<emitting file>", ...], "<meaning>")` to
   `EXEC_MARKERS` and a `tokens::TAG` constant in `src/exec_markers.rs`; add
   the JSON kind to `EVENT_SCHEMAS`.
3. In gates, reference `tokens::TAG`, never a string literal — especially in
   negative assertions.

**A new gate test.**
1. Put it in `crates/nsl-cli/tests/<subsystem>_<what>_gate.rs`; write a
   small `.nsl` into a `tempfile::tempdir()`, run
   `Command::new(env!("CARGO_BIN_EXE_nsl"))`, and assert on the exit status
   and marker lines with `exec_markers::tokens`.
2. Run the control first (the same program without the feature) so the gate
   cannot pass on a broken harness.
3. A GPU arm gets `#[ignore = "requires CUDA GPU"]` (a reason is mandatory),
   the `-- --ignored --test-threads=1` line in the file header if it needs
   serial execution, and `scripts/gpu-cert.sh --write-manifest` to update
   `ci/gpu-cert-manifest.tsv`.
4. Name e2e example tests `e2e_*` only if they belong to the example suite
   CI skips in the workspace run and executes serially afterwards.
5. If the test pins a documented claim (a count, a path, a version), prefer
   extending `scripts/check-doc-agreement.sh` so the claim and the tree are
   compared in one place.
