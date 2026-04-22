<!-- owner: @bwiemz -->

# Development Setup

Get a working NSL build in under 30 minutes. For PR process (fork, branch, commit format, review), see [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Prerequisites

### Rust toolchain

Pinned via [`rust-toolchain.toml`](../../rust-toolchain.toml): `1.95.0`.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

rustup will auto-install the pinned toolchain on first `cargo` invocation in the repo.

### Platform-specific

- **Linux/macOS** — `gcc` or `clang` (usually pre-installed)
- **Windows** — [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) for `link.exe`. `cargo` auto-detects it.
- **GPU (optional)** — NVIDIA CUDA Toolkit. Set `CUDA_PATH` if it is not in the default location.

You do **not** need CUDA to build and test the compiler. CPU-only paths exercise most of the pipeline. CUDA is required for GPU kernel verification and for running the CSHA / WRGA / FlashAttention e2e tests.

## First build

```bash
git clone https://github.com/bwiemz/NSL.git
cd NSL
cargo build --release -p nsl-cli
```

The binary lands at `target/release/nsl`. Symlink it or add `target/release/` to your `PATH`.

## Cargo cheat sheet

```bash
cargo build                  # debug build of all crates
cargo build --release        # optimized
cargo test                   # run all unit + integration tests
cargo test -p nsl-codegen    # one crate
cargo test <name>            # filter by test name
cargo clippy -- -D warnings  # zero-warnings lint gate (required for merge)
cargo fmt                    # format all crates
```

## `nsl` CLI dev loop

The following subcommands are verified from `crates/nsl-cli/src/main.rs`:

- **`nsl run main.nsl`** — compiles the source to a temp binary in the system temp directory (`%TEMP%` on Windows / `$TMPDIR` on Unix) and executes it immediately. Does **not** JIT PTX at launch; the full frontend+codegen pipeline runs first, then the resulting native binary is spawned.
  - `--profile-memory` — writes `memory_profile.json` on exit
  - `--profile-kernels` — writes `kernel_profile.json` on exit
  - `--profile` — enables all profilers and merges output into `profile.json`
  - `--monitor` — activates the health monitor for train programs (auto-detects `train` block)
  - `--inspect` — activates `@inspect` decorator hooks; dumps tensor stats to `.nsl-inspect/`
  - `--cuda-sync` — forces synchronous CUDA launches (better error attribution, slower)
- **`nsl build main.nsl`** — emits a native binary and embeds PTX
  - `--dump-ir` — prints the Cranelift IR for each function
  - `--standalone` — produces a zero-dependency standalone bundle (requires `-w`/`--weights`)
  - `--emit-obj` — emit only the object file (skip linking)
  - `--wrga-report` — print WRGA allocation report after codegen
- **`nsl check main.nsl`** — runs the frontend (lex → parse → type-check) without codegen
  - `--shapes` — emit a shape-trace table showing inferred tensor dimensions at each op
  - `--nan-analysis` — compile-time NaN/Inf risk scan (M45)
  - `--deterministic` — flag non-deterministic ops (M46)
  - `--weight-analysis --weights <path>` — dead/sparse weight report (M52)
- **`nsl test main.nsl`** — discovers and runs `@test`-annotated functions in the file
- **`nsl profile main.nsl`** — predictive performance report for a target GPU (static cost model, does not require a GPU at report time)
  - `--target h100` — GPU target (default: `h100`; also `a100`, `rtx-4090`, etc.)
  - `--memory` — include the HBM-usage memory timeline in the report
  - `--json` — emit machine-readable JSON instead of a formatted table
- **`nsl debug main.nsl`** — M45 tensor trace reader; find NaNs, diff traces, or export Chrome JSON
  - `--export-chrome <out.json>` — export tensor trace to Chrome Trace Event Format
- **`nsl fmt main.nsl`** — format NSL source files in-place
  - `--check` — exit non-zero if formatting changes would be made (CI mode)
- **`nsl export main.nsl`** — export a model to ONNX or checkpoint to safetensors
- **`nsl convert`** — convert between checkpoint formats (`.nslm` ↔ `.safetensors`)
- **`nsl init <name>`** — scaffold a new NSL project
- **`nsl zk`** — M55 ZK inference circuit operations (`stats`, `prove`, `verify`)
- **`nsl tokenize`** — train a BPE tokenizer from source files

## IDE

- **rust-analyzer** — settings auto-discovered from workspace `Cargo.toml`
- Useful extensions:
  - `usernamehw.errorlens` — surface errors inline
  - `vadimcn.vscode-lldb` — Rust debugger

## DevTools & telemetry <a id="devtools"></a>

NSL ships compile-time-gated telemetry. Use these to verify your change did not regress compiler invariants.

### `nsl profile` — predictive performance table

```bash
# Print an op-cost table for H100 at batch=4, seq=2048
nsl profile main.nsl --target h100 --batch 4 --seq 2048

# Machine-readable JSON
nsl profile main.nsl --json

# Include HBM memory timeline
nsl profile main.nsl --memory
```

`nsl profile` uses a static cost model and does **not** require a GPU to be present. It is useful for catching regressions in codegen shape before running on hardware.

### `nsl run --profile` — runtime Chrome trace

```bash
nsl run main.nsl --profile
# Writes profile.json (merged memory + kernel trace)
```

Load `profile.json` in [`chrome://tracing/`](chrome://tracing/) or [Perfetto UI](https://ui.perfetto.dev/). Shows per-op timing, GPU kernel launches, and memory allocations as recorded by the runtime.

### `nsl run --monitor` — train-loop health monitor

When `--monitor` is passed for a program with a `train` block, the CLI auto-detects the training loop and activates the health monitor instead of the kernel-timing path. Useful for watching loss curves and gradient norms during development.

### `memory_timeline.rs` — HBM slab packing verification

Source: [`crates/nsl-codegen/src/profiling/memory_timeline.rs`](../../crates/nsl-codegen/src/profiling/memory_timeline.rs).

Renders an ASCII staircase showing how HBM slabs are packed across training steps. **The target is 0 MB/step allocation growth.** A flat plateau means aliasing is correct; a rising slope means the slab plan missed an opportunity and the allocator is leaking.

```bash
# Activate via nsl profile --memory
nsl profile main.nsl --memory
```

Interpreting the staircase:

- Horizontal axis: training step
- Vertical "height" characters: live slab bytes
- Flat plateau = correct (no growth)
- Rising slope = investigate aliasing in the WRGA memory plan

### `nsl run --inspect` — `@inspect` decorator hooks

When `--inspect` is passed, any `@inspect`-annotated tensors in the program dump their stats (shape, dtype, min/max/mean, NaN flag) to the `.nsl-inspect/` directory. Useful for verifying tensor values at specific sites without modifying source.

```bash
nsl run main.nsl --inspect
# Inspect dumps land in .nsl-inspect/<step>/<site-name>.json
```

### `nsl check --shapes` — compile-time shape trace

```bash
nsl check main.nsl --shapes
```

Prints a table of every tensor-typed expression with its inferred compile-time shape. Use this to confirm shape propagation is correct after adding a new op or transformation.

### Cranelift IR dumps

```bash
nsl build main.nsl --dump-ir
```

Prints the Cranelift IR for every function as codegen produces it. The flag is on `nsl build` only (not `nsl run`). Useful when debugging incorrect native code — compare the IR against the NSL AST to locate the mis-lowering.

### `nsl debug --export-chrome` — tensor trace viewer

```bash
nsl debug trace.bin --export-chrome out.json
```

Exports a recorded tensor trace to Chrome Trace Event Format. Load in `chrome://tracing/` or Perfetto UI. The `--find-nan` flag scans the trace for NaN sentinels (M45).

### Environment variables

The following `NSL_*` variables are read by the compiler or runtime at process start. All are opt-in; the binary is fully functional without any of them.

| Variable | Where read | Effect |
|---|---|---|
| `NSL_DEBUG` | `nsl-codegen` | Verbose compiler-internal logging (entry-point dispatch, module analysis) |
| `NSL_DEBUG_WENGERT` | `nsl-codegen` | Log Wengert/tape lowering decisions |
| `NSL_DEBUG_SOURCE_AD_OWNED` | `nsl-codegen` | Log source-AD ownership checks |
| `NSL_AUTOTUNE_VERBOSE` | `nsl-codegen` | Print autotune candidate timing as it runs |
| `NSL_AUTOTUNE_FALLBACK` | `nsl-codegen` | Force autotune to fall back to the default kernel |
| `NSL_PROFILE_MEMORY` | `nsl-runtime` | Runtime memory profiler; writes `memory_profile.json` |
| `NSL_PROFILE_KERNELS` | `nsl-runtime` | Runtime kernel profiler; writes `kernel_profile.json` |
| `NSL_GPU_MEM_REPORT` | `nsl-runtime` | Print GPU memory report on program exit |
| `NSL_GPU_MEM_LIMIT` | `nsl-runtime` | Cap the caching allocator's GPU memory budget (bytes) |
| `NSL_MEMSTATS` | `nsl-runtime` | Print caching-allocator statistics on exit |
| `NSL_CUDA_SYNC=1` | `nsl-runtime` | Synchronize after every CUDA call (equivalent to `CUDA_LAUNCH_BLOCKING=1`) |
| `NSL_WRGA_FUSED_CUDA=1` | `nsl-runtime` | Enable WRGA fused-MMA CUDA launch path |
| `NSL_WRGA_GPU_LAUNCH_COUNTER=1` | `nsl-runtime` | Count real WRGA GPU kernel launches (for tests) |
| `NSL_CSHA_DUMP_SAVE_STATE` | `nsl-codegen` / `nsl-runtime` | CSHA diagnostic probe; values: `wrow`, `wid`, `qstart`, `fmax`, `newmax`, `fsum`, `direct_s0` |
| `NSL_CSHA_DUMP_GRADS` | `nsl-runtime` | Dump CSHA backward gradient values to stderr |
| `NSL_STDLIB_PATH` | `nsl-codegen` | Override path to the NSL standard library source |

`NSL_MEMORY_TIMELINE`, `NSL_CRANELIFT_DUMP`, and `NSL_PTX_DUMP` do **not** exist in the codebase; use `nsl profile --memory`, `nsl build --dump-ir`, and the `NSL_DEBUG` flag respectively.

### TensorScope (M45, upcoming)

Runtime tensor-value tracing with NaN sentinel detection. The `nsl debug` subcommand (already shipped) is the read side of this infrastructure. Full interactive TensorScope UI: `docs/superpowers/specs/` (search `m45`). Not yet shipped.

### Checklist — did my change regress the memory model?

After a codegen or runtime change:

1. Run unit tests: `cargo test`
2. Run the memory timeline on a representative example:
   ```bash
   nsl profile main.nsl --memory
   ```
3. Verify the plateau is flat (0 MB/step growth)
4. If non-flat, diff the WengertList or WRGA slab plan against the previous commit

## Debugging

- `RUST_LOG=debug` — verbose tracing
- `RUST_BACKTRACE=1` — panic stacks
- `RUST_BACKTRACE=full` — full symbol stacks
- `CUDA_LAUNCH_BLOCKING=1` — forces synchronous CUDA launches (better error attribution); equivalent to `NSL_CUDA_SYNC=1` on the NSL side

## Recurring compile-error traps

The session history surfaces a recurring set of Rust compile errors when touching `nsl-codegen`:

- **E0063** — missing fields in struct initializer (e.g., `Compiler { ... }`). When adding a new field to a struct, every initializer site needs updating; `cargo build` will enumerate them all.
- **E0599** — method not found on `&mut Compiler`. Usually means a new method was referenced before being added; check the `impl Compiler` block.
- **E0308** — mismatched types. Often `Tensor<[...]>` vs `NslTensor`; check whether you are at the FFI boundary.

Anticipate these when adding a language feature — see [Adding-a-Language-Feature](Adding-a-Language-Feature.md).

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
