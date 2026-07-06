<!-- owner: @bwiemz -->

# Compiler Pipeline

How a `.nsl` file becomes a native binary, stage by stage. For the 10,000-ft view, see [Architecture-Overview](Architecture-Overview.md). For the optimization passes that run inside Stage 4, see [Optimization-Passes](Optimization-Passes.md).

## Stage 1 — Lexing

**Crate:** [`crates/nsl-lexer/`](../../crates/nsl-lexer/)

The lexer turns a `.nsl` source byte-stream into a token stream. The non-obvious parts:

- **Indentation handling** — NSL is Python-like, so the lexer emits synthetic `Indent` / `Dedent` tokens (not the parser). This logic lives in [`indent.rs`](../../crates/nsl-lexer/src/indent.rs): `IndentTracker::process_indent` compares the current column to a stack of previous indentation levels and emits the appropriate tokens. `IndentTracker::finalize` flushes any remaining `Dedent` tokens at EOF.
- **Keywords are lexed** — `model`, `train`, `grad`, `quant`, `kernel`, `datatype`, `serve`, `let`, `const`, `fn` and their siblings are all resolved in the lexer via [`keywords.rs`](../../crates/nsl-lexer/src/keywords.rs). The function `lookup_keyword` maps identifier strings directly to typed `TokenKind` variants (`TokenKind::Model`, `TokenKind::Train`, `TokenKind::Grad`, `TokenKind::Quant`, …). By the time any token reaches the parser, the distinction between a keyword and an identifier is already settled.

Start reading: [`lib.rs:17`](../../crates/nsl-lexer/src/lib.rs#L17) — `pub fn tokenize(source: &str, file_id: FileId, interner: &mut Interner) -> (Vec<Token>, Vec<Diagnostic>)` is the single public entry point. `FileId` (from `nsl-errors`) scopes diagnostics to the originating file; `Interner` is shared mutable state used to intern identifier strings — it must be the same instance passed to the parser.

## Stage 2 — Parsing

**Crate:** [`crates/nsl-parser/`](../../crates/nsl-parser/) + AST types in [`crates/nsl-ast/`](../../crates/nsl-ast/)

The parser builds the AST. It receives the flat `Vec<Token>` from Stage 1 and produces a `Module` whose node types are defined in `nsl-ast`. **New AST variants belong in `nsl-ast`; new parsing rules belong in `nsl-parser`.** If the keyword is new, add it to `lookup_keyword` in [`crates/nsl-lexer/src/keywords.rs`](../../crates/nsl-lexer/src/keywords.rs) first — the lexer must emit a typed `TokenKind` before the parser can match on it. Gotchas:

- **`model`, `train`, `grad` are keywords** — the parser's statement dispatcher pattern-matches on `TokenKind::Model`, `TokenKind::Quant`, `TokenKind::Train`, `TokenKind::Grad` (see [`stmt.rs:21`](../../crates/nsl-parser/src/stmt.rs#L21) — `parse_stmt`'s top-level `match`). You cannot name a variable `model`; the lexer already tagged it as `TokenKind::Model` in Stage 1. Note that [`parser.rs:139`](../../crates/nsl-parser/src/parser.rs#L139) also matches on keyword `TokenKind` variants, but that is the import-path helper that allows keyword names as path segments (e.g., `nsl.quant`) — not the statement dispatcher.
- **Named dimensions** like `Tensor<[batch="B", heads="H"], fp8>` are parsed as first-class AST nodes — not a later rewrite. See [`types.rs`](../../crates/nsl-parser/src/types.rs).
- **The pipe operator `|>`** has a specific precedence handled by the Pratt parser in [`pratt.rs`](../../crates/nsl-parser/src/pratt.rs). See [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md) for the precedence table.

Start reading: [`lib.rs:20`](../../crates/nsl-parser/src/lib.rs#L20) — `pub fn parse(tokens: &[Token], interner: &mut Interner) -> ParseResult` is the single public entry point; it returns the `Module` AST plus any parse diagnostics.

## Stage 3 — Semantic analysis

**Crate:** [`crates/nsl-semantic/`](../../crates/nsl-semantic/)

The semantic pass enforces:

- **Shape checking** — tensor operations must have compatible shapes, symbolic or concrete. Shape arithmetic is in [`shapes.rs`](../../crates/nsl-semantic/src/shapes.rs) (`check_elementwise`, `check_matmul`). See [`spec/02-tensor-type-system.nsl.md`](../../spec/02-tensor-type-system.nsl.md).
- **Ownership** — NSL's ownership analysis (linear types semantics shipped as M38a behind a flag) prevents double-use of consumed tensors. See [`ownership.rs`](../../crates/nsl-semantic/src/ownership.rs) and the walker in [`ownership_walker.rs`](../../crates/nsl-semantic/src/ownership_walker.rs).
- **Determinism analysis** — annotations like [`@no_grad`](Glossary.md#dec-no-grad), [`@checkpoint`](Glossary.md#dec-checkpoint) propagate through the AST via [`determinism.rs`](../../crates/nsl-semantic/src/determinism.rs).
- **Decorator resolution** — every `@<name>` decorator is looked up here and its semantic constraints applied; subsystem-specific passes include [`csha.rs`](../../crates/nsl-semantic/src/csha.rs), [`wrga.rs`](../../crates/nsl-semantic/src/wrga.rs), [`cpdt.rs`](../../crates/nsl-semantic/src/cpdt.rs), and others.
- **Effects and NaN analysis** — [`effects.rs`](../../crates/nsl-semantic/src/effects.rs) tracks side-effect annotations; [`nan_analysis.rs`](../../crates/nsl-semantic/src/nan_analysis.rs) flags compile-time NaN/Inf risks.

Shape-check failures produce compile-time errors with source-location spans. They are not runtime errors.

The top-level checker lives in [`checker/mod.rs`](../../crates/nsl-semantic/src/checker/mod.rs): `Checker::check_module` walks the `Module` AST and dispatches into sub-checkers in [`checker/expr.rs`](../../crates/nsl-semantic/src/checker/expr.rs), [`checker/decl.rs`](../../crates/nsl-semantic/src/checker/decl.rs), etc.

Start reading: [`lib.rs:83`](../../crates/nsl-semantic/src/lib.rs#L83) — `pub fn analyze(module: &Module, interner: &mut Interner) -> AnalysisResult` is the public entry point that runs all passes and returns a unified `AnalysisResult`.

## Stage 4 — Codegen

**Crate:** [`crates/nsl-codegen/`](../../crates/nsl-codegen/)

NSL uses **Cranelift as the sole function-emission backend**. Every function — host code, `train` block scaffolding, autodiff, FFI shims, optimizer state updates, `@export` shims — lowers to Cranelift IR → native machine code. The Cranelift dependency is declared in [`Cargo.toml`](../../crates/nsl-codegen/Cargo.toml) as `cranelift-codegen`, `cranelift-frontend`, `cranelift-module`, `cranelift-object`, and `cranelift-native`. The function-level compiler lives under [`compiler/`](../../crates/nsl-codegen/src/compiler/).

### Autodiff lowers to Cranelift, not PTX

This is the most common misconception about Stage 4. The `WengertList` — the symbolic backward graph produced by source AD — is lowered by [`wengert_lower.rs`](../../crates/nsl-codegen/src/wengert_lower.rs) into **Cranelift IR** (typically as FFI calls into the NSL runtime), not into PTX. Autodiff is a host-side concern; `wengert_lower.rs` takes a `&mut FunctionBuilder` (Cranelift) and emits Cranelift `Value`s.

### PTX synthesis is a separate pipeline

GPU kernels are not a second function-emission backend. PTX text is synthesized by kernel-compilation passes and then embedded as **Cranelift data sections** in the emitted binary. At runtime, `cuModuleLoadData` loads those bytes.

Two PTX synthesis paths exist:

1. **Direct AST → PTX (default CUDA target)** — [`kernel.rs`](../../crates/nsl-codegen/src/kernel.rs) (`KernelCompiler`) translates user-authored `KernelDef` AST nodes directly to PTX strings without going through `KernelIR`. This is the path user `kernel` blocks take on the default `--target cuda` (see the dispatch in [`compiler/kernel.rs`](../../crates/nsl-codegen/src/compiler/kernel.rs) `compile_single_kernel`). Constructs outside its supported subset are refused with a compile error.

2. **Portable subset (M47b, non-CUDA targets only)** — for `--target rocm|metal|webgpu`, `KernelDef` AST nodes are lowered by [`kernel_lower.rs`](../../crates/nsl-codegen/src/kernel_lower.rs) (`lower_kernel_to_ir`) into a backend-agnostic `KernelIR`, then emitted by the per-backend lowerers (`backend_amdgpu.rs`, `backend_metal.rs`, `backend_wgsl.rs`; [`backend_ptx.rs`](../../crates/nsl-codegen/src/backend_ptx.rs) exists for KIR→PTX but is not in the CUDA dispatch). Stores and control flow on this path are deferred to M47c and refuse with a compile error; there is no automatic fallback between the two paths.

Subsystem-specific fusions (CSHA, WRGA, FlashAttention-2) have their own templated PTX emitters. Their output also becomes a Cranelift data section. See [Optimization-Passes](Optimization-Passes.md) for the per-subsystem pass structure.

The resulting PTX text is embedded as a Cranelift data section inside the compiled native binary (see [`compiler/kernel.rs`](../../crates/nsl-codegen/src/compiler/kernel.rs) and [`compiler/main_entry.rs`](../../crates/nsl-codegen/src/compiler/main_entry.rs)). At runtime, the NSL runtime loads each PTX string on demand via `cuModuleLoadData` (see [`crates/nsl-runtime/src/cuda/mod.rs`](../../crates/nsl-runtime/src/cuda/mod.rs)). **There is no separate PTX compilation step at launch** — the PTX is baked into the binary at `nsl build` / `nsl run` time.

> ⚠️ **If you are looking for Hopper TMA / `mma.sync` / `cp.async.bulk` instructions, search the PTX emitters (`backend_ptx.rs`, subsystem fusion modules) — not the Cranelift IR dumps.** Cranelift only sees the embedded PTX as opaque bytes.

### Bare-metal target

For M54 (unikernel) targets, [`unikernel.rs`](../../crates/nsl-codegen/src/unikernel.rs) drives an alternate emission path that skips the host runtime linkage.

Between AST and codegen, a sequence of IR rewrite passes runs: FASE (planning) → source AD → WGGO → subsystem-specific (CSHA, WRGA, CPDT). These are described in [Optimization-Passes](Optimization-Passes.md).

Start reading: [`lib.rs:750`](../../crates/nsl-codegen/src/lib.rs#L750) — `pub fn compile_with_options(source: &str, opts: &CompileOptions) -> Result<Vec<u8>, CodegenError>` is the top-level entry point that runs all semantic-to-binary work.

## Stage 5 — Linking & runtime handoff

The emitted native binary links against `libnsl_runtime.a` (from [`crates/nsl-runtime/`](../../crates/nsl-runtime/)). PTX cubins are embedded in the binary at compile time and loaded on demand at runtime. The deployment mode affects what else gets bundled:

- **`nsl run main.nsl`** — compiles to a temp directory (under `<system-temp-dir>/nsl_run_<pid>/`, resolved via `std::env::temp_dir()` — `%TEMP%` on Windows, `$TMPDIR` on Unix), then immediately executes the resulting binary. The PTX is already embedded in that binary. No separate JIT step occurs at launch.
- **`nsl build main.nsl`** — compiles to the output path. PTX is embedded in the binary exactly as above, but the binary is kept rather than deleted.
- **`nsl build --standalone`** — additionally invokes the standalone pipeline (see [`crates/nsl-cli/src/standalone.rs`](../../crates/nsl-cli/src/standalone.rs) and [`crates/nsl-codegen/src/standalone.rs`](../../crates/nsl-codegen/src/standalone.rs)) to bundle the runtime + PTX + model weights into a single deployable. Weight embedding mode is controlled by `--embed-weights auto|always|never`.

Runtime internals — tensor layout, FFI conventions, caching allocator — are covered in [Runtime-Internals](Runtime-Internals.md).

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
