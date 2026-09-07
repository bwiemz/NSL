# nsl-semantic architecture

`nsl-semantic` is the crate between the parser and codegen. It takes a parsed
`nsl_ast::Module` plus an interner and produces an `AnalysisResult`: every
diagnostic the front end can raise without generating code, a `TypeMap` from
expression `NodeId` to resolved `Type`, the scope arena, and a set of
validated "side tables" (decorator configs, checkpoint policies, weight
indices) that codegen and the CLI consume directly. It has no I/O and no
knowledge of files: module loading, stdlib lookup and import ordering live in
`nsl-cli`, which calls this crate once per module in dependency order.

The crate is ~19K lines at the top level plus ~8.5K in `checker/`
(about 3K of which are tests). It depends only on `nsl-ast`, `nsl-lexer`
(for `Interner`) and `nsl-errors` (for `Diagnostic`/`Span`); `nsl-parser` is
a dev-dependency used by tests.

Two design rules explain most of the shape of the code:

1. **Refuse, do not ignore.** A config key, decorator name, optimizer kwarg
   or callback name the compiler does not consume is an error, not a
   forward-compat no-op. The module docs of `train_config.rs`,
   `optim_config.rs` and `decorator_registry.rs` each open with the bug that
   motivated this ("`epochz=8` passed `nsl check` clean and trained with
   defaults"). New code should follow the same rule.
2. **One resolver, several callers.** Where semantic and codegen both need
   the same facts (train header, optimizer sections, `@fase`/`@pca`
   decorators), the validator is a pure function in this crate and codegen
   calls it again instead of keeping a second table. Tables kept in two
   places drift; the drift gates in `tests/` exist for the cases where a
   second copy is unavoidable.

## Pipeline

The driver is `analyze_with_imports` in `crates/nsl-semantic/src/lib.rs`
(`analyze` is the no-imports wrapper the tests and benches use). Every stage
appends to one `Vec<Diagnostic>`; nothing short-circuits, so a single
`nsl check` reports all problems.

```
nsl_ast::Module + &mut Interner + ImportTypes + linear_types: bool
        │
        ▼
builtins::register_builtins(&mut ScopeMap, interner)         builtins.rs
        │   root scope gets print/len/zeros/…, is_builtin = true
        ▼
TypeChecker::new(...).check_module(module)                   checker/mod.rs
        │   collect_top_level_decls   (imports first, then fn/model/struct/… pre-declared)
        │   agent::check_linear_types_flag        (E0610 if `agent` without --linear-types)
        │   check_stmt for every top-level stmt   checker/stmt.rs → decl.rs, model.rs,
        │                                          block.rs, expr.rs, ops.rs
        │     ├─ TypeResolver::resolve (resolve.rs) turns TypeExpr → Type
        │     ├─ shapes::check_elementwise / check_matmul (shapes.rs)
        │     ├─ ShapeAlgebraSolver::prove_eq_normalized for reshape (shape_algebra.rs)
        │     ├─ train_config::resolve_train_config + optim_config::resolve_optim_config
        │     └─ per-decorator validators (cftp, cpkd, csha, wrga, cep, cpdt, wggo, …)
        │   effect_checker.propagate(); effect_checker.validate()   effects.rs
        ▼
closed decorator namespace walk                              lib.rs + decorator_registry.rs
        │   nsl_ast::decorator_walk::collect_decorators; unknown → error (+did-you-mean),
        │   documented-but-unimplemented → typed refusal
        ▼
export::validate_exports(module, interner)                   export.rs
        │   @export C-ABI signature rules; returns WeightIndexMap
        ▼
wrga::validate_wrga_custom_adapters(...)                     wrga.rs
        ▼
if linear_types: ownership_walker::analyze_ownership(...)    ownership_walker.rs / ownership.rs
        │   use-after-move, borrow rules; FunctionOwnershipInfo per fn
        ▼
agent pipeline (always runs)                                 agent.rs
        │   AgentRegistry::register_module → extract_apgs → detect_cycles,
        │   check_device_compatibility, check_fan_out,
        │   check_cross_agent_field_access, check_cross_agent_mutation
        ▼
AnalysisResult { diagnostics, type_map, scopes, ownership_info, *_configs, … }
```

Two analyses are **not** part of `analyze_with_imports` and are driven by
the CLI on demand: `nan_analysis::NanAnalyzer::analyze_module` (behind
`nsl check --nan-analysis`) and
`determinism::DeterminismChecker::scan_module` (behind
`nsl check --deterministic`), both wired in
`crates/nsl-cli/src/commands/check.rs`.

## Module map

Line counts are `wc -l` at the time of writing.

| Module (`crates/nsl-semantic/src/`) | Lines | Responsibility |
|---|---|---|
| `lib.rs` | 310 | Driver: `analyze`, `analyze_with_imports`, `AnalysisResult`, `FunctionOwnershipInfo`, `ImportTypes`; decorator-namespace walk. |
| `checker/mod.rs` | 489 | `TypeChecker` struct, `TypeMap`, `check_module`, `collect_top_level_decls`, scope/symbol helpers, pattern binding. |
| `checker/stmt.rs` | 1543 | `check_stmt`: statement dispatch and the big `StmtKind::Decorated` arm that validates ~34 decorator names by `dname == "..."`. |
| `checker/expr.rs` | 601 | `check_expr`: literal/ident/call/subscript/lambda/match typing; the single `type_map.insert(expr.id, ty)` site. |
| `checker/ops.rs` | 705 | Binary/unary ops, `check_call` (builtin special cases, reshape proof, arity/assignability), `check_member_access` (tensor/str/list method table). |
| `checker/decl.rs` | 358 | `check_fn_def`, struct/enum/trait defs, `check_import`/`check_from_import`; registers each fn with the `EffectChecker`. |
| `checker/model.rs` | 511 | `check_model_def`: two-pass field/method collection, `self` typing, member-level decorators (`@shard`, `@moe`, `@pipeline`, …). |
| `checker/block.rs` | 1114 | `check_train_block`, `check_distill_block`, `check_serve_block`, `check_tokenizer_def`, `check_dataset_def`; `DATA_SECTION_KEYS`. |
| `checker/tests.rs` | 3157 | Unit tests written against source snippets (see Tests and gates). |
| `types.rs` | 833 | `Type`, `Dim`, `DimExpr`, `Shape`, `DType`, `Device`, `Effect`; `is_assignable`, `wider_dtype`, `display_type`. |
| `scope.rs` | 255 | `ScopeMap` arena, `ScopeId`, `Scope`, `ScopeKind`, `SymbolInfo`. |
| `resolve.rs` | 755 | `TypeResolver`: syntactic `TypeExpr` → semantic `Type`, generic instantiation, dtype/device/shape/effect resolution. |
| `builtins.rs` | 1362 | `register_builtins`: the root-scope table of builtin functions and their `Type::Function` signatures. |
| `shapes.rs` | 422 | `check_elementwise` (broadcasting), `check_matmul`, `unify_dim`, `fmt_shape`/`fmt_dim`. |
| `shape_algebra.rs` | 1162 | `ShapeAlgebraSolver`: symbolic `DimExpr` equality/divisibility/bound proofs (Fourier–Motzkin), `ProofFailure`. |
| `train_config.rs` | 359 | The `train(...)` header contract: `TRAIN_CONFIG_KEYS`, `resolve_train_config`, `ResolvedTrainConfig`, `TrainConfigPurpose`. |
| `optim_config.rs` | 1387 | The `optimizer:`/`scheduler:`/`callbacks:` section contract: `OptimizerKind`, `ResolvedOptimizer`, `ResolvedScheduler`, `resolve_optim_config`, `VALID_ROLES`, `CALLBACK_NAMES`. |
| `decorator_registry.rs` | 256 | `KNOWN_DECORATORS`, `UNIMPLEMENTED_DECORATORS`, `find`, `unimplemented_refusal`, `suggest`. |
| `effects.rs` | 804 | `EffectSet` bitset, `classify_builtin_effects`, `EffectChecker` (register → propagate → validate), `CheckpointPolicy`. |
| `ownership.rs` | 900 | `OwnershipChecker`: per-binding `OwnershipState`, consume/use/borrow rules, branch symmetry, loop handling. |
| `ownership_walker.rs` | 392 | `analyze_ownership`: AST walk that drives `OwnershipChecker` per fn body; emits `FunctionOwnershipInfo`. |
| `ownership_autodiff.rs` | 253 | `classify_backward_access`: which tape ops need inputs/outputs alive for backward. |
| `export.rs` | 1318 | `@export` validation (C-ABI subset), `WeightIndexMap`. |
| `agent.rs` | 1826 | M56 agents: `AgentRegistry`, `ActionPortGraph`, E0601/E0602/E0603/E0607/E0608/E0610 checks. |
| `determinism.rs` | 344 | `DeterminismChecker`: non-deterministic op classification, RNG state tracking (CLI-driven). |
| `nan_analysis.rs` | 594 | `NanAnalyzer`: `log`/`sqrt`/division risk walk (CLI-driven). |
| `cftp.rs` | 510 | `@fase`, `@pca`, `@fused_lm_ce` validators and their config structs. |
| `cpkd.rs` | 219 | `@fused_kl_ce` on `distill` blocks. |
| `csha.rs` | 190 | `@csha(level=, target=, disable=)` → `CshaConfig`. |
| `wrga.rs` | 662 | `@wrga`, `@freeze`, `@adapter` validators; `validate_wrga_custom_adapters`. |
| `cep.rs` | 600 | `@cep_prune`, `@cep_search` validators. |
| `cpdt.rs` | 380 | `@cpdt(mode=, cluster=, …)` validator. |
| `wggo.rs` | 431 | `@wggo`, `@wggo_target` validators. |
| `cfie.rs` | 568 | `@cfie` serve-block config. |
| `moe.rs`, `pipeline.rs`, `speculative.rs`, `kv_compress.rs`, `context_parallel.rs`, `sparse.rs`, `sparse_layout.rs`, `fp8.rs`, `perf_budget.rs`, `grammar.rs`, `multimodal.rs`, `target.rs`, `vmap.rs`, `inspect.rs` | 24–491 each | One `validate_<name>_decorator` per model- or statement-level decorator. `sparse_layout.rs` (`@layout`) has a validator but no dispatch, which is why `layout` is on the unimplemented list. |

`benches/analyze.rs` benchmarks `analyze` over `examples/*.nsl`
(`cargo bench -p nsl-semantic`).

## Key types

### `Type` (`types.rs`)

`Type` is the resolved, canonical representation; `nsl_ast::types::TypeExpr`
is syntactic and never escapes `resolve.rs`. Variants worth knowing:

- Primitives `Int`, `Float`, `Bool`, `Str`, `Void`, and the specific numerics
  `F32`, `F64`, `Fp16`, `Bf16`, `Fp8E4m3`, `Fp8E5m2`, `Int4`..`Int64`,
  `Uint8`, `TernaryPacked`, `TernaryUnpacked`. `dtype_rank` puts these in
  (family, rank) order so `is_assignable` only widens within a family.
- Compound `List`, `Dict`, `Tuple`, `Optional`, `Union`, `NoneType`.
- Tensor family: `Tensor { shape, dtype, device }`, `Param { shape, dtype }`,
  `Buffer { shape, dtype }`, `Sparse { shape, dtype, format }`,
  `QuantizedTensor`. `Type::is_tensor` and `as_tensor_parts` see through
  `Borrow` and normalise `Param`/`Buffer` to (shape, dtype, `Device::Unknown`).
- Nominal: `Struct`, `Enum`, `Trait`, `Model { fields, methods, .. }`,
  `Agent { fields: (name, Type, AgentFieldOwnership), methods }`,
  `FixedModelArray { element_model, size }`.
- `Function { params, ret, effect: Effect }` — a param of `Type::Unknown`
  marks the function variadic for arity purposes (`check_call`).
- `Module { exports }` for `import nsl.math as math`.
- `Borrow(Box<Type>)` for `&T`.
- `TypeVar(Symbol)` for generics (tracked, not instantiated except through
  `resolve_generic`'s substitution of user struct/enum/trait/model params).
- `Unknown` (inference gave up; propagates silently) and `Error` (already
  reported; poison). `is_indeterminate` covers both and every checker path
  bails out on it so one root error does not cascade.

`is_assignable(source, target)` is the single compatibility predicate:
indeterminate is compatible with everything; `T → &T` auto-borrows and
`&T → T` is allowed for reads (the ownership pass catches consumption);
`Int → Float`; generic `Float`/`Int` narrow to specific dtypes; `List` is
covariant; `NoneType → Optional`; `Param → Tensor` when shape and dtype
match; `Tensor → Tensor` unifies dims pairwise with `shapes::unify_dim`,
treats rank 0 as "unknown shape, always compatible", and requires equal
dtype and compatible device.

### `TypeMap` and `NodeId`

`checker::TypeMap` is `HashMap<NodeId, Type>`. `TypeChecker::check_expr`
computes a type for every `ExprKind` and inserts it exactly once at the end
of the function (`self.type_map.insert(expr.id, ty.clone())`). Statements
have no entry; their sub-expressions do. Codegen holds `type_map: &TypeMap`
on `Compiler` (`crates/nsl-codegen/src/compiler/mod.rs`) and looks up by the
same `NodeId`, defaulting to `Type::Unknown` when missing — so an expression
the checker never visited is invisible, not an error, downstream. The
`tensor_method_result_typing.rs` integration test explains why this matters:
an Unknown-typed chain link is skipped by codegen ownership tracking and
leaks GPU memory.

### Scopes (`scope.rs`)

`ScopeMap` is an arena of `Scope { parent, symbols, kind }` indexed by
`ScopeId`; `ScopeId::ROOT` is the module scope and holds the builtins.
`ScopeKind` is `Module | Function | Method | Model | Block | Loop | Lambda`;
`enclosing_function` and `is_in_loop` walk the parent chain. `SymbolInfo`
carries `ty`, `def_span`, `is_const`, `is_param`, `is_used`, and
`is_builtin`. `declare` refuses a duplicate in the *same* scope, and
`TypeChecker::declare_symbol` turns that refusal into Python-style rebinding
(overwriting the type and clearing `is_builtin`). The `is_builtin` flag is
load-bearing: `check_call`'s by-name special cases (`zeros`, `exp`, `sum`,
…) only fire when the name is still the builtin; a user `fn sum(...)`
sheds the flag and is typed from its declaration. A span heuristic was
tried first and mislabelled the loader's synthesized glob imports (they
carry `Span::DUMMY` too).

The checker is two-pass at the top level (`collect_top_level_decls`:
imports first, then every fn/model/struct/enum/trait/tokenizer/dataset/
datatype/agent pre-declared, recursing through `StmtKind::Decorated`) so
forward references work, and two-pass again inside `check_model_def` so
method bodies see a complete `self`.

### Shapes and dims (`types.rs`)

```
Shape { dims: Vec<Dim> }
Dim   = Concrete(i64) | Symbolic(Symbol) | Named { name, size: Box<Dim> }
      | Bounded { name, upper_bound: i64 } | Computed(Box<DimExpr>) | Wildcard
DimExpr = Sym | Lit | Add | Mul | Div | Mod
```

`TypeResolver::resolve_dim` maps the AST's `DimExpr` one-to-one:
`[8, T, batch=8, seq<4096, _]` becomes `Concrete(8)`, `Symbolic(T)`,
`Named { batch, Concrete(8) }`, `Bounded { seq, 4096 }`, `Wildcard`. A
named dim whose value is a string becomes `Named { .., Wildcard }`.

**An empty `dims` vector means "unknown shape"**, not "scalar" —
`Shape::unknown()` and `Shape::scalar()` are the same value, and every
check treats rank 0 as "skip validation". Bare `Tensor` in a type
annotation resolves to `Tensor { Shape::unknown(), DType::Unknown,
Device::Unknown }`.

### Effects (`types.rs`, `effects.rs`)

`Effect` on a function type is `Concrete(EffectSet) | Var(Symbol) |
Union | Inferred`. `EffectSet` is a `u8` bitset with `PURE`, `IO`, `RANDOM`,
`MUTATION`, `COMMUNICATION`. Section "Ownership / effects" covers the pass.

### Training Configuration Contract (`train_config.rs`, `optim_config.rs`)

"Closed namespace" means the accepted key set is a compile-time constant and
anything outside it is an error with the key named. For the `train(...)`
header:

- `TRAIN_CONFIG_KEYS` = `model, epochs, grad_accumulation, grad_clip,
  checkpoint_save, checkpoint_every, checkpoint_load`. A unit test pins that
  the "expected …" label in the unknown-key diagnostic lists every key.
- `resolve_train_config(train: &TrainBlock, resolve_sym, purpose) ->
  Result<ResolvedTrainConfig, Vec<Diagnostic>>` walks `train.config`:
  a positional arg (`arg.name == None`) is refused; a key already in the
  `seen` set is a `duplicate train config key`; each known key has a literal
  and range rule (`epochs` and `grad_accumulation` integer literal >= 1 via
  `fold_int_literal`, which folds `-3` and `(2)`; `grad_clip` positive
  finite numeric literal below `f64::MAX`; `checkpoint_*` non-empty string
  literals; `model` an identifier); everything else hits the
  `unknown train config key` arm. Pairing rules (`checkpoint_save` needs
  `checkpoint_every` and vice versa) and the missing-`model` rule are judged
  on keys *written*, not values validated, so an author who wrote `model=5`
  is not also told `model` is missing.
- `TrainConfigPurpose::{UserTrainBlock, DistillLowering}`: distill
  lowering synthesises a `TrainBlock` whose model travels in typed fields,
  so `model=` is legitimately absent there.
- Values are stored in `ResolvedTrainConfig`; every consumer reads those
  fields rather than re-scanning `TrainBlock.config`.

The sections contract in `optim_config.rs` follows the same pattern one
level down: `OptimizerKind` (`Sgd, Adam, AdamW, Lion, Muon, Soap`) with a
per-optimizer `accepted_kwargs` table that is exactly the stdlib step
function's parameters (so `SGD(beta2=0.5)` is refused, `SOAP` has no
`no_decay`); `ResolvedScheduler` with one variant per
`stdlib/nsl/optim/schedulers.nsl` function and `fn_name()` giving the name
codegen mangles; `CALLBACK_NAMES` = `on_step, on_epoch, on_epoch_end` with
per-callback parameter tables; `VALID_ROLES` for `no_decay=[...]`, which
`crates/nsl-codegen/src/param_roles.rs` consumes. Defaults live here, not in
the lowering. `resolve_optim_config(sections, block_span, resolve_sym,
purpose)` takes the section slice so `check_distill_block` can pass
`DistillBlock.sections` with `DistillLowering`.

Both resolvers are called from `check_train_block`
(`checker/block.rs`) for diagnostics at check time and again from
`crates/nsl-codegen/src/stmt.rs` (the standard train lowering and its
pipelined twin) as the single source of the values it lowers.

### Decorators

There are three layers:

1. **Name membership** — `decorator_registry.rs`. `KNOWN_DECORATORS` is a
   static table of `KnownDecorator { name, read_by }` where `read_by` is the
   repo path of the consumer. `UNIMPLEMENTED_DECORATORS` lists names the
   documentation advertises that have no implementation (`layout`,
   `tie_weights`, `custom_vjp`, `broadcast`, `mac_array`, `unchecked`,
   `torch`) with a per-name refusal message. `lib.rs` walks every decorator
   in the module (including member-level ones the generic visitor skips) and
   emits `unknown decorator @x on <host>; did you mean @y?` (Levenshtein
   distance <= 2 via `suggest`) or the typed refusal.
   `NSL_ALLOW_UNKNOWN_DECORATORS=1` (set by `--allow-unknown-decorators`
   through `crates/nsl-cli/src/activation_enforce.rs`) demotes only the
   unknown-name error to a warning; refusals stay errors.
2. **Argument validation** — `checker/stmt.rs` (`StmtKind::Decorated`
   arm; `if dname == "cpdt" { crate::cpdt::validate_cpdt_decorator(...) }`
   style) for statement-level decorators, and `checker/model.rs` for
   member-level ones. Each feature module exposes
   `validate_<name>_decorator(deco, &resolve, &mut diagnostics) ->
   Option<Config>`.
3. **Config capture** — validated configs are pushed onto `TypeChecker`
   fields (`wrga_configs`, `csha_configs`, `fused_ce_configs`, `pca_configs`,
   …) and moved into `AnalysisResult` by the driver. `cpdt_decorator_span`
   enforces "exactly one `@cpdt` per program".

## Shape checking

Shape inference is local and forward-only: there is no unification across
statements, no constraint solving over a whole function, and dims are never
rewritten after a `let`. Each expression's shape comes from its operands'
shapes at the moment `check_expr` reaches it.

**What is known statically.** Anything written in a type annotation
(`Tensor<[B, T, D], f32>`), the result of `zeros/ones/rand/randn/empty/full`
called with a list *literal* of ints and identifiers
(`extract_shape_from_args`: ints → `Concrete`, identifiers → `Symbolic`,
anything else → unknown), and everything derived from those through the
rules below. `arange` and any tensor produced by a call to a user function
carry the callee's declared return shape.

**What is deferred to runtime.** Any expression that reaches an unknown
shape: rank 0 skips all checks and produces rank 0. `transpose` with
non-literal axes, `expand`/`unsqueeze`/`select`/`slice` (`view_result_type`
returns the receiver's dtype/device with unknown shape), `.sum()`/`.mean()`
(unknown shape), `.to(...)`, `Dim::Computed` (unifies as `Wildcard`, i.e.
runtime-checked), and reshape targets that cannot be proven (a *warning*,
because the runtime still checks element counts). Train-block `step(batch)`
parameters are typed `Dict<Str, Tensor<unknown>>`, so most training-loop
tensors are runtime-shaped by construction.

**Rules.**

- Elementwise (`+ - * / // % **` on two tensors) →
  `shapes::check_elementwise`: left-pad the shorter shape with
  `Concrete(1)`, then `broadcast_dim` per position — `unify_dim` first
  (wildcard with anything; equal concretes; same-name symbolics; symbolic
  with concrete becomes the concrete; named dims unify their sizes;
  `Bounded` with a concrete within the bound; two same-name `Bounded` take
  the tighter bound), then `Concrete(1)` against anything. Different
  symbolic names are *incompatible* by design (catches `[B, T] + [T, B]`).
  Result dtype is `wider_dtype(l, r)`; device mismatch (both known,
  different) is a separate error in `check_arithmetic`.
- Matmul (`@`) → `shapes::check_matmul`: both rank >= 2, `lhs[-1]` unifies
  with `rhs[-2]`, batch dims unify pairwise from the right over the shorter
  batch prefix; result is `lhs[..-1] ++ [rhs[-1]]`.
- `reshape([..])` → `check_call` builds the product of source and target
  dims as `DimExpr`s (`shape_to_product`, feeding `Bounded` bounds into the
  solver via `assert_bound`) and calls
  `ShapeAlgebraSolver::prove_eq_normalized`. `normalize` canonicalises the
  product so `B*T*D` and `D*(B*T)` compare equal; failure yields a warning
  `reshape may be invalid: <reason>`. The solver also offers
  `prove_divisible`, `prove_le`, `prove_ge` (Fourier–Motzkin over asserted
  bounds), currently unused by the checker.
- `transpose(i, j)` with literal axes swaps two dims. `.T` returns the
  receiver type unchanged (not swapped).
- Assignment / call arguments / return → `is_assignable` unifies dims
  pairwise after a rank check.
- Unary math builtins (`exp`, `log`, `sqrt`, …) and reductions (`mean`,
  `sum`, `reduce_max`, `gather`, `clamp`) return the first argument's type
  unchanged when it is a tensor — so a reduction *does not* drop rank in
  the static view.

**How a shape error is reported.** `check_elementwise`/`check_matmul`
return `Err(Diagnostic)` built with `Diagnostic::error(msg).with_label(op_span,
..)`; the caller pushes it and returns `Type::Error`, which poisons the
enclosing expression so nothing else fires. Messages are rendered with
`fmt_shape`/`fmt_dim`: `[3, <symbolic>, _, <4096]`. Note that symbolic and
named dims print as `<symbolic>`/`<named>` rather than by name — `fmt_dim`
has no interner. `nsl check --shapes` (`crates/nsl-cli/src/shape_debug.rs`,
`format_trace`) prints one line per `let`-bound tensor expression with its
resolved shape and a red mark when a diagnostic's primary label overlaps it;
`nsl check --dump-types` dumps the raw `TypeMap`.

## Name resolution and modules

Within one module, names resolve through `ScopeMap::lookup` walking parent
scopes up to `ROOT`; `ExprKind::Ident` misses produce ``undefined variable
`x` `` and `Type::Error`. Type names resolve in `TypeResolver::resolve_named`:
the builtin dtype/primitive spellings (`int`, `f32`, `bf16`, `i64`, `ternary`,
bare `Tensor`/`Param`/`Buffer`) are matched first, then the scope is
consulted for user structs/enums/traits/models/type variables, else
``undefined type `X` ``. `resolve_generic` handles `list<T>`, `dict<K,V>`,
`Optional<T>` and instantiates user generic types by substituting the
declared `type_params`.

Cross-module resolution is the CLI's job. `nsl-semantic` receives an
`ImportTypes = HashMap<Symbol, Type>` and `check_import`/`check_from_import`
(`checker/decl.rs`) declare each imported name in the module scope with the
type found there (falling back to `Type::Unknown` if absent, so a missing
export degrades to unknown rather than an error at this layer); alias
imports get a `Type::Module { exports }`; glob imports declare every entry.

`from nsl.nn.losses import mse_loss` resolves like this:

1. `crates/nsl-cli/src/loader.rs::load_all_modules` parses the entry file,
   prepends synthetic `from nsl.optim.<optimizer> import *` /
   `from nsl.optim.schedulers import *` statements for any train/distill
   block (`inject_train_block_imports`; that is why optimizer step functions
   are in scope without an import), and calls `discover_imports`.
2. `crates/nsl-cli/src/resolver.rs::resolve_import` turns the path
   `[nsl, nn, losses]` into `nsl/nn/losses.nsl` and looks first relative to
   the importing file, then in each `stdlib_roots()` entry in order:
   `$NSL_STDLIB_PATH`, `<exe_dir>/stdlib`, `<exe_dir>/../lib/stdlib`,
   `./stdlib`. The result is canonicalised and becomes the module-graph key.
3. Modules are topologically sorted (`topological_sort`; cycles are an
   error) and analysed in dependency order. For each module the loader
   builds `import_types` from the already-analysed dependency's exports
   (`inject_import_types` for `from … import`, `inject_alias_import` for
   `import … as`) and calls `nsl_semantic::analyze_with_imports`.
4. After analysis, `extract_exports` reads the module's top-level
   fn/struct/enum/model names out of `analysis.scopes` (root scope), so a
   dependent module's `mse_loss` symbol gets the real
   `Type::Function { params, ret, .. }` of the stdlib definition.

The single-file entry points (`nsl check`, `crates/nsl-cli/src/pipeline.rs`,
`shape_debug.rs`) pass an empty import map, so imported names there are
`Unknown` and calls through them are unchecked. Tests that need stdlib
(`train_config_contract_gate.rs`) set `NSL_STDLIB_PATH` explicitly.

## Ownership / effects / other analyses

### Effects (`effects.rs`, M51)

Three phases inside `check_module`. Registration: `check_fn_def` collects
the names of callees encountered while checking the body
(`current_callees`, pushed in `check_call`), computes local effects as the
union of `classify_builtin_effects(callee)` over them, and calls
`EffectChecker::register_function(name, local_effects, callees)`.
Decorators mark intent: `@pure` → `mark_pure`, `@deterministic` →
`mark_deterministic`, `@checkpoint` → `mark_checkpointed` /
`mark_checkpointed_with_policy(CheckpointPolicy::{Full, Selective})`
(policy strings `selective_postnorm` and `custom` are refused), `@paged_kv`
on a model → `mark_paged_kv_model`. Propagation: fixed-point union over the
call graph, unknown callees classified as builtins. Validation: `@pure`
must be `PURE`; `@deterministic` must lack `RANDOM` unless the function has
an explicit `Rng` parameter (`has_explicit_rng`); `@checkpoint` requires
purity because recompute must reproduce the forward. The resulting
`checkpoint_policies` map and `paged_kv_models` set are exported on
`AnalysisResult` for the CLI loader and codegen's flash-attention backward
refusal.

### Ownership / linear types (`ownership.rs`, `ownership_walker.rs`, M38a)

Only runs when `analyze_with_imports(.., linear_types = true)`, i.e.
`nsl check --linear-types` / `nsl run --linear-types`. `analyze_ownership`
walks each top-level `FnDef`, registers tensor-typed parameters and
let-bindings with an `OwnershipChecker` (`register_binding`; `@shared`
let-bindings are refcounted and may be used many times), and translates
uses into `consume` / `use_binding` / `register_borrow` / `release_borrow`
calls. `OwnershipState` tracks consumed/shared/borrowed; `snapshot` /
`restore` plus `check_branch_symmetry` / `check_multi_branch_symmetry`
enforce that every `if`/`match` arm consumes the same set; `enter_loop` /
`exit_loop` catch consumption inside a loop body; `check_unconsumed` flags
leftovers. The `Type::Borrow` rules in `is_assignable` (auto-borrow on
call, `&T → T` allowed for reads) are the type-level half; the walker is
the flow-sensitive half. Output is `FunctionOwnershipInfo { linear_params,
shared_params }` per function, which `crates/nsl-codegen/src/func.rs` reads
to decide parameter ownership at the ABI.

`ownership_autodiff.rs` is a small table (`classify_backward_access`)
used by codegen's tape planner to know which tensors the backward of each
op reads; it lives here so the classification sits next to the ownership
rules.

### Agents (`agent.rs`, M56)

`agent Foo:` declarations are gated behind `--linear-types` (E0610, emitted
once after `collect_top_level_decls`). The rest of the pipeline runs
regardless so all agent errors show in one pass: `AgentRegistry`,
`extract_apgs` (action-port graphs), `detect_cycles` (E0603),
`check_device_compatibility` (E0607 cross-GPU, E0608 cross-device without
`@auto_device_transfer`), `check_fan_out`,
`check_cross_agent_field_access` (E0601 exclusive field) and
`check_cross_agent_mutation` (E0602). Error codes are embedded in the
message text; `nsl_errors::Diagnostic` has no code field.

### Determinism and NaN analysis (`determinism.rs`, `nan_analysis.rs`)

Standalone AST walkers with their own `diagnostics` vectors, run by
`crates/nsl-cli/src/commands/check.rs` under `--deterministic` and
`--nan-analysis` (each re-parses the file). `DeterminismChecker::scan_module`
classifies ops (`classify_op`, `deterministic_variant`) and tracks explicit
RNG state; `NanAnalyzer::analyze_module` flags `log`/`sqrt`/division on
unconstrained values using `ValueConstraint`.

## Interface to codegen

Codegen never re-runs semantic analysis; it consumes `AnalysisResult`
fields that the CLI threads through `CompileOptions` or passes directly.
Grepping `nsl_semantic::` in `crates/nsl-codegen/src` and
`crates/nsl-cli/src` gives this surface:

| Consumed | Where it goes |
|---|---|
| `checker::TypeMap` | `Compiler.type_map: &TypeMap` (`compiler/mod.rs`); read by `expr/*`, `stmt.rs`, `ownership.rs`, `memory_planner.rs`, `dynamic_shapes.rs`, `profiling/*`, `source_ad.rs`. |
| `types::{Type, Dim, Shape, DType, Device}` | Everywhere codegen inspects a type; `types::display_type` for messages; `Dim::Bounded`/`Symbolic`/`Computed` in `dynamic_shapes.rs`. |
| `train_config::resolve_train_config`, `TrainConfigPurpose`, `ResolvedTrainConfig` | `stmt.rs` train lowering (standard and pipelined) — the backstop call. |
| `optim_config::resolve_optim_config`, `ResolvedScheduler`, `OptimizerKind`, `VALID_ROLES` | `stmt.rs` optimizer/scheduler emission; `param_roles.rs`. |
| `AnalysisResult.weight_index_map` (`export::WeightIndexMap`) | `CompileOptions.weights.index_map` → `Compiler.weight_index_map`; lowers `self.W` in `@export` methods to an indexed load. |
| `AnalysisResult.ownership_info` | `func.rs` via the features struct. |
| `effects::CheckpointPolicy`, `AnalysisResult.checkpoint_policies`, `paged_kv_models` | CLI loader → `WengertExtractor::with_checkpoint_policy`; flash-attention backward refusal. |
| `csha::CshaConfig`, `cftp::{FusedCeConfig, PcaConfig, FaseConfig, FaseMode, PcaStrategy, FusedCeDtypeHint}`, `cpkd::FusedKlCeConfig`, `cep::{CepPruneConfig, CepSearchConfig, ..}`, `wrga::{WrgaConfig, FreezeConfig, AdapterConfig, AdapterKind}`, `cpdt::*` | `CompileOptions` side-channels populated by `crates/nsl-cli/src/loader.rs` and `pipeline.rs`; codegen's per-feature drivers read them. |
| `cftp::validate_fase_decorator`, `validate_pca_decorator`, `cep::validate_cep_*` | Called again from codegen / CLI (one-resolver pattern). |
| `decorator_registry::KNOWN_DECORATORS` | `crates/nsl-codegen/src/activation.rs` (decorator activation contract). |
| `scope::ScopeMap`, `AnalysisResult.scopes` | Loader `extract_exports`; `ScopeMap::new()` in codegen's `profiling/walker.rs` and `entry_points.rs`. |
| `nsl_semantic::analyze` / `analyze_with_imports` | CLI (`loader.rs`, `pipeline.rs`, `commands/check.rs`, `shape_debug.rs`); in codegen, `lib.rs`, `compiler/entry_points.rs`, `profiling/walker.rs`, `calibration/binary_codegen.rs`, `test_helpers.rs`. |

Everything in the table is stable in the sense that codegen matches on it
directly; there is no facade. Adding a field to `AnalysisResult` means
also threading it through `loader.rs` (multi-module builds) and
`pipeline.rs` (single-file builds), or it will silently be dropped on one
path.

## Invariants

- `check_expr` inserts into `type_map` for every expression it visits, at
  exactly one site (`checker/expr.rs`). Sub-checkers that need an operand's
  type read it from `type_map` after calling `check_expr` (e.g. `check_call`
  reading `object.id`), so ordering inside a checker function matters.
- `Type::Error` is only returned after a diagnostic has been pushed;
  `Type::Unknown` never comes with a diagnostic. Every checker path tests
  `is_indeterminate()` first and propagates without reporting.
- Rank 0 `Shape` means unknown; no rule may treat it as a scalar.
- `is_builtin` is set only by `register_builtins` and cleared by
  `declare_symbol` on user redeclaration; `check_call`'s by-name special
  cases must key on it, never on `Span::DUMMY`.
- `TRAIN_CONFIG_KEYS`, each `OptimizerKind::accepted_kwargs` table, and
  `CALLBACK_NAMES` are the *complete* accepted sets; a key accepted here
  must be consumed by codegen in the same change. Unknown, duplicate,
  positional and non-literal entries are errors, never defaults.
- `resolve_train_config` / `resolve_optim_config` are pure and are called
  by both the checker and codegen; they must stay free of checker state.
- Every decorator name a checker branches on (`dname == "..."`) has a row
  in `KNOWN_DECORATORS` whose `read_by` file mentions the name as a string
  literal (both directions gated in `crates/nsl-semantic/tests/decorator_namespace_gate.rs`).
  A name in `UNIMPLEMENTED_DECORATORS` must not also be known.
- Exactly one `@cpdt` decorator per program (`cpdt_decorator_span`).
- Import processing precedes declaration pre-registration inside
  `collect_top_level_decls`, and pre-registration precedes body checking, so
  forward references and imported types are always visible.
- `AnalysisResult.diagnostics` is the union of every stage; the driver never
  stops early. The CLI decides exit status by counting `Level::Error`.
- The crate does no file I/O and reads only one environment variable
  (`NSL_ALLOW_UNKNOWN_DECORATORS`). Stdlib paths are the CLI's concern.

## Tests and gates

There are ~500 `#[test]` functions in the crate (497 by grep at time of
writing): 130 in `crates/nsl-semantic/src/checker/tests.rs`, the rest as
`#[cfg(test)] mod tests` blocks inside individual modules (`shapes.rs`,
`shape_algebra.rs`, `scope.rs`, `train_config.rs`, `optim_config.rs`,
`decorator_registry.rs`, `effects.rs`, `ownership.rs`, `agent.rs`, each
feature validator) and 76 across the 14 files in
`crates/nsl-semantic/tests/`.

The dominant style is *snippet tests*: `checker/tests.rs::check_source(src)
-> Vec<Diagnostic>` lexes, parses and analyses an inline NSL string and the
test asserts on diagnostic messages. `train_fixture` / `train_config_errors`
and `train_sections_fixture` wrap a header or section list in a minimal
model/train program for the contract tests (`train_unknown_key_is_refused_
at_check_time`, `train_checkpoint_pairing_is_enforced_both_directions`,
`scheduler_typo_kwarg_is_refused_not_defaulted`, …). Integration tests in
`tests/` use `nsl_semantic::analyze` on parsed source and inspect
`AnalysisResult` fields; `tensor_method_result_typing.rs` shows how to read
a specific expression's type out of `type_map` by capturing `value.id`
before analysis.

There are no snapshot or golden-file tests in this crate. The gates that
pin semantic behaviour are:

- `crates/nsl-semantic/tests/decorator_namespace_gate.rs` — registry ↔ tree drift, both
  directions, by reading the checker sources.
- `train_config.rs::the_expected_keys_label_lists_every_accepted_key` —
  key table ↔ diagnostic label drift.
- `crates/nsl-semantic/tests/tensor_method_result_typing.rs` — the `check_member_access` method
  table must keep chain links `Tensor`-typed.
- `crates/nsl-semantic/tests/cpdt_decorator_single_constraint.rs`,
  `crates/nsl-semantic/tests/checkpoint_policy_parsing.rs`, `crates/nsl-semantic/tests/paged_kv_models_tracker.rs`,
  `crates/nsl-semantic/tests/csha_decorator_binding.rs` — side-table capture.
- CLI end-to-end gates in `crates/nsl-cli/tests/` that spawn the `nsl`
  binary: `train_config_contract_gate.rs` (typo'd key refused at `nsl
  check` and on the run path, valid header still builds — sets
  `NSL_STDLIB_PATH`), `train_config_resume_gate.rs`,
  `train_checkpoint_gate.rs`, `shape_debug.rs` (`--shapes` trace format),
  `m56_linear_types_run.rs` (`nsl run --linear-types` accepts agents
  without E0610), `cpdt_decorator_activation_gate.rs`,
  `fase_decorator_activation_gate.rs`, `train_sections_e2e.rs`.

Run them with:

```
cargo test -p nsl-semantic                              # unit + tests/
cargo test -p nsl-semantic --test decorator_namespace_gate
cargo test -p nsl-cli --test train_config_contract_gate # needs stdlib/ in repo
cargo bench -p nsl-semantic                             # benches/analyze.rs
```

CI (`.github/workflows/ci.yml`) additionally builds the `analyze` bench
under `cargo clippy -D warnings`, so bench code must stay lint-clean.

## Where to add a new X

### A new builtin function or tensor method signature

1. Free function: add a `def("name", Type::Function { params, ret, effect:
   Effect::Inferred })` entry in `builtins::register_builtins`
   (`builtins.rs`). Use `Type::Unknown` in `params` if arity must be
   flexible. If the result type depends on the arguments (shape from a list
   literal, "returns the tensor it was given"), add a by-name case inside
   the `!user_declared` block of `check_call` (`checker/ops.rs`) — keep it
   inside that block so a user redeclaration wins.
2. Tensor method: add an arm to the `Type::Tensor | Param | Buffer` match in
   `check_member_access` (`checker/ops.rs`) returning a `Type::Function`.
   Use `obj_ty.clone()` as the return for shape-preserving methods and
   `Self::view_result_type(&obj_ty)` for shape-changing ones. Never let a
   real method fall to `_ => Type::Unknown`. Arity must match the codegen
   dispatch in `crates/nsl-codegen/src/expr/advanced.rs`.
3. If the method changes shape statically, add a case to the "Tensor method
   shape inference" block in `check_call`, next to `reshape`/`expand`/
   `transpose`.
4. Effects: if the builtin performs IO/RNG/mutation, add it to
   `classify_builtin_effects` (`effects.rs`) so `@pure`/`@deterministic`
   validation sees it.
5. Tests: a snippet test in `checker/tests.rs`, and for tensor methods an
   assertion in `crates/nsl-semantic/tests/tensor_method_result_typing.rs` that the chain link
   is `Type::Tensor`.

### A new `train(...)` config key

1. Add the key to `TRAIN_CONFIG_KEYS` and to `EXPECTED_KEYS_LABEL` in
   `train_config.rs` (the unit test fails if you forget the label).
2. Add a field to `ResolvedTrainConfig` with its default and doc-comment the
   validated range.
3. Add a match arm in `resolve_train_config` using `fold_int_literal` /
   `fold_numeric_literal` or a string-literal match; refuse non-literals and
   out-of-range values with a labelled `Diagnostic::error`. If the key only
   makes sense with another key, add a pairing rule judged on `seen`, not
   on values.
4. Consume the new field in `crates/nsl-codegen/src/stmt.rs` at the
   `resolve_train_config` call sites (standard and pipelined lowering) in
   the same change; an accepted-but-unread key is the bug this module exists
   to prevent.
5. Tests: add a case to the "Training Configuration Contract" section of
   `checker/tests.rs` (both a refusal and an acceptance), extend
   `train_valid_full_header_produces_no_config_errors` to write the key,
   and if the key changes runtime behaviour add a CLI gate next to
   `crates/nsl-cli/tests/train_config_contract_gate.rs`.

### A new optimizer or scheduler kind

Optimizer:

1. Add a variant to `OptimizerKind` (`optim_config.rs`) and update
   `as_str`, `display`, `parse`, `accepted_kwargs` (exactly the stdlib step
   function's parameters, plus `no_decay` if it applies weight decay) and
   `OPTIMIZER_NAMES_LABEL`.
2. Add the stdlib module `stdlib/nsl/optim/<name>.nsl` and teach
   `inject_train_block_imports` in `crates/nsl-cli/src/loader.rs` (and the
   test-only `crates/nsl-codegen/src/stdlib_loader.rs`) the name → module
   mapping so the step function is auto-imported.
3. Fill `ResolvedOptimizer` defaults in `resolve_optimizer_expr` and wire the
   lowering in `crates/nsl-codegen/src/stmt.rs` (step-fn mangling, state
   buffer count) keyed on `OptimizerKind::as_str`.
4. Tests: kwarg refusal/acceptance in the "Optimizer/scheduler/callbacks
   section contract" section of `checker/tests.rs`; an end-to-end gate like
   `crates/nsl-cli/tests/muon_optimizer_gate.rs`.

Scheduler:

1. Add a `ResolvedScheduler` variant with its kwargs in stdlib signature
   order, its `fn_name` (must match the function in
   `stdlib/nsl/optim/schedulers.nsl`), both spellings in
   `canonical_scheduler_name`, and `SCHEDULER_NAMES_LABEL`.
2. Add the per-scheduler kwarg table and defaults in the scheduler branch of
   `resolve_optim_config`.
3. The lowering passes the variant's fields positionally after
   `(base_lr, step)`; confirm the order in `stmt.rs`.

### A new shape rule for a tensor op

1. If it is a binary operator: extend `check_arithmetic` or
   `check_matmul_op` in `checker/ops.rs`, keeping the rule itself in
   `shapes.rs` as a pure `fn(&Shape, &Shape, Span) -> Result<Shape,
   Diagnostic>` so it can be unit-tested without a checker.
2. If it is a call or method: add a case in `check_call`'s tensor-method
   block or the by-name builtin block, computing the result `Shape` from
   `extract_shape_from_args` and the receiver's shape. Return an unknown
   shape (not `Type::Unknown`) when the arguments are non-literal.
3. If the rule is an arithmetic fact over symbolic dims (element-count
   equality, divisibility for head splits, bounds), express it with
   `DimExpr` and `ShapeAlgebraSolver` — `assert_bound`/`assert_divisible`
   then `prove_eq_normalized`/`prove_divisible`. Prefer a warning when the
   runtime still checks, an error only when it does not.
4. Always handle rank 0 (unknown) by returning unknown, and unify with
   `shapes::unify_dim` rather than comparing `Dim` values directly so
   wildcards, named and bounded dims keep working.
5. Tests: pure rules in `shapes.rs`'s or `shape_algebra.rs`'s `mod tests`
   (build shapes with `Dim::Concrete` and `make_sym`); end-to-end typing via
   `crates/nsl-semantic/tests/tensor_method_result_typing.rs`-style `type_map` lookup; if the
   trace should show it, `crates/nsl-cli/tests/shape_debug.rs`.

### A new decorator

1. Write `validate_<name>_decorator(deco, &resolve, &mut diagnostics) ->
   Option<Config>` in a new or existing feature module (pattern:
   `csha.rs`); keyword args only, literal values only, unknown args refused.
2. Dispatch it from the `StmtKind::Decorated` arm in `checker/stmt.rs`
   (`if dname == "<name>" { ... }`, checking the host statement kind) or from
   `check_model_def` in `checker/model.rs` for member-level decorators.
3. Add a `k("<name>", "crates/nsl-semantic/src/checker/stmt.rs")` row to
   `KNOWN_DECORATORS`; the drift gate fails otherwise. Remove the name from
   `UNIMPLEMENTED_DECORATORS` if it was a documented ghost.
4. If codegen needs the config, add a `Vec<Config>` field to `TypeChecker`
   and `AnalysisResult`, move it in `analyze_with_imports`, and thread it
   through `crates/nsl-cli/src/loader.rs` and `pipeline.rs` into
   `CompileOptions`.
5. Tests: validator unit tests in the module, a capture test in `tests/`
   (pattern: `crates/nsl-semantic/tests/csha_decorator_binding.rs`), and if activation is
   observable a CLI gate like `cpdt_decorator_activation_gate.rs`.

### A new diagnostic

1. Build it with `nsl_errors::Diagnostic::error(msg)` / `warning(msg)` and
   attach the narrowest useful span with `.with_label(span, "...")`
   (`with_secondary_label`, `with_note` exist). Label the offending token,
   not the whole block; say what was expected.
2. Push it on `self.diagnostics` (checker) or the pass's own vector, then
   return `Type::Error` from a typing path so the poison suppresses
   follow-on errors. Do not report on `is_indeterminate()` inputs.
3. There is no error-code field; the agent pass embeds `E06xx:` in the
   message text. Follow that only within `agent.rs`.
4. Warnings versus errors: an error means codegen would produce a program
   that differs from the source; a warning means the runtime still checks.
   Demotions via environment variables exist only for
   `NSL_ALLOW_UNKNOWN_DECORATORS`.
5. Test it with `check_source` in `checker/tests.rs`, asserting on a
   distinctive substring of the message, and — if users hit it from the CLI
   — a `crates/nsl-cli/tests/*_gate.rs` that asserts the text on stderr.
