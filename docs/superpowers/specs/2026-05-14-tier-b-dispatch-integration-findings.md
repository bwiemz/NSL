# Tier B Dispatch — V-dispatch-integration Findings

**Date:** 2026-05-14
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Spec:** [`2026-05-14-pca-tier-b-dispatch-design.md`](2026-05-14-pca-tier-b-dispatch-design.md) §5
**Plan:** [`2026-05-14-pca-tier-b-dispatch-implementation.md`](../plans/2026-05-14-pca-tier-b-dispatch-implementation.md) Task D-1
**Status:** verification only — no source edits.

## Verification protocol

Enumerated every direct caller of `synthesize_flash_attention_ptx_v2` and every direct caller of its `*_selected*` wrappers in `crates/nsl-codegen/src` via `git grep`. Classified each production caller as **α / β / γ** (per §5.2) by reading the enclosing function and tracing upward to ask: *at this call site, is the runtime `seq_len` (the runtime row count of the Q tensor) in scope as a value, in scope as a type parameter, or unavailable?* Tests (`#[cfg(test)] mod tests` blocks) were enumerated but excluded from the α/β/γ tally because they don't run during a user compile.

## Grep commands run (reproducibility)

```
$ git grep -n 'synthesize_flash_attention_ptx_v2[^_]' crates/nsl-codegen/src
crates/nsl-codegen/src/flash_attention_selector.rs:9:    flash_attention_kernel_name_v2, shared_mem_bytes_v2, synthesize_flash_attention_ptx_v2,
crates/nsl-codegen/src/flash_attention_selector.rs:52:    synthesize_flash_attention_ptx_v2(config)
crates/nsl-codegen/src/flash_attention_v2/mod.rs:25:pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
crates/nsl-codegen/src/flash_attention_v2/mod.rs:32:/// `synthesize_flash_attention_ptx_v2` (no-op guarantee, spec §3.4.6).
crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2045:            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2072:            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs:30:/// (used by the existing `synthesize_flash_attention_ptx_v2` path).
crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs:154:/// Runtime validation called by `synthesize_flash_attention_ptx_v2`.

$ git grep -n 'synthesize_flash_attention_ptx_selected' crates/nsl-codegen/src
crates/nsl-codegen/src/compiler/kernel.rs:750:            crate::flash_attention_selector::synthesize_flash_attention_ptx_selected_with_diag(
crates/nsl-codegen/src/compiler/kernel.rs:1050:                let ptx_bytes = crate::flash_attention_selector::synthesize_flash_attention_ptx_selected_with_diag(
crates/nsl-codegen/src/compiler/kernel.rs:1173:            let ptx_bytes = crate::flash_attention_selector::synthesize_flash_attention_ptx_selected_with_diag(
crates/nsl-codegen/src/flash_attention_selector.rs:47:pub fn synthesize_flash_attention_ptx_selected_with_diag(
crates/nsl-codegen/src/flash_attention_selector.rs:76:pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
crates/nsl-codegen/src/flash_attention_selector.rs:77:    synthesize_flash_attention_ptx_selected_with_diag(config, &mut Vec::new())

$ git grep -n 'synthesize_flash_attention_ptx_selected\b' crates/nsl-codegen/src
crates/nsl-codegen/src/flash_attention_selector.rs:76:pub fn synthesize_flash_attention_ptx_selected(config: &FlashAttentionConfig) -> Vec<u8> {
```

The no-diag `synthesize_flash_attention_ptx_selected` shim has zero in-tree callers (it's a backwards-compat API surface, unused).

## Caller classification

### Layer 0 — direct callers of `synthesize_flash_attention_ptx_v2`

| Call site | Production? | Class | Justification |
|-----------|-------------|-------|---------------|
| `flash_attention_selector.rs:52` | yes (internal route) | — | Forwards from `synthesize_flash_attention_ptx_selected_with_diag`; classified one level up. |
| `phases/forward/csha_hooks.rs:2045` | no (test) | — | Inside `#[cfg(test)] mod tests` (the module opens at `csha_hooks.rs:1378`). Test is `a4_rope_epilogue_placed_before_attention_body`. |
| `phases/forward/csha_hooks.rs:2072` | no (test) | — | Same `#[cfg(test)] mod tests` block. Test is `a4_rope_k_epilogue_placed_before_s_compute_in_full_ptx`. |

Verified the `#[cfg(test)]` boundary with `git grep -n '^#\[cfg(test)\]' crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs` → returns `1378:#[cfg(test)]` followed by `1379:mod tests {`. Lines 2045 and 2072 both fall inside that gate.

### Layer 1 — production callers of the `_selected_with_diag` wrapper

| Call site | Enclosing fn | Class | Justification |
|-----------|--------------|-------|---------------|
| `compiler/kernel.rs:750` | `maybe_synthesize_csha_training_ptx(&mut self, stmts: &[Stmt])` (def @ line 657) | **β** | Operates on the AST `&[Stmt]` slice. Builds `training_config` from `flash_attention_context.config` + a CSHA preset (`csha_extras` at line 685, `backward_block_kv: i64 = 32` at line 709, `backward_block_q: i64 = 32` at line 739). No `seq_len` is in scope: the enclosing fn signature takes only `&[Stmt]` and `&mut self`; no caller upstream produces or threads through a runtime row count. The comment at lines 710–730 explicitly acknowledges seq_len is unknown at codegen time — see "When `block_q > seq_len` (the common case for short toy programs — e2e smoke runs seq_len=32 with the default block_q=64 from the forward inference config)" and "the runtime dispatcher reads sequence length from the tensor shape, not the PTX-baked block_kv". |
| `compiler/kernel.rs:1050` | `compile_flash_attention_kernels(&mut self, stmts: &[Stmt])` (def @ line 503) — autotune variant loop | **β** | Loop iterates over `(block_q, block_kv)` Cartesian product produced by `crate::autotune::cartesian_product(&tune_params)` (line 997). Each iteration constructs a `test_config` (line 1012) from `block_q`, `block_kv`, `default_head_dim` (resolved from `@flash_attention(head_dim=N)` decorator at line 969), `causal`/`paged`/`rope_q`/`rope_style`/`gqa_group_size` (parsed from decorators in the loop @ lines 875–949), and `gpu_sm` (from `compile_options.target`). `segment_masked: false` is hard-coded (line 1023). No `seq_len` produced or threaded — same reason: this is AOT compilation of a decorator-annotated `kernel` block; the Q/K/V tensor shapes are bound at the FFI boundary at runtime, not at this codegen pass. |
| `compiler/kernel.rs:1173` | `compile_flash_attention_kernels(&mut self, stmts: &[Stmt])` — single-config fallback (no `@autotune`) | **β** | Same enclosing fn as 1050. Hard-codes `block_q: 64`, `block_kv: 64`, `head_dim: default_head_dim`, `segment_masked: false` (lines 1158–1170). Identical seq_len availability story: AOT, no runtime tensor binding has occurred. |

### Detailed trace per Layer-1 caller

#### Trace 1 — `compiler/kernel.rs:750`

- **L0:** `kernel.rs:750` invokes `synthesize_flash_attention_ptx_selected_with_diag(&training_config, &mut diags)`.
- **L1:** Enclosing fn `maybe_synthesize_csha_training_ptx` (line 657) signature: `fn maybe_synthesize_csha_training_ptx(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError>`. Inputs: `self` (codegen state) and the AST. No `seq_len: i64` or `seq_len: u32` parameter; no extraction from `stmts` in this fn. The closest dimensional resolve is `resolve_csha_d_model_from_stmts` (line 677), which inspects layer-decl shapes for `d_model`, NOT for seq.
- **L2:** Callers of `maybe_synthesize_csha_training_ptx`: I did not need to walk farther — `seq_len` is not a parameter at L1, and L1's body explicitly comments (lines 710–730) that runtime seq_len is unknown at this stage and the dispatcher computes `grid_x` at launch time. **Verdict: β.**

#### Trace 2 — `compiler/kernel.rs:1050`

- **L0:** `kernel.rs:1050` synthesizes PTX for each `(block_q, block_kv)` autotune variant.
- **L1:** Enclosing fn `compile_flash_attention_kernels(&mut self, stmts: &[Stmt])` (line 503). Same `&[Stmt]` shape — AST input, no runtime dim. The loop body builds `test_config` (line 1012) from autotune Cartesian product values plus decorator-parsed flags. `segment_masked: false` is hard-coded; head_dim derives from `@flash_attention(head_dim=N)` (line 969); no seq_len input.
- **L2:** Callers of `compile_flash_attention_kernels` are higher-level codegen entry points (e.g., the module-level `compile_kernels`, line 94, which also takes `&[Stmt]`). No runtime tensor shape is in scope at this layer either — same AOT compilation context. **Verdict: β.**

#### Trace 3 — `compiler/kernel.rs:1173`

- **L0:** `kernel.rs:1173` is the `else` branch of the autotune check (line 971) — the no-`@autotune` single-config path.
- **L1:** Same enclosing fn as Trace 2 (`compile_flash_attention_kernels`). Hard-codes `block_q: 64, block_kv: 64, segment_masked: false` (lines 1158–1170). Same β reasoning. **Verdict: β.**

## Layer-2 check — could seq_len be threaded down?

I considered whether seq_len could become available at the call sites if a caller chose to thread it through. Two structural facts argue **no, not without a separate design effort**:

1. **The codegen passes operate on `&[Stmt]` (AST) before any tensor instantiation.** `compile_flash_attention_kernels` and `maybe_synthesize_csha_training_ptx` run during module compilation — at the same point that PTX bytes are embedded into the `.rodata` of the Cranelift module. The host-side `nsl_flash_attention_csha_*` FFI dispatcher reads Q's shape at *call* time and recomputes `grid_x` then (see kernel.rs:726, "the runtime `grid_x = (seq_len + block_q - 1) / block_q` calculation in `nsl_flash_attention_csha_*`"). PTX is shape-agnostic by construction.
2. **`FlashAttentionConfig` is the compile-time-constant struct that names the PTX kernel.** `flash_attention_kernel_name_v2(config)` mangles `block_q`/`block_kv`/`head_dim`/CSHA flags into the symbol name, which is then `cuModuleGetFunction`'d at launch time. Adding `seq_len` to the config (option 3) would either change the kernel name per seq_len (one PTX blob per sequence length — multiplicative explosion of PTX bytes embedded), or seq_len would be a config field deliberately ignored by the kernel-name mangler (which the §5.3 "type-system honesty" criterion warns against).

Neither makes option 3 (γ-shaped) attractive, and option 4 (synthesizer arg) has no production caller that can supply the arg.

## Tally

- **α** (seq_len in scope at call time): **0**
- **β** (seq_len not available — early-compilation context): **3** (`kernel.rs:750, 1050, 1173`)
- **γ** (seq_len exists as a type parameter, not a value): **0**
- Test-only callers excluded from tally: 2 (`csha_hooks.rs:2045, 2072`)

## Outcome decision

**Option 2 — sparsity-only collapsed heuristic.**

Per spec §5.2: "Mostly β: option 2 (sparsity-only collapse — drop seq_len gate). D-3 implementation simplifies; floor derivation in D-2 still useful as future v2 input but doesn't gate dispatch."

All 3 production callers fall in β. There is no α population to special-case, so the §5.2 mixed-α/β branch ("option 4 for α callers; β callers stay no-op") does not apply — every production caller is β. The minimum-viable dispatch heuristic at D-3 will therefore reduce to:

```rust
pub fn should_emit_tier_b(config: &FlashAttentionConfig) -> bool {
    config.segment_masked
}
```

(or equivalent — D-3 decides the final signature and any additional sparsity gates).

### Justification

1. **Type-system honesty (spec §5.3):** seq_len is runtime-variable and intentionally absent from `FlashAttentionConfig` today. The struct is the kernel-name mangling key; smuggling seq_len in would either bloat PTX-embed counts (one blob per length bucket) or violate the invariant that the struct names the kernel.
2. **No production source for seq_len at the call sites:** all 3 callers operate on `&[Stmt]` AST input during AOT module compilation. seq_len is bound only at the host-side FFI boundary at launch time, in a different layer of the stack entirely. The kernel.rs:710–730 comment block documents this design choice explicitly.
3. **Option 4 ruled out:** adding a `seq_len: u32` arg to `synthesize_flash_attention_ptx_v2` would require every production caller to either pass a sentinel ("unknown") or pass a synthetic worst-case value. The first defeats the gate (Tier B would never fire under the sentinel); the second is unsound (codegen would emit Tier B against a value that may bear no relation to runtime input).
4. **D-2 floor derivation retains value as a future v2 input:** the floor measurement informs whether Tier B is profitable at low seq_len. If a future planner-side dispatch layer (out of scope for this milestone — see V-M35.2a "planner-side dispatch is its own downstream spec" in MEMORY.md) gains access to a static seq_len bound (e.g. via shape inference on bound dataset configs), the floor value documents the lower-seq_len cutoff that layer should enforce. Today, it goes into a comment beside the heuristic.

## D-2 floor-scope conditional (per spec §6.5)

Per §6.5 bullet 2 ("If V-dispatch-integration surfaces (α) + (β): floor applies only to (α) callers. The (β) callers operate under sparsity-only collapsed heuristic"), this finding (β-only, zero α) is a degenerate case of that bullet: **the floor applies to zero current production callers**. D-2 should still run the measurement — both because the spec §6 sequencing prescribes it and because the floor value is the load-bearing artifact a future planner-side dispatch layer would consume.

Concretely for D-2:

- **Measurement is still useful:** sweep wall-time win % across seq_len ∈ {64, 128, 256, 512, 1024} on a representative segment-mask workload; record the curve shape.
- **Floor value is recorded as a "reserved for future v2" constant**, not wired into the D-3 heuristic.
- **D-2 findings doc cross-references this doc** as the reason its output is informational rather than gating.
- **D-3 heuristic does NOT consume the floor.** D-3's `should_emit_tier_b(config)` body reads only `config.segment_masked` (and any other purely-config-time sparsity gates the §5.2 collapsed-heuristic mode admits).

## Cross-references

- Design spec: [`2026-05-14-pca-tier-b-dispatch-design.md`](2026-05-14-pca-tier-b-dispatch-design.md) §5 (V-dispatch-integration), §6.5 (floor scope conditional).
- Plan: [`2026-05-14-pca-tier-b-dispatch-implementation.md`](../plans/2026-05-14-pca-tier-b-dispatch-implementation.md) Task D-1.
- Code touched (read-only): `crates/nsl-codegen/src/flash_attention_selector.rs:47-77`, `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:1378,2045,2072`, `crates/nsl-codegen/src/compiler/kernel.rs:503,657,750,1050,1173`.
- Background: `MEMORY.md` PCA Tier B section — "`should_emit_tier_b(config)` continues returning false today; planner-side dispatch is its own downstream spec."
