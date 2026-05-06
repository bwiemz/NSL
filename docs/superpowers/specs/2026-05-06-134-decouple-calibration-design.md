# #134: Decouple calibration harness from `compile_train_block`

**Issue:** [#134](https://github.com/bwiemz/NSL/issues/134) — Decouple calibration harness from `compile_train_block`
**Status:** Design (Q1–Q4 brainstormed, ready for plan)
**Date:** 2026-05-06
**Stacks on:** PRs #139 (WGGO Phase 2 IR/FFI infrastructure), #140 (`ObservationSet::BackwardGradients` semantics), #141 (`#[ignore]` message pinning)

---

## 1. Background

PR #132 shipped 29 of 32 tasks of WGGO Phase 2 backward-pass calibration. PRs #135–#139 closed the remaining IR/FFI infrastructure gaps. PR #140 fixed `ObservationSet::BackwardGradients(_) → needs_forward_pass=true`, the latent observation-framework bug that routed WGGO-only calibration through the simulated `build_sidecar_from_stub` path. PR #141 pinned the merge-gate test's `#[ignore]` message at #134 as the resolution path for the remaining architectural work.

The merge-gate end-to-end test (`crates/nsl-codegen/tests/wggo_backward_pipeline.rs::end_to_end_backward_subprocess_matches_analytical_reference`) remains `#[ignore]`'d. Running it after PR #140 surfaced **hop 6**: `emit_calibration_model_object` (`crates/nsl-codegen/src/calibration/binary_codegen.rs:1771-1779`) requires a non-empty `calibration_retention` list (AWQ projections) because it reads `channels` / `model_name` / `transpose_fields` from the first projection. WGGO-only flows have empty `calibration_retention` but populated `calibration_grad_retention`; the function rejects on the empty list before reaching the WGGO codegen path.

PR #139's end-to-end validation surfaced two more drift points beyond hop 6:

- The calibration-firing site inside `compile_train_block` (`crates/nsl-codegen/src/stmt.rs:3966-4046`) has drifted from `real_subprocess_entry` (`crates/nsl-codegen/src/calibration/binary_codegen.rs:62-258`). Path 2 (`real_subprocess_entry`) has WGGO-targets auto-derivation and `calibration_grad_retention` plumbing that PR #139 added; Path 1 (`compile_train_block`'s calibration block) does not.
- Calibration-only flows currently require a synthetic `train` block in fixtures (see `tests/fixtures/wggo_attention_mlp_real.nsl`) because `compile_and_calibrate` only fires the harness from inside `compile_train_block`.

These three manifestations — hop 6, Path 1/Path 2 drift, synthetic `train` block requirement — share a single root cause: the calibration entry points were written AWQ-first and have ossified that assumption across multiple sites. **#134 is the architectural cleanup that removes the ossification at its source rather than patching individual manifestations.**

## 2. Non-goals

- **Generalized calibration harness for hooks beyond AWQ + WGGO.** Future hooks (Fisher information, KL divergence, etc.) will likely need their own descriptor types. This spec converges AWQ + WGGO on a single canonical entry path; broader hook-system generalization is not part of this work.
- **`compile_main`'s reduced pipeline ergonomics.** Stdlib path resolution, `main` signature collision, train-DSL stdlib gaps — see §3 for explicit rejection rationale.
- **Bit-equivalent behavior change for AWQ.** AWQ's currently-shipped sidecar bytes are the merge gate, not a target for cleanup. Any change to AWQ's output is a regression unless explicitly justified via CHANGELOG-CALIBRATION (see §6).

## 3. Out-of-scope rejections

PR #139's end-to-end validation surfaced three gaps in `compile_main`'s reduced pipeline that PR #139 worked around by switching the WGGO test entry point to `real_subprocess_entry` (mirroring AWQ's pattern):

1. **Stdlib path resolution.** Test's CWD is `crates/nsl-codegen`, but `cwd/stdlib` doesn't exist there; loader needs `NSL_STDLIB_PATH` or a workspace-aware default.
2. **`main` signature collision.** `fn main()` user `() -> ()` vs C-ABI `(i32, i64) -> i32` after `declare_user_functions_with_linkage`.
3. **Train-DSL stdlib gaps.** `mse_loss` not in step-body scope; `nsl_optim_sgd__sgd_step` undefined unless full pipeline runs.

**Out of scope for #134.** These remain present in `compile_main`'s reduced pipeline after #134, but no caller exercises them — `compile_and_calibrate` calls `real_subprocess_entry` directly under (c-i) (see §4), bypassing the reduced pipeline for calibration; production callers don't hit the affected code paths. **Tracked separately as "Ergonomics: compile_main's reduced pipeline for non-train-block fixtures."** The gaps are *dormant*, not deleted; future code paths through the reduced pipeline would rediscover them. Out of scope for #134 because (c-i) makes them non-blocking, not because they're fixed.

This is the same discipline as the §10 module-orphan policy in PCA Tier B — temporary or dormant states need explicit framing about what they are and aren't, so future maintainers correctly read codebase state.

## 4. Convergence shape: (c-i)

Today there are two drifted entry paths:

- **Path 1 (production):** `compile_and_calibrate` → `compile_main` → `compile_train_block` → `run_harness_production`
- **Path 2 (test, post-PR-#139):** `real_subprocess_entry` (`binary_codegen.rs:62-258`), invoked directly by AWQ tests and (now) the WGGO merge-gate test

The convergence problem #134 solves is "make these one path." Three sub-options for how the merge happens — (c-i): Path 2 becomes canonical and Path 1's calibration block is deleted; (c-ii): Path 1 is promoted and Path 2 becomes a shim; (c-iii): a new shared function is extracted and both paths become thin shims. **(c-i) is pinned.**

### 4.1 (c-i) definition

> `real_subprocess_entry` becomes the canonical calibration entry-point implementation. `compile_and_calibrate` calls it directly when `calibration_data.is_some()`, after `compile_main` returns. The calibration block in `compile_train_block` (`stmt.rs:3966-4046`) is deleted; `compile_train_block` returns to compiling train blocks only. AWQ tests' existing call sites to `real_subprocess_entry` are unchanged. Hop 6's fix lands in `real_subprocess_entry`'s codegen path (see §5), where the union `calibration_retention ∪ calibration_grad_retention` is read and `emit_calibration_model_object` is called with both projection sets.

### 4.2 Why (c-i) over (c-ii) / (c-iii)

- **Path 2 has the WGGO-targets auto-derivation and `calibration_grad_retention` plumbing already.** Path 1 does not. Promoting Path 2 means less migration work than promoting Path 1.
- **Hop 6 lives in `binary_codegen.rs`, which is `real_subprocess_entry`'s home.** Fixing hop 6 in (c-i) is "modify the canonical implementation in place." Fixing it in (c-ii) is "extract the AWQ-only assumption from `binary_codegen.rs:1771-1779`, generalize it in the new orchestration code, then have the shim forward to the generalization." More moving parts.
- **(c-iii) leaves both paths in place as shims.** That preserves the architectural drift in a slightly different form (now both shims drift independently from the shared core). The convergence is what makes #134 worth doing; (c-iii) doesn't deliver it.

### 4.3 Architectural framing

The mental-model match: *calibration runs around the compiled code, not inside it.* `compile_train_block` should compile train blocks; `compile_main` should compile main; `compile_and_calibrate` should orchestrate. (c-i) puts calibration where it conceptually belongs — at the wrapper level, post-compilation — rather than embedded mid-compilation in `compile_train_block`.

## 5. Hop 6 generalization

### 5.1 Current code

```rust
// crates/nsl-codegen/src/calibration/binary_codegen.rs:1771-1779
let projections = compile_opts
    .calibration_retention
    .clone()
    .ok_or_else(|| HarnessError::Infrastructure {
        reason: "emit_calibration_model_object requires AWQ projections".into(),
    })?;
let first_projection = projections.first().ok_or_else(|| HarnessError::Infrastructure {
    reason: "emit_calibration_model_object requires a non-empty projection list".into(),
})?;
let channels = first_projection.weight_shape[1] as i64;
```

The function uses `first_projection` for `channels`, `model_name` (split before first `.`), and `transpose_fields`. WGGO targets carry equivalent metadata via `class_name` + `w_*_shape`.

### 5.2 Generalization

`emit_calibration_model_object` reads from the union `calibration_retention ∪ calibration_grad_retention`. Concretely: if `calibration_retention` is empty, fall back to `calibration_grad_retention`'s targets to derive `channels`, `model_name`, `transpose_fields`. Both populated is also valid (mixed AWQ + WGGO calibration); the union pulls metadata from whichever has projections covering the relevant layer.

The function signature is unchanged. The internal logic adds:

```rust
let projections: Vec<ProjectionMeta> = match (
    compile_opts.calibration_retention.as_ref(),
    compile_opts.calibration_grad_retention.as_ref(),
) {
    (Some(retn), _) if !retn.is_empty() => retn.clone(),
    (_, Some(grads)) if !grads.is_empty() => grads.iter().map(grad_target_to_projection_meta).collect(),
    _ => return Err(HarnessError::Infrastructure {
        reason: "emit_calibration_model_object requires either AWQ or WGGO projections".into(),
    }),
};
```

`grad_target_to_projection_meta` is a thin adapter: WGGO `WggoGradTarget` → the same `ProjectionMeta` shape AWQ uses. Both have `class_name` / `weight_shape`; the adapter unifies field naming.

**Adapter shape pinned:** `grad_target_to_projection_meta` emits **one** `ProjectionMeta` per target, using **`w_o_shape`** for the `weight_shape` field. Rationale: `emit_calibration_model_object` reads `channels = first_projection.weight_shape[1]` (output dimension); `w_o_shape[1]` is the corresponding output dimension for attention's output projection. This matches AWQ's semantic for layers where the output channel count drives the calibration model's per-channel structure. WGGO targets carry four weight shapes (`w_q_shape`, `w_k_shape`, `w_v_shape`, `w_o_shape`); picking `w_o_shape` is load-bearing for "AWQ byte-identical by inspection" — a different choice could shift the channels value on mixed AWQ + WGGO fixtures and trigger a snapshot regression diagnosed as a symptom (wrong channels) rather than the cause (wrong shape selection).

If a future caller needs per-W_* granularity, the adapter can be extended to emit four `ProjectionMeta` entries per target — but v1's hop 6 fix needs only one entry per layer to satisfy the existing `channels` / `model_name` / `transpose_fields` reads.

### 5.3 AWQ behavior under §5.2

For AWQ flows, `calibration_retention` is non-empty, so the `(Some(retn), _) if !retn.is_empty()` arm matches and the function reads exactly what it read before. **AWQ flow is byte-identical by inspection** — the new code path is entirely behind a condition AWQ never satisfies.

This is what makes §5 the lowest-risk logic change in the commit sequence (see §8): it unblocks WGGO without touching AWQ's code path.

## 6. AWQ regression discipline

The merge gate for #134 has two checks: artifact-level (compile-time output bytes) and output-level (runtime sidecar bytes). Both are enforced; under (c-i) specifically, output-level drift should be **zero by construction** because `compile_main` is unchanged and calibration fires post-`compile_main` against the same fixture in the same order.

### 6.1 Determinism precondition

Before the AWQ baseline can be captured, the sidecar must be byte-deterministic across runs. Three potential sources of non-determinism worth verifying:

- **`HashMap` iteration order in serialization paths.** `serde_json` preserves the input map's iteration order; `HashMap` source produces randomized output across runs. Fix: replace with `BTreeMap` at the serialization boundary.
- **Floating-point representation drift.** f32/f64 → JSON formatting depends on Rust toolchain version. Pin by using a deterministic formatter or a fixed string format.
- **Timestamp / environment fields.** "Created at" timestamps, compiler version strings — every run produces different bytes. Probably absent; verify by inspection.

**Verification protocol:**

```bash
# Run the snapshot test 10 times; verify all 10 produce identical output.
for i in {1..10}; do cargo test --package nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar; done

# Threading-variation check: serial vs parallel must produce the same output.
cargo test --package nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar -- --test-threads 1
cargo test --package nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar -- --test-threads 8
```

If output varies, locate the source and fix it before capturing the baseline. **Do not paper over non-determinism with a hash digest** — hashing only conceals the same underlying bug.

### 6.2 Snapshot mechanism

- **(A-2)** First-commit-of-#134 baseline capture (split per §8.1). Subsequent commits in the same PR must not change the snapshot.
- **(B-1)** Full sidecar JSON content as the snapshot body. Pretty-formatted via `serde_json::to_string_pretty` with sorted keys; diff-friendly; immediate diagnostic value when it changes.
- **(C-1)** `insta` snapshot test. Already used in `nsl-codegen` (`tests/snapshots/`). `cargo insta accept` workflow handles legitimate updates.
- **(D-1)** Standard `cargo test` trigger. Snapshot test runs as part of `nsl-codegen`'s CI test suite; no special trigger needed.

Snapshot file lives at `crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap`.

### 6.3 CHANGELOG-CALIBRATION + CI enforcement

- **Location:** `CHANGELOG-CALIBRATION.md` at repo root, dedicated to calibration-affecting changes.
- **Format:** dated entries with PR number, snapshot files affected, cause (e.g., "new hook added field X," "serialization format upgraded for Y"), and bit-equivalence evidence (e.g., "AWQ behavior unchanged; only WGGO field added").
- **Enforcement:** CI check on PRs modifying `crates/nsl-codegen/tests/snapshots/awq_*.snap` (or whichever path holds the snapshot file). The check verifies a corresponding entry exists in `CHANGELOG-CALIBRATION.md` referencing the PR number. Implementation: ~20-line shell script in `.github/workflows/`.

Convention-only enforcement decays. CI enforcement persists across team turnover. This is the same discipline as the merge-gate test's CI integration in PR #139.

For #134 specifically: the snapshot must NOT change. Any drift is evidence of an implementation bug, not an architectural design problem with (c-i).

### 6.4 Rejected alternative: dependency-free crate

Splitting the snapshot test into its own dependency-free crate so it can run in CI without rebuilding `nsl-codegen` was considered and rejected:

- The snapshot's input is the calibration pipeline's output, which requires `nsl-codegen` to compute. A dependency-free crate would either re-implement the pipeline (massive cost) or pre-compute the sidecar in CI and check against it (CI complexity for no test-isolation benefit).
- The snapshot test runs in `nsl-codegen`'s own test suite, paying no rebuild cost beyond what `nsl-codegen` PRs already incur. The "rebuild cost" argument doesn't hold for the test's actual call site.

Documented here so future maintainers don't reach for this alternative without understanding the previous reasoning. Same discipline as the §10 module-orphan policy in PCA Tier B and the deprecated-shim lifecycle in the calibration FFI spec.

## 7. Risk analysis under (c-i)

Two risk shapes:

- **(α) Compile-time output drift.** Bytes in the compiled artifact (`.o` files, linked binary) change because the firing order affects what's emitted into the calibration scaffolding. Detected at compile time by `awq_full_pipeline`'s existing behavioral checks; tightened by (c-i)'s artifact-level bit-equivalence check (see §6.1's existing test asserts).
- **(β) Runtime calibration output drift.** Bytes in the calibration sidecar JSON change because the firing order affects what AWQ observes during calibration. Detected by the new output-level snapshot test (§6.2).

Under (c-i) specifically:

- (α) is bounded to whatever `compile_main` itself produces, which (c-i) doesn't change. The wrapper-level firing affects nothing inside `compile_main`'s output.
- (β) is **zero by construction** — `compile_main` is unchanged and calibration fires post-`compile_main` against the same binary in the same order; only the wrapper-level orchestration differs.

If either regresses, the convergence implementation is incorrect, not the (c-i) decision. The merge gate is "both checks pass"; failure indicates a bug to fix, not a design rollback.

## 8. Commit sequence (Sequence C)

### 8.1 Sequence

1. **Commit 1a — Determinism verification.** Run AWQ sidecar generation 10× with varying thread counts and concurrency. If output is byte-identical across runs, this commit is a no-op (just the verification script in `scripts/verify-awq-determinism.sh` plus a one-line CI integration). If non-determinism is found, this commit's scope grows to include the fix — typically `HashMap` → `BTreeMap` at the serialization boundary. **Commit 1b cannot land until commit 1a verifies determinism.**
2. **Commit 1b — Baseline + CHANGELOG + CI.** Captures the AWQ baseline snapshot (§6.2), creates `CHANGELOG-CALIBRATION.md` with the initial entry referencing #134, lands the CI enforcement workflow (`.github/workflows/calibration-snapshot-changelog.yml`). Once 1a verifies determinism, 1b is mechanical.
3. **Commit 2 — Hop 6 fix.** §5.2 generalization in `emit_calibration_model_object`. AWQ unchanged by inspection (§5.3); WGGO unblocked at hop 6 specifically. Merge-gate test moves to next hop or passes; either way, snapshot must not change.
4. **Commit 3 — Wrapper-level firing replaces train-block firing (atomic move).** Single-commit move: deletes the calibration-firing call from `compile_train_block` (`stmt.rs:3966-4046`), adds an equivalent call in `compile_and_calibrate` post-`compile_main` that invokes `real_subprocess_entry` directly. **No parallel-firing intermediate state** — Sequence B's parallel-firing trick is rejected per §8.4 and is not introduced here. Snapshot must not change. AWQ tests still pass; merge-gate test still ignored.
5. **Commit 4 — Train-block calibration block cleanup (optional).** Deletes any orphaned helper code in `stmt.rs:~3966-4046` that was referenced only by the now-removed firing call. If commit 3's atomic move was sufficient (no orphaned helpers), **this commit is empty and is skipped** — verification matrix collapses to 5 rows.
6. **Commit 5 — Remove synthetic train block + un-`#[ignore]` merge-gate test.** Removes the synthetic `train` block from `tests/fixtures/wggo_attention_mlp_real.nsl` (it was a workaround for Path 1's `compile_train_block`-only firing; under (c-i) it's no longer load-bearing). Removes the `#[ignore]` attribute from `end_to_end_backward_subprocess_matches_analytical_reference`. Both changes in the same commit, because removing the train block IS the verification that Path 2 works without a train block. Snapshot remains unchanged; merge-gate test now passes; `#[ignore]` discipline (PR #141) honored to its resolution.

### 8.2 Why 1a/1b split

Determinism verification is investigative work whose outcome determines whether infrastructure capture is valid. Folding it into the same commit as the baseline capture risks landing infrastructure on an unverified foundation. Splitting:

- Makes commit 1b's "infrastructure landing" half truly safe — only reached if determinism is verified.
- Isolates the "fixing non-determinism" work as its own auditable commit if needed. Six months from now, "why is this `BTreeMap` here?" is answered by commit 1a's diff.
- Adds one commit to the PR; the cost is negligible against the audit-trail benefit.

### 8.3 Discipline-first ordering principle

The reasoning behind Sequence C generalizes: **when a PR introduces both regression discipline and behavioral changes, the discipline lands first.** Otherwise the behavioral changes are made without the discipline available to verify them, and any drift is detected only after the discipline catches up. This is the same principle as the §11 anti-treadmill rule in PCA Tier B applied to PR sequencing: regression discipline that's added last is regression discipline that didn't catch the changes that made it necessary.

Future PRs facing the same "should regression machinery be first or last?" question cite this principle.

### 8.4 Why not Sequence A or B

- **Sequence A (test-first but CI-last):** lands hop 6 *before* the CI infrastructure. Commit 2's "byte-identical-by-inspection" claim doesn't get verified by CI; only later commits do. Worse audit trail than C.
- **Sequence B (parallel firing as belt-and-suspenders):** doubles AWQ test execution time during the parallel-firing window and adds two-firing-paths complexity that has to be undone. The snapshot test under (B-1)+(C-1) already provides byte-equivalence verification with less mechanism. Belt-and-suspenders that costs more than it adds.

## 9. Verification matrix

Each commit must satisfy the columns marked `✓`; columns marked `—` are not applicable.

| Commit                                    | AWQ behavioral test | Snapshot byte-equiv | Determinism (10× repeat) | Merge-gate test         |
| ----------------------------------------- | ------------------- | ------------------- | ------------------------ | ----------------------- |
| 1a — Determinism verification             | —                   | —                   | ✓                        | `#[ignore]`             |
| 1b — Baseline + CHANGELOG + CI            | ✓                   | ✓ (capture)         | ✓                        | `#[ignore]`             |
| 2 — Hop 6 fix                             | ✓                   | ✓ (no change)       | ✓                        | `#[ignore]` (next-hop)  |
| 3 — Wrapper-level firing (atomic move)    | ✓                   | ✓ (no change)       | ✓                        | `#[ignore]`             |
| 4 — Train-block cleanup (skip if empty)   | ✓                   | ✓ (no change)       | ✓                        | `#[ignore]`             |
| 5 — Un-`#[ignore]` + remove synth-train   | ✓                   | ✓ (no change)       | ✓                        | **PASSES**              |

`AWQ behavioral test`: `cargo test -p nsl-codegen --test awq_full_pipeline` — 6/6 PASS.
`Snapshot byte-equiv`: `cargo insta test --check` — captured baseline matches.
`Determinism`: 10× repeat of snapshot test produces identical output.
`Merge-gate test`: `cargo test -p nsl-codegen --test wggo_backward_pipeline -- --include-ignored` — 1/1 PASS at commit 5.

## 10. Architectural debt remaining after #134

After #134 lands, the following is fixed:

- ✓ Calibration harness fires unconditionally when `calibration_data.is_some()`, regardless of `train` block presence
- ✓ Path 1 / Path 2 drift eliminated (Path 2 is canonical; Path 1 deleted)
- ✓ Hop 6 (AWQ-only `emit_calibration_model_object`) generalized
- ✓ Synthetic `train` block in `wggo_attention_mlp_real.nsl` no longer required
- ✓ Merge-gate test (`end_to_end_backward_subprocess_matches_analytical_reference`) un-`#[ignore]`'d and passing
- ✓ AWQ regression discipline in CI (snapshot test + CHANGELOG enforcement)

Remaining architectural debt (out of scope, tracked separately):

- **Reduced-pipeline ergonomics** (PR-#139's other gaps): stdlib path, `main` signature, train-DSL scope. Dormant under (c-i); see §3.
- **Generalized hook-system descriptor schema**: future hooks (Fisher info, KL divergence) will need their own descriptor types beyond AWQ + WGGO. Not in #134's scope; see §2.
- **Windows support for calibration.** Calibration is currently Linux/macOS-only by platform guard (`#[cfg(any(target_os = "linux", target_os = "macos"))]` in `crates/nsl-runtime/src/calibration/sidecar.rs`); the WGGO Phase 2 merge-gate completion spec's §6.4 atomicity caveats document the Windows divergence. Adding Windows is a separate scope item.

## 11. Verification plan (post-merge)

After #134's PR merges:

- [ ] CI green on supported platforms (Linux, macOS): `awq_full_pipeline` 6/6 PASS, `wggo_backward_pipeline` 1/1 PASS (no `#[ignore]`). Windows calibration support is not in #134's scope per §10.
- [ ] AWQ snapshot file (`awq_full_pipeline__awq_sidecar_baseline.snap`) byte-identical to commit 1b's capture
- [ ] `CHANGELOG-CALIBRATION.md` exists with #134's entry and no other entries
- [ ] Merge-gate test runs as standard `cargo test` without `--include-ignored`
- [ ] Issue #134 closed; merge-gate `#[ignore]` discipline (PR #141) honored to its resolution

## 12. Estimated scope

- **Spec design (this document):** ~0.5 day, completed
- **Implementation:** ~1.5 days for Sequence C (1a/1b/2/3/4/5), assuming determinism verification finds no major non-determinism source
- **Total:** ~2 days from brainstorm-start to PR-ready

If determinism verification surfaces non-trivial non-determinism (e.g., requires changing serialization fields across multiple modules), commit 1a's scope grows and the implementation budget extends to ~2.5 days.

---

## Appendix A: References

- Issue: [#134](https://github.com/bwiemz/NSL/issues/134)
- Stacks on: [PR #139](https://github.com/bwiemz/NSL/pull/139), [PR #140](https://github.com/bwiemz/NSL/pull/140), [PR #141](https://github.com/bwiemz/NSL/pull/141)
- Related specs:
  - `docs/superpowers/specs/2026-04-22-awq-real-subprocess-completion-design.md` — AWQ real-subprocess completion (PR #125)
  - `docs/superpowers/specs/2026-05-02-wggo-phase2-merge-gate-completion-design.md` — WGGO Phase 2 merge-gate completion (local-only, gitignored; covers §3.3 of this spec's discipline)
- Memory invariants in scope:
  - `feedback_pre_implementation_verification.md` — verify load-bearing cross-module assumptions in specs before writing code
  - `feedback_use_worktrees.md` — use worktrees for implementation
- Code references:
  - `crates/nsl-codegen/src/lib.rs:702-826` — `compile_and_calibrate`
  - `crates/nsl-codegen/src/stmt.rs:3966-4046` — calibration block in `compile_train_block` (deleted in commit 4)
  - `crates/nsl-codegen/src/calibration/binary_codegen.rs:62-258` — `real_subprocess_entry` (canonical entry under (c-i))
  - `crates/nsl-codegen/src/calibration/binary_codegen.rs:1771-1779` — hop 6 site (generalized in commit 2)
  - `crates/nsl-codegen/src/calibration/observation.rs:127-137` — `ObservationSet::needs_forward_pass()` (post-PR-#140)
  - `crates/nsl-codegen/tests/wggo_backward_pipeline.rs:179-201` — merge-gate test `#[ignore]` message (post-PR-#141)
