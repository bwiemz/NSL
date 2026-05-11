# #134: Decouple calibration harness from `compile_train_block` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge AWQ and WGGO calibration onto a single canonical entry point (`real_subprocess_entry`); delete the calibration block in `compile_train_block`; un-`#[ignore]` the WGGO Phase 2 merge-gate test.

**Architecture:** Sequence C with 1a/1b split per spec §8.1. Regression discipline (determinism verification + snapshot test + CHANGELOG-CALIBRATION + CI) lands first; hop 6 generalization lands second; wrapper-level atomic firing-move lands third; optional cleanup; un-`#[ignore]` last. AWQ flow byte-identical by inspection at every commit boundary (snapshot test verifies).

**Tech Stack:** Rust, `insta` 1.40 (yaml feature), `serde_json`, `cargo`, GitHub Actions (bash).

**Spec:** `docs/superpowers/specs/2026-05-06-134-decouple-calibration-design.md`

---

## File Structure

| Path | Status | Responsibility | Commit |
|------|--------|----------------|--------|
| `scripts/verify-awq-determinism.sh` | New | Run AWQ snapshot test 10× with thread variation; assert byte-identical output | 1a |
| `crates/nsl-codegen/tests/awq_full_pipeline.rs` | Modify | Add `snapshot_awq_sidecar_baseline` insta test that captures full Sidecar JSON | 1b |
| `crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap` | New | Captured baseline (created by `cargo insta accept` in 1b) | 1b |
| `CHANGELOG-CALIBRATION.md` | New | Initial entry referencing PR for #134 implementation | 1b |
| `.github/workflows/calibration-snapshot-changelog.yml` | New | CI check: PRs touching `awq_*.snap` must update CHANGELOG-CALIBRATION.md | 1b |
| `crates/nsl-codegen/src/calibration/binary_codegen.rs` | Modify | Hop 6 generalization + `grad_target_to_projection_meta` adapter | 2 |
| `crates/nsl-codegen/src/lib.rs` | Modify | Add wrapper-level firing call to `compile_and_calibrate` (post-`compile_main`) | 3 |
| `crates/nsl-codegen/src/stmt.rs` | Modify | Delete calibration block at lines 3960-4046 | 3 |
| `crates/nsl-codegen/src/stmt.rs` | Modify | Delete any now-orphaned helpers (only if commit 3 left any) | 4 (optional) |
| `tests/fixtures/wggo_attention_mlp_real.nsl` | Modify | Update header comments (workaround caveat no longer applies) | 5 |
| `crates/nsl-codegen/tests/wggo_backward_pipeline.rs` | Modify | Remove `#[ignore]` from merge-gate test | 5 |

---

## Prerequisites

- [ ] **Step 0.1: Verify PR #141 is merged**

PR #141 pins the merge-gate test's `#[ignore]` message to #134 as the resolution path. If still open, merge it before starting. Check:

```bash
gh pr view 141 --json state
```

Expected: `"state":"MERGED"`. If `OPEN`, ask the user to merge before proceeding.

- [ ] **Step 0.2: Verify on `main` branch with clean tree**

```bash
git checkout main && git pull --ff-only origin main && git status
```

Expected: `On branch main`, `working tree clean`. If dirty, stash or commit first.

- [ ] **Step 0.3: Create implementation worktree**

```bash
git worktree add ../NSL.worktrees/134-impl -b feat/134-decouple-calibration main
cd ../NSL.worktrees/134-impl
```

All subsequent steps run from this worktree.

- [ ] **Step 0.4: Confirm baseline AWQ test currently passes**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline
```

Expected: `6 passed; 0 failed`. If any test fails on unmodified `main`, stop and investigate — the baseline is broken before #134 work starts.

---

## Task 1a — Determinism verification

**Goal:** Verify the AWQ sidecar JSON is byte-deterministic across runs before capturing it as a snapshot. If non-determinism is found (most likely culprit: `HashMap` iteration order in serialization paths), fix the source — never mask with a hash digest.

**Files:**

- Create: `scripts/verify-awq-determinism.sh`
- Modify (conditional, only if non-determinism found): wherever the source is — typically a `HashMap` → `BTreeMap` swap

### Step 1a.1: Write the determinism verification script

- [ ] **Write `scripts/verify-awq-determinism.sh`**

```bash
#!/usr/bin/env bash
# verify-awq-determinism.sh — Spec §6.1 precondition for #134.
#
# Runs the AWQ end-to-end pipeline 10 times serially, then twice more with
# differing thread counts, and verifies the captured Sidecar JSON is
# byte-identical across runs. Exits 0 on success, 1 on any divergence.
#
# Used at commit 1a of #134 to verify the AWQ sidecar is deterministic
# before commit 1b captures it as an insta baseline. Spec §6.1 forbids
# papering over non-determinism with a hash digest — fix the source.

set -euo pipefail

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)/crates/nsl-codegen"
TEST_NAME="end_to_end_real_subprocess_matches_analytical_reference"

echo "verify-awq-determinism: capturing 10 sequential runs of $TEST_NAME"
for i in $(seq 1 10); do
    cargo test -p nsl-codegen --test awq_full_pipeline -- \
        --test-threads 1 --nocapture "$TEST_NAME" \
        > "$TMP_DIR/run_$i.log" 2>&1 \
        || { echo "FAIL: run $i exited non-zero. Log:"; cat "$TMP_DIR/run_$i.log"; exit 1; }
done

echo "verify-awq-determinism: capturing 2 thread-varied runs"
cargo test -p nsl-codegen --test awq_full_pipeline -- \
    --test-threads 8 --nocapture "$TEST_NAME" \
    > "$TMP_DIR/threads_8.log" 2>&1

echo "verify-awq-determinism: extracting Sidecar JSON from each run"
# The AWQ end-to-end test prints the serialized sidecar to stdout when
# the SIDECAR_DUMP=1 env var is set. We re-run with that env var to
# extract one canonical JSON per run.
for run in run_1 run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9 run_10 threads_8; do
    SIDECAR_DUMP=1 cargo test -p nsl-codegen --test awq_full_pipeline -- \
        --test-threads 1 --nocapture "$TEST_NAME" \
        2>&1 | grep -A 9999 '^SIDECAR_JSON_START' \
              | grep -B 9999 '^SIDECAR_JSON_END' \
              | grep -v '^SIDECAR_JSON_' \
        > "$TMP_DIR/${run}.json"
done

echo "verify-awq-determinism: comparing runs"
DIFFS=0
REF="$TMP_DIR/run_1.json"
for run in run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9 run_10 threads_8; do
    if ! diff -q "$REF" "$TMP_DIR/${run}.json" > /dev/null; then
        echo "DIVERGENCE: $run differs from run_1"
        diff "$REF" "$TMP_DIR/${run}.json" | head -20
        DIFFS=$((DIFFS + 1))
    fi
done

if [ "$DIFFS" -ne 0 ]; then
    echo ""
    echo "FAIL: $DIFFS run(s) diverged from baseline."
    echo "Likely sources: HashMap iteration order in Sidecar serialization."
    echo "Fix at source per spec §6.1 — do NOT mask with a hash digest."
    exit 1
fi

echo "PASS: all 11 runs produced byte-identical Sidecar JSON."
```

### Step 1a.2: Make the script executable

- [ ] **Run:** `chmod +x scripts/verify-awq-determinism.sh`

### Step 1a.3: Add `SIDECAR_DUMP=1` instrumentation to the AWQ test

The verification script greps for `SIDECAR_JSON_START` / `SIDECAR_JSON_END` sentinels. Add the dump path in `end_to_end_real_subprocess_matches_analytical_reference`.

- [ ] **Locate the test:** `crates/nsl-codegen/tests/awq_full_pipeline.rs`

Find the function `end_to_end_real_subprocess_matches_analytical_reference` (around line 624 per spec Appendix A). Inside, after the sidecar is read back from the subprocess (search for `serde_json::from_slice` or `Sidecar` deserialization), add:

```rust
// #134 §6.1 determinism verification: when SIDECAR_DUMP=1, print the
// canonical sidecar JSON between sentinel lines so scripts/verify-awq-
// determinism.sh can extract and compare across runs. Sentinels are
// distinctive enough to not collide with arbitrary test output.
if std::env::var("SIDECAR_DUMP").is_ok() {
    let canonical = serde_json::to_string_pretty(&sidecar)
        .expect("Sidecar serializes to JSON");
    eprintln!("SIDECAR_JSON_START");
    eprintln!("{canonical}");
    eprintln!("SIDECAR_JSON_END");
}
```

Insert *after* the line that materializes `sidecar: Sidecar` (the deserialized result the test will assert on).

### Step 1a.4: Run the verification script

- [ ] **Run:** `bash scripts/verify-awq-determinism.sh`

Expected if AWQ is already deterministic: `PASS: all 11 runs produced byte-identical Sidecar JSON.` (this is the happy path; the `Sidecar.hooks` field already uses `BTreeMap` per `crates/nsl-codegen/src/calibration/sidecar.rs:49`, so most likely cause of non-determinism is absent).

If non-determinism is found, proceed to Step 1a.5. Otherwise skip to Step 1a.7.

### Step 1a.5 (conditional): Investigate and fix non-determinism

- [ ] **Run with diff output to identify the drifting field:**

```bash
diff /tmp/verify-awq-*/run_1.json /tmp/verify-awq-*/run_2.json | head -30
```

- [ ] **Common diagnostics:**
  - If the diff shows reordered map keys → likely a `HashMap<String, _>` in a Sidecar-adjacent struct. Grep for `HashMap<String` in `crates/nsl-codegen/src/calibration/` and replace with `BTreeMap` at the serialization boundary.
  - If the diff shows different float representations (`0.1` vs `0.10000000149`) → likely a `f32/f64` field. Pin formatting with `serde_json::Number` or a `#[serde(serialize_with = "...")]` adapter.
  - If the diff shows different timestamps / version strings → likely a non-deterministic field. Either remove or pin to a fixed value during tests.

- [ ] **Fix at source.** Do not introduce a hash digest, sorted-output post-processor, or "stable serializer wrapper" — the bug is the data structure, not the serializer.

### Step 1a.6 (conditional): Re-run verification after fix

- [ ] **Run:** `bash scripts/verify-awq-determinism.sh`

Expected: `PASS: all 11 runs produced byte-identical Sidecar JSON.`

If still failing, the fix is incomplete — investigate further before proceeding.

### Step 1a.7: Commit

- [ ] **Stage:**

```bash
git add scripts/verify-awq-determinism.sh crates/nsl-codegen/tests/awq_full_pipeline.rs
# If non-determinism was found and fixed:
# git add <files that fixed the non-determinism source>
```

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
chore(134): determinism verification for AWQ sidecar JSON

Commit 1a of #134's Sequence C (spec §8.1). Adds scripts/verify-awq-
determinism.sh which runs the AWQ end-to-end pipeline 11 times (10
serial + 1 thread-varied) and asserts the produced Sidecar JSON is
byte-identical across runs. Adds SIDECAR_DUMP=1 instrumentation to
the AWQ end-to-end test so the script can extract canonical JSON
from each run via sentinel markers.

Determinism is the precondition for commit 1b's insta baseline
capture: if the sidecar isn't byte-deterministic, the baseline
snapshot will flake on every CI run and the (c-i) zero-by-
construction merge gate becomes unverifiable.

[If non-determinism was fixed:] Also fixes <SOURCE> by replacing
<HashMap with BTreeMap | pinning float formatting | etc>. Spec §6.1
forbids papering over non-determinism with hash digests — the source
fix lives here so commit 1a's diff is the audit trail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1b — Baseline snapshot + CHANGELOG + CI enforcement

**Goal:** Capture the AWQ Sidecar JSON as an `insta` snapshot baseline, create `CHANGELOG-CALIBRATION.md`, and land the CI workflow that enforces "snapshot changes require CHANGELOG entry."

**Files:**

- Modify: `crates/nsl-codegen/tests/awq_full_pipeline.rs` — add `snapshot_awq_sidecar_baseline` test
- Create: `crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap` (auto-generated by `cargo insta accept`)
- Create: `CHANGELOG-CALIBRATION.md`
- Create: `.github/workflows/calibration-snapshot-changelog.yml`

### Step 1b.1: Add the snapshot test to `awq_full_pipeline.rs`

- [ ] **Locate the test file:** `crates/nsl-codegen/tests/awq_full_pipeline.rs`

- [ ] **Add the snapshot test at the end of the file** (after the existing `end_to_end_real_subprocess_matches_analytical_reference` test at line 656):

```rust
/// #134 §6.2 — AWQ sidecar bit-identical regression test.
///
/// Captures the full Sidecar JSON as an `insta` snapshot. Under #134's
/// (c-i) convergence shape, this snapshot is **zero-by-construction**: the
/// wrapper-level firing change does not affect `compile_main`'s output and
/// fires calibration against the same fixture in the same order. Any drift
/// after commit 1b is captured indicates an implementation bug in commits
/// 2-5, not a (c-i) design problem.
///
/// Legitimate future changes (new hooks, format upgrades) must update both
/// this snapshot AND `CHANGELOG-CALIBRATION.md` — CI enforces the pairing.
#[test]
fn snapshot_awq_sidecar_baseline() {
    let data_path = fixture("awq_calib_data.safetensors");
    let weights_path = fixture("awq_calib_weights.safetensors");
    let (projections, compile_bundle) = awq_fixture_compile_bundle();

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let cfg = HarnessConfig {
        checkpoints: vec![weights_path.clone()],
        calibration_data: data_path.clone(),
        samples: 8,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let sidecar = real_subprocess_entry(&cfg, &registry)
        .expect("real subprocess pipeline runs end-to-end")
        .sidecar;

    // Serialize to a canonical pretty-printed JSON. serde_json sorts BTreeMap
    // keys by construction; Sidecar.hooks is BTreeMap<String, Vec<u8>>. Float
    // fields use default serde_json formatting (deterministic per Rust toolchain).
    let canonical = serde_json::to_string_pretty(&sidecar)
        .expect("Sidecar serializes to JSON");

    insta::assert_snapshot!("awq_sidecar_baseline", canonical);
}
```

**Implementer note:** the setup verbatim-copies `end_to_end_real_subprocess_matches_analytical_reference` (at `crates/nsl-codegen/tests/awq_full_pipeline.rs:623-645`). The only divergence is the assertion: instead of analytical-reference comparison, snapshot the full Sidecar JSON. The `awq_fixture_compile_bundle()` helper (declared at line 77) takes no arguments and returns `(Vec<DiscoveredProjection>, Arc<CalibrationCompileBundle>)`. `AwqCalibrationHook::from_discovered` is the canonical constructor used by the AWQ test suite (not `AwqCalibrationHook::new`).

### Step 1b.2: Run the test to generate the snapshot

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline -- --include-ignored 2>&1 | tail -30
```

Expected: test FAILS with `assertion failed: snapshot missing` (or similar insta message). Insta creates a `.snap.new` file in `crates/nsl-codegen/tests/snapshots/`.

### Step 1b.3: Accept the snapshot

- [ ] **Run:**

```bash
cargo insta accept --include-hidden 2>&1 | tail -5
```

This renames `awq_full_pipeline__awq_sidecar_baseline.snap.new` → `.snap`.

- [ ] **Verify:**

```bash
ls crates/nsl-codegen/tests/snapshots/ | grep awq_sidecar_baseline
```

Expected: `awq_full_pipeline__awq_sidecar_baseline.snap` (no `.new` suffix).

### Step 1b.4: Re-run the test to confirm it passes

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline
```

Expected: `test snapshot_awq_sidecar_baseline ... ok`.

### Step 1b.5: Run the determinism script against the new test

- [ ] **Update `scripts/verify-awq-determinism.sh`** to target `snapshot_awq_sidecar_baseline` instead of the existing end-to-end test (replace `TEST_NAME="end_to_end_real_subprocess_matches_analytical_reference"` with `TEST_NAME="snapshot_awq_sidecar_baseline"`).

- [ ] **Run:**

```bash
bash scripts/verify-awq-determinism.sh
```

Expected: `PASS: all 11 runs produced byte-identical Sidecar JSON.`

### Step 1b.6: Create `CHANGELOG-CALIBRATION.md`

- [ ] **Create at repo root:** `CHANGELOG-CALIBRATION.md`

```markdown
# CHANGELOG — Calibration

Dedicated changelog for calibration-affecting changes. Per #134 §6.3, any
PR modifying `crates/nsl-codegen/tests/snapshots/awq_*.snap` (or other
calibration snapshot files) MUST add a corresponding entry here.
CI enforces the pairing (`.github/workflows/calibration-snapshot-changelog.yml`).

**Entry format:**

```
## YYYY-MM-DD — PR #NNN — <short title>

**Snapshot files:** <comma-separated list>
**Cause:** <one sentence: what changed and why>
**Bit-equivalence evidence:** <how the impact on AWQ behavior was verified>
```

---

## 2026-05-06 — PR #TBD — Initial AWQ sidecar baseline

**Snapshot files:** `crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap`

**Cause:** Initial baseline capture for #134's (c-i) regression discipline.
Captures the AWQ Sidecar JSON from `end_to_end_real_subprocess_matches_
analytical_reference`'s fixture under the unmodified `main` branch
(commit 1a verified determinism via `scripts/verify-awq-determinism.sh`).

**Bit-equivalence evidence:** Snapshot captured from unmodified `main`
before any #134 logic changes. Determinism verified across 11 runs
(10 serial + 1 thread-varied). Subsequent commits in #134 (hop 6 fix,
wrapper-level firing, train-block deletion, un-ignore) must NOT change
this snapshot per (c-i)'s zero-by-construction merge gate.
```

**Note for implementer:** Replace `PR #TBD` with the actual PR number after opening the implementation PR (or leave it and amend later).

### Step 1b.7: Create the CI workflow

- [ ] **Create:** `.github/workflows/calibration-snapshot-changelog.yml`

```yaml
name: Calibration snapshot CHANGELOG enforcement

on:
  pull_request:
    paths:
      - 'crates/nsl-codegen/tests/snapshots/awq_*.snap'

jobs:
  enforce-changelog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Verify CHANGELOG-CALIBRATION.md has entry for this PR
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
          BASE_REF: ${{ github.event.pull_request.base.ref }}
        run: |
          set -euo pipefail
          # Find awq_*.snap files changed in this PR.
          CHANGED_SNAPSHOTS=$(git diff --name-only "origin/${BASE_REF}...HEAD" \
            -- 'crates/nsl-codegen/tests/snapshots/awq_*.snap' || true)
          if [ -z "$CHANGED_SNAPSHOTS" ]; then
            echo "No AWQ snapshot files changed; skipping CHANGELOG check."
            exit 0
          fi
          echo "Changed AWQ snapshots:"
          echo "$CHANGED_SNAPSHOTS"
          echo ""
          # Verify CHANGELOG-CALIBRATION.md was modified in this PR.
          if ! git diff --name-only "origin/${BASE_REF}...HEAD" | grep -q '^CHANGELOG-CALIBRATION\.md$'; then
            echo "FAIL: PR modifies AWQ snapshot file(s) but does not update CHANGELOG-CALIBRATION.md."
            echo "Per #134 §6.3, any snapshot-touching PR must add a corresponding entry."
            echo "See CHANGELOG-CALIBRATION.md for the entry format."
            exit 1
          fi
          # Verify the CHANGELOG entry references this PR number.
          if ! grep -q "PR #${PR_NUMBER}" CHANGELOG-CALIBRATION.md; then
            echo "FAIL: CHANGELOG-CALIBRATION.md was modified but does not reference PR #${PR_NUMBER}."
            echo "Add an entry of the form: '## YYYY-MM-DD — PR #${PR_NUMBER} — <title>'"
            exit 1
          fi
          echo "PASS: CHANGELOG-CALIBRATION.md has entry referencing PR #${PR_NUMBER}."
```

### Step 1b.8: Commit

- [ ] **Stage:**

```bash
git add crates/nsl-codegen/tests/awq_full_pipeline.rs \
        crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap \
        CHANGELOG-CALIBRATION.md \
        .github/workflows/calibration-snapshot-changelog.yml \
        scripts/verify-awq-determinism.sh
```

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
chore(134): AWQ sidecar baseline + CHANGELOG-CALIBRATION + CI enforcement

Commit 1b of #134's Sequence C (spec §8.1). Lands the regression
machinery that gates commits 2-5:

- Snapshot test `snapshot_awq_sidecar_baseline` captures the AWQ
  Sidecar JSON via insta. Under (c-i) the snapshot is zero-by-
  construction across commits 2-5; any drift is an implementation
  bug, not a design problem.
- CHANGELOG-CALIBRATION.md at repo root documents calibration-
  affecting changes. Initial entry references this PR.
- .github/workflows/calibration-snapshot-changelog.yml enforces:
  any PR modifying awq_*.snap files must update CHANGELOG-CALIBRATION.

Discipline-first ordering principle (spec §8.3): regression machinery
lands before behavioral changes. Otherwise the behavioral changes are
made without the discipline available to verify them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Hop 6 generalization

**Goal:** Fix `emit_calibration_model_object` so it accepts the union of `calibration_retention ∪ calibration_grad_retention`. AWQ flow byte-identical by inspection (its `calibration_retention` is non-empty; the new code path is behind a condition AWQ never satisfies).

**Files:**

- Modify: `crates/nsl-codegen/src/calibration/binary_codegen.rs` — generalize hop 6 + add `grad_target_to_projection_meta` adapter

### Step 2.1: Read the current code

- [ ] **Open `crates/nsl-codegen/src/calibration/binary_codegen.rs`** and locate `emit_calibration_model_object` (starts at line 1749). The hop 6 code is lines 1763-1800 (the discovery + empty-check + projections extraction).

The current logic:

```rust
let mut compile_opts = opts.clone();
if compile_opts.calibration_retention.is_none() {
    let discovered =
        crate::calibration::pre_scan_awq_projections_from_ast(ast, &bundle.interner);
    if !discovered.is_empty() {
        compile_opts.calibration_retention = Some(discovered);
    }
}
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
// ... model_name and transpose_fields use first_projection ...
```

### Step 2.2: Add the `grad_target_to_projection_meta` adapter

- [ ] **Add this function** above `emit_calibration_model_object` (around line 1748):

```rust
/// Adapt a WGGO Phase 2 `WggoGradTarget` to a `DiscoveredProjection` so
/// `emit_calibration_model_object` can read `channels` / `model_name` /
/// `transpose_fields` from a uniform projection list regardless of
/// whether the source was AWQ (`@quantize(dtype="awq4")`) or WGGO
/// (`@wggo_target(...)`).
///
/// Spec §5.2: the adapter emits **one** `DiscoveredProjection` per target,
/// using **`w_o_shape`** for the `weight_shape` field. Rationale:
/// `emit_calibration_model_object` reads `channels = first_projection.
/// weight_shape[1]` (output dimension); `w_o_shape[1]` is the corresponding
/// output dimension for attention's output projection. Picking `w_o_shape`
/// is load-bearing for "AWQ byte-identical by inspection" — a different
/// choice could shift the channels value on mixed AWQ + WGGO fixtures and
/// trigger a snapshot regression diagnosed as a symptom (wrong channels)
/// rather than the cause (wrong shape selection).
///
/// If a future caller needs per-W_* granularity, this adapter can be
/// extended to emit four entries per target — but v1's hop 6 fix needs
/// only one entry per layer to satisfy the existing reads.
fn grad_target_to_projection_meta(
    target: &crate::calibration::discovery::WggoGradTarget,
) -> crate::calibration::discovery::DiscoveredProjection {
    crate::calibration::discovery::DiscoveredProjection {
        // Use the output projection's qualified path so model_name
        // (split before first '.') and transpose_fields (strip-prefix
        // model_name + '.') resolve identically to the AWQ semantic.
        projection: target.w_o.clone(),
        weight_shape: target.w_o_shape,
    }
}
```

### Step 2.3: Replace the hop 6 empty-check with the union match

- [ ] **Replace lines 1763-1779** (the existing pre-scan + empty-check + `.ok_or_else` block) with:

```rust
let mut compile_opts = opts.clone();

// #134 §5.2 hop 6 generalization: emit_calibration_model_object reads
// from the union calibration_retention ∪ calibration_grad_retention.
// AWQ flow: calibration_retention is non-empty (or AWQ pre-scan
// discovers it from the AST), so the first arm matches and the function
// reads exactly what it read before — byte-identical by inspection.
// WGGO-only flow: calibration_retention is None or empty but
// calibration_grad_retention is populated; the second arm synthesizes
// projections from WGGO targets using grad_target_to_projection_meta
// (one entry per target, using w_o_shape per spec §5.2).
if compile_opts.calibration_retention.is_none() {
    let discovered =
        crate::calibration::pre_scan_awq_projections_from_ast(ast, &bundle.interner);
    if !discovered.is_empty() {
        compile_opts.calibration_retention = Some(discovered);
    }
}

let projections: Vec<crate::calibration::discovery::DiscoveredProjection> = match (
    compile_opts.calibration_retention.as_ref(),
    compile_opts.calibration_grad_retention.as_ref(),
) {
    (Some(retn), _) if !retn.is_empty() => retn.clone(),
    (_, Some(grads)) if !grads.is_empty() => {
        grads.iter().map(grad_target_to_projection_meta).collect()
    }
    _ => {
        return Err(HarnessError::Infrastructure {
            reason: "emit_calibration_model_object requires either AWQ \
                     projections (calibration_retention) or WGGO targets \
                     (calibration_grad_retention)"
                .into(),
        });
    }
};
let first_projection = projections.first().expect(
    "projections vec built from non-empty match arms — first() cannot fail",
);
let channels = first_projection.weight_shape[1] as i64;
```

### Step 2.4: Run the AWQ behavioral test

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline
```

Expected: `6 passed; 0 failed`. AWQ flow takes the first match arm and behaves identically to before.

### Step 2.5: Run the snapshot test

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline
```

Expected: `test snapshot_awq_sidecar_baseline ... ok`. The snapshot must NOT change — if it does, the generalization accidentally affected AWQ's code path (which it shouldn't per §5.3).

### Step 2.6: Run the determinism script

- [ ] **Run:** `bash scripts/verify-awq-determinism.sh`

Expected: `PASS: all 11 runs produced byte-identical Sidecar JSON.`

### Step 2.7: Run the merge-gate test (still expected to be ignored)

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test wggo_backward_pipeline -- --include-ignored 2>&1 | tail -20
```

Expected outcomes:
- **Best case:** Test PASSES. Hop 6 was the only remaining blocker; the test will be un-`#[ignore]`'d in commit 5.
- **Acceptable case:** Test fails with a different error than `"requires AWQ projections"` or `"non-empty projection list"`. That means hop 6 was fixed but a "next hop" exists — document the new error in commit 5's notes for follow-up.
- **Failure case:** Test still fails with the hop 6 message. The generalization was incomplete — investigate before proceeding.

**Capture the test's output** in a temporary log for the commit message:

```bash
cargo test -p nsl-codegen --test wggo_backward_pipeline -- --include-ignored 2>&1 \
  > /tmp/134-commit-2-mergegate.log || true
tail -40 /tmp/134-commit-2-mergegate.log
```

### Step 2.8: Commit

- [ ] **Stage:** `git add crates/nsl-codegen/src/calibration/binary_codegen.rs`

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
fix(134): hop 6 — emit_calibration_model_object accepts WGGO targets

Commit 2 of #134's Sequence C (spec §5.2). Generalizes
emit_calibration_model_object to read from the union of
calibration_retention U calibration_grad_retention rather than
requiring non-empty calibration_retention specifically.

AWQ flow byte-identical by inspection (§5.3): calibration_retention
is non-empty for AWQ, so the first match arm matches and the function
reads exactly what it read before. The new WGGO arm is behind a
condition AWQ never satisfies.

WGGO flow: calibration_retention is None or empty but calibration_
grad_retention is populated. The second match arm synthesizes
DiscoveredProjections from WggoGradTargets via grad_target_to_
projection_meta — one entry per target, using w_o_shape for the
weight_shape field. Spec §5.2 pins w_o_shape because emit_calibration_
model_object reads channels = first_projection.weight_shape[1]
(output dimension) and w_o_shape[1] is the corresponding output
dimension for attention's output projection.

Verification:
- cargo test -p nsl-codegen --test awq_full_pipeline: 6/6 PASS
- snapshot_awq_sidecar_baseline: PASS (snapshot unchanged)
- scripts/verify-awq-determinism.sh: PASS (11 runs byte-identical)
- merge-gate test still #[ignore]'d (un-ignored in commit 5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — Wrapper-level firing (atomic move)

**Goal:** Single-commit move: delete the calibration-firing call from `compile_train_block` and add an equivalent call in `compile_and_calibrate` that invokes `real_subprocess_entry` directly post-`compile_main`. **No parallel-firing intermediate state.**

**Files:**

- Modify: `crates/nsl-codegen/src/lib.rs` — add firing call in `compile_and_calibrate` after `compile_main` returns
- Modify: `crates/nsl-codegen/src/stmt.rs` — delete the calibration block at lines 3960-4046

### Step 3.1: Read the calibration block to be deleted

- [ ] **Open `crates/nsl-codegen/src/stmt.rs`** at lines 3960-4046.

This is the block inside `compile_train_block` that:
1. Checks `self.compile_options.calibration_data.is_some()`
2. Discovers AWQ projections via `self.discover_awq_projections()`
3. Registers `AwqCalibrationHook` and conditionally `WggoGradientHook`
4. Builds a `HarnessConfig` and calls `crate::calibration::run_harness_production(&registry, &cfg)`

The hook-registration logic (AWQ + WGGO) and the `HarnessConfig` construction are the parts we'll move to `compile_and_calibrate`. The post-call sidecar handling (line 4044 onward) is also moved.

- [ ] **Read all the way to the end of the block** (line ~4090) to capture the post-call handling code that needs to move too:

```bash
awk 'NR>=3960 && NR<=4090 {print NR": "$0}' crates/nsl-codegen/src/stmt.rs | head -150
```

### Step 3.2: Read `compile_and_calibrate` to find the insertion point

- [ ] **Open `crates/nsl-codegen/src/lib.rs`** at lines 702-826.

The function builds a `Compiler`, runs `pre_finalize` (which calls `compile_main` which currently triggers the calibration block), then extracts `compiler.compile_options.calibration_sidecar`.

**Insertion point:** between `compile_main` returning (line 802) and `compile_pending_lambdas` (line 803), OR after `pre_finalize` completes successfully (line 808) but before the sidecar extraction (line 813). The latter is cleaner architecturally — calibration runs around the compiled code, not interleaved with arena emission.

### Step 3.3: Write the new firing call in `compile_and_calibrate`

- [ ] **Modify `crates/nsl-codegen/src/lib.rs`**, inside `compile_and_calibrate`'s closure, after `pre_finalize`'s logic completes. The cleanest spot is right after the closure but before extracting the sidecar (around line 810).

Actually re-reading the code structure: the closure assigns to `pre_finalize: Result<(), CodegenError>` which is `?`'d at line 816. The harness firing should happen INSIDE the closure to share its error-handling, after `compile_main` returns successfully.

- [ ] **Locate line 802** (`compiler.compile_main(&parsed.module.stmts)?;`) and **insert immediately after it:**

```rust
// #134 (c-i) wrapper-level firing — spec §4.1 + §8.1 commit 3.
// Previously: compile_train_block fired the calibration harness as a
// side-effect inside compile_main, coupling calibration to the
// presence of a `train` block.
// Now: calibration fires here, at the wrapper level, regardless of
// whether the source contains a `train` block. Path 1 (compile_train_
// block's calibration block at stmt.rs:3960-4046) is deleted in this
// same commit. real_subprocess_entry is invoked directly — same
// canonical path that AWQ + WGGO end-to-end tests already use.
if let Some(ref data_path) = compiler.compile_options.calibration_data.clone() {
    let mut registry = crate::calibration::registry::HookRegistry::new();
    let awq_projections = compiler.discover_awq_projections().unwrap_or_default();
    if let Some(pre_scan) = compiler.compile_options.calibration_retention.as_ref() {
        crate::calibration::discovery::check_discovery_agreement(
            pre_scan,
            &awq_projections,
        )
        .map_err(|err| CodegenError::new(err.to_string()))?;
    }
    if !awq_projections.is_empty() {
        let proj_refs: Vec<crate::calibration::ProjectionRef> = awq_projections
            .iter()
            .map(|dp| dp.projection.clone())
            .collect();
        registry.register(Box::new(
            crate::calibration::awq_hook::AwqCalibrationHook::new(proj_refs),
        ));
        compiler.compile_options.calibration_retention = Some(awq_projections);
    }
    if let Some(targets) = compiler.compile_options.calibration_grad_retention.as_ref() {
        if !targets.is_empty() {
            registry.register(Box::new(
                crate::calibration::wggo_gradient_hook::WggoGradientHook::new(
                    targets.clone(),
                ),
            ));
        }
    }
    if registry.is_empty() {
        eprintln!(
            "warning: --calibration-data {} supplied but no calibration hooks \
             registered (no consumers yet — this is a no-op in MVP)",
            data_path.display()
        );
    } else {
        let mode = match compiler
            .compile_options
            .calibration_mode
            .as_deref()
            .unwrap_or("required")
        {
            "best-effort" => crate::calibration::HarnessMode::BestEffort,
            _ => crate::calibration::HarnessMode::Required,
        };
        let cfg = crate::calibration::HarnessConfig {
            checkpoints: compiler
                .compile_options
                .weight_file
                .as_ref()
                .map(|p| vec![p.clone()])
                .unwrap_or_default(),
            calibration_data: data_path.clone(),
            samples: compiler.compile_options.calibration_samples,
            batch_size: compiler.compile_options.calibration_batch_size,
            timeout_secs: compiler.compile_options.calibration_timeout_secs,
            mode,
            projections: compiler
                .compile_options
                .calibration_retention
                .clone()
                .unwrap_or_default(),
            compile_bundle: compiler.compile_options.calibration_compile_bundle.clone(),
        };
        match crate::calibration::binary_codegen::real_subprocess_entry(&registry, &cfg) {
            Ok(out) => {
                eprintln!(
                    "[calibration] {} ({} hooks)",
                    out.outcome_repr,
                    out.sidecar.hooks.len()
                );
                compiler.compile_options.calibration_sidecar = Some(out.sidecar);
            }
            Err(err) => {
                return Err(CodegenError::new(format!("calibration: {err}")));
            }
        }
    }
}
```

**Note for implementer:** The code is largely a verbatim copy of `stmt.rs:3960-4046` with two changes:
1. `self.` → `compiler.` (the compile_and_calibrate context has a `compiler` local, not `self`)
2. The harness call is `real_subprocess_entry` instead of `run_harness_production` (per (c-i)'s convergence — both are valid entry points but `real_subprocess_entry` is the canonical one we're standardizing on)

**Critical:** verify the `run_harness_production` vs `real_subprocess_entry` signature compatibility. They may differ in argument order. Adjust the call site accordingly — the spec's intent is "canonical entry," not strict signature equivalence.

### Step 3.4: Delete the calibration block from `compile_train_block`

- [ ] **Open `crates/nsl-codegen/src/stmt.rs`** at lines 3960-4046.

- [ ] **Delete the entire block** starting from `if let Some(ref data_path) = self.compile_options.calibration_data {` (line ~3966) through its closing `}` (which is the matching `}` for the outer `if let`, around line 4046 or wherever the block ends).

**Critical:** read 10 lines BEFORE line 3960 and 10 lines AFTER line 4046 first, so you understand the surrounding control flow. The deletion must not leave a dangling `else`, mismatched braces, or orphaned variable declarations.

### Step 3.5: Build to verify no orphaned references

- [ ] **Run:**

```bash
cargo build -p nsl-codegen --tests 2>&1 | tail -40
```

Expected: clean build with no errors. Common failures:
- `cannot find function ...` — a helper in `stmt.rs` that was used only by the deleted block. If it's `pub`, leave it for commit 4 to evaluate. If it's `pub(crate)` or private and unused elsewhere, deletion candidate for commit 4.
- `unused variable ...` warnings in `stmt.rs` — orphaned bindings from the deleted block. Remove.
- `mismatched braces` — re-check the deletion bounds; you may have removed one too many or too few lines.

### Step 3.6: Run the AWQ behavioral tests

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline
```

Expected: `6 passed; 0 failed`. AWQ's flow now goes through `compile_and_calibrate → compile_main → (no more calibration block) → compile_and_calibrate's wrapper firing`. The sidecar is populated the same way as before, just from a different call site.

### Step 3.7: Run the snapshot test

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline
```

Expected: PASS. Snapshot unchanged — per (c-i)'s zero-by-construction claim, the wrapper-level firing produces byte-identical output because it fires against the same compiled binary in the same order.

If the snapshot fails: the move was not atomic-equivalent. Diff the snapshot via `cargo insta review` to identify the drift, then fix. Do NOT `cargo insta accept` — the drift is the bug, not the snapshot.

### Step 3.8: Run the determinism script

- [ ] **Run:** `bash scripts/verify-awq-determinism.sh`

Expected: `PASS: all 11 runs produced byte-identical Sidecar JSON.`

### Step 3.9: Run the full test suite (broader safety net)

- [ ] **Run:**

```bash
cargo test -p nsl-codegen 2>&1 | tail -10
```

Expected: all tests pass (modulo the still-`#[ignore]`'d merge-gate test).

### Step 3.10: Commit

- [ ] **Stage:** `git add crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/stmt.rs`

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
feat(134): wrapper-level firing — calibration runs around compilation

Commit 3 of #134's Sequence C (spec §8.1, atomic move per §8.4
rejection of Sequence B). Moves calibration harness firing from
inside compile_train_block to the compile_and_calibrate wrapper,
post-compile_main.

(c-i) convergence: real_subprocess_entry is the canonical entry-
point implementation. compile_and_calibrate calls it directly when
calibration_data.is_some(), after compile_main returns. The
calibration block in compile_train_block (stmt.rs:3960-4046) is
deleted in this same commit; compile_train_block returns to
compiling train blocks only.

No parallel-firing intermediate state. The atomic move + snapshot
test (commit 1b) provides the byte-equivalence verification with
less mechanism than Sequence B's parallel-firing trick.

Mental-model match (spec §4.3): calibration runs around the compiled
code, not inside it. compile_train_block compiles train blocks;
compile_main compiles main; compile_and_calibrate orchestrates.

Verification:
- cargo test -p nsl-codegen --test awq_full_pipeline: 6/6 PASS
- snapshot_awq_sidecar_baseline: PASS (snapshot unchanged — wrapper-
  level firing is byte-equivalent to train-block firing by (c-i)'s
  zero-by-construction analysis)
- scripts/verify-awq-determinism.sh: PASS
- merge-gate test still #[ignore]'d (un-ignored in commit 5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Train-block cleanup (conditional)

**Goal:** Remove any helper functions in `stmt.rs` that were referenced only by the now-deleted calibration block. If no orphans exist, this task is a no-op and is skipped.

### Step 4.1: Identify candidate orphans

- [ ] **Search `stmt.rs` for helpers that were only referenced from the deleted block.**

Common patterns from the deleted block:
- `self.discover_awq_projections()` — likely referenced elsewhere (AWQ test fixtures); not an orphan.
- A `HarnessConfig` construction helper — if any was extracted, check if it's used elsewhere.

```bash
# Replace <helper> with each candidate function name and check references:
git grep -n "<helper>" -- ':!*.snap' ':!*.md' ':!docs/'
```

If a helper has zero references outside `stmt.rs` AND zero references inside `stmt.rs` after commit 3, it's an orphan.

### Step 4.2: Decide whether commit 4 is needed

- [ ] **If no orphans found:** skip to "Task 4 conclusion" below. **No commit 4.**

- [ ] **If orphans found:** delete them and continue with the cleanup commit.

### Step 4.3 (conditional): Delete orphaned helpers

- [ ] **Open `crates/nsl-codegen/src/stmt.rs`** and delete each orphaned function.

For each deletion:
1. Verify no references remain: `git grep -n "<helper_name>"`
2. Delete the function definition
3. Re-build: `cargo build -p nsl-codegen --tests`

### Step 4.4 (conditional): Run AWQ + snapshot + determinism + full suite

- [ ] `cargo test -p nsl-codegen --test awq_full_pipeline` — 6 PASS
- [ ] `cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline` — PASS
- [ ] `bash scripts/verify-awq-determinism.sh` — PASS
- [ ] `cargo test -p nsl-codegen` — all green (modulo `#[ignore]`)

### Step 4.5 (conditional): Commit

- [ ] **Stage:** `git add crates/nsl-codegen/src/stmt.rs`

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
chore(134): remove now-orphaned helpers from compile_train_block

Commit 4 of #134's Sequence C (spec §8.1, optional cleanup).
Deletes <list of orphaned helpers> in stmt.rs that were referenced
only by the calibration block removed in commit 3.

Verification:
- AWQ + snapshot + determinism + full suite all green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4 conclusion

If no orphans were found, this task is **skipped** entirely. The verification matrix collapses from 6 rows to 5; commit 5 follows commit 3 directly.

---

## Task 5 — Un-`#[ignore]` merge-gate test + fixture cleanup

**Goal:** Remove the `#[ignore]` attribute from `end_to_end_backward_subprocess_matches_analytical_reference`; update fixture header comments to reflect that #134 is now resolved (the workaround caveat no longer applies).

**Files:**

- Modify: `tests/fixtures/wggo_attention_mlp_real.nsl` — update header comments
- Modify: `crates/nsl-codegen/tests/wggo_backward_pipeline.rs` — remove `#[ignore]`

### Step 5.1: Update the fixture header comments

- [ ] **Open `tests/fixtures/wggo_attention_mlp_real.nsl`** at lines 6-16.

The current comment block says the test "bypasses `compile_and_calibrate`" because of "reduced-pipeline gaps (main signature collision, stdlib path resolution for train-block compilation) which issue #134 tracks for proper architectural fix."

After #134 lands, the test still uses `real_subprocess_entry` directly (because it's the canonical entry point under (c-i)), but the framing changes from "workaround" to "canonical."

- [ ] **Replace lines 6-16** with:

```
# The merge-gate test (`wggo_backward_pipeline.rs`) calls
# `real_subprocess_entry` directly — the canonical calibration entry
# point under #134's (c-i) convergence shape. `compile_and_calibrate`
# also invokes `real_subprocess_entry` post-`compile_main`, so this
# test exercises the same path that production calibration takes.
#
# As a result this fixture has no `train` block — calibration fires
# from `real_subprocess_entry` based on registered hooks, not from
# `compile_train_block` (whose calibration block was deleted in #134).
# The `@wggo_target` decorator on the forward method is the only
# declaration needed.
```

### Step 5.2: Remove the `#[ignore]` from the merge-gate test

- [ ] **Open `crates/nsl-codegen/tests/wggo_backward_pipeline.rs`** and locate the `#[ignore = "..."]` attribute on `end_to_end_backward_subprocess_matches_analytical_reference` (lines 179-200 per spec Appendix A; will be updated by PR #141 if not yet merged).

- [ ] **Delete the entire `#[ignore = "..."]` attribute line(s)** (the multi-line attribute starting `#[ignore = "Blocked on #134 ...` and ending `... See #134 for scope rationale."]`).

The test should now look like:

```rust
#[test]
fn end_to_end_backward_subprocess_matches_analytical_reference() {
    let data_path = fixture("wggo_calib_data.safetensors");
    // ... rest of test body unchanged ...
}
```

### Step 5.3: Run the merge-gate test (no longer ignored)

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test wggo_backward_pipeline 2>&1 | tail -20
```

Expected: `test end_to_end_backward_subprocess_matches_analytical_reference ... ok`.

If FAIL: the test hits a "next hop" beyond hop 6 that #134 didn't address. Capture the failure:

```bash
cargo test -p nsl-codegen --test wggo_backward_pipeline -- --nocapture 2>&1 | tail -60
```

Then **stop** and discuss with the user — a next-hop failure means #134's scope needs to expand or the next hop needs its own follow-up issue.

### Step 5.4: Run the AWQ behavioral tests

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline
```

Expected: `7 passed; 0 failed` (6 original + 1 new snapshot test).

### Step 5.5: Run the snapshot test

- [ ] **Run:**

```bash
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline
```

Expected: PASS. Snapshot unchanged from commit 1b.

### Step 5.6: Run the determinism script

- [ ] **Run:** `bash scripts/verify-awq-determinism.sh`

Expected: `PASS: all 11 runs produced byte-identical Sidecar JSON.`

### Step 5.7: Run the full test suite

- [ ] **Run:**

```bash
cargo test -p nsl-codegen 2>&1 | tail -10
```

Expected: all tests pass (no `#[ignore]` skips for the merge-gate test).

### Step 5.8: Commit

- [ ] **Stage:** `git add tests/fixtures/wggo_attention_mlp_real.nsl crates/nsl-codegen/tests/wggo_backward_pipeline.rs`

- [ ] **Commit:**

```bash
git commit -m "$(cat <<'EOF'
feat(134): un-#[ignore] WGGO Phase 2 merge-gate test + fixture cleanup

Commit 5 of #134's Sequence C (spec §8.1, final commit). Removes the
#[ignore] attribute from end_to_end_backward_subprocess_matches_
analytical_reference, which has been ignored since PR #132 with
sequentially refined ignore messages (PR #135 → #136 → #137 → #139 →
#140 → #141). #134's (c-i) convergence + hop 6 generalization + train-
block calibration deletion makes the test pass end-to-end.

Also updates wggo_attention_mlp_real.nsl's header comments: previously
described the bypass-compile_and_calibrate pattern as a workaround for
#134's pending reshape; now describes it as exercising the canonical
calibration entry point (real_subprocess_entry) that compile_and_
calibrate itself invokes post-compile_main.

The #[ignore] discipline pinned in PR #141 is honored to its
resolution: the message pointed at #134 as the resolution path; this
commit resolves #134, and the resolution is the un-ignore.

Verification:
- cargo test -p nsl-codegen --test wggo_backward_pipeline: 1/1 PASS
  (no --include-ignored needed)
- cargo test -p nsl-codegen --test awq_full_pipeline: 7/7 PASS
- snapshot_awq_sidecar_baseline: PASS (snapshot unchanged from 1b)
- scripts/verify-awq-determinism.sh: PASS
- cargo test -p nsl-codegen: all green

Closes #134.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (before pushing the PR)

- [ ] **Run the complete verification matrix** (spec §9):

```bash
# Row 1a-5: determinism
bash scripts/verify-awq-determinism.sh

# AWQ behavioral
cargo test -p nsl-codegen --test awq_full_pipeline

# Snapshot byte-equivalence
cargo test -p nsl-codegen --test awq_full_pipeline snapshot_awq_sidecar_baseline

# Merge-gate test (no --include-ignored)
cargo test -p nsl-codegen --test wggo_backward_pipeline

# Full nsl-codegen suite
cargo test -p nsl-codegen
```

All five must pass. If any fail, do NOT push — fix the regression first.

- [ ] **Run `cargo clippy` to catch dead code from deletions:**

```bash
cargo clippy -p nsl-codegen --tests 2>&1 | tail -30
```

Address any new warnings (unused imports, dead code) introduced by the deletions in commits 3-4.

- [ ] **Update `CHANGELOG-CALIBRATION.md`'s PR number** if you left it as `#TBD` in commit 1b:

```bash
# Find the implementation PR number you'll use when opening, then:
sed -i "s/PR #TBD/PR #<actual-number>/" CHANGELOG-CALIBRATION.md
git add CHANGELOG-CALIBRATION.md
git commit --amend --no-edit  # only if commit 1b is still local; otherwise add as a fixup
```

If commit 1b is already pushed, leave `#TBD` and update via a follow-up commit on the same branch.

- [ ] **Push the branch:**

```bash
git push -u origin feat/134-decouple-calibration
```

- [ ] **Open the PR:**

```bash
gh pr create --title "feat(134): decouple calibration harness from compile_train_block" \
  --body "$(cat <<'EOF'
## Summary

Implements [#134](https://github.com/bwiemz/NSL/issues/134) per the design spec at `docs/superpowers/specs/2026-05-06-134-decouple-calibration-design.md` (merged in PR #142). Six commits following Sequence C:

1. **1a — Determinism verification.** Adds `scripts/verify-awq-determinism.sh` + `SIDECAR_DUMP` instrumentation. Verifies AWQ sidecar is byte-deterministic before baseline capture.
2. **1b — Baseline + CHANGELOG + CI.** Captures `snapshot_awq_sidecar_baseline` insta snapshot. Creates `CHANGELOG-CALIBRATION.md`. Lands `.github/workflows/calibration-snapshot-changelog.yml` enforcement.
3. **2 — Hop 6 fix.** Generalizes `emit_calibration_model_object` to accept the union of AWQ + WGGO projections. AWQ flow byte-identical by inspection.
4. **3 — Wrapper-level firing (atomic move).** Calibration fires from `compile_and_calibrate` post-`compile_main`, not from `compile_train_block`. Path 1 (calibration block in `stmt.rs:3960-4046`) deleted.
5. **4 — Train-block cleanup.** (Conditional: skipped if commit 3 left no orphans.)
6. **5 — Un-`#[ignore]` merge-gate test.** WGGO Phase 2 merge-gate test passes end-to-end without `#[ignore]`. Fixture header comments updated.

## Verification

- [x] `cargo test -p nsl-codegen --test awq_full_pipeline`: 7/7 PASS (6 original + 1 snapshot)
- [x] `cargo test -p nsl-codegen --test wggo_backward_pipeline`: 1/1 PASS (no `--include-ignored`)
- [x] `bash scripts/verify-awq-determinism.sh`: PASS (11 runs byte-identical)
- [x] `cargo test -p nsl-codegen`: all green
- [x] AWQ snapshot byte-identical to commit 1b's capture

## (c-i) zero-by-construction merge gate

Under #134's (c-i) convergence shape, the AWQ Sidecar JSON snapshot is unchanged from commit 1b through commit 5. The wrapper-level firing fires against the same compiled binary in the same order; only the orchestration call site moves. Any drift would have indicated an implementation bug — none was observed.

Closes #134.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist

Before declaring the plan complete:

- [ ] Every task has explicit file paths, real code blocks, runnable commands, and expected outputs.
- [ ] Type names match across tasks: `DiscoveredProjection` (not `ProjectionMeta`), `WggoGradTarget`, `HarnessConfig`, `HarnessMode`, `HookRegistry`.
- [ ] No `TODO`, `TBD`, or "implement later" markers in any step.
- [ ] Sequence C 1a/1b split is honored: 1a's determinism verification is independent of 1b's baseline capture.
- [ ] Commit 3 is the atomic move (no parallel-firing intermediate state).
- [ ] Commit 4 is explicitly marked conditional / skippable.
- [ ] Each commit message references the spec section it implements.
- [ ] The verification matrix (spec §9) is exercised at every commit boundary.
- [ ] AWQ "byte-identical by inspection" claim is verified by the snapshot test at every commit.

---

## Appendix A — Reference paths and types

- Spec: `docs/superpowers/specs/2026-05-06-134-decouple-calibration-design.md`
- Issue: [#134](https://github.com/bwiemz/NSL/issues/134)
- Stacks on:
  - PR #139 (merged) — WGGO Phase 2 IR/FFI infrastructure
  - PR #140 (merged) — `ObservationSet::BackwardGradients` semantics
  - PR #141 (open) — pin merge-gate `#[ignore]` to #134
  - PR #142 (merged) — #134 design spec

Key types:

- `crate::calibration::discovery::DiscoveredProjection` (NOT `ProjectionMeta`) — fields: `projection: ProjectionRef`, `weight_shape: [u32; 2]`
- `crate::calibration::discovery::WggoGradTarget` — fields: `layer_key`, `class_name`, `head_dim`, `w_{q,k,v,o}: ProjectionRef`, `w_{q,k,v,o}_shape: [u32; 2]`, `w_{q,k,v,o}_index: u32`
- `crate::CompileOptions` — fields used here: `calibration_retention: Option<Vec<DiscoveredProjection>>`, `calibration_grad_retention: Option<Vec<WggoGradTarget>>`, `calibration_data: Option<PathBuf>`, `calibration_mode: Option<String>`, `weight_file: Option<PathBuf>`, `calibration_compile_bundle: Option<Arc<CalibrationCompileBundle>>`
- `crate::calibration::sidecar::Sidecar` — `hooks: BTreeMap<String, Vec<u8>>` (deterministic by construction)

Key code locations:

- `crates/nsl-codegen/src/lib.rs:702-826` — `compile_and_calibrate`
- `crates/nsl-codegen/src/stmt.rs:3960-4046` — calibration block to delete (commit 3)
- `crates/nsl-codegen/src/calibration/binary_codegen.rs:62-258` — `real_subprocess_entry` (canonical entry)
- `crates/nsl-codegen/src/calibration/binary_codegen.rs:1749` — `emit_calibration_model_object` start
- `crates/nsl-codegen/src/calibration/binary_codegen.rs:1763-1779` — hop 6 location (commit 2 modifies)
- `crates/nsl-codegen/src/calibration/discovery.rs:64-117` — `DiscoveredProjection` and `WggoGradTarget` definitions
- `crates/nsl-codegen/tests/awq_full_pipeline.rs:624` — `end_to_end_real_subprocess_matches_analytical_reference` (snapshot test mirrors its setup)
- `crates/nsl-codegen/tests/wggo_backward_pipeline.rs:179-201` — merge-gate test `#[ignore]` (PR #141 pins; commit 5 removes)
