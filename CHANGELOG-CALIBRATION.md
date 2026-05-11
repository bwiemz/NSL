# CHANGELOG — Calibration

Dedicated changelog for calibration-affecting changes. Per #134 §6.3, any
PR modifying `crates/nsl-codegen/tests/snapshots/awq_*.snap` (or other
calibration snapshot files) MUST add a corresponding entry here.
CI enforces the pairing (`.github/workflows/calibration-snapshot-changelog.yml`).

**Entry format:**

    ## YYYY-MM-DD — PR #NNN — <short title>

    **Snapshot files:** <comma-separated list>
    **Cause:** <one sentence: what changed and why>
    **Bit-equivalence evidence:** <how the impact on AWQ behavior was verified>

---

## 2026-05-06 — PR #145 — Initial AWQ sidecar baseline

**Snapshot files:** `crates/nsl-codegen/tests/snapshots/awq_full_pipeline__awq_sidecar_baseline.snap`

**Cause:** Initial baseline capture for #134's (c-i) regression discipline.
Captures the AWQ Sidecar JSON from `end_to_end_real_subprocess_matches_analytical_reference`'s
fixture under the unmodified `main` branch (commit 1a verified determinism
via `scripts/verify-awq-determinism.sh`).

**Bit-equivalence evidence:** Snapshot captured from unmodified `main`
before any #134 logic changes. Determinism verified across 11 runs
(10 serial + 1 thread-varied). Subsequent commits in #134 (hop 6 fix,
wrapper-level firing, train-block deletion, un-ignore) must NOT change
this snapshot per (c-i)'s zero-by-construction merge gate.
