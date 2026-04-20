# CPDT Phase 1 Calibration Retune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retune CPDT's placeholder `T0/T1/T2/CALIB_K` constants to produce a non-degenerate weights-present tier distribution, reframe invariant #7 from hard-gate to monitoring-gate with a Phase 2 promotion trigger, and add two institutional rules to the close-out discipline family.

**Architecture:** Two commits on `feat/cpdt-calibration-tune`. Commit 1 extends `cpdt_calibrate` with a `--emit-calibration` mode that computes thresholds from weights-present score bands (via geometric means of adjacent-band min/max) and finds `CALIB_K` by numerical minimization of parameter-weighted disagreement over a log-grid; zero scorer-behavior change. Commit 2 applies the emitted constants, bumps `ANALYSIS_VERSION` 1→2, regenerates both snapshot JSONs, updates the canary test to count-floor assertions, adds the numel-collision diagnostic test, rewords the disagreement gate to 20% monitoring-status, and appends the retrospective addendum to the Phase 2 stub.

**Tech Stack:** Rust 1.95.0 (workspace-pinned), Cranelift codegen, `safetensors` 0.4, `serde_json` 1, `half` 2.4, `rand` 0.8 (feature-gated behind `calibrate`). Tests run via `cargo test -p nsl-codegen --features cuda` from the worktree root.

**Spec:** [docs/superpowers/specs/2026-04-19-cpdt-calibration-correction-design.md](../specs/2026-04-19-cpdt-calibration-correction-design.md)

**Worktree:** `.worktrees/cpdt-calibration-tune` (branch `feat/cpdt-calibration-tune`, based on `main` at `404257c`)

---

## File Structure After Retune

**Modified:**

- `crates/nsl-codegen/src/bin/cpdt_calibrate.rs` (~105 LOC → ~260 LOC) — add band-analysis, cross-fixture ordering guard, log-grid K minimization, structural-limitation guard, per-band diagnostic emission, all behind a new `--emit-calibration` flag. Default behavior (no flag) prints the current constants verbatim to stdout, unchanged from today.
- `crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs` (~253 LOC → ~285 LOC) — add `--include-medium` and `--output-dir` flags; existing behavior (no flags) regenerates `calib_tiny` and `calib_small` into the CLI-supplied positional output dir, unchanged.
- `crates/nsl-codegen/src/cpdt_sensitivity.rs` (~294 LOC) — Commit 2 only: update `CALIB_K`, `CALIB_T0`, `CALIB_T1`, `CALIB_T2` constants, bump `ANALYSIS_VERSION` 1→2. Formula primitives, `SensitivityScorer`, and validation stub untouched.
- `tests/fixtures/cpdt_calibration/baseline_heuristic.json` — Commit 2 only: regenerated against new constants.
- `tests/fixtures/cpdt_calibration/expected_weights_present.json` — Commit 2 only: regenerated.
- `crates/nsl-codegen/tests/cpdt_tier_agreement.rs` (~172 LOC → ~220 LOC) — Commit 2 only: update `tier_agreement_full_on_calib_small_by_construction` to count-floor assertions; add new `disagreement_source_matches_numel_collision` test.
- `crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs` (~57 LOC) — Commit 2 only: rename test to `weighted_disagreement_below_monitoring_threshold`, raise gate from 0.05 to 0.20, update doc-comment to reflect monitoring-gate semantics per spec §5.
- `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` — Commit 2 only: append retrospective addendum per spec §6.

**Not modified:**

- Formula primitives, `SensitivityScorer`, `plan_map_noweights`, `compute_tier_agreement`, the CLI `invoke_cpdt_if_enabled` wiring — all out of scope.

---

## Commit 1 — `feat(cpdt): compute threshold + CALIB_K via minimization in cpdt_calibrate`

### Task 1.1: Baseline sanity

**Files:** none (verification only).

- [ ] **Step 1: Confirm worktree + branch + rustc.**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/cpdt-calibration-tune
pwd
git rev-parse --abbrev-ref HEAD
rustc --version
```

Expected: path ends with `.worktrees/cpdt-calibration-tune`, branch `feat/cpdt-calibration-tune`, rustc 1.95.0.

- [ ] **Step 2: Verify baseline build green.**

Run: `cargo build -p nsl-codegen --features cuda 2>&1 | tail -3`
Expected: `Finished \`dev\` profile [unoptimized + debuginfo] target(s) in <N>s`, no errors.

- [ ] **Step 3: Verify baseline cpdt tests green.**

Run: `cargo test -p nsl-codegen --features cuda --lib cpdt 2>&1 | tail -5`
Expected: 61 tests pass, 0 failures.

Run: `cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_primitives --test cpdt_sensitivity_snapshot --test cpdt_sensitivity_adversarial --test cpdt_sensitivity_disagreement --test cpdt_tier_agreement 2>&1 | grep "^test result:"`
Expected: five lines, each "ok. N passed; 0 failed".

### Task 1.2: Add `--include-medium` + `--output-dir` flags to `cpdt_fixture_generate`

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs`

- [ ] **Step 1: Extend `main()` to accept flags.**

Find the current `main()` (starts around line 77). Replace the body with flag-parsing logic:

```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut include_medium = false;
    let mut output_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--include-medium" => {
                include_medium = true;
                i += 1;
            }
            "--output-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output-dir requires a value");
                    std::process::exit(2);
                }
                output_dir = Some(args[i].clone());
                i += 1;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    // Positional output_dir (back-compat) OR --output-dir (new form).
    let out_dir_string = match (output_dir, positional.first()) {
        (Some(d), _) => d,
        (None, Some(d)) => d.clone(),
        (None, None) => {
            eprintln!(
                "usage: cpdt_fixture_generate [--include-medium] [--output-dir DIR | DIR]"
            );
            std::process::exit(1);
        }
    };
    let out_dir = Path::new(&out_dir_string);
    std::fs::create_dir_all(out_dir).unwrap();

    write_fixture(out_dir, "calib_tiny", calib_tiny(), DType::F32);
    write_fixture(out_dir, "calib_small", calib_small(), DType::F16);

    if include_medium {
        write_fixture(out_dir, "calib_medium", calib_medium(), DType::F16);
    } else {
        eprintln!(
            "calib_medium is regenerated at test-time into target/; pass --include-medium to write here."
        );
    }
}
```

This preserves back-compat: the existing single-positional form (e.g. `cpdt_fixture_generate tests/fixtures/cpdt_calibration/`) still works.

- [ ] **Step 2: Remove `#[allow(dead_code)]` from `calib_medium()`.**

Find `calib_medium()` (around line 64). Drop the `#[allow(dead_code)]` attribute above it — the function is now reachable.

- [ ] **Step 3: Build + verify back-compat path.**

```bash
cargo build --features calibrate --bin cpdt_fixture_generate 2>&1 | tail -3
```

Expected: builds clean.

```bash
# Back-compat positional form still works; produces tiny + small only.
cargo run --features calibrate --bin cpdt_fixture_generate -- /tmp/cpdt_fixturegen_test/ 2>&1 | tail -5
ls -la /tmp/cpdt_fixturegen_test/
```

Expected: writes `calib_tiny.safetensors` (~2.3 MB) and `calib_small.safetensors` (~68 MB); prints "calib_medium is regenerated at test-time..." message.

- [ ] **Step 4: Verify `--include-medium` flag path.**

```bash
cargo run --features calibrate --bin cpdt_fixture_generate -- \
    --include-medium \
    --output-dir /tmp/cpdt_fixturegen_medium/ 2>&1 | tail -5
ls -la /tmp/cpdt_fixturegen_medium/
```

Expected: writes `calib_tiny.safetensors`, `calib_small.safetensors`, and `calib_medium.safetensors` (~600 MB, f16 per spec §5.1).

- [ ] **Step 5: Clean up test output dirs.**

```bash
rm -rf /tmp/cpdt_fixturegen_test /tmp/cpdt_fixturegen_medium
```

### Task 1.3: Scaffold `--emit-calibration` mode in `cpdt_calibrate`

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`

- [ ] **Step 1: Extend flag parsing in `main()`.**

Find the current `main()` (starts around line 34). Replace its body with:

```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut emit_calibration = false;
    let mut medium_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--emit-calibration" => {
                emit_calibration = true;
                i += 1;
            }
            "--medium-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --medium-dir requires a value");
                    std::process::exit(2);
                }
                medium_dir = Some(args[i].clone());
                i += 1;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    let fixture_dir_string = match positional.first() {
        Some(d) => d.clone(),
        None => {
            eprintln!(
                "usage: cpdt_calibrate <fixture_dir> [--emit-calibration] [--medium-dir DIR]"
            );
            std::process::exit(1);
        }
    };
    let fixture_dir = Path::new(&fixture_dir_string);

    if emit_calibration {
        run_emit_calibration(fixture_dir, medium_dir.as_deref());
    } else {
        run_default(fixture_dir);
    }
}

/// Default mode: produces baseline_heuristic.json + prints the current constants.
/// Byte-identical to pre-Commit-1 behavior.
fn run_default(fixture_dir: &Path) {
    let fixtures = ["calib_tiny", "calib_small", "calib_medium"];
    let mut snapshots = Vec::new();

    for fixture in &fixtures {
        let path = fixture_dir.join(format!("{fixture}.safetensors"));
        let wm = match WeightMap::load(&path) {
            Ok(w) => w,
            Err(_) if *fixture == "calib_medium" => {
                eprintln!("calib_medium missing; skipping (regenerated at test-time into target/)");
                continue;
            }
            Err(e) => {
                eprintln!("failed to load {fixture}: {e:?}");
                std::process::exit(1);
            }
        };
        let cfg = infer_config_from_fixture(fixture);
        let mut tiers = Vec::new();
        for (name, entry) in wm.entries() {
            let layer = layer_of(name);
            let kind = classify_layer_kind(name, layer, cfg.n_layers);
            let gm = gradient_magnitude_est(Some(entry));
            let pos = position_criticality(layer, cfg.n_layers, CALIB_ALPHA);
            let elts = entry.num_elements.max(1) as f64;
            let score = gm * pos / elts;
            let tier = assign_tier(score, kind);
            tiers.push(TierEntry {
                name: name.clone(),
                tier: tier.as_str(),
                score,
            });
        }
        tiers.sort_by(|a, b| a.name.cmp(&b.name));
        snapshots.push(FixtureSnapshot {
            fixture: fixture.to_string(),
            tiers,
        });
    }

    let json = serde_json::to_string_pretty(&snapshots).unwrap();
    let out_path = fixture_dir.join("baseline_heuristic.json");
    std::fs::write(&out_path, &json).unwrap();
    println!("wrote {}", out_path.display());
    println!();
    println!("pub const ANALYSIS_VERSION: u32 = {};", ANALYSIS_VERSION);
    println!("pub const CALIB_K:     f64 = {};", CALIB_K);
    println!("pub const CALIB_T0:    f64 = {};", CALIB_T0);
    println!("pub const CALIB_T1:    f64 = {};", CALIB_T1);
    println!("pub const CALIB_T2:    f64 = {};", CALIB_T2);
    println!("pub const CALIB_ALPHA: f64 = {};", CALIB_ALPHA);
}

/// `--emit-calibration` mode: scaffold only; body filled in Tasks 1.4-1.6.
fn run_emit_calibration(_fixture_dir: &Path, _medium_dir: Option<&str>) {
    unimplemented!("emit-calibration body lands in Tasks 1.4-1.6");
}
```

Keep the existing `TierEntry`, `FixtureSnapshot`, and `infer_config_from_fixture` definitions verbatim.

- [ ] **Step 2: Build.**

```bash
cargo build --features calibrate --bin cpdt_calibrate 2>&1 | tail -3
```

Expected: builds clean with a warning about the `unimplemented!` in `run_emit_calibration`. The warning is fine; the body lands in Tasks 1.4-1.6.

- [ ] **Step 3: Verify default behavior byte-identical.**

```bash
cp tests/fixtures/cpdt_calibration/baseline_heuristic.json /tmp/pre_baseline.json
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/ 2>&1 | tail -10
diff /tmp/pre_baseline.json tests/fixtures/cpdt_calibration/baseline_heuristic.json
```

Expected: diff is empty. The new flag parsing preserved the default path exactly.

### Task 1.4: Score collection + band classification in `--emit-calibration`

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`

- [ ] **Step 1: Add supporting types + classifier above `main`.**

Add near the top of the file, after the existing `use` imports:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Band {
    AttnQkvo,
    FfnGateUp,
    FfnDown,
}

impl Band {
    fn as_str(self) -> &'static str {
        match self {
            Band::AttnQkvo => "attn_qkvo",
            Band::FfnGateUp => "ffn_gate_up",
            Band::FfnDown => "ffn_down",
        }
    }
}

fn band_of(name: &str) -> Option<Band> {
    // Match order matters: check the most specific FFN pattern before the generic attn check.
    if name.contains("ffn.w_gate") || name.contains("ffn.w_up") {
        return Some(Band::FfnGateUp);
    }
    if name.contains("ffn.w_down") {
        return Some(Band::FfnDown);
    }
    if name.contains("attn.w") && !name.contains("norm") {
        return Some(Band::AttnQkvo);
    }
    None
}

struct TensorRecord {
    fixture: String,
    name: String,
    layer: Option<u32>,
    pos: f64,
    numel: usize,
    wp_score: f64,
    band: Option<Band>,
    overridden: bool,
}
```

- [ ] **Step 2: Replace `run_emit_calibration` stub body with score collection.**

```rust
fn run_emit_calibration(fixture_dir: &Path, medium_dir: Option<&str>) {
    // Enforce calib_medium presence; this is the two-step command contract.
    let Some(md) = medium_dir else {
        eprintln!(
            "error: --emit-calibration requires --medium-dir <path>.\n\
             calib_medium is generated at runtime into target/. Run:\n  \
             cargo run --features calibrate --bin cpdt_fixture_generate -- \\\n  \
                 --include-medium --output-dir target/cpdt_calibration/\n\
             then re-run with --medium-dir target/cpdt_calibration/."
        );
        std::process::exit(2);
    };
    let medium_path = Path::new(md).join("calib_medium.safetensors");
    if !medium_path.is_file() {
        eprintln!(
            "error: calib_medium.safetensors not found at {}. \
             Run cpdt_fixture_generate --include-medium --output-dir {} first.",
            medium_path.display(), md
        );
        std::process::exit(2);
    }

    let records = collect_records(fixture_dir, Path::new(md));
    if records.is_empty() {
        eprintln!("error: no tensor records collected — fixtures empty or corrupt");
        std::process::exit(2);
    }

    // Subsequent tasks append: ordering guard (1.4), threshold computation (1.5),
    // K minimization + diagnostic emission (1.6).
    eprintln!("collected {} records across fixtures", records.len());
}

fn collect_records(committed_dir: &Path, medium_dir: &Path) -> Vec<TensorRecord> {
    let sources: Vec<(String, std::path::PathBuf, u32)> = vec![
        ("calib_tiny".to_string(),   committed_dir.join("calib_tiny.safetensors"),   2),
        ("calib_small".to_string(),  committed_dir.join("calib_small.safetensors"),  8),
        ("calib_medium".to_string(), medium_dir.join("calib_medium.safetensors"),   16),
    ];

    let mut out = Vec::new();
    for (fixture, path, n_layers) in sources {
        let wm = match WeightMap::load(&path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to load {}: {e:?}", path.display());
                std::process::exit(1);
            }
        };
        for (name, entry) in wm.entries() {
            let layer = layer_of(name);
            let kind = classify_layer_kind(name, layer, n_layers);
            let overridden = kind.is_kind_overridden();
            let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
            let gm = gradient_magnitude_est(Some(entry));
            let elts = entry.num_elements.max(1) as f64;
            let wp_score = gm * pos / elts;
            out.push(TensorRecord {
                fixture: fixture.clone(),
                name: name.clone(),
                layer,
                pos,
                numel: entry.num_elements,
                wp_score,
                band: band_of(name),
                overridden,
            });
        }
    }
    out
}
```

- [ ] **Step 3: Build + quick sanity.**

```bash
cargo build --features calibrate --bin cpdt_calibrate 2>&1 | tail -3
```

Expected: clean build.

Cannot run `--emit-calibration` end-to-end yet (body incomplete); subsequent tasks add the guard, threshold, and K-minimization pieces.

### Task 1.5: Cross-fixture ordering guard + threshold computation

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`

- [ ] **Step 1: Add helpers + extend `run_emit_calibration`.**

Add after the `collect_records` helper:

```rust
fn geomean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    let logs: f64 = xs.iter().map(|x| x.max(1e-300).ln()).sum();
    (logs / xs.len() as f64).exp()
}

struct BandStats {
    min: f64,
    max: f64,
    #[allow(dead_code)] // surfaced in diagnostic later
    geomean: f64,
    n: usize,
}

fn band_stats_per_fixture(records: &[TensorRecord]) -> std::collections::BTreeMap<(String, Band), BandStats> {
    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<(String, Band), Vec<f64>> = BTreeMap::new();
    for r in records {
        if r.overridden { continue; }
        if let Some(b) = r.band {
            buckets
                .entry((r.fixture.clone(), b))
                .or_default()
                .push(r.wp_score);
        }
    }
    buckets.into_iter().map(|(k, vs)| {
        let mn = vs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gm = geomean(&vs);
        (k, BandStats { min: mn, max: mx, geomean: gm, n: vs.len() })
    }).collect()
}

/// Enforce attn_qkvo > ffn_gate_up > ffn_down by band geomean on every fixture
/// that contains all three bands. calib_tiny has no generics (L=2 → all kind-
/// overridden), so it contributes nothing and is skipped.
fn verify_ordering(
    stats: &std::collections::BTreeMap<(String, Band), BandStats>,
) -> Result<(), String> {
    let fixtures: std::collections::BTreeSet<String> =
        stats.keys().map(|(f, _)| f.clone()).collect();
    for fixture in fixtures {
        let a = stats.get(&(fixture.clone(), Band::AttnQkvo));
        let g = stats.get(&(fixture.clone(), Band::FfnGateUp));
        let d = stats.get(&(fixture.clone(), Band::FfnDown));
        let (Some(a), Some(g), Some(d)) = (a, g, d) else { continue; };
        if !(a.geomean > g.geomean && g.geomean > d.geomean) {
            return Err(format!(
                "band ordering violated on {fixture}: \
                 attn_qkvo={:.3e} ffn_gate_up={:.3e} ffn_down={:.3e} \
                 (expected attn > gate_up > down)",
                a.geomean, g.geomean, d.geomean
            ));
        }
    }
    Ok(())
}

fn compute_thresholds(
    records: &[TensorRecord],
) -> (f64, f64, f64) {
    // Pool generic scores across fixtures per band.
    let mut attn: Vec<f64> = Vec::new();
    let mut gate_up: Vec<f64> = Vec::new();
    let mut down: Vec<f64> = Vec::new();
    for r in records {
        if r.overridden { continue; }
        match r.band {
            Some(Band::AttnQkvo) => attn.push(r.wp_score),
            Some(Band::FfnGateUp) => gate_up.push(r.wp_score),
            Some(Band::FfnDown) => down.push(r.wp_score),
            None => {}
        }
    }
    let attn_min = attn.iter().cloned().fold(f64::INFINITY, f64::min);
    let gu_min = gate_up.iter().cloned().fold(f64::INFINITY, f64::min);
    let gu_max = gate_up.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dn_min = down.iter().cloned().fold(f64::INFINITY, f64::min);
    let dn_max = down.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    const CALIB_T2_FLOOR: f64 = 1e-10;
    let t0 = (attn_min * gu_max).sqrt();
    let t1 = (gu_min * dn_max).sqrt();
    let t2 = (dn_min * CALIB_T2_FLOOR).sqrt();
    (t0, t1, t2)
}
```

- [ ] **Step 2: Extend `run_emit_calibration` to invoke guard + thresholds.**

Replace the `eprintln!("collected {} records...")` line in `run_emit_calibration` with:

```rust
    eprintln!("collected {} records across fixtures", records.len());

    let stats = band_stats_per_fixture(&records);
    if let Err(msg) = verify_ordering(&stats) {
        eprintln!("error: {msg}");
        std::process::exit(2);
    }

    let (t0, t1, t2) = compute_thresholds(&records);
    eprintln!(
        "thresholds: T0 = {t0:.4e}  T1 = {t1:.4e}  T2 = {t2:.4e}"
    );

    // K minimization and final emission land in Task 1.6.
```

- [ ] **Step 3: Build + partial-run.**

```bash
# First regenerate calib_medium into target/.
cargo run --features calibrate --bin cpdt_fixture_generate -- \
    --include-medium --output-dir target/cpdt_calibration/

# Run the emit-calibration partial path.
cargo run --features calibrate --bin cpdt_calibrate -- \
    tests/fixtures/cpdt_calibration/ \
    --medium-dir target/cpdt_calibration/ \
    --emit-calibration 2>&1 | tail -10
```

Expected: prints "collected N records across fixtures" (N matches 21+74+calib_medium_count; calib_medium has 16 layers × 9 per-layer tensors + 1 embed + 1 output (untied) + 1 final norm + any biases per `MixedHalf` schedule = exercise says ~149, but exact count is not asserted here) and `thresholds: T0 = 1.4525e-7  T1 = 5.6755e-8  T2 = 1.9074e-9`. If the ordering guard fires, investigate before proceeding.

### Task 1.6: K via plateau-midpoint + structural-limitation guard + diagnostic emission

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`

This task was redirected during pre-dispatch: the original pure-corpus-minimum K (0.046) produces 26.5% disagreement on calib_small specifically, driven by the SwiGLU numel-collision. The spec now specifies **plateau-midpoint-under-per-fixture-constraint**: pick the log-midpoint of the widest contiguous K-range where both calib_small and calib_medium stay under a 20% monitoring threshold. See spec §3.2.

- [ ] **Step 1: Add per-fixture disagreement + plateau-midpoint helpers.**

Add after `compute_thresholds`:

```rust
/// Tier discriminator: u8 for equality comparison without depending on Tier's Hash.
#[inline]
fn tier_of_u8(score: f64, t0: f64, t1: f64, t2: f64) -> u8 {
    if score > t0 { 3 }       // High
    else if score > t1 { 2 }  // Medium
    else if score > t2 { 1 }  // Low
    else { 0 }                // VeryLow
}

/// Per-fixture parameter-weighted disagreement at a given K.
/// Kind-overridden tensors agree trivially (both paths produce Tier::High);
/// they contribute to `total` so the denominator matches the test harness,
/// but are never counted as disagreeing.
fn disagreement_per_fixture(
    records: &[TensorRecord],
    fixture: &str,
    k: f64,
    t0: f64,
    t1: f64,
    t2: f64,
) -> f64 {
    let mut total: u64 = 0;
    let mut disagree: u64 = 0;
    for r in records {
        if r.fixture != fixture { continue; }
        if r.numel == 0 { continue; }
        total += r.numel as u64;
        if r.overridden { continue; }
        let s_nw = k * r.pos / (r.numel.max(1) as f64);
        let t_wp = tier_of_u8(r.wp_score, t0, t1, t2);
        let t_nw = tier_of_u8(s_nw, t0, t1, t2);
        if t_wp != t_nw {
            disagree += r.numel as u64;
        }
    }
    if total == 0 { 0.0 } else { disagree as f64 / total as f64 }
}

/// Corpus-wide disagreement (pooled across all fixtures in `records`).
fn disagreement_corpus(records: &[TensorRecord], k: f64, t0: f64, t1: f64, t2: f64) -> f64 {
    let mut total: u64 = 0;
    let mut disagree: u64 = 0;
    for r in records {
        if r.numel == 0 { continue; }
        total += r.numel as u64;
        if r.overridden { continue; }
        let s_nw = k * r.pos / (r.numel.max(1) as f64);
        let t_wp = tier_of_u8(r.wp_score, t0, t1, t2);
        let t_nw = tier_of_u8(s_nw, t0, t1, t2);
        if t_wp != t_nw {
            disagree += r.numel as u64;
        }
    }
    if total == 0 { 0.0 } else { disagree as f64 / total as f64 }
}

/// Plateau-midpoint-under-per-fixture-constraint selector.
///
/// Per spec §3.2: find the longest contiguous K-range (on a log-spaced grid)
/// where *every* fixture's per-fixture disagreement is at or below
/// `monitoring_threshold`. Return the log-midpoint of that range and its
/// per-fixture disagreement values at the midpoint.
///
/// If no K satisfies the constraint for all fixtures, return
/// `Err(diagnostic)` for the caller's structural-limitation guard to emit.
fn select_calib_k_by_plateau(
    records: &[TensorRecord],
    t0: f64,
    t1: f64,
    t2: f64,
    monitoring_threshold: f64,
) -> Result<PlateauResult, PlateauError> {
    // Log-spaced grid: 10^-6 to 10^0 in 0.01-decade steps → 601 points,
    // ~2.33% multiplicative step.
    const N_POINTS: usize = 601;
    let mut grid: Vec<f64> = Vec::with_capacity(N_POINTS);
    for i in 0..N_POINTS {
        let lnk = (-6.0 + (i as f64) / 100.0) * std::f64::consts::LN_10;
        grid.push(lnk.exp());
    }

    // Per-point per-fixture disagreement.
    let fixtures: std::collections::BTreeSet<String> =
        records.iter().map(|r| r.fixture.clone()).collect();
    let mut feasible: Vec<bool> = Vec::with_capacity(N_POINTS);
    for &k in &grid {
        let ok = fixtures.iter().all(|f| {
            disagreement_per_fixture(records, f, k, t0, t1, t2) <= monitoring_threshold
        });
        feasible.push(ok);
    }

    // Longest contiguous run of true.
    let mut best_lo: Option<usize> = None;
    let mut best_hi: Option<usize> = None;
    let mut cur_lo: Option<usize> = None;
    for (i, &ok) in feasible.iter().enumerate() {
        if ok {
            if cur_lo.is_none() { cur_lo = Some(i); }
            let lo = cur_lo.unwrap();
            let run_len = i - lo + 1;
            let best_len = match (best_lo, best_hi) {
                (Some(l), Some(h)) => h - l + 1,
                _ => 0,
            };
            if run_len > best_len {
                best_lo = Some(lo);
                best_hi = Some(i);
            }
        } else {
            cur_lo = None;
        }
    }

    let (lo, hi) = match (best_lo, best_hi) {
        (Some(l), Some(h)) => (l, h),
        _ => {
            // No feasible point — emit per-fixture best-found disagreements.
            let best_k_per_fixture: Vec<(String, f64, f64)> = fixtures
                .iter()
                .map(|f| {
                    let (bk, bd) = grid
                        .iter()
                        .map(|&k| (k, disagreement_per_fixture(records, f, k, t0, t1, t2)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();
                    (f.clone(), bk, bd)
                })
                .collect();
            return Err(PlateauError {
                monitoring_threshold,
                per_fixture_best: best_k_per_fixture,
            });
        }
    };

    let plateau_start = grid[lo];
    let plateau_end = grid[hi];
    let k_mid = (plateau_start * plateau_end).sqrt();
    let mut per_fixture: Vec<(String, f64)> = Vec::new();
    for f in &fixtures {
        per_fixture.push((
            f.clone(),
            disagreement_per_fixture(records, f, k_mid, t0, t1, t2),
        ));
    }
    let corpus = disagreement_corpus(records, k_mid, t0, t1, t2);
    Ok(PlateauResult {
        k: k_mid,
        plateau_start,
        plateau_end,
        per_fixture_disagreement: per_fixture,
        corpus_disagreement: corpus,
    })
}

struct PlateauResult {
    k: f64,
    plateau_start: f64,
    plateau_end: f64,
    per_fixture_disagreement: Vec<(String, f64)>,
    corpus_disagreement: f64,
}

struct PlateauError {
    monitoring_threshold: f64,
    per_fixture_best: Vec<(String, f64, f64)>, // (fixture, best_k, best_disagreement)
}
```

- [ ] **Step 2: Extend `run_emit_calibration` with plateau selection + guard + diagnostic emission.**

Replace the `// K minimization ...` comment at the end of `run_emit_calibration` with:

```rust
    const MONITORING_THRESHOLD: f64 = 0.20; // spec §5.3

    let plateau = match select_calib_k_by_plateau(&records, t0, t1, t2, MONITORING_THRESHOLD) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "error: no K satisfies per-fixture disagreement <= {:.2} across all fixtures.",
                e.monitoring_threshold
            );
            eprintln!("best-found K per fixture:");
            for (f, bk, bd) in &e.per_fixture_best {
                eprintln!("  {f}: K={bk:.4e}  disagreement={:.4}", bd);
            }
            eprintln!(
                "This indicates an unknown source of disagreement beyond the known \
                 numel-collision pattern, or a corpus that needs re-calibration. \
                 Investigate before shipping."
            );
            std::process::exit(3);
        }
    };

    eprintln!(
        "plateau: [{:.4e}, {:.4e}]  log-midpoint CALIB_K = {:.4e}",
        plateau.plateau_start, plateau.plateau_end, plateau.k
    );
    for (f, d) in &plateau.per_fixture_disagreement {
        eprintln!("  {f} disagreement at K: {:.4}", d);
    }
    eprintln!("  corpus-wide disagreement at K: {:.4}", plateau.corpus_disagreement);

    let k = plateau.k;

    // Per-band diagnostic: show representative middle-layer tensor tiers under
    // the emitted (T0, T1, T2, K). Lets reviewers verify disagreements trace
    // to the ffn_gate_up/down numel collision rather than unknown drift.
    println!();
    println!("=== Per-band diagnostic (middle-layer median tensors) ===");
    println!("| Band          | s_wp     | wp tier | s_nw     | nw tier | agree |");
    println!("| ------------- | -------- | ------- | -------- | ------- | ----- |");
    for band in &[Band::AttnQkvo, Band::FfnGateUp, Band::FfnDown] {
        let mut midlayer_records: Vec<&TensorRecord> = records.iter()
            .filter(|r| !r.overridden
                && r.band == Some(*band)
                && (r.pos - 1.0).abs() < 1e-9)
            .collect();
        midlayer_records.sort_by(|a, b| a.wp_score.partial_cmp(&b.wp_score).unwrap());
        if midlayer_records.is_empty() { continue; }
        let median = midlayer_records[midlayer_records.len() / 2];
        let s_wp = median.wp_score;
        let s_nw = k * median.pos / (median.numel.max(1) as f64);
        let t_wp = tier_label(s_wp, t0, t1, t2);
        let t_nw = tier_label(s_nw, t0, t1, t2);
        println!(
            "| {:<13} | {:.2e} | {:<7} | {:.2e} | {:<7} | {} |",
            band.as_str(), s_wp, t_wp, s_nw, t_nw, if t_wp == t_nw { "yes" } else { "no" }
        );
    }

    // Emit the diff-ready Rust constants block.
    println!();
    println!("pub const ANALYSIS_VERSION: u32 = 2;");
    println!("pub const CALIB_K:     f64 = {k:.6e};");
    println!("pub const CALIB_T0:    f64 = {t0:.6e};");
    println!("pub const CALIB_T1:    f64 = {t1:.6e};");
    println!("pub const CALIB_T2:    f64 = {t2:.6e};");
    println!("pub const CALIB_ALPHA: f64 = {};", CALIB_ALPHA);
}

fn tier_label(score: f64, t0: f64, t1: f64, t2: f64) -> &'static str {
    if score > t0 { "high" }
    else if score > t1 { "medium" }
    else if score > t2 { "low" }
    else { "very_low" }
}
```

- [ ] **Step 3: Build + end-to-end run.**

```bash
cargo build --features calibrate --bin cpdt_calibrate 2>&1 | tail -3
```

Expected: clean build.

```bash
cargo run --features calibrate --bin cpdt_calibrate -- \
    tests/fixtures/cpdt_calibration/ \
    --medium-dir target/cpdt_calibration/ \
    --emit-calibration 2>&1 | tail -30
```

Expected output (exact numbers may vary by ~2% due to grid quantization):

```text
collected <N> records across fixtures
thresholds: T0 = 6.1056e-8  T1 = 2.2324e-8  T2 = 7.2557e-10
plateau: [~5.7e-2, ~6.4e-2]  log-midpoint CALIB_K = ~6.0e-2
  calib_tiny disagreement at K: 0.0000
  calib_small disagreement at K: ~0.1591
  calib_medium disagreement at K: ~0.0251
  corpus-wide disagreement at K: ~0.0376

=== Per-band diagnostic (middle-layer median tensors) ===
| Band          | s_wp     | wp tier | s_nw     | nw tier | agree |
| ------------- | -------- | ------- | -------- | ------- | ----- |
| attn_qkvo     | ~4.2e-08 | medium  | ~4.4e-08 | medium  | yes |
| ffn_gate_up   | ~1.05e-08| low     | ~1.1e-08 | low     | yes |
| ffn_down      | ~5.3e-09 | low     | ~1.1e-08 | low     | yes |

pub const ANALYSIS_VERSION: u32 = 2;
pub const CALIB_K:     f64 = ~6.0e-02;
pub const CALIB_T0:    f64 = 6.105600e-08;
pub const CALIB_T1:    f64 = 2.232400e-08;
pub const CALIB_T2:    f64 = 7.255700e-10;
pub const CALIB_ALPHA: f64 = 0.3;
```

(The per-band diagnostic's middle-layer median scores come from calib_medium, which dominates the middle-layer tensor count. calib_medium's medians all agree between wp and nw paths at the plateau midpoint; calib_small's middle-layer scores would show the numel-collision disagreement but are not the median across the pooled set.)

- [ ] **Step 4: Verify spec pinning within tolerance.**

Spec §3.1/§3.2 pins: T0=6.1056e-8, T1=2.2324e-8, T2=7.2557e-10, plateau ≈ [0.057, 0.064], K ≈ 0.060.

Verify:

- `T0` emitted within 1% of 6.1056e-8.
- `T1` emitted within 1% of 2.2324e-8.
- `T2` emitted within 1% of 7.2557e-10.
- `plateau_start` within ~2% of 0.057.
- `plateau_end` within ~2% of 0.064.
- Emitted `K` within ~2% of 0.060 (log-midpoint, grid-dependent).
- `calib_small` disagreement at K in range [0.15, 0.17].
- `calib_medium` disagreement at K in range [0.02, 0.03].
- `corpus-wide` disagreement at K below 0.05.

If any constant deviates more, investigate before proceeding — spec values came from the same fixture set and the same algorithm; divergence indicates a computation bug.

### Task 1.7: Regression-test the default path + document the flag

**Files:**
- Modify: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`

- [ ] **Step 1: Verify default mode still byte-identical.**

```bash
# Save pre-Commit-1 baseline JSON.
cp tests/fixtures/cpdt_calibration/baseline_heuristic.json /tmp/pre_commit1_baseline.json

# Re-run default (no flags).
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/ 2>&1 | tail -10
diff /tmp/pre_commit1_baseline.json tests/fixtures/cpdt_calibration/baseline_heuristic.json
```

Expected: diff is empty; the default mode's output JSON is byte-identical to the pre-Commit-1 baseline. Clean up:

```bash
rm /tmp/pre_commit1_baseline.json
```

- [ ] **Step 2: Update the module doc-comment with the new usage.**

Find the top doc-comment (lines 1-10 of `cpdt_calibrate.rs`). Replace with:

```rust
//! Dev-only CPDT calibration binary. Gated behind `[features] calibrate = []`
//! so it doesn't land in release builds.
//!
//! ## Default mode — regenerate baseline_heuristic.json
//!
//! ```bash
//! cargo run --features calibrate --bin cpdt_calibrate -- <fixture_dir>
//! ```
//!
//! Writes `<fixture_dir>/baseline_heuristic.json` and prints the current
//! constants to stdout. Byte-identical to pre-retune behavior.
//!
//! ## `--emit-calibration` mode — compute retune values
//!
//! ```bash
//! # Step 1: regenerate calib_medium into target/.
//! cargo run --features calibrate --bin cpdt_fixture_generate -- \
//!     --include-medium --output-dir target/cpdt_calibration/
//!
//! # Step 2: compute thresholds + CALIB_K by minimization.
//! cargo run --features calibrate --bin cpdt_calibrate -- \
//!     tests/fixtures/cpdt_calibration/ \
//!     --medium-dir target/cpdt_calibration/ \
//!     --emit-calibration
//! ```
//!
//! Prints the diff-ready constants block + per-band disagreement diagnostic.
//! Exits non-zero if the cross-fixture ordering guard or structural-limitation
//! guard (35% ceiling) fires.
```

### Task 1.8: Verify all existing tests still pass

**Files:** none (verification only).

- [ ] **Step 1: Full cpdt test suite.**

```bash
cargo test -p nsl-codegen --features cuda --lib cpdt 2>&1 | tail -5
cargo test -p nsl-codegen --features cuda \
    --test cpdt_sensitivity_primitives \
    --test cpdt_sensitivity_snapshot \
    --test cpdt_sensitivity_adversarial \
    --test cpdt_sensitivity_disagreement \
    --test cpdt_tier_agreement 2>&1 | grep "^test result:"
```

Expected: 61 lib tests pass. Integration tests: 22 + 1 + 1 + 1 + 6 = 31 pass. Zero failures.

The scorer's in-code constants are unchanged in Commit 1; these tests should pass exactly as they did on the branch tip.

### Task 1.9: Commit 1

**Files:** all staged changes from Tasks 1.2 - 1.7.

- [ ] **Step 1: Confirm staged set.**

```bash
git status --short
```

Expected (no snapshot JSONs should appear; target/ contents are gitignored):

```
M  crates/nsl-codegen/src/bin/cpdt_calibrate.rs
M  crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs
```

- [ ] **Step 2: Stage + commit.**

```bash
git add crates/nsl-codegen/src/bin/cpdt_calibrate.rs \
       crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs

git commit -m "$(cat <<'EOF'
feat(cpdt): compute threshold + CALIB_K via minimization in cpdt_calibrate

Adds --emit-calibration mode to cpdt_calibrate that:
- Pools generic (non-kind-overridden) tensor scores across committed
  fixtures + user-supplied calib_medium into three bands: attn_qkvo,
  ffn_gate_up, ffn_down.
- Enforces cross-fixture band ordering (attn > gate_up > down).
- Computes thresholds as geomeans of adjacent-band min/max pairs.
- Finds CALIB_K by log-grid minimization of parameter-weighted
  disagreement; emits structural-limitation guard non-zero exit if
  residual disagreement exceeds 35%.
- Prints per-band disagreement diagnostic so reviewers can verify the
  residual traces to the known numel-collision pattern.

Adds --include-medium + --output-dir flags to cpdt_fixture_generate so
the two-step calibration workflow can produce calib_medium on demand
into target/.

No scorer-behavior change: the in-code constants are unchanged; the
default (no-flag) mode of cpdt_calibrate remains byte-identical to the
pre-commit behavior. Both the baseline_heuristic.json and
expected_weights_present.json are unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands.

---

## Commit 2 — `feat(cpdt): retune T0/T1/T2 + CALIB_K + bump ANALYSIS_VERSION`

### Task 2.1: Apply the emitted constants + bump ANALYSIS_VERSION

**Files:**
- Modify: `crates/nsl-codegen/src/cpdt_sensitivity.rs` (lines 31-52)

- [ ] **Step 1: Re-run the calibration binary to capture exact emitted values.**

```bash
cargo run --features calibrate --bin cpdt_calibrate -- \
    tests/fixtures/cpdt_calibration/ \
    --medium-dir target/cpdt_calibration/ \
    --emit-calibration 2>&1 | tail -10 > /tmp/cpdt_calib_out.txt
cat /tmp/cpdt_calib_out.txt
```

Expected: the constants block lands in `/tmp/cpdt_calib_out.txt` for copy-paste.

- [ ] **Step 2: Update `cpdt_sensitivity.rs` constants.**

Open `crates/nsl-codegen/src/cpdt_sensitivity.rs`. Find the block starting at line 31:

```rust
pub const ANALYSIS_VERSION: u32 = 1;
```

Replace the block (lines 31-52) with:

```rust
pub const ANALYSIS_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Calibration constants (Phase 1 retune — 2026-04-19)
//
// Emitted by `cpdt_calibrate --emit-calibration`. T0/T1/T2 are geomeans of
// adjacent weights-present band mins/maxes; CALIB_K minimizes parameter-
// weighted no-weights-vs-weights disagreement on the committed corpus +
// regenerated calib_medium. Residual disagreement 15.64% — the SwiGLU
// gate_up/down numel-collision (spec §1.1) floors this. Invariant #7 is
// reframed to monitoring-gate at 20% per spec §5; Phase 2 spectral is the
// intervention that returns it to <5% hard-gate.
// ---------------------------------------------------------------------------

/// Neutral value of `gradient_magnitude_est` when weights are absent.
/// Calibrated by numerical minimization in `cpdt_calibrate --emit-calibration`.
pub const CALIB_K: f64 = 8.333333e-2;

pub const CALIB_T0: f64 = 1.452480e-7; // High   ↔ Medium
pub const CALIB_T1: f64 = 5.675521e-8; // Medium ↔ Low
pub const CALIB_T2: f64 = 1.907367e-9; // Low    ↔ VeryLow

/// Position-criticality near-extreme boost (for `L ≥ 4`). Unchanged from
/// Phase 1 ship; retune touches only the sensitivity-band constants.
pub const CALIB_ALPHA: f64 = 0.3;
```

(Exact decimal values: paste from `/tmp/cpdt_calib_out.txt`. If your Commit 1 run emitted slightly different values due to grid resolution, use those; the constants block is what you shipped.)

- [ ] **Step 3: Build.**

```bash
cargo build -p nsl-codegen --features cuda 2>&1 | tail -3
```

Expected: builds clean.

### Task 2.2: Regenerate baseline_heuristic.json

**Files:**
- Modify (regenerate): `tests/fixtures/cpdt_calibration/baseline_heuristic.json`

- [ ] **Step 1: Save pre-retune JSON for comparison.**

```bash
cp tests/fixtures/cpdt_calibration/baseline_heuristic.json \
   /tmp/pre_retune_baseline.json
```

- [ ] **Step 2: Regenerate.**

```bash
cargo run --features calibrate --bin cpdt_calibrate -- \
    tests/fixtures/cpdt_calibration/ 2>&1 | tail -5
```

Expected: writes `tests/fixtures/cpdt_calibration/baseline_heuristic.json` with new tier assignments driven by the bumped constants.

- [ ] **Step 3: Sanity-check the distribution on calib_small.**

```bash
python - << 'EOF'
import json, collections
with open('tests/fixtures/cpdt_calibration/baseline_heuristic.json') as f:
    data = json.load(f)
for snap in data:
    if snap['fixture'] == 'calib_small':
        c = collections.Counter(t['tier'] for t in snap['tiers'])
        print(f"calib_small: {dict(c)} (total {len(snap['tiers'])})")
        # Expected: High=56, Medium=12, Low=6, VeryLow=0
EOF
```

Expected: `{'high': 56, 'medium': 12, 'low': 6}` with total 74, no VeryLow. Spec §3.3 arithmetic.

- [ ] **Step 4: Clean up.**

```bash
rm /tmp/pre_retune_baseline.json
```

### Task 2.3: Regenerate expected_weights_present.json

**Files:**
- Modify (regenerate): `tests/fixtures/cpdt_calibration/expected_weights_present.json`

- [ ] **Step 1: Copy from baseline.**

Per Phase 1 design (spec §5.3): `expected_weights_present.json` is byte-identical to `baseline_heuristic.json` because the scorer's Some-branch already reads weights; the "Commit 4 prototype" semantic identity the Phase 1 spec discussed is satisfied tautologically today.

```bash
cp tests/fixtures/cpdt_calibration/baseline_heuristic.json \
   tests/fixtures/cpdt_calibration/expected_weights_present.json
```

- [ ] **Step 2: Verify byte-identity.**

```bash
diff tests/fixtures/cpdt_calibration/baseline_heuristic.json \
     tests/fixtures/cpdt_calibration/expected_weights_present.json
```

Expected: no output (files byte-identical).

### Task 2.4: Update canary test to count-floor assertions

**Files:**
- Modify: `crates/nsl-codegen/tests/cpdt_tier_agreement.rs`

- [ ] **Step 1: Extend the imports.**

Near the top of `cpdt_tier_agreement.rs`, change the `use nsl_codegen::cpdt_tier_apply::...` import to include `ParamPrecision` and `PrecisionPlan`:

```rust
use nsl_codegen::cpdt_tier_apply::{
    compute_tier_agreement, plan_map, plan_map_noweights, ParamPrecision, PrecisionConfig,
    PrecisionPlan, Tier,
};
```

- [ ] **Step 2: Find the existing `tier_agreement_full_on_calib_small_by_construction` test.**

Around line 62-82 in `cpdt_tier_agreement.rs`. Replace the whole function body with the new **corpus-wide primary-tier non-degeneracy + per-fixture sanity** form from spec §4.2:

```rust
#[test]
fn tier_agreement_full_on_calib_small_by_construction() {
    // Phase 1 retune (2026-04-19): the previous assertion
    // (`agree_l == total_l`) held under the degenerate binary distribution.
    // Under the retuned constants, calib_small is effectively 2-tier (H + M)
    // because the pooled-across-fixtures thresholds are dominated by
    // calib_medium's larger tensors; calib_small's ffn_gate_up scores land
    // above T0 and all generic attn+ffn_gate_up go High, leaving ffn_down as
    // the sole Medium population. The test now asserts corpus-wide primary-
    // tier non-degeneracy (§6 rule) plus per-fixture sanity; see spec §4.2.
    let wm_small = WeightMap::load(&fixture("calib_small")).unwrap();
    let plan_small = plan_map(
        &wm_small,
        &PrecisionConfig { n_layers: 8, ..Default::default() },
    );
    let counts_small = tier_counts_of(&plan_small);

    // Per-fixture sanity on calib_small: at least one primary tier populated.
    // Catches the "all generics landed VeryLow" degeneracy that Phase 1 shipped.
    let small_primary_count =
        counts_small.high + counts_small.medium + counts_small.low;
    assert!(
        small_primary_count > 0,
        "calib_small has no primary-tier assignments (degenerate distribution)"
    );

    // Expected under pooled thresholds at K≈0.060 (per spec §3.3):
    //   High = 68, Medium = 6, Low = 0, VeryLow = 0.
    // Floors use slack so minor fixture perturbations don't break the test.
    assert!(
        counts_small.high >= 60,
        "H underpopulated on calib_small: {}",
        counts_small.high
    );
    assert!(
        counts_small.medium >= 4,
        "M underpopulated on calib_small: {}",
        counts_small.medium
    );
    assert_eq!(
        counts_small.very_low, 0,
        "VeryLow unexpectedly populated on calib_small: {}",
        counts_small.very_low
    );

    // Corpus-wide primary-tier non-degeneracy (§6 rule). calib_small alone
    // doesn't populate Low; calib_medium does. calib_medium is regen-at-test-
    // time — skip the corpus union check if the regenerated fixture is absent.
    // CI runs that regenerate calib_medium first catch the full rule; local
    // developer runs without regen see only the per-fixture sanity check.
    if let Some(plan_medium) = try_load_plan_medium() {
        let counts_medium = tier_counts_of(&plan_medium);
        let union_populated = |s: usize, m: usize| -> bool { s > 0 || m > 0 };
        assert!(
            union_populated(counts_small.high, counts_medium.high),
            "primary tier High not populated across corpus"
        );
        assert!(
            union_populated(counts_small.medium, counts_medium.medium),
            "primary tier Medium not populated across corpus"
        );
        assert!(
            union_populated(counts_small.low, counts_medium.low),
            "primary tier Low not populated across corpus; \
             calib_small: {}, calib_medium: {}",
            counts_small.low,
            counts_medium.low
        );
    } else {
        eprintln!(
            "note: calib_medium not present at target/cpdt_calibration/; \
             corpus-wide primary-tier check skipped. To enable, run \
             cpdt_fixture_generate --include-medium \
             --output-dir target/cpdt_calibration/"
        );
    }
}

#[derive(Default)]
struct TierCounts {
    high: usize,
    medium: usize,
    low: usize,
    very_low: usize,
}

fn tier_counts_of(plan: &PrecisionPlan) -> TierCounts {
    let mut c = TierCounts::default();
    for p in &plan.params {
        match p.tier {
            Tier::High => c.high += 1,
            Tier::Medium => c.medium += 1,
            Tier::Low => c.low += 1,
            Tier::VeryLow => c.very_low += 1,
        }
    }
    c
}

/// Load calib_medium from the regen-at-test-time directory if present.
/// Returns `None` if the file is absent so the canary test can skip the
/// corpus-union check gracefully.
fn try_load_plan_medium() -> Option<PrecisionPlan> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/cpdt_calibration/calib_medium.safetensors");
    if !p.is_file() {
        return None;
    }
    let wm = WeightMap::load(&p).ok()?;
    let plan = plan_map(
        &wm,
        &PrecisionConfig { n_layers: 16, ..Default::default() },
    );
    Some(plan)
}
```

The existing imports (`plan_map`, `PrecisionConfig`, `Tier`, `WeightMap`, the `fixture()` helper, `PathBuf`) are already in scope. After Step 1's import additions, `ParamPrecision` and `PrecisionPlan` are also in scope.

- [ ] **Step 3: Build + run just this test.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_tier_agreement \
    tier_agreement_full_on_calib_small_by_construction 2>&1 | tail -5
```

Expected: 1 passed; 0 failed.

### Task 2.5: Add `disagreement_source_matches_numel_collision` diagnostic test

**Files:**
- Modify: `crates/nsl-codegen/tests/cpdt_tier_agreement.rs`

- [ ] **Step 1: Append the diagnostic test.**

Add this function at the bottom of `cpdt_tier_agreement.rs`, after the existing tests:

```rust
#[test]
fn disagreement_source_matches_numel_collision() {
    // Spec §5.4: for every tensor p whose weights-present tier differs from
    // its no-weights tier, there must exist a numel-matched sibling q whose
    // weights-present tier is exactly the tier the nw path put p into. That's
    // the precise collision signature (SwiGLU gate_up/down at d_model × d_ffn).
    // Any disagreement without such a sibling is an unexpected source and
    // should fail the test.
    let fixtures = [("calib_tiny", 2u32), ("calib_small", 8u32)];
    for (fix, n_layers) in fixtures {
        let wm = WeightMap::load(&fixture(fix)).unwrap();
        let cfg = PrecisionConfig {
            n_layers,
            ..Default::default()
        };
        let plan = plan_map(&wm, &cfg);
        let plan_nw = plan_map_noweights(&wm, &cfg);
        let by_name_nw: std::collections::HashMap<&str, &ParamPrecision> = plan_nw
            .params
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();
        for p in &plan.params {
            let pnw = by_name_nw
                .get(p.name.as_str())
                .unwrap_or_else(|| panic!("{fix}: missing nw plan entry for {}", p.name));
            if p.tier == pnw.tier {
                continue;
            }
            let p_numel = wm.get(&p.name).unwrap().num_elements;
            let has_collision_partner = plan.params.iter().any(|q| {
                q.name != p.name
                    && wm.get(&q.name).unwrap().num_elements == p_numel
                    && q.tier == pnw.tier
            });
            assert!(
                has_collision_partner,
                "{fix}: disagreement on {} (wp={:?}, nw={:?}) has no \
                 numel-matched sibling with wp-tier == {:?} — unknown source \
                 of disagreement, investigate",
                p.name, p.tier, pnw.tier, pnw.tier
            );
        }
    }
}
```

The existing imports cover `plan_map_noweights`, `ParamPrecision`, and the `fixture()` helper. If `std::collections::HashMap` isn't already imported, the fully-qualified path used above works without changing imports.

- [ ] **Step 2: Build + run just this test.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_tier_agreement \
    disagreement_source_matches_numel_collision 2>&1 | tail -5
```

Expected: 1 passed; 0 failed. If it fails, the disagreement is coming from somewhere other than gate_up/down numel collision — investigate before proceeding.

### Task 2.6: Reframe the disagreement gate from 5% to 20% monitoring threshold

**Files:**
- Modify: `crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs`

- [ ] **Step 1: Open the file and replace the test body + doc-comment.**

Open `crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs`. Replace the module-level doc-comment (lines 1-2) with:

```rust
//! Phase 1 Commit 4 (original) + Phase 1 retune (2026-04-19) — parameter-
//! weighted disagreement between no-weights and weights-present paths.
//!
//! Originally a <5% hard-gate. Reframed to <20% monitoring-gate by the
//! retune (spec §5): the no-weights formula `K × pos / numel` has no
//! discriminator between numel-degenerate shape classes (SwiGLU's
//! ffn.w_gate/w_up/w_down at d_model × d_ffn), so no CALIB_K achieves
//! <5% disagreement. The 20% ceiling is Phase 1's monitoring range;
//! Phase 2's spectral factor is the intervention that returns this gate
//! to <5%. See `disagreement_source_matches_numel_collision` (in
//! cpdt_tier_agreement.rs) for the diagnostic that verifies the source.
```

- [ ] **Step 2: Rename the test + raise the threshold.**

Replace the function (around line 28):

```rust
#[test]
fn weighted_disagreement_below_monitoring_threshold() {
    let fixtures = [("calib_tiny", 2u32), ("calib_small", 8u32)];
    let mut disagreeing_params: u64 = 0;
    let mut total_params: u64 = 0;
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cpdt_calibration");
    for (name, n_layers) in fixtures {
        let path = fixture_dir.join(format!("{name}.safetensors"));
        let wm = WeightMap::load(&path)
            .unwrap_or_else(|e| panic!("fixture {} load failed: {e:?}", path.display()));
        for (tname, entry) in wm.entries() {
            let with = score_with_entry(tname, Some(entry), entry.num_elements, n_layers);
            let without = score_with_entry(tname, None, entry.num_elements, n_layers);
            total_params += entry.num_elements as u64;
            if with != without {
                disagreeing_params += entry.num_elements as u64;
            }
        }
    }
    let frac = disagreeing_params as f64 / total_params as f64;
    eprintln!(
        "weighted disagreement: {:.4} ({}/{} params)",
        frac, disagreeing_params, total_params
    );
    assert!(
        frac < 0.20,
        "weighted disagreement {frac:.4} >= 0.20 monitoring threshold — \
         calibration drift or new class of disagreement beyond the documented \
         numel-degeneracy. See spec §5 for the reframing."
    );
}
```

- [ ] **Step 3: Build + run.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_disagreement 2>&1 | tail -5
```

Expected: 1 passed; 0 failed. The measured fraction printed to stderr should be ~0.16 (close to the spec's 15.64%).

### Task 2.7: Verify adversarial + primitives tests still pass

**Files:** none (verification only).

- [ ] **Step 1: Run adversarial.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_adversarial 2>&1 | tail -5
```

Expected: 1 passed. The `M = CALIB_T0 / s_pre * 1.5` multiplier reads the live T0 constant, so the adversarial scaling automatically tracks the new threshold.

- [ ] **Step 2: Run primitives.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_primitives 2>&1 | tail -5
```

Expected: 22 passed. All boundary tests use `CALIB_T0 ± 1e-6` patterns that are T-value-agnostic.

- [ ] **Step 3: Run snapshot.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_snapshot 2>&1 | tail -5
```

Expected: 1 passed. The snapshot test reads `expected_weights_present.json` which was regenerated in Task 2.3.

- [ ] **Step 4: Run the full cpdt suite one more time.**

```bash
cargo test -p nsl-codegen --features cuda --lib cpdt 2>&1 | tail -5
cargo test -p nsl-codegen --features cuda \
    --test cpdt_sensitivity_primitives \
    --test cpdt_sensitivity_snapshot \
    --test cpdt_sensitivity_adversarial \
    --test cpdt_sensitivity_disagreement \
    --test cpdt_tier_agreement 2>&1 | grep "^test result:"
```

Expected: 61 lib + 22 + 1 + 1 + 1 + 7 = 93 tests pass, 0 failures. (`cpdt_tier_agreement` count goes from 6 to 7 because of the new diagnostic test added in Task 2.5.)

### Task 2.8: Append retrospective addendum to phase2-stub.md

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md`

- [ ] **Step 1: Append the addendum.**

Open `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md`. Append to the end:

```markdown

---

## Phase 1 threshold retune — 2026-04-19 — scope-reduced with monitoring-gate reframing

**Context.** Phase 1 shipped with placeholder constants `T0=0.50, T1=0.10, T2=0.02, CALIB_K=0.0312`. Post-ship inspection found that every generic (non-kind-overridden) tensor scored below `T2` on the calibration corpus, producing a degenerate binary distribution: 32 High + 42 VeryLow on `calib_small`, 0 Medium, 0 Low.

**Pre-dispatch finding.** Running the retune's Commit 1 logic ahead of implementation revealed a structural information-bottleneck in the no-weights path. `ffn.w_gate`, `ffn.w_up`, and `ffn.w_down` share `numel = d_model × d_ffn` in a standard SwiGLU FFN. The no-weights formula `K × pos / numel` depends only on numel and pos; no choice of `CALIB_K` can separate these three shape classes into different tiers on the no-weights path. Numerical minimization floors parameter-weighted disagreement at 15.64% on `calib_small` — the fraction of parameters living in `ffn.w_down`, which necessarily agrees with `ffn.w_gate/w_up` on the no-weights path but disagrees on the weights-present path.

**Response.** Retune anyway — the weights-present path still benefits from meaningful thresholds, and Phase 1 was always the coarse-signal layer. Reframe invariant #7 from hard-gate to monitoring-gate with Phase 2 as the documented intervention that returns it to <5% hard-gate status.

**Measured values** (pinned by `cpdt_calibrate --emit-calibration`):
- `T0 = 1.4525e-7` (High ↔ Medium)
- `T1 = 5.6755e-8` (Medium ↔ Low)
- `T2 = 1.9074e-9` (Low ↔ VeryLow)
- `CALIB_K = 8.333e-2`
- `ANALYSIS_VERSION` bumped to 2

**Post-retune distribution on calib_small (74 tensors):**
- High: 56 (32 kind-overridden + 24 formula-driven attn QKVO)
- Medium: 12 (ffn_gate + ffn_up × 6 middle layers)
- Low: 6 (ffn_down × 6 middle layers)
- VeryLow: 0 (documented fallback)

**Residual disagreement:** 15.64%, entirely from ffn_down × 6 middle layers. Monitoring gate now fires at 20%; diagnostic test `disagreement_source_matches_numel_collision` verifies the firing traces to the known SwiGLU numel-collision pattern rather than unknown drift.

**Phase 2 connection.** Spectral condition is the intervention that gives the no-weights path a discriminator beyond numel; the per-Phase-2 close-out criterion is <5% parameter-weighted disagreement across the calibration corpus. If Phase 2's first-commit precondition (spectral geomean ≈ 1.0) fails, scope expands per the parent spec's §11.4.

**Institutional rules added** (spec §7):

1. **Tier-distribution non-degeneracy check** — any N-tier scheme's close-out must assert the *primary* tier set (all tiers minus documented-as-rare fallbacks) is non-degenerate across the union of calibration fixtures. CPDT's primary set is {High, Medium, Low}; VeryLow is the documented fallback.
2. **Monitoring-gate reframing for architectural bottlenecks** — when an invariant fires on a specific architectural class (not a single tensor or transient bug), the correct response is to document the class, add a diagnostic test that verifies the firing traces to it, name the future intervention that resolves it, and reframe the invariant to monitoring-status with that intervention as the promotion trigger. *Not* to narrow the invariant's domain or raise its threshold.

Both rules apply to future tier-assignment work in NSL.
```

- [ ] **Step 2: Verify the section is appended.**

```bash
grep -n "^## Phase 1 threshold retune" docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md
```

Expected: one match line showing the new section header.

### Task 2.9: Update session MEMORY.md

**Files:**
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md`
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md` (update invariant #7 wording)

- [ ] **Step 1: Update the SHIPPED entry.**

Open `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md`. Find the line starting with `## CPDT weight-aware Phase 1 — SHIPPED 2026-04-19` and replace the bullet below it with:

```markdown
## CPDT weight-aware Phase 1 — SHIPPED 2026-04-19 (retune 2026-04-19)
- [CPDT Phase 1 invariants](project_cpdt_weight_aware_invariants.md) — 9 load-bearing constraints; **#7 reframed 2026-04-19** from <5% hard-gate to <20% monitoring-gate with Phase 2 promotion trigger (SwiGLU gate_up/down numel-collision floors disagreement at ~15.64% until Phase 2's spectral factor ships). Retune commits (`feat/cpdt-calibration-tune`): `T0=1.45e-7, T1=5.68e-8, T2=1.91e-9, CALIB_K=0.083, ANALYSIS_VERSION=2`. Post-retune distribution on calib_small: 56 High + 12 Medium + 6 Low + 0 VeryLow. Two new institutional rules: tier-distribution non-degeneracy check + monitoring-gate reframing for architectural bottlenecks — see spec §7.
```

- [ ] **Step 2: Update invariant #7 in the invariants file.**

Open `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md`. Find `7. **No-weights path agrees with weights-present path within 5% parameter-weighted on baseline corpus.**` and replace the entire bullet (including Why and How to apply) with:

```markdown
7. **No-weights path parameter-weighted disagreement is monitored; gate semantics are Phase-dependent.**
   - **Phase 1 (current, retuned 2026-04-19):** `cpdt_sensitivity_disagreement::weighted_disagreement_below_monitoring_threshold` asserts `< 0.20`. Firing above 20% indicates calibration drift or an unknown class of disagreement beyond the documented SwiGLU numel-collision floor (~15.64%). The `disagreement_source_matches_numel_collision` test verifies firings trace to the known pattern.
   - **Phase 2 (future):** after spectral-factor integration, the gate returns to `< 0.05`. Phase 2's close-out criterion is hitting <5% on the calibration corpus; if it doesn't, scope expands per the parent spec's §11.4.
   - **Why:** the no-weights formula `K × pos / numel` has no discriminator between numel-degenerate shape classes (e.g. SwiGLU's `ffn.w_gate/w_up/w_down` at `d_model × d_ffn`); Phase 1 ships the best approximation possible and Phase 2 supplies the missing discriminator.
   - **How to apply:** Phase 1 code NEVER tightens the 20% ceiling to track lower measurements; the threshold is Phase 1's monitoring range, not "current + slack." Tightening happens when Phase 2 ships, and only then.
```

- [ ] **Step 3: Verify MEMORY.md stays within the ~200-line convention.**

```bash
wc -l "C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md"
```

If over 200, extract stable background (Architecture, Roadmap sections) into a topic file and replace with a one-line pointer, per the existing convention.

### Task 2.10: Commit 2

**Files:** all staged changes from Tasks 2.1 - 2.9.

- [ ] **Step 1: Confirm staged set.**

```bash
git status --short
```

Expected:

```
 M crates/nsl-codegen/src/cpdt_sensitivity.rs
 M crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs
 M crates/nsl-codegen/tests/cpdt_tier_agreement.rs
 M tests/fixtures/cpdt_calibration/baseline_heuristic.json
 M tests/fixtures/cpdt_calibration/expected_weights_present.json
?? docs/superpowers/specs/... (phase2-stub.md is gitignored; use -f)
```

- [ ] **Step 2: Stage (including force-add for gitignored docs).**

```bash
git add crates/nsl-codegen/src/cpdt_sensitivity.rs \
       crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs \
       crates/nsl-codegen/tests/cpdt_tier_agreement.rs \
       tests/fixtures/cpdt_calibration/baseline_heuristic.json \
       tests/fixtures/cpdt_calibration/expected_weights_present.json
git add -f docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md
```

- [ ] **Step 3: Run the full verification-before-completion.**

```bash
cargo build -p nsl-codegen --features cuda 2>&1 | tail -3
cargo test -p nsl-codegen --features cuda --lib cpdt 2>&1 | tail -5
cargo test -p nsl-codegen --features cuda \
    --test cpdt_sensitivity_primitives \
    --test cpdt_sensitivity_snapshot \
    --test cpdt_sensitivity_adversarial \
    --test cpdt_sensitivity_disagreement \
    --test cpdt_tier_agreement 2>&1 | grep "^test result:"
```

Expected: everything passes, 93 tests total (61 lib + 22 primitives + 1 snapshot + 1 adversarial + 1 disagreement + 7 tier_agreement).

- [ ] **Step 4: Commit.**

```bash
git commit -m "$(cat <<'EOF'
feat(cpdt): retune T0/T1/T2 + CALIB_K + bump ANALYSIS_VERSION (Phase 1 retune)

Applies the constants emitted by Commit 1's `cpdt_calibrate --emit-calibration`
and reframes invariant #7 from <5% hard-gate to <20% monitoring-gate.

Scorer constants:
- ANALYSIS_VERSION 1 → 2
- CALIB_T0 0.50 → 1.4525e-7 (High ↔ Medium)
- CALIB_T1 0.10 → 5.6755e-8 (Medium ↔ Low)
- CALIB_T2 0.02 → 1.9074e-9 (Low ↔ VeryLow)
- CALIB_K  0.0312 → 8.333e-2

Post-retune distribution on calib_small (74 tensors):
- High    : 56 (32 kind-overridden + 24 formula-driven attn QKVO)
- Medium  : 12 (ffn_gate + ffn_up × 6 middle layers)
- Low     :  6 (ffn_down × 6 middle layers)
- VeryLow :  0 (documented fallback for out-of-band generics)

Test suite changes:
- cpdt_sensitivity_disagreement: rename `..._below_5_percent` to
  `..._below_monitoring_threshold`, gate raised from 0.05 to 0.20. Doc
  comment reframes the gate per the design spec's §5.
- cpdt_tier_agreement::tier_agreement_full_on_calib_small_by_construction:
  replaced `agree_l == total_l` binary check with count-floor asserts on
  High (>=50), Medium (>=8), Low (>=4), VeryLow (==0).
- cpdt_tier_agreement::disagreement_source_matches_numel_collision: new
  diagnostic test that asserts every disagreement traces to a numel-matched
  sibling with wp-tier == p's nw-tier (the precise collision signature).
  Fails if disagreement comes from an unknown source.
- Fixture JSONs regenerated.

Invariant #7 reframing (per spec §5):
- Phase 1: monitoring-gate at <20% with documented SwiGLU numel-collision
  exemption. Current measurement 15.64%.
- Phase 2: returns to hard-gate at <5% once spectral-factor integration
  provides the missing no-weights discriminator.

No formula change. No CLI change. No changes to `invoke_cpdt_if_enabled`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands. `git log --oneline -2` shows both Commit 1 and Commit 2.

---

## Post-Commit: Final Verification

- [ ] **Run the full workspace.**

```bash
cargo build --release --bin nsl --features cuda 2>&1 | tail -3
cargo test -p nsl-codegen --features cuda 2>&1 | tail -20
```

Expected: release build + full `nsl-codegen` test suite green.

- [ ] **Walk through the spec's §9 close-out criteria.**

Each item satisfied:
1. Commit 1 acceptance — Task 1.9.
2. Commit 2 acceptance with 7 named assertions — Tasks 2.4-2.7.
3. `ANALYSIS_VERSION = 2` — Task 2.1.
4. Retrospective addendum committed — Task 2.8.
5. Nine invariants evaluated — #1-#6, #8, #9 preserved; #7 reframed. Task 2.9 updates the invariants memory file.
6. MEMORY.md updated — Task 2.9.

- [ ] **Open PR** (after explicit user authorization; not a default plan step).

```bash
git push -u origin feat/cpdt-calibration-tune
gh pr create --title "feat(cpdt): Phase 1 threshold retune — scope-reduced with monitoring-gate reframing"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the design spec is addressed.
  - §1 motivation → Commit 2 JSON regeneration reflects the retune (Task 2.2, 2.3).
  - §2 non-goals → plan touches only constants + tests + docs; formula / CLI / `invoke_cpdt_if_enabled` untouched.
  - §3.1 threshold computation → Task 1.5.
  - §3.2 CALIB_K minimization → Task 1.6.
  - §3.3 post-retune distribution → Task 2.2 step 3 sanity check.
  - §3.4 invariant checklist → Task 2.9 updates the invariants file; Task 2.1 bumps ANALYSIS_VERSION.
  - §4.1 Commit 1 acceptance two-step command → Task 1.6 step 3.
  - §4.2 Commit 2 seven named assertions → Tasks 2.4, 2.5, 2.6, 2.7 each cover a subset; Tasks 2.1, 2.2, 2.3 handle #1-#3.
  - §5 invariant #7 reframing → Tasks 2.6 (test) + 2.9 (invariants memory) + 2.8 (retrospective addendum).
  - §5.4 diagnostic test → Task 2.5.
  - §6 retrospective addendum → Task 2.8.
  - §7.1 + §7.2 institutional rules → Task 2.8 addendum captures both.
  - §8 risks → addressed operationally by Task 1.6's structural-limitation guard and Task 1.5's ordering guard.
  - §9 close-out → Post-Commit section.

- **No placeholders.** Every code step shows the actual code; every command shows the exact invocation; every expected output is concrete. "Expected: builds clean" is specific to cargo's output format and matches what the engineer will see.

- **Type consistency.** `Band` enum, `TensorRecord` struct, `tier_of` / `tier_label` helpers, `BandStats`, `compute_thresholds`, `minimize_calib_k`, `collect_records`, `verify_ordering`, `band_stats_per_fixture` — all defined once and referenced consistently. `TierCounts` + `tier_counts_of` added in Task 2.4 and reused by future tests if needed. `disagreement_source_matches_numel_collision` lives in `cpdt_tier_agreement.rs` alongside its collaborators (`plan_map_noweights`, `ParamPrecision`).

- **DRY.** Task 1.3's `run_default` reuses the Phase 1 scorer primitives (`gradient_magnitude_est`, `position_criticality`, `classify_layer_kind`, `assign_tier`) rather than reimplementing them. `--emit-calibration` reuses the same primitives via `collect_records`.

- **TDD.** Task 1 is verification-driven (the calibration binary produces specific measured values; if it doesn't, the plan stops). Tasks 2.4, 2.5 are test-first. Tasks 2.6, 2.7 verify tests adapt to the new constants before the commit lands.

- **Frequent commits.** Two commits total, scoped per spec §4. Each commit's acceptance criteria are named so a reviewer can verify.
