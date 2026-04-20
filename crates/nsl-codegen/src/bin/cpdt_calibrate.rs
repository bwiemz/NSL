//! Dev-only CPDT calibration binary. Gated behind `[features] calibrate = []`
//! so it doesn't land in release builds.
//!
//! # Default mode — regenerate baseline_heuristic.json
//!
//! ```bash
//! cargo run --features calibrate --bin cpdt_calibrate -- <fixture_dir>
//! ```
//!
//! Writes `<fixture_dir>/baseline_heuristic.json` and prints the current
//! in-code constants to stdout. Byte-identical to pre-retune behavior.
//!
//! # `--emit-calibration` mode — compute retune values (Phase 1 retune)
//!
//! Two-step procedure:
//!
//! ```bash
//! # Step 1: regenerate calib_medium into target/.
//! cargo run --features calibrate --bin cpdt_fixture_generate -- \
//!     --include-medium --output-dir target/cpdt_calibration/
//!
//! # Step 2: compute thresholds + CALIB_K by plateau-midpoint selection.
//! cargo run --features calibrate --bin cpdt_calibrate -- \
//!     tests/fixtures/cpdt_calibration/ \
//!     --medium-dir target/cpdt_calibration/ \
//!     --emit-calibration
//! ```
//!
//! Prints the diff-ready constants block (T0/T1/T2/CALIB_K) plus plateau
//! boundaries, per-fixture disagreement, corpus-wide disagreement, and a
//! per-band diagnostic (middle-layer median tensors' wp vs nw tier
//! assignments at the emitted K). Exits non-zero if the cross-fixture
//! ordering guard fires or the plateau is empty (no K satisfies per-
//! fixture disagreement <= 20% for every fixture).

use std::path::Path;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    ANALYSIS_VERSION, CALIB_ALPHA, CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2,
};
use nsl_codegen::cpdt_tier_apply::PrecisionConfig;
use nsl_codegen::weight_aware::WeightMap;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    // Only classify weight tensors, not bias or other parameters.
    // Bias tensors are zero-initialized and would collapse the geomean to ~0.
    if !name.ends_with(".weight") {
        return None;
    }
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

#[derive(Serialize)]
struct TierEntry {
    name: String,
    tier: &'static str,
    score: f64,
}

#[derive(Serialize)]
struct FixtureSnapshot {
    fixture: String,
    tiers: Vec<TierEntry>,
}

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
                if other.starts_with("--") {
                    eprintln!("error: unknown flag or missing space after flag name: {other}");
                    std::process::exit(2);
                }
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

/// `--emit-calibration` mode: loads all three fixtures, scores tensors, guards ordering,
/// and emits computed T0/T1/T2 thresholds to stderr.
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

    // Subsequent tasks append: ordering guard (1.5), threshold computation (1.5),
    // K minimization + diagnostic emission (1.6).
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

    // Emit the diff-ready Rust constants block. `--emit-calibration` is a
    // retune operation, which always changes tier boundaries (T0/T1/T2) —
    // invariant #1 requires bumping ANALYSIS_VERSION in the same commit.
    // Emit the current-version-plus-one so the copy-paste produces a
    // bump-compliant patch regardless of what version the source is on.
    println!();
    println!(
        "pub const ANALYSIS_VERSION: u32 = {}; // bumped from {} per invariant #1",
        ANALYSIS_VERSION + 1,
        ANALYSIS_VERSION,
    );
    println!("pub const CALIB_K:     f64 = {k:.6e};");
    println!("pub const CALIB_T0:    f64 = {t0:.6e};");
    println!("pub const CALIB_T1:    f64 = {t1:.6e};");
    println!("pub const CALIB_T2:    f64 = {t2:.6e};");
    println!("pub const CALIB_ALPHA: f64 = {};", CALIB_ALPHA);
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
    // Log-spaced grid: 10^-6 to 10^0 in 0.01-decade steps → 601 endpoints
    // spanning 600 intervals, ~2.33% multiplicative step per interval.
    // Spec §3.2's pseudocode says "600 points"; 601 is the endpoint count
    // that produces 600 intervals — see spec's measured-result paragraph.
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
            // Strict >, so ties are broken in favor of the first (lowest-K)
            // run encountered — the more conservative feasible K when
            // evidence is ambiguous.
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

fn tier_label(score: f64, t0: f64, t1: f64, t2: f64) -> &'static str {
    if score > t0 { "high" }
    else if score > t1 { "medium" }
    else if score > t2 { "low" }
    else { "very_low" }
}

fn infer_config_from_fixture(fixture: &str) -> PrecisionConfig {
    let n_layers = match fixture {
        "calib_tiny" => 2,
        "calib_small" => 8,
        "calib_medium" => 16,
        _ => 8,
    };
    PrecisionConfig {
        n_layers,
        ..Default::default()
    }
}
