//! CPKD Innovation 5: CEP-guided student architecture design.
//!
//! Given a teacher architecture (a CEP [`ModelSpec`]), a parameter budget,
//! and a hardware target, search the `@search`-decorated axes for the best
//! student architecture and produce a teacher->student layer mapping for
//! feature matching during distillation.
//!
//! Reuses the CEP machinery wholesale:
//!   * candidate enumeration + roofline ranking — [`crate::cep_search::architecture_search`]
//!   * per-layer teacher importance — [`crate::cep_importance::analyse_weight_map`]
//!
//! The layer mapping is uniform in depth: the teacher's layers are cut into
//! as many contiguous segments as the student has layers, each student
//! layer learns from one teacher layer of its own segment (the most
//! important one when teacher weights are given, the deepest otherwise),
//! and the last student layer always learns from the last teacher layer.
//! Ranking teacher layers by importance alone and taking the top k could
//! hand a 12-layer student teacher layers 20 to 31, and with no weights at
//! all (every score tied) it mapped the student onto the teacher's FIRST k
//! layers.
//!
//! Two v1 deviations from stock CEP, both deliberate:
//!   * **Parameter budget is enforced caller-side.** `Constraints`
//!     (cep_search.rs) has no budget field — feasibility only checks peak
//!     memory + latency — so we filter `ranked_candidates` by
//!     [`ModelSpec::param_count`] after the search, preserving the
//!     feasible-first, score-descending ranking.
//!   * **The position-factor prior is normalized out** before ranking
//!     teacher layers. `analyse_weight_map` multiplies every head/FFN score
//!     by the U-shaped `position_factor` (first/last layers +30%) because
//!     those layers resist *pruning*; for teacher->student feature matching
//!     we want the informativeness signal without that positional prior, so
//!     each layer's `total_score` is divided by its position factor.
//!
//! ADVISORY in v1: the design is reported (plan/CLI), not lowered — no
//! generated student source or sliced weights come out of this module.

use serde::Serialize;

use crate::cep_importance::{position_factor, ImportanceTable, RooflineSlackTable};
use crate::cep_oracle::{CompilationProfile, ModelSpec};
use crate::cep_rewrite::SearchAxes;
use crate::cep_search::{architecture_search, Constraints, NasObjective};
use crate::weight_aware::WeightMap;

/// Input to [`design_student`].
#[derive(Debug, Clone)]
pub struct StudentDesignInput<'a> {
    /// Search axes (from `@search` decorators via `cep_extract::extract_search_axes`).
    pub axes: SearchAxes,
    /// The teacher architecture the student will distill from.
    pub teacher_spec: ModelSpec,
    /// Teacher weights for layer-importance scoring. `None` degenerates to
    /// uniform scores (see [`design_student`] docs).
    pub weights: Option<&'a WeightMap>,
    /// GPU model name resolved against the `gpu_specs` database.
    pub target_gpu: &'a str,
    /// Maximum student parameter count (inclusive).
    pub target_params: u64,
    /// NAS ranking objective.
    pub objective: NasObjective,
}

/// One teacher->student feature-matching pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerMapEntry {
    pub teacher_layer: u32,
    pub student_layer: u32,
    /// Position-normalized teacher-layer importance (`total_score /
    /// position_factor`), the score the layer was preferred by within its
    /// depth segment. `None` when the layer has no score: no teacher weights
    /// were given, or the importance pass matched none of this layer's
    /// weight names. A missing score is not an importance of zero.
    pub teacher_importance: Option<f64>,
}

/// Output of [`design_student`].
#[derive(Debug, Clone, Serialize)]
pub struct StudentDesign {
    /// The chosen student architecture.
    pub spec: ModelSpec,
    /// Oracle profile of the chosen student on the target GPU.
    pub profile: CompilationProfile,
    /// Candidates enumerated by the architecture search (post shape-algebra
    /// filtering, pre budget filtering).
    pub candidates_considered: u32,
    /// Candidates whose `param_count()` fit within the budget.
    pub candidates_within_budget: u32,
    /// Teacher->student layer pairing for feature matching, ascending by
    /// teacher layer. Covers `min(student_layers, teacher_layers)` entries.
    pub layer_mapping: Vec<LayerMapEntry>,
    /// Whether teacher weights were given, i.e. whether importance scores
    /// took part in the mapping at all.
    pub teacher_scored: bool,
    /// Teacher layers the importance pass produced no score for although
    /// weights were given (none of their weight names matched). Empty when
    /// `teacher_scored` is false.
    pub unscored_teacher_layers: Vec<u32>,
}

/// Per-layer teacher importance with the U-shaped position prior divided out.
///
/// `analyse_weight_map` bakes `position_factor(l, L)` into every head and FFN
/// score (it models *pruning* sensitivity); dividing `total_score` by the
/// factor recovers a positionally unbiased informativeness signal for
/// teacher-layer selection. A layer missing from the table has no score
/// (`None`): the importance pass matched none of its weight names. It used
/// to be reported as an importance of 0.0, which reads as a measurement.
pub fn normalized_layer_scores(table: &ImportanceTable, n_layers: u32) -> Vec<Option<f64>> {
    (0..n_layers)
        .map(|l| {
            table
                .layers
                .iter()
                .find(|li| li.layer == l)
                .map(|li| li.total_score / position_factor(l, n_layers))
        })
        .collect()
}

/// Pair each student layer with one teacher layer, spread over the teacher's
/// depth.
///
/// With `T` teacher layers and `k = min(student_n_layers, T)` mapped student
/// layers, student layer `i` owns the teacher segment
/// `[i*T/k, (i+1)*T/k)`. It is paired with the highest-scoring layer of that
/// segment; a layer without a score ranks below every scored one, and a tie
/// goes to the DEEPER layer, so with no scores at all this is the usual
/// uniform map (student `i` from teacher `(i+1)*T/k - 1`). The last student
/// layer is always paired with the last teacher layer: the student's output
/// is fitted to the teacher's output, so their final representations are the
/// ones that must correspond.
///
/// The pairs come out ascending in both layers, one per segment, so they
/// can neither cluster nor skip a stretch of the teacher's depth. If the
/// student is deeper than the teacher, only `T` entries are produced (extra
/// student layers are left unmapped — v1 has no fan-out mapping).
pub fn select_layer_mapping(scores: &[Option<f64>], student_n_layers: u32) -> Vec<LayerMapEntry> {
    let teacher_n = scores.len();
    let k = (student_n_layers as usize).min(teacher_n);
    (0..k)
        .map(|student| {
            let (start, end) = (student * teacher_n / k, (student + 1) * teacher_n / k);
            let teacher = if student + 1 == k {
                teacher_n - 1
            } else {
                // `max_by` keeps the LAST maximum, which is the deeper layer.
                (start..end)
                    .max_by(|&a, &b| {
                        let key = |l: usize| scores[l].unwrap_or(f64::NEG_INFINITY);
                        key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(start)
            };
            LayerMapEntry {
                teacher_layer: teacher as u32,
                student_layer: student as u32,
                teacher_importance: scores[teacher],
            }
        })
        .collect()
}

/// Search the axes for the best student under the parameter budget and build
/// the teacher->student layer mapping.
///
/// Teacher importance: with `weights` present, the CEP importance pass runs
/// with a NEUTRAL roofline slack (empty [`RooflineSlackTable`] — every layer
/// defaults to 1.0). The slack refinement CEP's prune driver applies
/// (`slack_from_profile`, cep.rs) is private and profile-coupled; v1
/// deliberately uses neutral slack so the layer score is a pure
/// weight-magnitude x spectral x position signal (position then normalized
/// out). Without `weights` no layer has a score and the mapping is the
/// uniform-depth one (see [`select_layer_mapping`]); with them, importance
/// chooses within each depth segment.
pub fn design_student(input: StudentDesignInput) -> Result<StudentDesign, String> {
    let Some(gpu) = crate::gpu_specs::find_gpu(input.target_gpu) else {
        return Err(format!(
            "CPKD: unknown student-design target GPU '{}' (supported: {})",
            input.target_gpu,
            crate::cep::supported_gpus_list()
        ));
    };

    // Peak-memory/latency unconstrained in v1; the budget is enforced below.
    let constraints = Constraints::default();
    let outcome = architecture_search(&input.axes, gpu, &constraints, input.objective);
    let considered = outcome.candidates_enumerated;

    if outcome.ranked_candidates.is_empty() {
        return Err(
            "CPKD: the @search axes enumerate no valid candidates \
             (shape-algebra filters: n_kv_heads | n_heads, n_heads | d_model, \
             d_model pow2-or-mult-64, d_ff mult-64); widen or fix the axes"
                .to_string(),
        );
    }

    // Caller-side budget filter. `retain`-style stable filtering preserves
    // the feasible-first, score-descending ranking from the search.
    let within: Vec<&crate::cep_search::EvaluatedCandidate> = outcome
        .ranked_candidates
        .iter()
        .filter(|c| c.spec.param_count() <= input.target_params)
        .collect();
    let within_budget = within.len() as u32;

    let Some(chosen) = within.first() else {
        let smallest = outcome
            .ranked_candidates
            .iter()
            .map(|c| c.spec.param_count())
            .min()
            .unwrap_or(0);
        return Err(format!(
            "CPKD: no student candidate fits the parameter budget of {} params: \
             the smallest of {} enumerated candidates has {} params; \
             widen the @search axes (smaller d_model / n_layers / d_ff) or raise the budget",
            input.target_params, considered, smallest
        ));
    };

    let teacher_n = input.teacher_spec.n_layers;
    let scores = match input.weights {
        Some(wm) => {
            let table = crate::cep_importance::analyse_weight_map(
                wm,
                &input.teacher_spec.n_heads,
                teacher_n,
                &RooflineSlackTable::default(),
            );
            normalized_layer_scores(&table, teacher_n)
        }
        // No weights, no scores: the mapping is by depth alone.
        None => vec![None; teacher_n as usize],
    };
    let layer_mapping = select_layer_mapping(&scores, chosen.spec.n_layers);
    let teacher_scored = input.weights.is_some();
    let unscored_teacher_layers = if teacher_scored {
        (0..teacher_n).filter(|&l| scores[l as usize].is_none()).collect()
    } else {
        Vec::new()
    };

    Ok(StudentDesign {
        spec: chosen.spec.clone(),
        profile: chosen.profile.clone(),
        candidates_considered: considered,
        candidates_within_budget: within_budget,
        layer_mapping,
        teacher_scored,
        unscored_teacher_layers,
    })
}

/// Human-readable parameter count (K/M/B), mirroring the CLI budget syntax.
fn format_params(n: u64) -> String {
    let nf = n as f64;
    if n >= 1_000_000_000 {
        format!("{:.2}B", nf / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", nf / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", nf / 1e3)
    } else {
        format!("{n}")
    }
}

fn dims_line(spec: &ModelSpec) -> String {
    // NAS candidates are uniform per layer (SearchAxes::enumerate), so the
    // first layer's entries describe the whole stack.
    format!(
        "d_model={}, n_layers={}, n_heads={}, n_kv_heads={}, d_ff={}",
        spec.d_model,
        spec.n_layers,
        spec.n_heads.first().copied().unwrap_or(0),
        spec.n_kv_heads.first().copied().unwrap_or(0),
        spec.d_ff.first().copied().unwrap_or(0),
    )
}

/// Render the student-design report (CepPlan::render_report conventions:
/// plain `writeln!`, no unicode tables).
pub fn render_design_report(
    design: &StudentDesign,
    teacher_spec: &ModelSpec,
    target_params: u64,
    target_gpu: &str,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    writeln!(s, "=== CPKD Student Design Report ===").unwrap();
    writeln!(s, "Target: {target_gpu}").unwrap();
    writeln!(
        s,
        "Parameter budget: {} ({} params)",
        format_params(target_params),
        target_params
    )
    .unwrap();
    writeln!(s).unwrap();

    let tp = teacher_spec.param_count();
    let sp = design.spec.param_count();
    let reduction = if tp > 0 {
        100.0 * (1.0 - sp as f64 / tp as f64)
    } else {
        0.0
    };
    writeln!(
        s,
        "Teacher: params={} ({})",
        format_params(tp),
        dims_line(teacher_spec)
    )
    .unwrap();
    writeln!(
        s,
        "Student: params={} ({:.1}% reduction) ({})",
        format_params(sp),
        reduction,
        dims_line(&design.spec)
    )
    .unwrap();
    writeln!(
        s,
        "Profile: latency={:.1}us, peak memory={:.2}GB, roofline utilization={:.1}%",
        design.profile.estimated_latency_us,
        design.profile.peak_memory_bytes as f64 / 1e9,
        100.0 * design.profile.roofline_utilization
    )
    .unwrap();
    writeln!(
        s,
        "Candidates: {} enumerated, {} within budget",
        design.candidates_considered, design.candidates_within_budget
    )
    .unwrap();
    writeln!(s).unwrap();

    writeln!(
        s,
        "Teacher -> student layer mapping (one teacher layer per student layer, \
         spread over the teacher's depth, last to last; {}):",
        if design.teacher_scored {
            "within a depth segment the layer of highest position-normalized importance"
        } else {
            "no teacher weights given, so by depth alone"
        }
    )
    .unwrap();
    if design.layer_mapping.is_empty() {
        writeln!(s, "  (none)").unwrap();
    }
    for e in &design.layer_mapping {
        let importance = match e.teacher_importance {
            Some(v) => format!("  (importance {v:.4})"),
            None if design.teacher_scored => "  (no importance score)".to_string(),
            None => String::new(),
        };
        writeln!(s, "  student {:>2} <- teacher {:>2}{importance}", e.student_layer, e.teacher_layer).unwrap();
    }
    if !design.unscored_teacher_layers.is_empty() {
        let layers: Vec<String> = design.unscored_teacher_layers.iter().map(u32::to_string).collect();
        writeln!(
            s,
            "Teacher layers with no importance score (none of their weight names matched): {}",
            layers.join(", ")
        )
        .unwrap();
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_importance::LayerImportance;
    use crate::cep_oracle::{Activation, NormType};

    /// 2x2 grid: d_model in {128, 256} x d_ff in {256, 512} — all four pass
    /// the enumerate() shape filters (128/256 pow2, d_ff mult-64, 4 % 4 == 0,
    /// d % 4 == 0).
    fn tiny_axes() -> SearchAxes {
        SearchAxes {
            d_model: vec![128, 256],
            n_layers: vec![2],
            n_heads: vec![4],
            n_kv_heads: vec![4],
            d_ff: vec![256, 512],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 1000,
            head_dim: 32,
            max_seq: 128,
            batch: 1,
            dtype_bytes: 2,
        }
    }

    fn teacher() -> ModelSpec {
        ModelSpec::uniform(256, 4, 4, 4, 32, 512, 1000)
    }

    fn input_with_budget(budget: u64) -> StudentDesignInput<'static> {
        StudentDesignInput {
            axes: tiny_axes(),
            teacher_spec: teacher(),
            weights: None,
            target_gpu: "H100-SXM",
            target_params: budget,
            objective: NasObjective::ParamEfficiency,
        }
    }

    fn candidate_param_counts() -> Vec<u64> {
        tiny_axes().enumerate().iter().map(|s| s.param_count()).collect()
    }

    // --- budget filter -----------------------------------------------------

    #[test]
    fn param_counts_order_with_dims() {
        // Sanity on the filter boundaries: bigger d_model / d_ff means
        // strictly more params.
        let small = ModelSpec::uniform(128, 2, 4, 4, 32, 256, 1000).param_count();
        let mid_ff = ModelSpec::uniform(128, 2, 4, 4, 32, 512, 1000).param_count();
        let mid_d = ModelSpec::uniform(256, 2, 4, 4, 32, 256, 1000).param_count();
        let large = ModelSpec::uniform(256, 2, 4, 4, 32, 512, 1000).param_count();
        assert!(small < mid_ff && small < mid_d);
        assert!(mid_ff < large && mid_d < large);
    }

    #[test]
    fn unbounded_budget_admits_all_candidates() {
        let design = design_student(input_with_budget(u64::MAX)).expect("design succeeds");
        assert_eq!(design.candidates_considered, 4);
        assert_eq!(design.candidates_within_budget, 4);
        assert!(design.spec.param_count() <= u64::MAX);
        // Chosen must be a real enumerated candidate.
        assert!(candidate_param_counts().contains(&design.spec.param_count()));
    }

    #[test]
    fn tight_budget_selects_smallest_candidate_only() {
        let smallest = *candidate_param_counts().iter().min().unwrap();
        let design = design_student(input_with_budget(smallest)).expect("design succeeds");
        assert_eq!(design.candidates_within_budget, 1);
        assert_eq!(design.spec.param_count(), smallest);
        // Candidates carry the axes' fixed max_seq (128), not uniform()'s
        // 1024 default.
        let mut expected = ModelSpec::uniform(128, 2, 4, 4, 32, 256, 1000);
        expected.max_seq = 128;
        assert_eq!(design.spec, expected);
    }

    #[test]
    fn budget_below_all_candidates_errs_loudly_naming_smallest() {
        let smallest = *candidate_param_counts().iter().min().unwrap();
        let err = design_student(input_with_budget(smallest - 1)).unwrap_err();
        assert!(err.contains("no student candidate fits"), "err: {err}");
        assert!(err.contains(&smallest.to_string()), "err: {err}");
        assert!(err.contains("widen the @search axes"), "err: {err}");
    }

    #[test]
    fn empty_search_space_errs_loudly() {
        let mut axes = tiny_axes();
        // d_ff = 100 is not a multiple of 64 — every candidate is filtered.
        axes.d_ff = vec![100];
        let input = StudentDesignInput {
            axes,
            teacher_spec: teacher(),
            weights: None,
            target_gpu: "H100-SXM",
            target_params: u64::MAX,
            objective: NasObjective::ParamEfficiency,
        };
        let err = design_student(input).unwrap_err();
        assert!(err.contains("no valid candidates"), "err: {err}");
    }

    #[test]
    fn unknown_gpu_errs_with_supported_list() {
        let input = StudentDesignInput {
            axes: tiny_axes(),
            teacher_spec: teacher(),
            weights: None,
            target_gpu: "NoSuchGPU-9000",
            target_params: u64::MAX,
            objective: NasObjective::ParamEfficiency,
        };
        let err = design_student(input).unwrap_err();
        assert!(err.contains("unknown student-design target GPU"), "err: {err}");
        assert!(err.contains("H100-SXM"), "err lists supported GPUs: {err}");
    }

    // --- layer mapping -----------------------------------------------------

    fn teachers(mapping: &[LayerMapEntry]) -> Vec<u32> {
        mapping.iter().map(|e| e.teacher_layer).collect()
    }

    fn scored(v: &[f64]) -> Vec<Option<f64>> {
        v.iter().copied().map(Some).collect()
    }

    /// With no scores the map is the uniform one: student `i` of `k` from
    /// teacher `(i + 1) * T / k - 1`. Worked by hand for T = 12, k = 4 and
    /// T = 32, k = 12.
    #[test]
    fn mapping_without_scores_is_uniform_in_depth() {
        assert_eq!(teachers(&select_layer_mapping(&[None; 12], 4)), vec![2, 5, 8, 11]);
        assert_eq!(
            teachers(&select_layer_mapping(&[None; 32], 12)),
            vec![1, 4, 7, 9, 12, 15, 17, 20, 23, 25, 28, 31]
        );
        let students: Vec<u32> = select_layer_mapping(&[None; 12], 4).iter().map(|e| e.student_layer).collect();
        assert_eq!(students, vec![0, 1, 2, 3]);
    }

    /// The defect this replaced: every important teacher layer at the deep
    /// end. Top-k by importance mapped a 4-layer student onto teacher layers
    /// 8..=11; the map must still take one layer from each third of the
    /// teacher and end on its last layer.
    #[test]
    fn mapping_cannot_cluster_at_one_end() {
        let mut scores = vec![0.1; 12];
        for (l, s) in scores.iter_mut().enumerate().skip(8) {
            *s = 10.0 + l as f64;
        }
        let mapping = select_layer_mapping(&scored(&scores), 4);
        let t = teachers(&mapping);
        assert_eq!(t.len(), 4);
        for (i, &layer) in t.iter().enumerate() {
            let (start, end) = (i as u32 * 3, (i as u32 + 1) * 3);
            assert!((start..end).contains(&layer), "student {i} got teacher {layer}, outside {start}..{end}");
        }
        assert_eq!(*t.last().unwrap(), 11, "last student layer maps to the last teacher layer");
    }

    /// Within a segment the highest score wins; the last student layer is
    /// pinned to the last teacher layer whatever the scores say.
    #[test]
    fn importance_chooses_within_a_segment_and_last_maps_to_last() {
        //            seg 0        seg 1        seg 2 (pinned to 5)
        let scores = [0.5, 2.0, 3.0, 1.0, 9.0, 0.2];
        let mapping = select_layer_mapping(&scored(&scores), 3);
        assert_eq!(
            mapping,
            vec![
                LayerMapEntry { teacher_layer: 1, student_layer: 0, teacher_importance: Some(2.0) },
                LayerMapEntry { teacher_layer: 2, student_layer: 1, teacher_importance: Some(3.0) },
                LayerMapEntry { teacher_layer: 5, student_layer: 2, teacher_importance: Some(0.2) },
            ]
        );
    }

    #[test]
    fn mapping_ties_break_toward_the_deeper_layer_and_are_deterministic() {
        let scores = scored(&[1.0; 6]);
        let mapping = select_layer_mapping(&scores, 3);
        assert_eq!(teachers(&mapping), vec![1, 3, 5]);
        assert_eq!(select_layer_mapping(&scores, 3), mapping);
    }

    /// A layer with no score never beats a scored one, however small.
    #[test]
    fn an_unscored_layer_ranks_below_every_scored_one() {
        let scores = [Some(1e-9), None, Some(0.5), Some(0.7)];
        let mapping = select_layer_mapping(&scores, 2);
        assert_eq!(teachers(&mapping), vec![0, 3]);
        assert_eq!(mapping[0].teacher_importance, Some(1e-9));
    }

    #[test]
    fn mapping_caps_at_teacher_depth() {
        let mapping = select_layer_mapping(&scored(&[2.0, 1.0]), 4);
        assert_eq!(teachers(&mapping), vec![0, 1]);
        assert!(select_layer_mapping(&[], 3).is_empty());
        assert!(select_layer_mapping(&scored(&[1.0, 2.0]), 0).is_empty());
    }

    #[test]
    fn normalized_scores_divide_out_position_factor() {
        // Bake the U-shaped prior into total_score, then check it comes
        // back out.
        let n = 4u32;
        let base = [2.0, 1.0, 1.0, 2.0];
        let table = ImportanceTable {
            heads: Vec::new(),
            ffns: Vec::new(),
            layers: (0..n)
                .map(|l| LayerImportance {
                    layer: l,
                    attention_score: 0.0,
                    ffn_score: 0.0,
                    total_score: base[l as usize] * position_factor(l, n),
                })
                .collect(),
        };
        let normalized = normalized_layer_scores(&table, n);
        for (got, want) in normalized.iter().zip(base.iter()) {
            let got = got.expect("every layer is in the table");
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn normalization_changes_selection_vs_raw_scores() {
        // position_factor(0, 4) = 1.3, position_factor(1, 4) = 1.1.
        // Raw scores favor layer 0 (1.43 > 1.32) but the normalized signal
        // favors layer 1 (1.2 > 1.1) — proving the prior is removed. Two
        // student layers: the first chooses within teacher layers {0, 1}.
        let n = 4u32;
        let raw = [1.43, 1.32, 0.5, 0.1];
        let table = ImportanceTable {
            heads: Vec::new(),
            ffns: Vec::new(),
            layers: (0..n)
                .map(|l| LayerImportance {
                    layer: l,
                    attention_score: 0.0,
                    ffn_score: 0.0,
                    total_score: raw[l as usize],
                })
                .collect(),
        };
        let normalized = normalized_layer_scores(&table, n);
        let mapping = select_layer_mapping(&normalized, 2);
        assert_eq!(teachers(&mapping), vec![1, 3]);
    }

    /// A layer the importance pass did not score has NO score, not zero.
    #[test]
    fn missing_table_layers_have_no_score() {
        let table = ImportanceTable::default();
        assert_eq!(normalized_layer_scores(&table, 3), vec![None, None, None]);
    }

    #[test]
    fn design_without_weights_maps_by_depth_and_reports_no_importance() {
        // Teacher has 4 layers, the student 2: students 0 and 1 learn from
        // teacher layers 1 and 3. (This used to be teacher layers 0 and 1,
        // each printed with an "importance" of 1.0 nothing had measured.)
        let design = design_student(input_with_budget(u64::MAX)).expect("design succeeds");
        assert_eq!(design.spec.n_layers, 2);
        assert_eq!(teachers(&design.layer_mapping), vec![1, 3]);
        let students: Vec<u32> = design.layer_mapping.iter().map(|e| e.student_layer).collect();
        assert_eq!(students, vec![0, 1]);
        assert!(design.layer_mapping.iter().all(|e| e.teacher_importance.is_none()));
        assert!(!design.teacher_scored);
        assert!(design.unscored_teacher_layers.is_empty());
        let report = render_design_report(&design, &teacher(), 1_000_000, "H100-SXM");
        assert!(report.contains("no teacher weights given, so by depth alone"), "{report}");
        assert!(!report.contains("importance 1.0000"), "{report}");
    }

    // --- report ------------------------------------------------------------

    #[test]
    fn report_contains_header_dims_and_mapping() {
        let design = design_student(input_with_budget(u64::MAX)).expect("design succeeds");
        let report = render_design_report(&design, &teacher(), 1_000_000, "H100-SXM");
        assert!(report.starts_with("=== CPKD Student Design Report ===\n"));
        assert!(report.contains("Target: H100-SXM"));
        assert!(report.contains("Parameter budget: 1.00M (1000000 params)"));
        assert!(report.contains("Teacher: params="));
        assert!(report.contains("% reduction"));
        assert!(report.contains("d_model="));
        assert!(report.contains("Candidates: 4 enumerated, 4 within budget"));
        assert!(report.contains("Teacher -> student layer mapping"));
        assert!(report.contains("student  0 <- teacher  1"));
        assert!(report.contains("student  1 <- teacher  3"));
        // ASCII only (repo doctrine for report text consumed in terminals).
        assert!(report.is_ascii());
    }

    #[test]
    fn format_params_covers_suffix_ranges() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(125_000), "125.0K");
        assert_eq!(format_params(125_000_000), "125.00M");
        assert_eq!(format_params(1_100_000_000), "1.10B");
    }
}
