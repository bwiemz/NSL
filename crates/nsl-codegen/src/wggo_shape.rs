//! WGGO — §2.4 inter-layer shape-compatibility constraint.
//!
//! The paper's feasibility model (§2.4) carries a hard constraint that the
//! solver's per-layer decisions must never violate:
//!
//! ```text
//! ∀ l: output_dim(l, h[l], f[l]) == input_dim(l+1, h[l+1], f[l+1])
//!      OR reshape_possible(output_dim(l), input_dim(l+1))
//! ```
//!
//! i.e. every layer's *output* activation width must equal the *input* width
//! the next layer expects, or a view/reshape must bridge them.  Without this
//! gate WGGO could emit a structurally invalid model — e.g. a thinned layer
//! whose narrowed output feeds an un-thinned successor that still expects the
//! full width.
//!
//! ## What changes a layer's inter-layer width
//!
//! For a standard **residual-stream transformer** the inter-layer activation
//! width is `d_model` and is *invariant* under WGGO's structural decisions:
//! head pruning and FFN thinning are absorbed by the output projection
//! (`W_O` / `W_down`), and a pruned layer degenerates to a residual identity.
//! So for the uniform transformers WGGO targets today this constraint is
//! satisfied-by-construction and [`validate`] returns no violations.
//!
//! Its teeth are for the cases that *can* go wrong:
//!   * **Non-residual graphs** — `OptGraph` is built from an arbitrary Wengert
//!     list (a plain stacked `matmul → matmul` chain has no residual stream),
//!     where a width decision *does* change the inter-layer tensor.
//!   * **Heterogeneous-width models** — adjacent layers with genuinely
//!     different declared widths that no reshape can bridge.
//!   * **Future dim-changing decisions** — e.g. structured width-pruning of
//!     `d_model` itself; the moment such a decision narrows an inter-layer
//!     tensor, this gate catches an incompatible successor instead of letting
//!     codegen emit a broken model.
//!
//! The module is deliberately decoupled from the solver types: it operates on
//! plain [`LayerShapeInfo`] records (layer id, input/output widths, producer
//! edges), so `wggo::run` adapts the assembled plan into these records and
//! treats any returned [`ShapeViolation`] as a refusal (surfaced as a warning,
//! consistent with the project invariant that an unmet transformation
//! precondition must refuse, never silently weaken).

use serde::Serialize;

/// Paper §2.4 `reshape_possible(a, b)`.
///
/// A producer activation of width `a` can feed a consumer expecting width `b`
/// *without* an explicit re-projection iff one width evenly divides the other:
/// a `[.., a]` tensor then views/reshapes (or broadcasts) into `[.., b]`
/// without changing the element count along unrelated axes.  Equal widths are
/// the trivial divisible case.  Zero widths are never reshape-compatible (a
/// degenerate/unknown extent, not a real bridge).
pub fn reshape_possible(a: u64, b: u64) -> bool {
    a != 0 && b != 0 && (a.is_multiple_of(b) || b.is_multiple_of(a))
}

/// Compatibility verdict for a single producer→consumer activation edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeCompat {
    /// Widths are equal — no bridge needed.
    Exact,
    /// Widths differ but a reshape/view bridges them ([`reshape_possible`]).
    Reshape,
    /// Widths differ and no reshape bridges them — structurally invalid.
    Incompatible,
}

/// Classify one producer→consumer edge.
///
/// Unknown widths (`None` on either side) classify as [`ShapeCompat::Exact`]:
/// we cannot *prove* an incompatibility, and the gate must never reject a
/// valid model on missing information (fail-safe — only provable violations
/// are flagged).
pub fn classify(producer_out: Option<u64>, consumer_in: Option<u64>) -> ShapeCompat {
    match (producer_out, consumer_in) {
        // A zero width is a degenerate / unknown extent (a defensively
        // zero-initialised shape), not a meaningful equal-width match.  Treat
        // it as unknown — the same fail-safe as `None` — consistent with
        // `reshape_possible`, which already rejects zero operands.  Without
        // this arm `classify(Some(0), Some(0))` would falsely report `Exact`
        // and `classify(Some(0), Some(5))` would falsely report `Incompatible`.
        (Some(0), _) | (_, Some(0)) => ShapeCompat::Exact,
        (Some(a), Some(b)) if a == b => ShapeCompat::Exact,
        (Some(a), Some(b)) if reshape_possible(a, b) => ShapeCompat::Reshape,
        (Some(_), Some(_)) => ShapeCompat::Incompatible,
        _ => ShapeCompat::Exact,
    }
}

/// Inter-layer shape record for one layer.
///
/// `input_dim` / `output_dim` are the widths of the activation tensor the
/// layer *consumes* / *produces* on the inter-layer stream (for a residual
/// transformer both are `d_model`).  `None` means "unknown" — see
/// [`classify`] for the fail-safe treatment.  `depends_on` lists the layer
/// indices whose outputs feed this layer (mirrors `wggo_graph::Layer`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayerShapeInfo {
    pub layer: u32,
    pub name: String,
    pub input_dim: Option<u64>,
    pub output_dim: Option<u64>,
    pub depends_on: Vec<u32>,
}

/// A provably-incompatible producer→consumer edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShapeViolation {
    pub producer: u32,
    pub producer_name: String,
    pub consumer: u32,
    pub consumer_name: String,
    pub producer_out: u64,
    pub consumer_in: u64,
}

impl ShapeViolation {
    /// One-line diagnostic for the `[wggo]` stderr / report surface.
    pub fn message(&self) -> String {
        format!(
            "shape_incompatible layer:{} ({}) outputs width {} but layer:{} ({}) expects width {} \
             with no reshape bridge",
            self.producer,
            self.producer_name,
            self.producer_out,
            self.consumer,
            self.consumer_name,
            self.consumer_in,
        )
    }
}

/// Validate the §2.4 shape-compatibility constraint over the whole layer graph.
///
/// Walks every producer→consumer edge (each consumer's `depends_on`) and
/// collects the edges that are provably [`ShapeCompat::Incompatible`].  An
/// empty result means the constraint holds.  Edges whose endpoints have
/// unknown widths are treated as compatible (fail-safe).
pub fn validate(infos: &[LayerShapeInfo]) -> Vec<ShapeViolation> {
    use std::collections::HashMap;
    let by_id: HashMap<u32, &LayerShapeInfo> = infos.iter().map(|i| (i.layer, i)).collect();
    let mut out = Vec::new();
    for consumer in infos {
        for &prod_id in &consumer.depends_on {
            let Some(producer) = by_id.get(&prod_id) else {
                continue;
            };
            if classify(producer.output_dim, consumer.input_dim) == ShapeCompat::Incompatible {
                out.push(ShapeViolation {
                    producer: producer.layer,
                    producer_name: producer.name.clone(),
                    consumer: consumer.layer,
                    consumer_name: consumer.name.clone(),
                    // `classify` returns `Incompatible` only when both widths
                    // are `Some(_)` and non-zero, so these are infallible here.
                    // `expect` (not `unwrap_or(0)`) keeps the invariant
                    // self-documenting: a future `classify` change that broke it
                    // would surface loudly instead of emitting a
                    // self-contradictory "width 0 → width 0" violation.
                    producer_out: producer
                        .output_dim
                        .expect("classify Incompatible implies producer output_dim is Some"),
                    consumer_in: consumer
                        .input_dim
                        .expect("classify Incompatible implies consumer input_dim is Some"),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn info(layer: u32, input: Option<u64>, output: Option<u64>, deps: &[u32]) -> LayerShapeInfo {
        LayerShapeInfo {
            layer,
            name: format!("blocks.{layer}"),
            input_dim: input,
            output_dim: output,
            depends_on: deps.to_vec(),
        }
    }

    #[test]
    fn reshape_possible_is_true_for_equal_and_divisible_widths() {
        assert!(reshape_possible(512, 512)); // equal
        assert!(reshape_possible(512, 256)); // a multiple of b
        assert!(reshape_possible(256, 512)); // b multiple of a
        assert!(reshape_possible(768, 256)); // 768 = 3 * 256
    }

    #[test]
    fn reshape_possible_is_false_for_coprime_and_zero_widths() {
        assert!(!reshape_possible(512, 768)); // neither divides the other
        assert!(!reshape_possible(100, 30));
        assert!(!reshape_possible(0, 512)); // degenerate
        assert!(!reshape_possible(512, 0));
    }

    #[test]
    fn classify_distinguishes_exact_reshape_incompatible() {
        assert_eq!(classify(Some(512), Some(512)), ShapeCompat::Exact);
        assert_eq!(classify(Some(512), Some(256)), ShapeCompat::Reshape);
        assert_eq!(classify(Some(512), Some(768)), ShapeCompat::Incompatible);
    }

    #[test]
    fn classify_treats_unknown_width_as_compatible() {
        // Fail-safe: missing information must never reject a valid model.
        assert_eq!(classify(None, Some(512)), ShapeCompat::Exact);
        assert_eq!(classify(Some(512), None), ShapeCompat::Exact);
        assert_eq!(classify(None, None), ShapeCompat::Exact);
    }

    #[test]
    fn classify_treats_zero_width_as_unknown_not_a_match() {
        // A zero width is degenerate, not a real equal-width match: it must be
        // fail-safe (Exact), never a false Exact-on-both-zero or a false
        // Incompatible-on-zero-vs-nonzero.
        assert_eq!(classify(Some(0), Some(0)), ShapeCompat::Exact);
        assert_eq!(classify(Some(0), Some(512)), ShapeCompat::Exact);
        assert_eq!(classify(Some(512), Some(0)), ShapeCompat::Exact);
    }

    #[test]
    fn validate_passes_uniform_residual_chain() {
        // The transformer norm: every layer carries d_model = 512 in and out.
        let infos = vec![
            info(0, Some(512), Some(512), &[]),
            info(1, Some(512), Some(512), &[0]),
            info(2, Some(512), Some(512), &[1]),
        ];
        assert!(validate(&infos).is_empty());
    }

    #[test]
    fn validate_flags_thinned_producer_feeding_unthinned_consumer() {
        // The audit's exact danger: layer 0 narrows its output to 384 (a
        // width-defining, non-residual layer that was thinned) but layer 1
        // still expects the full 512, and 512 is not a multiple of 384.
        let infos = vec![
            info(0, Some(512), Some(384), &[]),
            info(1, Some(512), Some(512), &[0]),
        ];
        let v = validate(&infos);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].producer, 0);
        assert_eq!(v[0].consumer, 1);
        assert_eq!(v[0].producer_out, 384);
        assert_eq!(v[0].consumer_in, 512);
    }

    #[test]
    fn validate_allows_reshape_bridgeable_mismatch() {
        // 512 → 256 is a clean view (256 divides 512): no violation.
        let infos = vec![
            info(0, Some(512), Some(256), &[]),
            info(1, Some(256), Some(256), &[0]),
        ];
        assert!(validate(&infos).is_empty());
    }

    #[test]
    fn validate_reports_one_violation_per_bad_edge() {
        // Layer 2 fans in from both 0 (ok) and 1 (bad).
        let infos = vec![
            info(0, Some(512), Some(512), &[]),
            info(1, Some(512), Some(300), &[0]),
            info(2, Some(512), Some(512), &[0, 1]),
        ];
        let v = validate(&infos);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].producer, 1);
        assert_eq!(v[0].consumer, 2);
    }

    #[test]
    fn validate_skips_edges_with_unknown_widths() {
        let infos = vec![
            info(0, None, None, &[]),
            info(1, Some(512), Some(512), &[0]),
        ];
        assert!(validate(&infos).is_empty());
    }

    #[test]
    fn violation_message_names_both_layers_and_widths() {
        let v = ShapeViolation {
            producer: 3,
            producer_name: "blocks.3".into(),
            consumer: 4,
            consumer_name: "blocks.4".into(),
            producer_out: 384,
            consumer_in: 512,
        };
        let m = v.message();
        assert!(m.contains("blocks.3"));
        assert!(m.contains("blocks.4"));
        assert!(m.contains("384"));
        assert!(m.contains("512"));
    }

    #[test]
    fn validate_ignores_dangling_dependency_ids() {
        // depends_on references a layer not present — skip, don't panic.
        let infos = vec![info(1, Some(512), Some(512), &[99])];
        assert!(validate(&infos).is_empty());
    }
}
