//! WGGO — §2.4 inter-layer shape-compatibility constraint.
//!
//! The paper's feasibility model (§2.4) carries a hard constraint that the
//! solver's per-layer decisions must never violate:
//!
//! ```text
//! ∀ l: output_dim(l, h[l], f[l]) == input_dim(l+1, h[l+1], f[l+1])
//! ```
//!
//! i.e. every layer's *output* activation width must equal the *input* width
//! the next layer expects.
//!
//! ## Which decision can break it
//!
//! The model being compiled is type-checked, so as written every edge
//! already satisfies the constraint. Of WGGO's structural decisions, head
//! pruning and FFN thinning are absorbed by the layer's own output
//! projection (`W_O` / `W_down`) and leave the width a layer hands on
//! untouched. The one decision that changes an inter-layer tensor is
//! pruning a whole layer: the layer is replaced by the identity on its
//! input, so its successors receive a tensor as wide as its *input* where
//! they were written for its *output*. That is sound exactly when the two
//! widths are equal, which holds for a block on a residual stream and fails
//! for a layer that changes the width (a projection between two stream
//! widths, a stacked `matmul → matmul` chain with no residual).
//!
//! [`validate_pruned`] checks that, on the widths read from the model
//! ([`crate::wggo_model_shape::LayerGeometry`]). `wggo::run` treats any
//! returned [`ShapeViolation`] as a refusal: it emits no structural
//! transforms at all, consistent with the project invariant that an unmet
//! transformation precondition must refuse, never silently weaken.
//!
//! An earlier form compared every producer→consumer edge using one
//! `d_model` for every layer, so it could not fire, and accepted any pair
//! of widths where one divides the other as "reshape-bridgeable", which is
//! not a property of a reshape: a `[.., 512]` tensor does not become a
//! `[.., 256]` one without changing another dimension, and nothing inserts
//! such a reshape.

use serde::Serialize;

/// Inter-layer shape record for one layer.
///
/// `input_dim` / `output_dim` are the widths (last dimension) of the
/// activation tensor the layer *consumes* from outside and *produces* for
/// other layers. `None` means "unknown" — the layer reads or writes tensors
/// of more than one width, or a shape the compiler could not prove.
/// `depends_on` lists the layer indices whose outputs feed this layer
/// (mirrors `wggo_graph::Layer`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayerShapeInfo {
    pub layer: u32,
    pub name: String,
    pub input_dim: Option<u64>,
    pub output_dim: Option<u64>,
    pub depends_on: Vec<u32>,
}

/// A pruned layer that provably changes the width of its stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShapeViolation {
    pub layer: u32,
    pub layer_name: String,
    pub input_dim: u64,
    pub output_dim: u64,
}

impl ShapeViolation {
    /// One-line diagnostic for the `[wggo]` stderr / report surface.
    pub fn message(&self) -> String {
        format!(
            "shape_incompatible layer:{} ({}) cannot be pruned: it reads a tensor of width {} \
             and hands on one of width {}, so replacing it with the identity changes what its \
             successors receive",
            self.layer, self.layer_name, self.input_dim, self.output_dim,
        )
    }
}

/// Check the §2.4 constraint for the layers a plan prunes.
///
/// `pruned` holds one record per pruned layer. A layer is a violation when
/// both its widths are known, non-zero and different. A layer with an
/// unknown (or zero, i.e. defensively initialised) width is not flagged:
/// the gate must never reject a valid model on missing information, only on
/// a proven mismatch.
pub fn validate_pruned(pruned: &[LayerShapeInfo]) -> Vec<ShapeViolation> {
    pruned
        .iter()
        .filter_map(|l| match (l.input_dim, l.output_dim) {
            (Some(i), Some(o)) if i != 0 && o != 0 && i != o => Some(ShapeViolation {
                layer: l.layer,
                layer_name: l.name.clone(),
                input_dim: i,
                output_dim: o,
            }),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn info(layer: u32, input: Option<u64>, output: Option<u64>) -> LayerShapeInfo {
        LayerShapeInfo {
            layer,
            name: format!("blocks.{layer}"),
            input_dim: input,
            output_dim: output,
            depends_on: Vec::new(),
        }
    }

    #[test]
    fn a_pruned_residual_block_is_compatible() {
        // A block on a 512-wide residual stream is the identity's shape.
        assert!(validate_pruned(&[info(2, Some(512), Some(512))]).is_empty());
    }

    #[test]
    fn a_pruned_layer_that_changes_the_width_is_a_violation() {
        // 512 -> 256: the successor was written for 256 and would get 512.
        // One width dividing the other does not make this safe.
        let v = validate_pruned(&[info(3, Some(512), Some(256))]);
        assert_eq!(
            v,
            vec![ShapeViolation {
                layer: 3,
                layer_name: "blocks.3".into(),
                input_dim: 512,
                output_dim: 256,
            }]
        );
        // And the other direction, and a coprime pair.
        assert_eq!(validate_pruned(&[info(0, Some(256), Some(512))]).len(), 1);
        assert_eq!(validate_pruned(&[info(0, Some(512), Some(768))]).len(), 1);
    }

    #[test]
    fn unknown_or_zero_widths_are_not_flagged() {
        // Fail-safe: missing information must never reject a valid model.
        assert!(validate_pruned(&[info(0, None, Some(512))]).is_empty());
        assert!(validate_pruned(&[info(0, Some(512), None)]).is_empty());
        assert!(validate_pruned(&[info(0, None, None)]).is_empty());
        assert!(validate_pruned(&[info(0, Some(0), Some(512))]).is_empty());
        assert!(validate_pruned(&[info(0, Some(512), Some(0))]).is_empty());
    }

    #[test]
    fn each_offending_layer_is_reported_once() {
        let v = validate_pruned(&[
            info(1, Some(512), Some(512)),
            info(2, Some(512), Some(300)),
            info(5, Some(64), Some(32)),
        ]);
        assert_eq!(v.iter().map(|x| x.layer).collect::<Vec<_>>(), vec![2, 5]);
    }

    #[test]
    fn violation_message_names_the_layer_and_both_widths() {
        let m = ShapeViolation {
            layer: 3,
            layer_name: "blocks.3".into(),
            input_dim: 384,
            output_dim: 512,
        }
        .message();
        assert!(m.contains("blocks.3"));
        assert!(m.contains("384"));
        assert!(m.contains("512"));
        assert!(m.contains("shape_incompatible"));
    }
}
