use std::collections::BTreeSet;

/// Opaque reference to a transformer layer by `OptGraph`-style name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerRef(pub String);

impl LayerRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Opaque reference to a named parameter tensor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParamRef(pub String);

impl ParamRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// Opaque reference to a named linear projection inside the model.
/// Matches parameter-path naming used in the checkpoint and Wengert
/// list's param-op names (e.g. `"blocks.0.attn.wq"`,
/// `"blocks.0.mlp.up_proj"`).  First-class observation target for
/// hooks that want to see the tensor feeding into a specific matmul.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProjectionRef(pub String);

impl ProjectionRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// What a hook needs the calibration run to observe.  The driver unions
/// these across all enabled hooks into an `ObservationPlan`.
#[derive(Debug, Clone)]
pub enum ObservationSet {
    Empty,
    ForwardActivations(Vec<LayerRef>),
    BackwardGradients(Vec<LayerRef>),
    Weights(Vec<ParamRef>),
    LinearInputActivations(Vec<ProjectionRef>),
    Union(Vec<ObservationSet>),
}

/// Concrete plan for the calibration binary: the union of every hook's
/// requirements, with duplicates collapsed.
#[derive(Debug, Clone, Default)]
pub struct ObservationPlan {
    pub forward_activations: BTreeSet<LayerRef>,
    pub backward_gradients: BTreeSet<LayerRef>,
    pub weights: BTreeSet<ParamRef>,
    pub linear_input_activations: BTreeSet<ProjectionRef>,
}

impl ObservationPlan {
    pub fn union_of(sets: &[ObservationSet]) -> Self {
        let mut plan = Self::default();
        for s in sets {
            plan.add(s);
        }
        plan
    }

    fn add(&mut self, s: &ObservationSet) {
        match s {
            ObservationSet::Empty => {}
            ObservationSet::ForwardActivations(v) => {
                for l in v {
                    self.forward_activations.insert(l.clone());
                }
            }
            ObservationSet::BackwardGradients(v) => {
                for l in v {
                    self.backward_gradients.insert(l.clone());
                }
            }
            ObservationSet::Weights(v) => {
                for p in v {
                    self.weights.insert(p.clone());
                }
            }
            ObservationSet::LinearInputActivations(v) => {
                for p in v {
                    self.linear_input_activations.insert(p.clone());
                }
            }
            ObservationSet::Union(children) => {
                for c in children {
                    self.add(c);
                }
            }
        }
    }

    pub fn needs_backward(&self) -> bool {
        !self.backward_gradients.is_empty()
    }
}

impl ObservationSet {
    /// Returns `true` if any variant inside this set requires the
    /// forward pass to produce observable intermediate tensors
    /// (i.e. `ForwardActivations` or `LinearInputActivations` with
    /// at least one target).  Used by the calibration driver to
    /// decide whether the real subprocess codegen path (which emits
    /// model-forward IR) is required vs. the in-process stub.
    pub fn needs_forward_pass(&self) -> bool {
        match self {
            ObservationSet::Empty
            | ObservationSet::BackwardGradients(_)
            | ObservationSet::Weights(_) => false,
            ObservationSet::ForwardActivations(v) => !v.is_empty(),
            ObservationSet::LinearInputActivations(v) => !v.is_empty(),
            ObservationSet::Union(children) => children.iter().any(Self::needs_forward_pass),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_collapses_duplicates_across_hooks() {
        let a = ObservationSet::ForwardActivations(vec![
            LayerRef::new("blocks.0"),
            LayerRef::new("blocks.1"),
        ]);
        let b = ObservationSet::ForwardActivations(vec![
            LayerRef::new("blocks.1"),
            LayerRef::new("blocks.2"),
        ]);
        let plan = ObservationPlan::union_of(&[a, b]);
        assert_eq!(plan.forward_activations.len(), 3);
        assert!(plan.forward_activations.contains(&LayerRef::new("blocks.0")));
        assert!(plan.forward_activations.contains(&LayerRef::new("blocks.1")));
        assert!(plan.forward_activations.contains(&LayerRef::new("blocks.2")));
    }

    #[test]
    fn needs_backward_reflects_backward_gradients() {
        let a = ObservationSet::ForwardActivations(vec![LayerRef::new("x")]);
        assert!(!ObservationPlan::union_of(&[a]).needs_backward());

        let b = ObservationSet::BackwardGradients(vec![LayerRef::new("x")]);
        assert!(ObservationPlan::union_of(&[b]).needs_backward());
    }

    #[test]
    fn nested_union_flattens() {
        let inner = ObservationSet::Union(vec![
            ObservationSet::Weights(vec![ParamRef::new("w1")]),
            ObservationSet::Weights(vec![ParamRef::new("w2")]),
        ]);
        let outer = ObservationSet::Union(vec![
            inner,
            ObservationSet::Weights(vec![ParamRef::new("w3")]),
        ]);
        let plan = ObservationPlan::union_of(&[outer]);
        assert_eq!(plan.weights.len(), 3);
    }

    #[test]
    fn projection_ref_is_hashable_and_ordered() {
        let mut set = BTreeSet::new();
        set.insert(ProjectionRef::new("blocks.0.attn.wq"));
        set.insert(ProjectionRef::new("blocks.0.attn.wk"));
        set.insert(ProjectionRef::new("blocks.0.attn.wq")); // dedup
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn linear_input_activations_unions_into_plan() {
        let a = ObservationSet::LinearInputActivations(vec![
            ProjectionRef::new("blocks.0.attn.wq"),
            ProjectionRef::new("blocks.0.attn.wk"),
        ]);
        let b = ObservationSet::LinearInputActivations(vec![
            ProjectionRef::new("blocks.0.attn.wk"),
            ProjectionRef::new("blocks.0.attn.wv"),
        ]);
        let plan = ObservationPlan::union_of(&[a, b]);
        assert_eq!(plan.linear_input_activations.len(), 3);
        assert!(plan.linear_input_activations.contains(&ProjectionRef::new("blocks.0.attn.wq")));
        assert!(plan.linear_input_activations.contains(&ProjectionRef::new("blocks.0.attn.wk")));
        assert!(plan.linear_input_activations.contains(&ProjectionRef::new("blocks.0.attn.wv")));
    }

    #[test]
    fn linear_input_activations_nested_in_union() {
        let inner = ObservationSet::Union(vec![
            ObservationSet::LinearInputActivations(vec![ProjectionRef::new("p1")]),
            ObservationSet::LinearInputActivations(vec![ProjectionRef::new("p2")]),
        ]);
        let plan = ObservationPlan::union_of(&[inner]);
        assert_eq!(plan.linear_input_activations.len(), 2);
    }

    #[test]
    fn needs_forward_pass_detects_every_variant_correctly() {
        assert!(!ObservationSet::Empty.needs_forward_pass());
        assert!(!ObservationSet::BackwardGradients(vec![LayerRef::new("x")]).needs_forward_pass());
        assert!(!ObservationSet::Weights(vec![ParamRef::new("w")]).needs_forward_pass());
        assert!(!ObservationSet::ForwardActivations(vec![]).needs_forward_pass());
        assert!(ObservationSet::ForwardActivations(vec![LayerRef::new("l")]).needs_forward_pass());
        assert!(!ObservationSet::LinearInputActivations(vec![]).needs_forward_pass());
        assert!(ObservationSet::LinearInputActivations(vec![ProjectionRef::new("p")]).needs_forward_pass());
        // Union: true if any child needs it
        assert!(ObservationSet::Union(vec![
            ObservationSet::Weights(vec![ParamRef::new("w")]),
            ObservationSet::LinearInputActivations(vec![ProjectionRef::new("p")]),
        ]).needs_forward_pass());
        assert!(!ObservationSet::Union(vec![
            ObservationSet::Empty,
            ObservationSet::BackwardGradients(vec![LayerRef::new("x")]),
        ]).needs_forward_pass());
    }

    #[test]
    fn empty_plan_needs_nothing() {
        let plan = ObservationPlan::union_of(&[ObservationSet::Empty]);
        assert!(plan.forward_activations.is_empty());
        assert!(plan.backward_gradients.is_empty());
        assert!(plan.weights.is_empty());
        assert!(!plan.needs_backward());
    }
}
