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

/// What a hook needs the calibration run to observe.  The driver unions
/// these across all enabled hooks into an `ObservationPlan`.
#[derive(Debug, Clone)]
pub enum ObservationSet {
    Empty,
    ForwardActivations(Vec<LayerRef>),
    BackwardGradients(Vec<LayerRef>),
    Weights(Vec<ParamRef>),
    Union(Vec<ObservationSet>),
}

/// Concrete plan for the calibration binary: the union of every hook's
/// requirements, with duplicates collapsed.
#[derive(Debug, Clone, Default)]
pub struct ObservationPlan {
    pub forward_activations: BTreeSet<LayerRef>,
    pub backward_gradients: BTreeSet<LayerRef>,
    pub weights: BTreeSet<ParamRef>,
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
