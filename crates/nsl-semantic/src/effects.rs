//! M51: Effect system — tracks IO, Random, Mutation, Communication through the type system.
//!
//! Pipeline: Phase 1 (local inference) → Phase 2 (call graph propagation) → Phase 3 (assertion validation)

use std::collections::{HashMap, HashSet};
use nsl_errors::Diagnostic;

// ---------------------------------------------------------------------------
// EffectSet — bitset of computational effects
// ---------------------------------------------------------------------------

/// Set of effects a function may perform. Bitset for efficient composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EffectSet(u8);

impl EffectSet {
    pub const PURE: EffectSet = EffectSet(0);
    pub const IO: EffectSet = EffectSet(0b0001);
    pub const RANDOM: EffectSet = EffectSet(0b0010);
    pub const MUTATION: EffectSet = EffectSet(0b0100);
    pub const COMMUNICATION: EffectSet = EffectSet(0b1000);

    /// Union of two effect sets (caller inherits callee's effects).
    pub fn union(self, other: EffectSet) -> EffectSet {
        EffectSet(self.0 | other.0)
    }

    /// Check if this set contains a specific effect.
    pub fn contains(self, effect: EffectSet) -> bool {
        (self.0 & effect.0) == effect.0
    }

    /// Check if this set is pure (no effects).
    pub fn is_pure(self) -> bool {
        self.0 == 0
    }

    /// Human-readable effect list.
    pub fn display(&self) -> String {
        if self.is_pure() { return "Pure".to_string(); }
        let mut parts = Vec::new();
        if self.contains(Self::IO) { parts.push("IO"); }
        if self.contains(Self::RANDOM) { parts.push("Random"); }
        if self.contains(Self::MUTATION) { parts.push("Mutation"); }
        if self.contains(Self::COMMUNICATION) { parts.push("Communication"); }
        parts.join(" + ")
    }

    /// Check if this set is deterministic (no Random).
    ///
    /// KNOWN LIMITATION (M51a): This rejects ALL Random, even with explicit Rng.
    /// The spec allows @deterministic functions with explicit Rng parameters.
    /// M51b should add has_explicit_rng() detection to distinguish implicit vs explicit RNG.
    pub fn is_deterministic(self) -> bool {
        !self.contains(Self::RANDOM)
    }
}

impl std::ops::BitOr for EffectSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { self.union(rhs) }
}

impl std::ops::BitOrAssign for EffectSet {
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

// ---------------------------------------------------------------------------
// Built-in effect classification
// ---------------------------------------------------------------------------

/// Classify a known function/builtin by its effects.
pub fn classify_builtin_effects(name: &str) -> EffectSet {
    match name {
        // IO effects
        "print" | "println" | "eprint" | "eprintln"
        | "read_file" | "write_file" | "open" | "close"
        | "nsl_trace_record_op" | "nsl_trace_flush"
        => EffectSet::IO,

        // Random effects
        "rand" | "randn" | "random" | "random_normal" | "random_uniform"
        | "dropout" | "bernoulli"
        => EffectSet::RANDOM,

        // Mutation effects
        "nsl_tensor_add_inplace" | "nsl_tensor_zero_inplace"
        | "nsl_tensor_copy_data" | "nsl_grad_zero"
        => EffectSet::MUTATION,

        // Communication effects
        "nsl_tp_all_reduce_sum" | "nsl_tp_all_gather" | "nsl_tp_broadcast"
        | "nsl_tp_barrier" | "nsl_pipeline_send" | "nsl_pipeline_recv"
        | "nsl_pipeline_send_grad" | "nsl_pipeline_recv_grad"
        | "nccl_all_reduce" | "nccl_all_gather" | "nccl_broadcast"
        => EffectSet::COMMUNICATION,

        // Known pure operations (tensor math, reductions, activations, creation)
        "relu" | "gelu" | "silu" | "sigmoid" | "tanh" | "softmax" | "log_softmax"
        | "exp" | "log" | "sqrt" | "abs" | "sign" | "clamp" | "tensor_sin" | "tensor_cos" | "rotate_half"
        | "matmul" | "nsl_tensor_matmul" | "nsl_tensor_add" | "nsl_tensor_sub"
        | "nsl_tensor_mul" | "nsl_tensor_div" | "nsl_tensor_neg"
        | "sum" | "mean" | "max" | "min" | "argmax"
        | "nsl_tensor_sum" | "nsl_tensor_mean" | "nsl_tensor_reduce_max"
        | "reshape" | "transpose" | "unsqueeze" | "squeeze" | "cat" | "stack" | "slice"
        | "zeros" | "ones" | "full" | "arange"
        | "nsl_tensor_zeros" | "nsl_tensor_ones" | "nsl_tensor_full"
        | "nsl_tensor_reshape" | "nsl_tensor_transpose" | "nsl_tensor_clone"
        | "cross_entropy" | "mse_loss" | "l1_loss" | "bce_loss"
        | "layernorm" | "rmsnorm" | "embedding_lookup"
        | "nsl_tensor_softmax" | "nsl_tensor_gather"
        => EffectSet::PURE,

        // Unknown/unclassified: conservatively assume ALL effects.
        // This is the safe default — an unrecognized builtin in a @pure
        // function will cause a compile error, forcing explicit classification.
        _ => EffectSet::IO | EffectSet::RANDOM | EffectSet::MUTATION | EffectSet::COMMUNICATION,
    }
}

// ---------------------------------------------------------------------------
// EffectChecker
// ---------------------------------------------------------------------------

/// Semantic pass that infers and validates function effects.
pub struct EffectChecker {
    /// Inferred effect set for each function.
    fn_effects: HashMap<String, EffectSet>,

    /// Call graph: caller → list of callees.
    call_graph: HashMap<String, Vec<String>>,

    /// Functions annotated @pure.
    pure_fns: HashSet<String>,

    /// Functions annotated @deterministic.
    deterministic_fns: HashSet<String>,

    /// Functions annotated @checkpoint.
    checkpointed_fns: HashSet<String>,

    pub diagnostics: Vec<Diagnostic>,
}

impl EffectChecker {
    pub fn new() -> Self {
        EffectChecker {
            fn_effects: HashMap::new(),
            call_graph: HashMap::new(),
            pure_fns: HashSet::new(),
            deterministic_fns: HashSet::new(),
            checkpointed_fns: HashSet::new(),
            diagnostics: Vec::new(),
        }
    }

    // --- Phase 0: Registration ---

    /// Register a function's direct (local) effects and its callees.
    pub fn register_function(&mut self, name: &str, local_effects: EffectSet, callees: Vec<String>) {
        self.fn_effects.insert(name.to_string(), local_effects);
        self.call_graph.insert(name.to_string(), callees);
    }

    /// Mark a function as @pure.
    pub fn mark_pure(&mut self, name: &str) {
        self.pure_fns.insert(name.to_string());
    }

    /// Mark a function as @deterministic.
    pub fn mark_deterministic(&mut self, name: &str) {
        self.deterministic_fns.insert(name.to_string());
    }

    /// Mark a function as @checkpoint.
    pub fn mark_checkpointed(&mut self, name: &str) {
        self.checkpointed_fns.insert(name.to_string());
    }

    // --- Phase 2: Propagation ---

    /// Propagate effects through the call graph (transitive closure).
    ///
    /// After this, fn_effects[f] contains the union of f's local effects
    /// and all effects of functions f calls (directly or transitively).
    pub fn propagate(&mut self) {
        // Fixed-point iteration: keep propagating until no changes
        let mut changed = true;
        while changed {
            changed = false;
            let names: Vec<String> = self.fn_effects.keys().cloned().collect();
            for name in &names {
                let callees = self.call_graph.get(name).cloned().unwrap_or_default();
                let mut total = self.fn_effects.get(name).copied().unwrap_or(EffectSet::PURE);
                for callee in &callees {
                    let callee_effects = self.fn_effects.get(callee).copied()
                        .unwrap_or_else(|| classify_builtin_effects(callee));
                    total |= callee_effects;
                }
                if total != self.fn_effects.get(name).copied().unwrap_or(EffectSet::PURE) {
                    self.fn_effects.insert(name.clone(), total);
                    changed = true;
                }
            }
        }
    }

    // --- Phase 3: Validation ---

    /// Validate all @pure, @deterministic, @checkpoint assertions.
    pub fn validate(&mut self) {
        // @pure: must have no effects
        for name in self.pure_fns.clone() {
            let effects = self.fn_effects.get(&name).copied().unwrap_or(EffectSet::PURE);
            if !effects.is_pure() {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "@pure function '{}' has effects: {}",
                        name, effects.display()
                    )),
                );
            }
        }

        // @deterministic: must have no Random effect
        for name in self.deterministic_fns.clone() {
            let effects = self.fn_effects.get(&name).copied().unwrap_or(EffectSet::PURE);
            if !effects.is_deterministic() {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "@deterministic function '{}' has Random effect — \
                         use explicit Rng parameter instead of implicit RNG",
                        name
                    )),
                );
            }
        }

        // @checkpoint: requires @pure (recomputation must produce same result)
        for name in self.checkpointed_fns.clone() {
            let effects = self.fn_effects.get(&name).copied().unwrap_or(EffectSet::PURE);
            if !effects.is_pure() {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "@checkpoint function '{}' has effects: {} — \
                         @checkpoint recomputes during backward, requires @pure",
                        name, effects.display()
                    )),
                );
            }
        }
    }

    // --- Queries ---

    /// Get the inferred effects for a function.
    pub fn get_effects(&self, name: &str) -> EffectSet {
        self.fn_effects.get(name).copied().unwrap_or(EffectSet::PURE)
    }

    /// Check if a function is pure.
    pub fn is_pure(&self, name: &str) -> bool {
        self.get_effects(name).is_pure()
    }

    /// Run the full analysis: propagate + validate.
    pub fn analyze(&mut self) {
        self.propagate();
        self.validate();
    }
}

impl Default for EffectChecker {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EffectSet tests ---

    #[test]
    fn pure_is_zero() {
        assert!(EffectSet::PURE.is_pure());
        assert_eq!(EffectSet::PURE.0, 0);
    }

    #[test]
    fn union_combines_effects() {
        let io_random = EffectSet::IO | EffectSet::RANDOM;
        assert!(io_random.contains(EffectSet::IO));
        assert!(io_random.contains(EffectSet::RANDOM));
        assert!(!io_random.contains(EffectSet::MUTATION));
    }

    #[test]
    fn display_single() {
        assert_eq!(EffectSet::IO.display(), "IO");
        assert_eq!(EffectSet::RANDOM.display(), "Random");
        assert_eq!(EffectSet::PURE.display(), "Pure");
    }

    #[test]
    fn display_combined() {
        let effects = EffectSet::IO | EffectSet::RANDOM;
        assert_eq!(effects.display(), "IO + Random");
    }

    #[test]
    fn deterministic_check() {
        assert!(EffectSet::PURE.is_deterministic());
        assert!(EffectSet::IO.is_deterministic()); // IO is deterministic
        assert!(!EffectSet::RANDOM.is_deterministic()); // Random is not
        assert!(EffectSet::COMMUNICATION.is_deterministic()); // Communication alone is ok
    }

    // --- classify_builtin_effects tests ---

    #[test]
    fn classify_known_builtins() {
        assert_eq!(classify_builtin_effects("print"), EffectSet::IO);
        assert_eq!(classify_builtin_effects("rand"), EffectSet::RANDOM);
        assert_eq!(classify_builtin_effects("dropout"), EffectSet::RANDOM);
        assert_eq!(classify_builtin_effects("nsl_tp_all_reduce_sum"), EffectSet::COMMUNICATION);
        assert_eq!(classify_builtin_effects("nsl_tensor_add_inplace"), EffectSet::MUTATION);
    }

    #[test]
    fn classify_known_pure_ops() {
        assert_eq!(classify_builtin_effects("relu"), EffectSet::PURE);
        assert_eq!(classify_builtin_effects("matmul"), EffectSet::PURE);
        assert_eq!(classify_builtin_effects("softmax"), EffectSet::PURE);
        assert_eq!(classify_builtin_effects("cross_entropy"), EffectSet::PURE);
    }

    #[test]
    fn classify_unknown_is_conservative() {
        // Unknown builtins conservatively assumed to have all effects
        let all = classify_builtin_effects("some_unknown_function");
        assert!(all.contains(EffectSet::IO));
        assert!(all.contains(EffectSet::RANDOM));
        assert!(all.contains(EffectSet::MUTATION));
        assert!(all.contains(EffectSet::COMMUNICATION));
    }

    // --- EffectChecker tests ---

    #[test]
    fn pure_function_validates() {
        let mut checker = EffectChecker::new();
        checker.register_function("attention", EffectSet::PURE, vec!["matmul".into(), "softmax".into()]);
        checker.mark_pure("attention");
        checker.analyze();
        assert!(checker.diagnostics.is_empty());
        assert!(checker.is_pure("attention"));
    }

    #[test]
    fn pure_function_with_io_fails() {
        let mut checker = EffectChecker::new();
        checker.register_function("noisy", EffectSet::PURE, vec!["print".into()]);
        checker.mark_pure("noisy");
        checker.analyze();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("IO"));
    }

    #[test]
    fn effect_propagation_through_call_graph() {
        let mut checker = EffectChecker::new();
        // f calls g, g calls print (IO)
        checker.register_function("g", EffectSet::PURE, vec!["print".into()]);
        checker.register_function("f", EffectSet::PURE, vec!["g".into()]);
        checker.analyze();
        assert!(checker.get_effects("g").contains(EffectSet::IO));
        assert!(checker.get_effects("f").contains(EffectSet::IO)); // propagated
    }

    #[test]
    fn deterministic_with_random_fails() {
        let mut checker = EffectChecker::new();
        checker.register_function("train_step", EffectSet::PURE, vec!["dropout".into()]);
        checker.mark_deterministic("train_step");
        checker.analyze();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("Random"));
    }

    #[test]
    fn deterministic_without_random_passes() {
        let mut checker = EffectChecker::new();
        checker.register_function("forward", EffectSet::PURE, vec!["matmul".into(), "relu".into()]);
        checker.mark_deterministic("forward");
        checker.analyze();
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn checkpoint_requires_pure() {
        let mut checker = EffectChecker::new();
        checker.register_function("block", EffectSet::PURE, vec!["dropout".into()]);
        checker.mark_checkpointed("block");
        checker.analyze();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.diagnostics[0].message.contains("@checkpoint"));
    }

    #[test]
    fn checkpoint_pure_passes() {
        let mut checker = EffectChecker::new();
        checker.register_function("block", EffectSet::PURE, vec!["matmul".into(), "relu".into()]);
        checker.mark_checkpointed("block");
        checker.analyze();
        assert!(checker.diagnostics.is_empty());
    }

    #[test]
    fn transitive_closure_multi_hop() {
        let mut checker = EffectChecker::new();
        // a → b → c → rand (Random)
        checker.register_function("c", EffectSet::PURE, vec!["rand".into()]);
        checker.register_function("b", EffectSet::PURE, vec!["c".into()]);
        checker.register_function("a", EffectSet::PURE, vec!["b".into()]);
        checker.mark_pure("a");
        checker.analyze();
        // a should fail @pure because it transitively calls rand
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(checker.get_effects("a").contains(EffectSet::RANDOM));
    }
}
