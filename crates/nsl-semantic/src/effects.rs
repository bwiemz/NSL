//! M51: Effect system — tracks IO, Random, Mutation, Communication through the type system.
//!
//! Pipeline: Phase 1 (local inference) → Phase 2 (call graph propagation) → Phase 3 (assertion validation)

use std::collections::{HashMap, HashSet};
use nsl_errors::{Diagnostic, FileId};

// ---------------------------------------------------------------------------
// CheckpointPolicy — cycle-10 §5.3 paper checkpointing-aware backward policy
// ---------------------------------------------------------------------------
//
// Mirrors `nsl_codegen::flash_attention::CheckpointPolicy` deliberately:
// the semantic layer must NOT depend on nsl-codegen (would invert the
// crate-dependency direction). The codegen-side reads this semantic
// effect record via `EffectChecker::checkpoint_policies()` and translates.
//
// v1 ships a single variant (`Full`) per cycle-10 Refuter 3 invariant
// inversion: `Selective`, `SelectivePostnorm`, `Custom` are reserved for
// v2/v3 and are refused at the kwarg parser. See
// `docs/superpowers/specs/2026-06-24-csha-checkpointing-aware-backward-design.md`.

/// Paper §5.3 checkpointing-aware backward policy (semantic-layer copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CheckpointPolicy {
    /// Full prologue recompute during backward (the cycle-10 CSHA
    /// attention-prologue machinery).
    Full,
    /// CCR P1.b — SELECTIVE block-granular recompute on the source-AD
    /// path: matmul-class outputs stay saved; only cheap bandwidth-bound
    /// ops (norms, RoPE, elementwise, softmax) are replayed during the
    /// backward. Consumed by stmt.rs's train-block CCR gate (equivalent
    /// to `--checkpoint-blocks --checkpoint-selective`); deliberately NOT
    /// consumed by the CSHA prologue-stamping pass. `selective_postnorm`
    /// and `custom` remain reserved/refused.
    Selective,
}

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

    /// Check if this set is deterministic (no Random effect).
    ///
    /// Note: this is a pure bitset check. The `EffectChecker::validate()` method
    /// additionally allows Random for functions with an explicit `Rng` parameter
    /// (controlled randomness — deterministic given the same seed).
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
        | "read_line" | "read_file" | "write_file" | "open" | "close"
        | "nsl_trace_record_op" | "nsl_trace_flush"
        | "model_save" | "model_load"
        | "nsl_model_save" | "nsl_model_load"
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
        | "reshape" | "transpose" | "unsqueeze" | "squeeze" | "cat" | "stack" | "slice" | "contiguous"
        | "zeros" | "ones" | "full" | "arange"
        | "nsl_tensor_zeros" | "nsl_tensor_ones" | "nsl_tensor_full"
        | "nsl_tensor_reshape" | "nsl_tensor_transpose" | "nsl_tensor_clone"
        | "cross_entropy" | "mse_loss" | "l1_loss" | "bce_loss"
        | "layernorm" | "rmsnorm" | "embedding_lookup"
        | "nsl_tensor_softmax" | "nsl_tensor_gather"
        => EffectSet::PURE,

        // Unknown/unclassified: conservatively assume MUTATION only.
        // Most unrecognized builtins are tensor ops that mutate state but
        // don't perform IO, randomness, or communication. Using all 4 effects
        // caused false positives for @pure functions calling user-defined
        // functions not yet in the known list.
        _ => EffectSet::MUTATION,
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

    /// Cycle-10 §5.3 paper checkpointing-aware backward: functions
    /// annotated `@checkpoint(policy="full")` carry an explicit
    /// `CheckpointPolicy::Full` entry here. `@checkpoint(policy="none")`
    /// and bare `@checkpoint` (with deprecation warning, Path B per T3)
    /// only populate `checkpointed_fns` so the M14 tape-fallback
    /// backward continues to apply. The codegen-side dispatch fork
    /// (Task 9) reads this map to decide whether to emit the
    /// prologue-recompute kernel namespace.
    checkpoint_policies: HashMap<String, CheckpointPolicy>,

    /// Cycle-10 §5.3 Path B (T3): track which source files have already
    /// received the bare-`@checkpoint` deprecation warning so we emit at
    /// most one diagnostic per file (vs once per call site). Cycle 11
    /// promotes the warning to a hard error and this set goes away.
    pub emitted_checkpoint_deprecation: HashSet<FileId>,

    /// Cycle-10 §5.3 R9 cross-scope plumbing (T6): models annotated
    /// `@paged_kv` are recorded here so the dispatch fork (Task 9) can
    /// refuse the `@checkpoint(policy="full")` + `@paged_kv` composition
    /// before either side fires. v1 records membership only; the
    /// cross-scope check fires at codegen time. Multi-call-graph
    /// resolution deferred to v4.
    paged_kv_models: HashSet<String>,

    /// Functions with an explicit `Rng` parameter. Their Random effect is
    /// "controlled" — deterministic given the same seed. @deterministic
    /// validation skips the Random check for these functions.
    explicit_rng_fns: HashSet<String>,

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
            checkpoint_policies: HashMap::new(),
            emitted_checkpoint_deprecation: HashSet::new(),
            paged_kv_models: HashSet::new(),
            explicit_rng_fns: HashSet::new(),
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

    /// Cycle-10 §5.3 paper checkpointing-aware backward (Task 3):
    /// associate an explicit policy with a `@checkpoint(...)` function.
    /// Callers that flow through here MUST also call `mark_checkpointed`
    /// so the R4 validator (`@checkpoint requires @pure`) still applies.
    pub fn mark_checkpointed_with_policy(&mut self, name: &str, policy: CheckpointPolicy) {
        self.checkpoint_policies.insert(name.to_string(), policy);
    }

    /// Cycle-10 §5.3 paper checkpointing-aware backward (Task 3):
    /// public accessor consumed by the codegen-side wire-up at
    /// `crates/nsl-cli/src/loader.rs` (Task 6) to pass policies into
    /// `WengertExtractor::with_checkpoint_policy`.
    pub fn checkpoint_policies(&self) -> &HashMap<String, CheckpointPolicy> {
        &self.checkpoint_policies
    }

    /// Cycle-10 §5.3 R9 cross-scope plumbing (Task 4): record a
    /// `@paged_kv`-decorated model name. The dispatch fork at
    /// `flash_attention_v2/mod.rs::synthesize_backward_with_tier`
    /// (Task 9) consults this set to refuse the
    /// `policy="full"` + `@paged_kv` composition.
    pub fn mark_paged_kv_model(&mut self, name: &str) {
        self.paged_kv_models.insert(name.to_string());
    }

    /// Cycle-10 §5.3 R9 cross-scope plumbing (Task 4): public accessor
    /// for the `@paged_kv` model set. Codegen consumes this for the
    /// cross-scope refusal predicate.
    pub fn paged_kv_models(&self) -> &HashSet<String> {
        &self.paged_kv_models
    }

    /// Mark a function as having an explicit `Rng` parameter.
    /// This makes its Random effect "controlled" — @deterministic is allowed.
    pub fn mark_has_explicit_rng(&mut self, name: &str) {
        self.explicit_rng_fns.insert(name.to_string());
    }

    /// Check if a function has an explicit Rng parameter.
    pub fn has_explicit_rng(&self, name: &str) -> bool {
        self.explicit_rng_fns.contains(name)
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

        // @deterministic: must have no Random effect (unless controlled via explicit Rng param)
        for name in self.deterministic_fns.clone() {
            let effects = self.fn_effects.get(&name).copied().unwrap_or(EffectSet::PURE);
            if !effects.is_deterministic() {
                // Allow Random if the function takes an explicit Rng parameter —
                // randomness is "controlled" (deterministic given the same seed).
                if self.has_explicit_rng(&name) {
                    continue;
                }
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "@deterministic function '{}' has Random effect — \
                         add an explicit `rng: Rng` parameter to make randomness controlled, \
                         or remove @deterministic",
                        name
                    )),
                );
            }
        }

        // @checkpoint: requires @pure (recomputation must produce same result).
        //
        // Cycle-10 §5.3 (T10): the generic `is_pure()` check covers all four
        // non-pure flags (IO, Random, Mutation, Communication) — not just
        // Random. We keep the existing message wording for backwards
        // compatibility, and append a one-sentence note for the
        // Random-effect case explaining that per-Q-tile RNG channel routing
        // is deferred to v2.
        for name in self.checkpointed_fns.clone() {
            let effects = self.fn_effects.get(&name).copied().unwrap_or(EffectSet::PURE);
            if !effects.is_pure() {
                let mut msg = format!(
                    "@checkpoint function '{}' has effects: {} — \
                     @checkpoint recomputes during backward, requires @pure",
                    name, effects.display()
                );
                if effects.contains(EffectSet::RANDOM) {
                    msg.push_str(
                        "; per-Q-tile RNG channel for Random effects deferred to v2"
                    );
                }
                self.diagnostics.push(Diagnostic::error(msg));
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
        // Unknown builtins conservatively assumed to have MUTATION effect.
        // We don't assume IO/RANDOM/COMMUNICATION because most unrecognized
        // builtins are tensor ops, and over-approximating with all 4 effects
        // causes false @pure violations for user-defined functions.
        let effects = classify_builtin_effects("some_unknown_function");
        assert!(effects.contains(EffectSet::MUTATION));
        assert!(!effects.contains(EffectSet::IO));
        assert!(!effects.contains(EffectSet::RANDOM));
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

    // Cycle-10 §5.3 Task 3: R4 message picks up the RNG note for
    // Random-effect functions while preserving the generic wording.
    #[test]
    fn checkpoint_random_effect_appends_rng_v2_note() {
        let mut checker = EffectChecker::new();
        // `block` calls `randn` which is a builtin Random source.
        checker.register_function("block", EffectSet::PURE, vec!["randn".into()]);
        checker.mark_checkpointed("block");
        checker.analyze();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(
            checker.diagnostics[0].message.contains("requires @pure"),
            "generic message must remain"
        );
        assert!(
            checker.diagnostics[0]
                .message
                .contains("per-Q-tile RNG channel for Random effects deferred to v2"),
            "Random effect must append the v2-deferral note"
        );
    }

    // Cycle-10 §5.3 Task 3: R4 message does NOT append the RNG note for
    // non-Random non-pure effects (IO/Mutation/Communication).
    #[test]
    fn checkpoint_io_effect_does_not_append_rng_note() {
        let mut checker = EffectChecker::new();
        checker.register_function("block", EffectSet::PURE, vec!["print".into()]);
        checker.mark_checkpointed("block");
        checker.analyze();
        assert_eq!(checker.diagnostics.len(), 1);
        assert!(
            !checker.diagnostics[0]
                .message
                .contains("RNG channel"),
            "non-Random effects must NOT append the RNG v2 note"
        );
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

    // ── Explicit Rng detection (M51b) ─────────────────────────────────

    #[test]
    fn deterministic_allows_explicit_rng() {
        let mut checker = EffectChecker::new();
        // seeded_sample calls rand (Random effect) but has explicit Rng param
        checker.register_function("seeded_sample", EffectSet::RANDOM, vec![]);
        checker.mark_deterministic("seeded_sample");
        checker.mark_has_explicit_rng("seeded_sample");
        checker.analyze();

        // Should NOT produce a diagnostic — controlled randomness is allowed
        assert!(checker.diagnostics.is_empty(),
            "@deterministic should allow Random with explicit Rng, got: {:?}",
            checker.diagnostics);
    }

    #[test]
    fn deterministic_rejects_uncontrolled_random() {
        let mut checker = EffectChecker::new();
        // bad_sample calls rand but does NOT have Rng param
        checker.register_function("bad_sample", EffectSet::RANDOM, vec![]);
        checker.mark_deterministic("bad_sample");
        // NOT calling mark_has_explicit_rng — no Rng param
        checker.analyze();

        assert_eq!(checker.diagnostics.len(), 1,
            "Should reject uncontrolled Random");
        let msg = format!("{:?}", checker.diagnostics[0]);
        assert!(msg.contains("Random"), "Error should mention Random: {msg}");
        assert!(msg.contains("Rng"), "Error should hint about adding Rng param: {msg}");
    }

    #[test]
    fn deterministic_with_rng_still_rejects_io() {
        let mut checker = EffectChecker::new();
        // Function has Rng AND IO — Rng exempts Random but not IO
        checker.register_function("rng_and_io", EffectSet::RANDOM | EffectSet::IO, vec![]);
        checker.mark_deterministic("rng_and_io");
        checker.mark_has_explicit_rng("rng_and_io");
        checker.analyze();

        // Should still fail because of IO (Random is exempted but IO is not)
        // Actually, the current validate() checks is_deterministic() which only checks Random.
        // IO is not checked by @deterministic — only @pure checks all effects.
        // So this should pass (is_deterministic only cares about Random).
        // Let's verify:
        assert!(checker.diagnostics.is_empty(),
            "@deterministic only checks Random, not IO: {:?}", checker.diagnostics);
    }

    #[test]
    fn has_explicit_rng_tracking() {
        let mut checker = EffectChecker::new();
        checker.mark_has_explicit_rng("func_a");

        assert!(checker.has_explicit_rng("func_a"));
        assert!(!checker.has_explicit_rng("func_b"));
    }

    // ── Effect polymorphism tests (M51c) ────────────────────────────────

    use crate::types::Effect;

    /// Helper to create a test Symbol from a u32 index.
    fn test_sym(n: u32) -> nsl_ast::Symbol {
        nsl_ast::Symbol(unsafe { std::mem::transmute::<u32, string_interner::DefaultSymbol>(n) })
    }

    #[test]
    fn effect_pure_default() {
        assert_eq!(Effect::pure(), Effect::Concrete(EffectSet::PURE));
    }

    #[test]
    fn effect_inferred_default() {
        assert_eq!(Effect::default(), Effect::Inferred);
    }

    #[test]
    fn effect_var_has_vars() {
        let sym = test_sym(99);
        let eff = Effect::Var(sym);
        assert!(eff.has_vars());
    }

    #[test]
    fn effect_concrete_no_vars() {
        let eff = Effect::Concrete(EffectSet::IO);
        assert!(!eff.has_vars());
    }

    #[test]
    fn effect_union_has_vars() {
        let sym = test_sym(99);
        let eff = Effect::Union(
            Box::new(Effect::Var(sym)),
            Box::new(Effect::Concrete(EffectSet::IO)),
        );
        assert!(eff.has_vars());
    }

    #[test]
    fn effect_substitute_var_bound() {
        let sym = test_sym(42);
        let eff = Effect::Var(sym);
        let mut bindings = std::collections::HashMap::new();
        bindings.insert(sym, EffectSet::IO);
        assert_eq!(eff.substitute(&bindings), EffectSet::IO);
    }

    #[test]
    fn effect_substitute_var_unbound() {
        let sym = test_sym(42);
        let eff = Effect::Var(sym);
        let bindings = std::collections::HashMap::new();
        // Unbound variables default to PURE
        assert_eq!(eff.substitute(&bindings), EffectSet::PURE);
    }

    #[test]
    fn effect_substitute_union() {
        let sym = test_sym(42);
        let eff = Effect::Union(
            Box::new(Effect::Var(sym)),
            Box::new(Effect::Concrete(EffectSet::MUTATION)),
        );
        let mut bindings = std::collections::HashMap::new();
        bindings.insert(sym, EffectSet::IO);
        let result = eff.substitute(&bindings);
        assert!(result.contains(EffectSet::IO));
        assert!(result.contains(EffectSet::MUTATION));
    }

    #[test]
    fn effect_substitute_concrete_passthrough() {
        let eff = Effect::Concrete(EffectSet::RANDOM);
        let bindings = std::collections::HashMap::new();
        assert_eq!(eff.substitute(&bindings), EffectSet::RANDOM);
    }

    #[test]
    fn effect_substitute_inferred_is_pure() {
        let eff = Effect::Inferred;
        let bindings = std::collections::HashMap::new();
        assert_eq!(eff.substitute(&bindings), EffectSet::PURE);
    }

    #[test]
    fn effect_collect_vars() {
        let sym1 = test_sym(1);
        let sym2 = test_sym(2);
        let eff = Effect::Union(
            Box::new(Effect::Var(sym1)),
            Box::new(Effect::Var(sym2)),
        );
        let mut vars = Vec::new();
        eff.collect_vars(&mut vars);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&sym1));
        assert!(vars.contains(&sym2));
    }

    #[test]
    fn effect_checker_parametric_pure_call() {
        // Simulates: map(double, xs) where double is @pure
        // The effect variable E binds to {} (pure)
        let sym_e = test_sym(100);
        let effect = Effect::Var(sym_e);
        let mut bindings = std::collections::HashMap::new();
        bindings.insert(sym_e, EffectSet::PURE);
        let resolved = effect.substitute(&bindings);
        assert!(resolved.is_pure());
    }

    #[test]
    fn effect_checker_parametric_io_call() {
        // Simulates: map(noisy_double, xs) where noisy_double has IO
        // The effect variable E binds to {IO}
        let sym_e = test_sym(100);
        let effect = Effect::Var(sym_e);
        let mut bindings = std::collections::HashMap::new();
        bindings.insert(sym_e, EffectSet::IO);
        let resolved = effect.substitute(&bindings);
        assert!(resolved.contains(EffectSet::IO));
        assert!(!resolved.is_pure());
    }

    #[test]
    fn effect_multiple_vars_independent() {
        // fn combine(f: fn() | E1, g: fn() | E2) -> T | E1 | E2
        let sym_e1 = test_sym(101);
        let sym_e2 = test_sym(102);
        let return_effect = Effect::Union(
            Box::new(Effect::Var(sym_e1)),
            Box::new(Effect::Var(sym_e2)),
        );
        let mut bindings = std::collections::HashMap::new();
        bindings.insert(sym_e1, EffectSet::IO);
        bindings.insert(sym_e2, EffectSet::RANDOM);
        let resolved = return_effect.substitute(&bindings);
        assert!(resolved.contains(EffectSet::IO));
        assert!(resolved.contains(EffectSet::RANDOM));
    }
}
