//! Calibration-hook registry.  Explicit (not `inventory`-based)
//! because the compiler already enumerates passes via a short list of
//! imports — we stay with that idiom.  The driver builds a registry
//! at the start of calibration, `register()`s every enabled hook, and
//! iterates in sorted-id order for determinism.

use crate::calibration::hooks::CalibrationHook;

/// Collection of hooks participating in the current calibration run.
pub struct HookRegistry {
    hooks: Vec<Box<dyn CalibrationHook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Add a hook.  Panics on duplicate `id()` — two hooks writing the
    /// same sidecar key would silently clobber each other.
    pub fn register(&mut self, hook: Box<dyn CalibrationHook>) {
        let id = hook.id();
        if self.hooks.iter().any(|h| h.id() == id) {
            panic!("duplicate CalibrationHook id: {id}");
        }
        self.hooks.push(hook);
        self.hooks.sort_by(|a, b| a.id().cmp(b.id()));
    }

    pub fn enabled_ids_sorted(&self) -> Vec<String> {
        self.hooks.iter().map(|h| h.id().to_string()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn CalibrationHook> {
        self.hooks.iter().map(|h| h.as_ref())
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::ctx::CalibCtx;
    use crate::calibration::hooks::{CalibrationHook, CalibrationResult};
    use crate::calibration::observation::ObservationSet;

    struct HookA;
    impl CalibrationHook for HookA {
        fn id(&self) -> &'static str { "a" }
        fn requires(&self) -> ObservationSet { ObservationSet::Empty }
        fn emit_init(&self, _: &mut CalibCtx) {}
        fn emit_per_step(&self, _: &mut CalibCtx) {}
        fn emit_finalize(&self, _: &mut CalibCtx) -> CalibrationResult {
            CalibrationResult::Ok(vec![])
        }
    }
    struct HookB;
    impl CalibrationHook for HookB {
        fn id(&self) -> &'static str { "b" }
        fn requires(&self) -> ObservationSet { ObservationSet::Empty }
        fn emit_init(&self, _: &mut CalibCtx) {}
        fn emit_per_step(&self, _: &mut CalibCtx) {}
        fn emit_finalize(&self, _: &mut CalibCtx) -> CalibrationResult {
            CalibrationResult::Ok(vec![])
        }
    }

    #[test]
    fn registry_collects_and_sorts_ids() {
        let mut r = HookRegistry::new();
        r.register(Box::new(HookB));
        r.register(Box::new(HookA));
        let ids = r.enabled_ids_sorted();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn registry_rejects_duplicate_ids() {
        let mut r = HookRegistry::new();
        r.register(Box::new(HookA));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            r.register(Box::new(HookA))
        }));
        assert!(result.is_err(), "duplicate register should panic");
    }

    #[test]
    fn registry_iterates_by_id_order() {
        let mut r = HookRegistry::new();
        r.register(Box::new(HookB));
        r.register(Box::new(HookA));
        let order: Vec<&str> = r.iter().map(|h| h.id()).collect();
        assert_eq!(order, vec!["a", "b"]);
    }
}
