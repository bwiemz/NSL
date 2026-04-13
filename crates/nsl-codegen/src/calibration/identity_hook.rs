//! Test-only calibration hook that writes a caller-supplied fixed
//! byte payload to the sidecar.  Used by the harness's end-to-end
//! tests (Task 13/15) to prove the full flow works without a real
//! consumer.  Consumers like AWQ / WGGO Phase 2 land in follow-up
//! plans.

use crate::calibration::ctx::CalibCtx;
use crate::calibration::hooks::{CalibrationHook, CalibrationResult};
use crate::calibration::observation::ObservationSet;

pub struct IdentityHook {
    payload: Vec<u8>,
}

impl IdentityHook {
    pub fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }
}

impl CalibrationHook for IdentityHook {
    fn id(&self) -> &'static str {
        "identity"
    }

    fn requires(&self) -> ObservationSet {
        ObservationSet::Empty
    }

    fn emit_init(&self, _ctx: &mut CalibCtx) {}
    fn emit_per_step(&self, _ctx: &mut CalibCtx) {}

    fn emit_finalize(&self, _ctx: &mut CalibCtx) -> CalibrationResult {
        if self.payload.is_empty() {
            CalibrationResult::Degenerate {
                reason: "identity hook got empty payload (would write nothing useful)".into(),
            }
        } else {
            CalibrationResult::Ok(self.payload.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::ctx::CalibCtx;
    use crate::calibration::hooks::{CalibrationHook, CalibrationResult};
    use crate::calibration::observation::ObservationSet;

    #[test]
    fn identity_hook_writes_fixed_payload() {
        let h = IdentityHook::new(b"hello-calib".to_vec());
        assert_eq!(h.id(), "identity");
        assert!(matches!(h.requires(), ObservationSet::Empty));
        let mut ctx = CalibCtx::stub_for_tests();
        h.emit_init(&mut ctx);
        h.emit_per_step(&mut ctx);
        match h.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(b) => assert_eq!(b, b"hello-calib"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn identity_hook_returns_degenerate_on_empty_payload() {
        let h = IdentityHook::new(Vec::new());
        let mut ctx = CalibCtx::stub_for_tests();
        match h.emit_finalize(&mut ctx) {
            CalibrationResult::Degenerate { reason } => {
                assert!(reason.contains("empty"));
            }
            other => panic!("expected Degenerate, got {other:?}"),
        }
    }
}
