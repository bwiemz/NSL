//! Emits the `calibration_main()` entrypoint and links it against the
//! user's compiled model.  Task 14 fleshes this out; for now it's a
//! placeholder that unblocks Task 13's seam.

use crate::calibration::registry::HookRegistry;
use crate::calibration::{HarnessConfig, HarnessError, HarnessOutput};

/// Placeholder for the real subprocess entry point.  Task 14 replaces
/// the body with codegen + subprocess spawn + sidecar read.  Until
/// then production callers (tests excluded) should not reach this.
pub fn real_subprocess_entry(
    _cfg: &HarnessConfig,
    _registry: &HookRegistry,
) -> Result<HarnessOutput, HarnessError> {
    Err(HarnessError::Infrastructure {
        reason: "real subprocess entry not yet implemented (Task 14)".into(),
    })
}
