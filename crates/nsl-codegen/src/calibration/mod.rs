//! Compile-time calibration harness — see
//! `docs/superpowers/specs/2026-04-13-calibration-harness-design.md`.

pub mod cache;
pub mod ctx;
pub mod data_loader;
pub mod hooks;
pub mod observation;
pub mod sidecar;

pub use ctx::CalibCtx;
pub use hooks::{CalibrationHook, CalibrationResult};
pub use observation::{LayerRef, ObservationPlan, ObservationSet, ParamRef};
