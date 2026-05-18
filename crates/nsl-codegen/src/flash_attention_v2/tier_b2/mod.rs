//! CSHA Tier B.2 — MMA-throughput backward kernel emitter family.
//!
//! Phase 1: scaffold only. The kernel emitters land in Phase 2.
//!
//! Spec: docs/superpowers/specs/2026-05-18-csha-tier-b2-backward-design.md

pub mod dispatch;

pub use dispatch::BackwardTier;
