//! CPDT Part III — MoE dead-expert pruning (v1).
//!
//! Executes the dead-expert decision as a real weight-drop transform: slice the
//! router columns + drop the expert blocks for low-affinity experts, driven by
//! an `index_remap` single source of truth. A non-WGGO compile pass makes it
//! reachable with `--cpdt --weights` (no `--wggo`), sidestepping Part II's
//! WGGO/source-AD activation blocker. See
//! `docs/superpowers/specs/2026-05-27-cpdt-moe-dead-expert-pruning-v1-design.md`.
