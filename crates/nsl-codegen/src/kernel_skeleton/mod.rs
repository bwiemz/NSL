//! Shared PTX kernel boilerplate for FA and WRGA.
//!
//! This module owns only the ~15 lines of PTX that both FA and WRGA
//! genuinely emit identically: the file header, SMEM declaration, thread/
//! lane/warp register dance, and a few param-load line-level helpers.
//! Kernel-specific logic (tiling, MMA choreography, epilogue math, store
//! layout) stays in each caller's own emitter.
//!
//! **Extension discipline:** every helper below has a matching set of
//! pinned PTX snapshots in `tests/snapshots/`.  Adding a new variant to
//! FA or WRGA that the skeleton doesn't cover must also add a new snapshot
//! — the gap is caught at the skeleton layer, not requiring trace-back
//! from a failing FA snapshot.

pub mod header;
pub mod indexing;
pub mod pad;
pub mod params;
pub mod smem;

#[cfg(test)]
mod tests;
