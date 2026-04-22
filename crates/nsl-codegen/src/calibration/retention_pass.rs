//! Retention pass — Task 4 of the calibration forward-pass plan.
//!
//! Provides `build_arena_layout`, a pure function that maps a slice of
//! `DiscoveredProjection`s into an `ArenaLayout` with sequential byte offsets.
//! The layout is used by:
//!
//! 1. Module finalisation: to allocate the `.bss` `__nsl_calib_retention_arena`
//!    global that stores input activations during the calibration forward pass.
//! 2. The splice point in `expr/mod.rs`: to compute the per-projection offset
//!    before emitting the `emit_splice_memcpy` call.

use crate::calibration::retention::ArenaLayout;
use crate::calibration::discovery::DiscoveredProjection;

/// Build a flat `ArenaLayout` from a slice of discovered projections.
///
/// Each entry occupies `batch * seq * in_features * 4` bytes (f32).
/// `in_features` is `weight_shape[1]` (the number of input channels).
/// Offsets are assigned sequentially in the order projections appear in
/// the slice.
///
/// # MVP note
///
/// `batch` and `seq` are passed in directly.  For the MVP (Task 4) the
/// caller hard-codes `batch = 8, seq = 4` (matching the Task 10 fixture).
/// Task 6 will read these from the calibration-data header and pass them
/// through `CompileOptions`.
pub fn build_arena_layout(
    projections: &[DiscoveredProjection],
    batch: u32,
    seq: u32,
) -> ArenaLayout {
    let mut entries = Vec::with_capacity(projections.len());
    let mut offset: u32 = 0;
    for p in projections {
        let in_features = p.weight_shape[1];
        let nbytes = batch * seq * in_features * 4; // f32 = 4 bytes per element
        entries.push((p.projection.clone(), offset, nbytes));
        offset += nbytes;
    }
    ArenaLayout { entries }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::observation::ProjectionRef;

    #[test]
    fn two_projection_arena_sized_on_activations() {
        let ps = vec![
            DiscoveredProjection {
                projection: ProjectionRef("model.up_proj".into()),
                weight_shape: [128, 64],
            },
            DiscoveredProjection {
                projection: ProjectionRef("model.down_proj".into()),
                weight_shape: [64, 128],
            },
        ];
        let layout = build_arena_layout(&ps, 8, 4);
        assert_eq!(layout.entries.len(), 2);
        // entry 0: batch=8, seq=4, in_features=64, f32 → 8*4*64*4 = 8192
        assert_eq!(layout.entries[0].2, 8 * 4 * 64 * 4, "up_proj nbytes wrong");
        // entry 1 offset must equal entry 0's nbytes
        assert_eq!(layout.entries[1].1, 8192, "down_proj offset wrong");
        // entry 1: batch=8, seq=4, in_features=128 → 8*4*128*4 = 16384
        assert_eq!(layout.entries[1].2, 8 * 4 * 128 * 4, "down_proj nbytes wrong");
        // total = 8192 + 16384 = 24576
        assert_eq!(layout.total_bytes(), 24_576);
    }

    #[test]
    fn single_projection_offset_starts_at_zero() {
        let ps = vec![DiscoveredProjection {
            projection: ProjectionRef("m.w".into()),
            weight_shape: [32, 16],
        }];
        let layout = build_arena_layout(&ps, 2, 2);
        assert_eq!(layout.entries[0].1, 0);
        assert_eq!(layout.entries[0].2, 2 * 2 * 16 * 4); // 256
    }

    #[test]
    fn empty_projections_gives_zero_total() {
        let layout = build_arena_layout(&[], 8, 4);
        assert_eq!(layout.entries.len(), 0);
        assert_eq!(layout.total_bytes(), 0);
    }

    // Splice behaviour (retention = Some / None) is verified via the Task 10
    // end-to-end pipeline test (awq_full_pipeline.rs), since the codegen
    // context provides no IR-inspection API for unit tests.
}
