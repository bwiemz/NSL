//! Per-projection scratch buffer registry for calibration-time
//! activation retention.  See design §6.
//!
//! The registry is populated during calibration codegen: for every
//! `ProjectionRef` in `ObservationPlan::linear_input_activations`,
//! an entry with shape info and arena byte-offset gets recorded.
//! Hooks read from this registry (via the `CalibCtx` wrapper in
//! Task 6) during `emit_per_step`.

use std::collections::BTreeMap;

use crate::calibration::observation::ProjectionRef;

/// Shape of the tensor feeding into a projection.  `[batch, seq,
/// in_channels]` for the common transformer case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<u64>,
}

impl TensorShape {
    pub fn new(dims: Vec<u64>) -> Self { Self { dims } }
    pub fn total_elements(&self) -> u64 { self.dims.iter().product() }
    pub fn in_channels(&self) -> u64 { *self.dims.last().unwrap_or(&1) }
}

/// One retention-table entry — the scratch buffer allocated for a
/// projection's input activation.  The buffer is sized at codegen
/// time and reused across all calibration samples.
#[derive(Debug, Clone)]
pub struct RetainedProjection {
    pub projection: ProjectionRef,
    pub shape: TensorShape,
    /// Offset into the calibration binary's static scratch-buffer
    /// arena.  Real IR pointer lands here at codegen time; tests use
    /// a stub offset.
    pub arena_offset_bytes: u64,
    /// Element size in bytes (typically 4 for f32, 2 for f16/bf16).
    pub elem_bytes: u8,
}

#[derive(Debug, Clone, Default)]
pub struct RetentionTable {
    entries: BTreeMap<ProjectionRef, RetainedProjection>,
    next_offset: u64,
}

impl RetentionTable {
    pub fn new() -> Self { Self::default() }

    /// Register a projection for retention.  Returns the
    /// `arena_offset_bytes` that the splice-point codegen will use as
    /// the memcpy destination.  Panics on duplicate registration
    /// (the ObservationPlan's BTreeSet should have deduplicated).
    pub fn register(
        &mut self,
        projection: ProjectionRef,
        shape: TensorShape,
        elem_bytes: u8,
    ) -> u64 {
        if self.entries.contains_key(&projection) {
            panic!("duplicate projection in retention table: {}", projection.0);
        }
        let offset = self.next_offset;
        let bytes = shape.total_elements() * elem_bytes as u64;
        self.next_offset += bytes;
        self.entries.insert(
            projection.clone(),
            RetainedProjection {
                projection,
                shape,
                arena_offset_bytes: offset,
                elem_bytes,
            },
        );
        offset
    }

    pub fn get(&self, projection: &ProjectionRef) -> Option<&RetainedProjection> {
        self.entries.get(projection)
    }

    pub fn total_arena_bytes(&self) -> u64 { self.next_offset }

    pub fn iter(&self) -> impl Iterator<Item = &RetainedProjection> {
        self.entries.values()
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_computes_offsets_sequentially() {
        let mut t = RetentionTable::new();
        let o1 = t.register(
            ProjectionRef::new("blocks.0.attn.wq"),
            TensorShape::new(vec![1, 2048, 4096]),
            2,
        );
        let o2 = t.register(
            ProjectionRef::new("blocks.0.attn.wk"),
            TensorShape::new(vec![1, 2048, 1024]),
            2,
        );
        assert_eq!(o1, 0);
        // 1 * 2048 * 4096 * 2 = 16777216
        assert_eq!(o2, 16_777_216);
        assert_eq!(t.total_arena_bytes(), 16_777_216 + 1 * 2048 * 1024 * 2);
    }

    #[test]
    fn register_rejects_duplicates() {
        let mut t = RetentionTable::new();
        t.register(
            ProjectionRef::new("blocks.0.attn.wq"),
            TensorShape::new(vec![1, 512, 4096]),
            2,
        );
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            t.register(
                ProjectionRef::new("blocks.0.attn.wq"),
                TensorShape::new(vec![1, 512, 4096]),
                2,
            )
        }));
        assert!(r.is_err());
    }

    #[test]
    fn get_returns_shape_and_offset() {
        let mut t = RetentionTable::new();
        t.register(
            ProjectionRef::new("p"),
            TensorShape::new(vec![4, 128, 512]),
            4,
        );
        let entry = t.get(&ProjectionRef::new("p")).unwrap();
        assert_eq!(entry.shape.dims, vec![4, 128, 512]);
        assert_eq!(entry.elem_bytes, 4);
        assert_eq!(entry.arena_offset_bytes, 0);
    }

    #[test]
    fn iter_visits_all_registrations_sorted() {
        let mut t = RetentionTable::new();
        t.register(ProjectionRef::new("b"), TensorShape::new(vec![1]), 4);
        t.register(ProjectionRef::new("a"), TensorShape::new(vec![1]), 4);
        let names: Vec<String> = t.iter().map(|e| e.projection.0.clone()).collect();
        // BTreeMap sorts by key
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }
}
