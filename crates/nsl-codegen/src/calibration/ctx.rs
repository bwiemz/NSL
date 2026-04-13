//! Runtime context passed to `CalibrationHook::emit_*` methods.
//!
//! In production (Task 9), `CalibCtx` wraps a Cranelift
//! `FunctionBuilder` + retention table + running-buffer map, and the
//! API methods emit IR directly.  In unit-test mode, the same API is
//! backed by concrete Rust state so hooks can be exercised without
//! Cranelift.

use std::collections::BTreeMap;

use crate::calibration::observation::ProjectionRef;
use crate::calibration::retention::RetentionTable;

/// Opaque handle returned by `alloc_running_vec`; hooks store these
/// in their `self` fields to reference buffers across `emit_*` calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferHandle(u32);

#[derive(Debug)]
pub struct CalibCtx<'a> {
    pub sample_idx: u32,
    pub total_samples: u32,
    retention: &'a RetentionTable,
    running_buffers: BTreeMap<BufferHandle, Vec<f32>>,
    running_names: BTreeMap<String, BufferHandle>,
    next_handle: u32,
    // Task 9 adds: fbuilder: &'a mut cranelift::FunctionBuilder,
}

impl<'a> CalibCtx<'a> {
    /// Back-door constructor for unit tests.  Production code uses
    /// the codegen-driven constructor in Task 9.
    pub fn for_tests(retention: &'a RetentionTable) -> Self {
        Self {
            sample_idx: 0,
            total_samples: 0,
            retention,
            running_buffers: BTreeMap::new(),
            running_names: BTreeMap::new(),
            next_handle: 0,
        }
    }

    /// Construct a no-op context for unit tests that don't need a
    /// retention table.  Preserved for backward compatibility with
    /// pre-existing harness MVP tests.
    pub fn stub_for_tests() -> Self {
        static EMPTY: std::sync::OnceLock<RetentionTable> = std::sync::OnceLock::new();
        let retention = EMPTY.get_or_init(RetentionTable::new);
        Self {
            sample_idx: 0,
            total_samples: 0,
            retention,
            running_buffers: BTreeMap::new(),
            running_names: BTreeMap::new(),
            next_handle: 0,
        }
    }

    /// Shape of the retained tensor for a projection — comes from the
    /// `ObservationPlan`.  Panics when the projection wasn't
    /// registered, which is a hook-author bug (hook claimed to
    /// observe it in `requires()` but then didn't).
    pub fn projection_input_shape(&self, r: &ProjectionRef) -> &[u64] {
        &self
            .retention
            .get(r)
            .unwrap_or_else(|| panic!("projection {} not in retention table (missing from requires()?)", r.0))
            .shape
            .dims
    }

    /// Allocate a named running-reduction buffer.  The hook stores
    /// the returned handle and uses it in `emit_per_step` +
    /// `emit_finalize`.
    pub fn alloc_running_vec(&mut self, name: &str, len: usize) -> BufferHandle {
        if self.running_names.contains_key(name) {
            panic!("duplicate running buffer name: {name}");
        }
        let h = BufferHandle(self.next_handle);
        self.next_handle += 1;
        self.running_names.insert(name.to_string(), h);
        self.running_buffers.insert(h, vec![0.0; len]);
        h
    }

    /// Read back a running buffer's final values during emit_finalize.
    pub fn finalize_read(&self, h: BufferHandle) -> Vec<f32> {
        self.running_buffers.get(&h).expect("unknown BufferHandle").clone()
    }

    // ---- stub-mode helpers (test-only) ------------------------------
    //
    // Task 9 replaces these with Cranelift IR emission routines that
    // generate instructions inside the calibration binary's per-step
    // loop.  The *public* trait surface (`running_max_abs`,
    // `read_projection_input`) stays the same; only the backing
    // implementation changes.

    /// Test-only: inject a buffer's final state, bypassing the
    /// reduction loop.
    pub fn stub_set_buffer(&mut self, h: BufferHandle, values: Vec<f32>) {
        let buf = self.running_buffers.get_mut(&h).expect("unknown BufferHandle");
        assert_eq!(buf.len(), values.len());
        *buf = values;
    }

    /// Test-only: apply the running-max-abs reduction over a concrete
    /// slice.  Simulates what `running_max_abs` will do at IR level.
    pub fn stub_running_max_abs(&mut self, h: BufferHandle, src: &[f32]) {
        let buf = self.running_buffers.get_mut(&h).expect("unknown BufferHandle");
        assert_eq!(buf.len(), src.len(), "src length mismatch for running_max_abs");
        for (b, v) in buf.iter_mut().zip(src.iter()) {
            *b = b.max(v.abs());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::observation::ProjectionRef;
    use crate::calibration::retention::{RetentionTable, TensorShape};

    fn table_with(p: &str, shape: Vec<u64>) -> RetentionTable {
        let mut t = RetentionTable::new();
        t.register(ProjectionRef::new(p), TensorShape::new(shape), 4);
        t
    }

    #[test]
    fn projection_input_shape_returns_registered_shape() {
        let table = table_with("p", vec![2, 8, 16]);
        let ctx = CalibCtx::for_tests(&table);
        assert_eq!(ctx.projection_input_shape(&ProjectionRef::new("p")), &[2u64, 8, 16][..]);
    }

    #[test]
    fn alloc_running_vec_returns_unique_handles() {
        let table = RetentionTable::new();
        let mut ctx = CalibCtx::for_tests(&table);
        let h1 = ctx.alloc_running_vec("a", 32);
        let h2 = ctx.alloc_running_vec("b", 64);
        assert_ne!(h1, h2);
    }

    #[test]
    fn alloc_running_vec_rejects_duplicate_name() {
        let table = RetentionTable::new();
        let mut ctx = CalibCtx::for_tests(&table);
        ctx.alloc_running_vec("same", 32);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut c2 = ctx;
            c2.alloc_running_vec("same", 32)
        }));
        assert!(r.is_err());
    }

    #[test]
    fn finalize_read_returns_running_max_values_in_stub_mode() {
        let table = RetentionTable::new();
        let mut ctx = CalibCtx::for_tests(&table);
        let h = ctx.alloc_running_vec("buf", 4);
        ctx.stub_set_buffer(h, vec![1.0, -2.0, 3.5, 0.0]);
        let out = ctx.finalize_read(h);
        assert_eq!(out, vec![1.0, -2.0, 3.5, 0.0]);
    }

    #[test]
    fn running_max_abs_updates_buffer_in_stub_mode() {
        let table = RetentionTable::new();
        let mut ctx = CalibCtx::for_tests(&table);
        let h = ctx.alloc_running_vec("buf", 3);
        ctx.stub_running_max_abs(h, &[1.0, -5.0, 2.0]);
        ctx.stub_running_max_abs(h, &[-3.0, 4.0, -1.5]);
        assert_eq!(ctx.finalize_read(h), vec![3.0, 5.0, 2.0]);
    }
}
