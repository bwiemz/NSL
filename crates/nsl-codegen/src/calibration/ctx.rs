//! Runtime context passed to `CalibrationHook::emit_*` methods.
//!
//! The stub-mode path (`stub_*` methods) is used by unit tests so
//! hooks can be exercised without Cranelift.  The real IR emission for
//! the 2D max-abs reduction is in `binary_codegen::emit_2d_max_abs_loop`
//! and is driven by `ObservePlanEntry` / `FinalizePlanEntry` from Task 5.

use std::collections::BTreeMap;

use crate::calibration::observation::ProjectionRef;
use crate::calibration::retention::RetentionTable;

/// Opaque handle returned by `alloc_running_vec`; hooks store these
/// in their `self` fields to reference buffers across `emit_*` calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferHandle(u32);

/// Descriptor for a BSS global declared by a hook's `emit_init`.
/// Returned by [`CalibCtx::lookup_bss_global`] in test/stub mode.
#[derive(Debug, Clone)]
pub struct BssGlobalEntry {
    pub name: String,
    pub size_bytes: u32,
}

#[derive(Debug)]
pub struct CalibCtx<'a> {
    pub sample_idx: u32,
    pub total_samples: u32,
    retention: &'a RetentionTable,
    running_buffers: BTreeMap<BufferHandle, Vec<f32>>,
    running_names: BTreeMap<String, BufferHandle>,
    next_handle: u32,
    /// Stub-mode per-projection arena data (keyed by projection name).
    /// Populated by `stub_set_arena_buffer`; consumed by
    /// `emit_per_channel_max_abs_update` in stub mode.
    stub_arena_data: BTreeMap<String, Vec<f32>>,
    /// BSS globals declared by hook `emit_init` calls (stub + production).
    /// Keyed by symbol name; value is size in bytes.
    bss_globals: BTreeMap<String, u32>,
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
            stub_arena_data: BTreeMap::new(),
            bss_globals: BTreeMap::new(),
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
            stub_arena_data: BTreeMap::new(),
            bss_globals: BTreeMap::new(),
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

    // ---- emit_observe_batch helpers ------------------------------------------

    /// Stub-mode: write arena activation data for a projection, keyed by
    /// projection name.  In production (Task 9) the arena is a `.bss` global
    /// filled by the splice-point IR; in tests we inject it here so that
    /// `emit_per_channel_max_abs_update` can find it.
    ///
    /// Named `stub_set_arena_buffer` to distinguish from the handle-based
    /// `stub_set_buffer` used for running-max buffers.
    pub fn stub_set_arena_buffer(&mut self, name: &str, values: &[f32]) {
        self.stub_arena_data.insert(name.to_string(), values.to_vec());
    }

    /// Stub-mode: read back the running-max-abs vector for a named projection.
    /// Returns an empty `Vec` when the name is unknown.
    pub fn stub_running_max_abs_named(&self, name: &str) -> Vec<f32> {
        let h = match self.running_names.get(name) {
            Some(&h) => h,
            None => return Vec::new(),
        };
        self.running_buffers.get(&h).cloned().unwrap_or_default()
    }

    // ---- BSS-global declaration (emit_init path) ----------------------------

    /// Declare a BSS global with the given `symbol` name and `size_bytes`.
    ///
    /// On the real Cranelift IR path (Task 12+) this will emit a
    /// zero-initialised data section of the requested size.  In stub / test
    /// mode it records the declaration into an in-memory map so tests can
    /// verify correct size via [`lookup_bss_global`].
    ///
    /// Panics if the same symbol is declared twice (hook-author bug).
    pub fn declare_bss_global(&mut self, symbol: &str, size_bytes: u32) {
        if self.bss_globals.contains_key(symbol) {
            panic!("duplicate BSS global declaration: {symbol}");
        }
        self.bss_globals.insert(symbol.to_string(), size_bytes);
    }

    /// Test helper: look up a BSS global by symbol name.
    ///
    /// Returns `Some(BssGlobalEntry)` when the symbol was previously declared
    /// via [`declare_bss_global`], or `None` if it was never declared.
    pub fn lookup_bss_global(&self, symbol: &str) -> Option<BssGlobalEntry> {
        self.bss_globals.get(symbol).map(|&size_bytes| BssGlobalEntry {
            name: symbol.to_string(),
            size_bytes,
        })
    }

    /// Emit a per-input-channel max-abs reduction over a 2-D slice of the
    /// arena with shape `[rows, channels]` (row-major, f32).
    ///
    /// Parameters:
    /// - `src_name`   — projection name used in stub mode to look up the
    ///                  in-memory arena buffer (populated via `stub_set_arena_buffer`).
    /// - `src_offset` — byte offset into the `.bss` arena global; used only
    ///                  on the real Cranelift IR path (Task 9+).
    /// - `rows`       — first dimension of the activation slice
    ///                  (batch × seq, collapsed).
    /// - `channels`   — in-features (second dimension).
    /// - `running`    — handle to the running-max buffer (length == channels).
    ///
    /// Stub path: iterates the concrete buffer in `stub_arena_data[src_name]`
    /// and updates `running_buffers[running]` element-wise with `max(|v|, cur)`.
    ///
    /// Real IR path: emits a Cranelift outer loop over `rows`, calling
    /// `emit_running_max_abs_f32` for each row's sub-slice.
    pub fn emit_per_channel_max_abs_update(
        &mut self,
        src_name: &str,
        _src_offset: u32,
        rows: u32,
        channels: u32,
        running: BufferHandle,
    ) {
        if channels == 0 {
            return;
        }

        // ── Stub path (unit tests only) ─────────────────────────────────────
        // The real IR path is in binary_codegen::emit_2d_max_abs_loop.
        {
            let src = self.stub_arena_data.get(src_name).cloned().unwrap_or_default();
            let buf = self
                .running_buffers
                .get_mut(&running)
                .expect("emit_per_channel_max_abs_update: unknown BufferHandle");

            if buf.len() < channels as usize {
                buf.resize(channels as usize, 0.0);
            }

            for r in 0..rows as usize {
                for c in 0..channels as usize {
                    let idx = r * channels as usize + c;
                    if idx >= src.len() {
                        break;
                    }
                    let a = src[idx].abs();
                    if a > buf[c] {
                        buf[c] = a;
                    }
                }
            }
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

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types as cl_types, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;

/// Emit Cranelift IR for: `for i in 0..len { buf[i] = max(buf[i], |src[i]|) }`.
///
/// - `buf_ptr` (i64): pointer to a running-max f32 array (read + written in place).
/// - `src_ptr` (i64): pointer to a sample f32 array (read only).
/// - `len_val` (i32): number of f32 elements to process.
///
/// This is the core reduction shared by every AWQ-style calibration hook
/// (AWQ activation scales, FP8 amax tracking, GPTQ activation-stats inputs).
/// Task 11 will wire this into the per-step loop of the production calibration
/// binary; this helper is self-contained IR emission.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn emit_running_max_abs_f32(
    fb: &mut FunctionBuilder,
    buf_ptr: Value,
    src_ptr: Value,
    len_val: Value,
) {
    // Loop via block-param induction variable (no shared Variable slot, so
    // this helper is safe to call from any `FunctionBuilder` context).
    let header = fb.create_block();
    let body = fb.create_block();
    let exit = fb.create_block();
    fb.append_block_param(header, cl_types::I32);

    let zero_i32 = fb.ins().iconst(cl_types::I32, 0);
    fb.ins().jump(header, &[zero_i32]);

    // header: if i < len goto body(i) else goto exit
    fb.switch_to_block(header);
    let i_cur = fb.block_params(header)[0];
    let cond = fb.ins().icmp(IntCC::UnsignedLessThan, i_cur, len_val);
    fb.ins().brif(cond, body, &[], exit, &[]);

    // body
    fb.switch_to_block(body);
    fb.seal_block(body);
    let four = fb.ins().iconst(cl_types::I32, 4);
    let off32 = fb.ins().imul(i_cur, four);
    let off64 = fb.ins().sextend(cl_types::I64, off32);
    let src_addr = fb.ins().iadd(src_ptr, off64);
    let buf_addr = fb.ins().iadd(buf_ptr, off64);
    let src_val = fb.ins().load(cl_types::F32, MemFlags::new(), src_addr, 0);
    let abs_val = fb.ins().fabs(src_val);
    let buf_val = fb.ins().load(cl_types::F32, MemFlags::new(), buf_addr, 0);
    let new_val = fb.ins().fmax(buf_val, abs_val);
    fb.ins().store(MemFlags::new(), new_val, buf_addr, 0);
    let one = fb.ins().iconst(cl_types::I32, 1);
    let i_next = fb.ins().iadd(i_cur, one);
    fb.ins().jump(header, &[i_next]);

    fb.seal_block(header);
    fb.switch_to_block(exit);
    fb.seal_block(exit);
}

#[cfg(test)]
mod ir_tests {
    use super::emit_running_max_abs_f32;
    use cranelift_codegen::ir::{types as cl_types, AbiParam, InstBuilder, Function, UserFuncName, Signature};
    use cranelift_codegen::isa::CallConv;
    use cranelift_codegen::settings;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    #[test]
    fn running_max_abs_ir_contains_abs_max_and_store() {
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(cl_types::I64)); // buf_ptr
        sig.params.push(AbiParam::new(cl_types::I64)); // src_ptr
        sig.params.push(AbiParam::new(cl_types::I32)); // len
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut func, &mut fb_ctx);

        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);
        let buf_ptr = fb.block_params(entry)[0];
        let src_ptr = fb.block_params(entry)[1];
        let len_val = fb.block_params(entry)[2];

        emit_running_max_abs_f32(&mut fb, buf_ptr, src_ptr, len_val);
        fb.ins().return_(&[]);
        fb.finalize();

        // Verify the IR verifies.
        let flags = settings::Flags::new(settings::builder());
        cranelift_codegen::verifier::verify_function(&func, &flags)
            .expect("emitted IR should verify");

        let ir_text = format!("{}", func.display());
        assert!(
            ir_text.contains("fabs") || ir_text.contains("band"),
            "IR should take abs of src (fabs or bitmask band): {ir_text}"
        );
        assert!(
            ir_text.contains("fmax") || (ir_text.contains("fcmp") && ir_text.contains("select")),
            "IR should compute max via fmax or fcmp+select: {ir_text}"
        );
        assert!(
            ir_text.contains("store"),
            "IR should store result back to buf: {ir_text}"
        );
    }
}
