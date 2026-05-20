//! Per-projection scratch buffer registry for calibration-time
//! activation retention.  See design §6.
//!
//! The registry is populated during calibration codegen: for every
//! `ProjectionRef` in `ObservationPlan::linear_input_activations`,
//! an entry with shape info and arena byte-offset gets recorded.
//! Hooks read from this registry (via the `CalibCtx` wrapper in
//! Task 6) during `emit_per_step`.

use std::collections::BTreeMap;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types as cl_types, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;

use crate::calibration::observation::ProjectionRef;

/// Emit Cranelift IR that copies `nbytes` from `src_ptr` into
/// `arena_ptr + offset`.  Emits a byte-at-a-time loop — correct for
/// any size, though slower than a `memcpy` libcall for large copies.
/// Task 11 may add a libcall fast path once the Module is available
/// at the splice point.
///
/// Zero-byte copies are a no-op (emits nothing).
pub fn emit_splice_memcpy(
    fb: &mut FunctionBuilder,
    arena_ptr: Value,
    offset: u32,
    src_ptr: Value,
    nbytes: u64,
) {
    if nbytes == 0 {
        return;
    }
    let off_val = fb.ins().iconst(cl_types::I64, offset as i64);
    let dst = fb.ins().iadd(arena_ptr, off_val);

    let header = fb.create_block();
    let body = fb.create_block();
    let tail = fb.create_block();
    fb.append_block_param(header, cl_types::I64);

    let zero = fb.ins().iconst(cl_types::I64, 0);
    fb.ins().jump(header, &[zero]);

    // header: if i < nbytes goto body else goto tail
    fb.switch_to_block(header);
    let i = fb.block_params(header)[0];
    let limit = fb.ins().iconst(cl_types::I64, nbytes as i64);
    let cmp = fb.ins().icmp(IntCC::UnsignedLessThan, i, limit);
    fb.ins().brif(cmp, body, &[], tail, &[]);

    // body: *(dst + i) = *(src + i); i += 1; goto header
    fb.switch_to_block(body);
    fb.seal_block(body);
    let src_i = fb.ins().iadd(src_ptr, i);
    let dst_i = fb.ins().iadd(dst, i);
    let b = fb.ins().load(cl_types::I8, MemFlags::new(), src_i, 0);
    fb.ins().store(MemFlags::new(), b, dst_i, 0);
    let one = fb.ins().iconst(cl_types::I64, 1);
    let i2 = fb.ins().iadd(i, one);
    fb.ins().jump(header, &[i2]);

    fb.seal_block(header);
    fb.switch_to_block(tail);
    fb.seal_block(tail);
}

/// Flat memory layout of the per-projection scratch-buffer arena baked
/// into the calibration binary.  Built once during codegen; handed to
/// `emit_observe_batch` so hooks know where each projection's activations
/// land at calibration runtime.
#[derive(Debug, Default, Clone)]
pub struct ArenaLayout {
    pub entries: Vec<(
        ProjectionRef,
        u32, /* byte offset */
        u32, /* nbytes */
    )>,
}

impl ArenaLayout {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn total_bytes(&self) -> u32 {
        self.entries.last().map(|(_, off, n)| off + n).unwrap_or(0)
    }
}

/// Per-projection layout in `__nsl_calib_grad_arena`. Spec §4.3.
///
/// **Overflow contract:** entries' `byte_size` and `total_bytes` use `saturating_*`
/// arithmetic on `u32`. Pathological shapes that overflow will silently alias —
/// debug builds `debug_assert!` to surface this, but release builds saturate.
/// In practice, no production WGGO target approaches `u32::MAX` (4 GiB per matrix).
#[derive(Debug, Clone, PartialEq)]
pub struct GradArenaLayout {
    /// (projection, byte_offset, byte_size). One entry per registered
    /// target's W_q/W_k/W_v/W_o, in target × (q,k,v,o) order.
    pub entries: Vec<(ProjectionRef, u32, u32)>,
    /// Total .bss size; `__nsl_calib_grad_arena` is declared at this size.
    pub total_bytes: u32,
}

pub fn build_grad_arena_layout(
    targets: &[crate::calibration::discovery::WggoGradTarget],
) -> GradArenaLayout {
    let mut entries = Vec::with_capacity(targets.len() * 4);
    let mut offset: u32 = 0;
    for target in targets {
        for (proj, shape) in [
            (&target.w_q, &target.w_q_shape),
            (&target.w_k, &target.w_k_shape),
            (&target.w_v, &target.w_v_shape),
            (&target.w_o, &target.w_o_shape),
        ] {
            let byte_size = shape[0].saturating_mul(shape[1]).saturating_mul(4); // f32 = 4 bytes
            debug_assert!(
                byte_size < u32::MAX,
                "GradArenaLayout: byte_size for projection {:?} ({}x{}x4) saturated to u32::MAX — \
                 a real 4 GiB tensor would alias subsequent entries; spec §4.3 assumes this never happens.",
                proj.0, shape[0], shape[1]
            );
            entries.push((proj.clone(), offset, byte_size));
            offset = offset.saturating_add(byte_size);
        }
    }
    GradArenaLayout {
        entries,
        total_bytes: offset,
    }
}

/// Shape of the tensor feeding into a projection.  `[batch, seq,
/// in_channels]` for the common transformer case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<u64>,
}

impl TensorShape {
    pub fn new(dims: Vec<u64>) -> Self {
        Self { dims }
    }
    pub fn total_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn in_channels(&self) -> u64 {
        *self.dims.last().unwrap_or(&1)
    }
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
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn total_arena_bytes(&self) -> u64 {
        self.next_offset
    }

    pub fn iter(&self) -> impl Iterator<Item = &RetainedProjection> {
        self.entries.values()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod grad_arena_tests {
    use super::*;
    use crate::calibration::discovery::WggoGradTarget;
    use crate::calibration::observation::ProjectionRef;

    #[test]
    fn build_grad_arena_layout_assigns_per_projection_offsets() {
        let targets = vec![WggoGradTarget {
            layer_key: "gpt.blocks.0.attn".into(),
            class_name: "Attention".into(),
            head_dim: 128,
            w_q: ProjectionRef("gpt.blocks.0.attn.q_proj".into()),
            w_k: ProjectionRef("gpt.blocks.0.attn.k_proj".into()),
            w_v: ProjectionRef("gpt.blocks.0.attn.v_proj".into()),
            w_o: ProjectionRef("gpt.blocks.0.attn.o_proj".into()),
            w_q_shape: [4096, 4096],
            w_k_shape: [4096, 4096],
            w_v_shape: [4096, 4096],
            w_o_shape: [4096, 4096],
            w_q_index: 0,
            w_k_index: 1,
            w_v_index: 2,
            w_o_index: 3,
        }];
        let layout = build_grad_arena_layout(&targets);
        // 4 projections × (4096 × 4096 × 4 bytes) = 4 × 64 MiB = 256 MiB
        assert_eq!(layout.entries.len(), 4);
        assert_eq!(layout.entries[0].0 .0, "gpt.blocks.0.attn.q_proj");
        assert_eq!(layout.entries[0].1, 0); // first offset is zero
        assert_eq!(layout.entries[0].2, 4096 * 4096 * 4);
        // Subsequent offsets are the cumulative sum
        assert_eq!(layout.entries[1].1, 4096 * 4096 * 4);
        assert_eq!(layout.total_bytes, 4 * 4096 * 4096 * 4);
    }

    #[test]
    fn build_grad_arena_layout_handles_multiple_layers() {
        let targets: Vec<_> = (0..3)
            .map(|i| WggoGradTarget {
                layer_key: format!("gpt.blocks.{i}.attn"),
                class_name: "Attention".into(),
                head_dim: 64,
                w_q: ProjectionRef(format!("gpt.blocks.{i}.attn.q_proj")),
                w_k: ProjectionRef(format!("gpt.blocks.{i}.attn.k_proj")),
                w_v: ProjectionRef(format!("gpt.blocks.{i}.attn.v_proj")),
                w_o: ProjectionRef(format!("gpt.blocks.{i}.attn.o_proj")),
                w_q_shape: [128, 128],
                w_k_shape: [128, 128],
                w_v_shape: [128, 128],
                w_o_shape: [128, 128],
                w_q_index: 0,
                w_k_index: 1,
                w_v_index: 2,
                w_o_index: 3,
            })
            .collect();
        let layout = build_grad_arena_layout(&targets);
        assert_eq!(layout.entries.len(), 12); // 3 layers × 4 projections
        assert_eq!(layout.total_bytes, 12 * 128 * 128 * 4);

        // Verify the q,k,v,o order within layer 0
        assert_eq!(layout.entries[0].0 .0, "gpt.blocks.0.attn.q_proj");
        assert_eq!(layout.entries[1].0 .0, "gpt.blocks.0.attn.k_proj");
        assert_eq!(layout.entries[2].0 .0, "gpt.blocks.0.attn.v_proj");
        assert_eq!(layout.entries[3].0 .0, "gpt.blocks.0.attn.o_proj");

        // First entry of layer 1 (after layer 0's 4 q/k/v/o entries)
        assert_eq!(layout.entries[4].0 .0, "gpt.blocks.1.attn.q_proj");
        assert_eq!(layout.entries[4].1, 4 * 128 * 128 * 4);
    }
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

    #[test]
    fn splice_memcpy_ir_copies_bytes_from_src_to_arena_offset() {
        use cranelift_codegen::ir::{
            types as cl_types, AbiParam, Function, InstBuilder, Signature, UserFuncName,
        };
        use cranelift_codegen::isa::CallConv;
        use cranelift_codegen::settings;
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

        let flags = settings::Flags::new(settings::builder());

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(cl_types::I64));
        sig.params.push(AbiParam::new(cl_types::I64));

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut func, &mut fb_ctx);
        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);
        let arena = fb.block_params(entry)[0];
        let src = fb.block_params(entry)[1];

        emit_splice_memcpy(&mut fb, arena, 128, src, 64);
        fb.ins().return_(&[]);
        fb.finalize();

        cranelift_codegen::verifier::verify_function(&func, &flags).expect("IR should verify");

        let ir = format!("{}", func.display());
        assert!(
            ir.contains("load") && ir.contains("store"),
            "IR should load from src + store to arena: {ir}"
        );
        assert!(
            ir.contains("128") || ir.contains("0x80"),
            "IR should reference the arena offset: {ir}"
        );
    }

    #[test]
    fn splice_memcpy_zero_bytes_is_safe() {
        use cranelift_codegen::ir::{
            types as cl_types, AbiParam, Function, InstBuilder, Signature, UserFuncName,
        };
        use cranelift_codegen::isa::CallConv;
        use cranelift_codegen::settings;
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

        let flags = settings::Flags::new(settings::builder());

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(cl_types::I64));
        sig.params.push(AbiParam::new(cl_types::I64));

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut func, &mut fb_ctx);
        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);
        let arena = fb.block_params(entry)[0];
        let src = fb.block_params(entry)[1];

        emit_splice_memcpy(&mut fb, arena, 0, src, 0);
        fb.ins().return_(&[]);
        fb.finalize();

        cranelift_codegen::verifier::verify_function(&func, &flags)
            .expect("zero-byte IR should verify");
    }
}
