//! The wire constants both halves of the ABI must agree on.
//!
//! `nsl-codegen` emits code that reads tensor headers by byte offset, tags
//! dtypes by number, packs parameter-plan bits, names allocation surfaces
//! and bakes the ABI version into generated C headers; `nsl-runtime`
//! implements the other side of each. Until roadmap A3 these numbers were
//! declared in the runtime and *imported* by the compiler — which made a
//! whole compiler build depend on the runtime's dependency tree (tokenizers,
//! safetensors, hf-hub, …) for the sake of a few dozen integers. They now
//! live here, in the crate that exists to be the ABI's source of truth, and
//! BOTH crates read them from it: the runtime re-exports every name at its
//! historical path (so `nsl_runtime::tensor::DTYPE_F32` still resolves) and
//! pins the one value it cannot re-export — the tensor header's data offset
//! is a property of the `NslTensor` struct — with a compile-time assertion.
//!
//! Nothing here may depend on anything: this crate is dependency-free so it
//! can sit under both the compiler and the runtime without adding a build
//! edge either of them notices.

/// The `NslTensor` header, as compiled code addresses it.
pub mod tensor {
    /// Byte offset of the `data` pointer inside `NslTensor`: the header is
    /// `#[repr(C)] { magic: u32, data: *mut c_void, … }`, so the pointer
    /// sits at the first 8-aligned slot after the 4-byte magic. The runtime
    /// asserts `offset_of!(NslTensor, data)` against this at compile time;
    /// codegen's tensor loads (`expr/mod.rs`, the calibration binary) and
    /// the C header use it directly.
    pub const DATA_OFFSET: usize = 8;
    /// The live-handle marker in the first word. Spells "NSLT" in ASCII.
    pub const MAGIC: u32 = 0x4E53_4C54;
    /// The poison written into the magic word when a tensor is freed.
    pub const FREED: u32 = 0x0000_DEAD;
}

/// The `NslTensor.dtype` (u16) tag space. THIS is the single source of
/// truth; the runtime's `tensor_parallel::collective`, `dlpack` and the
/// codegen's `cpdt_precision_exec` are pinned to these by compile-time
/// assertions and the runtime's `dtype_abi_lock` test. Add new tags at the
/// next free slot — never reuse a value (tag 4 was once overloaded for i32
/// tokens; `DTYPE_I32` exists so that can never recur).
pub mod dtype {
    pub const DTYPE_F64: u16 = 0;
    pub const DTYPE_F32: u16 = 1;
    pub const DTYPE_FP16: u16 = 2;
    pub const DTYPE_BF16: u16 = 3;
    pub const DTYPE_INT8: u16 = 4;
    pub const DTYPE_FP8E4M3: u16 = 5;
    pub const DTYPE_FP8E5M2: u16 = 6;
    pub const DTYPE_U16_TOKEN: u16 = 7;
    pub const DTYPE_U16_SEGMENT: u16 = 8;
    pub const DTYPE_I32: u16 = 9;
    /// Blockwise-quantized int8 (`nsl_tensor_quant_int8_blockwise`): `len`
    /// int8 values padded to 4 bytes, then one f32 scale per 64-value block,
    /// in ONE buffer. It carries its own tag because its buffer is not
    /// `len` bytes — the runtime's `data_byte_size` sizes it by the packed
    /// formula, which is what makes its free and clone match its allocation
    /// (found by Miri as a mismatched-layout dealloc while it was tagged
    /// `DTYPE_INT8`). No C-API / DLPack representation: exporting one is
    /// refused like any other unsupported dtype.
    pub const DTYPE_INT8_BLOCKWISE: u16 = 10;
    /// First user-defined dtype tag.
    pub const DTYPE_CUSTOM_START: u16 = 256;
}

/// The runtime C-ABI version (`nsl_abi_version()`, and the
/// `NSL_ABI_VERSION_*` macros in every generated header).
pub mod version {
    /// **Major**: bump on any *breaking* change to an exported symbol's
    /// signature or semantics, or to the `NslTensorDesc` memory layout. A
    /// host whose header major differs from the runtime's must refuse to
    /// run.
    pub const NSL_ABI_VERSION_MAJOR: u32 = 1;
    /// **Minor**: bump for backward-compatible additions (new exported
    /// symbols, new trailing optional behavior). A host built against minor
    /// `m` can use a runtime with minor `>= m` and the same major.
    ///
    /// Minor 1 (item 7): `nsl_model_call_into`, `nsl_model_call_alloc`,
    /// `nsl_model_get_export_signature`, `nsl_dispatch_apply_scalar_result`,
    /// and ownership-transferring outputs from `nsl_model_call_dlpack` /
    /// `nsl_model_forward_dlpack`. `NslTensorDesc` stays 48 bytes.
    pub const NSL_ABI_VERSION_MINOR: u32 = 1;

    /// The version packed as `(major << 16) | minor` — what the runtime's
    /// `nsl_abi_version()` returns and what a generated header's
    /// `NSL_ABI_VERSION` macro expands to.
    pub const fn packed() -> i64 {
        ((NSL_ABI_VERSION_MAJOR as i64) << 16) | (NSL_ABI_VERSION_MINOR as i64)
    }
}

/// Parameter-residency plan bits (`nsl_param_plan_declare`): codegen bakes
/// them per parameter, the runtime confirms each parameter landed where the
/// plan said. An unknown bit means the two disagree about the encoding.
pub mod param_plan {
    /// Registered with a residency backend (not device-resident for the
    /// whole step).
    pub const PLAN_STREAMED: i64 = 1 << 0;
    /// Authoritative storage is bf16 with stochastic rounding
    /// (`--param-dtype bf16-sr`).
    pub const PLAN_BF16_SR: i64 = 1 << 1;
    /// Tensor-granular sharded across ranks (`--zero-stage 3`).
    pub const PLAN_SHARDED: i64 = 1 << 2;
    /// Elementwise sharded within the zero-3 backend (item 11); only valid
    /// alongside [`PLAN_SHARDED`].
    pub const PLAN_ELEMENTWISE: i64 = 1 << 3;
    /// Every bit this ABI version defines.
    pub const PLAN_KNOWN_BITS: i64 = PLAN_STREAMED | PLAN_BF16_SR | PLAN_SHARDED | PLAN_ELEMENTWISE;
}

/// Allocation-surface tags: the wire values of `nsl_gpu_set_alloc_surface`
/// / `nsl_gpu_get_alloc_surface`. The runtime's `SurfaceTag` enum takes its
/// discriminants from these; codegen's train-block brackets emit them.
pub mod surface {
    pub const SURFACE_OTHER: u8 = 0;
    pub const SURFACE_WEIGHTS: u8 = 1;
    pub const SURFACE_OPTIM_M: u8 = 2;
    pub const SURFACE_OPTIM_V: u8 = 3;
    pub const SURFACE_M_PARTIAL: u8 = 4;
    pub const SURFACE_GRADS: u8 = 5;
    pub const SURFACE_ACTIVATIONS: u8 = 6;
    pub const SURFACE_ATTN_WORKSPACE: u8 = 7;
    /// Number of surfaces (array size for per-surface counters).
    pub const NUM_SURFACES: usize = 8;
}

/// PCA Tier B: the seq_len floor the runtime dispatch gate applies and the
/// conservative maximum codegen bakes into the Tier-B-on PTX's shared-memory
/// allocation. Both were measured on an RTX 5070 Ti (sm_120); see the
/// findings docs cited from `nsl_runtime::pca_tier_b_runtime`.
pub mod pca_tier_b {
    /// Empirical seq_len floor (wall-time win >= 10% per dispatch spec §6).
    pub const TIER_B_SEQ_LEN_FLOOR: u32 = 128;
    /// Conservative-max seq_len baked into the Tier-B-on PTX SMEM allocation.
    pub const TIER_B_MAX_BAKED_SEQ_LEN: u32 = 16384;
}

/// The C-API tensor descriptor (`NslTensorDesc` in every generated header):
/// the one `repr(C)` struct a host, the runtime and the compiler's emitted
/// wrappers all address by byte offset.
///
/// Declared here (roadmap A3, "data layouts and records") so the compiler
/// takes its size from the struct rather than from a literal it had to keep
/// in lockstep, and so the runtime's `c_api::NslTensorDesc` is this type
/// re-exported. Layout, 48 bytes, 8-byte aligned — pinned by the constant
/// assertions below and, from the C side, by
/// `crates/nsl-codegen/tests/c_header_compiles.rs`:
///
/// ```text
/// offset  0: data         (*mut c_void, 8)
/// offset  8: shape        (*mut i64,    8)
/// offset 16: strides      (*mut i64,    8)   NULL = contiguous
/// offset 24: ndim         (i32,         4)
/// offset 28: dtype        (i32,         4)   canonical tag space (`dtype`)
/// offset 32: device_type  (i32,         4)   0 = CPU, 1 = CUDA
/// offset 36: device_id    (i32,         4)   GPU index (0 for CPU)
/// offset 40: tape_id      (i64,         8)   autodiff identity (0 = untracked)
/// ```
///
/// Any change to this layout is a **major** ABI version bump
/// ([`version`]).
pub mod tensor_desc {
    use core::ffi::c_void;

    /// Tensor descriptor matching the C header. The C API speaks the
    /// canonical runtime dtype tag space verbatim (`super::dtype`): the
    /// historical inverted 0=f32/1=f64 convention was removed in the P4
    /// item-16 dtype/ABI migration.
    ///
    /// `tape_id` carries the source tensor's autodiff tape id verbatim so
    /// that a desc round-trip (`nsl_tensor_to_desc` → `desc_to_nsl_tensor`)
    /// does not strip the id. Required for the per-call grad context
    /// (Spec B): the loss seed in `run_backward_core` keys on `t.tape_id`,
    /// which would fall through to the raw-pointer fallback if the desc
    /// dropped the id. `tape_id == 0` means the source tensor was never
    /// autodiff-tracked (constants, freshly-allocated wrappers, inputs from
    /// non-grad code paths); `tape_id > 0` matches the source tensor's
    /// `tape_id` as assigned by `Tape::get_or_assign_id`.
    ///
    /// `#[derive(Default)]` is kept for the runtime's scratch-desc
    /// allocation sites; a struct-literal site must name `tape_id`.
    #[repr(C)]
    #[derive(Default)]
    pub struct NslTensorDesc {
        /// Element buffer.
        pub data: *mut c_void,
        /// `ndim` dimension sizes.
        pub shape: *mut i64,
        /// `ndim` element strides, or NULL for contiguous row-major.
        pub strides: *mut i64,
        /// Rank.
        pub ndim: i32,
        /// Canonical NSL dtype tag: 0=f64, 1=f32, 2=f16, 3=bf16, 4=int8,
        /// 5=fp8e4m3, 6=fp8e5m2, 7=u16-token, 8=u16-segment, 9=int32.
        pub dtype: i32,
        /// 0=CPU, 1=CUDA
        pub device_type: i32,
        /// GPU index (0 for CPU)
        pub device_id: i32,
        /// Autodiff tape id of the source tensor, copied verbatim across
        /// desc round-trips. `0` means "untracked".
        pub tape_id: i64,
    }

    /// `sizeof(NslTensorDesc)`: what emitted code multiplies an index by to
    /// step through a descriptor array, and the size of the stack slot it
    /// builds a scratch descriptor in.
    pub const SIZE: usize = core::mem::size_of::<NslTensorDesc>();

    // The layout the header documents, checked on the struct itself so a
    // field reorder or a type change cannot compile.
    const _: () = {
        assert!(SIZE == 48);
        assert!(core::mem::align_of::<NslTensorDesc>() == 8);
        assert!(core::mem::offset_of!(NslTensorDesc, data) == 0);
        assert!(core::mem::offset_of!(NslTensorDesc, shape) == 8);
        assert!(core::mem::offset_of!(NslTensorDesc, strides) == 16);
        assert!(core::mem::offset_of!(NslTensorDesc, ndim) == 24);
        assert!(core::mem::offset_of!(NslTensorDesc, dtype) == 28);
        assert!(core::mem::offset_of!(NslTensorDesc, device_type) == 32);
        assert!(core::mem::offset_of!(NslTensorDesc, device_id) == 36);
        assert!(core::mem::offset_of!(NslTensorDesc, tape_id) == 40);
    };
}

/// The resolved train-configuration record (`nsl_set_train_config_record`):
/// codegen renders one fixed-order `k=v,k=v` string from the resolved
/// config at train-block entry, the `.optim` sidecar carries it verbatim,
/// and resume diffs saved-vs-live per key. The two key classes below are
/// the record's *schema* — which keys exist and how their drift is
/// judged — so both the renderer (codegen) and the checker (the runtime's
/// `train_config_record`) read them from here.
pub mod train_config {
    /// Keys whose drift changes the meaning of restored optimizer state or
    /// the restored step counter (`accum` is the optimizer-step divisor and
    /// the bias-correction clock). A resume under different values is not
    /// a continuation: the runtime aborts, no escape. Explicit list, not
    /// the negation of the other — a key in neither class is deliberately
    /// silent (the #519 doctrine).
    pub const MOMENT_KEYS: &[&str] = &[
        "opt", "accum", "beta1", "beta2", "eps", "wd", "momentum", "dampening",
        "nesterov", "ns_steps", "adamw_lr", "no_decay",
    ];

    /// Keys whose drift changes the future trajectory only; the runtime
    /// aborts by default and `NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1` turns
    /// the refusal into a loud acknowledgment. `sp4..sp6` are reserved
    /// ahead of any 4+-parameter scheduler: a parameter rendered under a
    /// key in neither class would be silently unguarded.
    pub const TRAJECTORY_KEYS: &[&str] =
        &["lr", "sched", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "clip"];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical dtype table — do not renumber; append at the next free
    /// slot. Mirrors the runtime's `dtype_abi_lock`, which now tests the
    /// re-exports of exactly these.
    #[test]
    fn dtype_tags_are_locked() {
        use dtype::*;
        assert_eq!(
            [
                DTYPE_F64, DTYPE_F32, DTYPE_FP16, DTYPE_BF16, DTYPE_INT8, DTYPE_FP8E4M3,
                DTYPE_FP8E5M2, DTYPE_U16_TOKEN, DTYPE_U16_SEGMENT, DTYPE_I32,
                DTYPE_INT8_BLOCKWISE,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(DTYPE_CUSTOM_START, 256);
    }

    /// The two record classes are disjoint, so a key's drift has one
    /// verdict.
    #[test]
    fn train_config_key_classes_are_disjoint() {
        for k in train_config::MOMENT_KEYS {
            assert!(!train_config::TRAJECTORY_KEYS.contains(k), "{k} in both classes");
        }
    }

    #[test]
    fn the_packed_version_round_trips() {
        let p = version::packed();
        assert_eq!((p >> 16) as u32, version::NSL_ABI_VERSION_MAJOR);
        assert_eq!((p & 0xffff) as u32, version::NSL_ABI_VERSION_MINOR);
    }

    #[test]
    fn plan_bits_are_distinct_and_all_known() {
        use param_plan::*;
        let bits = [PLAN_STREAMED, PLAN_BF16_SR, PLAN_SHARDED, PLAN_ELEMENTWISE];
        for (i, a) in bits.iter().enumerate() {
            assert_eq!(a.count_ones(), 1);
            for b in &bits[i + 1..] {
                assert_eq!(a & b, 0);
            }
        }
        assert_eq!(bits.iter().fold(0, |acc, b| acc | b), PLAN_KNOWN_BITS);
    }

    #[test]
    fn surfaces_are_dense_from_zero() {
        use surface::*;
        let tags = [
            SURFACE_OTHER, SURFACE_WEIGHTS, SURFACE_OPTIM_M, SURFACE_OPTIM_V, SURFACE_M_PARTIAL,
            SURFACE_GRADS, SURFACE_ACTIVATIONS, SURFACE_ATTN_WORKSPACE,
        ];
        assert_eq!(tags.len(), NUM_SURFACES);
        for (i, t) in tags.iter().enumerate() {
            assert_eq!(*t as usize, i);
        }
    }

    #[test]
    fn the_tensor_header_constants_are_what_compiled_code_assumes() {
        assert_eq!(tensor::DATA_OFFSET, 8);
        assert_eq!(&tensor::MAGIC.to_be_bytes(), b"NSLT");
        assert_ne!(tensor::MAGIC, tensor::FREED);
    }
}
