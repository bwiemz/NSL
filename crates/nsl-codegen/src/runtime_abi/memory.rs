//! Arenas, device slabs and CUDA graph regions.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_MEMORY: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Milestone C p2 Stage-2B: the placed transient arena. `bind` arms a
    // single-shot, size-exact pin that the device allocator consumes; `unbind`
    // disarms it so an op that took a non-allocating path cannot leak the pin
    // into an unrelated allocation.
    ("nsl_arena_init", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_arena_bind", &[types::I64, types::I64, types::I64], None),
    ("nsl_arena_unbind", &[], None),
    ("nsl_arena_unbind_verify", &[types::I64], None),
    ("nsl_arena_declare_slot", &[types::I64, types::I64], None),
    ("nsl_arena_check", &[], Some(types::I64)),
    ("nsl_arena_check_step", &[types::I64], None),
    ("nsl_arena_destroy", &[], None),
    // M36: GPU memory slab (compile-time planned device memory arena)
    ("nsl_gpu_slab_init", &[types::I64], Some(types::I64)),
    (
        "nsl_slab_offset",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_gpu_slab_destroy", &[], None),
    ("nsl_gpu_slab_active", &[], Some(types::I64)),
    (
        "nsl_tensor_from_slab",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // P5 item 19: opportunistic per-region CUDA graph capture/replay
    ("nsl_cuda_graphs_enable", &[types::I64], None), // (accum_window)
    ("nsl_cuda_graph_region_begin", &[types::I64], None), // (region_id)
    ("nsl_cuda_graph_region_end", &[types::I64], None), // (region_id)
    ("nsl_cuda_graphs_report", &[], None),
];
