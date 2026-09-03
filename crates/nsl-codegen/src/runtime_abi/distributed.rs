//! Tensor, pipeline and ZeRO parallelism.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_DISTRIBUTED: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // --- M41: Disaggregated inference ---
    (
        "nsl_disagg_init",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_disagg_get_role", &[], Some(types::I64)),
    ("nsl_disagg_get_rank", &[], Some(types::I64)),
    ("nsl_disagg_destroy", &[], Some(types::I64)),
    (
        "nsl_disagg_worker_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_disagg_worker_destroy", &[], Some(types::I64)),
    ("nsl_disagg_prefill_loop", &[types::I64], Some(types::I64)),
    ("nsl_disagg_decode_loop", &[types::I64], Some(types::I64)),
    // --- M41b: KV transfer backends ---
    (
        "nsl_kv_transfer_init",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_transfer_send",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_transfer_recv",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_transfer_destroy", &[], Some(types::I64)),
    // --- M30: Tensor parallelism ---
    ("nsl_tp_init", &[], Some(types::I64)),
    ("nsl_tp_rank", &[], Some(types::I64)),
    ("nsl_tp_world_size", &[], Some(types::I64)),
    (
        "nsl_tp_all_reduce_sum",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tp_all_gather",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tp_broadcast",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tp_barrier", &[], Some(types::I64)),
    ("nsl_tp_destroy", &[], Some(types::I64)),
    // --- M43: Pipeline parallelism ---
    (
        "nsl_pipeline_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_send",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_recv",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_send_grad",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_pipeline_recv_grad",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    ("nsl_pipeline_barrier", &[], Some(types::I64)),
    ("nsl_pipeline_destroy", &[], Some(types::I64)),
    // --- M43: ZeRO optimizer (ABI-fixed: match runtime signatures exactly) ---
    ("nsl_zero_init", &[types::I64, types::I64], Some(types::I64)), // (stage, world_size)
    ("nsl_zero_partition", &[types::I64], Some(types::I64)),        // (num_params)
    ("nsl_zero_partition_bytes", &[types::I64, types::I64], Some(types::I64)), // (param_list, num_params)
    (
        "nsl_zero_reduce_grads",
        &[types::I64, types::I64],
        Some(types::I64),
    ), // (grad_ptr, num_elems)
    ("nsl_zero_step", &[], Some(types::I64)),                       // ()
    // D3 (ZeRO-1): post-step parameter sync — broadcast each param from
    // its owner rank (idx % world_size) so all ranks hold the full model.
    ("nsl_zero_sync_params", &[types::I64, types::I64], Some(types::I64)), // (param_list, num_params)
    ("nsl_zero_destroy", &[], Some(types::I64)),
    ("nsl_zero_owns_param", &[types::I64], Some(types::I64)), // (param_idx) -> 1 if owned
    // (accum_list, num_params) -> NslList of owned indices; zeroes non-owners'
    // m_partial. Caller frees the list.
    (
        "nsl_zero_owned_step_indices",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // P3 ZeRO-3: tensor-granular parameter sharding (items 12-14).
    ("nsl_zero3_enable", &[], Some(types::I64)),
    ("nsl_zero3_note_param", &[types::I64, types::I64], Some(types::I64)), // (tensor, idx)
    // Item 11: elementwise 1/ws sharding (--zero-elementwise). Mark is
    // plan-driven and precedes the registration belt (the sr-note pattern);
    // the step runs on EVERY rank over its own slice — scalar order mirrors
    // nsl_fase_fused_adamw_step.
    // (tensor, idx, sr) — `sr` is the plan entry's storage decision (item 16x11)
    ("nsl_zero3_mark_elementwise", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // Owner-only moments: THIS RANK's slice-sized m/v for an elementwise
    // param. Sized from the carved ElemShard, so it MUST be emitted after
    // the weight-stream register belt (it aborts otherwise). Notes its own
    // element count against `optim_elems`.
    ("nsl_zero3_alloc_elem_moment", &[types::I64, types::I64], Some(types::I64)), // (theta, idx)
    (
        "nsl_zero3_elem_adamw_step",
        &[
            types::I64, types::I64, types::I64, types::I64, // theta, m, v, idx
            types::F64, types::F64, types::F64, types::F64, types::F64, // lr, b1, omb1, b2, omb2
            types::F64, types::F64, types::F64, types::F64, // eps, wd, bc1_inv, bc2_inv
        ],
        Some(types::I64),
    ),
    // Item 16×11: the composed bf16-sr elementwise step — same scalar order
    // plus the trailing optimizer step the SR counter stream is keyed on.
    (
        "nsl_zero3_elem_sr_adamw_step",
        &[
            types::I64, types::I64, types::I64, types::I64, // theta, m, v, idx
            types::F64, types::F64, types::F64, types::F64, types::F64, // lr, b1, omb1, b2, omb2
            types::F64, types::F64, types::F64, types::F64, // eps, wd, bc1_inv, bc2_inv
            types::I64, // step
        ],
        Some(types::I64),
    ),
    // D3 v2: record an owned optimizer-moment allocation's element count so the
    // G3 gate can prove per-rank optimizer state shrank to ~1/world_size.
    ("nsl_zero_note_optim_alloc", &[types::I64], Some(types::I64)), // (tensor_ptr) -> running elems
    // (tensor_ptr) -> running REPLICA elems; the MomentFill::Full half that
    // nsl_zero_note_optim_alloc deliberately excludes.
    (
        "nsl_zero_note_replicated_optim_alloc",
        &[types::I64],
        Some(types::I64),
    ),
];
