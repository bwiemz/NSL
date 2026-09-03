//! Talking to the outside world: tokenizers, safetensors, ONNX, DLPack,
//! multimodal, unikernel and agent runtimes.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_INTEROP: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Tokenizer functions (M15)
    ("nsl_byte_tokenizer_new", &[], Some(types::I64)),
    (
        "nsl_bpe_train",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tokenizer_load", &[types::I64], Some(types::I64)),
    ("nsl_tokenizer_save", &[types::I64, types::I64], None),
    (
        "nsl_tokenizer_encode",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tokenizer_decode",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tokenizer_vocab_size", &[types::I64], Some(types::I64)),
    (
        "nsl_tokenizer_encode_batch",
        &[types::I64, types::I64, types::I8, types::I8, types::I64],
        Some(types::I64),
    ),
    // GPU runtime functions (M17)
    ("nsl_cuda_init", &[], Some(types::I64)),
    (
        "nsl_kernel_launch",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // User `kernel` block launch: args array holds NslTensor handles; the
    // runtime extracts each `.data` device pointer and builds the kernelParams.
    (
        "nsl_kernel_launch_tensors",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
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
        "nsl_tensor_to_device",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_to_device_like",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Safetensors I/O (M18b)
    (
        "nsl_safetensors_load",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_safetensors_save",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // HuggingFace Hub download + weight loading (M18b)
    (
        "nsl_hf_load",
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
    // Trace infrastructure for ONNX export (M18b Task 7)
    ("nsl_trace_start", &[], None),
    ("nsl_trace_register_input", &[types::I64, types::I64], None),
    ("nsl_trace_register_output", &[types::I64, types::I64], None),
    ("nsl_trace_stop", &[], Some(types::I64)),
    // ONNX export (M18b Tasks 9-10)
    (
        "nsl_onnx_export",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // --- M39b: vmap runtime ---
    (
        "nsl_vmap_check_batch",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M48: Multimodal primitives ---
    (
        "nsl_patch_embed",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_mel_spectrogram",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Explicit-sample-rate variant; the 4-arg form assumes 16 kHz.
    (
        "nsl_mel_spectrogram_sr",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_cross_attention",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_image_resize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_image_normalize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_stft",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_audio_resample",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M62: Legacy Interop — DLPack bridge + C API ---
    ("nsl_dlpack_export", &[types::I64], Some(types::I64)),
    ("nsl_dlpack_import", &[types::I64], Some(types::I64)),
    ("nsl_dlpack_free", &[types::I64], None),
    ("nsl_model_create", &[types::I64], Some(types::I64)),
    ("nsl_model_destroy", &[types::I64], Some(types::I64)),
    (
        "nsl_model_forward",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_model_forward_dlpack",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_model_backward",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_model_get_version", &[], Some(types::I64)),
    ("nsl_get_last_error", &[], Some(types::I64)),
    ("nsl_clear_error", &[], Some(types::I64)),
    // --- M54b: Unikernel runtime ---
    (
        "nsl_unikernel_init",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_unikernel_model_alloc",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_unikernel_kv_alloc",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_unikernel_model_pool_stats", &[], Some(types::I64)),
    ("nsl_unikernel_shutdown", &[], Some(types::I64)),
    ("nsl_unikernel_gpu_init", &[types::I64], Some(types::I64)),
    ("nsl_unikernel_gpu_ready", &[], Some(types::I64)),
    ("nsl_unikernel_gpu_device_id", &[], Some(types::I64)),
    // --- M56 v1 agent runtime FFI (Task 16). Signatures from spec §3.4. ---
    // All raw pointers are I64 per the workspace convention; time: u64 is also I64.
    ("nsl_agent_pool_new", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_agent_pool_destroy", &[types::I64], None),
    ("nsl_agent_pool_acquire", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_agent_pool_release", &[types::I64, types::I64], None),
    ("nsl_agent_scheduler_step", &[types::I64], Some(types::I32)),
    ("nsl_agent_mailbox_write", &[types::I64, types::I64, types::I64], Some(types::I32)),
    ("nsl_agent_mailbox_read", &[types::I64], Some(types::I64)),
];
