//! The registry. Sorted by name; `tests/registry_agreement.rs` fails when a
//! variable is read but missing here, or listed here but no longer read.
//!
//! Columns: name, kind, accepted values, default when unset, tier, where it
//! is read (compile = by `nsl build`; runtime = by the compiled program or
//! `nsl run`), one-line doc. `accepted` quotes what the code compares
//! against — `1 only` means anything else (including `0`) is ignored.

use crate::{EnvVar, Kind, ReadAt, Tier};

macro_rules! var {
    ($name:literal, $kind:ident, $accepted:literal, $default:literal, $tier:ident, $read_at:ident, $doc:literal) => {
        EnvVar {
            name: $name,
            kind: Kind::$kind,
            accepted: $accepted,
            default: $default,
            tier: Tier::$tier,
            read_at: ReadAt::$read_at,
            doc: $doc,
        }
    };
}

pub static REGISTRY: &[EnvVar] = &[
    var!(
        "NSL_ALIGN_DEBUG",
        Bool,
        "any value (presence via is_ok; all 3 sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print packed-batch device-alignment moves and add_inplace dtype-mismatch details to stderr as [align-debug] lines."
    ),
    var!(
        "NSL_ALLOW_UNKNOWN_DECORATORS",
        Bool,
        "1 only (exact match; set by --allow-unknown-decorators)",
        "off",
        Safety,
        Compile,
        "Demotes the unknown-decorator-name error to a warning; documented-but-unimplemented decorators still refuse."
    ),
    var!(
        "NSL_ALLOW_UNRESOLVED_LIVE_ADJOINT",
        Bool,
        "1 only (exact match)",
        "off (hard error)",
        Safety,
        Compile,
        "Migration escape hatch: turns the source-AD live-adjoint-with-missing-inputs hard error into a loud warning and skips the op."
    ),
    var!(
        "NSL_ARENA_CHECK",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Verify every transient-arena guard byte at the end of each training step (O(arena) D2H copy) and report corrupted regions."
    ),
    var!(
        "NSL_ARENA_DEBUG",
        Bool,
        "1 only (both crates)",
        "off",
        Diagnostic,
        Both,
        "Transient-arena detail: at compile time eligibility/refusal/placement lines (with --transient-arena); at runtime [arena-debug] slot pin mismatches."
    ),
    var!(
        "NSL_ARENA_REPORT",
        Bool,
        "1 only (exact match)",
        "off (also on under --memory-report or --transient-arena)",
        Diagnostic,
        Compile,
        "Prints the transient-arena liveness/slot report for the backward tape to stderr; analysis only, no codegen change."
    ),
    var!(
        "NSL_ARENA_SLOT_LIMIT",
        Int,
        "integer slot count (trimmed); unparseable → 0 = place nothing (fail-closed)",
        "unlimited (i64::MAX)",
        Perf,
        Runtime,
        "Place only transient-arena slots with index below N; bisection aid for arena corruption. Unparseable values place nothing."
    ),
    var!(
        "NSL_ASYNC_ALLOC",
        Bool,
        "1 only",
        "off",
        Perf,
        Runtime,
        "Allocate GPU memory via cuMemAllocAsync from the driver's default pool instead of the caching allocator; incompatible with CUDA graphs."
    ),
    var!(
        "NSL_AUTOTUNE_FALLBACK",
        Bool,
        "any value (set, even empty; docs say 1)",
        "off",
        Perf,
        Compile,
        "Skips @autotune measurement/cost-model selection and picks the median candidate of every tuning parameter (no-GPU mode)."
    ),
    var!(
        "NSL_AUTOTUNE_VERBOSE",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Logs @autotune cache hits/misses, cache-key inputs, per-variant failures and the timing/cost report to stderr."
    ),
    var!(
        "NSL_BENCH_DUMP_PTX",
        Bool,
        "any value (set, even empty; var_os)",
        "off",
        Diagnostic,
        Compile,
        "PCA Tier-B bench harness only: writes the synthesized flash-attention fwd/bwd PTX to the system temp dir for post-mortem."
    ),
    var!(
        "NSL_BENCH_PRINT_DECISIONS",
        Bool,
        "any value (set, even empty; var_os)",
        "off",
        Diagnostic,
        Compile,
        "PCA Tier-B bench harness only: prints the tile skip-decisions buffer histogram and its first 128 bytes to stderr."
    ),
    var!(
        "NSL_BF16_CACHE_PROBE",
        Bool,
        "presence marks the child; parent sets 1",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child of the bf16 weight-cast cache GPU gate; the parent returns when it is set."
    ),
    var!(
        "NSL_BF16_CAST_CACHE_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print bf16 weight-cast cache hits/recasts/evictions to stderr at exit; NSL_EVENTS gets the JSON twin regardless."
    ),
    var!(
        "NSL_BF16_LT_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print cublasLt bf16 GEMM issued/tuned/fallback counts to stderr at exit; NSL_EVENTS gets the JSON twin regardless."
    ),
    var!(
        "NSL_BF16_LT_PROBE",
        Enum,
        "on = Lt-armed child, off = GemmEx child; anything else = parent",
        "unset",
        Test,
        Test,
        "Test harness only: selects which re-exec'd child of the cublasLt bf16 GEMM gate runs (on = Lt path, off = GemmEx control)."
    ),
    var!(
        "NSL_BF16_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd timing child of the bf16 matmul mode gate, which prints one PROBE line for the parent."
    ),
    var!(
        "NSL_CAPI_TRACE",
        Bool,
        "any value (presence via var_os().is_some())",
        "off",
        Diagnostic,
        Runtime,
        "Trace C-API calls (model_create, load_weight, dispatch) to stderr as [nsl-capi] lines."
    ),
    var!(
        "NSL_CCR_DEBUG",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Prints the CCR adjoint last-use free count and the number of weight-gradient fusion chains kept contiguous."
    ),
    var!(
        "NSL_CCR_SEGMENT_FREE",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for CCR per-segment forward early-free (memory only; already skipped under --weight-stream or --layerwise-accum)."
    ),
    var!(
        "NSL_COLLECTIVES",
        Enum,
        "sim or empty / sim-gpu / nccl; anything else refuses init (rc -2)",
        "sim",
        Behavior,
        Runtime,
        "Collective backend for --devices N ZeRO runs: CPU-shm simulated, GPU-staged test backend, or real NCCL (needs an nccl build)."
    ),
    var!(
        "NSL_CPDT_FORCE_STALE_PLAN",
        Bool,
        "1 only (exact match)",
        "off",
        Safety,
        Compile,
        "Test knob: forces the stale-CPDT-precision-plan refusal so the moment-dtype staleness gate is testable; the build fails."
    ),
    var!(
        "NSL_CSHA_DUMP_GRADS",
        Bool,
        "any value except 0 or empty (var_os; all 3 sites)",
        "off",
        Diagnostic,
        Runtime,
        "Dump CSHA fused-attention kernel names, saved activations and gradient stats to stderr; also writes the forward PTX to the temp dir."
    ),
    var!(
        "NSL_CSHA_DUMP_SAVE_STATE",
        Enum,
        "exact strings; empty/other = real values; direct_* only with save_activations_for_backward",
        "off (real row_max/row_sum saved)",
        Behavior,
        Compile,
        "Debug probe: emits PTX that overwrites the saved flash-attention softmax state with lane/warp/register probes; corrupts the backward."
    ),
    var!(
        "NSL_CSHA_MULTITILE_DW_VALIDATION",
        Bool,
        "any value except 0 or empty (var_os)",
        "off (refuse)",
        Safety,
        Runtime,
        "Let the CSHA multi-tile fused backward launch for batch>1 or heads>1 instead of refusing; gradients are NOT guaranteed correct."
    ),
    var!(
        "NSL_CSLA_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print the CSLA layerwise window-backward phase count to stderr at exit; NSL_EVENTS gets the JSON twin regardless."
    ),
    var!(
        "NSL_CSLA_REPORT",
        Bool,
        "1 only (exact match)",
        "off",
        Diagnostic,
        Compile,
        "Prints the layerwise-accumulation (CSLA) layer grouping and tied/cross-layer classification report; analysis only."
    ),
    var!(
        "NSL_CUDA_DEVICE",
        Int,
        "integer device ordinal (negative clamps to 0; unparseable ignored)",
        "rank % device_count under the --devices spawner, else 0",
        Platform,
        Runtime,
        "CUDA device ordinal this process binds; overrides the rank-striped choice made under the --devices spawner."
    ),
    var!(
        "NSL_CUDA_GRAPHS",
        Bool,
        "0 disables; anything else ignored",
        "on when the program was compiled with --cuda-graphs",
        Perf,
        Runtime,
        "Kill switch for per-region CUDA graph capture/replay in programs built with --cuda-graphs; 0 keeps every region eager."
    ),
    var!(
        "NSL_CUDA_GRAPH_LOG",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Log per-region CUDA graph record/capture/replay transitions to stderr (only read when graph capture is enabled)."
    ),
    var!(
        "NSL_CUDA_GRAPH_TEST_DIVERGE_FIRST",
        Bool,
        "1 only",
        "off",
        Test,
        Runtime,
        "Test hook: force each graph region's first replay verification to report a mismatch at op 0, exercising the eager-repair path."
    ),
    var!(
        "NSL_CUDA_SYNC",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Synchronize the device after every kernel launch so async CUDA errors surface at the launch site; disables CUDA graph capture."
    ),
    var!(
        "NSL_DEBUG",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Verbose compiler notes: Cranelift calling convention, kernel-profile pre-pass skips/failures and manifest writes."
    ),
    var!(
        "NSL_DEBUG_ESCAPE",
        Bool,
        "1 only (exact match)",
        "off",
        Diagnostic,
        Compile,
        "Prints the interprocedural escape-analysis results and per-call-site captive-argument flags to stderr."
    ),
    var!(
        "NSL_DEBUG_MEM_ALL",
        Bool,
        "1 only (both sites)",
        "off (alloc summary prints steps 0-2, memory report steps 0-5)",
        Diagnostic,
        Runtime,
        "Print the GPU allocation summary and memory report on every step instead of only the first few."
    ),
    var!(
        "NSL_DEBUG_MEM_TRACE",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Trace every caching-allocator block handout/free and every tensor free (pointer, refcount, context) to stderr for leak hunting."
    ),
    var!(
        "NSL_DEBUG_SOURCE_AD_OWNED",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Prints a per-op-kind histogram of the tensors the source-AD forward/backward lowering owns and bulk-frees."
    ),
    var!(
        "NSL_DEBUG_WENGERT",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Dumps the primal Wengert tape (VarId, name, op, inputs) and reports adjoint ops skipped for missing inputs."
    ),
    var!(
        "NSL_DLPACK_REFUSAL_LIB",
        Path,
        "path to the built shared library",
        "unset",
        Test,
        Test,
        "Test harness only: shared-library path handed to the DLPack unsupported-dtype refusal child scenarios."
    ),
    var!(
        "NSL_DLPACK_REFUSAL_SCENARIO",
        Str,
        "desc42 or a model scenario name",
        "unset (child returns)",
        Test,
        Test,
        "Test harness only: names the DLPack unsupported-dtype refusal scenario the re-exec'd child runs."
    ),
    var!(
        "NSL_DLPACK_REFUSAL_WEIGHTS",
        Path,
        "path to the weights file",
        "unset",
        Test,
        Test,
        "Test harness only: weights path handed to the DLPack unsupported-dtype refusal child scenarios."
    ),
    var!(
        "NSL_DTYPE_REFUSAL_SCENARIO",
        Str,
        "scenario name",
        "unset (child returns)",
        Test,
        Test,
        "Test harness only: names the poisoned GPU dtype-refusal call the re-exec'd child makes (expected to abort)."
    ),
    var!(
        "NSL_DX_NORM_DIAG_INPUT",
        Enum,
        "random selects deterministic-random inputs; anything else = ones",
        "ones",
        Test,
        Test,
        "Test harness only: input pattern for the CSHA dx-norm readback diagnostic (ones mirrors the failing smoke, random varies values)."
    ),
    var!(
        "NSL_EMBEDDING_BWD_CPU",
        Bool,
        "1 only",
        "off (GPU scatter-add)",
        Behavior,
        Runtime,
        "Force the deterministic host scatter for the embedding backward instead of the GPU atomic scatter-add (bisections, M46 audits)."
    ),
    var!(
        "NSL_EVENTS",
        Path,
        "any non-empty path (opened append/create; open or write failure warns once and disables)",
        "off",
        Diagnostic,
        Runtime,
        "Append one JSON event per line to this file (machine twins of the bracketed stderr counters) and register every counter reporter."
    ),
    var!(
        "NSL_FASE_BATCH_SUMSQ",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Behavior,
        Compile,
        "FASE two-phase clip: batches the grad-norm sum-of-squares into one f64-accumulating call; 0 restores per-param f32 reads (bits differ)."
    ),
    var!(
        "NSL_FASE_FUSED_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print fused FASE optimizer-step launch, block-table build and multi-param batched/fallback counts at exit; NSL_EVENTS gets JSON."
    ),
    var!(
        "NSL_FASE_FUSED_OVERRIDE",
        List,
        "comma-separated 1/true/0/false, exactly one per WGGO layer; other token or wrong length ignores the whole spec with a warning",
        "off (plan's own per-layer fase_fused)",
        Behavior,
        Compile,
        "Test knob: replaces the WGGO plan's per-layer FASE fused/full-buffer accumulation mode table (needs WGGO overrides)."
    ),
    var!(
        "NSL_FASE_FUSED_STEP",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for the single fused AdamW step kernel (bit-exact with the interpreted ~15-launch step); 0 refuses under bf16-sr."
    ),
    var!(
        "NSL_FASE_MULTI_STEP",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for the multi-tensor pointer-table fused-AdamW launch (bit-identical); 0 falls back to the per-parameter fused step."
    ),
    var!(
        "NSL_FA_BWD_MULTIWARP",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for the 4-warp flash-attention MMA backward emission (_w4 launch contract); 0 restores single-warp, bit-identical."
    ),
    var!(
        "NSL_FA_FWD_MMA",
        Bool,
        "0 disables; anything else or unset = on",
        "on (when block/head-dim shape admits)",
        Behavior,
        Compile,
        "Kill switch for the tensor-core (mma.sync) flash-attention forward; 0 selects the scalar kernel, which rounds differently."
    ),
    var!(
        "NSL_FLASH_ALLOW_FAILED",
        Bool,
        "1 only (both sites)",
        "off (abort)",
        Safety,
        Runtime,
        "Continue with an unwritten (all-zero) attention output after a failed FlashAttention forward launch or f16 widen instead of aborting."
    ),
    var!(
        "NSL_FLASH_BWD_CPU",
        Enum,
        "1 forces the CPU reference; 0 opts out of --deterministic routing (warns once); anything else = flag decides",
        "unset: --deterministic decides, else GPU",
        Behavior,
        Runtime,
        "Route the FlashAttention backward to the deterministic CPU reference (1) or opt out of that routing under --deterministic (0)."
    ),
    var!(
        "NSL_FLASH_DEBUG",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Narrate the FlashAttention backward dispatch decision (CPU vs GPU, GQA mode, view materialization) on stderr as [flash-bwd] lines."
    ),
    var!(
        "NSL_FLASH_GQA_BWD_CPU",
        Bool,
        "1 only",
        "off",
        Behavior,
        Runtime,
        "Route grouped-query (GQA) FlashAttention backward to the CPU reference instead of the GPU expand-KV/native kernels."
    ),
    var!(
        "NSL_FLASH_GQA_BWD_EXPAND",
        Bool,
        "1 only",
        "off (native grouped kernel)",
        Behavior,
        Runtime,
        "Force the legacy expand-KV envelope for the GQA FlashAttention GPU backward instead of the native grouped phase-2 kernel (A/B)."
    ),
    var!(
        "NSL_FORCE_DEPENDENCY_ORDER_VIOLATION",
        Bool,
        "1 only (exact match)",
        "off",
        Safety,
        Compile,
        "Test knob: forces the pass-manager dependency-order refusal (build fails naming the knob) so the enforcement arm is gate-testable."
    ),
    var!(
        "NSL_FUSED_EW_ABORT_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child that feeds a malformed fused-elementwise descriptor and must abort."
    ),
    var!(
        "NSL_FUSED_EW_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print fused elementwise-chain launch vs decomposed-replay fallback counts at exit; NSL_EVENTS gets the JSON twin regardless."
    ),
    var!(
        "NSL_FUSED_LCE_ALLOW_NON_F32_FFI",
        Bool,
        "any value (presence via var_os().is_some())",
        "off",
        Diagnostic,
        Runtime,
        "Deprecated v6 knob with NO effect; if set, prints a one-time warning pointing at NSL_FUSED_LCE_REFUSE_NON_F32."
    ),
    var!(
        "NSL_FUSED_LCE_GEMM",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Behavior,
        Compile,
        "Kill switch for the GEMM-chunked fused linear+CE path (large vocab / biasless heads); 0 restores v1 kernels and refuses biasless heads."
    ),
    var!(
        "NSL_FUSED_LCE_HINT_PIN",
        Bool,
        "0 disables the abort; anything else keeps it",
        "on (abort on hint mismatch)",
        Safety,
        Runtime,
        "Set to 0 to keep running (after printing the diagnostic) when @fused_lm_ce shape hints disagree with the real tensors, instead of aborting."
    ),
    var!(
        "NSL_FUSED_LCE_REFUSE_NON_F32",
        Bool,
        "any value (presence via var_os().is_none() check, so even 0 refuses); second site is a test EnvGuard",
        "off",
        Safety,
        Runtime,
        "Reinstate the v6 refusal: fused linear-CE FFIs return -1 for non-f32 (f16/bf16) operands instead of running the kernel."
    ),
    var!(
        "NSL_FUSE_ELEMENTWISE_BWD",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for fusing single-reader elementwise chains of the backward tape into one kernel (bit-exact by construction)."
    ),
    var!(
        "NSL_FUSE_NORM_RESIDUAL",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for folding the residual Add into the fused RMSNorm dx backward kernel (bit-exact; needs --fuse-rmsnorm-backward)."
    ),
    var!(
        "NSL_FUSE_ROPE_NEG",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for the fused rotate_half_neg backward op; 0 emits the rotate_half + Neg pair (rounding-free, bit-exact either way)."
    ),
    var!(
        "NSL_FUSE_SCALAR_IMM",
        Bool,
        "0 disables; anything else or unset = on",
        "on",
        Perf,
        Compile,
        "Kill switch for rewriting x OP const backward ops into scalar-immediate launches (bit-exact; skips the host scalar alloc/HtoD)."
    ),
    var!(
        "NSL_GATHER_DIM_GPU",
        Bool,
        "0 disables; anything else on",
        "on",
        Perf,
        Runtime,
        "Set to 0 to route non-dim-0 f32 gather back through the host copy path instead of the device gather-dim kernel (A/B)."
    ),
    var!(
        "NSL_GDS_ENABLED",
        Bool,
        "1 only",
        "off (host-staged read + cuMemcpyHtoD)",
        Platform,
        Runtime,
        "Declare GPUDirect Storage (cuFile) available for file-to-GPU reads; the cuFile path is unimplemented and returns Unsupported."
    ),
    var!(
        "NSL_GPU_CE_BACKWARD",
        Bool,
        "0 disables; anything else on",
        "on",
        Behavior,
        Runtime,
        "Set to 0 to compute the cross-entropy backward via the host bounce instead of the device-resident kernel (~1e-6 relative rounding diff)."
    ),
    var!(
        "NSL_GPU_COUNT",
        Int,
        "integer GPU count (>=2 = CUDA IPC available; unparseable ignored)",
        "assume IPC available",
        Platform,
        Runtime,
        "GPU count hint for the disaggregated KV-transfer NVLink/CUDA-IPC probe; fewer than 2 disables the NVLink backend."
    ),
    var!(
        "NSL_GPU_GRAD_CLIP",
        Bool,
        "0 disables; anything else on",
        "on",
        Behavior,
        Runtime,
        "Set to 0 to compute the gradient-clip global norm on the host instead of the device (f64-accumulated; differs only by reduction order)."
    ),
    var!(
        "NSL_GPU_MEM_LIMIT",
        Int,
        "bytes with optional K/M/G suffix (case-insensitive), e.g. 24G; 0, empty or unparseable = unlimited",
        "0 (unlimited)",
        Perf,
        Runtime,
        "Cap in bytes (K/M/G suffix allowed) on what the GPU caching allocator may reserve from the driver; growth past it fails as OOM."
    ),
    var!(
        "NSL_GPU_MEM_REPORT",
        Bool,
        "any value (presence via is_ok)",
        "off",
        Diagnostic,
        Runtime,
        "Print a GPU memory report (epilog frees, live allocator blocks, driver stats) to stderr at process exit."
    ),
    var!(
        "NSL_GPU_SMOKE",
        Bool,
        "any value (presence)",
        "off (test skipped)",
        Test,
        Test,
        "Set to run the on-GPU PCA RoPE smoke scaffold test, which is skipped by default to keep CI quiet."
    ),
    var!(
        "NSL_GRAD_INTEGRITY",
        Bool,
        "1 only",
        "off (also armed by --grad-integrity)",
        Diagnostic,
        Runtime,
        "Arm the process-exit [grad-integrity] report (worst-case finite/nonzero/missing gradient counts over steps)."
    ),
    var!(
        "NSL_GRAD_INTEGRITY_FAULT",
        Str,
        "double:<idx> or drop:<idx> (parameter index); anything else ignored",
        "off",
        Test,
        Runtime,
        "Test hook: miscount parameter <idx>'s gradient contributions on purpose (double or drop) so the grad-integrity failing arm is reachable."
    ),
    var!(
        "NSL_HOST_PROFILE",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Time host-side runtime entry points (kernel launch, alloc/free, scalar reads, memcpy, tensor alloc) and print [host-profile] reports."
    ),
    var!(
        "NSL_KEEP_TEMP",
        Bool,
        "1 only (exact match)",
        "off (scratch dir deleted)",
        Diagnostic,
        Compile,
        "Retains the per-build compile scratch directory (.o, PTX, intermediates) instead of deleting it after build/run."
    ),
    var!(
        "NSL_KERNEL_LAUNCH_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Print the fused-adapter kernel launch count as [nsl-kernel-count] at exit; NSL_EVENTS gets the JSON twin regardless."
    ),
    var!(
        "NSL_KV_BACKEND",
        Enum,
        "shared_mem / nvlink / rdma / tcp; anything else = shared_mem",
        "auto-detect (RDMA/TCP multi-node, NVLink multi-GPU, else shared_mem)",
        Platform,
        Runtime,
        "Transport for disaggregated-inference KV-cache transfer, overriding hardware auto-detection."
    ),
    var!(
        "NSL_LEGACY_NULL_STREAM",
        Bool,
        "1 only (both sites)",
        "off (per-thread compute stream)",
        Perf,
        Runtime,
        "Kill switch: launch kernels on the legacy NULL stream instead of the per-thread compute stream; also disables CUDA graph capture."
    ),
    var!(
        "NSL_LOCAL_RANK",
        Int,
        "any value, presence only (CLI spawner guard) / integer rank 0..world_size-1, unparsable -> 0, out of range refused by ZeRO init (runtime)",
        "unset: CLI spawns N ranks; runtime assumes rank 0",
        Platform,
        Both,
        "This process's distributed rank; the SPMD spawner sets it per child (its presence stops re-spawning) and the runtime reads it."
    ),
    var!(
        "NSL_MATMUL_BF16",
        Bool,
        "1 only engages; 0 or anything else is a no-op (falls through to TF32/pedantic resolution)",
        "off (TF32 unless NSL_MATMUL_TF32=0 or strict-matmul)",
        Behavior,
        Runtime,
        "Run cuBLAS matmuls with bf16 tensor-core compute (bf16 operand storage); beats NSL_MATMUL_TF32, loses to NSL_MATMUL_PEDANTIC."
    ),
    var!(
        "NSL_MATMUL_BF16_CAST_CACHE",
        Bool,
        "0 disables; anything else on",
        "on (active only under NSL_MATMUL_BF16=1 with RNE rounding and no --cuda-graphs)",
        Perf,
        Runtime,
        "Set to 0 to disable the persistent bf16 weight-image cache and re-cast every GEMM operand from fresh scratch."
    ),
    var!(
        "NSL_MATMUL_BF16_LT",
        Bool,
        "1 only",
        "off (cublasGemmEx DFALT)",
        Behavior,
        Runtime,
        "Issue bf16-storage GEMMs through cublasLt with explicit kernel selection and a real workspace instead of cublasGemmEx (changes bits)."
    ),
    var!(
        "NSL_MATMUL_BF16_LT_TUNE",
        Bool,
        "0 disables; anything else on",
        "on (self-disables under --cuda-graphs, --deterministic, or >1 GiB scratch)",
        Behavior,
        Runtime,
        "Set to 0 to skip timing cublasLt candidates per GEMM shape and take the untimed heuristic[0] kernel instead."
    ),
    var!(
        "NSL_MATMUL_BF16_LT_VERBOSE",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Print each cublasLt autotune result per GEMM shape (winner vs heuristic[0] time, candidate count) to stderr."
    ),
    var!(
        "NSL_MATMUL_BF16_LT_WORKSPACE_MIB",
        Int,
        "integer MiB, clamped to 4096 (unparseable → 64)",
        "64",
        Behavior,
        Runtime,
        "Size in MiB of the process-lifetime cublasLt GEMM workspace; larger values admit split-k/wide-tile kernels (affects kernel choice)."
    ),
    var!(
        "NSL_MATMUL_BF16_MIN_RATIO",
        Float,
        "any f64 (unparsable → 512)",
        "512",
        Behavior,
        Runtime,
        "Min arithmetic-intensity ratio mnk/(a+b elements) for a bf16-mode GEMM to cast operands to bf16; below it the GEMM stays f32/TF32."
    ),
    var!(
        "NSL_MATMUL_BF16_ROUND",
        Enum,
        "sr = stochastic; anything else = RNE",
        "rne",
        Behavior,
        Runtime,
        "Rounding for the bf16 operand cast under NSL_MATMUL_BF16=1: sr = stochastic (fresh dither per launch, blocks graph capture), else RNE."
    ),
    var!(
        "NSL_MATMUL_NO_BATCH_COLLAPSE",
        Bool,
        "1 only (anything else = off)",
        "off (collapse enabled)",
        Behavior,
        Runtime,
        "Set to 1 to keep every batched matmul on the naive nsl_bmm_f32 kernel instead of collapsing to one sgemm / strided-batched cuBLAS GEMM."
    ),
    var!(
        "NSL_MATMUL_PEDANTIC",
        Bool,
        "1 only",
        "off",
        Behavior,
        Runtime,
        "Set to 1 to force cuBLAS pedantic full-FP32 math for every matmul, overriding the TF32 default and NSL_MATMUL_BF16."
    ),
    var!(
        "NSL_MATMUL_TF32",
        Bool,
        "1 = TF32 / 0 = FP32 cores; anything else ignored (falls to default)",
        "TF32 on (pedantic under the strict-matmul build feature)",
        Behavior,
        Runtime,
        "1 enables / 0 disables TF32 tensor-core math for cuBLAS f32 matmuls; other values ignored. NSL_MATMUL_PEDANTIC/BF16=1 take precedence."
    ),
    var!(
        "NSL_MATMUL_TRANSPOSE_VIEWS",
        Bool,
        "1 = OP_T / 0 = materialise; anything else = math-mode default",
        "auto: on under TF32/BF16 math mode, off under FP32-cores/pedantic",
        Perf,
        Runtime,
        "1/0 forces whether a 2-D transposed operand goes to cuBLAS as OP_T instead of being materialised; default on under TF32/BF16, else off."
    ),
    var!(
        "NSL_MEMSTATS",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print a GPU caching-allocator memory summary (peaks, top allocation contexts) to stderr at process exit."
    ),
    var!(
        "NSL_MUON_BATCH_MB",
        Int,
        "integer MiB >= 16 (below 16 or unparsable → 256)",
        "256",
        Perf,
        Runtime,
        "Device workspace budget in MiB (min 16) for the batched Muon Newton-Schulz path; sets how many matrices share one launch."
    ),
    var!(
        "NSL_MUON_BATCH_TF32",
        Bool,
        "0 = off; anything else (incl. unset) = on",
        "on (TF32)",
        Behavior,
        Runtime,
        "Set to 0 to force strict FP32 (no TF32 tensor cores) for the batched Muon Newton-Schulz GEMMs."
    ),
    var!(
        "NSL_MUON_PROF",
        Enum,
        "1 = synced / 2 = enqueue-only; anything else = off",
        "off (0)",
        Diagnostic,
        Runtime,
        "Muon optimizer stage profiler: 1 = synced per-region wall time, 2 = enqueue-only host time; table printed to stderr at exit."
    ),
    var!(
        "NSL_NCCL_LIB_DIR",
        Path,
        "directory path (not validated)",
        "unset: plain -lnccl from the system search path",
        Platform,
        Compile,
        "Directory of a non-system libnccl added as -L and -Wl,-rpath at the final link (only when the compiler has the nccl feature)."
    ),
    var!(
        "NSL_NCCL_TIMEOUT_SECS",
        Int,
        "integer seconds (u64; unparsable → 300)",
        "300",
        Safety,
        Runtime,
        "Seconds to wait for an NCCL collective to complete before aborting every surviving rank (dead-peer watchdog)."
    ),
    var!(
        "NSL_NO_ESCAPE_ANALYSIS",
        Bool,
        "1 only (exact match)",
        "off (analysis runs)",
        Perf,
        Compile,
        "Disables parameter escape analysis so every call argument is assumed to escape; fresh temporaries passed to calls strand (more memory)."
    ),
    var!(
        "NSL_OFFLOAD_PAGEABLE",
        Bool,
        "1 only",
        "off (pinned)",
        Perf,
        Runtime,
        "Set to 1 to hold offloaded optimizer state in pageable heap memory instead of pinned (cuMemAllocHost) host buffers."
    ),
    var!(
        "NSL_OFFLOAD_SYNC",
        Bool,
        "1 only",
        "off (async)",
        Perf,
        Runtime,
        "Set to 1 to force the optimizer-state offload copy-back onto the synchronous path (disables the async DtoH overlap)."
    ),
    var!(
        "NSL_PARAM_PLAN_FAULT",
        Int,
        "integer parameter index (trimmed i64); unparsable = off",
        "off (no fault)",
        Test,
        Runtime,
        "Test hook: corrupts the declared residency plan of parameter index N so the plan-verify FATAL arm is reachable; never set for a real run."
    ),
    var!(
        "NSL_PARAM_PLAN_REPORT",
        Bool,
        "1 only (exact match)",
        "off",
        Diagnostic,
        Compile,
        "Prints the per-parameter placement plan (weight-stream/ZeRO/bf16-sr classes) derived under --layerwise-accum to stderr."
    ),
    var!(
        "NSL_PARTIAL_BCAST_CHILD",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child that issues a partial-batch-broadcast matmul and must be refused."
    ),
    var!(
        "NSL_PASS_TRACE",
        Bool,
        "1 only (exact match)",
        "off",
        Diagnostic,
        Compile,
        "Prints the compiler pass execution trace, per-pass dispositions and the pass-bus report to stderr after compilation."
    ),
    var!(
        "NSL_PCA_DUMP_OUT_ROWS",
        List,
        "comma-separated row indices below seq_len",
        "unset (no dump)",
        Test,
        Test,
        "PCA Tier-A backward correctness test: dump these attention output rows to stderr for kernel debugging."
    ),
    var!(
        "NSL_PCA_DUMP_ROW_STATS",
        List,
        "comma-separated row indices below seq_len",
        "unset (no dump)",
        Test,
        Test,
        "PCA Tier-A backward correctness test: dump saved row_max/row_sum softmax stats for these rows to stderr."
    ),
    var!(
        "NSL_PHASE_TIMING",
        Bool,
        "1 only (exact match)",
        "off",
        Diagnostic,
        Compile,
        "Bakes device-sync + clock probes into the training loop so the program prints per-micro-batch fwd/bwd/optimizer wall times (source-AD only)."
    ),
    var!(
        "NSL_PROFILE_ADJOINT",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Prints a launch-count histogram of the generated backward ops (pre- and post-fusion) plus the const-left binary site count."
    ),
    var!(
        "NSL_PROFILE_KERNELS",
        Bool,
        "present (any value, even empty)",
        "off",
        Diagnostic,
        Runtime,
        "When set (any value), start the GPU kernel profiler at startup and write kernel_profile.json in the working directory at exit."
    ),
    var!(
        "NSL_PROFILE_KERNELS_POOL",
        Int,
        "positive integer (trimmed); 0/unparsable → 4096; capped at 65536",
        "4096",
        Diagnostic,
        Runtime,
        "Number of cuEvent pairs (max 65536) in the kernel profiler's event pool; only used when NSL_PROFILE_KERNELS is set."
    ),
    var!(
        "NSL_PROFILE_MEMORY",
        Bool,
        "present (any value, even empty)",
        "off",
        Diagnostic,
        Runtime,
        "When set (any value), start the block-pool memory profiler at startup and write memory_profile.json at exit."
    ),
    var!(
        "NSL_RDMA_DEVICE",
        Str,
        "any non-empty = RDMA available; empty = unavailable",
        "auto-probe /sys/class/infiniband (Linux), else unavailable",
        Platform,
        Runtime,
        "RDMA NIC name for the disaggregated KV-transfer probe: non-empty forces RDMA available, empty forces unavailable, unset auto-probes."
    ),
    var!(
        "NSL_RECONCILE_DEBUG",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to narrate every add_inplace gradient reconciliation (device/dtype/layout mismatch) instead of warning once."
    ),
    var!(
        "NSL_REMOTE_HOSTS",
        Bool,
        "present (any value)",
        "single-node (unless SLURM_NNODES > 1)",
        Platform,
        Runtime,
        "When set (any value), marks the disaggregated KV-transfer auto-selection as multi-node, preferring RDMA then TCP over NVLink/shm."
    ),
    var!(
        "NSL_RESUME_ALLOW_TRAJECTORY_DRIFT",
        Bool,
        "1 only",
        "off (abort on trajectory drift)",
        Safety,
        Runtime,
        "Set to 1 to let checkpoint_load resume under changed lr/schedule/clip values with an acknowledgment instead of aborting."
    ),
    var!(
        "NSL_ROLE",
        Enum,
        "router / prefill / decode; anything else = router (both sites identical)",
        "router",
        Behavior,
        Runtime,
        "Disaggregated-inference worker role for this process: router (default, monolithic), prefill, or decode."
    ),
    var!(
        "NSL_RUNTIME_LIB_PATH_OVERRIDE",
        Path,
        "existing path to the nsl runtime static library; a missing path is silently ignored",
        "toolchain <exe>/../lib, then the cargo build path",
        Platform,
        Compile,
        "Overrides where the linker finds the nsl runtime static library for the final link."
    ),
    var!(
        "NSL_SCOPE_TRACE",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print tracked/freed/kept tensor counts to stderr at every tensor scope end."
    ),
    var!(
        "NSL_SDPA_FUSED_DEBUG",
        Bool,
        "present (any value, even empty)",
        "off",
        Diagnostic,
        Runtime,
        "When set (any value), print why the fused SDPA forward declined and fell back to the decomposed attention path."
    ),
    var!(
        "NSL_SDPA_FUSED_DISABLE",
        Bool,
        "1 only",
        "off (fused kernel enabled)",
        Behavior,
        Runtime,
        "Set to 1 to disable the fused flash-attention SDPA forward kernel and always run the decomposed attention path."
    ),
    var!(
        "NSL_SDPA_TIER_B_DISABLE",
        Bool,
        "1 only",
        "off (Tier B used when eligible)",
        Behavior,
        Runtime,
        "Set to 1 to force the base segment-masked attention kernel instead of the PCA Tier-B segment-skip PTX variant."
    ),
    var!(
        "NSL_SIMULATED_TP",
        Bool,
        "1 = simulated; anything else = not simulated (refused when world_size > 1); both sites identical",
        "1 (simulated)",
        Platform,
        Runtime,
        "1 (default) selects the CPU shared-memory simulated collective backend; any other value with world size > 1 is refused."
    ),
    var!(
        "NSL_SKIP_CUDA_TESTS",
        Bool,
        "any value (presence)",
        "off (tests probe nsl_cuda_init)",
        Test,
        Test,
        "Set to make every GPU test skip itself instead of probing CUDA; for machines whose driver is present but must not be touched."
    ),
    var!(
        "NSL_SKIP_GPU_DRAIN",
        Bool,
        "1 only",
        "off (drain each step)",
        Perf,
        Runtime,
        "Set to 1 to skip the per-step release of idle caching-allocator GPU memory back to the driver."
    ),
    var!(
        "NSL_SOURCE_AD_ITEM_FOLD",
        Bool,
        "0 disables; anything else or unset = on (cached at first read per process)",
        "on",
        Perf,
        Compile,
        "Kill switch for folding config-tensor .item() reads into tape constants; 0 restores the synchronous device readback (value-identical)."
    ),
    var!(
        "NSL_SR_HIST",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to sample SR-bf16 parameter updates each step and print a |delta| exponent histogram with stall rate at teardown."
    ),
    var!(
        "NSL_STDLIB_PATH",
        Path,
        "existing directory; a non-directory is silently ignored",
        "<exe dir>/stdlib, <exe>/../lib/stdlib, then ./stdlib",
        Platform,
        Compile,
        "Highest-priority root directory searched for stdlib .nsl modules (imports and train-block optimizer modules)."
    ),
    var!(
        "NSL_STRIDED_COPY_RUN",
        Bool,
        "0 = off; anything else = on",
        "on",
        Perf,
        Runtime,
        "Set to 0 to route every strided-view materialisation through the generic copy kernel instead of the run-planned fast path."
    ),
    var!(
        "NSL_SUM_DIM_COALESCED",
        Bool,
        "0 = off; anything else = on",
        "on",
        Behavior,
        Runtime,
        "Set to 0 to disable the coalesced sum_dim kernel route (sequential accumulation, not bit-exact vs the tree-reduce kernel)."
    ),
    var!(
        "NSL_SUM_DIM_LOG",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to log every GPU sum_dim launch's reduce/outer/inner shape and fast-path choice to stderr."
    ),
    var!(
        "NSL_SUM_DIM_SHORT_MAX",
        Int,
        "integer >= 0 (u64; unparsable → 1); 0 disables both fast routes",
        "1",
        Behavior,
        Runtime,
        "Max reduce length (elements) routed to the one-thread-per-output sum_dim kernel; 1 is bit-exact, >1 reassociates, 0 disables both."
    ),
    var!(
        "NSL_SUM_SQ_BLOCKS",
        Int,
        "integer >= 1 (i64; else 256)",
        "256",
        Behavior,
        Runtime,
        "Grid-block cap per tensor for the f64-accumulated gradient sum-of-squares kernel; changes partial-sum order, 1 = single block."
    ),
    var!(
        "NSL_SUM_SQ_CPU",
        Bool,
        "1 only",
        "off (GPU kernel)",
        Behavior,
        Runtime,
        "Set to 1 to compute a GPU tensor's sum of squares with the exact f64 CPU reduction instead of the f32-accumulating kernel."
    ),
    var!(
        "NSL_TAPE_ALLOW_DISCONNECTED",
        Bool,
        "1 only",
        "off (abort)",
        Safety,
        Runtime,
        "Set to 1 to let a strict tape backward proceed when no parameter received a gradient instead of aborting as disconnected."
    ),
    var!(
        "NSL_TAPE_DEBUG_DUMP",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print each parameter's tape key, length and returned gradient length after tape backward."
    ),
    var!(
        "NSL_TENSOR_TO_DESC_NULL_CHILD",
        Bool,
        "present (any value)",
        "unset (child test body returns immediately)",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child of the null-desc C-API test so its body runs; no effect elsewhere."
    ),
    var!(
        "NSL_TF32_DISPATCH_EXPECT_COPIED",
        Bool,
        "1 = expect the strided-copy arm, 0 = expect OP_T; anything else panics",
        "unset (child panics)",
        Test,
        Test,
        "Test harness only: the parent's declaration of which transpose arm the TF32 dispatch child must observe."
    ),
    var!(
        "NSL_TF32_DISPATCH_EXPECT_TF32",
        Bool,
        "1 = expect TF32 drift, 0 = expect full-f32; anything else panics",
        "unset (child panics)",
        Test,
        Test,
        "Test harness only: the parent's declaration of the math mode the TF32 dispatch child must observe."
    ),
    var!(
        "NSL_TF32_DISPATCH_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child of the matmul-dispatch-under-TF32 gate."
    ),
    var!(
        "NSL_TF32_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd timing child of the TF32 matmul mode gate, which prints one PROBE line for the parent."
    ),
    var!(
        "NSL_TP_BARRIER_TIMEOUT_SECS",
        Int,
        "integer seconds (u64; unparsable → 300)",
        "300",
        Safety,
        Runtime,
        "Seconds a rank spins in the shared-memory collective barrier before aborting on a presumed dead peer."
    ),
    var!(
        "NSL_TP_SHM_PATH",
        Path,
        "path; presence alone enables rank-striped device binding, opened as shm when world_size > 1",
        "unset: device 0, single rank; refused (-3) by zero_init if world_size > 1",
        Platform,
        Runtime,
        "Path of the spawner's shared-memory file for multi-rank collectives; required when world size > 1, also enables rank-striped GPU binding."
    ),
    var!(
        "NSL_TRANSPOSE_DEFAULT_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the virgin-process child that resolves the real transpose-views default for the parent."
    ),
    var!(
        "NSL_TRANSPOSE_MEASURE_PROBE",
        Bool,
        "1 only",
        "unset",
        Test,
        Test,
        "Test harness only: marks the re-exec'd child that times OP_T vs materialised transposed operands per shape."
    ),
    var!(
        "NSL_WEIGHTS_PATH",
        Path,
        "path",
        "unset (search next to binary / executable only)",
        Platform,
        Runtime,
        "Explicit path to the .nslweights sidecar, tried after the compiled-binary and executable directories."
    ),
    var!(
        "NSL_WGGO_FORCE_STALE_TABLE",
        Bool,
        "1 only (exact match)",
        "off",
        Safety,
        Compile,
        "Test knob: simulates a rejected WGGO pre-plan whose replan diverges on FASE modes, forcing the stale-mode-table refusal."
    ),
    var!(
        "NSL_WGGO_PREPASS_DEBUG",
        Bool,
        "any value (set, even empty)",
        "off",
        Diagnostic,
        Compile,
        "Dumps the WGGO pre-pass Wengert fingerprint inputs (ops, leaves, hashed count) and the final fingerprint to stderr."
    ),
    var!(
        "NSL_WGRAD_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print fused weight-gradient GEMM vs decomposed-fallback counts to stderr at exit."
    ),
    var!(
        "NSL_WGRAD_DEBUG",
        Bool,
        "1 only",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print operand device/dtype/layout every time the fused weight-gradient GEMM falls back to the decomposed path."
    ),
    var!(
        "NSL_WORLD_SIZE",
        Int,
        "integer (i32; unparsable → 1)",
        "1",
        Platform,
        Runtime,
        "Total number of tensor-parallel ranks for nsl_tp_init; values > 1 require NSL_TP_SHM_PATH."
    ),
    var!(
        "NSL_WRGA_FUSED_CUDA",
        Bool,
        "1 only (read per call, not cached)",
        "off (CPU fallback math)",
        Behavior,
        Runtime,
        "Set to 1 to let WRGA fused adapter ops (LoRA/IA3/GatedLoRA) launch the synthesized CUDA kernel instead of the CPU fallback math."
    ),
    var!(
        "NSL_WRGA_GPU_LAUNCH_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print the count of real GPU fused-adapter kernel launches to stderr at exit (0 = every call fell back to CPU)."
    ),
    var!(
        "NSL_WS_COUNTER",
        Bool,
        "1 only (both sites)",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print weight-stream upload/evict/writeback/traffic counters to stderr at exit."
    ),
    var!(
        "NSL_WS_DECISION_JSON",
        Path,
        "any non-empty path",
        "unset (no file written)",
        Diagnostic,
        Runtime,
        "Path to write the weight-stream residency decision (reserve, must-free, pinned, PCIe traffic) as JSON at exit."
    ),
    var!(
        "NSL_WS_RESIDENT",
        Bool,
        "0 = off; anything else = on",
        "on",
        Perf,
        Runtime,
        "Set to 0 to disable the capacity-aware residency policy and stream every registered weight unconditionally."
    ),
    var!(
        "NSL_WS_RESIDENT_RESERVE_MIB",
        Int,
        "integer MiB (trimmed usize)",
        "50% of total VRAM",
        Perf,
        Runtime,
        "VRAM reserve in MiB the weight-stream residency policy keeps free for activations; weights stream until the reserve is covered."
    ),
    var!(
        "NSL_ZERO_BUCKET_MB",
        Float,
        "MB as f64, fractional ok, negatives clamp to 0; 0 = per-tensor",
        "25",
        Perf,
        Runtime,
        "Bucket cap in MB (fractional accepted) for flattened ZeRO collectives; 0 disables bucketing (one collective per tensor)."
    ),
    var!(
        "NSL_ZERO_COUNTER",
        Bool,
        "1 only (all three sites)",
        "off",
        Diagnostic,
        Runtime,
        "Set to 1 to print ZeRO collective counts at exit and the owned-parameter index set once per process to stderr."
    ),
];
