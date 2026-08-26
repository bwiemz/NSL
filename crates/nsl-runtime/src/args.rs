use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;

use crate::list::{nsl_list_new, nsl_list_push};
use crate::memory::checked_alloc;

/// Wrapper so we can store raw argv pointer in a Mutex (it's only accessed under lock).
struct ArgvPtr(*const *const c_char);
unsafe impl Send for ArgvPtr {}

static ARGS: Mutex<(i32, ArgvPtr)> = Mutex::new((0, ArgvPtr(std::ptr::null())));

#[no_mangle]
pub extern "C" fn nsl_args_init(argc: i32, argv: i64) {
    let mut args = ARGS.lock().unwrap();
    *args = (argc, ArgvPtr(argv as *const *const c_char));

    // Auto-start memory profiler when NSL_PROFILE_MEMORY env var is set.
    if std::env::var("NSL_PROFILE_MEMORY").is_ok() {
        crate::profiling::nsl_profiler_start(0);
    }

    // Auto-start kernel profiler when NSL_PROFILE_KERNELS env var is set.
    if std::env::var("NSL_PROFILE_KERNELS").is_ok() {
        crate::kernel_profiler::nsl_kernel_profiler_start();
    }

    // ELTLS instrumentation: print GPU memory report at program exit when
    // NSL_GPU_MEM_REPORT env var is set. Piggybacks on the C `atexit`
    // function (available on both glibc and MSVCRT/UCRT) so the report
    // fires after the compiled Cranelift `main` returns.
    if std::env::var("NSL_GPU_MEM_REPORT").is_ok() {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_gpu_mem_report_atexit);
        }
    }

    // Item 17: when NSL_EVENTS=<path> is set, every counter reporter below
    // ALSO registers, emitting its machine-readable twin to the events file
    // (crate::events). Each hook re-reads its own env var to decide whether
    // the human stderr line prints, so setting NSL_EVENTS alone produces a
    // silent run with a complete events file — machine output is decoupled
    // from human verbosity flags. NSL_EVENTS does NOT arm [grad-integrity]:
    // arming buys per-step measurement cost, which stays an explicit opt-in.
    let events_on = std::env::var("NSL_EVENTS").map(|v| !v.is_empty()).unwrap_or(false);

    // B.3 Task 5: print fused-adapter kernel-launch count at exit when
    // NSL_KERNEL_LAUNCH_COUNTER=1.  The counter is always live (cheap
    // atomic increment); the env var only gates the report.
    if events_on || std::env::var("NSL_KERNEL_LAUNCH_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(crate::fused_adapter::nsl_fused_adapter_launch_count_atexit);
        }
    }

    // B.3 Task 5.6 hardening: GPU-specific launch counter, enabled when
    // NSL_WRGA_GPU_LAUNCH_COUNTER=1.  Distinguishes real-GPU execution
    // from CPU fallback so tests can assert the fused CUDA path actually
    // fired (not just the math came out right).
    if events_on || std::env::var("NSL_WRGA_GPU_LAUNCH_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(crate::fused_adapter::nsl_fused_adapter_gpu_launch_count_atexit);
        }
    }

    // p9: fused FASE optimizer-step launch count, enabled when
    // NSL_FASE_FUSED_COUNTER=1. Lets the differential gate assert the fused
    // path actually fired (anti-vacuity), mirroring the WRGA counter pattern.
    // The counter is always live; the env var only gates the report.
    if events_on || std::env::var("NSL_FASE_FUSED_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_fase_fused_step_count_atexit);
        }
    }

    // Item 7: fused weight-gradient GEMM vs decomposed-fallback counts,
    // enabled when NSL_WGRAD_COUNTER=1. The fallback inside
    // `nsl_tensor_wgrad_accum` is SILENT by design (a shape the compiler's
    // pre-pass misjudged must degrade to correct-but-slow, not abort), which
    // is exactly why the parity gate needs to see which path actually ran.
    if events_on || std::env::var("NSL_WGRAD_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_wgrad_count_atexit);
        }
    }

    // mfu-fusion C3: fused elementwise-chain launch vs decomposed-replay
    // counts, enabled when NSL_FUSED_EW_COUNTER=1 (or NSL_EVENTS, which gets
    // the machine twin regardless). Same anti-vacuity rationale as the wgrad
    // pair: the replay fallback is warn-once, so a green bit-exactness gate
    // could otherwise be testing the decomposed chain against itself. The
    // report fn IS the atexit hook (it re-reads its own env var for the
    // human line, per the Item-17 pattern above).
    if events_on || std::env::var("NSL_FUSED_EW_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(crate::tensor::fused_chain::nsl_fused_ew_counters_report);
        }
    }

    // D1 (CSLA Stage-2): layerwise window-backward count, enabled when
    // NSL_CSLA_COUNTER=1. Lets the differential gate assert the buffered
    // backward phase actually fired (anti-vacuity), same pattern as above.
    // D2b: weight-stream upload/evict counts, enabled when NSL_WS_COUNTER=1.
    if events_on || std::env::var("NSL_WS_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_weight_stream_count_atexit);
        }
    }

    if events_on || std::env::var("NSL_CSLA_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_csla_window_count_atexit);
        }
    }

    // Milestone B: machine-readable residency-decision record. When
    // NSL_WS_DECISION_JSON=<path>, an atexit writes the policy's inputs and
    // outcome as JSON — the decision existed only as a human stderr line
    // before, so no artifact could bank WHY the policy pinned what it pinned.
    if let Ok(path) = std::env::var("NSL_WS_DECISION_JSON") {
        if !path.is_empty() {
            *WS_DECISION_JSON_PATH.lock().unwrap() = Some(path);
            extern "C" {
                fn atexit(cb: extern "C" fn()) -> i32;
            }
            unsafe {
                atexit(nsl_ws_decision_json_atexit);
            }
        }
    }

    // D3 (ZeRO-1): collective counts, enabled when NSL_ZERO_COUNTER=1. The
    // parity gate asserts EXACT all_reduce/broadcast totals — with
    // replicated data a no-op reduce is otherwise invisible (identical
    // grads make sum/ws == g either way).
    if events_on || std::env::var("NSL_ZERO_COUNTER").ok().as_deref() == Some("1") {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(nsl_zero_count_atexit);
        }
    }

    // P0.3: gradient-integrity report. The --grad-integrity flag arms this
    // directly from the emitted train setup (nsl_grad_integrity_arm), so it
    // works with no env var; NSL_GRAD_INTEGRITY=1 also arms it here (the arm is
    // Once-guarded, so the two paths never double-register). Prints the
    // worst-case-over-steps snapshot so a gate can prove every trainable
    // parameter received a finite, mostly-nonzero gradient on EVERY step —
    // catching the #396 silent-drop at runtime.
    if std::env::var("NSL_GRAD_INTEGRITY").ok().as_deref() == Some("1") {
        crate::grad_integrity::nsl_grad_integrity_arm();
    }
}

extern "C" fn nsl_zero_count_atexit() {
    let (rank, ws, all_reduce, broadcast, optim_elems, bucket_members, repl_optim_elems) =
        crate::zero::zero_counter_snapshot();
    // EVERY rank prints this line to the SAME inherited stderr pipe, often
    // exiting simultaneously. `eprintln!` issues one write(2) per format
    // fragment, so concurrent ranks can TEAR each other's lines mid-string
    // (observed: "[zero] ws=2[zero] ws= rank=0 ..." — the SPMD gates then
    // flake on the missing line). Format first, write once: a single
    // write_all under PIPE_BUF is atomic on POSIX pipes.
    let rs = crate::zero::ZERO_REDUCE_SCATTER_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let ag = crate::zero::ZERO_ALL_GATHER_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    // `repl_optim_elems` is APPENDED, never inserted, and that ordering is
    // load-bearing: `optim_elems=` is a SUBSTRING of `repl_optim_elems=`, so a
    // parser that matches by bare substring reads the right field only while
    // `optim_elems` appears FIRST on the line.
    //
    // The three live parsers survive it two different ways, neither of which
    // is "a trailing space":
    //   * zero_spmd_gate.rs and zero_gpu_collectives_gate.rs split on
    //     whitespace and `strip_prefix("optim_elems=")` — token-exact, so
    //     `repl_optim_elems=8320` simply is not a match. Order-independent.
    //   * zero3_gate.rs anchors with a LEADING space (` optim_elems=`), which
    //     `repl_optim_elems=` cannot contain (the preceding char is `_`).
    // A bare-substring parser would be order-dependent; don't add one.
    //
    // Keep new fields at the end regardless — the historical
    // `ends_with("reduce_scatter=6")` parser broke the day item 11 appended
    // `all_gather=`, and terminal anchoring is never safe on this line.
    // Item 17: the structured twin is built from the SAME snapshot as the
    // stderr line, so the two renderings cannot disagree about values. The
    // stderr line stays gated by NSL_ZERO_COUNTER (this hook also registers
    // under NSL_EVENTS alone).
    crate::events::emit(
        "zero_counters",
        None,
        &[
            ("ws", crate::events::i(ws)),
            ("rank", crate::events::i(rank)),
            ("all_reduce", crate::events::u(all_reduce)),
            ("broadcast", crate::events::u(broadcast)),
            ("optim_elems", crate::events::u(optim_elems)),
            ("bucket_members", crate::events::u(bucket_members)),
            ("reduce_scatter", crate::events::u(rs)),
            ("all_gather", crate::events::u(ag)),
            ("repl_optim_elems", crate::events::u(repl_optim_elems)),
        ],
    );
    if std::env::var("NSL_ZERO_COUNTER").ok().as_deref() != Some("1") {
        return;
    }
    let line = format!(
        "[zero] ws={ws} rank={rank} all_reduce={all_reduce} broadcast={broadcast} optim_elems={optim_elems} bucket_members={bucket_members} reduce_scatter={rs} all_gather={ag} repl_optim_elems={repl_optim_elems}\n"
    );
    use std::io::Write;
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
}

/// Target path for the residency-decision JSON, set once at reporter setup.
/// A static because `atexit` callbacks take no arguments.
static WS_DECISION_JSON_PATH: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

extern "C" fn nsl_ws_decision_json_atexit() {
    let Some(path) = WS_DECISION_JSON_PATH.lock().unwrap().clone() else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    let ld = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed);
    let total = ld(&crate::weight_stream::WS_PLAN_TOTAL);
    let decided = total != u64::MAX;
    // Plan fields are meaningful only after a decision ran; emit them as
    // JSON null otherwise so a consumer cannot mistake "never decided" for
    // "reserve was 0". Hand-built JSON — the repo convention (checkpoint.rs).
    let plan_fields = if decided {
        format!(
            "\"reserve_bytes\":{},\"must_free_bytes\":{},\"free_at_decision_bytes\":{},\
             \"total_vram_bytes\":{}",
            ld(&crate::weight_stream::WS_PLAN_RESERVE),
            ld(&crate::weight_stream::WS_PLAN_MUST_FREE),
            ld(&crate::weight_stream::WS_PLAN_FREE_AT_DECISION),
            total,
        )
    } else {
        "\"reserve_bytes\":null,\"must_free_bytes\":null,\"free_at_decision_bytes\":null,\
         \"total_vram_bytes\":null"
            .to_string()
    };
    let json = format!(
        "{{\"decided\":{},\"registered\":{},\"pinned\":{},\"pinned_bytes\":{},\
         \"streamed_bytes\":{},\"h2d_traffic_bytes\":{},\"d2h_traffic_bytes\":{},{}}}\n",
        decided,
        ld(&crate::weight_stream::WS_REGISTERED),
        ld(&crate::weight_stream::WS_PINNED),
        ld(&crate::weight_stream::WS_PINNED_BYTES),
        ld(&crate::weight_stream::WS_STREAMED_BYTES),
        ld(&crate::weight_stream::WS_H2D_TRAFFIC_BYTES),
        ld(&crate::weight_stream::WS_D2H_TRAFFIC_BYTES),
        plan_fields,
    );
    if let Err(e) = std::fs::write(&path, json) {
        eprintln!("[weight-stream] warning: could not write decision json to {path}: {e}");
    }
}

extern "C" fn nsl_weight_stream_count_atexit() {
    // Item 10: pack_uploads/pack_evicts count contiguous-layer-pack transfers
    // (arena mode). In per-param mode both are 0; in arena mode they are far
    // below uploads/evicts — the batching evidence the arena gate asserts.
    // h2d_bytes/d2h_bytes are APPENDED (Milestone B traffic counters, exact
    // bytes) — the zero-line lore above applies here too: existing gates
    // strip_prefix-then-parse leading fields, so new fields go at the END.
    //
    // Item 17: ONE snapshot feeds both the stderr line and the structured
    // event, so the two renderings cannot disagree about values. The stderr
    // line stays gated by NSL_WS_COUNTER (this hook also registers under
    // NSL_EVENTS alone). This event retires the most brittle consumer in the
    // repo: a nine-capture-group python regex that pinned this line's every
    // field, order, and colon punctuation.
    let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
    let uploads = ld(&crate::weight_stream::WS_UPLOADS);
    let evicts = ld(&crate::weight_stream::WS_EVICTS);
    let writeback = ld(&crate::weight_stream::WS_EVICTS_WB);
    let registered = ld(&crate::weight_stream::WS_REGISTERED);
    let ptr_moves = ld(&crate::weight_stream::WS_PTR_MOVES);
    let pack_uploads = ld(&crate::weight_stream::WS_PACK_UPLOADS);
    let pack_evicts = ld(&crate::weight_stream::WS_PACK_EVICTS);
    let prefetches = ld(&crate::weight_stream::WS_PREFETCHES);
    let async_wb = ld(&crate::weight_stream::WS_ASYNC_WB);
    let h2d_bytes = ld(&crate::weight_stream::WS_H2D_TRAFFIC_BYTES);
    let d2h_bytes = ld(&crate::weight_stream::WS_D2H_TRAFFIC_BYTES);
    crate::events::emit(
        "weight_stream_counters",
        None,
        &[
            ("uploads", crate::events::u(uploads)),
            ("evicts", crate::events::u(evicts)),
            ("writeback", crate::events::u(writeback)),
            ("registered", crate::events::u(registered)),
            ("ptr_moves", crate::events::u(ptr_moves)),
            ("pack_uploads", crate::events::u(pack_uploads)),
            ("pack_evicts", crate::events::u(pack_evicts)),
            ("prefetches", crate::events::u(prefetches)),
            ("async_wb", crate::events::u(async_wb)),
            ("h2d_bytes", crate::events::u(h2d_bytes)),
            ("d2h_bytes", crate::events::u(d2h_bytes)),
        ],
    );
    let stderr_on = std::env::var("NSL_WS_COUNTER").ok().as_deref() == Some("1");
    if stderr_on {
        eprintln!(
            "[weight-stream] uploads: {uploads} evicts: {evicts} writeback: {writeback} \
             registered: {registered} ptr_moves: {ptr_moves} pack_uploads: {pack_uploads} \
             pack_evicts: {pack_evicts} prefetches: {prefetches} async_wb: {async_wb} \
             h2d_bytes: {h2d_bytes} d2h_bytes: {d2h_bytes}"
        );
    }
    // Capacity-aware residency posture. Its OWN line, never appended fields:
    // the counters line above is consumed by strip_prefix-then-parse gates.
    // `pinned == registered` means no parameter bytes crossed PCIe at all.
    #[cfg(feature = "cuda")]
    {
        let pinned = ld(&crate::weight_stream::WS_PINNED);
        if registered > 0 {
            // EXACT bytes, not just MiB: a small model's whole parameter set
            // is a few hundred KiB, and `{:.0}` MiB rounds that to "0" — a
            // counter that reads zero on exactly the fixtures the gates use
            // is worse than no counter (caught by the residency gate).
            // MiB is kept alongside for humans reading a 1B run.
            let pinned_b = ld(&crate::weight_stream::WS_PINNED_BYTES);
            let streamed_b = ld(&crate::weight_stream::WS_STREAMED_BYTES);
            let summary = crate::weight_stream::residency_summary().unwrap_or_default();
            crate::events::emit(
                "weight_stream_residency",
                None,
                &[
                    ("pinned", crate::events::u(pinned)),
                    ("registered", crate::events::u(registered)),
                    ("pinned_bytes", crate::events::u(pinned_b)),
                    ("streamed_bytes", crate::events::u(streamed_b)),
                    ("summary", serde_json::Value::from(summary.clone())),
                ],
            );
            if stderr_on {
                let mib = |b: u64| b as f64 / 1048576.0;
                let line = format!(
                    "[weight-stream] residency: pinned={} of {} param(s) \
                     pinned_bytes={} streamed_bytes={} \
                     pinned_mib={:.1} streamed_mib={:.1} {}\n",
                    pinned,
                    registered,
                    pinned_b,
                    streamed_b,
                    mib(pinned_b),
                    mib(streamed_b),
                    summary,
                );
                use std::io::Write;
                let _ = std::io::stderr().lock().write_all(line.as_bytes());
            }
        }
    }
}

extern "C" fn nsl_csla_window_count_atexit() {
    let phases = crate::csla_stat::nsl_csla_window_count();
    crate::events::emit(
        "csla_counters",
        None,
        &[("window_backward_phases", crate::events::i(phases))],
    );
    if std::env::var("NSL_CSLA_COUNTER").ok().as_deref() == Some("1") {
        eprintln!("[csla] window backward phases: {phases}");
    }
}

extern "C" fn nsl_fase_fused_step_count_atexit() {
    // Each counter gets its OWN line: the launches line is consumed by
    // strip_prefix-then-parse gates, so appending a field there would break
    // them (the append-breaks-terminal-anchors class).
    //
    // Formatted once and written once, for the reason the `[zero]` reporter
    // documents: under `--devices N` every rank prints to the SAME inherited
    // stderr pipe, often exiting together, and `eprintln!` issues one write(2)
    // per format fragment — so concurrent ranks TEAR each other's lines
    // mid-string and the parsing gate flakes on a missing field. This report
    // became multi-rank the moment the batched optimizer step started firing
    // under `--zero-stage`.
    // The multi-params line carries `rank=`: with `--devices N` every rank
    // writes one, and a gate pairing them by ARRIVAL ORDER against per-rank
    // facts (which parameters that rank owns) would silently cross them
    // whenever the ranks happen to exit in the other order.
    let (rank, _ws, _ar, _bc, _oe, _bm, _repl) = crate::zero::zero_counter_snapshot();
    let launches = crate::fase_step::nsl_fase_fused_step_count();
    let table_builds = crate::fase_step::nsl_fase_blk_table_builds();
    let batched = crate::fase_step::nsl_fase_multi_batched_params();
    let fallback = crate::fase_step::nsl_fase_multi_fallback_params();
    crate::events::emit(
        "fase_fused_counters",
        None,
        &[
            ("rank", crate::events::i(rank)),
            ("fused_step_launches", crate::events::i(launches)),
            ("block_table_builds", crate::events::i(table_builds)),
            ("multi_batched", crate::events::i(batched)),
            ("multi_fallback", crate::events::i(fallback)),
        ],
    );
    if std::env::var("NSL_FASE_FUSED_COUNTER").ok().as_deref() != Some("1") {
        return;
    }
    let report = format!(
        "[fase-fused] optimizer fused-step launches: {launches}\n\
         [fase-fused] block-table builds: {table_builds}\n\
         [fase-fused] multi params: rank={rank} batched={batched} fallback={fallback}\n",
    );
    use std::io::Write;
    let _ = std::io::stderr().lock().write_all(report.as_bytes());
}

extern "C" fn nsl_wgrad_count_atexit() {
    let fused = crate::tensor::arithmetic::nsl_wgrad_fused_count();
    let fallback = crate::tensor::arithmetic::nsl_wgrad_fallback_count();
    crate::events::emit(
        "wgrad_counters",
        None,
        &[
            ("fused_gemm", crate::events::i(fused)),
            ("decomposed_fallback", crate::events::i(fallback)),
        ],
    );
    if std::env::var("NSL_WGRAD_COUNTER").ok().as_deref() == Some("1") {
        eprintln!("[wgrad-accum] fused GEMM: {fused}, decomposed fallback: {fallback}");
    }
}

extern "C" fn nsl_gpu_mem_report_atexit() {
    let epilog_frees = crate::tensor::nsl_debug_epilog_free_count();
    eprintln!("--- GPU memory report ---");
    eprintln!("epilog_frees_total: {epilog_frees}");
    #[cfg(feature = "cuda")]
    {
        // Print live caching-allocator block summary (step=0 so it prints).
        crate::tensor::nsl_debug_gpu_alloc_summary(0);
        // Print driver + allocator stats (step=0 so it prints).
        crate::tensor::nsl_debug_gpu_mem(0);
    }
}

#[no_mangle]
pub extern "C" fn nsl_args() -> i64 {
    let args = ARGS.lock().unwrap();
    let (argc, ref argv_wrapper) = *args;
    let argv = argv_wrapper.0;
    let list = nsl_list_new();
    for i in 0..argc {
        let arg = unsafe { *argv.add(i as usize) };
        let cstr = unsafe { CStr::from_ptr(arg) };
        let bytes = cstr.to_bytes_with_nul();
        let copy = checked_alloc(bytes.len());
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy, bytes.len());
        }
        nsl_list_push(list, copy as i64);
    }
    list
}
