//! CSLA Stage-2 (D1a) — window-buffered schedule parity gates.
//!
//! `--layerwise-accum` restructures the FASE-Deferred training loop: the N
//! micro-batches of each accumulation window run their forwards first (saving
//! only the adjoint-read primal values + batch dicts), then the backward
//! phase replays the whole window through a runtime loop over the buffered
//! micro-batches. The transformation must be BIT-EXACT vs the interleaved
//! `--checkpoint-blocks` baseline: same kernels, same inputs, same
//! per-parameter accumulation order (micro-batch ascending), so the loss
//! stream and the saved model bytes must match at full precision.
//!
//! Gate discipline (campaign standard):
//!   - baseline arm FIRST, run TWICE (run-to-run determinism probe);
//!   - anti-vacuity via the NSL_CSLA_COUNTER=1 atexit report — the layerwise
//!     arm must report EXACTLY the expected number of window-backward phases
//!     and the baseline must report 0, so a silently-inert flag can't pass;
//!   - CCR must not have declined ("running without checkpointing");
//!   - refusal cases assert the loud errors, not silent fallbacks.
//!
//! Window-semantics coverage:
//!   - `csla_parity_ffn_cpu`: 13 micro-batches, N=2, epochs=1 -> 6 windows +
//!     a TRAILING PARTIAL WINDOW (teardown sweep; tail must not step);
//!   - `csla_parity_ffn_cpu_epoch_straddle`: epochs=2 -> window 7 spans the
//!     epoch boundary (global step counter, DataLoader reset between);
//!   - GPU twins (#[ignore]) + the packed-GQA composition (per-micro-batch
//!     packing-metadata re-install on the replay path).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src(name: &str) -> String {
    std::fs::read_to_string(repo_root().join("crates/nsl-cli/tests/fixtures").join(name))
        .unwrap_or_else(|_| panic!("{name} fixture missing"))
}

fn program(fixture: &str, gpu: bool, save_path: &Path, rewrites: &[(&str, &str)]) -> String {
    let mut src = fixture_src(fixture);
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    assert!(src.contains("CSLA_SAVE_PATH"));
    src = src.replace(
        "CSLA_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    );
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing");
        src = src.replace(from, to);
    }
    src
}

struct RunOutput {
    loss_stream: String,
    stdout: String,
    stderr: String,
    success: bool,
}

fn run_program(source: &str, tag: &str, cuda: bool, deterministic: bool, extra_args: &[&str]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_csla_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("csla_gate.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-q");
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"]);
    if deterministic {
        cmd.arg("--deterministic");
    }
    cmd.args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Anti-vacuity counter report on every arm; deterministic embedding
        // scatter for the CPU cases (harmless there, load-bearing on GPU
        // EmbedCpu-style runs).
        .env("NSL_CSLA_COUNTER", "1")
        .env("NSL_WS_COUNTER", "1")
        .env("NSL_EMBEDDING_BWD_CPU", "1");
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    let mut loss_stream = String::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            l if in_stream => {
                loss_stream.push_str(l);
                loss_stream.push('\n');
            }
            _ => {}
        }
    }
    RunOutput {
        loss_stream,
        stdout,
        stderr,
        success: output.status.success(),
    }
}

/// Parse the number between SDPA_FUSED_BEGIN/END markers (bare integer).
fn sdpa_fused_count(out: &RunOutput) -> Option<i64> {
    let mut in_block = false;
    for line in out.stdout.lines() {
        match line.trim() {
            "SDPA_FUSED_BEGIN" => in_block = true,
            "SDPA_FUSED_END" => in_block = false,
            l if in_block => {
                let cleaned = l.trim().trim_start_matches("tensor([").trim_end_matches("])");
                if let Ok(v) = cleaned.parse::<f64>() {
                    return Some(v as i64);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse `[weight-stream] uploads: N evicts: M writeback: W`.
fn ws_counts(stderr: &str) -> (i64, i64, i64) {
    let rest = stderr
        .lines()
        .find_map(|l| l.strip_prefix("[weight-stream] uploads: "))
        .expect("NSL_WS_COUNTER report missing");
    let mut it = rest.split(" evicts: ");
    let uploads: i64 = it.next().unwrap().trim().parse().unwrap();
    let mut it2 = it.next().unwrap().split(" writeback: ");
    let evicts: i64 = it2.next().unwrap().trim().parse().unwrap();
    // The line now carries trailing " registered: N ptr_moves: M" fields, so
    // take only the first whitespace-delimited token as the writeback count.
    let writeback: i64 = it2
        .next()
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    (uploads, evicts, writeback)
}

/// Parse the `registered: R` and `ptr_moves: M` suffix of the weight-stream
/// counter line (added by the P0.1 view gate: per-param pointer-movement
/// evidence — a registered param had its device storage freed+reallocated; a
/// view-rooted excluded param never registers, so its pointer stays put).
fn ws_ptr_counts(stderr: &str) -> (i64, i64) {
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[weight-stream] uploads: "))
        .expect("NSL_WS_COUNTER report missing");
    let reg = line
        .split("registered: ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("registered count missing");
    let moves = line
        .split("ptr_moves: ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("ptr_moves count missing");
    (reg, moves)
}

/// Parse the bare integer between `<TAG>_BEGIN` / `<TAG>_END` stdout markers.
fn marker_i64(stdout: &str, tag: &str) -> Option<i64> {
    let begin = format!("{tag}_BEGIN");
    let end = format!("{tag}_END");
    let mut in_block = false;
    for line in stdout.lines() {
        let t = line.trim();
        if t == begin {
            in_block = true;
        } else if t == end {
            in_block = false;
        } else if in_block {
            if let Ok(v) = t.parse::<f64>() {
                return Some(v as i64);
            }
        }
    }
    None
}

/// Parse the `[csla] window backward phases: N` atexit report.
fn window_phase_count(stderr: &str) -> Option<i64> {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("[csla] window backward phases: "))
        .and_then(|n| n.trim().parse::<i64>().ok())
}

/// The parity core: baseline (`--checkpoint-blocks`) twice as a determinism
/// probe, then the layerwise arm; bit-exact loss stream + model bytes when
/// the environment is deterministic; exact window-phase count both arms.
fn parity_case(
    fixture: &str,
    cuda: bool,
    deterministic: bool,
    tag: &str,
    rewrites: &[(&str, &str)],
    common_args: &[&str],
    expected_windows: i64,
) {
    parity_case_with_schedule(
        fixture,
        cuda,
        deterministic,
        tag,
        rewrites,
        common_args,
        expected_windows,
        None,
    );
}

/// `expected_schedule`: the exact `[csla] layer-major schedule: ...` stderr
/// line (D1b anti-vacuity — proves the k-range layer-major shape engaged,
/// not a degenerate all-epilogue schedule that would replicate D1a).
#[allow(clippy::too_many_arguments)]
fn parity_case_with_schedule(
    fixture: &str,
    cuda: bool,
    deterministic: bool,
    tag: &str,
    rewrites: &[(&str, &str)],
    common_args: &[&str],
    expected_windows: i64,
    expected_schedule: Option<&str>,
) -> (RunOutput, RunOutput) {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_saves_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base_a.nslm");
    let save_b = tmp.join("base_b.nslm");
    let save_c = tmp.join("csla.nslm");

    // 1. Baseline determinism probe (two identical runs, baseline arm first).
    let base_a = run_program(
        &program(fixture, cuda, &save_a, rewrites),
        &format!("{tag}_a"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks"], common_args].concat(),
    );
    assert!(base_a.success, "baseline run A failed:\n{}", base_a.stderr);
    let base_b = run_program(
        &program(fixture, cuda, &save_b, rewrites),
        &format!("{tag}_b"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks"], common_args].concat(),
    );
    assert!(base_b.success, "baseline run B failed:\n{}", base_b.stderr);
    let bytes_a = std::fs::read(&save_a).expect("baseline A model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("baseline B model_save missing");
    let env_deterministic = bytes_a == bytes_b && base_a.loss_stream == base_b.loss_stream;

    // Baseline must never enter the window backward.
    assert_eq!(
        window_phase_count(&base_a.stderr),
        Some(0),
        "baseline arm reported csla window phases != 0:\n{}",
        base_a.stderr
    );

    // 2. Layerwise arm.
    let csla = run_program(
        &program(fixture, cuda, &save_c, rewrites),
        &format!("{tag}_c"),
        cuda,
        deterministic,
        &[&["--checkpoint-blocks", "--layerwise-accum"], common_args].concat(),
    );
    assert!(csla.success, "--layerwise-accum run failed:\n{}", csla.stderr);
    assert!(
        !csla.stderr.contains("running without checkpointing"),
        "CCR declined under --layerwise-accum (should have been a hard error):\n{}",
        csla.stderr
    );
    // Anti-vacuity: the window backward fired exactly once per complete
    // accumulation window (trailing partial windows never fire, matching the
    // baseline's discarded m_partial tail).
    assert_eq!(
        window_phase_count(&csla.stderr),
        Some(expected_windows),
        "csla arm window-phase count != expected {expected_windows}:\n{}",
        csla.stderr
    );
    // D1b anti-vacuity: the layer-major schedule shape itself.
    if let Some(sched) = expected_schedule {
        assert!(
            csla.stderr.contains(sched),
            "expected layer-major schedule line '{sched}' in:\n{}",
            csla.stderr
        );
    }
    let bytes_c = std::fs::read(&save_c).expect("csla model_save missing");

    if env_deterministic {
        assert_eq!(
            base_a.loss_stream, csla.loss_stream,
            "loss stream diverged under --layerwise-accum (must be bit-exact)"
        );
        assert!(
            bytes_a == bytes_c,
            "saved model bytes diverged under --layerwise-accum (must be bit-exact)"
        );
    } else {
        eprintln!(
            "[csla-parity] WARNING: baseline is not run-to-run deterministic \
             in this environment ({tag}); falling back to loss-prefix comparison"
        );
        let first_a = base_a.loss_stream.lines().next().unwrap_or("");
        let first_c = csla.loss_stream.lines().next().unwrap_or("");
        assert_eq!(
            first_a, first_c,
            "first loss (pure forward on deterministic init) must still match"
        );
    }
    (base_a, csla)
}

/// CPU gate — 6 complete windows + a trailing partial window (13 micro-
/// batches, N=2): exercises the save/replay machinery AND the teardown sweep.
#[test]
fn csla_parity_ffn_cpu() {
    parity_case_with_schedule(
        "csla_layerwise_ffn.nsl",
        false,
        false,
        "ffn_cpu",
        &[],
        &[],
        6,
        Some("[csla] layer-major schedule: 3 ranges, 6 layer-grouped params, 2 epilogue params"),
    );
}

/// CPU gate — loss tensor READ BY THE ADJOINT (review M2): tanh's backward
/// reads its own output, so wrapping the loss in tanh() makes the loss VarId
/// a window-buffered import and engages the loss_slot machinery (b<N-1 skip
/// in the replay loop + the should_step-conditional per-iteration free) that
/// the cross_entropy fixtures never touch.
#[test]
fn csla_parity_ffn_cpu_loss_read_by_adjoint() {
    parity_case(
        "csla_layerwise_ffn.nsl",
        false,
        false,
        "ffn_loss_read",
        &[(
            "let loss = cross_entropy(flat_logits, flat_labels)",
            "let loss = tanh(cross_entropy(flat_logits, flat_labels))",
        )],
        &[],
        6,
    );
}

/// CPU gate — epochs=2: 26 micro-batches -> 13 windows, one of which spans
/// the epoch boundary (buffered saves from epoch 1's tail + epoch 2's head).
#[test]
fn csla_parity_ffn_cpu_epoch_straddle() {
    parity_case(
        "csla_layerwise_ffn.nsl",
        false,
        false,
        "ffn_straddle",
        &[("epochs=1", "epochs=2")],
        &[],
        13,
    );
}

/// GPU twin of the FFN gate under `--deterministic` (M46 kernels).
/// `cargo test -p nsl-cli --features cuda --test csla_layerwise_gate -- --ignored`
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_ffn_gpu() {
    parity_case_with_schedule(
        "csla_layerwise_ffn.nsl",
        true,
        true,
        "ffn_gpu",
        &[],
        &[],
        6,
        Some("[csla] layer-major schedule: 3 ranges, 6 layer-grouped params, 2 epilogue params"),
    );
}

/// Packed-GQA composition, CPU: buffered packed-batch dict state
/// (segment_ids / position_ids / input_ids per micro-batch), the per-b
/// packing-registry re-install, and the packed EmbeddingBackward reading
/// buffered input_ids at replay time. CPU because attention-on-CUDA is
/// refused in D1a (review H2: the fused SDPA dispatch's logsumexp rides an
/// SSA side-band the replay cannot see) — the GPU arm is the refusal case
/// in csla_refusals; the D1b tape-carry restores GPU attention parity.
#[test]
fn csla_parity_packed_gqa_cpu() {
    parity_case_with_schedule(
        "csla_layerwise_packed_gqa.nsl",
        false,
        false,
        "packed_cpu",
        &[],
        // CPU compile target: the default "cuda" target emits the fused
        // SDPA dispatch table even for CPU-placed models, which the H2
        // refusal (correctly) rejects. Both arms share the target.
        &["--target", "cpu"],
        8,
        Some("[csla] layer-major schedule: 3 ranges, 16 layer-grouped params, 2 epilogue params"),
    );
}

/// GPU packed-GQA parity: the fused SDPA dispatch is emitted (head_dim=32
/// is in the variant table); the GQA form's expanded (strided) K/V declines
/// the fused launch at RUNTIME on both arms — parity of the
/// dispatch/decline plumbing. Under the Block checkpoint policy the SDPA
/// out is a recompute victim, so the LSE carry is INERT here (0 slots,
/// asserted) — the replay's recompute clone re-establishes the aux
/// side-band locally.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_packed_gqa_gpu() {
    let (base, csla) = parity_case_with_schedule(
        "csla_layerwise_packed_gqa.nsl",
        true,
        true,
        "packed_gpu",
        &[],
        &[],
        8,
        Some("[csla] layer-major schedule: 3 ranges, 16 layer-grouped params, 2 epilogue params"),
    );
    assert!(
        csla.stderr.contains("[csla] lse tape-carry: 0 slots"),
        "expected the inert-carry marker under the Block policy:\n{}",
        csla.stderr
    );
    let base_n = sdpa_fused_count(&base).expect("SDPA_FUSED markers missing (baseline)");
    let csla_n = sdpa_fused_count(&csla).expect("SDPA_FUSED markers missing (csla)");
    assert_eq!(
        base_n, csla_n,
        "fused-launch counts diverged between arms (base {base_n} vs csla {csla_n})"
    );
}

/// GPU packed-MHA parity, Block policy — the RECOMPUTE path: kv_heads ==
/// heads gives contiguous K/V so the fused forward FIRES, but under
/// --checkpoint-blocks the SDPA out is a recompute victim: the replay's
/// spliced clone RE-LAUNCHES the fused forward per micro-batch and
/// re-establishes the Value-keyed aux locally (carry inert — 0 slots,
/// asserted). Launch-count equality across arms is structural (each clone
/// replays once per micro-batch = the baseline's once per iteration).
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_packed_mha_gpu_fused() {
    let (base, csla) = parity_case_with_schedule(
        "csla_layerwise_packed_gqa.nsl",
        true,
        true,
        "packed_mha_gpu",
        &[(
            "GroupedQueryAttention(64, 2, 1, 32, 0.0)",
            "GroupedQueryAttention(64, 2, 2, 32, 0.0)",
        )],
        &[],
        8,
        Some("[csla] layer-major schedule: 3 ranges, 16 layer-grouped params, 2 epilogue params"),
    );
    assert!(
        csla.stderr.contains("[csla] lse tape-carry: 0 slots"),
        "expected the inert-carry marker under the Block policy:\n{}",
        csla.stderr
    );
    let base_n = sdpa_fused_count(&base).expect("SDPA_FUSED markers missing (baseline)");
    let csla_n = sdpa_fused_count(&csla).expect("SDPA_FUSED markers missing (csla)");
    assert!(
        base_n > 0,
        "fused SDPA forward never fired on the MHA baseline — the gate is vacuous"
    );
    assert_eq!(
        base_n, csla_n,
        "fused-launch counts diverged between arms (base {base_n} vs csla {csla_n})"
    );
}

/// GPU packed-MHA parity, SELECTIVE policy — the LIVE tape-carry (review
/// F1): --checkpoint-selective SAVES the SDPA outs, so they are adjoint
/// imports, the save phase buffers the aux entries (2 slots, asserted
/// exactly), and the replay re-binds each micro-batch's REAL forward-saved
/// logsumexp into the aux side-band feeding the fused phase-2 backward.
/// This is the configuration the carry machinery exists for.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_packed_mha_gpu_selective_carry() {
    let (base, csla) = parity_case_with_schedule(
        "csla_layerwise_packed_gqa.nsl",
        true,
        true,
        "packed_mha_sel_gpu",
        &[(
            "GroupedQueryAttention(64, 2, 1, 32, 0.0)",
            "GroupedQueryAttention(64, 2, 2, 32, 0.0)",
        )],
        &["--checkpoint-selective"],
        8,
        Some("[csla] layer-major schedule: 3 ranges, 16 layer-grouped params, 2 epilogue params"),
    );
    assert!(
        csla.stderr.contains("[csla] lse tape-carry: 2 slots"),
        "expected the LIVE carry marker (2 buffered LSE slots) under the \
         Selective policy:\n{}",
        csla.stderr
    );
    let base_n = sdpa_fused_count(&base).expect("SDPA_FUSED markers missing (baseline)");
    let csla_n = sdpa_fused_count(&csla).expect("SDPA_FUSED markers missing (csla)");
    assert!(
        base_n > 0,
        "fused SDPA forward never fired on the MHA baseline — the gate is vacuous"
    );
    assert_eq!(
        base_n, csla_n,
        "fused-launch counts diverged between arms (base {base_n} vs csla {csla_n})"
    );
}

/// D2a m/v-streaming parity (GPU): `--optim-state-offload` composes with the
/// layerwise schedule — m/v allocate host-pinned and stage through the
/// wrap_offload envelope AT THE PER-LAYER UPDATE SITES (one layer's staged
/// tensors in flight at a time, drained per group update). Baseline arm =
/// the P3 interleaved offload path; staging is byte-preserving, so parity
/// stays bit-exact.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_ffn_gpu_mv_streaming() {
    parity_case_with_schedule(
        "csla_layerwise_ffn.nsl",
        true,
        true,
        "ffn_gpu_mv",
        &[],
        &["--optim-state-offload"],
        6,
        Some("[csla] layer-major schedule: 3 ranges, 6 layer-grouped params, 2 epilogue params"),
    );
}

/// D2b whole-loop weight-streaming parity (GPU): layer-grouped params live
/// in pinned host mirrors and hold device buffers only inside their
/// brackets — the forward (sliced per CCR segment) uploads each block
/// before its segment and evicts read-only after its last primal touch;
/// the window backward re-uploads per replay range and writes back after
/// that layer's update; teardown restores for model_save. Streaming is
/// byte-preserving, so the two arms (csla vs csla + --weight-stream) must
/// be BIT-EXACT; anti-vacuity = exact transfer counters (incl. the
/// writeback subset), the placement-pinned schedule line, and the
/// weights-at-peak bracket probe.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_weight_stream_parity_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_ws_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("csla.nslm");
    let save_b = tmp.join("csla_ws.nslm");

    let base = run_program(
        &program("csla_layerwise_ffn.nsl", true, &save_a, &[]),
        "ws_a",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(base.success, "csla arm failed:\n{}", base.stderr);
    let ws = run_program(
        &program("csla_layerwise_ffn.nsl", true, &save_b, &[]),
        "ws_b",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum", "--weight-stream"],
    );
    assert!(ws.success, "--weight-stream arm failed:\n{}", ws.stderr);

    // Anti-vacuity: the streaming cycle actually ran, with the EXACT part-2
    // transfer arithmetic. 13 micro-batch forwards × 6 streamed params
    // (upload before the layer's segment, read-only evict after its last
    // primal touch) + 6 windows × 6 (range-head upload, post-update
    // writeback evict) = 114 each; the WRITEBACK subset must be exactly
    // 6 windows × 6 = 36 — total counts alone cannot pin it, because a
    // dropped writeback leg is absorbed by the idempotent step-top register
    // belt at identical totals while training on stale mirrors (review
    // D2b-2-5). Register and teardown moves are deliberately uncounted
    // (mirror bootstrap + final restore).
    let (uploads, evicts, writeback) = ws_counts(&ws.stderr);
    assert_eq!(
        (uploads, evicts, writeback),
        (114, 114, 36),
        "weight-stream transfer counts drifted from the designed schedule \
         (13 fwd × 6 + 6 win × 6 = 114 each; 36 writeback)"
    );
    // D2b part 2: the forward really was sliced per CCR segment (prologue +
    // 2 blocks + epilogue) with all 6 layer params on the streaming plan,
    // AND the brackets sit on the right slices — the per-slice vectors pin
    // PLACEMENT (review D2b-2-2: counts + parity are invariant under a
    // widened plan that re-creates full forward residency).
    assert!(
        ws.stderr.contains(
            "[weight-stream] forward streaming: 4 slices, 6 streamed params \
             (6 touched by the primal); uploads/slice [0,3,3,0] \
             evicts/slice [0,3,3,0]"
        ),
        "forward-streaming schedule line missing or changed:\n{}",
        ws.stderr
    );
    // Runtime end-to-end bracket probe: WEIGHTS-surface bytes resident AT
    // the global allocator peak. The baseline holds the FULL model tag-1
    // (`.to(cuda)` brackets the transfer in SURFACE_WEIGHTS): embed 16,384
    // + norms + 2×(norm_f + w_up + w_down) = ~148 KB. Tight streaming
    // brackets keep at most the epilogue residents + ~one block uploaded
    // (~82 KB → ratio ~0.56); a widened EMISSION (all blocks resident
    // through the peak) restores ~1.0. Assert ≤ 0.7 with the same
    // allocator-rounding headroom the m_partial gate uses, plus the
    // baseline ≥ the streamed total as anti-vacuity (the accounting must
    // actually see full residency for the ratio to mean anything).
    let ws_weights_at_peak = marker_i64(&ws.stdout, "WEIGHTS_AT_PEAK")
        .expect("WEIGHTS_AT_PEAK markers missing (ws arm)");
    let base_weights_at_peak = marker_i64(&base.stdout, "WEIGHTS_AT_PEAK")
        .expect("WEIGHTS_AT_PEAK markers missing (baseline arm)");
    assert!(
        base_weights_at_peak >= 131_584,
        "baseline weights-at-peak ({base_weights_at_peak}) below the \
         streamed total — the surface accounting is not seeing residency"
    );
    let ratio = ws_weights_at_peak as f64 / base_weights_at_peak as f64;
    assert!(
        ratio <= 0.7,
        "weights-surface bytes at the global peak did not shrink \
         (ws {ws_weights_at_peak} vs base {base_weights_at_peak}, ratio \
         {ratio:.3}) — forward brackets have widened"
    );
    // Baseline arm must not stream.
    assert!(
        base.stderr
            .lines()
            .find_map(|l| l.strip_prefix("[weight-stream] uploads: "))
            .map(|c| c.starts_with('0'))
            .unwrap_or(true),
        "baseline arm streamed weights:\n{}",
        base.stderr
    );

    // Bit-exact: streaming is byte-preserving end to end (mirror round
    // trips + post-update writebacks + final restore).
    assert_eq!(
        base.loss_stream, ws.loss_stream,
        "loss stream diverged under --weight-stream"
    );
    let bytes_a = std::fs::read(&save_a).expect("csla model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("ws model_save missing");
    assert!(
        bytes_a == bytes_b,
        "saved model bytes diverged under --weight-stream"
    );
}

/// P0.1 weight-streaming PASSTHROUGH-VIEW gate (GPU): the #397 fix extended the
/// view-of-θ residency exclusion from AD-internal struct variants to the
/// user-facing view METHODS (.reshape / .contiguous / .expand / ...), which
/// lower to `PrimalOp::Passthrough`. The fixture buffers the tied LM-head view
/// of `embed` through a reshape -> contiguous -> reshape -> transpose chain, so
/// the primal_view_of walk MUST traverse Passthrough nodes to reach `embed` and
/// keep it resident. If it did not, evicting `embed` would free its storage,
/// the later upload would reallocate elsewhere, and the buffered head view would
/// read recycled memory.
///
/// The gate verifies all three signals the review asked for, not just loss:
///   - MODEL BYTES: bit-exact vs the non-streaming baseline (a recycled-memory
///     read would corrupt the saved model);
///   - POINTER MOVEMENT: the 6 streamed layer params register and reallocate
///     (registered == 6, ptr_moves > 0), while the view-rooted `embed` NEVER
///     registers (so its device pointer never moves);
///   - TRANSFER COUNTERS: the exact designed upload/evict/writeback schedule.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_weight_stream_passthrough_view_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_wsview_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("view_base.nslm");
    let save_b = tmp.join("view_ws.nslm");

    let base = run_program(
        &program("csla_weight_stream_view_methods.nsl", true, &save_a, &[]),
        "wsview_a",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(base.success, "baseline arm failed:\n{}", base.stderr);
    let ws = run_program(
        &program("csla_weight_stream_view_methods.nsl", true, &save_b, &[]),
        "wsview_b",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum", "--weight-stream"],
    );
    assert!(ws.success, "--weight-stream arm failed:\n{}", ws.stderr);

    // The view-of-θ exclusion fired: at least one param (the tied `embed`)
    // stays resident because a buffered Passthrough view aliases its storage.
    assert!(
        ws.stderr.contains("param(s) stay resident"),
        "view-of-θ exclusion line missing — the Passthrough view chain did not \
         keep `embed` resident:\n{}",
        ws.stderr
    );

    // Transfer counters: same designed schedule as the plain FFN fixture
    // (13 fwd × 6 + 6 win × 6 = 114; 36 writeback).
    let (uploads, evicts, writeback) = ws_counts(&ws.stderr);
    assert_eq!(
        (uploads, evicts, writeback),
        (114, 114, 36),
        "weight-stream transfer counts drifted from the designed schedule"
    );

    // Pointer movement: exactly the 6 streamed LAYER params register (their
    // storage is freed on evict), and they reallocate to fresh addresses. The
    // view-rooted `embed` is EXCLUDED, so it never registers — registered
    // counts only the streamed params.
    let (registered, ptr_moves) = ws_ptr_counts(&ws.stderr);
    assert_eq!(
        registered, 6,
        "expected exactly the 6 streamed layer params to register (embed excluded)"
    );
    assert!(
        ptr_moves > 0,
        "streamed params must actually move (evict frees, upload reallocs) — \
         got ptr_moves={ptr_moves}"
    );
    // Baseline must not stream at all.
    let (base_reg, _) = ws_ptr_counts(&base.stderr);
    assert_eq!(base_reg, 0, "baseline arm registered params for streaming");

    // Model bytes: bit-exact end to end. Streaming is byte-preserving ONLY if
    // the buffered head view never read recycled `embed` memory — this is the
    // correctness payoff of keeping `embed` resident through the Passthrough
    // view chain.
    assert_eq!(
        base.loss_stream, ws.loss_stream,
        "loss stream diverged under --weight-stream (recycled-memory read?)"
    );
    let bytes_a = std::fs::read(&save_a).expect("baseline model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("ws model_save missing");
    assert!(
        bytes_a == bytes_b,
        "saved model bytes diverged under --weight-stream — the buffered \
         Passthrough view of `embed` likely read recycled storage (the #397 \
         exclusion regressed)"
    );
}

/// Fused-CE tape-carry parity (GPU): @fused_lm_ce(dtype=f32) composes with
/// the layerwise schedule. Both arms carry the decorator (fused-vs-fused —
/// fused numerics differ from the composite by design), so the diff is the
/// schedule alone and parity is BIT-EXACT. The carry buffers one [B*S] f32
/// logsumexp per micro-batch and re-binds `fused_ce_fwd_lse` by the seeded
/// loss Value per replay; the materialized [B*S, V] logits chain never
/// exists on either arm. Anti-vacuity: exact launch counters (fwd 13 both
/// arms; bwd 13 interleaved vs 12 replayed — the trailing partial window
/// never replays) + the exact carry-slot stderr line; a composite fallback
/// or an inert carry cannot pass.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_parity_fused_lmce_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_fce_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("fce_base.nslm");
    let save_b = tmp.join("fce_csla.nslm");

    let base = run_program(
        &program("csla_fused_lmce.nsl", true, &save_a, &[]),
        "fce_a",
        true,
        true,
        &["--checkpoint-blocks"],
    );
    assert!(base.success, "fused baseline arm failed:\n{}", base.stderr);
    // Determinism probe: the fused backward scatters dW/dbias via
    // red.global.add.f32 — atomic-order ULP nondeterminism, exactly the
    // embedding-backward situation. Baseline-vs-itself decides whether
    // bit-exact comparison is even available on this kernel/driver combo.
    let save_a2 = tmp.join("fce_base2.nslm");
    let base2 = run_program(
        &program("csla_fused_lmce.nsl", true, &save_a2, &[]),
        "fce_a2",
        true,
        true,
        &["--checkpoint-blocks"],
    );
    assert!(base2.success, "fused baseline rerun failed:\n{}", base2.stderr);
    // Probe on streams AND raw model bytes (review D2c-6): the exact
    // branch compares bytes, so the probe must certify byte-level
    // determinism, not just the printed (rounded) losses.
    let backward_deterministic = base.loss_stream == base2.loss_stream
        && std::fs::read(&save_a).ok() == std::fs::read(&save_a2).ok();
    let csla = run_program(
        &program("csla_fused_lmce.nsl", true, &save_b, &[]),
        "fce_b",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(csla.success, "fused csla arm failed:\n{}", csla.stderr);

    // Schedule engaged (not a degenerate all-epilogue shape) + exactly one
    // carried fused-CE LSE slot.
    assert_eq!(
        window_phase_count(&csla.stderr),
        Some(6),
        "csla arm window count:\n{}",
        csla.stderr
    );
    assert_eq!(window_phase_count(&base.stderr), Some(0));
    assert!(
        csla.stderr.contains("[csla] fused-ce tape-carry: 1 slots"),
        "fused-CE carry slot line missing or wrong:\n{}",
        csla.stderr
    );

    // Launch counters: the fused kernel actually ran, the designed number
    // of times, on BOTH arms.
    let fwd_b = marker_i64(&base.stdout, "FUSED_LCE_FWD").expect("base fwd marker");
    let bwd_b = marker_i64(&base.stdout, "FUSED_LCE_BWD").expect("base bwd marker");
    let fwd_c = marker_i64(&csla.stdout, "FUSED_LCE_FWD").expect("csla fwd marker");
    let bwd_c = marker_i64(&csla.stdout, "FUSED_LCE_BWD").expect("csla bwd marker");
    assert_eq!(
        (fwd_b, bwd_b, fwd_c, bwd_c),
        (13, 13, 13, 12),
        "fused-CE launch counts drifted (base fwd/bwd, csla fwd/bwd)"
    );

    // Semantic anchors (targets-dtype + ghost-adjoint lessons): bit-exact
    // parity alone is satisfiable by DETERMINISTIC GARBAGE — the s64-vs-f32
    // targets overread AND the i8-vs-i32 tag-4 misread both produced
    // identical wrong losses on both arms, and the un-drained fused-LCE
    // prune left ghost adjoints that silently dropped every param grad
    // (flat loss, still bit-exact between arms). Two independent anchors:
    //   1. the first loss must sit in the sane CE band for this init —
    //      ln(128)=4.852 plus the logit-variance term (~5.65 measured
    //      against the composite/no-train ground truth);
    //   2. the model must actually TRAIN: last window loss well below the
    //      first (weight-decay-only "training" stays flat).
    let parse = |l: &str| -> Option<f64> {
        l.trim()
            .trim_start_matches("tensor([")
            .trim_end_matches("])")
            .parse()
            .ok()
    };
    let losses: Vec<f64> = base.loss_stream.lines().filter_map(parse).collect();
    let first_loss = *losses.first().expect("empty loss stream");
    // Pinned to the composite-derived ground truth for this deterministic
    // init + data (review D2c-4: the old (4.5, 6.4) band ADMITTED the
    // measured tag-4-garbled value 5.19 — a band is not a discriminator).
    assert!(
        (first_loss - 5.6485).abs() < 0.02,
        "first fused-CE loss {first_loss} != composite ground truth 5.6485 \
         — the kernel is not computing real cross-entropy (targets dtype \
         bridge regressed?)"
    );
    // Whole-stream mean separates trained (measured 4.78) from
    // grads-dropped/frozen (measured 5.36 at lr~0 — per-block data
    // difficulty makes single-loss descent checks noisy).
    let mean_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
    assert!(
        mean_loss < 5.1,
        "fused-CE mean loss {mean_loss} looks frozen (trained ~4.78, \
         grads-dropped ~5.36) — ghost-adjoint prune regressed?"
    );

    // Parity: with a deterministic backward, demand bit-exactness (loss
    // stream + model bytes). With the atomic-scatter backward (the current
    // kernel), run-to-run ULP noise makes bit-exactness unavailable even
    // baseline-vs-baseline — fall back to: FIRST loss bit-equal (the
    // pre-update forward has no atomics; a schedule bug that corrupts
    // inputs shows up here exactly) + every later step within a tight
    // relative band (ULP noise compounds to ~1e-6 over 6 steps at lr 2e-3;
    // a wrong replay shows orders of magnitude more).
    if backward_deterministic {
        assert_eq!(
            base.loss_stream, csla.loss_stream,
            "loss stream diverged under --layerwise-accum with @fused_lm_ce"
        );
        let bytes_a = std::fs::read(&save_a).expect("base model_save missing");
        let bytes_b = std::fs::read(&save_b).expect("csla model_save missing");
        assert!(
            bytes_a == bytes_b,
            "saved model bytes diverged under --layerwise-accum with @fused_lm_ce"
        );
    } else {
        let csla_losses: Vec<f64> = csla.loss_stream.lines().filter_map(parse).collect();
        assert_eq!(
            losses.len(),
            csla_losses.len(),
            "loss stream lengths diverged"
        );
        assert_eq!(
            losses.first(),
            csla_losses.first(),
            "FIRST fused-CE loss diverged — the replay's forward inputs or \
             LSE carry are wrong (this is pre-update and atomic-free)"
        );
        for (i, (a, b)) in losses.iter().zip(&csla_losses).enumerate() {
            let rel = (a - b).abs() / a.abs().max(1e-9);
            assert!(
                rel < 1e-4,
                "step {i}: csla loss {b} deviates from baseline {a} \
                 (rel {rel:.2e}) beyond atomic-ULP noise"
            );
        }
    }
}

/// The narrowed refusal: @fused_lm_ce with a non-f32 dtype stays refused
/// under the layerwise schedule (step-scoped cast shadow tensors cannot be
/// carried), with the loud diagnostic — not a silent composite fallback.
#[test]
fn csla_fused_lmce_non_f32_refused() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_fce_ref_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("unused.nslm");
    let out = run_program(
        &program(
            "csla_fused_lmce.nsl",
            false,
            &save,
            &[("dtype=\"f32\"", "dtype=\"f16\"")],
        ),
        "fce_ref",
        false,
        false,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(!out.success, "non-f32 fused-CE under csla must refuse");
    assert!(
        out.stderr
            .contains("supports @fused_lm_ce only with dtype=\"f32\""),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Fused-CE × streaming composition (GPU): the exact 1B production stack —
/// @fused_lm_ce + --layerwise-accum + --optim-state-offload +
/// --weight-stream (review D2c-7: previously only the manual 1B run
/// covered it). Streaming and staging are byte-preserving and the forward
/// is deterministic, so the FIRST loss must be bit-equal between the
/// csla arm and the fully-streamed arm; later steps within atomic-ULP
/// noise. Transfer counters and the placement-pinned schedule line carry
/// over from the ws gates (same 2-block structure: 6 streamed params).
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_fused_lmce_streams_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_fces_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("fces_base.nslm");
    let save_b = tmp.join("fces_stream.nslm");

    let base = run_program(
        &program("csla_fused_lmce.nsl", true, &save_a, &[]),
        "fces_a",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(base.success, "fused csla arm failed:\n{}", base.stderr);
    let ws = run_program(
        &program("csla_fused_lmce.nsl", true, &save_b, &[]),
        "fces_b",
        true,
        true,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--optim-state-offload",
            "--weight-stream",
        ],
    );
    assert!(ws.success, "fused streamed arm failed:\n{}", ws.stderr);

    assert_eq!(window_phase_count(&ws.stderr), Some(6));
    assert!(
        ws.stderr.contains("[csla] fused-ce tape-carry: 1 slots"),
        "carry inert under the streamed composition:\n{}",
        ws.stderr
    );
    assert_eq!(
        ws_counts(&ws.stderr),
        (114, 114, 36),
        "transfer counts drifted under the fused composition"
    );
    assert!(
        ws.stderr.contains(
            "[weight-stream] forward streaming: 4 slices, 6 streamed params \
             (6 touched by the primal); uploads/slice [0,3,3,0] \
             evicts/slice [0,3,3,0]"
        ),
        "forward-streaming schedule line missing under the fused \
         composition:\n{}",
        ws.stderr
    );

    let parse = |l: &str| -> Option<f64> {
        l.trim()
            .trim_start_matches("tensor([")
            .trim_end_matches("])")
            .parse()
            .ok()
    };
    let a: Vec<f64> = base.loss_stream.lines().filter_map(parse).collect();
    let b: Vec<f64> = ws.loss_stream.lines().filter_map(parse).collect();
    assert_eq!(a.len(), b.len(), "loss stream lengths diverged");
    assert_eq!(
        base.loss_stream.lines().next(),
        ws.loss_stream.lines().next(),
        "FIRST loss diverged under byte-preserving streaming — the fused \
         forward read different bytes"
    );
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let rel = (x - y).abs() / x.abs().max(1e-9);
        assert!(
            rel < 1e-4,
            "step {i}: streamed loss {y} deviates from csla {x} (rel {rel:.2e})"
        );
    }
}

/// D2b × D2a composition (GPU): --weight-stream WITH --optim-state-offload —
/// the exact 1B production configuration (the 9.10 GB measurement). The
/// per-layer update site interleaves the offload envelope's staged-m/v
/// drain with the θ writeback evict; no other gate exercises that
/// interleaving (review D2b-2-4). Both arms carry --optim-state-offload so
/// the diff is the streaming flag alone; bit-exact + exact counters.
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_weight_stream_offload_parity_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_wso_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("csla_off.nslm");
    let save_b = tmp.join("csla_off_ws.nslm");

    let base = run_program(
        &program("csla_layerwise_ffn.nsl", true, &save_a, &[]),
        "wso_a",
        true,
        true,
        &["--checkpoint-blocks", "--layerwise-accum", "--optim-state-offload"],
    );
    assert!(base.success, "csla+offload arm failed:\n{}", base.stderr);
    let ws = run_program(
        &program("csla_layerwise_ffn.nsl", true, &save_b, &[]),
        "wso_b",
        true,
        true,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--optim-state-offload",
            "--weight-stream",
        ],
    );
    assert!(ws.success, "csla+offload+ws arm failed:\n{}", ws.stderr);

    assert_eq!(
        ws_counts(&ws.stderr),
        (114, 114, 36),
        "transfer counts drifted under the offload composition"
    );
    assert!(
        ws.stderr.contains(
            "[weight-stream] forward streaming: 4 slices, 6 streamed params \
             (6 touched by the primal); uploads/slice [0,3,3,0] \
             evicts/slice [0,3,3,0]"
        ),
        "forward-streaming schedule line missing under the offload \
         composition:\n{}",
        ws.stderr
    );
    assert_eq!(
        base.loss_stream, ws.loss_stream,
        "loss stream diverged under --weight-stream + --optim-state-offload"
    );
    let bytes_a = std::fs::read(&save_a).expect("csla+offload model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("csla+offload+ws model_save missing");
    assert!(
        bytes_a == bytes_b,
        "saved model bytes diverged under --weight-stream + --optim-state-offload"
    );
}

/// D1b memory-win gate (GPU): the whole point of the layer-major schedule —
/// the m_partial surface peak drops from the full-model window (baseline
/// allocates every accumulator up front) to max(one layer) + the epilogue
/// globals. FFN fixture: full = 37056 elems, layerwise peak = 16448 + 4160 =
/// 20608 elems → ratio 0.556; assert <= 0.7 for allocator-rounding headroom,
/// plus both peaks nonzero (anti-vacuity).
#[test]
#[ignore = "requires CUDA GPU"]
fn csla_mpartial_surface_shrinks_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_surface_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let peak_of = |tag: &str, extra: &[&str]| -> f64 {
        let src = program(
            "csla_layerwise_ffn.nsl",
            true,
            &tmp.join(format!("{tag}.nslm")),
            &[],
        );
        let root = repo_root();
        let dir = tmp.join(tag);
        std::fs::create_dir_all(&dir).unwrap();
        let prog = dir.join("surface.nsl");
        std::fs::write(&prog, src).unwrap();
        let out = Command::new(env!("CARGO"))
            .args(["run", "-q", "--features", "cuda"])
            .arg("--manifest-path")
            .arg(root.join("Cargo.toml"))
            .args(["-p", "nsl-cli", "--", "run", "--source-ad", "--deterministic"])
            .args(extra)
            .arg(&prog)
            .current_dir(&dir)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .env("NSL_EMBEDDING_BWD_CPU", "1")
            .output()
            .expect("spawn nsl run");
        assert!(
            out.status.success(),
            "{tag} run failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut in_peak = false;
        for line in stdout.lines() {
            match line.trim() {
                "MPARTIAL_PEAK_BEGIN" => in_peak = true,
                "MPARTIAL_PEAK_END" => in_peak = false,
                l if in_peak => {
                    let cleaned = l
                        .trim()
                        .trim_start_matches("tensor([")
                        .trim_end_matches("])");
                    if let Ok(v) = cleaned.parse::<f64>() {
                        return v;
                    }
                }
                _ => {}
            }
        }
        panic!("MPARTIAL_PEAK markers missing in {tag} stdout:\n{stdout}");
    };

    let base_peak = peak_of("base", &["--checkpoint-blocks"]);
    let csla_peak = peak_of("csla", &["--checkpoint-blocks", "--layerwise-accum"]);
    assert!(
        base_peak > 0.0 && csla_peak > 0.0,
        "m_partial surface peaks must be nonzero (base {base_peak}, csla {csla_peak})"
    );
    let ratio = csla_peak / base_peak;
    assert!(
        ratio <= 0.7,
        "layerwise m_partial peak did not shrink: csla {csla_peak} / base {base_peak} \
         = {ratio:.3} (expected <= 0.7; schedule projects 0.556 on this fixture)"
    );
    eprintln!(
        "[csla-surface] m_partial peak: baseline {base_peak}B -> layerwise {csla_peak}B \
         (ratio {ratio:.3})"
    );
}

/// Deferral-must-refuse: every unsupported composition dies loudly at
/// compile time instead of silently running the baseline schedule.
#[test]
fn csla_refusals() {
    let tmp = std::env::temp_dir().join(format!("nsl_csla_refusals_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // (a) grad_clip: two-phase clip needs the full-window global norm.
    let clip_src = program(
        "csla_layerwise_ffn.nsl",
        false,
        &tmp.join("clip.nslm"),
        &[(
            "grad_accumulation=2)",
            "grad_accumulation=2, grad_clip=1.0)",
        )],
    );
    let clip = run_program(
        &clip_src,
        "refuse_clip",
        false,
        false,
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(!clip.success, "grad_clip + --layerwise-accum must refuse");
    assert!(
        clip.stderr.contains("incompatible with grad_clip"),
        "expected the grad_clip refusal, got:\n{}",
        clip.stderr
    );

    // (b) missing --checkpoint-blocks: clap-level `requires` error.
    let plain_src = program("csla_layerwise_ffn.nsl", false, &tmp.join("noccr.nslm"), &[]);
    let noccr = run_program(&plain_src, "refuse_noccr", false, false, &["--layerwise-accum"]);
    assert!(
        !noccr.success,
        "--layerwise-accum without --checkpoint-blocks must refuse"
    );
    assert!(
        noccr.stderr.contains("checkpoint-blocks") || noccr.stderr.contains("checkpoint_blocks"),
        "expected the clap requires error naming checkpoint-blocks, got:\n{}",
        noccr.stderr
    );

    // (c) missing --source-ad: codegen admission error. Bypass the shared
    // runner (it always passes --source-ad) with a direct invocation.
    let root = repo_root();
    let prog = tmp.join("no_source_ad.nsl");
    std::fs::write(
        &prog,
        program("csla_layerwise_ffn.nsl", false, &tmp.join("nosad.nslm"), &[]),
    )
    .unwrap();
    let out = Command::new(env!("CARGO"))
        .arg("run")
        .arg("-q")
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--checkpoint-blocks",
            "--layerwise-accum",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        !out.status.success(),
        "--layerwise-accum without --source-ad must refuse"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("requires --source-ad"),
        "expected the source-ad refusal, got:\n{stderr}"
    );
}
