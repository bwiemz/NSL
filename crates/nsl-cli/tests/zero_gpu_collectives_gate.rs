//! P4 item 14: CUDA-aware ZeRO collectives.
//!
//! The `--collectives` axis: `sim` (CPU-shm reference), `sim-gpu` (CUDA-aware
//! TEST backend — device-pointer API staged through the shm reduce; validates
//! the GPU plumbing on ONE GPU), `nccl` (real NCCL transport; multiple ranks
//! per device are refused by NCCL itself, so the ws>=2 NCCL leg is a
//! scheduled HARDWARE gate: on a single-GPU box it asserts the documented
//! symmetric refusal, on a multi-GPU box it asserts parity).
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test zero_gpu_collectives_gate -- --ignored`
//! NCCL: same with `--features cuda,nccl` (plus NSL_NCCL_LIB_DIR/LD_LIBRARY_PATH).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    loss: String,
    stderr: String,
}

fn run_gpu(tag: &str, extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zgpu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("out.nslm");
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "m.to(cuda)")
    .replace(
        "CSLA_SAVE_PATH",
        &save.display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic", "--zero-stage", "1"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_ZERO_COUNTER", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    RunOut {
        success: out.status.success(),
        loss,
        stderr,
    }
}

/// Unknown backend value fails fast at the CLI, before any spawn.
#[test]
fn unknown_collectives_value_refuses() {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--collectives", "bogus", "x.nsl"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--collectives must be"),
        "wrong refusal:\n{stderr}"
    );
}

/// The CUDA-aware TEST backend: a 2-rank GPU-RESIDENT ZeRO-1 run must be
/// BIT-EXACT against the single-rank GPU baseline (replicated data:
/// (g+g)/2 == g in f32), with real device-pointer collectives counted and
/// the byte-balanced moment shards summing to the unsharded total.
#[test]
#[ignore = "requires CUDA GPU"]
fn sim_gpu_two_rank_parity_bit_exact() {
    let base = run_gpu("base", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);
    assert!(!base.loss.is_empty(), "empty baseline loss stream");

    let spmd = run_gpu("simgpu", &["--devices", "2", "--collectives", "sim-gpu"]);
    assert!(spmd.success, "2-rank sim-gpu run failed:\n{}", spmd.stderr);
    assert_eq!(
        spmd.loss, base.loss,
        "2-rank GPU ZeRO-1 diverged from the single-rank GPU baseline\nstderr:\n{}",
        spmd.stderr
    );

    // Anti-vacuity: bucketed collectives actually ran on both ranks. Each step
    // reduces ONE bucket and syncs TWO owner groups, so 6 steps give
    // all_reduce=6, broadcast=12, with all 96 tensor-memberships covered.
    //
    // These counts were all_reduce=12 / broadcast=18 when this gate was written
    // (2026-07-20), on the premise that "not every gradient lands on-device —
    // one CPU f64 group rides alongside the GPU f32 bucket". That premise was a
    // BUG, not a property: `model.to(device)` skipped every field declared
    // AFTER an inline model array, and this fixture is `blocks: [Blk; 2]`
    // followed by `norm_out`. f1030f76 (2026-07-22) fixed it, so the whole
    // model is now GPU-resident, the CPU group is gone, and the two buckets
    // collapse into one. bucket_members stays 96 — the same tensors are
    // covered, in fewer buckets. The bit-exactness assertion above is the real
    // correctness property and it still holds.
    for rank in 0..2 {
        let line = spmd
            .stderr
            .lines()
            .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
            .unwrap_or_else(|| panic!("no [zero] line for rank {rank}:\n{}", spmd.stderr));
        assert!(
            line.contains("all_reduce=6 ")
                && line.contains("broadcast=12 ")
                && line.contains("bucket_members=96 "),
            "wrong collective counts: {line}"
        );
    }
    let elems = |s: &str, pat: &str| -> u64 {
        s.lines()
            .find(|l| l.contains(pat))
            .and_then(|l| l.split("optim_elems=").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let full = elems(&base.stderr, "[zero] ws=1 rank=0");
    let r0 = elems(&spmd.stderr, "[zero] ws=2 rank=0");
    let r1 = elems(&spmd.stderr, "[zero] ws=2 rank=1");
    assert!(full > 0 && r0 > 0 && r1 > 0, "missing optim_elems");
    assert_eq!(r0 + r1, full, "byte-balanced shards must cover the total");
    assert!(r0 < full && r1 < full, "both ranks must actually shard");
}

/// P4 item 16 (ZeRO-2) on GPU: partitioned gradients via device
/// reduce_scatter, still bit-exact against the single-rank GPU baseline.
#[test]
#[ignore = "requires CUDA GPU"]
fn sim_gpu_stage2_reduce_scatter_bit_exact() {
    let base = run_gpu("s2base", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);

    // run_gpu passes --zero-stage 1 already; clap rejects duplicates, so
    // stage-2 runs use the dedicated runner below.
    let spmd = run_gpu_stage2("s2spmd", &[]);
    assert!(spmd.success, "2-rank stage-2 sim-gpu run failed:\n{}", spmd.stderr);
    assert_eq!(
        spmd.loss, base.loss,
        "GPU stage-2 diverged from the single-rank GPU baseline\n{}",
        spmd.stderr
    );
    for rank in 0..2 {
        let line = spmd
            .stderr
            .lines()
            .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
            .unwrap_or_else(|| panic!("no [zero] line for rank {rank}:\n{}", spmd.stderr));
        // 6 steps × 1 GPU f32 group = 6 scatters; no gradient may travel
        // through an all_reduce. Was 12 on the same stale premise corrected in
        // the stage-1 gate above: the second (CPU f64) group only existed
        // because `model.to(device)` skipped fields after an inline model
        // array, fixed in f1030f76. all_reduce=0 is the load-bearing half of
        // this assertion and is unchanged. (Label-anchored with delimiters —
        // the original `ends_with("reduce_scatter=6")` broke the day item 11
        // appended `all_gather=` to the line: positional/terminal anchors rot
        // on append, labels do not.)
        assert!(
            line.contains(" reduce_scatter=6 ") && line.contains("all_reduce=0 "),
            "stage-2 GPU collective counts wrong: {line}"
        );
    }
}

fn run_gpu_stage2(tag: &str, envs: &[(&str, &str)]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zgpu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("out.nslm");
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "m.to(cuda)")
    .replace(
        "CSLA_SAVE_PATH",
        &save.display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args([
        "run",
        "--source-ad",
        "--deterministic",
        "--zero-stage",
        "2",
        "--devices",
        "2",
        "--collectives",
        "sim-gpu",
    ])
    .arg(&prog)
    .current_dir(&tmp)
    .env("NSL_STDLIB_PATH", root.join("stdlib"))
    .env("NSL_ZERO_COUNTER", "1")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    RunOut {
        success: out.status.success(),
        loss,
        stderr,
    }
}

/// M3 (stage-2 shm wall) on the DEVICE path: a sub-MB bucket cap forces
/// padded sub-groups + intra-tensor byte-range chunks through the CUDA-aware
/// scatter (device pack, staging-buffer average, offset unpack). Must stay
/// BIT-EXACT vs the single-rank GPU baseline with a multiplied scatter count.
#[test]
#[ignore = "requires CUDA GPU"]
fn sim_gpu_stage2_chunked_subgroups_bit_exact() {
    let base = run_gpu("s2chbase", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);

    let spmd = run_gpu_stage2("s2chspmd", &[("NSL_ZERO_BUCKET_MB", "0.02")]);
    assert!(
        spmd.success,
        "2-rank chunked stage-2 sim-gpu run failed:\n{}",
        spmd.stderr
    );
    assert_eq!(
        spmd.loss, base.loss,
        "chunked GPU stage-2 diverged from the single-rank GPU baseline\n{}",
        spmd.stderr
    );
    for rank in 0..2 {
        let line = spmd
            .stderr
            .lines()
            .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
            .unwrap_or_else(|| panic!("no [zero] line for rank {rank}:\n{}", spmd.stderr));
        let scatters: u64 = line
            .split("reduce_scatter=")
            .nth(1)
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no reduce_scatter= on [zero] line: {line}"));
        assert!(
            scatters > 12 && line.contains("all_reduce=0 "),
            "GPU chunking must multiply the scatter count past the unchunked \
             12 (got {scatters}) with no grad all_reduce: {line}"
        );
    }
}

/// The real NCCL transport, ws=2. On a multi-GPU box: full parity. On a
/// single-GPU box NCCL refuses multiple ranks per device — the run must
/// fail FAST with the documented symmetric ncclInvalidUsage refusal on
/// every rank (never a hang, never a silent success).
#[test]
#[ignore = "requires CUDA GPU + nccl-featured build"]
fn nccl_two_rank_parity_or_documented_refusal() {
    let base = run_gpu("nbase", &[]);
    assert!(base.success, "GPU baseline failed:\n{}", base.stderr);

    let spmd = run_gpu("nccl2", &["--devices", "2", "--collectives", "nccl"]);
    if spmd.success {
        // Multi-GPU hardware: demand full parity + counters.
        assert_eq!(
            spmd.loss, base.loss,
            "2-rank NCCL ZeRO-1 diverged from the single-rank GPU baseline"
        );
        assert!(
            spmd.stderr.contains("NCCL communicator up"),
            "NCCL banner missing:\n{}",
            spmd.stderr
        );
        for rank in 0..2 {
            let line = spmd
                .stderr
                .lines()
                .find(|l| l.contains(&format!("[zero] ws=2 rank={rank}")))
                .unwrap_or_else(|| panic!("no [zero] line for rank {rank}"));
            assert!(
                line.contains("all_reduce=12 ") && line.contains("broadcast=18 "),
                "wrong collective counts: {line}"
            );
        }
    } else {
        // Single-GPU hardware: the documented loud, symmetric refusal.
        assert!(
            spmd.stderr.contains("ncclInvalidUsage")
                || spmd.stderr.contains("NCCL backend init failed"),
            "expected the documented NCCL same-device refusal, got:\n{}",
            spmd.stderr
        );
        assert!(
            spmd.stderr.contains("rank 0") && spmd.stderr.contains("rank 1"),
            "refusal must be symmetric across ranks:\n{}",
            spmd.stderr
        );
    }
}

// ── Item 8 × D3: the batched optimizer step under ZeRO ──────────────────────
//
// `multi_scalars` admission carried `&& !zero_enabled`, so every
// `--zero-stage` run silently reverted item 8's batched fused-AdamW launch to
// the per-param loop — a shipped optimization that was 100% off in a supported
// configuration, with no diagnostic anywhere. Nothing could see it: the only
// evidence of batching was a one-shot `[fase-multi]` marker whose ABSENCE no
// gate asserted, and `--zero-stage` runs never set NSL_FASE_FUSED_COUNTER.

struct FaseOut {
    success: bool,
    loss: String,
    stderr: String,
    model: Vec<u8>,
}

/// Same fixture and flags as `run_gpu`, plus the fused-step counter report and
/// the saved model bytes. `env` lets an arm force the per-param loop.
fn run_gpu_fase(tag: &str, extra: &[&str], env: &[(&str, &str)]) -> FaseOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_zfase_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("out.nslm");
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap()
    .replace("# GPU_PLACEMENT", "m.to(cuda)")
    .replace(
        "CSLA_SAVE_PATH",
        &save.display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic", "--zero-stage", "1"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_ZERO_COUNTER", "1")
        .env("NSL_FASE_FUSED_COUNTER", "1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    FaseOut {
        success: out.status.success(),
        loss,
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        model: std::fs::read(&save).unwrap_or_default(),
    }
}

/// Parse `batched=` / `fallback=` off the `[fase-fused] multi params:` line,
/// ORDERED BY RANK.
///
/// Sorting by the line's own `rank=` matters: the ranks are separate processes
/// writing to one pipe, so arrival order is whichever exits first. Pairing a
/// count with a per-rank fact by position crosses them roughly half the time —
/// which is exactly how this gate first failed, reporting rank 0's count
/// against rank 1's owned set.
///
/// Field lookup is LABEL-anchored (`" name="`), not `ends_with`: a field
/// appended to that line later must not silently turn this into a no-match,
/// which is how three sibling parsers rotted in #477.
fn multi_params(stderr: &str, field: &str) -> Vec<u64> {
    let mut rows: Vec<(u64, u64)> = stderr
        .lines()
        .filter(|l| l.contains("[fase-fused] multi params:"))
        .filter_map(|l| {
            let get = |k: &str| {
                l.split_whitespace()
                    .find_map(|t| t.strip_prefix(&format!("{k}="))?.parse::<u64>().ok())
            };
            Some((get("rank")?, get(field)?))
        })
        .collect();
    rows.sort_by_key(|(r, _)| *r);
    rows.into_iter().map(|(_, v)| v).collect()
}

/// Parse the `[zero] owned-step rank=R n=N idx=a,b,c` lines into
/// `(rank, n, indices)`, sorted by rank. An empty `idx=` yields an empty set
/// (a rank can legitimately own nothing).
fn owned_sets(stderr: &str) -> Vec<(i64, i64, Vec<i64>)> {
    let mut out: Vec<(i64, i64, Vec<i64>)> = stderr
        .lines()
        .filter_map(|l| l.trim().strip_prefix("[zero] owned-step "))
        .filter_map(|rest| {
            let field = |k: &str| {
                rest.split_whitespace()
                    .find_map(|t| t.strip_prefix(k).map(|v| v.to_string()))
            };
            let rank = field("rank=")?.parse::<i64>().ok()?;
            let n = field("n=")?.parse::<i64>().ok()?;
            let idx = field("idx=").unwrap_or_default();
            let idxs = idx
                .split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<i64>().ok())
                .collect();
            Some((rank, n, idxs))
        })
        .collect();
    out.sort_by_key(|(r, _, _)| *r);
    out
}

/// A 2-rank ZeRO-1 GPU run must batch its OWNER SUBSET into the fused
/// multi-tensor launch, and be bit-identical to the per-param loop it
/// replaces.
#[test]
#[ignore = "requires CUDA GPU"]
fn zero_stage1_batches_the_owner_subset_gpu() {
    let batched = run_gpu_fase(
        "batched",
        &["--devices", "2", "--collectives", "sim-gpu"],
        &[],
    );
    assert!(batched.success, "batched run failed:\n{}", batched.stderr);
    assert!(!batched.loss.is_empty(), "empty loss stream");
    assert!(!batched.model.is_empty(), "no model saved");

    // The control: NSL_FASE_MULTI_STEP=0 is the compile-time kill switch for
    // the batched arm, so this run takes exactly the per-param owner-gated
    // loop that shipped before this change.
    let per_param = run_gpu_fase(
        "perparam",
        &["--devices", "2", "--collectives", "sim-gpu"],
        &[("NSL_FASE_MULTI_STEP", "0")],
    );
    assert!(
        per_param.success,
        "per-param control failed:\n{}",
        per_param.stderr
    );

    // 1. Numerically identical to the loop it replaces — the actual claim.
    assert_eq!(
        batched.loss, per_param.loss,
        "the batched owner subset diverged from the per-param loop\nstderr:\n{}",
        batched.stderr
    );
    assert_eq!(
        batched.model, per_param.model,
        "trained weights differ between the batched and per-param ZeRO paths"
    );

    // 2. The batched arm actually fired. Before this change this marker never
    //    appeared under any --zero-stage, so this assertion is what goes red
    //    if the `!zero_enabled` exclusion (or the subset launch) comes back.
    assert!(
        batched.stderr.contains("[fase-multi] batched"),
        "no batched-launch marker under --zero-stage 1:\n{}",
        batched.stderr
    );

    // 3. EVERY owned param batched, not just one. `batched + fallback` is the
    //    parameter total the multi launcher saw on each rank; a fallback here
    //    would mean some owned param quietly took the slow arm.
    let b = multi_params(&batched.stderr, "batched");
    let f = multi_params(&batched.stderr, "fallback");
    assert_eq!(b.len(), 2, "expected one report line per rank:\n{}", batched.stderr);
    assert_eq!(f.len(), 2, "expected one report line per rank:\n{}", batched.stderr);
    for r in 0..2 {
        assert!(
            b[r] > 0,
            "rank {r} batched no parameters (batched={}, fallback={}):\n{}",
            b[r],
            f[r],
            batched.stderr
        );
        assert_eq!(
            f[r], 0,
            "rank {r} fell back to the per-param arm for {} parameter(s) — an \
             owned param is not eligible for the batched launch:\n{}",
            f[r], batched.stderr
        );
    }

    // 4. Ownership is a PARTITION — proven on the index sets, not on counts.
    //    Counts cannot distinguish a partition from an overlap plus a matching
    //    gap ({0,1,2} + {2,3,4} sums the same as {0,1,2} + {3,4}: one
    //    parameter stepped twice, one never stepped, both totals right). And
    //    bit-exactness cannot see it either, because ZeRO-1 broadcasts the
    //    owner's parameters after the step, so an over-stepping rank's extra
    //    work is overwritten. The sets are the only witness.
    let sets = owned_sets(&batched.stderr);
    assert_eq!(
        sets.len(),
        2,
        "expected one [zero] owned-step line per rank:\n{}",
        batched.stderr
    );
    let n = sets[0].1;
    assert_eq!(sets[1].1, n, "ranks disagree on the parameter count");
    let (a, b_set) = (&sets[0].2, &sets[1].2);
    let overlap: Vec<i64> = a.iter().filter(|i| b_set.contains(i)).copied().collect();
    assert!(
        overlap.is_empty(),
        "parameter(s) {overlap:?} are owned by BOTH ranks — each would be \
         stepped twice, and the post-step broadcast would hide it"
    );
    let mut union: Vec<i64> = a.iter().chain(b_set.iter()).copied().collect();
    union.sort_unstable();
    let expected: Vec<i64> = (0..n).collect();
    assert_eq!(
        union, expected,
        "the ranks' owned sets do not cover every parameter — an uncovered one \
         is never stepped by anyone"
    );

    // And each rank's batched count must be its owned-set size once per step —
    // the counter is CUMULATIVE over the run, so the invariant is
    // divisibility plus an equal step count, not equality with the set size.
    assert!(
        a.len() > 0 && !b_set.is_empty(),
        "a rank owns nothing; this fixture is meant to split its parameters"
    );
    assert_eq!(
        b[0] as usize % a.len(),
        0,
        "rank 0 batched {} parameters over the run, which is not a whole number \
         of passes over its {}-parameter owned set",
        b[0],
        a.len()
    );
    assert_eq!(
        b[1] as usize % b_set.len(),
        0,
        "rank 1 batched {} parameters over the run, which is not a whole number \
         of passes over its {}-parameter owned set",
        b[1],
        b_set.len()
    );
    assert_eq!(
        b[0] as usize / a.len(),
        b[1] as usize / b_set.len(),
        "the ranks batched their owned sets a different number of times \
         ({} vs {}) — they are not stepping in lockstep",
        b[0] as usize / a.len(),
        b[1] as usize / b_set.len()
    );

    // 5. The control really disabled batching (proves assertion 2 is not
    //    asserting a constant).
    assert_eq!(
        multi_params(&per_param.stderr, "batched"),
        vec![0, 0],
        "NSL_FASE_MULTI_STEP=0 did not disable the batched arm:\n{}",
        per_param.stderr
    );
}
