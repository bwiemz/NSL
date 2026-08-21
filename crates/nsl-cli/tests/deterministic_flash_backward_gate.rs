//! M46: `--deterministic` must actually deliver run-to-run bit-reproducibility
//! for a program whose backward reaches the flash-attention kernel.
//!
//! THE BUG THIS PINS. The flash phase-2 backward accumulates **dQ** with a
//! cross-CTA `atom.global.add.f32`, so its rounding depends on scheduling
//! order: two runs of the same program produce slightly different gradients.
//! (It is dQ, not dK/dV. Each CTA owns one (batch-head, kv-tile) and loops the
//! Q tiles, keeping dK/dV in its own SMEM; dQ is the only cross-block
//! reduction. The contending-CTA count is batch x heads x ceil(seq/block_kv),
//! which is the quantity the fixture's geometry preserves.) `flash_attention.rs`
//! documented that and offered `NSL_FLASH_BWD_CPU=1`, naming "M46-style
//! determinism audits" as a caller that needs the knob — but M46 *is*
//! `--deterministic`, and the flag was never wired to it. Only two runtime
//! sites gated on `is_deterministic()` (the embedding backward and one
//! `sum_dim` route), so any model with attention silently lost the guarantee.
//! Measured at Coder-50M before the fix: 3 replicates diverged with max
//! step-gap 1.09e-4, first difference at the step after the first optimizer
//! update; `NSL_FLASH_BWD_CPU=1` alone drove it to 0.000e+00 while
//! `NSL_EMBEDDING_BWD_CPU=1` (1.39e-4) and `NSL_SDPA_FUSED_DISABLE=1`
//! (1.78e-4) did not.
//!
//! WHY THE EXISTING GATE MISSED IT, which is the more useful lesson.
//! `srbf16_e2e_deterministic_and_sane_gpu` was cited as proof of bit-identity
//! under this flag. Its fixture declares itself "deliberately attention-free",
//! so it never runs the offending kernel. A determinism gate certifies exactly
//! the ops its fixture executes, and nothing else.
//!
//! WHY THIS GATE IS NOT VACUOUS. Three independent guards, because each alone
//! is satisfiable by a broken setup:
//!   1. the deterministic arm's three replicates must be byte-identical;
//!   2. the deterministic arm must REPORT the CPU-reference route (a gate that
//!      only diffs loss streams passes when the routing never fired);
//!   3. the plain arm must dispatch the GPU atomic kernel AT THE PINNED
//!      GEOMETRY and must DIVERGE. Guard 3 is what stops a future edit from
//!      shrinking the fixture into a size where the race cannot express — an
//!      earlier draft at 2 heads / seq 256 ran the atomic kernel and was still
//!      bit-identical across replicates, which would have made guards 1 and 2
//!      pass with the bug fully present.
//!
//! The divergence assertion is a race, so it was measured rather than assumed:
//! at the pinned geometry, 6 replicates produced 6/6 distinct streams and 15/15
//! differing pairs. If it ever flakes, that is itself worth reading — either the
//! fixture shrank or the kernel gained a deterministic reduction.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

fn spawn_drain<R: Read + Send + 'static>(
    mut pipe: R,
    buf: Arc<Mutex<Vec<u8>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.lock().unwrap().extend_from_slice(&chunk[..n]),
            }
        }
    })
}

/// Run the fixture once. `NSL_FLASH_DEBUG=1` is always set: the routing
/// decision is the thing under test, so the gate must be able to see it.
fn run_fixture(tag: &str, deterministic: bool) -> RunOutput {
    run_fixture_env(tag, deterministic, &[])
}

fn run_fixture_env(tag: &str, deterministic: bool, envs: &[(&str, &str)]) -> RunOutput {
    let root = repo_root();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/deterministic_flash_bwd.nsl"),
    )
    .expect("deterministic_flash_bwd fixture missing")
    .replace("# GPU_PLACEMENT", "m.to(cuda)");

    let tmp = std::env::temp_dir().join(format!("nsl_detflash_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("deterministic_flash_bwd.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "7", "--source-ad"]);
    if deterministic {
        cmd.arg("--deterministic");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_FLASH_DEBUG", "1")
        .envs(envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn nsl run");
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    // Drain concurrently with the wait: a compile step alone can exceed a pipe
    // buffer, and reading only after waiting deadlocks (bugs.md, 2026-07-18).
    let out_reader = spawn_drain(child.stdout.take().unwrap(), out_buf.clone());
    let err_reader = spawn_drain(child.stderr.take().unwrap(), err_buf.clone());
    let status = child.wait().expect("wait nsl run");
    let _ = out_reader.join();
    let _ = err_reader.join();

    // Bind the guards to locals: building the struct from
    // `String::from_utf8_lossy(&buf.lock().unwrap())` inline keeps the
    // MutexGuard temporaries alive past the end of the block.
    let stdout = {
        let g = out_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    let stderr = {
        let g = err_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    RunOutput {
        success: status.success(),
        stdout,
        stderr,
    }
}

/// The printed loss stream, as TEXT. Comparing the rendered digits rather than
/// parsed f64s keeps the check bit-exact — two distinct floats can round to the
/// same parsed value under a lossy parse, and "bit-identical" is the claim.
fn loss_text(stdout: &str) -> Vec<String> {
    let seg = stdout
        .split("LOSS_STREAM_BEGIN")
        .nth(1)
        .and_then(|s| s.split("LOSS_STREAM_END").next())
        .unwrap_or("");
    seg.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("tensor(["))
        .map(str::to_string)
        .collect()
}

/// The routing witness, read from the program's own STDOUT.
///
/// This used to scrape an `NSL_FLASH_DEBUG` stderr line, which review correctly
/// called out: the runtime counter added alongside the fix was exported but
/// unreachable (no builtin registration, and `nsl run` execs the program as a
/// separate process, so a test linking nsl-runtime reads its own always-zero
/// copy). The counter is now a real builtin — `flash_bwd_det_routed_count()` —
/// and the fixture prints it, so the gate asserts the value the runtime
/// actually incremented rather than the presence of a debug string.
fn det_route_count(stdout: &str) -> usize {
    stdout
        .split("DET_ROUTED")
        .nth(1)
        .and_then(|s| s.lines().map(str::trim).find(|l| !l.is_empty()))
        .and_then(|l| {
            l.trim_start_matches("tensor([")
                .split(|c: char| !c.is_ascii_digit())
                .find(|p| !p.is_empty())
                .and_then(|d| d.parse::<usize>().ok())
        })
        .unwrap_or_else(|| panic!("no DET_ROUTED witness in stdout:\n{stdout}"))
}

#[test]
#[ignore = "requires CUDA GPU"]
fn deterministic_flag_makes_flash_backward_bit_reproducible() {
    // ── the arm under test: --deterministic must be bit-reproducible ──────
    let mut det_runs = Vec::new();
    for r in 0..3 {
        let out = run_fixture(&format!("det{r}"), true);
        assert!(
            out.success,
            "deterministic run {r} failed:\nstdout:\n{}\nstderr:\n{}",
            out.stdout, out.stderr
        );
        let losses = loss_text(&out.stdout);
        assert!(
            losses.len() >= 8,
            "deterministic run {r} produced {} losses, expected >= 8 — the \
             fixture did not train:\n{}",
            losses.len(),
            out.stdout
        );
        // Guard 2: the route must actually have been taken. Without this the
        // test passes when --deterministic silently stops routing and the
        // kernel merely happens to agree.
        assert!(
            det_route_count(&out.stdout) > 0,
            "deterministic run {r} never reported the CPU-reference backward \
             route — --deterministic is no longer reaching the flash backward, \
             so any bit-identity below is accidental:\n{}",
            out.stderr
        );
        det_runs.push(losses);
    }
    assert_eq!(
        det_runs[0], det_runs[1],
        "--deterministic is not bit-reproducible: replicates 0 and 1 differ"
    );
    assert_eq!(
        det_runs[0], det_runs[2],
        "--deterministic is not bit-reproducible: replicates 0 and 2 differ"
    );

    // ── the control arm: without the flag this MUST still diverge ─────────
    // Guard 3. If this arm ever goes bit-identical, the gate above has stopped
    // proving anything: either the fixture shrank below the size where the
    // atomic race expresses (an earlier draft at 2 heads / seq 256 did exactly
    // that), or the phase-2 kernel gained a deterministic reduction — in which
    // case this gate should be revisited rather than deleted.
    let mut plain_runs = Vec::new();
    let mut plain_stderr = String::new();
    for r in 0..3 {
        let out = run_fixture(&format!("plain{r}"), false);
        assert!(
            out.success,
            "plain run {r} failed:\nstdout:\n{}\nstderr:\n{}",
            out.stdout, out.stderr
        );
        assert_eq!(
            det_route_count(&out.stdout),
            0,
            "plain run {r} took the deterministic CPU route without \
             --deterministic — the routing predicate is inverted or the env is \
             dirty:\n{}",
            out.stderr
        );
        if r == 0 {
            plain_stderr = out.stderr.clone();
        }
        plain_runs.push(loss_text(&out.stdout));
    }

    // The pinned geometry, read off the kernel's own dispatch line. This is
    // what stops a future edit from shrinking the fixture into vacuity while
    // every other assertion still passes.
    let dispatch = plain_stderr
        .lines()
        .find(|l| l.contains("[flash-bwd] GPU backward dispatched"))
        .unwrap_or_else(|| {
            panic!(
                "no GPU flash backward was dispatched without --deterministic, \
                 so this fixture does not exercise the kernel the gate exists \
                 to certify:\n{plain_stderr}"
            )
        });
    for want in ["heads=8", "seq=512", "head_dim=64", "blocks=(32,32)"] {
        assert!(
            dispatch.contains(want),
            "flash backward dispatched at the wrong geometry: expected {want} \
             in `{dispatch}`. The fixture's contention (batch 2 x 8 heads x 16 \
             kv tiles = 256 CTAs all atomically accumulating into the same dQ \
             rows) is measured, not incidental — at 2 heads / seq 256 that is \
             32 CTAs, the kernel runs, and it is still bit-identical, which \
             would make this whole gate vacuous"
        );
    }

    let distinct = plain_runs.iter().collect::<std::collections::HashSet<_>>();
    assert!(
        distinct.len() > 1,
        "WITHOUT --deterministic all 3 replicates were byte-identical. Measured \
         at this geometry: 6/6 distinct streams, 15/15 differing pairs. Either \
         the fixture shrank below the size where the dK/dV atomic race \
         expresses, or the phase-2 kernel is now deterministic — in both cases \
         the positive arm above has stopped being evidence and this gate needs \
         re-deriving, not deleting"
    );
}

/// The `NSL_FLASH_BWD_CPU` escape hatch is TRI-STATE, and each state is pinned
/// here because the middle one (unset) is the only one the main test covers.
///
/// The routing is correct by default but costs ~36x per step at Coder-50M, and
/// several existing GPU gates pass `--deterministic` for reasons unrelated to
/// attention. `=0` buys their speed back at the price of the guarantee, so it
/// has to be LOUD — a silent opt-out would reintroduce exactly the situation
/// this campaign fixed: a run that believes it is reproducible and is not.
///
/// Note what is asserted and what is not. The warning and the route count are
/// deterministic facts about the routing decision. Whether the opted-out run
/// then actually DIVERGES is a race, and the main test already carries that
/// assertion once, where its reliability was measured (6/6 distinct streams).
/// Asserting it twice would double the flake surface for no extra coverage.
#[test]
#[ignore = "requires CUDA GPU"]
fn flash_bwd_cpu_env_overrides_deterministic_loudly() {
    // `=0` under --deterministic: opt out, and say so.
    let out = run_fixture_env("optout", true, &[("NSL_FLASH_BWD_CPU", "0")]);
    assert!(
        out.success,
        "opt-out run failed:\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    assert_eq!(
        det_route_count(&out.stdout),
        0,
        "NSL_FLASH_BWD_CPU=0 did not disable the deterministic route — the \
         opt-out is dead, so anyone setting it silently keeps paying the ~36x \
         cost they set it to avoid:\n{}",
        out.stderr
    );
    assert!(
        out.stderr.contains("NSL_FLASH_BWD_CPU=0 overrides --deterministic"),
        "opting out of --deterministic's bit-reproducibility printed no \
         warning. A run that quietly loses the guarantee it was asked for is \
         the bug this gate exists to prevent:\n{}",
        out.stderr
    );

    // `=1` WITHOUT --deterministic: force the CPU reference anyway. This is the
    // pre-existing contract the fix had to preserve, not replace.
    let out = run_fixture_env("force", false, &[("NSL_FLASH_BWD_CPU", "1")]);
    assert!(
        out.success,
        "forced-CPU run failed:\nstdout:\n{}\nstderr:\n{}",
        out.stdout, out.stderr
    );
    assert!(
        det_route_count(&out.stdout) > 0,
        "NSL_FLASH_BWD_CPU=1 no longer forces the CPU reference backward \
         without --deterministic — the original knob regressed:\n{}",
        out.stderr
    );
}
