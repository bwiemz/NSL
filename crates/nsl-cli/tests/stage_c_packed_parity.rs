//! PCA Stage C — packed-attention parity gates.
//!
//! Three differentials over the shared fixture
//! `fixtures/stage_c_packed_gqa.nsl` (2-block GQA, head_dim=32, seq=64,
//! DataLoader packing=true):
//!
//! * `packed_matches_masked_on_cpu` (ungated) — `forward_packed` must
//!   reproduce `forward_masked` on CPU, where the fused FFI declines and
//!   both run the decomposed additive-mask chain. The FORWARD is
//!   op-identical; the BACKWARD differs by construction (Stage B: per-op
//!   decomposed adjoints in f64; Stage C: the fused-extract op's
//!   segment-aware f32 CPU reference), so agreement here validates the
//!   Stage-C CPU reference backward against the Stage-B oracle.
//!
//! * `packed_fused_matches_decomposed_on_gpu` (cuda, ignored) — the real
//!   Stage C gate: the same packed program on the GPU with the fused
//!   segment-masked flash kernels ON (default) vs OFF
//!   (`NSL_SDPA_FUSED_DISABLE=1`), checkpoint-compared. Launch-level proof
//!   comes from the once-per-process `sdpa fused forward: launched` marker.
//!
//! * `wggo_reports_fused_consumption_on_gpu` (cuda, ignored) — under
//!   `--pretrain-optimized` the plan's segment_id packing decision must be
//!   reported consumed through the Stage-C channel:
//!   `-> fused segment-masked flash kernel (Stage C plain family)`.

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

fn fixture_src() -> String {
    std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/stage_c_packed_gqa.nsl"),
    )
    .expect("stage_c_packed_gqa.nsl fixture missing")
}

/// Rewrite fixture markers by exact match; every replace asserts it fired.
///
/// `continuous_pos` swaps the DataLoader's per-document position_ids for a
/// continuous 0..seq arange — `forward_with_positions` with continuous
/// positions is bitwise-identical rope to `forward()`/`forward_masked`'s
/// internal arange, which is what makes the packed==masked differential
/// like-for-like now that packed training resets RoPE positions per
/// document by default.
fn program(masked_form: bool, gpu: bool, continuous_pos: bool, save_path: &Path) -> String {
    let mut src = fixture_src();
    if masked_form {
        // forward_packed is 5-arg (no dense mask — masking derives from
        // seg); the fixture still plumbs `mask` so this rewrite to the
        // Stage-B oracle has it in scope.
        let from = "self.attn.forward_packed(rmsnorm(h, self.norm_a, 0.00001), seg, pos, training)";
        let to = "self.attn.forward_masked(rmsnorm(h, self.norm_a, 0.00001), mask, training)";
        assert!(src.contains(from), "packed call marker not found — resync fixture");
        src = src.replace(from, to);
    }
    if continuous_pos {
        let from = "batch.position_ids  # POSITION_SOURCE";
        let to = "arange(0.0, 64.0).reshape([1, 64]).expand([2, 64]).contiguous()";
        assert!(src.contains(from), "position-source marker not found — resync fixture");
        src = src.replace(from, to);
    }
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    assert!(src.contains("STAGE_C_SAVE_PATH"));
    src.replace("STAGE_C_SAVE_PATH", &save_path.display().to_string())
}

struct RunOutput {
    losses: Vec<f64>,
    stderr: String,
    success: bool,
    stdout: String,
}

fn run_program(source: &str, tag: &str, cuda: bool, extra_env: &[(&str, &str)], extra_args: &[&str]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!(
        "nsl_stage_c_parity_{tag}_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("stage_c.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-q");
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    let mut losses = Vec::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            t if in_stream => {
                // CPU losses print as bare floats; GPU losses print as
                // `tensor([X])` (device-resident scalar repr).
                let inner = t
                    .strip_prefix("tensor([")
                    .and_then(|s| s.strip_suffix("])"))
                    .unwrap_or(t);
                if let Ok(v) = inner.parse::<f64>() {
                    losses.push(v);
                }
            }
            _ => {}
        }
    }
    RunOutput { losses, stderr, success: output.status.success(), stdout }
}

fn assert_trains(r: &RunOutput, tag: &str) {
    assert!(r.success, "{tag} run failed:\n{}", r.stderr);
    assert!(
        r.losses.len() >= 8,
        "{tag}: expected a loss stream, got {} values\nstdout:\n{}",
        r.losses.len(),
        r.stdout
    );
    let first = r.losses[0];
    let last = *r.losses.last().unwrap();
    assert!(
        last.is_finite() && last > 0.0 && last < first,
        "{tag}: loss did not improve (first={first} last={last})"
    );
}

fn checkpoint_max_diff(a: &Path, b: &Path) -> (f64, String) {
    let ca = nslm::read(a);
    let cb = nslm::read(b);
    assert_eq!(
        ca.len(),
        cb.len(),
        "checkpoint param-count mismatch: {} vs {}",
        ca.len(),
        cb.len()
    );
    let mut max_diff = 0.0f64;
    let mut worst = String::new();
    for (name, va) in &ca {
        let vb = cb
            .get(name)
            .unwrap_or_else(|| panic!("param {name} missing from second checkpoint"));
        assert_eq!(va.len(), vb.len(), "param {name} length mismatch");
        for (x, y) in va.iter().zip(vb) {
            let d = (*x as f64 - *y as f64).abs();
            if d > max_diff {
                max_diff = d;
                worst = name.clone();
            }
        }
    }
    (max_diff, worst)
}

/// CPU: packed == masked. Forward chains are op-identical; the packed
/// BACKWARD runs the Stage-C segment-aware f32 CPU reference while masked
/// runs Stage-B per-op f64 adjoints — so the tolerance is f32-epsilon-ish
/// accumulated over ~16 optimizer steps, not exact.
#[test]
fn packed_matches_masked_on_cpu() {
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let save_packed = tmp.join(format!("stage_c_cpu_packed_{pid}.nslm"));
    let save_masked = tmp.join(format!("stage_c_cpu_masked_{pid}.nslm"));

    // Continuous positions on BOTH sides: forward_masked applies continuous
    // arange rope internally, so the packed side must match it to isolate
    // the attention-builtin differential from the (intentional) per-document
    // position reset. Reset consumption is proven separately below.
    let packed = run_program(&program(false, false, true, &save_packed), "cpu_packed", false, &[], &[]);
    assert_trains(&packed, "cpu packed");
    let masked = run_program(&program(true, false, true, &save_masked), "cpu_masked", false, &[], &[]);
    assert_trains(&masked, "cpu masked");

    let (max_diff, worst) = checkpoint_max_diff(&save_packed, &save_masked);
    assert!(
        max_diff < 1e-4,
        "packed vs masked CPU checkpoints diverged: max_diff={max_diff:.3e} at {worst}"
    );
    let _ = std::fs::remove_file(&save_packed);
    let _ = std::fs::remove_file(&save_masked);
}

/// CPU: per-document RoPE position reset is CONSUMED. Training with the
/// DataLoader's reset position_ids vs continuous positions must produce
/// materially different checkpoints on multi-document rows (the fixture
/// packs 8 docs per row, so most tokens' rotary phases change). A vacuous
/// pass here would mean forward_with_positions silently ignores its
/// argument — the position-space analog of the Tier-B launch-count proof.
#[test]
fn position_reset_changes_packed_training_on_cpu() {
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let save_reset = tmp.join(format!("stage_c_cpu_reset_{pid}.nslm"));
    let save_cont = tmp.join(format!("stage_c_cpu_cont_{pid}.nslm"));

    let reset = run_program(&program(false, false, false, &save_reset), "cpu_reset", false, &[], &[]);
    assert_trains(&reset, "cpu reset-positions");
    let cont = run_program(&program(false, false, true, &save_cont), "cpu_cont", false, &[], &[]);
    assert_trains(&cont, "cpu continuous-positions");

    let (max_diff, worst) = checkpoint_max_diff(&save_reset, &save_cont);
    assert!(
        max_diff > 1e-6,
        "reset vs continuous positions produced IDENTICAL checkpoints \
         (max_diff={max_diff:.3e} at {worst}) — position_ids are not being consumed"
    );
    let _ = std::fs::remove_file(&save_reset);
    let _ = std::fs::remove_file(&save_cont);
}

#[cfg(feature = "cuda")]
fn gpu_present() -> bool {
    Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// GPU: fused segment-masked kernels vs the decomposed fallback, same
/// program, same data. Checkpoints agree within a fused-flash-vs-decomposed
/// f32 tolerance; the once-per-process launch marker proves the fused path
/// actually fired (and that the kill-switch actually disabled it).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU (two real training runs)"]
fn packed_fused_matches_decomposed_on_gpu() {
    if !gpu_present() {
        eprintln!("[skip] no GPU visible");
        return;
    }
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let save_fused = tmp.join(format!("stage_c_gpu_fused_{pid}.nslm"));
    let save_plain = tmp.join(format!("stage_c_gpu_decomp_{pid}.nslm"));

    let fused = run_program(&program(false, true, false, &save_fused), "gpu_fused", true, &[], &[]);
    assert_trains(&fused, "gpu fused");
    assert!(
        fused.stderr.contains("sdpa fused forward: launched"),
        "fused run must print the first-launch marker:\n{}",
        fused.stderr
    );

    let plain = run_program(
        &program(false, true, false, &save_plain),
        "gpu_decomp",
        true,
        &[("NSL_SDPA_FUSED_DISABLE", "1")],
        &[],
    );
    assert_trains(&plain, "gpu decomposed");
    assert!(
        !plain.stderr.contains("sdpa fused forward: launched"),
        "disabled run must NOT launch the fused kernel:\n{}",
        plain.stderr
    );

    let (max_diff, worst) = checkpoint_max_diff(&save_fused, &save_plain);
    assert!(
        max_diff < 2e-2,
        "fused vs decomposed GPU checkpoints diverged: max_diff={max_diff:.3e} at {worst}"
    );
    // And they must not be trivially identical — that would mean the fused
    // path silently declined every step despite printing nothing.
    assert!(
        max_diff > 0.0,
        "checkpoints bit-identical — fused path likely never ran"
    );
    let _ = std::fs::remove_file(&save_fused);
    let _ = std::fs::remove_file(&save_plain);
}

/// GPU + WGGO: the plan's segment_id decision must be consumed through the
/// Stage-C fused channel and reported as such per layer.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU"]
fn wggo_reports_fused_consumption_on_gpu() {
    if !gpu_present() {
        eprintln!("[skip] no GPU visible");
        return;
    }
    let tmp = std::env::temp_dir();
    let save = tmp.join(format!("stage_c_gpu_wggo_{}.nslm", std::process::id()));
    let r = run_program(
        &program(false, true, false, &save),
        "gpu_wggo",
        true,
        &[],
        &["--pretrain-optimized"],
    );
    assert_trains(&r, "gpu wggo");
    assert!(
        r.stderr
            .contains("fused segment-masked flash kernel (Stage C plain family"),
        "plan consumption must report the Stage-C fused channel:\n{}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("packing_requires_packed_dataset"),
        "no packing-rejection noise expected:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_file(&save);
}

mod nslm {
    //! Minimal .nslm reader — faithful copy of the proven parser used by the
    //! 4.3 gate in pretrain_loss_decrease_gpu_e2e.rs (test `common/` modules
    //! don't cross crate boundaries).
    use std::collections::HashMap;
    use std::path::Path;

    struct Entry {
        name: String,
        dtype: String,
        offset: usize,
        nbytes: usize,
    }

    pub fn read(path: &Path) -> HashMap<String, Vec<f32>> {
        let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert!(buf.len() >= 16 && &buf[0..4] == b"NSLM", "bad magic in {path:?}");
        let header_size = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + header_size;
        let json = std::str::from_utf8(&buf[16..header_end]).expect("header utf8");
        let pad = (64 - (header_end % 64)) % 64;
        let data_start = header_end + pad;

        let mut out = HashMap::new();
        for e in parse_entries(json) {
            let slab = &buf[data_start + e.offset..data_start + e.offset + e.nbytes];
            let values: Vec<f32> = match e.dtype.as_str() {
                "f32" => slab
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "f64" => slab
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                other => panic!("unsupported dtype {other} in {path:?}"),
            };
            out.insert(e.name, values);
        }
        out
    }

    fn parse_entries(json: &str) -> Vec<Entry> {
        let mut entries = Vec::new();
        let mut cursor = 0;
        while let Some(name_idx) = json[cursor..].find("\"name\":\"") {
            let abs = cursor + name_idx + 8;
            let name_end = json[abs..].find('"').expect("malformed name");
            let name = json[abs..abs + name_end].to_string();
            cursor = abs + name_end;
            let dtype = extract_str(json, cursor, "dtype");
            let offset = extract_int(json, cursor, "offset");
            let nbytes = extract_int(json, cursor, "nbytes");
            entries.push(Entry { name, dtype, offset, nbytes });
        }
        entries
    }

    fn extract_str(json: &str, from: usize, key: &str) -> String {
        let pattern = format!("\"{key}\":\"");
        let abs = from + json[from..].find(&pattern).expect(key) + pattern.len();
        let end = json[abs..].find('"').expect(key);
        json[abs..abs + end].to_string()
    }

    fn extract_int(json: &str, from: usize, key: &str) -> usize {
        let pattern = format!("\"{key}\":");
        let abs = from + json[from..].find(&pattern).expect(key) + pattern.len();
        let end = json[abs..].find(|c: char| c == ',' || c == '}').expect(key);
        json[abs..abs + end].trim().parse().expect(key)
    }
}
