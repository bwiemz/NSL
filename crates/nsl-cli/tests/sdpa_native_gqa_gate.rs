//! `scaled_dot_product_attention` takes grouped K/V (`kv_heads` dividing
//! `heads`), and `nsl.nn.gqa` now passes them grouped instead of expanding K
//! and V to every query head in NSL source.
//!
//! The claim pinned here: on every path that computes the same sums in the
//! same order, a program that passes grouped K/V trains BIT FOR BIT like the
//! program that expanded them. The fixture (`fixtures/sdpa_native_gqa.nsl`)
//! is compiled twice — `ATTN = GroupedQueryAttention` (the stdlib, grouped
//! K/V) and `ATTN = ExpandedGqa` (the stdlib's forward as it was, expanding
//! K/V before the call) — and the loss streams and trained weights are
//! compared as printed text:
//!
//! - tape AD and source AD on the host: the decomposed attention expands
//!   grouped K/V itself (`nsl_sdpa_expand_kv`, the same op chain), and the
//!   CPU reference backward computes per query head and sums each group with
//!   the expansion's adjoint;
//! - on the GPU under `--deterministic`: the tensor-core forward reads
//!   grouped K/V in place (each query head reads the same values as from the
//!   expanded copy), and the backward is the CPU reference;
//! - the previous attention kernels (`NSL_TC_ATTENTION=0`): the fused forward
//!   expands grouped K/V on the device;
//! - `NSL_SDPA_GQA_EXPAND=1` (the kill switch) with the tensor-core backward:
//!   K/V expanded, dK/dV per query head and summed with the expansion's
//!   adjoint, f32 and bf16 activation storage.
//!
//! The one path that is NOT bit-identical is the default tensor-core
//! backward: its dK/dV kernel sums each group's query heads in one f32
//! accumulator (and under bf16 storage rounds once instead of per query head
//! and again after the sum). The gate checks that it is the path taken (the
//! dispatch line names `kv_heads`), that the forward and therefore the first
//! loss are still identical, and that the streams stay close.

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Attn {
    Grouped,
    Expanded,
}

#[derive(Clone, Copy)]
struct Arm<'a> {
    attn: Attn,
    gpu: bool,
    source_ad: bool,
    deterministic: bool,
    bf16: bool,
    envs: &'a [(&'a str, &'a str)],
}

struct RunOutput {
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

fn run(tag: &str, arm: Arm<'_>) -> RunOutput {
    let root = repo_root();
    let attn = match arm.attn {
        Attn::Grouped => "GroupedQueryAttention",
        Attn::Expanded => "ExpandedGqa",
    };
    let mut src =
        std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/sdpa_native_gqa.nsl"))
            .expect("sdpa_native_gqa fixture missing")
            .replace("attn: ATTN = ATTN(", &format!("attn: {attn} = {attn}("));
    if arm.gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    if arm.bf16 {
        src = src.replace(
            "grad_accumulation=2):",
            "grad_accumulation=2, precision=\"bf16\"):",
        );
    }
    let tmp = std::env::temp_dir().join(format!("nsl_native_gqa_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("sdpa_native_gqa.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--seed", "11"]);
    if arm.source_ad {
        cmd.arg("--source-ad");
    }
    if arm.deterministic {
        cmd.arg("--deterministic");
    }
    cmd.arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_FLASH_DEBUG", "1")
        .envs(arm.envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn nsl run");
    let out_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let err_buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let out_reader = spawn_drain(child.stdout.take().unwrap(), out_buf.clone());
    let err_reader = spawn_drain(child.stderr.take().unwrap(), err_buf.clone());
    let status = child.wait().expect("wait nsl run");
    let _ = out_reader.join();
    let _ = err_reader.join();
    let stdout = {
        let g = out_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    let stderr = {
        let g = err_buf.lock().unwrap();
        String::from_utf8_lossy(&g).to_string()
    };
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        status.success(),
        "{tag} ({:?}) failed\nstdout:\n{stdout}\nstderr:\n{stderr}",
        arm.attn
    );
    RunOutput { stdout, stderr }
}

/// The printed lines between two markers, as text (bit-exact comparison:
/// distinct floats never print alike). Non-finite values fail the gate, so
/// two runs that both went NaN cannot read as identical.
fn section(stdout: &str, begin: &str, end: &str) -> Vec<String> {
    let lines: Vec<String> = stdout
        .split(begin)
        .nth(1)
        .and_then(|s| s.split(end).next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        // A device scalar prints as `tensor([..])`, a host one as the bare number.
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect();
    for l in &lines {
        let lower = l.to_ascii_lowercase();
        assert!(
            !lower.contains("nan") && !lower.contains("inf"),
            "non-finite value: {l:?}"
        );
    }
    lines
}

fn losses(out: &RunOutput) -> Vec<String> {
    let l = section(&out.stdout, "LOSS_STREAM_BEGIN", "LOSS_STREAM_END");
    assert_eq!(l.len(), 8, "8 micro-batch losses expected:\n{}", out.stdout);
    l
}

fn weights(out: &RunOutput) -> Vec<String> {
    let w = section(&out.stdout, "WEIGHTS_BEGIN", "WEIGHTS_END");
    assert_eq!(w.len(), 5, "5 weight sums expected:\n{}", out.stdout);
    w
}

fn value(line: &str) -> f64 {
    line.trim_start_matches("tensor([")
        .split(|c: char| c == ']' || c == ',')
        .next()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no value in {line:?}"))
}

/// Train the grouped and the expanded program in one configuration and
/// require the same loss stream and trained weights, bit for bit.
fn assert_identical(tag: &str, arm: Arm<'_>) -> (RunOutput, RunOutput) {
    let g = run(
        &format!("{tag}_g"),
        Arm {
            attn: Attn::Grouped,
            ..arm
        },
    );
    let e = run(
        &format!("{tag}_e"),
        Arm {
            attn: Attn::Expanded,
            ..arm
        },
    );
    assert_eq!(losses(&g), losses(&e), "{tag}: loss streams differ");
    assert_eq!(weights(&g), weights(&e), "{tag}: trained weights differ");
    (g, e)
}

const HOST: Arm<'static> = Arm {
    attn: Attn::Grouped,
    gpu: false,
    source_ad: false,
    deterministic: false,
    bf16: false,
    envs: &[],
};

#[test]
fn grouped_kv_trains_like_expanded_kv_under_tape_ad_on_the_host() {
    assert_identical("tape_host", HOST);
}

#[test]
fn grouped_kv_trains_like_expanded_kv_under_source_ad_on_the_host() {
    assert_identical(
        "src_host",
        Arm {
            source_ad: true,
            ..HOST
        },
    );
}

const GPU: Arm<'static> = Arm {
    attn: Attn::Grouped,
    gpu: true,
    source_ad: true,
    deterministic: true,
    bf16: false,
    envs: &[],
};

fn assert_tc_forward(out: &RunOutput) {
    assert!(
        out.stderr
            .contains("sdpa fused forward: launched (tensor-core kernels"),
        "the tensor-core forward did not run:\n{}",
        out.stderr
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn on_the_gpu_grouped_kv_is_the_expanded_computation_under_deterministic() {
    // Forward on the tensor-core kernels (grouped K/V read in place),
    // backward on the CPU reference (`--deterministic`).
    let (g, e) = assert_identical("gpu_det", GPU);
    assert_tc_forward(&g);
    assert_tc_forward(&e);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_previous_attention_kernels_take_grouped_kv() {
    let envs = [("NSL_TC_ATTENTION", "0")];
    let (g, _) = assert_identical("gpu_prev", Arm { envs: &envs, ..GPU });
    assert!(
        g.stderr.contains("sdpa fused forward: launched (segmask=0"),
        "the previous fused forward did not run on grouped K/V:\n{}",
        g.stderr
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tape_ad_on_the_gpu_takes_grouped_kv() {
    // The embedding backward's atomics are the only other nondeterministic
    // op of the fixture under tape AD; keep it on the host.
    let envs = [("NSL_EMBEDDING_BWD_CPU", "1")];
    assert_identical(
        "gpu_tape",
        Arm {
            source_ad: false,
            deterministic: false,
            envs: &envs,
            ..GPU
        },
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_kill_switch_reproduces_the_expanded_tensor_core_backward() {
    for bf16 in [false, true] {
        let envs = [("NSL_FLASH_BWD_CPU", "0"), ("NSL_SDPA_GQA_EXPAND", "1")];
        let (g, _) = assert_identical(
            &format!("gpu_kill_{bf16}"),
            Arm {
                bf16,
                envs: &envs,
                ..GPU
            },
        );
        assert_tc_forward(&g);
        assert!(
            g.stderr
                .contains("[flash-bwd] tensor-core backward dispatched")
                && g.stderr.contains("gqa_expanded=true"),
            "the kill switch did not run the tensor-core backward on expanded K/V:\n{}",
            g.stderr
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_tensor_core_backward_sums_each_group_in_place() {
    for bf16 in [false, true] {
        let envs = [("NSL_FLASH_BWD_CPU", "0")];
        let arm = Arm {
            bf16,
            envs: &envs,
            ..GPU
        };
        let g = run(
            &format!("gpu_native_{bf16}_g"),
            Arm {
                attn: Attn::Grouped,
                ..arm
            },
        );
        let e = run(
            &format!("gpu_native_{bf16}_e"),
            Arm {
                attn: Attn::Expanded,
                ..arm
            },
        );
        assert!(
            g.stderr
                .contains("[flash-bwd] tensor-core backward dispatched")
                && g.stderr.contains("kv_heads=2, gqa_expanded=false"),
            "grouped K/V did not reach the tensor-core backward in place:\n{}",
            g.stderr
        );
        let (lg, le) = (losses(&g), losses(&e));
        // The forward is the same arithmetic: the first micro-batch, before
        // any update, has the same loss.
        assert_eq!(lg[0], le[0], "bf16={bf16}: first loss differs");
        // Afterwards dK/dV differ in summation order only (and, under bf16
        // storage, in the per-head rounding the expanded program adds), and
        // AdamW at this fixture's lr 0.01 amplifies that over the updates as
        // the loss falls towards 0.1: rounding-level at f32, ~1% at bf16.
        let tol = if bf16 { 2e-2 } else { 1e-3 };
        for (a, b) in lg.iter().zip(&le) {
            let (a, b) = (value(a), value(b));
            assert!(
                (a - b).abs() <= tol * b.abs(),
                "bf16={bf16}: losses {a} vs {b}"
            );
        }
    }
}
