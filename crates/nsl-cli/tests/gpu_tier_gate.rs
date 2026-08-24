//! GPU-free gates for the certification-tier front-end (roadmap item 14):
//! scripts/gpu-tier.sh, scripts/gpu-guard.sh, and the guard wiring in
//! scripts/gpu-cert.sh, tools/gpu-test.sh and the models/benchmarks campaign
//! drivers.
//!
//! Everything here runs WITHOUT a device: nvidia-smi is a PATH shim whose
//! compute-app listing the test controls, and the lock path is per-test. The
//! refusal behaviour under test exists because of a real incident: a sweep
//! that printed "WARNING: GPU still at 24459 MiB" and proceeded lost three
//! measurements and two whole LR-sweep arms to an orphaned run's resident
//! allocation. A warning that does not gate is not a guard, so these tests
//! pin the GATING — refusals fire (with the offender named), and equally
//! important, they do NOT fire for an idle device, a sub-threshold desktop
//! compositor, or operations that never touch the device.
//!
//! Linux-only: the guard's mutual exclusion is util-linux `flock`, and the
//! certification lane's reference box is Linux. macOS CI has no `flock`
//! binary and no CUDA device, so these gates would test a configuration the
//! lane never runs in.
#![cfg(target_os = "linux")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nsl-cli is two levels below the repo root")
        .to_path_buf()
}

/// A PATH shim for nvidia-smi plus an isolated lock path. The shim answers
/// `--query-compute-apps` with the contents of `apps.csv` (written per test)
/// and any other query with a one-GPU listing.
struct FakeDevice {
    dir: tempfile::TempDir,
    path_env: String,
}

impl FakeDevice {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let bin = dir.path().join("bin");
        fs::create_dir(&bin).expect("mkdir bin");
        let shim = bin.join("nvidia-smi");
        fs::write(
            &shim,
            "#!/bin/sh\ncase \"$*\" in\n  *query-compute-apps*) cat \"$(dirname \"$0\")/../apps.csv\";;\n  *) echo \"GPU 0: Fake Device\";;\nesac\n",
        )
        .expect("write shim");
        let mut perms = fs::metadata(&shim).expect("stat").permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        fs::set_permissions(&shim, perms).expect("chmod");
        let path_env = format!(
            "{}:{}",
            bin.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let dev = FakeDevice { dir, path_env };
        dev.set_compute_apps("");
        dev
    }

    fn set_compute_apps(&self, csv: &str) {
        fs::write(self.dir.path().join("apps.csv"), csv).expect("write apps.csv");
    }

    fn lock_path(&self) -> PathBuf {
        self.dir.path().join("nsl-gpu.lock")
    }

    /// Run `program args...` from the repo root under the shimmed PATH and
    /// the isolated lock, with any inherited guard/lock state scrubbed.
    fn run(&self, program: &str, args: &[&str]) -> Output {
        Command::new(repo_root().join(program))
            .args(args)
            .current_dir(repo_root())
            .env("PATH", &self.path_env)
            .env("NSL_GPU_LOCK", self.lock_path())
            .env_remove("NSL_GPU_LOCK_HELD")
            .env_remove("NSL_GPU_GUARD")
            .env_remove("NSL_GPU_GUARD_THRESHOLD_MIB")
            .output()
            .expect("spawn")
    }
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn guard_refuses_a_foreign_compute_process_and_names_it() {
    let dev = FakeDevice::new();
    dev.set_compute_apps("12345, /tmp/nsl_run_999/prog, 22000\n");
    let out = dev.run("scripts/gpu-guard.sh", &["check"]);
    assert!(
        !out.status.success(),
        "a 22 GB foreign compute process must refuse, got: {}",
        stderr_of(&out)
    );
    let err = stderr_of(&out);
    assert!(
        err.contains("12345") && err.contains("REFUSING"),
        "the refusal must NAME the offender so the operator can kill it: {err}"
    );
}

#[test]
fn guard_passes_an_idle_device_and_a_subthreshold_compositor() {
    let dev = FakeDevice::new();
    // Empty listing: nothing on the device.
    let out = dev.run("scripts/gpu-guard.sh", &["check"]);
    assert!(out.status.success(), "idle must pass: {}", stderr_of(&out));
    // The measured idle-desktop reality on the reference box: the Wayland
    // compositor holds ~125 MiB as a compute app. Refusing on it would make
    // the guard fire on every desktop boot — and a guard that always fires
    // gets bypassed, which is how warnings-that-don't-gate are born.
    dev.set_compute_apps("1140, /usr/bin/kwin_wayland, 125\n");
    let out = dev.run("scripts/gpu-guard.sh", &["check"]);
    assert!(
        out.status.success(),
        "a sub-threshold compositor must not refuse: {}",
        stderr_of(&out)
    );
}

#[test]
fn guard_refuses_a_non_numeric_memory_reading() {
    // "[N/A]" means the guard cannot PROVE the device is idle; cannot-prove
    // refuses for the same reason a missing nvidia-smi does.
    let dev = FakeDevice::new();
    dev.set_compute_apps("777, /some/proc, [N/A]\n");
    let out = dev.run("scripts/gpu-guard.sh", &["check"]);
    assert!(
        !out.status.success(),
        "an unprovable reading must refuse: {}",
        stderr_of(&out)
    );
}

#[test]
fn guard_run_excludes_a_second_cooperating_run() {
    let dev = FakeDevice::new();
    let mut holder = Command::new(repo_root().join("scripts/gpu-guard.sh"))
        .args(["run", "--", "sleep", "20"])
        .current_dir(repo_root())
        .env("PATH", &dev.path_env)
        .env("NSL_GPU_LOCK", dev.lock_path())
        .env_remove("NSL_GPU_LOCK_HELD")
        .spawn()
        .expect("spawn holder");
    // The guard writes holder metadata only AFTER acquiring the flock, so
    // metadata-visible == lock-held.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let held = fs::read_to_string(dev.lock_path())
            .map(|s| s.contains("command"))
            .unwrap_or(false);
        if held {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "holder never acquired the lock"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let out = dev.run("scripts/gpu-guard.sh", &["run", "--", "true"]);
    let err = stderr_of(&out);
    assert!(
        !out.status.success(),
        "a second guarded run must refuse while the first holds the lock"
    );
    assert!(
        err.contains("another guarded GPU run") && err.contains("sleep 20"),
        "the refusal must show WHO holds the lock: {err}"
    );

    holder.kill().ok();
    holder.wait().ok();
}

#[test]
fn guard_run_gives_the_workload_its_own_process_group() {
    // The incident mechanism: killing a launcher pid left its `nsl run`
    // child holding 22 GB. The guard runs the workload as a process-GROUP
    // leader (setsid) and forwards signals to the group, so the whole tree
    // dies together. A group leader's pgid equals its own pid.
    let dev = FakeDevice::new();
    let out = dev.run(
        "scripts/gpu-guard.sh",
        &["run", "--", "bash", "-c", "echo $$ $(ps -o pgid= -p $$)"],
    );
    assert!(out.status.success(), "wrapped run failed: {}", stderr_of(&out));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut it = stdout.split_whitespace();
    let (pid, pgid) = (it.next().expect("pid"), it.next().expect("pgid"));
    assert_eq!(
        pid, pgid,
        "the workload must lead its own process group, got pid={pid} pgid={pgid}"
    );
}

#[test]
fn tier_dispatch_maps_each_tier_to_its_runner_and_refuses_unknown() {
    let dev = FakeDevice::new();
    for (tier, runner) in [
        ("smoke", "tools/gpu-test.sh"),
        ("certify", "scripts/gpu-cert.sh --run"),
        ("endurance", "models/benchmarks/endurance_1b.py"),
    ] {
        let out = dev.run("scripts/gpu-tier.sh", &[tier, "--dry-run"]);
        assert!(out.status.success(), "{tier} --dry-run failed");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains(runner),
            "tier {tier} must dispatch to {runner}, printed: {stdout}"
        );
    }
    // Pass-through arguments reach the runner.
    let out = dev.run("scripts/gpu-tier.sh", &["certify", "--dry-run", "--tier", "all"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--tier all"),
        "extra args must pass through: {stdout}"
    );
    // A typo'd tier must not read as a green run of nothing.
    let out = dev.run("scripts/gpu-tier.sh", &["nonsense"]);
    assert_eq!(out.status.code(), Some(2), "unknown tier must exit 2");
}

#[test]
fn cert_lane_and_canary_refuse_before_any_work_on_a_busy_device() {
    // The wiring half: gpu-cert.sh --run and tools/gpu-test.sh must consult
    // the guard BEFORE building or running anything. With the device busy,
    // both must die with the guard's message — proving the guard is invoked
    // at all, which no amount of testing gpu-guard.sh alone can show.
    let dev = FakeDevice::new();
    dev.set_compute_apps("4242, /tmp/nsl_run_1/prog, 21000\n");
    for script in ["scripts/gpu-cert.sh", "tools/gpu-test.sh"] {
        let args: &[&str] = if script.ends_with("gpu-cert.sh") {
            &["--run"]
        } else {
            &[]
        };
        let out = dev.run(script, args);
        let err = stderr_of(&out);
        assert!(
            !out.status.success(),
            "{script} must refuse on a busy device"
        );
        assert!(
            err.contains("gpu-guard: REFUSING") && err.contains("4242"),
            "{script} must surface the guard's refusal, got: {err}"
        );
    }
}

#[test]
fn listing_operations_stay_unguarded_on_a_busy_device() {
    // The refusal must be SPECIFIC to operations that touch the device.
    // A guard that also blocks `--list` on a busy box teaches operators to
    // export NSL_GPU_GUARD=0, and the guard is then dead on the day it
    // matters.
    let dev = FakeDevice::new();
    dev.set_compute_apps("4242, /tmp/nsl_run_1/prog, 21000\n");
    let out = dev.run("tools/gpu-test.sh", &["--list"]);
    assert!(
        out.status.success(),
        "--list runs nothing on the device and must not refuse: {}",
        stderr_of(&out)
    );
    let out = dev.run("scripts/gpu-tier.sh", &["smoke", "--dry-run"]);
    assert!(
        out.status.success(),
        "--dry-run must not refuse: {}",
        stderr_of(&out)
    );
}

#[test]
fn every_campaign_driver_calls_the_python_guard() {
    // Static wiring check across the fleet, paired with the EXECUTION check
    // below so it cannot rot into a comment-satisfiable tripwire.
    let root = repo_root();
    for script in [
        "endurance_1b.py",
        "p0_campaign.py",
        "srbf16_campaign.py",
        "muon_campaign.py",
        "matrix_bench.py",
        "residency_probe.py",
        "ad_differential_50m.py",
        "sr_differential_50m.py",
        "mfu_bench.py",
    ] {
        let text = fs::read_to_string(root.join("models/benchmarks").join(script))
            .unwrap_or_else(|e| panic!("read {script}: {e}"));
        assert!(
            text.contains("gpu_guard.acquire_or_refuse("),
            "{script} launches GPU work but never consults the guard"
        );
    }
}

#[test]
fn a_campaign_driver_actually_refuses_on_a_busy_device() {
    // Execution proof for the python twin, on the driver whose ancestor
    // caused the incident. A bogus --nsl path bounds the blast radius: if
    // the guard were ever skipped, the run dies immediately on a missing
    // binary — a different message than the one asserted here.
    let dev = FakeDevice::new();
    dev.set_compute_apps("4242, /tmp/nsl_run_1/prog, 21000\n");
    let out = Command::new("python3")
        .args([
            "models/benchmarks/endurance_1b.py",
            "--nsl",
            "/nonexistent/nsl",
            "--steps",
            "1",
        ])
        .current_dir(repo_root())
        .env("PATH", &dev.path_env)
        .env("NSL_GPU_LOCK", dev.lock_path())
        .env_remove("NSL_GPU_LOCK_HELD")
        .env_remove("NSL_GPU_GUARD")
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("python3 not on PATH — skipping the execution half");
            return;
        }
        Err(e) => panic!("spawn python3: {e}"),
    };
    let err = stderr_of(&out);
    assert!(
        !out.status.success(),
        "endurance_1b.py must refuse on a busy device"
    );
    assert!(
        err.contains("gpu-guard: REFUSING") && err.contains("4242"),
        "the python guard's refusal must fire before any launch: {err}"
    );
}
