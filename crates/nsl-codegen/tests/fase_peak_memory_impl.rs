//! Implementation of the peak-memory regression test.
//!
//! Uses `cargo run -p nsl-cli` to invoke `nsl build --shared-lib`, then
//! libloading to dlopen the produced library and call the exported symbols
//! directly in-process.
//!
//! This file is `#[path]`-included by `fase_peak_memory.rs` only when the
//! `test-hooks` feature is active.  When cargo discovers this file directly
//! (as an independent integration test crate), the `#![cfg]` below ensures
//! the test is skipped on non-test-hooks builds, keeping the default test run clean.
#![cfg(feature = "test-hooks")]

use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Workspace root: CARGO_MANIFEST_DIR/../../..
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fase_peak_memory.nsl")
}

/// Locate the nsl_runtime static lib that was built with `test-hooks` active.
///
/// The `find_feature_matched_runtime_lib` heuristic in `linker.rs` only
/// checks the CUDA feature, so when both test-hooks and non-test-hooks libs
/// coexist in `target/debug/deps/` it might select the wrong one.
///
/// We scan the fingerprints ourselves, filter to only those that list
/// "test-hooks" in their features, and return the newest matching lib.
/// If none is found we return `None` and let the linker use its own heuristic.
fn find_test_hooks_runtime_lib() -> Option<PathBuf> {
    // Walk from CARGO_MANIFEST_DIR up to find target/debug/deps.
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dir = start.clone();
    for _ in 0..6 {
        let fingerprints_dir = dir.join("target").join("debug").join(".fingerprint");
        let deps_dir = dir.join("target").join("debug").join("deps");
        if fingerprints_dir.is_dir() && deps_dir.is_dir() {
            return find_test_hooks_lib_in(&fingerprints_dir, &deps_dir);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

/// Return true iff the fingerprint JSON content has "test-hooks" in the
/// *active* features field (not in declared_features).
///
/// The fingerprint JSON format for the `features` field is:
///   `"features":"[\"default\", \"test-hooks\"]"`
/// We extract the substring after the first `"features":"` and before the
/// first closing `"` that ends the escaped array, then check for test-hooks.
fn active_features_include_test_hooks(json: &str) -> bool {
    // Find the "features": key (not "declared_features")
    let marker = "\"features\":\"";
    let Some(start) = json.find(marker) else {
        return false;
    };
    let after_key = &json[start + marker.len()..];
    // The value ends at the first unescaped `"` that closes the string.
    // Since the content is an escaped JSON array like `[\"default\"]`,
    // we find the `]"` sequence which marks the end of the value.
    let end = after_key
        .find("]\"")
        .map(|i| i + 1)
        .unwrap_or(after_key.len());
    let features_value = &after_key[..end];
    features_value.contains("test-hooks")
}

fn find_test_hooks_lib_in(fingerprints_dir: &Path, deps_dir: &Path) -> Option<PathBuf> {
    let ext = if cfg!(windows) { "lib" } else { "a" };
    let prefix = if cfg!(windows) {
        "nsl_runtime-"
    } else {
        "libnsl_runtime-"
    };

    let entries = std::fs::read_dir(fingerprints_dir).ok()?;
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        let Some(hash) = file_name_str.strip_prefix("nsl-runtime-") else {
            continue;
        };

        let fingerprint_path = entry.path().join("lib-nsl_runtime.json");
        let Ok(content) = std::fs::read_to_string(&fingerprint_path) else {
            continue;
        };

        // Only keep libs where "test-hooks" appears in the *active* features
        // (the "features" field, not "declared_features").
        // The JSON looks like: {"features":"[\"default\", \"test-hooks\"]",...}
        // We extract the value of the "features" key by finding the first
        // occurrence of `"features":"` and reading until the closing `"` that
        // follows the array literal.
        if !active_features_include_test_hooks(&content) {
            continue;
        }

        // Find the corresponding .lib / .a in deps.
        let lib_name = format!("{}{}.{}", prefix, hash, ext);
        let lib_path = deps_dir.join(&lib_name);
        if lib_path.exists() {
            let mtime = lib_path
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            candidates.push((mtime, lib_path));
        }
    }

    // Return the most recently modified test-hooks lib.
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    candidates.into_iter().next().map(|(_, p)| p)
}

/// Invoke `cargo run -p nsl-cli -- build --shared-lib [--source-ad] <fixture> -o <out>`
/// and return the .dll/.so/.dylib path.
///
/// Uses `env!("CARGO")` + `--manifest-path` (same pattern as
/// `fase_numerical_validation.rs`) so the test works whether invoked via
/// `cargo test` or as a standalone binary, and the correct feature flags are
/// forwarded automatically.
///
/// If a test-hooks runtime lib can be located in `target/debug/deps/`, it is
/// passed via `NSL_RUNTIME_LIB_PATH_OVERRIDE` to ensure the linker inside
/// the spawned nsl-cli process picks the right library.
fn build_fixture(source_ad: bool, out_dir: &Path) -> PathBuf {
    let fixture = fixture_path();
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let ext = if cfg!(windows) {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let suffix = if source_ad { "source_ad" } else { "tape_ad" };
    let lib_name = if cfg!(windows) {
        format!("fase_peak_memory_{}.{}", suffix, ext)
    } else {
        format!("libfase_peak_memory_{}.{}", suffix, ext)
    };
    let out_path = out_dir.join(&lib_name);

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli"])
        .arg("--features")
        .arg("test-hooks")
        .arg("--")
        .arg("build")
        .arg("--shared-lib")
        .arg(&fixture)
        .arg("-o")
        .arg(&out_path)
        .env("NSL_STDLIB_PATH", &stdlib_path);

    if source_ad {
        cmd.arg("--source-ad");
    }

    // Ensure the spawned nsl-cli uses the test-hooks runtime lib.
    // linker.rs's find_feature_matched_runtime_lib only checks the CUDA
    // feature, so it can select the wrong lib when both test-hooks and
    // non-test-hooks builds coexist.
    if let Some(lib_path) = find_test_hooks_runtime_lib() {
        eprintln!(
            "[fase_peak_memory] setting NSL_RUNTIME_LIB_PATH_OVERRIDE={}",
            lib_path.display()
        );
        cmd.env("NSL_RUNTIME_LIB_PATH_OVERRIDE", &lib_path);
    }

    let output = cmd
        .output()
        .expect("failed to spawn cargo run nsl-cli build");

    assert!(
        output.status.success(),
        "nsl build --shared-lib failed (source_ad={})\nstdout: {}\nstderr: {}",
        source_ad,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        out_path.exists(),
        "expected shared library at {:?} but was not produced",
        out_path
    );
    out_path
}

/// Load the shared library, reset the peak counter, run `main`, read the peak.
///
/// Each loaded library has its own copy of the peak-tracking state because the
/// test-hooks atomics live in the runtime which is statically linked into the
/// shared library.
fn run_and_measure_peak(dll_path: &Path) -> usize {
    unsafe {
        let lib = Library::new(dll_path)
            .unwrap_or_else(|e| panic!("load shared library {:?}: {}", dll_path, e));

        let reset: Symbol<unsafe extern "C" fn()> = lib
            .get(b"nsl_cpu_peak_reset\0")
            .expect("nsl_cpu_peak_reset symbol not exported");
        let peak: Symbol<unsafe extern "C" fn() -> i64> = lib
            .get(b"nsl_cpu_peak_bytes\0")
            .expect("nsl_cpu_peak_bytes symbol not exported");
        let main_fn: Symbol<unsafe extern "C" fn()> =
            lib.get(b"main\0").expect("main symbol not exported");

        reset();
        main_fn();
        peak() as usize
    }
}

#[test]
fn source_ad_peak_is_lower_than_tape_ad() {
    let out_dir = tempfile::TempDir::new().expect("tempdir");

    let tape_dll = build_fixture(/*source_ad=*/ false, out_dir.path());
    let source_dll = build_fixture(/*source_ad=*/ true, out_dir.path());

    let peak_tape = run_and_measure_peak(&tape_dll);
    let peak_source = run_and_measure_peak(&source_dll);

    // Fixture is a 2-parameter MLP with two 256x256 weight matrices (~512 KB
    // gradient each).  Tape-AD holds both live simultaneously (~1 MB); the
    // source-AD consume-per-param hook holds only one at a time (~512 KB
    // peak gradient memory).  Expected delta: >= 400 KB, leaving margin for
    // ~100 KB of bookkeeping noise.
    const MIN_DELTA_BYTES: usize = 400_000;

    println!("peak_tape   = {} bytes", peak_tape);
    println!("peak_source = {} bytes", peak_source);
    println!(
        "delta       = {} bytes",
        peak_tape.saturating_sub(peak_source)
    );

    assert!(
        peak_tape >= peak_source + MIN_DELTA_BYTES,
        "expected source-AD peak at least {} bytes below tape-AD; \
         got tape={} source={} (delta={})",
        MIN_DELTA_BYTES,
        peak_tape,
        peak_source,
        peak_tape.saturating_sub(peak_source),
    );
}
