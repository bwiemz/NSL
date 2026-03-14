//! End-to-end smoke tests for NSL examples.
//! Compiles and runs .nsl files, compares stdout against expected baselines.

use std::process::Command;

/// Truncate floating-point numbers in text to 4 decimal places.
fn normalize_floats(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        // Look for float patterns: optional minus, digits, dot, digits
        if chars[i].is_ascii_digit() || (chars[i] == '-' && i + 1 < n && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            if chars[i] == '-' {
                i += 1;
            }
            // Consume integer part
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Check for decimal point
            if i < n && chars[i] == '.' && i + 1 < n && chars[i + 1].is_ascii_digit() {
                i += 1; // skip dot
                let decimal_start = i;
                while i < n && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let decimal_digits = i - decimal_start;
                if decimal_digits >= 5 {
                    // Truncate to 4 decimal places
                    let num_str: String = chars[start..i].iter().collect();
                    if let Ok(val) = num_str.parse::<f64>() {
                        result.push_str(&format!("{:.4}", val));
                    } else {
                        let s: String = chars[start..i].iter().collect();
                        result.push_str(&s);
                    }
                } else {
                    let s: String = chars[start..i].iter().collect();
                    result.push_str(&s);
                }
            } else {
                // Just an integer — push as-is
                let s: String = chars[start..i].iter().collect();
                result.push_str(&s);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Replace absolute paths with <PATH>.
fn normalize_paths(text: &str) -> String {
    let mut result = text.to_string();
    // Windows paths
    while let Some(start) = result.find("C:\\") {
        if let Some(end) = result[start..].find(|c: char| c.is_whitespace()) {
            result.replace_range(start..start + end, "<PATH>");
        } else {
            result.replace_range(start.., "<PATH>");
            break;
        }
    }
    result
}

/// Strip MSVC linker noise ("Creating library ... and object ...") from output.
fn strip_linker_noise(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("Creating library"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize(text: &str) -> String {
    normalize_paths(&normalize_floats(&strip_linker_noise(text)))
}

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_example(name: &str) -> String {
    let root = workspace_root();
    let example_path = root.join(format!("examples/{}.nsl", name));
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        panic!(
            "nsl run failed for '{}' (exit {:?}):\nstderr: {}",
            name, output.status.code(), stderr
        );
    }
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn expected_output(name: &str) -> String {
    let root = workspace_root();
    let path = root.join(format!("tests/expected/{}.txt", name));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Missing baseline: {}", path.display()))
}

fn assert_output_matches(name: &str) {
    let actual = normalize(&run_example(name));
    let expected = normalize(&expected_output(name));
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Output mismatch for example '{}'",
        name
    );
}

#[test]
fn e2e_hello() {
    assert_output_matches("hello");
}

#[test]
fn e2e_features() {
    assert_output_matches("features");
}

#[test]
fn e2e_m9_tensors() {
    assert_output_matches("m9_tensors");
}

#[test]
fn e2e_m10_shape_check() {
    assert_output_matches("m10_shape_check");
}

#[test]
fn e2e_m11_model_basic() {
    assert_output_matches("m11_model_basic");
}

#[test]
fn e2e_m12_grad_basic() {
    assert_output_matches("m12_grad_basic");
}

#[test]
fn e2e_m12_grad_matmul() {
    assert_output_matches("m12_grad_matmul");
}

#[test]
fn e2e_m13_stdlib_import() {
    assert_output_matches("m13_stdlib_import");
}

#[test]
fn e2e_m14_sgd_basic() {
    assert_output_matches("m14_sgd_basic");
}

#[test]
fn e2e_m14_mse_test() {
    assert_output_matches("m14_mse_test");
}

#[test]
fn e2e_m23_byod_ternary() {
    assert_output_matches("m23_byod_ternary");
}

#[test]
fn e2e_m23_byod_block() {
    assert_output_matches("m23_byod_block");
}

#[test]
fn e2e_m23_byod_error() {
    assert_output_matches("m23_byod_error");
}

// ---------------------------------------------------------------------------
// M24 standalone export tests
// ---------------------------------------------------------------------------

/// Create a small safetensors file with a 4x3 weight matrix and a 3-element bias vector.
/// Returns the path to the created file.
fn create_small_safetensors(dir: &std::path::Path) -> std::path::PathBuf {
    use safetensors::Dtype;
    use std::collections::HashMap;

    // "weight": shape [4,3], f64, values 1.0..=12.0
    let weight_data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    // "bias": shape [3], f64, values 0.1, 0.2, 0.3
    let bias_data: Vec<f64> = vec![0.1, 0.2, 0.3];
    let bias_bytes: Vec<u8> = bias_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let weight_view =
        safetensors::tensor::TensorView::new(Dtype::F64, vec![4, 3], &weight_bytes).unwrap();
    let bias_view =
        safetensors::tensor::TensorView::new(Dtype::F64, vec![3], &bias_bytes).unwrap();

    let mut tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    tensors.insert("bias".to_string(), bias_view);
    tensors.insert("weight".to_string(), weight_view);

    let serialized = safetensors::tensor::serialize(&tensors, &None).unwrap();
    let path = dir.join("weights.safetensors");
    std::fs::write(&path, &serialized).unwrap();
    path
}

#[test]
fn e2e_m24_standalone_embedded() {
    let root = workspace_root();
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let weights_path = create_small_safetensors(tmp.path());
    let example_path = root.join("examples/m24_standalone_small.nsl");

    let exe_name = if cfg!(target_os = "windows") {
        "m24_embedded.exe"
    } else {
        "m24_embedded"
    };
    let exe_path = tmp.path().join(exe_name);

    // 1. Build standalone with embedded weights (default auto mode, small file)
    let build_output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "build", "--standalone"])
        .arg("-w")
        .arg(&weights_path)
        .arg("-o")
        .arg(&exe_path)
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl build --standalone");

    let build_stderr = String::from_utf8_lossy(&build_output.stderr);
    let build_stdout = String::from_utf8_lossy(&build_output.stdout);
    assert!(
        build_output.status.success(),
        "nsl build --standalone failed (exit {:?}):\nstdout: {}\nstderr: {}",
        build_output.status.code(),
        build_stdout,
        build_stderr
    );

    // Verify the build message mentions "weights embedded"
    assert!(
        build_stdout.contains("weights embedded"),
        "Expected 'weights embedded' in build output, got: {}",
        build_stdout
    );

    // 2. Run the standalone binary
    let run_output = Command::new(&exe_path)
        .output()
        .expect("failed to execute standalone binary");

    let run_stderr = String::from_utf8_lossy(&run_output.stderr);
    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        run_output.status.success(),
        "standalone binary failed (exit {:?}):\nstdout: {}\nstderr: {}",
        run_output.status.code(),
        run_stdout,
        run_stderr
    );

    // 3. Compare output against expected baseline
    let expected = normalize(&expected_output("m24_standalone_small"));
    let actual = normalize(&run_stdout);
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Standalone embedded output mismatch"
    );
}

#[test]
fn e2e_m24_standalone_sidecar() {
    let root = workspace_root();
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let weights_path = create_small_safetensors(tmp.path());
    let example_path = root.join("examples/m24_standalone_small.nsl");

    let exe_name = if cfg!(target_os = "windows") {
        "m24_sidecar.exe"
    } else {
        "m24_sidecar"
    };
    let exe_path = tmp.path().join(exe_name);
    let sidecar_path = tmp.path().join("m24_sidecar.nslweights");

    // 1. Build standalone with sidecar weights (--embed-weights=never)
    let build_output = Command::new(env!("CARGO"))
        .args([
            "run",
            "-q",
            "-p",
            "nsl-cli",
            "--",
            "build",
            "--standalone",
            "--embed-weights=never",
        ])
        .arg("-w")
        .arg(&weights_path)
        .arg("-o")
        .arg(&exe_path)
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl build --standalone --embed-weights=never");

    let build_stderr = String::from_utf8_lossy(&build_output.stderr);
    let build_stdout = String::from_utf8_lossy(&build_output.stdout);
    assert!(
        build_output.status.success(),
        "nsl build --standalone --embed-weights=never failed (exit {:?}):\nstdout: {}\nstderr: {}",
        build_output.status.code(),
        build_stdout,
        build_stderr
    );

    // Verify build message mentions "sidecar weights"
    assert!(
        build_stdout.contains("sidecar weights"),
        "Expected 'sidecar weights' in build output, got: {}",
        build_stdout
    );

    // Verify both binary and .nslweights sidecar file exist
    assert!(
        exe_path.exists(),
        "Standalone binary not found at {}",
        exe_path.display()
    );
    assert!(
        sidecar_path.exists(),
        "Sidecar .nslweights file not found at {}",
        sidecar_path.display()
    );

    // 2. Run the standalone binary
    let run_output = Command::new(&exe_path)
        .output()
        .expect("failed to execute standalone binary (sidecar)");

    let run_stderr = String::from_utf8_lossy(&run_output.stderr);
    let run_stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        run_output.status.success(),
        "standalone sidecar binary failed (exit {:?}):\nstdout: {}\nstderr: {}",
        run_output.status.code(),
        run_stdout,
        run_stderr
    );

    // 3. Compare output against expected baseline
    let expected = normalize(&expected_output("m24_standalone_small"));
    let actual = normalize(&run_stdout);
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Standalone sidecar output mismatch"
    );
}

// ---------------------------------------------------------------------------
// M25 Paged KV-cache and memory profiling tests
// ---------------------------------------------------------------------------

#[test]
fn e2e_m25_paged_kv() {
    assert_output_matches("m25_paged_kv");
}

#[test]
fn e2e_m25_profiling() {
    let root = workspace_root();
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let example_path = root.join("examples/m25_profiling.nsl");

    // Build the NSL program to a temp binary first.
    let exe_name = if cfg!(target_os = "windows") {
        "m25_profiling.exe"
    } else {
        "m25_profiling"
    };
    let exe_path = tmp.path().join(exe_name);

    let build_output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "build", "-o"])
        .arg(&exe_path)
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to build m25_profiling.nsl");

    let build_stderr = String::from_utf8_lossy(&build_output.stderr);
    assert!(
        build_output.status.success(),
        "nsl build failed:\nstderr: {}",
        build_stderr
    );

    // Run the compiled binary from the temp dir with NSL_PROFILE_MEMORY=1.
    // The atexit handler will write memory_profile.json in the CWD.
    let output = Command::new(&exe_path)
        .env("NSL_PROFILE_MEMORY", "1")
        .current_dir(tmp.path())
        .output()
        .expect("failed to execute profiling binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "profiling binary failed (exit {:?}):\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr,
    );

    // Verify stdout matches expected baseline.
    let expected = normalize(&expected_output("m25_profiling"));
    let actual = normalize(&stdout);
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Profiling test output mismatch"
    );

    // Verify memory_profile.json was created by the atexit handler.
    let profile_path = tmp.path().join("memory_profile.json");
    assert!(
        profile_path.exists(),
        "memory_profile.json not found — atexit handler may not have fired. stderr: {}",
        stderr,
    );

    // Parse and validate the JSON content.
    let json_str = std::fs::read_to_string(&profile_path)
        .expect("failed to read memory_profile.json");
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("memory_profile.json is not valid JSON");

    // Check traceEvents array exists and has events.
    let events = parsed["traceEvents"]
        .as_array()
        .expect("traceEvents should be an array");
    assert!(
        !events.is_empty(),
        "traceEvents should contain alloc/free events"
    );

    // Verify metadata has expected fields.
    let meta = &parsed["metadata"];
    assert!(
        meta["peak_blocks"].as_u64().unwrap_or(0) > 0,
        "peak_blocks should be > 0, got: {}",
        meta["peak_blocks"]
    );
    assert!(
        meta["total_allocs"].as_u64().unwrap_or(0) > 0,
        "total_allocs should be > 0, got: {}",
        meta["total_allocs"]
    );
    assert!(
        meta["total_frees"].as_u64().unwrap_or(0) > 0,
        "total_frees should be > 0, got: {}",
        meta["total_frees"]
    );
}

// ---------------------------------------------------------------------------
// M26 @autotune, @fuse, and kernel profiler tests
// ---------------------------------------------------------------------------

#[test]
fn e2e_m26_kernel_profiler() {
    assert_output_matches("m26_kernel_profiler");
}

#[test]
fn e2e_m26_fuse() {
    assert_output_matches("m26_fuse");
}

#[test]
fn e2e_m26_autotune() {
    assert_output_matches("m26_autotune");
}

// ---------------------------------------------------------------------------
// M27: FlashAttention-2
// ---------------------------------------------------------------------------

#[test]
fn e2e_m27_flash_attention() {
    assert_output_matches("m27_flash_attention");
}

#[test]
fn e2e_m27_paged_attention() {
    assert_output_matches("m27_paged_attention");
}

#[test]
fn e2e_m27_rope_gqa() {
    assert_output_matches("m27_rope_gqa");
}
