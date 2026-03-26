//! End-to-end smoke tests for NSL examples.
//! Compiles and runs .nsl files, compares stdout against expected baselines.

use std::process::Command;

/// Truncate floating-point numbers in text to 6 decimal places.
/// Using 6 instead of 4 to catch more subtle numerical regressions
/// while still tolerating platform-level float formatting differences.
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
                if decimal_digits >= 7 {
                    // Truncate to 6 decimal places
                    let num_str: String = chars[start..i].iter().collect();
                    if let Ok(val) = num_str.parse::<f64>() {
                        result.push_str(&format!("{:.6}", val));
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

// ---------------------------------------------------------------------------
// M28: Dynamic shapes & ragged tensors
// ---------------------------------------------------------------------------

#[test]
fn e2e_m28_dynamic_shapes() {
    assert_output_matches("m28_dynamic_shapes");
}

#[test]
fn e2e_m28_bounded_dims() {
    assert_output_matches("m28_bounded_dims");
}

#[test]
fn e2e_m28_dim_unification() {
    assert_output_matches("m28_dim_unification");
}

// ---------------------------------------------------------------------------
// M29: Continuous batching & serving engine
// ---------------------------------------------------------------------------

#[test]
fn e2e_m29_serve_basic() {
    assert_output_matches("m29_serve_basic");
}

#[test]
fn e2e_m29_continuous_batch() {
    assert_output_matches("m29_continuous_batch");
}

#[test]
fn e2e_m29_preemption() {
    assert_output_matches("m29_preemption");
}

// ---------------------------------------------------------------------------
// M30: Tensor parallelism
// ---------------------------------------------------------------------------

#[test]
fn e2e_m30_tp_basic() {
    assert_output_matches("m30_tp_basic");
}

#[test]
fn e2e_m30_shard_validation() {
    assert_output_matches("m30_shard_validation");
}

#[test]
fn e2e_m30_gqa_replication() {
    assert_output_matches("m30_gqa_replication");
}

// ---------------------------------------------------------------------------
// M31: Graph-level operator fusion
// ---------------------------------------------------------------------------

#[test]
fn e2e_m31_epilogue_fusion() {
    assert_output_matches("m31_epilogue_fusion");
}

#[test]
fn e2e_m31_reduction_fusion() {
    assert_output_matches("m31_reduction_fusion");
}

#[test]
fn e2e_m31_fuse_graph() {
    assert_output_matches("m31_fuse_graph");
}

// ---------------------------------------------------------------------------
// M32: Mixture of Experts
// ---------------------------------------------------------------------------

#[test]
fn e2e_m32_moe_basic() {
    assert_output_matches("m32_moe_basic");
}

// M32: MoE validation error — expected compile failure
#[test]
fn e2e_m32_moe_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m32_moe_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m32_moe_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("top_k") || stderr.contains("moe"),
        "Expected MoE validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// M33: Speculative Decoding
// ---------------------------------------------------------------------------

#[test]
fn e2e_m33_speculative_basic() {
    assert_output_matches("m33_speculative_basic");
}

// M33: @speculative validation error — expected compile failure
#[test]
fn e2e_m33_speculative_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m33_vocab_mismatch.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m33_vocab_mismatch, but it succeeded"
    );
    assert!(
        stderr.contains("num_tokens") || stderr.contains("speculative"),
        "Expected speculative validation error in stderr, got: {}",
        stderr
    );
}

#[test]
fn e2e_m33_speculative_decode() {
    assert_output_matches("m33_speculative_decode");
}

// ---------------------------------------------------------------------------
// M34: Context Parallelism (Ring Attention)
// ---------------------------------------------------------------------------

#[test]
fn e2e_m34_cp_basic() {
    assert_output_matches("m34_cp_basic");
}

// M34: @context_parallel validation error — expected compile failure
#[test]
fn e2e_m34_cp_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m34_cp_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m34_cp_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("ring_size") || stderr.contains("context_parallel"),
        "Expected context_parallel validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// M35: FP8 Compute & Sub-Byte Quantization
// ---------------------------------------------------------------------------

#[test]
fn e2e_m35_fp8_basic() {
    assert_output_matches("m35_fp8_basic");
}

// M35: @fp8_compute validation error — expected compile failure
#[test]
fn e2e_m35_fp8_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m35_fp8_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m35_fp8_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("fp8_compute") || stderr.contains("unknown argument"),
        "Expected fp8_compute validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// M36: Compile-Time Memory Planning
// ---------------------------------------------------------------------------

#[test]
fn e2e_m36_slab_basic() {
    assert_output_matches("m36_slab_basic");
}

// ---------------------------------------------------------------------------
// M37: Compile-Time Roofline & Cost Model
// ---------------------------------------------------------------------------

#[test]
fn e2e_m37_perf_budget_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m37_perf_budget_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m37_perf_budget_error, but it succeeded"
    );
    assert!(
        stderr.contains("perf_budget") || stderr.contains("unknown argument"),
        "Expected perf_budget validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// M38a: Linear Types Semantics
// ---------------------------------------------------------------------------

#[test]
fn e2e_m38_shared_basic() {
    assert_output_matches("m38_shared_basic");
}

#[test]
fn e2e_m38_shared_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m38_shared_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m38_shared_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("shared") || stderr.contains("let-binding"),
        "Expected @shared validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// M39: Automatic Batching (vmap)
// ---------------------------------------------------------------------------

#[test]
fn e2e_m39_vmap_validation_error() {
    let root = workspace_root();
    let example_path = root.join("examples/m39_vmap_validation_error.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for m39_vmap_validation_error, but it succeeded"
    );
    assert!(
        stderr.contains("vmap") || stderr.contains("unknown argument"),
        "Expected vmap validation error in stderr, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// NSLM ↔ Safetensors conversion tests
// ---------------------------------------------------------------------------

/// Build a minimal NSLM file in memory:
///   magic(4) + version_u32_le(4) + header_size_u64_le(8) + JSON + padding + raw f32 data
fn build_nslm_f32(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    // Build tensor data blocks and metadata
    let mut data_offset: u64 = 0;
    let mut param_jsons: Vec<String> = Vec::new();
    let mut raw_data: Vec<u8> = Vec::new();

    for (name, shape, values) in tensors {
        let nbytes = (values.len() * 4) as u64;
        let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        param_jsons.push(format!(
            r#"{{"name":"{}","shape":[{}],"dtype":"f32","offset":{},"nbytes":{}}}"#,
            name,
            shape_json.join(","),
            data_offset,
            nbytes
        ));
        for &v in values {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        data_offset += nbytes;
    }

    let header = format!(r#"{{"params":[{}]}}"#, param_jsons.join(","));
    let header_bytes = header.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"NSLM");
    out.extend_from_slice(&1u32.to_le_bytes());       // version
    out.extend_from_slice(&header_size.to_le_bytes()); // header_size u64
    out.extend_from_slice(header_bytes);

    // Pad to 64-byte alignment from byte 0
    let total_header = 4 + 4 + 8 + header_bytes.len();
    let padding = (64 - (total_header % 64)) % 64;
    out.extend(std::iter::repeat(0u8).take(padding));

    out.extend_from_slice(&raw_data);
    out
}

#[test]
fn e2e_convert_nslm_to_safetensors() {
    use safetensors::SafeTensors;

    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let nslm_path = tmp.path().join("model.nslm");
    let st_path = tmp.path().join("model.safetensors");

    // Build a NSLM file with two tensors
    let nslm_bytes = build_nslm_f32(&[
        ("weight", vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("bias", vec![3], vec![0.1f32, 0.2, 0.3]),
    ]);
    std::fs::write(&nslm_path, &nslm_bytes).unwrap();

    // Run nsl convert
    let root = workspace_root();
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "convert"])
        .arg(&nslm_path)
        .arg(&st_path)
        .current_dir(&root)
        .output()
        .expect("failed to run nsl convert");

    assert!(
        output.status.success(),
        "nsl convert (nslm→safetensors) failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(st_path.exists(), "output .safetensors not created");

    // Verify the safetensors file contains the correct tensors
    let st_bytes = std::fs::read(&st_path).unwrap();
    let st = SafeTensors::deserialize(&st_bytes).expect("failed to parse safetensors");
    let names: Vec<String> = {
        let mut v: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
        v.sort();
        v
    };
    assert_eq!(names, vec!["bias".to_string(), "weight".to_string()], "unexpected tensor names: {:?}", names);

    // Check weight values (f32)
    let weight_view = st.tensor("weight").unwrap();
    assert_eq!(weight_view.shape(), &[2, 3]);
    let weight_f32: Vec<f32> = weight_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(weight_f32.len(), 6);
    assert!((weight_f32[0] - 1.0f32).abs() < 1e-5, "weight[0] = {}", weight_f32[0]);
    assert!((weight_f32[5] - 6.0f32).abs() < 1e-5, "weight[5] = {}", weight_f32[5]);

    // Check bias values
    let bias_view = st.tensor("bias").unwrap();
    assert_eq!(bias_view.shape(), &[3]);
    let bias_f32: Vec<f32> = bias_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!((bias_f32[0] - 0.1f32).abs() < 1e-4, "bias[0] = {}", bias_f32[0]);
    assert!((bias_f32[2] - 0.3f32).abs() < 1e-4, "bias[2] = {}", bias_f32[2]);
}

#[test]
fn e2e_convert_safetensors_to_nslm() {
    use std::collections::HashMap;

    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let st_path = tmp.path().join("weights.safetensors");
    let nslm_path = tmp.path().join("weights.nslm");

    // Build a safetensors file with one tensor
    let data: Vec<f32> = vec![10.0f32, 20.0, 30.0, 40.0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &bytes).unwrap();
    let mut map: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    map.insert("mat".to_string(), view);
    let serialized = safetensors::tensor::serialize(&map, &None).unwrap();
    std::fs::write(&st_path, &serialized).unwrap();

    // Run nsl convert
    let root = workspace_root();
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "convert"])
        .arg(&st_path)
        .arg(&nslm_path)
        .current_dir(&root)
        .output()
        .expect("failed to run nsl convert");

    assert!(
        output.status.success(),
        "nsl convert (safetensors→nslm) failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(nslm_path.exists(), "output .nslm not created");

    // Parse NSLM manually and verify contents
    let nslm_bytes = std::fs::read(&nslm_path).unwrap();
    assert!(nslm_bytes.len() >= 16, "NSLM file too small");
    assert_eq!(&nslm_bytes[0..4], b"NSLM", "bad magic");
    let version = u32::from_le_bytes(nslm_bytes[4..8].try_into().unwrap());
    assert_eq!(version, 1);
    let header_size = u64::from_le_bytes(nslm_bytes[8..16].try_into().unwrap()) as usize;
    let header_json: serde_json::Value =
        serde_json::from_slice(&nslm_bytes[16..16 + header_size]).unwrap();
    let params = header_json["params"].as_array().unwrap();
    assert_eq!(params.len(), 1, "expected 1 param");
    assert_eq!(params[0]["name"].as_str().unwrap(), "mat");
    assert_eq!(params[0]["dtype"].as_str().unwrap(), "f32");

    // Verify data
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;
    let raw = &nslm_bytes[data_start..];
    assert!(raw.len() >= 16, "not enough raw data bytes");
    let v0 = f32::from_le_bytes(raw[0..4].try_into().unwrap());
    let v3 = f32::from_le_bytes(raw[12..16].try_into().unwrap());
    assert!((v0 - 10.0f32).abs() < 1e-5, "data[0] = {}", v0);
    assert!((v3 - 40.0f32).abs() < 1e-5, "data[3] = {}", v3);
}

#[test]
fn e2e_convert_nslm_safetensors_round_trip() {
    use safetensors::SafeTensors;

    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let nslm_orig = tmp.path().join("orig.nslm");
    let st_mid = tmp.path().join("mid.safetensors");
    let nslm_final = tmp.path().join("final.nslm");

    // Build a NSLM file
    let orig_values: Vec<f32> = (1..=9).map(|i| i as f32 * 0.5f32).collect();
    let nslm_bytes = build_nslm_f32(&[("layer", vec![3, 3], orig_values.clone())]);
    std::fs::write(&nslm_orig, &nslm_bytes).unwrap();

    let root = workspace_root();

    // Step 1: nslm → safetensors
    let out1 = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "convert"])
        .arg(&nslm_orig)
        .arg(&st_mid)
        .current_dir(&root)
        .output()
        .expect("step 1 failed");
    assert!(
        out1.status.success(),
        "round-trip step 1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    // Step 2: safetensors → nslm
    let out2 = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "convert"])
        .arg(&st_mid)
        .arg(&nslm_final)
        .current_dir(&root)
        .output()
        .expect("step 2 failed");
    assert!(
        out2.status.success(),
        "round-trip step 2 failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );

    // Verify final NSLM tensor values match originals (within f32 precision)
    let final_bytes = std::fs::read(&nslm_final).unwrap();
    let header_size = u64::from_le_bytes(final_bytes[8..16].try_into().unwrap()) as usize;
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;
    let raw = &final_bytes[data_start..];

    for (i, &expected) in orig_values.iter().enumerate() {
        let got = f32::from_le_bytes(raw[i * 4..(i + 1) * 4].try_into().unwrap());
        assert!(
            (got - expected).abs() < 1e-5,
            "round-trip mismatch at [{}]: expected {}, got {}",
            i, expected, got
        );
    }

    // Also verify via safetensors intermediate
    let st_bytes = std::fs::read(&st_mid).unwrap();
    let st = SafeTensors::deserialize(&st_bytes).unwrap();
    let layer = st.tensor("layer").unwrap();
    assert_eq!(layer.shape(), &[3, 3]);
}
