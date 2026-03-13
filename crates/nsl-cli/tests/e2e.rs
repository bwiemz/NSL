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

fn normalize(text: &str) -> String {
    normalize_paths(&normalize_floats(text))
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
