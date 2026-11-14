//! `nsl.quant.ops` type-checks and computes PyTorch's affine quantization.
//!
//! The module's signatures spell `QuantizedTensor`, the type
//! `nsl_qtensor_quantize` returns, but the checker had no name for it, so
//! `import nsl.quant.ops` failed with "undefined type `QuantizedTensor`" and
//! nothing in the suite type-checked the standard library to notice.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn stdlib_modules(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            stdlib_modules(&path, out);
        } else if path.extension().is_some_and(|e| e == "nsl") {
            out.push(path);
        }
    }
}

/// Every module of the standard library passes `nsl check`.
#[test]
fn every_stdlib_module_type_checks() {
    let stdlib = workspace_root().join("stdlib");
    let mut modules = Vec::new();
    stdlib_modules(&stdlib, &mut modules);
    modules.sort();
    assert!(modules.len() > 20, "found only {} stdlib modules", modules.len());
    let mut failures = Vec::new();
    for module in &modules {
        let out = Command::cargo_bin("nsl")
            .unwrap()
            .env("NSL_STDLIB_PATH", &stdlib)
            .arg("check")
            .arg(module)
            .output()
            .unwrap();
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let first = stderr.lines().find(|l| l.contains("error")).unwrap_or("").to_string();
            failures.push(format!("{}: {first}", module.display()));
        }
    }
    assert!(failures.is_empty(), "stdlib modules that do not type-check:\n{}", failures.join("\n"));
}

/// `dequantize(quantize(x, ...))` against PyTorch 2.15's fake quantization
/// with min/max observer parameters, the same scheme (asymmetric, range
/// widened to contain zero, codes 0..255 for int8 and 0..15 for int4):
/// `MinMaxObserver` / `PerChannelMinMaxObserver(dtype=quint8, quant_min=0,
/// quant_max=255|15)`, then `fake_quantize_per_{tensor,channel}_affine`.
/// The inputs avoid rounding ties, where the two round differently.
#[test]
fn quantize_then_dequantize_matches_pytorch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(
        &path,
        r#"from nsl.quant.ops import quantize, dequantize

let t = arange(-4.0, 4.0).reshape([2, 4])
print(dequantize(quantize(t, "int8", "per_tensor", 0, 0)))
let t2 = zeros([2, 4])
t2[0, 0] = 0.5
t2[0, 1] = -1.25
t2[0, 2] = 2.0
t2[0, 3] = 0.75
t2[1, 0] = 10.0
t2[1, 1] = -3.0
t2[1, 2] = 7.5
t2[1, 3] = 1.0
print(dequantize(quantize(t2, "int8", "per_channel", 0, 0)))
print(dequantize(quantize(t2, "int4", "per_tensor", 0, 0)))
print(dequantize(quantize(t2, "int4", "per_channel", 1, 0)))
"#,
    )
    .unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    let got: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect();
    #[rustfmt::skip]
    let want: [f64; 32] = [
        // int8, per tensor (scale 0.0274509806, zero point 146)
        -4.007843017578125, -2.992156982421875, -2.0039215087890625, -0.9882352948188782,
        0.0, 0.9882352948188782, 2.0039215087890625, 2.992156982421875,
        // int8, per channel along axis 0 (scales 0.0127451, 0.0509804; zero points 98, 59)
        0.49705883860588074, -1.2490196228027344, 2.0009803771972656, 0.751960813999176,
        9.992156982421875, -3.007843255996704, 7.494117736816406, 1.0196079015731812,
        // int4, per tensor (scale 0.8666667, zero point 3)
        0.8666666746139526, -0.8666666746139526, 1.7333333492279053, 0.8666666746139526,
        10.399999618530273, -2.5999999046325684, 7.800000190734863, 0.8666666746139526,
        // int4, per channel along axis 1 (zero points 0, 15, 0, 0)
        0.6666666865348816, -1.2000000476837158, 2.0, 0.7333333492279053,
        10.0, -3.0, 7.5, 1.0,
    ];
    assert_eq!(got.len(), want.len(), "printed {got:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        // Far below the smallest step (0.0127): a one-code difference fails.
        assert!((g - w).abs() < 1e-5, "element {i}: {g}, PyTorch {w} (all: {got:?})");
    }
}
