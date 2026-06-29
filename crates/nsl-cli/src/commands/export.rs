//! `nsl export` — ONNX / safetensors export driver.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

pub(crate) fn run_export(file: &PathBuf, output: Option<&std::path::Path>, format: Option<&str>) {
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Determine export mode from format flag, output extension, or input extension
    let mode = if let Some(fmt) = format {
        fmt.to_lowercase()
    } else if let Some(out) = output {
        out.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    } else {
        match ext {
            "nsl" => "onnx".to_string(),
            "nslm" => "safetensors".to_string(),
            _ => {
                eprintln!(
                    "error: cannot determine export format from '{}'.\n\
                     Use --format onnx or --format safetensors, or provide an output path with --output.",
                    file.display()
                );
                process::exit(1);
            }
        }
    };

    match mode.as_str() {
        "onnx" => {
            if ext != "nsl" {
                eprintln!(
                    "error: ONNX export requires an .nsl input file, got '{}'",
                    file.display()
                );
                process::exit(1);
            }
            // The NSL file should contain an export_model() function (or similar)
            // that calls to_onnx() internally. We just compile and run it.
            println!("Exporting ONNX from {}", file.display());
            crate::commands::build::run_run(file, &[], false, false, false, false, false, &nsl_codegen::CompileOptions::default());
        }
        "safetensors" => {
            if ext != "nslm" {
                eprintln!(
                    "error: safetensors conversion requires an .nslm input file, got '{}'",
                    file.display()
                );
                process::exit(1);
            }
            // Compute default output path: same stem with .safetensors extension
            let default_out;
            let out_path: &std::path::Path = if let Some(o) = output {
                o
            } else {
                default_out = file.with_extension("safetensors");
                &default_out
            };
            match crate::commands::convert::convert_nslm_to_safetensors(file, out_path) {
                Ok(()) => println!("Converted {} → {}", file.display(), out_path.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!(
                "error: unknown export format '{}'. Supported formats: onnx, safetensors",
                other
            );
            process::exit(1);
        }
    }
}
