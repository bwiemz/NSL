//! `nsl test` — compile-and-run NSL test files.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

pub(crate) fn run_test(file: &PathBuf, filter: Option<&str>) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend(file);

    // Compile in test mode — produces a binary with test-dispatch main()
    let (obj_bytes, test_fns) = match nsl_codegen::compile_test(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        false,
        &nsl_codegen::CompileOptions::default(),
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // Apply filter
    let tests: Vec<&String> = if let Some(f) = filter {
        test_fns.iter().filter(|name| name.contains(f)).collect()
    } else {
        test_fns.iter().collect()
    };

    if tests.is_empty() {
        if let Some(f) = filter {
            eprintln!("no tests match filter '{f}'");
        } else {
            eprintln!("no @test functions found");
        }
        process::exit(1);
    }

    // Write object file and link to a temp executable
    let temp_dir = std::env::temp_dir().join(format!("nsl_test_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("test");
    let obj_path = temp_dir.join(format!("{stem}.o"));
    let exe_name = if cfg!(target_os = "windows") {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    };
    let exe_path = temp_dir.join(&exe_name);

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    // Run each test by spawning the binary with --run <test_name>
    let mut passed = 0u32;
    let mut failed = 0u32;

    for test_name in &tests {
        let output = std::process::Command::new(&exe_path)
            .args(["--run", test_name])
            .output()
            .unwrap_or_else(|e| {
                eprintln!("error: could not execute '{}': {e}", exe_path.display());
                process::exit(1);
            });

        if output.status.success() {
            println!("{test_name} ... PASS");
            passed += 1;
        } else {
            println!("{test_name} ... FAIL");
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                for line in stderr.lines() {
                    println!("  {line}");
                }
            }
            failed += 1;
        }
    }

    // Clean up
    let _ = std::fs::remove_file(&exe_path);
    let _ = std::fs::remove_dir(&temp_dir);

    // Summary
    println!("\n{passed} passed, {failed} failed");

    if failed > 0 {
        process::exit(1);
    }
}

