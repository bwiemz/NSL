//! `nsl fmt` — source formatter driver.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use crate::formatter;
use std::process;

pub(crate) fn run_fmt(files: &[String], check: bool) {
    use std::path::Path;

    let mut total = 0u32;
    let mut changed = 0u32;
    let mut errors = 0u32;

    for pattern in files {
        // Treat as literal file path (glob support can come later)
        let path = Path::new(pattern);
        if !path.exists() {
            eprintln!("error: file not found: {}", pattern);
            errors += 1;
            continue;
        }
        total += 1;
        match formatter::format_file(path, check) {
            Ok(true) => {
                changed += 1;
                if check {
                    println!("Would reformat: {}", path.display());
                } else {
                    println!("Formatted: {}", path.display());
                }
            }
            Ok(false) => {} // already formatted
            Err(e) => {
                eprintln!("{}", e);
                errors += 1;
            }
        }
    }

    if total > 0 || errors > 0 {
        println!("{} file(s) checked, {} changed, {} error(s)", total, changed, errors);
    }

    if check && changed > 0 {
        process::exit(1);
    }
    if errors > 0 {
        process::exit(1);
    }
}
