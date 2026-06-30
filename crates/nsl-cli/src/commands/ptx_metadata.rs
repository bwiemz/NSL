//! `nsl ptx-metadata` — static per-kernel resource report for a `.ptx` file.
//!
//! Parses declared register counts, static shared-memory bytes, and the target
//! SM out of synthesized PTX text and prints a per-kernel report. Pure text
//! analysis — no GPU or CUDA toolkit required. The parser lives in
//! [`nsl_codegen::ptx_metadata`] so it is also reusable from codegen.

use std::path::Path;
use std::process;

pub(crate) fn run(file: &Path) {
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read {}: {e}", file.display());
            process::exit(1);
        }
    };

    let kernels = nsl_codegen::ptx_metadata::extract_ptx_metadata(&bytes);
    if kernels.is_empty() {
        // Exit non-zero, so the message is framed as an error (not a warning)
        // to stay consistent with the exit code for scripting callers.
        eprintln!(
            "error: no `.entry` kernels found in {} — is this a PTX module?",
            file.display(),
        );
        process::exit(1);
    }

    print!(
        "{}",
        nsl_codegen::ptx_metadata::format_ptx_metadata_report(&kernels),
    );
}
