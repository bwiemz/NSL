//! Issue #693: `nsl build --shared-lib` refuses an `@export` whose symbol
//! name collides with a symbol the artifact's statically-linked runtime calls
//! by name.
//!
//! The artifact defines each export as a global symbol in the same image as
//! its own copy of the runtime. On Mach-O and PE the static linker binds the
//! runtime's `memcpy` (or `pow`, `log`, …) references to that definition,
//! so every runtime copy ran the export wrapper — `dlpack_unsupported_dtype_refusal`
//! aborted on macOS and Windows for exactly this reason while Linux passed on
//! ELF load order. The refusal is uniform across platforms so the same source
//! means the same thing everywhere.
//!
//! The shadowed set is read from the runtime archive the link would use, not
//! a hand list, so this gate runs on every CI lane: it is what proves the
//! archive reader understands that lane's object format (ELF / Mach-O's `_`
//! prefix / COFF). If the reader missed a format, `memcpy` would not be
//! found and the build would proceed to link.

use assert_cmd::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn stdlib_path() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("stdlib")
}

/// `alpha` is an honest name. `memcpy` and `pow` are C-library functions the
/// runtime calls on every target: LLVM lowers copies to `memcpy`, and
/// `f64::powf` calls libm's (or the UCRT's) `pow`. Not a POSIX-only name like
/// `close` — Rust's std on Windows closes handles with `CloseHandle`, so the
/// Windows lane would not report it.
const SRC: &str = concat!(
    "\n@export\nfn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 2.0\n",
    "\n@export\nfn memcpy(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 3.0\n",
    "\n@export\nfn pow(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 4.0\n",
);

#[test]
fn an_export_named_after_a_runtime_import_is_refused_before_linking() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("m.nsl");
    fs::write(&src, SRC).unwrap();
    let lib = tmp.path().join("libm.shared");

    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", stdlib_path())
        .args(["build", "--shared-lib"])
        .arg(&src)
        .arg("-o")
        .arg(&lib)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !out.status.success(),
        "a `memcpy` export must be refused, but the build succeeded:\n{stderr}"
    );
    assert!(
        stderr.contains("@export names `memcpy`, `pow` collide with symbols"),
        "the refusal must name every offender, in source order:\n{stderr}"
    );
    assert!(
        !stderr.contains("`alpha`"),
        "an honest export name must not be reported:\n{stderr}"
    );
    assert!(stderr.contains("#693"), "the refusal must cite the issue:\n{stderr}");
    assert!(!lib.exists(), "a refused build must not leave a library behind");
    let leftovers: Vec<_> = fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".o") || n.ends_with(".obj"))
        .collect();
    assert!(leftovers.is_empty(), "refusal left objects behind: {leftovers:?}");
}
