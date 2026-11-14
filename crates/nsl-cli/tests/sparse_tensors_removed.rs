//! General sparse tensors (M50) were moved to the attic in 2026-11; the
//! code is preserved at tag `attic/sparse-tensors-2026-11`.
//!
//! `Sparse<[shape], dtype, format>` was a type no value could have: NSL had
//! no way to construct a sparse tensor, so the type, the code generation
//! behind it and the runtime sparse library could not be reached by any
//! program. A program that still writes the type must fail loudly, naming
//! the removal, rather than compile into something else.
//!
//! This replaces `nsl-semantic`'s `sparse_arithmetic_refusal` test: the
//! `Sparse + Sparse` program that test refused is the first case below, and
//! it is refused earlier now, at its first `Sparse<...>` annotation.

use std::path::PathBuf;
use std::process::{Command, Output};

const ATTIC_TAG: &str = "attic/sparse-tensors-2026-11";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nsl-cli sits two levels below the workspace root")
        .to_path_buf()
}

/// Write `source` to a scratch file and run `nsl <args...> <file>` on it.
fn nsl(args: &[&str], source: &str) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).expect("write program");
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(&path)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl")
}

#[test]
fn a_sparse_type_annotation_is_refused_with_the_attic_tag() {
    let programs = [
        // The program that used to check and compile with a sparse handle
        // registered as a dense tensor.
        "fn f(a: Sparse<[3, 3], f32, csr>, b: Sparse<[3, 3], f32, csr>) -> Tensor:\n    return a + b\n",
        "fn f(a: Sparse<[3, 3], f32, coo>, b: Tensor) -> Tensor:\n    return a @ b\n",
        "let d = ones([3, 3])\nlet s: Sparse<[3, 3], f32, bsr> = d\n",
        "model M:\n    w: Sparse<[4, 4], f32, csc> = zeros([4, 4])\n",
        // Bare `Sparse`, which names no user type: refused by the resolver.
        "fn f(a: Sparse, b: Tensor) -> Tensor:\n    return b\n",
    ];
    for src in programs {
        for args in [&["check"][..], &["build"][..]] {
            let out = nsl(args, src);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(!out.status.success(), "nsl {args:?} accepted a Sparse type:\n{src}\n{stderr}");
            assert!(
                stderr.contains("`Sparse<...>` types were removed") && stderr.contains(ATTIC_TAG),
                "nsl {args:?} must name the removal and the attic tag:\n{src}\n{stderr}"
            );
            assert!(stderr.contains("general sparse tensors (M50)"), "{stderr}");
        }
    }
}

/// One annotation, one error: the annotation becomes a wildcard, so the
/// rest of the declaration is not reported a second time.
#[test]
fn one_annotation_gives_one_error() {
    let out = nsl(&["check"], "fn f(a: Sparse<[3, 3], f32, csr>, b: Tensor) -> Tensor:\n    return b\n");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert_eq!(stderr.matches("types were removed").count(), 1, "{stderr}");
    assert!(stderr.contains("1 error(s) found"), "{stderr}");
}

#[test]
fn the_layout_decorator_is_refused_with_the_attic_tag() {
    let out = nsl(
        &["check", "--allow-unknown-decorators"],
        "@layout\nfn f(x: Tensor) -> Tensor:\n    return x\n",
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "@layout must stay an error under --allow-unknown-decorators:\n{stderr}");
    assert!(stderr.contains("@layout was removed") && stderr.contains(ATTIC_TAG), "{stderr}");
}

/// What must keep working: the same programs with dense tensors, and a
/// user type that merely has "Sparse" in its name.
#[test]
fn dense_programs_and_similar_names_still_check() {
    for src in [
        "fn f(a: Tensor<[3, 3], f32>, b: Tensor) -> Tensor:\n    return a + b\n",
        "struct SparseStats:\n    nnz: int\n\nfn f(s: SparseStats) -> int:\n    return s.nnz\n",
    ] {
        let out = nsl(&["check"], src);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{src}\n{stderr}");
    }
}
