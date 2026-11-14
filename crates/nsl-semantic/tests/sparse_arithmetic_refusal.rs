//! `Sparse + Sparse` and `Sparse * Sparse` are compile errors.
//!
//! Their result is a sparse value. The checker typed it as a tensor and
//! code generation registered the sparse handle the runtime returns as an
//! owned dense tensor, so the first dense operation on it, or the cleanup
//! at the end of the statement, would have received a sparse handle. No
//! program can construct a sparse value today, so the fault could not be
//! reached, fixed and tested by running; it is refused, and this test keeps
//! it refused when a constructor arrives.

fn errors(src: &str) -> Vec<String> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed.diagnostics.iter().all(|d| d.level != nsl_errors::Level::Error),
        "parse errors in {src:?}: {:?}",
        parsed.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    nsl_semantic::analyze(&parsed.module, &mut interner)
        .diagnostics
        .iter()
        .filter(|d| d.level == nsl_errors::Level::Error)
        .map(|d| format!("{} | {}", d.message, d.labels.iter().map(|l| l.message.as_str()).collect::<Vec<_>>().join(" / ")))
        .collect()
}

fn program(a: &str, b: &str, op: &str) -> String {
    format!("fn f(a: {a}, b: {b}) -> Tensor:\n    return a {op} b\n")
}

const SPARSE: &str = "Sparse<[3, 3], f32, csr>";

#[test]
fn sparse_plus_sparse_and_sparse_times_sparse_are_refused() {
    for op in ["+", "*"] {
        let e = errors(&program(SPARSE, SPARSE, op));
        assert_eq!(e.len(), 1, "`{op}`: {e:?}");
        assert!(e[0].contains(&format!("`{op}` on two Sparse tensors is not supported")), "{e:?}");
        assert!(e[0].contains("no sparse result type"), "{e:?}");
    }
    // Formats may differ; it is still two sparse operands.
    let e = errors(&program(SPARSE, "Sparse<[3, 3], f32, coo>", "+"));
    assert!(e.iter().any(|m| m.contains("on two Sparse tensors is not supported")), "{e:?}");
}

/// What the refusal must not catch: an operation with a dense operand, whose
/// result code generation does produce as a dense tensor.
#[test]
fn sparse_with_dense_arithmetic_still_checks() {
    for (a, b) in [(SPARSE, "Tensor"), ("Tensor", SPARSE)] {
        for op in ["+", "*", "-"] {
            let e = errors(&program(a, b, op));
            assert!(e.is_empty(), "{a} {op} {b}: {e:?}");
        }
    }
    let e = errors(&program("Tensor", "Tensor", "+"));
    assert!(e.is_empty(), "{e:?}");
}
