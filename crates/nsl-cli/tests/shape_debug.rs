use nsl_cli::shape_debug::{format_trace, ShapeDebugInput};

const SRC: &str = r#"
fn layernorm(a: Tensor<[batch=8, seq=2048, d=512], bf16>) -> Tensor<[batch=8, seq=2048, d=512], bf16>:
    return a

fn matmul(a: Tensor<[batch=8, seq=2048, d=512], bf16>,
          b: Tensor<[d=512, out=512], bf16>) -> Tensor<[batch=8, seq=2048, out=512], bf16>:
    return a @ b

fn forward(x: Tensor<[batch=8, seq=2048, d=512], bf16>,
           W: Tensor<[d=512, out=512], bf16>) -> Tensor<[batch=8, seq=2048, out=512], bf16>:
    let h = layernorm(x)
    let y = matmul(h, W)
    return y
"#;

#[test]
fn trace_emits_one_line_per_typed_let() {
    let input = ShapeDebugInput::from_source(SRC, "model.nsl").unwrap();
    let out = format_trace(&input);
    eprintln!("{out}");
    assert!(
        out.contains("layernorm(x)"),
        "output missing 'layernorm(x)':\n{out}"
    );
    assert!(
        out.contains("[8, 2048, 512]"),
        "output missing '[8, 2048, 512]':\n{out}"
    );
    assert!(
        out.contains("matmul(h, W)"),
        "output missing 'matmul(h, W)':\n{out}"
    );
    assert!(
        out.lines().filter(|l| l.contains("\u{2705}")).count() >= 2,
        "expected >=2 check marks:\n{out}"
    );
}

#[test]
fn trace_final_line_reports_total_flops() {
    let input = ShapeDebugInput::from_source(SRC, "model.nsl").unwrap();
    let out = format_trace(&input);
    let last = out
        .lines()
        .rev()
        .find(|l| l.contains("Total FLOPs"))
        .unwrap_or_else(|| panic!("no 'Total FLOPs' line in:\n{out}"));
    assert!(last.contains("FLOP"));
}

#[test]
fn shape_mismatch_renders_error_block() {
    let bad = r#"
fn forward(x: Tensor<[8, 2048, 512], bf16>) -> Tensor<[8, 2048, 384], bf16>:
    let y: Tensor<[8, 2048, 384], bf16> = x
    return y
"#;
    let input = ShapeDebugInput::from_source(bad, "bad.nsl").unwrap();
    let out = format_trace(&input);
    eprintln!("{out}");
    assert!(out.contains("\u{274C}"), "expected red X in:\n{out}");
    assert!(
        out.contains("Expected"),
        "expected 'Expected' block in:\n{out}"
    );
    assert!(out.contains("Cause"), "expected 'Cause' block in:\n{out}");
}
