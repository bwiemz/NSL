//! The `kernel` block sources the KIR front door is proved on (roadmap A2
//! step 3): `kernel_block_ptxas.rs` assembles them, `snapshot_tests.rs`
//! pins their PTX.

#![allow(dead_code)]

use nsl_ast::block::KernelDef;
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;

/// Every shape the lowering accepts, named for the snapshot.
pub const ALL: &[(&str, &str)] = &[
    (
        "kernel_block_vec_add",
        // tests/m17_kernel_test.nsl
        "kernel vec_add(a, b, c):\n    let i = thread_id()\n    c[i] = a[i] + b[i]\n",
    ),
    (
        "kernel_block_scale_add",
        // tests/m17_gpu_training_test.nsl
        "kernel scale_add(x, scale, bias, out):\n    let i = thread_id()\n    out[i] = x[i] * scale[i] + bias[i]\n",
    ),
    (
        "kernel_block_if_elif_else",
        "kernel bucket(a, out):\n    let i = thread_id()\n    let x = 0.0\n    if a[i] < 1.0:\n        x = 1.0\n    elif a[i] < 2.0:\n        x = 2.0\n    else:\n        x = 3.0\n    out[i] = x\n",
    ),
    (
        "kernel_block_guard_return",
        "kernel guard(a, out, n):\n    let i = thread_id()\n    if i >= n[0]:\n        return\n    out[i] = a[i] / 2.0\n",
    ),
    (
        "kernel_block_for_range",
        "kernel rowsum(a, out):\n    let i = thread_id()\n    let acc = 0.0\n    for j in range(0, 4):\n        acc += a[i * 4 + j]\n    out[i] = acc\n",
    ),
    (
        "kernel_block_while_break_continue",
        "kernel scan(a, out):\n    let i = thread_id()\n    let j = 0\n    let hits = 0\n    while j < 16:\n        j = j + 1\n        if a[i * 16 + j] < 0.0:\n            continue\n        if a[i * 16 + j] > 100.0:\n            break\n        hits = hits + 1\n    out[i] = hits\n",
    ),
    (
        "kernel_block_index_builtins",
        "kernel idx(out):\n    let x = thread_id()\n    let y = thread_id_y()\n    let b = block_id()\n    let d = block_dim()\n    if x % 2 == 0 and y < 4:\n        out[x + y * d + b] = 1.0\n    sync_threads()\n",
    ),
];

/// Parse NSL source and return the first `kernel` definition found (bare
/// or decorated), together with the interner.
pub fn parse_first_kernel(src: &str) -> (KernelDef, Interner) {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let errs: Vec<_> = lex_diags
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(errs.is_empty(), "lex errors: {errs:?}");
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let errs: Vec<_> = parsed
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(errs.is_empty(), "parse errors: {errs:?}");
    for stmt in &parsed.module.stmts {
        match &stmt.kind {
            StmtKind::KernelDef(k) => return (k.clone(), interner),
            StmtKind::Decorated { stmt: inner, .. } => {
                if let StmtKind::KernelDef(k) = &inner.kind {
                    return (k.clone(), interner);
                }
            }
            _ => {}
        }
    }
    panic!("no kernel definition in test source");
}
