//! `@rope` is refused: it asked `@flash_attention` for in-kernel RoPE, but no
//! launch site ever passed the kernel its cos/sin tables (they are null at
//! every call), and the non-CSHA kernel it selected rotates Q only. A model
//! that used it compiled and would have read from address ~0 on its first
//! launch. RoPE belongs before the attention call, as `nsl.nn.gqa` applies it.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn errors(src: &str) -> Vec<String> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    let mut msgs: Vec<String> = lex_diags.iter().map(|d| d.message.clone()).collect();
    msgs.extend(parse_result.diagnostics.iter().map(|d| d.message.clone()));
    msgs.extend(
        analysis
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .map(|d| d.message.clone()),
    );
    msgs
}

#[test]
fn rope_on_flash_attention_is_refused_with_the_reason() {
    for rope in ["@rope", "@rope(style=\"adjacent\")"] {
        let src = format!("@flash_attention\n{rope}\nfn forward(x: Tensor) -> Tensor:\n    return x\n");
        let errs = errors(&src);
        assert_eq!(errs.len(), 1, "{rope}: exactly one error, got {errs:?}");
        assert!(errs[0].contains("@rope"), "{rope}: {errs:?}");
        assert!(errs[0].contains("never wired"), "{rope}: {errs:?}");
        assert!(errs[0].contains("before the attention call"), "{rope}: {errs:?}");
    }
}

#[test]
fn flash_attention_without_rope_still_compiles_clean() {
    let errs = errors("@flash_attention\n@gqa(groups=4)\nfn forward(x: Tensor) -> Tensor:\n    return x\n");
    assert!(errs.is_empty(), "{errs:?}");
}
