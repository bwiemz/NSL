//! Refusal table tests (roadmap T1): each case is a malformed line
//! followed by `let after = 0`. The test pins the first diagnostic the
//! frontend reports and whether `let after = 0` still reached the module
//! as its own statement, i.e. whether the parser recovered at the end of
//! the bad line.
//!
//! Only the first diagnostic is pinned. What follows it is often a
//! cascade from the same mistake (see `recovery.rs`), and a table that
//! pinned the cascade would make every recovery improvement a test
//! failure. The cases that do not recover before the next line today are
//! listed with `false`; making one of them recover flips its entry.

#[path = "common/sexpr.rs"]
#[allow(dead_code, unused_macros)]
mod sexpr;

/// One `#[test]` per case: `name: source => first_diagnostic, recovers;`.
macro_rules! refusals {
    ($($name:ident: $src:expr => $first:expr, $recovers:expr;)*) => {$(
        #[test]
        fn $name() {
            let (first, recovered) = sexpr::refusal($src);
            assert_eq!(first, $first, "first diagnostic for {:?}", $src);
            assert_eq!(
                recovered, $recovers,
                "whether the line after {:?} survived",
                $src
            );
        }
    )*};
}

refusals! {
    let_without_a_name: "let = 1" => "error: expected pattern, found =", true;
    let_without_a_value: "let x = " => "error: expected expression, found newline", true;
    let_mut_is_not_nsl: "let mut x = 1" => "error: expected pattern, found Mut", true;
    let_type_missing: "let x: = 1" => "error: expected type, found =", true;
    tensor_type_missing_its_dtype: "let x: Tensor<[2], > = 1" => "error: expected identifier, found >", true;
    double_equals_sign: "x = = 1" => "error: expected expression, found =", true;
    compound_assign_without_a_value: "x +=" => "error: expected expression, found newline", true;
    two_expressions_on_a_line: "let x = 1 2" => "error: expected newline or end of statement, found 2", true;
    semicolon_terminator: "let x = 1;" => "error: expected newline or end of statement, found Semicolon", true;
    trailing_binary_operator: "let x = 1 +" => "error: expected expression, found newline", true;
    stray_closing_paren: "let x = )" => "error: expected expression, found )", true;
    return_of_return: "return return" => "error: expected expression, found return", true;
    keyword_argument_before_its_name: "f(=1)" => "error: expected expression, found =", true;
    empty_subscript: "xs[]" => "error: expected expression, found ]", true;
    lambda_without_its_closing_bar: "|x x" => "error: expected Bar, found identifier", true;
    import_without_a_path: "import" => "error: expected identifier, found newline", true;
    from_without_a_module: "from import x" => "error: expected import, found identifier", true;
}

refusals! {
    fn_without_a_name: "fn (x):\n    pass" => "error: expected identifier, found (", true;
    if_without_a_colon: "if x\n    pass" => "error: expected :, found newline", true;
    while_without_a_condition: "while:\n    pass" => "error: expected expression, found :", true;
    for_without_a_pattern: "for in xs:\n    pass" => "error: expected pattern, found in", true;
    for_without_in: "for x xs:\n    pass" => "error: expected in, found identifier", true;
    struct_without_a_name: "struct:\n    x: int" => "error: expected identifier, found :", true;
    enum_variant_that_is_a_number: "enum E:\n    1" => "error: expected identifier, found 1", true;
    decorator_without_a_name: "@\nfn f():\n    pass" => "error: expected identifier, found newline", true;
    model_member_that_is_an_expression: "model M:\n    1 + 2" => "error: expected identifier, found 1", true;
    match_arm_without_case: "match x:\n    1:\n        pass" => "error: expected newline or end of statement, found :", true;
    match_arms_not_indented: "match x:\ncase 1:\n    pass" => "error: expected INDENT, found case", true;
    else_without_if: "else:\n    pass" => "error: expected expression, found else", true;
    elif_without_if: "elif x:\n    pass" => "error: expected expression, found elif", true;
    case_outside_match: "case 1:\n    pass" => "error: expected expression, found case", true;
    unexpected_indent: "let x = 1\n    let y = 2" => "error: expected expression, found INDENT", true;
    indented_first_line: "  let x = 1" => "error: expected expression, found INDENT", true;
}

refusals! {
    lexer_unexpected_character: "let x = $" => "error: unexpected character: '$'", true;
    lexer_unterminated_string: "let s = \"unterminated" => "error: unterminated string literal", true;
    lexer_integer_too_large: "let n = 100000000000000000000" => "error: integer literal too large", true;
    unterminated_fstring_hole: "let x = f\"{\"" => "error: unterminated string literal", true;
}

// These do not recover before the next line today: the parser is still
// inside the unclosed construct, or inside a block it opened, when it
// reaches `let after = 0`.
refusals! {
    unclosed_call: "f(1, 2" => "error: expected ), found let", false;
    unclosed_list: "[1, 2" => "error: expected ], found let", false;
    unclosed_paren: "let x = (1" => "error: expected ), found let", false;
    unclosed_import_braces: "import a.{b" => "error: expected }, found let", false;
    dict_entry_without_a_value: "let x = {1: }" => "error: expected expression, found }", false;
    keyword_argument_without_a_value: "f(a=)" => "error: expected expression, found )", false;
    unclosed_parameter_list: "fn f(x:\n    pass" => "error: expected ), found let", false;
    if_body_not_indented: "if x:\npass" => "error: expected INDENT, found identifier", false;
    if_expression_without_a_colon: "let x = if c 1" => "error: expected :, found 1", false;
    struct_field_without_a_type: "struct S:\n    x" => "error: expected :, found newline", false;
    trait_member_that_is_not_a_fn: "trait T:\n    x = 1" => "error: expected fn, found identifier", false;
    async_before_let: "async let x = 1" => "error: expected fn, found let", false;
}

#[test]
fn every_bad_line_in_a_file_is_reported_and_the_good_ones_survive() {
    let parsed = sexpr::parse("let a = 1\nlet = 2\nlet b = 3\nlet = 4\nlet c = 5");
    assert_eq!(
        parsed.stmts,
        ["(let a 1)", "(let _)", "(let b 3)", "(let _)", "(let c 5)"]
    );
    let expected_pattern = parsed
        .diagnostics
        .iter()
        .filter(|d| *d == "error: expected pattern, found =")
        .count();
    assert_eq!(expected_pattern, 2, "{:#?}", parsed.diagnostics);
}
