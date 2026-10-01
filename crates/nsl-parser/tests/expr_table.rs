//! Expression table tests (roadmap T1): one `#[test]` per construct, each
//! pinning how the parser groups one small expression. The rendering is
//! the S-expression form of `common/sexpr.rs`; `(paren e)` marks explicit
//! parentheses, `(neg e)` unary minus, `(index x i)` a subscript and
//! `(slice lo hi step)` a slice, with `_` for an omitted bound.

#[path = "common/sexpr.rs"]
#[allow(dead_code)]
#[macro_use]
mod sexpr;

use sexpr::expr;

// Binary precedence, loosest to tightest:
// `|>`, `or`, `and`, `is`/`in`, comparisons, `|`, `&`, ranges, `+ -`,
// `* / // %`, `@`, `**` (right-associative); then unary `-`, then postfix.
cases! { expr;
    mul_binds_tighter_than_add: "1 + 2 * 3" => "(+ 1 (* 2 3))";
    mul_before_add_on_the_left: "1 * 2 + 3" => "(+ (* 1 2) 3)";
    sub_is_left_associative: "1 - 2 - 3" => "(- (- 1 2) 3)";
    add_then_sub_is_left_associative: "a + b - c" => "(- (+ a b) c)";
    sub_then_add_is_left_associative: "a - b + c" => "(+ (- a b) c)";
    div_is_left_associative: "a / b / c" => "(/ (/ a b) c)";
    mul_and_div_share_a_level: "a * b / c" => "(/ (* a b) c)";
    floor_div_and_mod_share_a_level: "a // b % c" => "(% (// a b) c)";
    modulo: "a % b" => "(% a b)";
    pow_is_right_associative: "2 ** 3 ** 2" => "(** 2 (** 3 2))";
    pow_is_right_associative_without_spaces: "a**b**c" => "(** a (** b c))";
    unary_minus_binds_looser_than_pow: "-2 ** 2" => "(neg (** 2 2))";
    pow_takes_a_negated_exponent: "a ** -b" => "(** a (neg b))";
    pow_of_negative_literal: "2 ** -1" => "(** 2 (neg 1))";
    matmul_is_left_associative: "a @ b @ c" => "(@ (@ a b) c)";
    matmul_binds_tighter_than_add: "a @ b + c" => "(+ (@ a b) c)";
    matmul_binds_tighter_than_mul_on_the_left: "a @ b * c" => "(* (@ a b) c)";
    matmul_binds_tighter_than_mul_on_the_right: "a * b @ c" => "(* a (@ b c))";
    comparison_chain_is_left_associative: "a < b < c" => "(< (< a b) c)";
    comparisons_share_a_level: "a < b == c" => "(== (< a b) c)";
    le_then_ge: "a <= b >= c" => "(>= (<= a b) c)";
    not_equal: "a != b" => "(!= a b)";
    greater_than: "a > b" => "(> a b)";
    greater_or_equal: "a >= b" => "(>= a b)";
    bit_and_binds_tighter_than_bit_or: "a | b & c" => "(| a (& b c))";
    bit_and_binds_tighter_than_comparison: "a & b == c" => "(== (& a b) c)";
    bit_or_binds_tighter_than_comparison: "a | b == c" => "(== (| a b) c)";
    and_binds_tighter_than_or: "a or b and c" => "(or a (and b c))";
    not_binds_tighter_than_and: "not a and b" => "(and (not a) b)";
    and_not_or: "a and not b or c" => "(or (and a (not b)) c)";
    not_takes_a_whole_membership_test: "not a in b" => "(not (in a b))";
    not_takes_a_whole_comparison: "not a == b" => "(not (== a b))";
    membership: "a in b" => "(in a b)";
    identity_against_the_none_keyword: "a is none" => "(is a none)";
    membership_binds_tighter_than_and: "a in b and c" => "(and (in a b) c)";
    identity_binds_tighter_than_or: "a is b or c" => "(or (is a b) c)";
    boolean_literals_with_and: "true and false" => "(and true false)";
}

cases! { expr;
    pipe_is_left_associative: "a |> f |> g" => "(|> (|> a f) g)";
    pipe_binds_looser_than_or_on_the_left: "a or b |> f" => "(|> (or a b) f)";
    pipe_binds_looser_than_or_on_the_right: "a |> f or b" => "(|> a (or f b))";
    pipe_into_a_call: "a |> f(b)" => "(|> a (call f b))";
    pipe_from_a_list: "[x, y] |> f" => "(|> [x y] f)";
    range_exclusive: "1..10" => "(.. 1 10)";
    range_inclusive: "1..=10" => "(..= 1 10)";
    range_binds_looser_than_arithmetic: "a + 1..b * 2" => "(.. (+ a 1) (* b 2))";
    range_binds_tighter_than_comparison: "x == 0..3" => "(== x (.. 0 3))";
}

cases! { expr;
    unary_minus: "-x" => "(neg x)";
    double_unary_minus: "- -x" => "(neg (neg x))";
    triple_unary_minus: "- - - x" => "(neg (neg (neg x)))";
    double_not: "not not x" => "(not (not x))";
    not_of_negation: "not -x" => "(not (neg x))";
    minus_of_parenthesised_sum: "-(a + b)" => "(neg (paren (+ a b)))";
    minus_binds_looser_than_member_access: "-a.b" => "(neg (. a b))";
    minus_binds_looser_than_call: "-f(x)" => "(neg (call f x))";
    subtract_a_negation: "a - -b" => "(- a (neg b))";
    multiply_by_a_negative_literal: "2 * -3" => "(* 2 (neg 3))";
    zero_minus_one_is_binary: "0 - 1" => "(- 0 1)";
    await_takes_a_postfix_chain: "await a.b" => "(await (. a b))";
    await_a_call: "await f(x)" => "(await (call f x))";
}

cases! { expr;
    member_access_is_left_associative: "x.y.z" => "(. (. x y) z)";
    long_member_chain: "a.b.c.d" => "(. (. (. a b) c) d)";
    member_on_self: "self.w" => "(. self w)";
    curried_calls: "f(x)(y)" => "(call (call f x) y)";
    three_curried_calls: "f(a)(b)(c)" => "(call (call (call f a) b) c)";
    nested_calls: "f(g(h(x)))" => "(call f (call g (call h x)))";
    call_with_no_arguments: "f()" => "(call f)";
    method_call: "x.f(1, 2)" => "(call (. x f) 1 2)";
    call_then_member: "x.y(z).w" => "(. (call (. x y) z) w)";
    call_then_index: "a.b(c)[d]" => "(index (call (. a b) c) d)";
    index_then_member: "a[0].b" => "(. (index a 0) b)";
    member_index_member_call: "a.b[0].c()" => "(call (. (index (. a b) 0) c))";
    member_then_index: "x.shape[0]" => "(index (. x shape) 0)";
    keyword_argument: "f(a, b=1)" => "(call f a b=1)";
    keyword_arguments_only: "f(x=1, y=2)" => "(call f x=1 y=2)";
    lambda_as_keyword_argument: "f(a=|x| x)" => "(call f a=(lambda (x) x))";
    trailing_comma_in_call: "f(a, b,)" => "(call f a b)";
    call_arguments_across_lines: "f(\n    a,\n    b,\n)" => "(call f a b)";
}

cases! { expr;
    index: "a[0]" => "(index a 0)";
    index_twice: "x[i][j]" => "(index (index x i) j)";
    index_by_expression: "xs[i + 1]" => "(index xs (+ i 1))";
    index_by_negative_literal: "xs[-1]" => "(index xs (neg 1))";
    index_by_call: "xs[f(x)]" => "(index xs (call f x))";
    two_dimensional_index: "m[i, j]" => "(index m (dims i j))";
    slice_lower_and_upper: "a[1:2]" => "(index a (slice 1 2 _))";
    slice_all_three: "xs[i:j:k]" => "(index xs (slice i j k))";
    slice_step_only: "a[::2]" => "(index a (slice _ _ 2))";
    slice_everything: "a[:]" => "(index a (slice _ _ _))";
    slice_then_index: "a[:, 0]" => "(index a (dims (slice _ _ _) 0))";
    slice_and_index_mixed: "xs[1:2, 3]" => "(index xs (dims (slice 1 2 _) 3))";
    three_slices_with_a_negative_step: "a[1:, :3, ::-1]" =>
        "(index a (dims (slice 1 _ _) (slice _ 3 _) (slice _ _ (neg 1))))";
}

cases! { expr;
    list_literal: "[1, 2, 3]" => "[1 2 3]";
    empty_list: "[]" => "[]";
    nested_list: "[[1, 2], [3, 4]]" => "[[1 2] [3 4]]";
    trailing_comma_in_list: "[1, 2,]" => "[1 2]";
    list_across_lines: "[\n    1,\n    2,\n]" => "[1 2]";
    tuple: "(1, 2)" => "(tuple 1 2)";
    three_tuple: "(a, b, c)" => "(tuple a b c)";
    one_tuple_needs_the_comma: "(1,)" => "(tuple 1)";
    one_tuple_of_an_expression: "(a + b,)" => "(tuple (+ a b))";
    empty_tuple: "()" => "(tuple)";
    tuples_added: "(a,) + (b,)" => "(+ (tuple a) (tuple b))";
    parenthesised_name: "(a)" => "(paren a)";
    doubly_parenthesised_name: "((a))" => "(paren (paren a))";
    parentheses_override_precedence: "(a + b) * c" => "(* (paren (+ a b)) c)";
    parentheses_allow_a_line_break: "(1 +\n2)" => "(paren (+ 1 2))";
    dict_literal: "{\"a\": 1, \"b\": 2}" => "(dict \"a\":1 \"b\":2)";
    empty_dict: "{}" => "(dict)";
    dict_with_integer_key: "{1: \"a\"}" => "(dict 1:\"a\")";
    dict_with_list_value: "{\"k\": [1, 2]}" => "(dict \"k\":[1 2])";
    trailing_comma_in_dict: "{ \"a\": 1, }" => "(dict \"a\":1)";
    dict_across_lines: "{\n    \"a\": 1,\n    \"b\": 2,\n}" => "(dict \"a\":1 \"b\":2)";
}

cases! { expr;
    list_comprehension: "[x * 2 for x in xs]" => "(listcomp (* x 2) (for x xs))";
    comprehension_with_filter: "[x for x in xs if x > 0]" => "(listcomp x (for x xs (if (> x 0))))";
    comprehension_with_two_filters: "[x for x in xs if a if b]" => "(listcomp x (for x xs (if a) (if b)))";
    comprehension_over_two_generators: "[(a, b) for a in xs for b in ys]" =>
        "(listcomp (tuple a b) (for a xs) (for b ys))";
    comprehension_over_a_call: "[f(x) for x in range(10)]" => "(listcomp (call f x) (for x (call range 10)))";
    lambda_one_parameter: "|x| x" => "(lambda (x) x)";
    lambda_body_is_a_whole_expression: "|x| x * 2" => "(lambda (x) (* x 2))";
    lambda_returning_a_lambda: "|x| |y| x + y" => "(lambda (x) (lambda (y) (+ x y)))";
    lambda_as_first_argument: "f(|x| x * 2, xs)" => "(call f (lambda (x) (* x 2)) xs)";
    lambda_passed_to_a_method: "xs.map(|x| x)" => "(call (. xs map) (lambda (x) x))";
}

cases! { expr;
    integer: "42" => "42";
    integer_with_underscores: "1_000" => "1000";
    hexadecimal: "0x1F" => "31";
    hexadecimal_plus_binary: "0x10 + 0b11" => "(+ 16 3)";
    float: "3.14" => "3.14";
    float_below_one: "0.5" => "0.5";
    float_with_underscore: "1_0.5" => "10.5";
    float_with_negative_exponent: "1e-3" => "0.001";
    float_with_exponent: "1e3" => "1000.0";
    float_with_mantissa_and_exponent: "1.5e10" => "15000000000.0";
    double_quoted_string: "\"str\"" => "\"str\"";
    single_quoted_string: "'single'" => "\"single\"";
    string_escape: "\"a\\nb\"" => "\"a\\nb\"";
    strings_of_both_quotes_added: "'a' + \"b\"" => "(+ \"a\" \"b\")";
    fstring_text_and_hole: "f\"hello {name}!\"" => "(f \"hello \" {name} \"!\")";
    fstring_two_adjacent_holes: "f\"{a}{b}\"" => "(f {a} {b})";
    fstring_hole_holds_an_expression: "f\"x={x + 1}\"" => "(f \"x=\" {(+ x 1)})";
    fstring_without_holes: "f\"plain\"" => "(f \"plain\")";
    true_literal: "true" => "true";
    false_literal: "false" => "false";
    none_keyword: "none" => "none";
    capitalised_none_is_an_identifier: "None" => "None";
}
