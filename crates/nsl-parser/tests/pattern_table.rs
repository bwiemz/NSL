//! Pattern table tests (roadmap T1): each case is parsed as the pattern of
//! a one-arm `match`, and the test pins the pattern it becomes in the
//! S-expression form of `common/sexpr.rs` — `(lit v)` a literal,
//! `(ctor Path args)` a constructor, `(ptuple ...)` a tuple, `..name` a
//! rest binding, a trailing `(if cond)` the arm's guard.

#[path = "common/sexpr.rs"]
#[allow(dead_code)]
#[macro_use]
mod sexpr;

use sexpr::pat;

cases! { pat;
    binding: "x" => "x";
    wildcard: "_" => "_";
    integer_literal: "1" => "(lit 1)";
    negative_integer_literal: "-1" => "(lit -1)";
    string_literal: "\"s\"" => "(lit \"s\")";
    bool_literal: "true" => "(lit true)";
    none_keyword_literal: "none" => "(lit none)";
    capitalised_none_is_a_binding: "None" => "None";
    tuple: "(a, b)" => "(ptuple a b)";
    tuple_with_wildcard: "(a, _)" => "(ptuple a _)";
    list: "[a, b]" => "[a b]";
    list_with_rest: "[first, *rest]" => "[first ..rest]";
    constructor: "Some(x)" => "(ctor Some x)";
    constructor_with_a_tuple_inside: "Some((a, b))" => "(ctor Some (ptuple a b))";
    qualified_constructor: "Shape.Rect(w, h)" => "(ctor Shape.Rect w h)";
    or_of_names: "Red | Green" => "(or Red Green)";
    or_of_three_literals: "1 | 2 | 3" => "(or (lit 1) (lit 2) (lit 3))";
    or_of_bools: "true | false" => "(or (lit true) (lit false))";
    guard: "n if n > 0" => "n (if (> n 0))";
    struct_fields: "{x, y}" => "(pstruct x y)";
    struct_field_subpattern_and_rest: "{x, y: (p, q), ..rest}" => "(pstruct x y:(ptuple p q) ..rest)";
}
