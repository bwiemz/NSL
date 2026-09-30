//! Statement and declaration table tests (roadmap T1): one `#[test]` per
//! form, each pinning the statement a few lines of source become. The
//! rendering is the S-expression form of `common/sexpr.rs`; `(do ...)` is
//! a block, and a train, grad, kernel or other domain block renders as its
//! kind alone (the goldens in `tests/parse/` pin those in full).

#[path = "common/sexpr.rs"]
#[allow(dead_code)]
#[macro_use]
mod sexpr;

use sexpr::parse_clean;

cases! { parse_clean;
    let_binding: "let x = 1" => "(let x 1)";
    let_with_type: "let x: int = 1" => "(let x :int 1)";
    const_binding: "const N = 4" => "(const N 4)";
    let_destructures_a_tuple: "let (a, b) = t" => "(let (ptuple a b) t)";
    let_destructures_a_list: "let [a, b] = xs" => "(let [a b] xs)";
    let_with_tensor_type: "let x: Tensor<[2, 3], f32> = zeros([2, 3])" =>
        "(let x :(Tensor [2 3] f32) (call zeros [2 3]))";
    let_binds_a_lambda: "let f = |x| x + 1" => "(let f (lambda (x) (+ x 1)))";
    let_binds_a_two_parameter_lambda: "let g = |x, y| x * y" => "(let g (lambda (x y) (* x y)))";
    let_binds_a_nullary_lambda: "let h = || 0" => "(let h (lambda () 0))";
    lambda_parameter_with_type: "let k = |x: int| x" => "(let k (lambda (x:int) x))";
    if_expression_with_else: "let y = if c:\n    1\nelse:\n    2" =>
        "(let y (ifexpr c (block (do (expr 1))) (block (do (expr 2)))))";
    if_expression_without_else_gets_a_none_placeholder: "let y = if c:\n    1" =>
        "(let y (ifexpr c (block (do (expr 1))) none))";
    three_lets_in_a_row: "let a = 1\nlet b = 2\nlet c = 3" => "(let a 1)\n(let b 2)\n(let c 3)";
}

cases! { parse_clean;
    assign: "x = 1" => "(= x 1)";
    add_assign: "x += 1" => "(+= x 1)";
    sub_assign: "x -= 1" => "(-= x 1)";
    mul_assign: "x *= 2" => "(*= x 2)";
    div_assign: "x /= 2" => "(/= x 2)";
    assign_to_a_member: "a.b = c" => "(= (. a b) c)";
    assign_to_an_index: "a[0] = 1" => "(= (index a 0) 1)";
    assign_to_a_slice: "a[1:2] = b" => "(= (index a (slice 1 2 _)) b)";
    call_statement: "f(x)" => "(expr (call f x))";
    pass_is_an_identifier_statement: "pass" => "(expr pass)";
    break_statement: "break" => "(break)";
    continue_statement: "continue" => "(continue)";
    bare_return: "return" => "(return)";
    return_a_value: "return 1" => "(return 1)";
    return_a_tuple: "return (a, b)" => "(return (tuple a b))";
}

cases! { parse_clean;
    if_only: "if a:\n    b" => "(if a (do (expr b)))";
    if_else: "if a:\n    b\nelse:\n    c" => "(if a (do (expr b)) (else (do (expr c))))";
    if_elif_elif_else: "if a:\n    b\nelif c:\n    d\nelif e:\n    f\nelse:\n    g" =>
        "(if a (do (expr b)) (elif c (do (expr d))) (elif e (do (expr f))) (else (do (expr g))))";
    if_with_two_statements: "if a:\n    b\n    c" => "(if a (do (expr b) (expr c)))";
    nested_if: "if a:\n    if b:\n        c" => "(if a (do (if b (do (expr c)))))";
    while_loop: "while x < 10:\n    x += 1" => "(while (< x 10) (do (+= x 1)))";
    while_let: "while let Some(v) = it.next():\n    print(v)" =>
        "(whilelet (ctor Some v) (call (. it next)) (do (expr (call print v))))";
    for_over_a_range: "for i in 0..10:\n    print(i)" => "(for i (.. 0 10) (do (expr (call print i))))";
    for_destructures_a_tuple: "for (i, x) in xs.enumerate():\n    print(i)" =>
        "(for (ptuple i x) (call (. xs enumerate)) (do (expr (call print i))))";
    break_and_continue_in_a_loop: "for x in xs:\n    if x:\n        break\n    continue" =>
        "(for x xs (do (if x (do (break))) (continue)))";
    match_with_literal_and_wildcard: "match x:\n    case 1:\n        a\n    case _:\n        b" =>
        "(match x (case (lit 1) (do (expr a))) (case _ (do (expr b))))";
    match_with_guard: "match x:\n    case n if n > 0:\n        a" =>
        "(match x (case n (if (> n 0)) (do (expr a))))";
    statement_after_a_block: "if a:\n    b\nc" => "(if a (do (expr b)))\n(expr c)";
    comment_lines_are_skipped: "# a comment\nlet x = 1\n# another\n" => "(let x 1)";
    blank_lines_are_skipped: "let x = 1\n\n\nlet y = 2" => "(let x 1)\n(let y 2)";
}

cases! { parse_clean;
    fn_bare_return: "fn f():\n    return" => "(fn f () (do (return)))";
    fn_return_type: "fn f() -> int:\n    return 1" => "(fn f () -> int (do (return 1)))";
    fn_typed_parameters_and_default: "fn f(a: int, b: float = 1.0) -> float:\n    return a + b" =>
        "(fn f (a:int b:float=1.0) -> float (do (return (+ a b))))";
    fn_variadic_parameter: "fn f(*args):\n    pass" => "(fn f (*args) (do (expr pass)))";
    fn_generic: "fn id<T>(x: T) -> T:\n    return x" => "(fn id<T> (x:T) -> T (do (return x)))";
    async_fn: "async fn fetch(url: str) -> str:\n    return await get(url)" =>
        "(fn fetch async (url:str) -> str (do (return (await (call get url)))))";
    fn_yields: "fn f():\n    yield 1" => "(fn f () (do (yield 1)))";
    fn_untyped_parameters: "fn add(a, b):\n    return a + b" => "(fn add (a b) (do (return (+ a b))))";
    decorated_fn: "@test\nfn t():\n    pass" => "(decorated @test (fn t () (do (expr pass))))";
    decorator_with_keyword_argument: "@inline(always=true)\nfn t():\n    pass" =>
        "(decorated (@inline always=true) (fn t () (do (expr pass))))";
    two_decorators: "@a\n@b(1)\nfn t():\n    pass" => "(decorated @a (@b 1) (fn t () (do (expr pass))))";
    dotted_decorator: "@nsl.checkpoint\nfn t():\n    pass" =>
        "(decorated @nsl.checkpoint (fn t () (do (expr pass))))";
}

cases! { parse_clean;
    struct_two_fields: "struct Point:\n    x: float\n    y: float" => "(struct Point x:float y:float)";
    struct_field_default: "struct Config:\n    lr: float = 0.001" => "(struct Config lr:float=0.001)";
    generic_struct: "struct Pair<T>:\n    a: T\n    b: T" => "(struct Pair<T> a:T b:T)";
    enum_unit_variants: "enum Color:\n    Red\n    Green" => "(enum Color Red Green)";
    enum_tuple_variants: "enum Shape:\n    Circle(float)\n    Rect(float, float)" =>
        "(enum Shape Circle(float) Rect(float float))";
    enum_valued_variants: "enum Code:\n    Ok = 0\n    Err = 1" => "(enum Code Ok=0 Err=1)";
    trait_with_a_method: "trait Show:\n    fn show(self) -> str:\n        pass" =>
        "(trait Show (fn show (self) -> str (do (expr pass))))";
    model_with_a_layer: "model M:\n    w: Tensor<[4], f32> = zeros([4])" =>
        "(model M () (layer w:(Tensor [4] f32) (call zeros [4])))";
    model_with_parameters_layer_and_method:
        "model M(d: int):\n    fc: Linear = Linear(d, d)\n    fn forward(self, x: Tensor) -> Tensor:\n        return self.fc(x)" =>
        "(model M (d:int) (layer fc:Linear (call Linear d d)) (method (fn forward (self x:Tensor) -> Tensor (do (return (call (. self fc) x))))))";
}

cases! { parse_clean;
    import_module: "import math" => "(import math)";
    import_dotted_module: "import nsl.nn" => "(import nsl.nn)";
    import_with_alias: "import nsl.nn as nn" => "(import nsl.nn as nn)";
    import_named_items: "import nsl.nn.{Linear, ReLU}" => "(import nsl.nn {Linear ReLU})";
    import_named_item_with_alias: "import nsl.nn.{Linear as L}" => "(import nsl.nn {Linear as L})";
    import_glob: "import nsl.nn.*" => "(import nsl.nn *)";
    from_import_one: "from nsl.nn import Linear" => "(from nsl.nn {Linear})";
    from_import_two: "from nsl.nn import Linear, ReLU" => "(from nsl.nn {Linear ReLU})";
    from_import_glob: "from nsl.nn import *" => "(from nsl.nn *)";
}
