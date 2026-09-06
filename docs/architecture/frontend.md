# Frontend architecture: nsl-lexer, nsl-parser, nsl-ast, nsl-errors

The frontend turns one `.nsl` source file into an AST plus a list of
diagnostics. It is four small crates that depend on each other in one
direction only: `nsl-errors` (spans and diagnostics, no NSL knowledge) is
below everything; `nsl-lexer` depends on it; `nsl-ast` depends on it and on
`string-interner`; `nsl-parser` depends on all three. Nothing in the
frontend knows about types, modules, or codegen — `nsl-semantic` consumes
the `Module` the parser returns, and `nsl-cli` (`crates/nsl-cli/src/loader.rs`,
`crates/nsl-cli/src/pipeline.rs`) is the driver that wires the stages
together and decides what an error-level diagnostic means for the build.

Two design choices shape everything below. The lexer is *indentation-aware*
(NSL is block-by-indentation, like Python), so it synthesizes `Newline`,
`Indent` and `Dedent` tokens and the parser never looks at whitespace. And
the frontend *never aborts*: both the lexer and the parser return a complete
result plus a `Vec<Diagnostic>`, producing `Error` tokens and `Error` nodes
where the input was wrong, so one bad line reports one diagnostic and the
rest of the file still parses.

```
  source text (String)                    crates/nsl-cli/src/loader.rs
        │                                 SourceMap::add_file → FileId
        ▼
  nsl_lexer::tokenize(&str, FileId, &mut Interner)
        │  → (Vec<Token>, Vec<Diagnostic>)        crates/nsl-lexer/src/lexer.rs
        │    Newline / Indent / Dedent / Eof synthesized here
        ▼
  nsl_parser::parse(&[Token], &mut Interner)
        │  → ParseResult { module: Module, diagnostics }   crates/nsl-parser/src/lib.rs
        │    recursive descent for statements, Pratt for expressions
        ▼
  nsl_ast::Module                          crates/nsl-ast/src/lib.rs
        │  every Stmt / Expr / TypeExpr / Pattern carries a Span (+ NodeId)
        ▼
  SourceMap::emit_diagnostic(&Diagnostic)  crates/nsl-errors/src/source.rs
       renders each diagnostic against the file it came from (codespan-reporting)
```

Sizes (`wc -l`): `nsl-errors/src` 285 lines; `nsl-lexer/src` 1,452
(`lexer.rs` 571); `nsl-ast/src` 1,698; `nsl-parser/src` 4,919, of which
`block.rs` (train, distill, quant, kernel, tokenizer, dataset, datatype,
serve) is 1,366 and `expr.rs` 770.

---

## nsl-errors

`crates/nsl-errors/src/` has three modules re-exported flat from `lib.rs`:
`span.rs`, `diagnostic.rs`, `source.rs`. Every crate above it imports
`nsl_errors::{Span, FileId, BytePos, Diagnostic, Level, SourceMap}`.

### Positions: `FileId`, `BytePos`, `Span` (`span.rs`)

- `FileId(pub usize)` is the index a `SourceMap` handed back from `add_file`.
- `BytePos(pub u32)` is a byte offset into that file's source. Bytes, not
  chars and not line/column: the lexer's `Cursor` works in bytes, and
  line/column is computed only at render time (by codespan-reporting in
  `emit_diagnostic`, or by the test harnesses' own `LineIndex`).
- `Span { file_id, start, end }` is the half-open range `[start, end)` in one
  file. `Span::merge` takes the min start and max end and **asserts both
  spans are in the same file** — this assertion is the reason for several
  parser rules below. `Span::DUMMY` (`FileId(0)`, `0..0`) marks synthesized
  nodes; it is a *valid* empty range at the start of file 0, which is why
  `SourceMap::contains_span` refuses it explicitly (see below).

### Diagnostics (`diagnostic.rs`)

`Diagnostic { level: Level, message, labels: Vec<Label>, notes: Vec<String> }`
with `Level::{Error, Warning, Info}`. A `Label { span, message, style }` has
`LabelStyle::Primary` or `Secondary`. There is no error code field; the
message is the identity. Construction is builder-style:

```rust
Diagnostic::error("expected identifier, found `)`")
    .with_label(span, "expected identifier")          // Primary
    .with_secondary_label(other_span, "opened here")   // Secondary
    .with_note("tabs are not allowed; use 4 spaces")
```

Nothing in this crate stores diagnostics; each stage returns its own
`Vec<Diagnostic>` and the driver decides what to do with them.

### Rendering: `SourceMap` (`source.rs`)

`SourceMap` wraps `codespan_reporting::files::SimpleFiles<String, String>`.
`add_file(name, source) -> FileId` registers a file; `SourceMap::silent()`
makes a map whose `emit_diagnostic` renders nothing (the benches and any
caller that drives the frontend repeatedly use it — diagnostics still come
back through return values, only the stderr rendering is dropped).

`emit_diagnostic(&self, diag)` converts to a codespan diagnostic
(`Level::Info` becomes codespan's `Note` severity), maps each `Label` by
`convert_label`, attaches notes, and writes to stderr with
`term::emit_to_write_style`. Two details are deliberate and commented in the
source: colour is decided by `std::io::stderr().is_terminal()` rather than
codespan's `ColorChoice::Auto`, because `Auto` only consults `TERM` and would
splice ANSI codes into a piped `nsl build 2> log`; and the renderer's result
is discarded (`let _ =`), so a label the map cannot locate prints the
`error: …` header and then nothing.

That silent failure is what `contains_span(&self, span) -> bool` exists for.
It returns true only if `span.file_id` names a file in this map, the range is
not inverted, and `end` is within the source — and never for `Span::DUMMY`.
`crates/nsl-cli/src/pipeline.rs` (`exit_on_codegen_error`) is the model
caller: a `CodegenError` with a span the map contains is rendered through
`emit_diagnostic` like a frontend diagnostic; otherwise it falls back to a
plain `codegen error: …` line. Any code that renders a span it did not get
from this same `SourceMap` should do the same check.

The rendered form is codespan-reporting's standard layout: `error:
<message>`, `--> <file>:<line>:<col>`, the excerpt with primary labels
underlined and secondary ones in the softer style, then one `= <note>` line
per note.

---

## nsl-lexer

Entry point (`crates/nsl-lexer/src/lib.rs`):

```rust
pub fn tokenize(source: &str, file_id: FileId, interner: &mut Interner)
    -> (Vec<Token>, Vec<Diagnostic>)
```

`Interner` is a type alias for
`StringInterner<BucketBackend<DefaultSymbol>>`; the same interner must be
passed on to `nsl_parser::parse` so the parser can resolve and intern
symbols consistently. `tokenize` constructs `lexer::Lexer::new(...)` and
calls `Lexer::tokenize(self)`, which consumes the lexer and returns both
vectors. There is no streaming interface; the whole token vector is built up
front (the parser indexes into it with `pos`).

### Tokens (`token.rs`, `keywords.rs`)

`Token { kind: TokenKind, span: Span }`. `TokenKind` is one flat enum
(244-line file) grouped by comment banners: literals (`IntLiteral(i64)`,
`FloatLiteral(f64)`, `StringLiteral(String)`, the five f-string kinds),
`Ident(Symbol)`, keywords (variable, function, control-flow, ML-block, module,
literal, logical, type, compound-type groups), operators, delimiters,
punctuation, the three synthetic indentation kinds `Newline` / `Indent` /
`Dedent`, and `DocComment(String)`, `Eof`, `Error(String)`.

`nsl_lexer::Symbol` is `string_interner::DefaultSymbol` itself; the AST
crate wraps it in its own `nsl_ast::Symbol` newtype (see below).

`keywords.rs::lookup_keyword(&str) -> Option<TokenKind>` is a single
`match`; adding a keyword is one arm there plus one variant in `TokenKind`.
Keywords are reserved everywhere — `tokenizer` is a keyword, so a program
that uses it as a variable name fails to parse (see the `EXPECTED_SKIPS`
list in `crates/nsl-cli/tests/fmt_roundtrip.rs` for a real example).

`impl Display for TokenKind` is what the parser's "expected X, found Y"
messages print; kinds without an explicit arm fall back to `Debug` (so an
unmapped operator shows as e.g. `DoubleStar` — add a `Display` arm when you
add a token that users will see in messages).

Two kinds are declared but never produced by the lexer: `Underscore` and
`Hash`. `_` goes down the identifier path in `scan_token` and comes out as
`Ident("_")`, and `#` always starts a comment. The `TokenKind::Underscore`
arms in `crates/nsl-parser/src/pattern.rs` and `types.rs` are therefore
dead: `case _:` parses today as `PatternKind::Ident("_")` (the
`match_and_patterns.ast.snap` golden shows `Ident(`_`)`), and wildcard
semantics are resolved downstream by name.

### The scanning loop (`lexer.rs`, `cursor.rs`)

`Cursor<'a>` (`cursor.rs`) is a byte-position cursor over `&str` with
`peek`, `peek_at(n)` (O(n) in chars), `advance`, `eat(char)`,
`eat_while(pred) -> &'a str`, and `span_from(start: BytePos) -> Span`. It
knows the `FileId`, so every span the lexer builds is in the right file.

`Lexer` holds the cursor, an `IndentTracker`, the interner, the output
`tokens` and `diagnostics`, and three bits of mode state:
`at_line_start`, `line_continuation`, and `fstring_stack`.
`Lexer::tokenize` loops `scan_token` until EOF, then appends a final
`Newline` if the last token was not one (so every statement ends in a
`Newline`, even on a file without a trailing newline), the DEDENTs that
close every still-open block (`IndentTracker::finalize`), and exactly one
`Eof`. The parser relies on that `Eof` being present and last.

`scan_token` is one big `match` on the current char. Operators use
`cursor.eat` for their longest-match second/third character (`-` then `>` is
`Arrow`, `.` `.` `=` is `DotDotEq`, `|` `>` is `Pipe`, bare `|` is `Bar`).
Brackets call `IndentTracker::open_bracket` / `close_bracket` as a side
effect of being emitted. `push_token(kind, start)` computes the span from
`start` to the cursor's current position.

### Indentation: `Newline`, `Indent`, `Dedent` (`indent.rs`)

`IndentTracker` keeps a stack of indentation levels (always starting
`[0]`) and a `bracket_depth`. The rules:

- A `\n` (or `\r\n`, or bare `\r`) emits a `Newline` token **unless** the
  lexer is inside brackets (`bracket_depth > 0`) or the previous line ended
  in a `\` continuation. Either way it sets `at_line_start`.
- At the start of the next line, `process_line_indentation` counts leading
  spaces, then peeks: a blank line, a comment-only line (`#` first), or EOF
  **returns without touching the stack**, so blank and comment lines never
  open or close a block whatever their indentation (the
  `crates/nsl-lexer/tests/lex/indentation.nsl` fixture pins this). A tab in
  the indentation is an error diagnostic. Otherwise it calls
  `IndentTracker::process_indent(level, span)`.
- `process_indent` returns `[Indent]` if the level is deeper than the top of
  stack (pushing it), nothing if equal, or one `Dedent` per popped level if
  shallower. If the new level does not match any level on the stack it
  returns `Err(String)`; the lexer turns that into a "dedent does not match
  any outer indentation level" diagnostic and emits **no** tokens for the
  line, so the parser sees the line as a continuation of the current block.
  Inside brackets `process_indent` returns nothing at all.
- `Indent` and `Dedent` tokens have *empty* spans at the first non-space
  character of the line (`3:5-3:5 Indent` in the goldens). `Newline` spans
  the newline character(s).

There is no fixed indent width; any deeper level opens a block ("use 4
spaces" in the tab message is convention, not enforced).

### Comments and continuations

`#` starts a line comment, consumed silently. `##` is a doc comment and
emits `DocComment(String)` (leading `## ` stripped) — but the parser's
`skip_newlines` skips `DocComment` alongside `Newline`, so doc comments are
tokenized and then dropped; nothing in the AST carries them today. A line
starting `#[` (attribute syntax) is skipped whole with a "for now" comment in
`scan_token`. A `\` immediately before a newline sets `line_continuation`,
which suppresses the `Newline` and skips the next line's indentation; a `\`
anywhere else is a `Backslash` token.

### Identifiers, keywords, f-strings

An identifier starts with an ASCII letter or `_` and continues with
`char::is_alphanumeric` (so Unicode letters are accepted after the first
char but not as it; a non-ASCII first char is an "unexpected character"
error). `lex_identifier` builds a `String`, `lookup_keyword` decides keyword
vs. `Ident`, and identifiers are interned with `get_or_intern`.

`f"..."` / `f'...'` (and triple-quoted forms) are a token *sequence*:
`FStringStart`, then `FStringText(String)` and
`FStringExprStart … <ordinary tokens> … FStringExprEnd` groups, then
`FStringEnd`. `lex_fstring_text_with_quote` scans text until `{`, pushes
`(brace_depth, quote, is_triple)` on `fstring_stack`, and returns to the
main loop, which lexes the expression normally while counting braces; the
`}` at depth 0 pops the stack and resumes text. `{{`/`}}` are literal. The
parser's `parse_fstring` consumes exactly this protocol.

### String and number literals (`strings.rs`, `numbers.rs`)

`lex_string(cursor, quote, start, diagnostics) -> TokenKind` handles `'`
and `"`; a third quote switches to `lex_triple_string`, which may span
lines. Single-line strings hitting a newline or EOF are "unterminated string
literal" and return `TokenKind::Error`. Escapes (`lex_escape`): `\n \t \r \\
\' \" \0`, `\xHH`, `\u{H..H}` (1–6 hex digits). An unknown escape is
reported *and* the character is kept (`\q` yields `q`), so the token is
still a `StringLiteral`.

`lex_number` dispatches on `0x`/`0o`/`0b` to `lex_hex`/`lex_octal`/
`lex_binary`, else `lex_decimal`. Underscores are allowed anywhere in the
digits and stripped before parsing. Two lookahead rules keep the grammar
unambiguous: a `.` is part of a float only if a digit follows (so `1..5` is
a range and `1.abs()` is a member access), and `e`/`E` is an exponent only
if digits (optionally signed) follow (so `123east` is `123` then an
identifier). Overflow of `i64`/`f64` parsing is a diagnostic plus an `Error`
token.

### Error recovery in the lexer

Every lexical error produces both a `Diagnostic` and a
`TokenKind::Error(String)` token at that position, and the lexer keeps
going. The parser sees the `Error` token as "found error: …" in whatever
`expect` it reaches, reports that, and recovers like for any unexpected
token — `crates/nsl-parser/tests/recovery.rs::lexer_errors_become_error_tokens_and_the_parser_carries_on`
pins the interaction. Bare `!` gets a targeted message ("did you mean
'not'?"). Tabs outside indentation are consumed with the same tab
diagnostic.

---

## nsl-ast

`crates/nsl-ast/src/lib.rs` declares the modules and the three shared
primitives. All node types derive `Debug, Clone, Serialize` (serde) — the
parser goldens are the pretty `Debug` form, and the CLI's `nsl check
--dump-ast` prints it.

### `Symbol`, `NodeId`, `Span`

- `nsl_ast::Symbol(pub string_interner::DefaultSymbol)` is a newtype so it
  can implement `Serialize` (as the raw index). `From<DefaultSymbol>` lets
  the parser write `sym.into()`; resolve with `interner.resolve(sym.0)`.
- `NodeId(pub u32)` comes from `NodeId::next()`, a process-global
  `AtomicU32`. Ids are unique within a process but not stable across runs or
  test orderings — that is why the goldens strip them. `NodeId::dummy()` is
  `u32::MAX`.
- `Span` is `nsl_errors::Span` re-exported.

The four "big" node structs — `Stmt`, `Expr`, `TypeExpr`, `Pattern` — all
have the same shape: `{ kind: XKind, span: Span, id: NodeId }`. `Block
{ stmts, span }` and `Module { stmts, span }` have a span but no id. The
declaration structs (`FnDef`, `ModelDef`, `StructDef`, `Decorator`, `Param`,
`Arg`, every struct in `block.rs`) carry a `span` but no `NodeId`.

### The node families

| File | Types | Notes |
|------|-------|-------|
| `stmt.rs` | `Stmt`, `Block`, `StmtKind` | `StmtKind` has 30 variants: `VarDecl`, the declaration wrappers (`FnDef`, `ModelDef`, `AgentDef`, `StructDef`, `EnumDef`, `TraitDef`), control flow (`If` with `elif_clauses: Vec<(Expr, Block)>`, `For`, `While`, `WhileLet`, `Match`, `Break`, `Continue`, `Return`, `Yield`), `Assign { target, op: AssignOp, value }`, `Import`/`FromImport`, the ML blocks (`TrainBlock`, `DistillBlock`, `GradBlock`, `QuantBlock`, `KernelDef`, `TokenizerDef`, `DatasetDef`, `DatatypeDef`, `ServeBlock`), `Decorated { decorators, stmt: Box<Stmt> }`, and `Expr`. |
| `expr.rs` | `Expr`, `ExprKind`, `FStringPart`, `SubscriptKind`, `Arg`, `LambdaParam`, `CompGenerator`, `MatchArm` | `ExprKind` has literals, `Ident`/`SelfRef`, `BinaryOp`/`UnaryOp`/`Pipe`, `MemberAccess`/`Subscript`/`Call`, `Lambda`, `BlockExpr`, `ListComp`, `IfExpr`, `MatchExpr`, `Range`, `Paren`, `Await`, and `Error`. `Arg { name: Option<Symbol>, value, span }` is both call arguments and decorator/config kwargs. |
| `operator.rs` | `BinOp` (20 variants incl. `MatMul`, `FloorDiv`, `Is`, `In`, `BitOr`, `BitAnd`), `UnaryOp { Neg, Not }`, `AssignOp` | The parser's `pratt::token_to_binop` maps tokens onto `BinOp`. |
| `types.rs` | `TypeExpr`, `TypeExprKind`, `DimExpr`, `DimValue`, `DeviceExpr` | Tensor-shaped types: `Tensor { shape: Vec<DimExpr>, dtype, device }`, `Param`, `Buffer`, `Sparse`; plus `Named`, `Generic`, `Function` (with optional `EffectExpr`), `Union`, `Tuple`, `Wildcard`, `FixedArray`, `Borrow`. `DimExpr` covers concrete, symbolic, named, bounded and wildcard dims. |
| `pattern.rs` | `Pattern`, `PatternKind`, `FieldPattern` | `Ident`, `Wildcard`, `Literal(Box<Expr>)`, `Tuple`, `List`, `Struct { fields, rest }`, `Constructor { path, args }`, `Or`, `Guarded`, `Rest`, `Typed`. |
| `decl.rs` | `FnDef`, `Param`, `TypeParam`, `EffectExpr`, `ModelDef`, `ModelMember`, `StructDef`, `StructField`, `EnumDef`, `EnumVariant`, `TraitDef`, `Decorator`, `ImportStmt`, `FromImportStmt`, `ImportItems`, `ImportItem` | `ModelMember::{LayerDecl { decorators, .. }, Method(FnDef, Vec<Decorator>)}` carry member-level decorators. |
| `agent.rs` | `AgentDef`, `AgentMember` | Mirrors `ModelDef`/`ModelMember` (`FieldDecl`, `Method`). |
| `block.rs` | `TrainBlock`, `TrainSection`, `CallbackDef`, `DistillBlock`, `GradBlock`, `QuantBlock` (+ `QuantKind`, `QuantDtype`, `QuantGranularity`, `CalibrationConfig`), `KernelDef`, `KeyValueEntry`, `TokenizerDef`/`TokenizerStmt`, `DatasetDef`, `DatatypeDef`/`DatatypeMethod`/`DatatypePtxBlock`, `ServeBlock`/`ServeConfigEntry`/`ServeSubBlock`/`EndpointDef`, `WrgaBlock`/`WrgaMode` | The ML-specific statement payloads. |

Some things worth knowing about `block.rs`:

- `TrainBlock { config: Vec<Arg>, sections: Vec<TrainSection>, span }`.
  `config` is the kwargs from `train(model=m, epochs=10):` — the parser
  accepts any keyword; the closed set lives in `nsl_semantic::train_config`.
  `TrainSection` is `Data(Vec<Stmt>)`, `Optimizer(Expr)`, `Scheduler(Expr)`,
  `Step { param, body }`, `Eval { param, body }`, `Callbacks(Vec<CallbackDef>)`,
  `Distribute(Expr)`, or `Stmt(Box<Stmt>)` for an ordinary statement in the
  body. Section names (`data:`, `step(batch):`) are *contextual
  identifiers*, not keywords.
- `DistillBlock` reuses `TrainSection` and adds `loss: Vec<Arg>` for its
  `loss:` section.
- `QuantBlock` is not a generic kwargs bag; the parser validates
  `quant static Name from Source:` and its `dtype`/`granularity`/`exclude`/
  `calibration` entries into typed fields at parse time.
- `WrgaBlock` / `WrgaMode` are **not produced by the parser**. They are the
  validated form of a `@wrga(...)` decorator that `nsl-semantic` and
  `nsl-codegen` share; the module comment explains why they live here.
- `ServeSubBlock` accepts any identifier as a section key; which keys mean
  something is decided in `nsl-semantic`.

### Decorators

`Decorator { name: Vec<Symbol>, args: Option<Vec<Arg>>, span }`. `name` is a
dotted path (`@nsl.export`, `@a.b.c`), one `Symbol` per segment. `args` is
`None` for `@x` and `Some(vec![])` for `@x()`. Decorators sit in five
carrier positions: `StmtKind::Decorated`, `ModelMember::LayerDecl.decorators`,
`ModelMember::Method(_, decos)`, `AgentMember::{FieldDecl.decorators,
Method(_, decos)}`, and `KernelDef.decorators`.

### Walks: `visitor.rs` and `decorator_walk.rs`

`visitor::Visitor` is the generic read-only walk: a trait with
`visit_module`, `visit_stmt`, `visit_expr`, `visit_type_expr`,
`visit_pattern`, `visit_block`, each defaulting to the matching free
`walk_*` function. Two things to know: `visit_type_expr` has **no** default
walk (types are leaves to the visitor), and `walk_stmt` visits bodies but
**not** the member-level decorator vectors.

`decorator_walk::collect_decorators(&Module) -> Vec<DecoratorUse<'_>>`
exists for exactly that gap: it finds every decorator in source order with
its `DecoratorHost` (`Function`, `Model`, `TrainBlock`, `DistillBlock`,
`Kernel`, `Agent`, `ModelLayer`, `ModelMethod`, `AgentField`, `AgentMethod`,
`OtherStmt`). It hand-recurses rather than implementing `Visitor` because
the trait's `&Stmt` has an anonymous lifetime and the collector needs to
keep `&'a Decorator`. `nsl_semantic` (`crates/nsl-semantic/src/lib.rs`)
runs it to enforce the closed decorator namespace against
`decorator_registry`; the module comment records the bug that motivated it
(`@totally_not_a_real_feature` compiled silently).

---

## nsl-parser

Entry point (`crates/nsl-parser/src/lib.rs`):

```rust
pub struct ParseResult { pub module: Module, pub diagnostics: Vec<Diagnostic> }
pub fn parse(tokens: &[Token], interner: &mut Interner) -> ParseResult
```

`parse` builds a `parser::Parser`, loops `stmt::parse_stmt` until `Eof`
(skipping newlines between statements), merges the first and last statement
spans for `Module.span` (or `Span::dummy()` for an empty file), and returns
`Parser::finish()` as the diagnostics. There is no `Result` anywhere in the
crate's public surface: a module always comes back.

### `Parser` (`parser.rs`)

```rust
pub struct Parser<'a> {
    tokens: &'a [Token], pos: usize,
    pub diagnostics: Vec<Diagnostic>,
    pub interner: &'a mut Interner,
    depth: u32, nesting_overflow: Option<usize>,
}
```

Token inspection: `peek()`, `peek_token()`, `peek_at(offset)`, `at(&kind)`
(compares `mem::discriminant`, so `at(&TokenKind::Ident(_))` needs a sample
symbol but matches any identifier), `at_any(&[..])`, `current_span()`,
`prev_span()`. Consumption: `advance()`, `eat(&kind) -> bool`,
`expect(&kind) -> Span`, `expect_ident() -> (Symbol, Span)`,
`expect_ident_or_keyword()` (import paths may contain keywords such as
`nsl.quant`), `skip_newlines()` (skips `Newline` **and** `DocComment`),
`expect_end_of_stmt()`. Helpers: `next_node_id()`, `intern(&str)`,
`resolve(Symbol)`.

The grammar itself is free functions taking `&mut Parser`, one file per
family (`stmt.rs`, `expr.rs`, `types.rs`, `pattern.rs`, `decl.rs`,
`block.rs`, `agent.rs`), plus `Parser::parse_block` in `parser.rs` because
every family needs it.

`advance()` never steps past the `Eof` token: consuming `Eof` returns it
again. The reason is `EOF_TOKEN`, a sentinel with `Span::DUMMY` in
`FileId(0)` that `peek_token` returns for an out-of-range `pos`; any
`Span::merge` of that with a span from a real file (every imported module
is `FileId(1)` or higher) would panic. The parse fuzz target found this on
`let x = f(` followed by a newline. The test harness in
`crates/nsl-parser/tests/common/frontend_check.rs` lexes as `FileId(1)` for
the same reason: with file 0 the panic would be masked.

### The diagnostics-accumulation model

Every `parse_*` function returns a node. When the input is wrong it pushes
onto `p.diagnostics` and returns *something* with a real span:

- `expect(kind)` reports "expected `kind`, found `other`" (both via
  `TokenKind`'s `Display`), advances **one** token unless at `Eof` (so a
  loop calling it cannot spin), and returns the span it was looking at.
- `expect_ident` does the same and returns the interned symbol `<error>`.
- `parse_prefix_or_atom` on a token that cannot start an expression reports
  "expected expression, found …", advances one token, and returns
  `ExprKind::Error`. `parse_stmt` at the nesting limit returns a
  `StmtKind::Expr(Error)`; `parse_primary_type` returns `Named("error")`;
  `parse_primary_pattern` returns `Wildcard`.
- `parse_expr_or_assign` reports "invalid assignment target" but still
  builds the `Assign`.

`Parser::finish()` is the only post-processing: after a nesting-limit error,
the recursion unwinds through every open construct and each would report a
bogus "expected `)`, found EOF"; `finish` truncates the vector at the
recorded index so only the diagnostics before the limit plus the limit error
survive.

The driver decides severity policy. `crates/nsl-cli/src/loader.rs` emits
every lexer and parser diagnostic and returns `Err` for the module if any
has `Level::Error`, so semantic analysis never sees an AST containing
`Error` nodes in a real build — but tools that only need tokens or a
best-effort tree (`nsl fmt`, `nsl check --dump-tokens`) keep going.

### Statements and blocks (`stmt.rs`, `parser.rs`)

`stmt::parse_stmt` is the single statement entry (top level, block bodies,
train/distill bodies, datatype methods, after `pub`/`priv` all route through
it). It wraps `parse_stmt_nested`, a `match p.peek()` dispatch: `@` goes to
`parse_decorated_stmt`; each statement keyword to its `parse_*_stmt` in
`stmt.rs`, `decl.rs`, `block.rs` or `agent.rs`; `Pub`/`Priv` to
`parse_visibility_prefixed` (which currently **drops** the visibility and
returns the inner statement); anything else to `parse_expr_or_assign`,
which parses an expression and then checks for an assignment operator.

Statement parsers follow one pattern: record `start = p.current_span()`,
consume the keyword, parse the pieces, and for a line statement call
`p.expect_end_of_stmt()`, for a suite call `p.parse_block()`. The node's
span is `start.merge(<last piece>.span)`.

`Parser::parse_block` (with `parse_block_nested`) is the suite grammar:
`skip_newlines`, `expect(Indent)`, statements until `Dedent`/`Eof`,
`eat(Dedent)`. Its span runs from the (empty) `Indent` span to the token at
the close. `match` is the one statement that reads `Indent`/`Dedent` itself
(`parse_match_stmt`) because its body is `case` arms, not statements.

`expect_end_of_stmt` accepts `Newline`, `Eof`, `Dedent`, or "the previous
token was a `Dedent`" (a statement ending in a block has already consumed
its newline), otherwise reports and calls `synchronize`.

One special case worth knowing: `parse_var_decl` peeks for `let x =
grad(targets):` and builds a `GradBlock` with `outputs: Some(pattern)`
instead of a `VarDecl`.

### Expressions: Pratt (`expr.rs`, `pratt.rs`)

`expr::parse_expr(p)` is `parse_expr_bp(p, 0)`. `parse_expr_bp_nested`
parses a prefix/atom with `parse_prefix_or_atom`, then loops:

1. **postfix** — if `pratt::postfix_binding_power(peek)` (`.`, `[`, `(` at
   30) is at least `min_bp`, `parse_postfix` builds `MemberAccess`,
   `Subscript` (via `parse_subscript`, which handles slices and multi-dim
   indices into `SubscriptKind`) or `Call` (via `parse_args`);
2. **pipe** `|>` — its own branch because it builds `ExprKind::Pipe`, not
   `BinaryOp`;
3. **range** `..` / `..=` — its own branch because the end is optional
   (`can_start_expr` decides) and it builds `ExprKind::Range`;
4. **infix** — `infix_binding_power` plus `token_to_binop` build `BinaryOp`.

Binding powers (`pratt.rs`; `(left, right)`, left < right is
left-associative):

| Operators | bp | |
|-----------|----|-|
| `\|>` | 2, 3 | lowest |
| `or` | 4, 5 | |
| `and` | 6, 7 | |
| `is`, `in` | 8, 9 | |
| `==` `!=` `<` `>` `<=` `>=` | 10, 11 | |
| `\|` | 12, 13 | bit-or / union |
| `&` | 14, 15 | |
| `..` `..=` | 16, 17 | |
| `+` `-` | 18, 19 | |
| `*` `/` `//` `%` | 20, 21 | |
| `@` | 22, 23 | matmul |
| `**` | 25, 24 | right-associative |
| prefix `-` | 24 | so `-a**b` is `-(a**b)` |
| prefix `not` | 8 | tighter than `and`, looser than comparisons |
| `await` | 27 | hard-coded in `parse_prefix_or_atom` |
| `.` `[` `(` | 30 | postfix |

Atoms (`parse_prefix_or_atom`): literals, `FStringStart` →
`parse_fstring`, identifiers, `self`, `(` → `parse_paren_or_tuple`,
`[` → `parse_list_or_comp`, `{` → `parse_dict`, `|` → `parse_lambda`
(`|x, y| body`; lambda parameter types use `types::parse_primary_type` so
`|` is not eaten as a union), `if` → `parse_if_expr`, `await`.
`can_start_expr` is the "could an expression begin here" predicate the range
branch and `return`/`yield` use; keep it in sync with the atom list.

### Types and patterns (`types.rs`, `pattern.rs`)

`types::parse_type` parses `parse_primary_type` then folds `A | B | C` into
one `Union`. `parse_primary_type_nested` handles names, generics, and the
tensor family by name (`Tensor<[B, S], f32, cuda>` via `parse_tensor_type`,
`parse_dim_list`, `parse_dim_expr`, `parse_device_expr`), `&T` borrows,
`[T; N]` fixed arrays, and `(A, B) -> C | E` via `parse_function_or_tuple_type`.
`parse_type_no_borrow` and `reject_borrow_in_type` implement the "no `&T`
in return position / nested" rule with a diagnostic rather than a grammar
restriction. `pattern::parse_pattern` parses `parse_primary_pattern` and
folds `a | b` into `Or`; `Typed` and `Guarded` come from the callers.

### Nesting limit

`MAX_NESTING = 128` (`parser.rs`). `enter_nesting(what)` / `leave_nesting()`
are paired at five sites — `parse_stmt`, `parse_expr_bp`, `parse_block`,
`parse_primary_type`, `parse_primary_pattern` — so every recursion into the
grammar is counted once. On overflow the first call records the error
("`what` nested more than 128 levels deep"), moves `pos` onto the input's
`Eof` token (not past it), and every subsequent `enter_nesting` returns
`false` so callers return an error node without recursing. The long comment
on `MAX_NESTING` records the calibration: the limit fits in `cargo test`'s
2 MB debug threads, and the deepest real file in the tree nests 13. Note
the bound is on *nesting*, not tree depth: `a + b + c + …` builds a deep
left-leaning tree in a loop, and passes that recurse over the tree (and the
tree's own drop glue) are not bounded — hence `-max_len=4096` on the
fuzzers.

### Recovery points

- `Parser::expect` / `expect_ident`: skip one token.
- `Parser::expect_end_of_stmt` → `Parser::synchronize`: skip tokens until a
  `Newline`, `Dedent`, `Eof`, **or a statement keyword** (`let`, `fn`,
  `model`, `if`, `for`, `return`, `train`, `agent`, …). Stopping *before*
  the keyword is what lets the next `parse_stmt` pick it up; the list is in
  `synchronize` and must be extended when a statement keyword is added.
- `Parser::skip_to_next_line`: for body loops that do not parse statements
  (tokenizer sections, dataset entries, key-value blocks). `synchronize`
  parked in front of a keyword such loops have no arm for, and the loop
  pushed the same diagnostic until the process ran out of memory (found by
  the fuzzer); this helper always consumes through the newline and the
  indented suite that follows, so a loop calling it every iteration
  terminates. `crates/nsl-parser/tests/body_recovery.rs` pins the shape.
- Two known gaps are pinned rather than fixed:
  `recovery.rs::unclosed_paren_and_trailing_operator_swallow_the_next_statement_today`
  documents that an unclosed `(` or a trailing operator consumes the
  following line because the lexer suppressed its `Newline` (brackets) or
  the expression parser kept going.

### Decorators and config namespaces

Statement-level decorators: `parse_decorated_stmt` collects one or more
`parse_decorator` results (each `@` + dotted ident path + optional
`(args)` via `parse_args`, so `@a.b(x=1)` gives `name = [a, b]`), skips
newlines, parses the next statement with `parse_stmt`, and wraps it in
`StmtKind::Decorated`. `@a` then `@b` then `fn` therefore gives one
`Decorated` with two decorators; `pub @a fn` goes through
`parse_visibility_prefixed` and yields the same shape.

Member-level decorators are parsed by **inline copies** of the same loop in
`decl.rs::parse_model_def_stmt` (into `ModelMember`), `agent.rs::parse_agent_def_stmt`
(into `AgentMember`), and `block.rs::parse_kernel_def_stmt` (into
`KernelDef.decorators`). `block.rs::parse_datatype_def_stmt` is different:
`@pack`, `@unpack`, `@backward @pack`, `@pack_ptx`, `@unpack_ptx` and
`@arithmetic_ptx` are *structural* there and become `DatatypeMethod` / `DatatypePtxBlock`
directly, not `Decorator` nodes.

The parser treats every decorator name as opaque. The closed namespace is
`nsl_semantic::decorator_registry` (`find`, `unimplemented_refusal`,
`suggest`), applied over `collect_decorators` in `nsl_semantic::analyze` /
`analyze_with_imports`; an unknown name is an error with a did-you-mean, demoted to a
warning only by `NSL_ALLOW_UNKNOWN_DECORATORS=1`. Likewise
`train(...)`/`distill(...)` kwargs are any `Arg` list here and are
validated by `nsl_semantic::train_config`; `serve` sub-block keys are any
identifier. The rule is: the parser fixes *shape*, semantic fixes
*vocabulary*, so adding a decorator or a train key usually touches no
frontend code at all (see `docs/wiki/Adding-a-Language-Feature.md` for the
`@export` walkthrough).

---

## Invariants

The frontend relies on these; several are machine-checked by
`crates/nsl-parser/tests/common/frontend_check.rs`, which both the fuzz
targets and `crates/nsl-parser/tests/fuzz_seeds.rs` run.

- **Spans are byte ranges into one file.** `start <= end`, `end <= source
  .len()`, both on `char` boundaries, and `file_id` is the file the source
  was lexed as. Checked for every token and every diagnostic label.
- **`Span::merge` panics across files**, so nothing in the parser may merge
  a real span with `Span::DUMMY` (file 0). Hence: `advance` never passes
  `Eof`, `enter_nesting` moves onto the last real token, and real inputs
  are never lexed as `FileId(0)` in the harnesses.
- **Every `Stmt`, `Expr`, `TypeExpr`, `Pattern` has a span and a `NodeId`;
  every `Block`, `Module`, declaration struct and `Decorator` has a span.**
  A synthesized node gets the span of the token it stands in for; only an
  empty `Module` gets `Span::dummy()`.
- **The token stream ends in exactly one `Eof`, preceded by a `Newline` and
  by a `Dedent` per open block.** Token start positions are non-decreasing.
  The parser assumes all three.
- **`Indent`/`Dedent` are balanced by construction** and never appear
  inside brackets; a mismatched dedent emits neither and reports.
- **Both stages return on every input** — no panic, no abort, no `Result`.
  The lexer always makes progress (every arm of `scan_token` advances);
  every parser loop that reports also advances (`expect`, `synchronize`,
  `skip_to_next_line`), and recursion is bounded by `MAX_NESTING`.
  `fuzz/fuzz_targets/{lex,parse}.rs` check this on arbitrary bytes;
  `crates/nsl-parser/tests/nesting_limit.rs` checks the bound at and one past the limit on a
  2 MB thread.
- **Every diagnostic has a label the `SourceMap` can render.** A label with
  `Span::DUMMY` or an out-of-map span prints nothing after the header;
  `contains_span` is the guard for spans of uncertain origin.
- **`NodeId`s are unique per process, not stable across runs.** Nothing may
  key persisted data on them; the goldens erase them.
- **The parser never interprets decorator names, train-config keys or serve
  section keys.** Vocabulary lives in `nsl-semantic`.
- **`nsl fmt` preserves the token stream's meaning.** For every tracked
  `.nsl` that parses clean, formatting succeeds, still parses clean, yields
  the same AST modulo spans and ids, and is idempotent
  (`crates/nsl-cli/tests/fmt_roundtrip.rs`). The formatter
  (`crates/nsl-cli/src/formatter.rs`) works on the *token stream*, not the
  AST, so it needs the lexer but not the parser.

---

## Tests and gates

All of these run under CI's `cargo test --workspace --no-fail-fast -- --skip e2e_`
(`.github/workflows/ci.yml`) except the cargo-fuzz targets, which need
nightly.

**Lexer goldens** — `crates/nsl-lexer/tests/goldens.rs` over
`crates/nsl-lexer/tests/lex/*.nsl` (six fixtures: `indentation`,
`keywords`, `numbers`, `operators`, `strings`, `err_lexer`), each with a
`<stem>.tokens.snap` (insta) listing diagnostics then one `line:col-line:col
Kind` per token, identifiers printed by name so interning order is
irrelevant. A fixture whose stem starts `err_` must produce at least one
error; every other must produce none; a `.snap` without a fixture fails.
Run `cargo test -p nsl-lexer --test goldens`. To add one: drop the `.nsl` in
`crates/nsl-lexer/tests/lex/`, run once, review the `.snap.new`, rename or `cargo insta
review`.

**Parser goldens** — `crates/nsl-parser/tests/goldens.rs` over
`crates/nsl-parser/tests/parse/*.nsl` (21 fixtures, six of them `err_*`)
with `<stem>.ast.snap`: lexer then parser diagnostics, then the `Debug`
module with `NodeId`s dropped, symbols resolved, and spans as
`line:col-line:col`. Same `err_` contract and workflow;
`cargo test -p nsl-parser --test goldens`.

**Error-recovery pins** (`crates/nsl-parser/tests/`):
`recovery.rs` (three independent errors are all reported and their
neighbours reach the AST; lexer error tokens do not stop the parser; the two
known swallow cases), `body_recovery.rs` (one diagnostic per bad line in
tokenizer/dataset/key-value bodies, under a 10 s deadline so a non-progress
regression fails instead of OOM-killing the test process),
`nesting_limit.rs` (every nesting shape parses at `MAX_NESTING`, one past it
is refused with one diagnostic, the cascade is dropped), `agent_parse.rs`.
Unit tests inside `src/block.rs` (serve/tokenizer/dataset) and
`src/types.rs` (borrow types). `cargo test -p nsl-parser`.

**Fuzzing** — `fuzz/` is its own cargo workspace (so the stable build never
sees `libfuzzer-sys`) with two targets, `lex` and `parse`, both one-liners
over `nsl_fuzz::check`, which is
`crates/nsl-parser/tests/common/frontend_check.rs` included by `#[path]`.
Commands are in `fuzz/Cargo.toml`:

```
cargo install cargo-fuzz
cargo +nightly fuzz run parse -- -max_len=4096 -timeout=10 -rss_limit_mb=4096
cargo +nightly fuzz run lex   -- -max_len=4096 -timeout=10 -rss_limit_mb=4096
```

The committed seed corpus is `fuzz/corpus/lex/*.nsl` (8) and
`fuzz/corpus/parse/*.nsl` (15, including the reproducers for the crashes
above: `deep_nesting.nsl`, `unclosed_call.nsl`, `tokenizer_body_keyword.nsl`).
`crates/nsl-parser/tests/fuzz_seeds.rs` runs the same invariants over those
seeds on stable, so a regression on a known crash is caught in CI; a new
crash becomes a regression test by copying its reproducer into the corpus
under a `.nsl` name. Only `.nsl` files are tracked — the hash-named inputs
libFuzzer adds are ignored.

**`nsl fmt` round-trip** — `crates/nsl-cli/tests/fmt_roundtrip.rs`
(`cargo test -p nsl-cli --test fmt_roundtrip`): for every `.nsl` under
`CORPUS_DIRS` that parses clean, and for three de-formatted spellings of it
(widened gaps, squeezed gaps, requoted strings), `format_source` must
succeed, re-parse clean, preserve the AST modulo spans/ids, and be a fixed
point. `EXPECTED_SKIPS` pins the files the frontend rejects with a reason,
and `CHANGED_FLOOR` keeps the property from going vacuous. The CLI is
`nsl fmt [--check] <files>` (`crates/nsl-cli/src/commands/fmt.rs`).

**By hand** — `nsl check <file> --dump-tokens` / `--dump-ast`
(`crates/nsl-cli/src/commands/check.rs`). **Benches** — `cargo bench -p
nsl-lexer` (bytes/s), `cargo bench -p nsl-parser` (tokens/s); baselines via
`scripts/bench.sh`; CI clippy-checks the bench sources.

---

## Where to add a new X

### A new token kind (operator or keyword)

1. Add the variant to `TokenKind` in `crates/nsl-lexer/src/token.rs`, in
   the matching banner group, and a `Display` arm so parser messages print
   it as source text rather than its `Debug` name.
2. Keyword: add the arm to `lookup_keyword` in
   `crates/nsl-lexer/src/keywords.rs`. Remember it becomes reserved
   everywhere; grep the corpus (`stdlib/`, `examples/`, `models/`) for the
   word used as a name first, and if it may appear in import paths add it to
   `Parser::expect_ident_or_keyword`.
   Operator: add the arm to `scan_token` in `crates/nsl-lexer/src/lexer.rs`,
   using `cursor.eat` for the longer match *before* the shorter one (see
   `-` → `->`/`-=`/`-`). A token that opens or closes a bracket must call
   `self.indent.open_bracket()` / `close_bracket()`.
3. If it can begin an expression, add it to `can_start_expr` in
   `crates/nsl-parser/src/expr.rs`; if it is a statement keyword, add it to
   `Parser::synchronize` in `parser.rs` so recovery stops before it.
4. Add a line to `crates/nsl-lexer/tests/lex/operators.nsl` or
   `keywords.nsl` and regenerate the `.tokens.snap`.
5. The formatter classifies tokens for spacing in
   `crates/nsl-cli/src/formatter.rs` (`ends_operand`, `opens_bracket`,
   `closes_or_punctuates`); run `cargo test -p nsl-cli --test fmt_roundtrip`
   and add the token there if a spelling fails to round-trip.

### A new statement

1. AST: add the payload struct to `crates/nsl-ast/src/block.rs` (ML block)
   or `decl.rs` (declaration), with a `span`, and the `StmtKind` variant in
   `stmt.rs`. Add the arm to `walk_stmt` in `visitor.rs`; if it carries
   bodies or decorator vectors, add the arm to `Collector::collect_stmt` in
   `decorator_walk.rs` and, if it can be decorated, to `host_of_inner` (and
   a `DecoratorHost` variant if the position is new).
2. Parser: write `parse_<x>_stmt(p: &mut Parser) -> Stmt` in `block.rs` or
   `decl.rs` following the pattern (record `start`, consume keyword, pieces,
   `p.parse_block()` for a suite or `p.expect_end_of_stmt()` for a line,
   `span: start.merge(...)`, `id: p.next_node_id()`). Dispatch it from
   `parse_stmt_nested` in `stmt.rs`. Do not add your own nesting counter;
   `parse_stmt` and `parse_block` already count. If the body is not a
   statement suite (key-value lines), every loop iteration that reports must
   call `skip_to_next_line`.
3. Keyword: steps 1–3 of the token recipe, including `synchronize`.
4. Tests: a fixture in `crates/nsl-parser/tests/parse/` (a clean one and,
   for the error shapes, an `err_*` one) and their `.ast.snap`; a seed in
   `fuzz/corpus/parse/`; if the body is a non-statement loop, a case in
   `crates/nsl-parser/tests/body_recovery.rs`.
5. Downstream, `nsl-semantic` and `nsl-codegen` `match` on `StmtKind`; the
   new variant will fail their exhaustiveness checks, which is the intended
   to-do list.

### A new expression form or operator

1. Binary operator: token (recipe above), a `BinOp` variant in
   `crates/nsl-ast/src/operator.rs`, an `infix_binding_power` row and a
   `token_to_binop` arm in `crates/nsl-parser/src/pratt.rs`. The Pratt loop
   in `parse_expr_bp_nested` then handles it; pick the `(l, r)` pair from
   the table above (`l < r` left-assoc, `l > r` right-assoc) and leave gaps
   between rows. A form that is not a plain `BinaryOp` (like `|>` and `..`)
   gets its own branch in that loop *before* the generic infix branch.
2. Prefix operator: `prefix_binding_power` in `pratt.rs` and an arm in
   `parse_prefix_or_atom`. Postfix: `postfix_binding_power` and an arm in
   `parse_postfix`.
3. New atom (a literal or bracketed form): an arm in
   `parse_prefix_or_atom`, an entry in `can_start_expr`, an `ExprKind`
   variant in `crates/nsl-ast/src/expr.rs`, and an arm in `walk_expr` in
   `visitor.rs`. Every recursive operand must go through `parse_expr_bp`
   (not `parse_prefix_or_atom`) so it is nesting-counted.
4. Tests: extend `crates/nsl-parser/tests/parse/operators.nsl` (or
   `calls_and_access.nsl`) with cases that pin the *tree shape* against
   neighbouring precedence rows — the golden shows the nesting, which is the
   only executable statement of precedence. If a deep chain of the new form
   is possible, add a shape to `shapes()` in `crates/nsl-parser/tests/nesting_limit.rs`.

### A new decorator

1. Usually: no frontend change. `@name`, `@a.b.c`, and `@name(k=v)` already
   parse into `Decorator` at every carrier position. Register the name in
   `crates/nsl-semantic/src/decorator_registry.rs` (or it is refused as
   unknown) and implement its meaning in semantic/codegen.
   `docs/wiki/Adding-a-Language-Feature.md` walks through `@export`.
2. If the decorator must be structural at parse time (it changes what the
   following construct *is*, as `@pack` does inside `datatype`), handle it in
   that host's parser in `block.rs` and give it a typed AST node rather than
   a `Decorator`.
3. If it is allowed on a construct that has no decorator carrier today, add
   a `Vec<Decorator>` to that node, parse it with the same loop shape as
   `decl.rs::parse_model_def_stmt`, add a `DecoratorHost` variant and a
   `Collector` arm in `decorator_walk.rs` so the namespace gate sees it, and
   note in `visitor.rs` that the generic walk skips it.
4. Tests: a parser golden only if syntax changed; otherwise semantic tests
   for the registry and the host-position contract.

### A new diagnostic

1. Build it where the fact is known:
   `Diagnostic::error("…").with_label(span, "…")` (+ `with_note` for the
   fix), and push onto `self.diagnostics` (lexer) or `p.diagnostics`
   (parser). Use the "expected X, found Y" phrasing with `TokenKind`'s
   `Display` for consistency with `expect`.
2. The label span must be a real span from a token or node in the current
   file — never `Span::DUMMY`, never a span merged from another file. In the
   parser, `p.current_span()` / `p.prev_span()` / a node's `.span` are the
   right sources.
3. After reporting, make sure the caller advances (`expect` does; a custom
   check must `advance`, `synchronize`, or `skip_to_next_line`), or the
   enclosing loop will report forever. `body_recovery.rs` is the pattern for
   proving that.
4. Warnings and infos flow through the same vector; the loader only fails on
   `Level::Error`, and a warning-only fixture is a plain golden (no `err_`
   prefix).
5. Pin it in an `err_*` fixture in `crates/nsl-lexer/tests/lex/` or `crates/nsl-parser/tests/parse/`; the
   golden records the message, the label position, and that the surrounding
   statements survive. If the shape came from a fuzz crash, also copy the
   reproducer into `fuzz/corpus/<target>/`.
