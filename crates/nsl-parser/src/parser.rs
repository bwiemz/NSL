use nsl_ast::Symbol;
use nsl_ast::{NodeId, Span};
use nsl_errors::Diagnostic;
use nsl_lexer::{Token, TokenKind};
use nsl_lexer::Interner;

/// Sentinel for an empty token stream, or one that does not end in `Eof`.
///
/// Its span is `Span::DUMMY`, which is in `FileId(0)` — merging it with a
/// span from any other file panics — so the parser must not otherwise
/// reach it: the lexer ends every stream with an `Eof` token, and
/// `advance` never steps past it.
const EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span::DUMMY,
};

/// The deepest nesting the parser accepts: statements, expressions,
/// blocks, types and patterns together, the outermost counting as level
/// one, so a nested `if` costs two levels (the statement and its block)
/// and a parenthesis one. One level more is refused with a diagnostic and
/// parsing stops.
///
/// The parser is recursive descent, so every level costs a few stack
/// frames and nothing else bounds it; at two thousand levels — found by
/// the parse fuzz target — it segfaults. The budget is set by the tightest
/// stack the parser runs on, `cargo test`'s 2 MB threads with the debug
/// build (`nsl` itself runs on a 16 MB thread): at this limit the
/// hungriest shape, nested tuple patterns, uses about 1 MB of that and
/// the others 200–700 KB; the release build needs at most 210 KB for any
/// shape. `tests/nesting_limit.rs` parses every nesting shape at the
/// limit on such a thread, so the margin is checked on every platform CI
/// runs, not assumed. When the limit was set, the deepest of the 418
/// tracked `.nsl` files (`stdlib/nsl/optim/muon.nsl`) nested 13 levels.
///
/// This bounds the parser's recursion, which is also the depth of any
/// *nested* construct in the tree. A flat chain — `a + b + c + …`, `x.a.b.c`,
/// `x |> f |> g` — parses in a loop and builds a tree as deep as the chain
/// is long without ever nesting; later passes that recurse over the tree
/// have no such bound.
pub const MAX_NESTING: u32 = 128;

/// The core parser state.
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    pub diagnostics: Vec<Diagnostic>,
    pub interner: &'a mut Interner,
    /// Current nesting level; see `enter_nesting`.
    depth: u32,
    /// Index in `diagnostics` of the nesting-limit error, once it has been
    /// reported.
    nesting_overflow: Option<usize>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token], interner: &'a mut Interner) -> Self {
        Self {
            tokens,
            pos: 0,
            diagnostics: Vec::new(),
            interner,
            depth: 0,
            nesting_overflow: None,
        }
    }

    /// The diagnostics to report, once parsing is finished.
    ///
    /// After the nesting limit was hit, the recursion unwinds through every
    /// open construct, and each would report a bogus "expected `)`, found
    /// end of file"; those are dropped, keeping everything reported before
    /// the limit and the limit error itself. A check that runs on a
    /// construct *after* its inner parse returns (the `&&T` borrow errors
    /// in `types.rs`) goes with the cascade when that inner parse is the
    /// one that hit the limit; it fires again once the nesting is reduced.
    pub fn finish(self) -> Vec<Diagnostic> {
        let mut diagnostics = self.diagnostics;
        if let Some(at) = self.nesting_overflow {
            diagnostics.truncate(at + 1);
        }
        diagnostics
    }

    // === Token inspection ===

    pub fn peek(&self) -> &TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    pub fn peek_token(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&EOF_TOKEN)
    }

    pub fn peek_at(&self, offset: usize) -> &TokenKind {
        self.tokens
            .get(self.pos + offset)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    pub fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(kind)
    }

    pub fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|k| self.at(k))
    }

    pub fn current_span(&self) -> Span {
        self.peek_token().span
    }

    pub fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            Span::dummy()
        }
    }

    // === Token consumption ===

    /// Consume and return the current token.
    ///
    /// Never steps past an `Eof` token: consuming it again returns it
    /// again. Stepping past it would leave `peek_token` on the sentinel
    /// `EOF_TOKEN`, whose `Span::DUMMY` is in `FileId(0)`, and the next
    /// `Span::merge` with a span of any other file — every imported module
    /// — would panic. Unclosed brackets reached that state on
    /// `let x = f(` followed by a newline (found by the parse fuzz
    /// target).
    pub fn advance(&mut self) -> &Token {
        let Some(tok) = self.tokens.get(self.pos) else {
            return &EOF_TOKEN;
        };
        if !matches!(tok.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        tok
    }

    pub fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn expect(&mut self, kind: &TokenKind) -> Span {
        if self.at(kind) {
            let tok = self.advance();
            tok.span
        } else {
            let span = self.current_span();
            self.diagnostics.push(
                Diagnostic::error(format!("expected {kind}, found {}", self.peek()))
                    .with_label(span, format!("expected {kind}")),
            );
            // Advance past the unexpected token to prevent infinite loops
            if !self.at(&TokenKind::Eof) {
                self.advance();
            }
            span
        }
    }

    pub fn expect_ident(&mut self) -> (Symbol, Span) {
        if let TokenKind::Ident(sym) = self.peek().clone() {
            let span = self.advance().span;
            (sym.into(), span)
        } else {
            let span = self.current_span();
            self.diagnostics.push(
                Diagnostic::error(format!("expected identifier, found {}", self.peek()))
                    .with_label(span, "expected identifier"),
            );
            // Advance past the unexpected token to prevent infinite loops
            if !self.at(&TokenKind::Eof) {
                self.advance();
            }
            let sym = self.interner.get_or_intern("<error>");
            (sym.into(), span)
        }
    }

    /// Expect an identifier or a keyword (for use in import paths where keywords are valid segments).
    pub fn expect_ident_or_keyword(&mut self) -> (Symbol, Span) {
        if let TokenKind::Ident(sym) = self.peek().clone() {
            let span = self.advance().span;
            return (sym.into(), span);
        }

        // Allow keywords as path segments in imports (e.g., nsl.quant, nsl.export)
        let keyword_name = match self.peek() {
            TokenKind::Model => Some("model"),
            TokenKind::Quant => Some("quant"),
            TokenKind::Train => Some("train"),
            TokenKind::Distill => Some("distill"),
            TokenKind::Grad => Some("grad"),
            TokenKind::Kernel => Some("kernel"),
            TokenKind::Tokenizer => Some("tokenizer"),
            TokenKind::Dataset => Some("dataset"),
            TokenKind::Import => Some("import"),
            TokenKind::From => Some("from"),
            TokenKind::Match => Some("match"),
            TokenKind::Struct => Some("struct"),
            TokenKind::Enum => Some("enum"),
            TokenKind::Trait => Some("trait"),
            TokenKind::SelfKw => Some("self"),
            TokenKind::True => Some("true"),
            TokenKind::False => Some("false"),
            TokenKind::None => Some("none"),
            _ => Option::None,
        };

        if let Some(name) = keyword_name {
            let span = self.advance().span;
            return (self.intern(name), span);
        }

        let span = self.current_span();
        self.diagnostics.push(
            Diagnostic::error(format!("expected identifier, found {}", self.peek()))
                .with_label(span, "expected identifier"),
        );
        // Advance past the unexpected token to prevent infinite loops
        if !self.at(&TokenKind::Eof) {
            self.advance();
        }
        let sym = self.interner.get_or_intern("<error>");
        (sym.into(), span)
    }

    /// Consume newlines and doc comments (skip over them).
    pub fn skip_newlines(&mut self) {
        while let TokenKind::Newline | TokenKind::DocComment(_) = self.peek() {
            self.advance();
        }
    }

    /// Expect a newline or EOF or DEDENT (end of statement).
    pub fn expect_end_of_stmt(&mut self) {
        let previous_was_dedent = self.pos > 0
            && matches!(self.tokens[self.pos - 1].kind, TokenKind::Dedent);

        if !self.at(&TokenKind::Newline)
            && !self.at(&TokenKind::Eof)
            && !self.at(&TokenKind::Dedent)
            && !previous_was_dedent
        {
            let span = self.current_span();
            self.diagnostics.push(
                Diagnostic::error(format!(
                    "expected newline or end of statement, found {}",
                    self.peek()
                ))
                .with_label(span, "here"),
            );
            // Try to recover by skipping to next newline
            self.synchronize();
        }
        self.eat(&TokenKind::Newline);
    }

    // === Nesting ===

    /// Enter one level of nesting: a statement, expression, block, type or
    /// pattern.
    ///
    /// Returns `false` when that would exceed `MAX_NESTING`; the caller then
    /// returns an error node without recursing. The first time this
    /// happens the overflow is reported and the parser is moved onto the
    /// input's `Eof` token, so the open constructs unwind against `Eof` —
    /// a state every truncated input already puts the parser in — and
    /// nothing after the limit is parsed. On `true`, pair with
    /// `leave_nesting`.
    pub fn enter_nesting(&mut self, what: &str) -> bool {
        if self.depth >= MAX_NESTING {
            if self.nesting_overflow.is_none() {
                let span = self.current_span();
                self.nesting_overflow = Some(self.diagnostics.len());
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "{what} nested more than {MAX_NESTING} levels deep"
                    ))
                    .with_label(span, "nesting limit reached here; the rest of the file was not parsed"),
                );
                // Onto the trailing `Eof` token, not past it: past it is
                // the `EOF_TOKEN` sentinel, whose span is in `FileId(0)`.
                self.pos = self.tokens.len().saturating_sub(1);
            }
            return false;
        }
        self.depth += 1;
        true
    }

    pub fn leave_nesting(&mut self) {
        self.depth -= 1;
    }

    // === Block parsing ===

    pub fn parse_block(&mut self) -> nsl_ast::stmt::Block {
        if !self.enter_nesting("block") {
            let span = self.current_span();
            return nsl_ast::stmt::Block {
                stmts: Vec::new(),
                span,
            };
        }
        let block = self.parse_block_nested();
        self.leave_nesting();
        block
    }

    fn parse_block_nested(&mut self) -> nsl_ast::stmt::Block {
        self.skip_newlines();
        let start = self.current_span();
        self.expect(&TokenKind::Indent);
        self.skip_newlines();

        let mut stmts = Vec::new();
        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();
            if self.at(&TokenKind::Dedent) || self.at(&TokenKind::Eof) {
                break;
            }
            stmts.push(super::stmt::parse_stmt(self));
            self.skip_newlines();
        }

        let end = self.current_span();
        self.eat(&TokenKind::Dedent);

        nsl_ast::stmt::Block {
            stmts,
            span: start.merge(end),
        }
    }

    // === Error recovery ===

    pub fn synchronize(&mut self) {
        loop {
            match self.peek() {
                TokenKind::Newline | TokenKind::Dedent | TokenKind::Eof => {
                    return;
                }
                TokenKind::Let
                | TokenKind::Const
                | TokenKind::Fn
                | TokenKind::Model
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::If
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Match
                | TokenKind::Return
                | TokenKind::Import
                | TokenKind::From
                | TokenKind::Train
                | TokenKind::Grad
                | TokenKind::Agent => {
                    return;
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Recovery for a body loop that does not parse statements (tokenizer,
    /// dataset, key-value blocks): drop the rest of the current line, its
    /// newline, and the indented suite that follows it, if any — `if b:`
    /// and its block are one bad line to such a loop, not one bad line
    /// and then an `Indent` it has no arm for.
    ///
    /// `synchronize` stops *before* a statement keyword so that a statement
    /// parser can pick it up. A loop that only accepts `key = value` lines
    /// never does, so calling `synchronize` on a line such as `if b` left
    /// the parser on `if`, and the loop pushed the same diagnostic forever
    /// (found by the parse fuzz target). This always consumes up to a
    /// `Newline`, `Dedent` or `Eof` and never a `Dedent` of the enclosing
    /// body, so a loop that calls it on every iteration terminates and
    /// still sees the end of its body.
    pub fn skip_to_next_line(&mut self) {
        while !self.at_any(&[TokenKind::Newline, TokenKind::Dedent, TokenKind::Eof]) {
            self.advance();
        }
        if !self.eat(&TokenKind::Newline) {
            return;
        }
        self.skip_newlines();
        if !self.at(&TokenKind::Indent) {
            return;
        }
        // The suite, tokens only: it is not parsed, so it reports nothing.
        let mut depth = 0u32;
        while !self.at(&TokenKind::Eof) {
            match self.peek() {
                TokenKind::Indent => depth += 1,
                TokenKind::Dedent => depth -= 1,
                _ => {}
            }
            self.advance();
            if depth == 0 {
                break;
            }
        }
    }

    // === Helpers ===

    pub fn next_node_id(&self) -> NodeId {
        NodeId::next()
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        self.interner.get_or_intern(s).into()
    }

    /// Resolve a symbol back to its string representation.
    pub fn resolve(&self, sym: Symbol) -> Option<&str> {
        self.interner.resolve(sym.0)
    }
}
