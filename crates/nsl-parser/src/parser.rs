use nsl_ast::Symbol;
use nsl_ast::{NodeId, Span};
use nsl_errors::Diagnostic;
use nsl_lexer::{Token, TokenKind};
use nsl_lexer::Interner;

/// Sentinel EOF token used when the parser has reached the end of input.
const EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span::DUMMY,
};

/// The core parser state.
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    pub diagnostics: Vec<Diagnostic>,
    pub interner: &'a mut Interner,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token], interner: &'a mut Interner) -> Self {
        Self {
            tokens,
            pos: 0,
            diagnostics: Vec::new(),
            interner,
        }
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

    pub fn advance(&mut self) -> &Token {
        if self.tokens.is_empty() {
            return &EOF_TOKEN;
        }
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() {
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
        if !self.at(&TokenKind::Newline)
            && !self.at(&TokenKind::Eof)
            && !self.at(&TokenKind::Dedent)
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

    // === Block parsing ===

    pub fn parse_block(&mut self) -> nsl_ast::stmt::Block {
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
                | TokenKind::Grad => {
                    return;
                }
                _ => {
                    self.advance();
                }
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
