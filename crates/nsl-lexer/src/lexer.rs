use crate::cursor::Cursor;
use crate::indent::IndentTracker;
use crate::keywords::lookup_keyword;
use crate::numbers::lex_number;
use crate::strings::lex_string;
use crate::token::{Token, TokenKind};
use nsl_errors::{BytePos, Diagnostic, FileId, Span};
use string_interner::StringInterner;

type Interner = StringInterner<string_interner::backend::BucketBackend<string_interner::DefaultSymbol>>;

/// The main lexer. Tokenizes NSL source code into a stream of tokens.
pub struct Lexer<'a> {
    cursor: Cursor<'a>,
    indent: IndentTracker,
    interner: &'a mut Interner,
    diagnostics: Vec<Diagnostic>,
    tokens: Vec<Token>,
    /// Whether we are at the start of a logical line (need to process indentation).
    at_line_start: bool,
    /// Whether the previous line ended with a line continuation `\`.
    line_continuation: bool,
    /// F-string nesting state: each entry is the brace depth within an f-string expression.
    /// Stack of (brace_depth, quote_char) for nested f-string expressions.
    fstring_stack: Vec<(u32, char)>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str, file_id: FileId, interner: &'a mut Interner) -> Self {
        Self {
            cursor: Cursor::new(source, file_id),
            indent: IndentTracker::new(),
            interner,
            diagnostics: Vec::new(),
            tokens: Vec::new(),
            at_line_start: true,
            line_continuation: false,
            fstring_stack: Vec::new(),
        }
    }

    pub fn tokenize(mut self) -> (Vec<Token>, Vec<Diagnostic>) {
        while !self.cursor.is_eof() {
            self.scan_token();
        }

        // Emit final newline if last line didn't end with one (or if source was empty)
        let needs_newline = match self.tokens.last() {
            None => true,
            Some(t) => t.kind != TokenKind::Newline,
        };
        if needs_newline {
            self.tokens.push(Token {
                kind: TokenKind::Newline,
                span: self.eof_span(),
            });
        }

        // Emit remaining DEDENTs
        let dedents = self.indent.finalize(self.eof_span());
        self.tokens.extend(dedents);

        // Emit EOF
        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: self.eof_span(),
        });

        (self.tokens, self.diagnostics)
    }

    fn eof_span(&self) -> Span {
        let pos = self.cursor.pos();
        Span::new(self.cursor.file_id(), pos, pos)
    }

    fn scan_token(&mut self) {
        // Handle f-string expression mode
        if !self.fstring_stack.is_empty() {
            if self.cursor.peek() == Some('}') {
                let (depth, _quote) = self.fstring_stack.last_mut().unwrap();
                if *depth == 0 {
                    // End of f-string expression — resume text with the correct quote char
                    let start = self.cursor.pos();
                    self.cursor.advance();
                    self.push_token(TokenKind::FStringExprEnd, start);
                    let (_d, quote) = self.fstring_stack.pop().unwrap();
                    // Continue lexing the rest of the f-string text with the original quote
                    self.lex_fstring_text_with_quote(quote);
                    return;
                } else {
                    *depth -= 1;
                    // Fall through to normal token scanning for the `}`
                }
            } else if self.cursor.peek() == Some('{') {
                // Nested brace in f-string expression
                if let Some((depth, _)) = self.fstring_stack.last_mut() {
                    *depth += 1;
                }
                // Fall through to normal `{` handling
            }
        }

        // Process indentation at line start
        if self.at_line_start {
            self.at_line_start = false;
            self.process_line_indentation();
            if self.cursor.is_eof() {
                return;
            }
        }

        // Skip spaces (not at line start — those were handled by indentation)
        while self.cursor.peek() == Some(' ') {
            self.cursor.advance();
        }

        if self.cursor.is_eof() {
            return;
        }

        let start = self.cursor.pos();
        let ch = self.cursor.peek().unwrap();

        match ch {
            // Newlines
            '\n' => {
                self.cursor.advance();
                if !self.line_continuation && !self.indent.in_brackets() {
                    self.push_token(TokenKind::Newline, start);
                }
                // Note: line_continuation is cleared in process_line_indentation,
                // not here, so it survives until the indentation check runs.
                self.at_line_start = true;
            }
            '\r' => {
                self.cursor.advance();
                self.cursor.eat('\n'); // consume optional \r\n
                if !self.line_continuation && !self.indent.in_brackets() {
                    self.push_token(TokenKind::Newline, start);
                }
                self.at_line_start = true;
            }

            // Tab (error in NSL)
            '\t' => {
                self.cursor.advance();
                self.diagnostics.push(
                    Diagnostic::error("tabs are not allowed; use 4 spaces for indentation")
                        .with_label(self.cursor.span_from(start), "tab character here"),
                );
            }

            // Comments
            '#' => {
                if self.cursor.peek_at(1) == Some('#') {
                    // Doc comment
                    self.cursor.advance(); // first #
                    self.cursor.advance(); // second #
                    // Skip optional space after ##
                    if self.cursor.peek() == Some(' ') {
                        self.cursor.advance();
                    }
                    let text = self.cursor.eat_while(|c| c != '\n');
                    self.push_token(TokenKind::DocComment(text.to_string()), start);
                } else if self.cursor.peek_at(1) == Some('[') {
                    // Attribute: #[cfg(...)]
                    // For now, skip the whole line as we don't handle attributes in the lexer
                    self.cursor.eat_while(|c| c != '\n');
                } else {
                    // Line comment — skip to end of line
                    self.cursor.eat_while(|c| c != '\n');
                }
            }

            // Line continuation
            '\\' => {
                self.cursor.advance();
                if self.cursor.peek() == Some('\n') || self.cursor.peek() == Some('\r') {
                    self.line_continuation = true;
                } else {
                    self.push_token(TokenKind::Backslash, start);
                }
            }

            // String literals
            '"' | '\'' => {
                let kind = lex_string(&mut self.cursor, ch, start, &mut self.diagnostics);
                self.push_token(kind, start);
            }

            // Numbers
            '0'..='9' => {
                let kind = lex_number(&mut self.cursor, start, &mut self.diagnostics);
                self.push_token(kind, start);
            }

            // Identifiers, keywords, and f-strings
            'a'..='z' | 'A'..='Z' | '_' => {
                // Check for f-string
                if ch == 'f' && matches!(self.cursor.peek_at(1), Some('"') | Some('\'')) {
                    self.cursor.advance(); // consume 'f'
                    let quote = self.cursor.peek().unwrap();
                    self.cursor.advance(); // consume opening quote
                    // Check for triple f-string
                    if self.cursor.peek() == Some(quote) && self.cursor.peek_at(1) == Some(quote) {
                        self.cursor.advance();
                        self.cursor.advance();
                        // Triple f-string — not yet supported; consume body until closing triple quotes
                        self.push_token(TokenKind::FStringStart, start);
                        self.diagnostics.push(
                            Diagnostic::error("triple-quoted f-strings are not yet supported")
                                .with_label(self.cursor.span_from(start), "here"),
                        );
                        loop {
                            match self.cursor.peek() {
                                None => break,
                                Some(c) if c == quote
                                    && self.cursor.peek_at(1) == Some(quote)
                                    && self.cursor.peek_at(2) == Some(quote) =>
                                {
                                    self.cursor.advance();
                                    self.cursor.advance();
                                    self.cursor.advance();
                                    break;
                                }
                                _ => { self.cursor.advance(); }
                            }
                        }
                        self.push_token(TokenKind::FStringEnd, start);
                    } else {
                        self.push_token(TokenKind::FStringStart, start);
                        self.lex_fstring_text_with_quote(quote);
                    }
                    return;
                }

                let ident = self.lex_identifier();
                if let Some(kw) = lookup_keyword(&ident) {
                    self.push_token(kw, start);
                } else {
                    let sym = self.interner.get_or_intern(&ident);
                    self.push_token(TokenKind::Ident(sym), start);
                }
            }

            // Operators and punctuation
            '+' => {
                self.cursor.advance();
                if self.cursor.eat('=') {
                    self.push_token(TokenKind::PlusEq, start);
                } else {
                    self.push_token(TokenKind::Plus, start);
                }
            }
            '-' => {
                self.cursor.advance();
                if self.cursor.eat('>') {
                    self.push_token(TokenKind::Arrow, start);
                } else if self.cursor.eat('=') {
                    self.push_token(TokenKind::MinusEq, start);
                } else {
                    self.push_token(TokenKind::Minus, start);
                }
            }
            '*' => {
                self.cursor.advance();
                if self.cursor.eat('*') {
                    self.push_token(TokenKind::DoubleStar, start);
                } else if self.cursor.eat('=') {
                    self.push_token(TokenKind::StarEq, start);
                } else {
                    self.push_token(TokenKind::Star, start);
                }
            }
            '/' => {
                self.cursor.advance();
                if self.cursor.eat('/') {
                    self.push_token(TokenKind::DoubleSlash, start);
                } else if self.cursor.eat('=') {
                    self.push_token(TokenKind::SlashEq, start);
                } else {
                    self.push_token(TokenKind::Slash, start);
                }
            }
            '%' => {
                self.cursor.advance();
                self.push_token(TokenKind::Percent, start);
            }
            '@' => {
                self.cursor.advance();
                self.push_token(TokenKind::At, start);
            }
            '=' => {
                self.cursor.advance();
                if self.cursor.eat('=') {
                    self.push_token(TokenKind::EqEq, start);
                } else if self.cursor.eat('>') {
                    self.push_token(TokenKind::FatArrow, start);
                } else {
                    self.push_token(TokenKind::Eq, start);
                }
            }
            '!' => {
                self.cursor.advance();
                if self.cursor.eat('=') {
                    self.push_token(TokenKind::NotEq, start);
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("unexpected character '!'; did you mean 'not'?")
                            .with_label(self.cursor.span_from(start), "here"),
                    );
                    self.push_token(TokenKind::Error("unexpected '!'".into()), start);
                }
            }
            '<' => {
                self.cursor.advance();
                if self.cursor.eat('=') {
                    self.push_token(TokenKind::LtEq, start);
                } else {
                    self.push_token(TokenKind::Lt, start);
                }
            }
            '>' => {
                self.cursor.advance();
                if self.cursor.eat('=') {
                    self.push_token(TokenKind::GtEq, start);
                } else {
                    self.push_token(TokenKind::Gt, start);
                }
            }
            '|' => {
                self.cursor.advance();
                if self.cursor.eat('>') {
                    self.push_token(TokenKind::Pipe, start);
                } else {
                    self.push_token(TokenKind::Bar, start);
                }
            }
            '&' => {
                self.cursor.advance();
                self.push_token(TokenKind::Ampersand, start);
            }
            '.' => {
                self.cursor.advance();
                if self.cursor.eat('.') {
                    if self.cursor.eat('=') {
                        self.push_token(TokenKind::DotDotEq, start);
                    } else if self.cursor.eat('.') {
                        self.push_token(TokenKind::Ellipsis, start);
                    } else {
                        self.push_token(TokenKind::DotDot, start);
                    }
                } else {
                    self.push_token(TokenKind::Dot, start);
                }
            }
            ',' => {
                self.cursor.advance();
                self.push_token(TokenKind::Comma, start);
            }
            ':' => {
                self.cursor.advance();
                self.push_token(TokenKind::Colon, start);
            }
            ';' => {
                self.cursor.advance();
                self.push_token(TokenKind::Semicolon, start);
            }
            '(' => {
                self.cursor.advance();
                self.indent.open_bracket();
                self.push_token(TokenKind::LeftParen, start);
            }
            ')' => {
                self.cursor.advance();
                self.indent.close_bracket();
                self.push_token(TokenKind::RightParen, start);
            }
            '[' => {
                self.cursor.advance();
                self.indent.open_bracket();
                self.push_token(TokenKind::LeftBracket, start);
            }
            ']' => {
                self.cursor.advance();
                self.indent.close_bracket();
                self.push_token(TokenKind::RightBracket, start);
            }
            '{' => {
                self.cursor.advance();
                self.indent.open_bracket();
                self.push_token(TokenKind::LeftBrace, start);
            }
            '}' => {
                self.cursor.advance();
                self.indent.close_bracket();
                self.push_token(TokenKind::RightBrace, start);
            }

            _ => {
                self.cursor.advance();
                self.diagnostics.push(
                    Diagnostic::error(format!("unexpected character: '{ch}'"))
                        .with_label(self.cursor.span_from(start), "here"),
                );
                self.push_token(TokenKind::Error(format!("unexpected '{ch}'")), start);
            }
        }
    }

    fn process_line_indentation(&mut self) {
        if self.line_continuation {
            self.line_continuation = false;
            // Skip indentation on continuation lines
            while self.cursor.peek() == Some(' ') {
                self.cursor.advance();
            }
            return;
        }

        // Count leading spaces
        let mut level = 0u32;
        while self.cursor.peek() == Some(' ') {
            self.cursor.advance();
            level += 1;
        }

        // Skip blank lines and comment-only lines
        match self.cursor.peek() {
            None | Some('\n') | Some('\r') => return,
            Some('#') => return, // comment-only line, don't change indent
            Some('\t') => {
                // Tab in indentation — error
                let start = self.cursor.pos();
                self.cursor.advance();
                self.diagnostics.push(
                    Diagnostic::error("tabs are not allowed; use 4 spaces for indentation")
                        .with_label(self.cursor.span_from(start), "tab character here"),
                );
                return;
            }
            _ => {}
        }

        let span = self.eof_span();
        match self.indent.process_indent(level, span) {
            Ok(tokens) => self.tokens.extend(tokens),
            Err(msg) => {
                self.diagnostics.push(
                    Diagnostic::error(msg).with_label(span, "invalid indentation"),
                );
            }
        }
    }

    fn lex_identifier(&mut self) -> String {
        let mut ident = String::new();
        while let Some(c) = self.cursor.peek() {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.cursor.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn lex_fstring_text_with_quote(&mut self, quote: char) {
        let start = self.cursor.pos();
        let mut text = String::new();

        loop {
            match self.cursor.peek() {
                None | Some('\n') | Some('\r') => {
                    if !text.is_empty() {
                        self.push_token(TokenKind::FStringText(text), start);
                    }
                    self.push_token(TokenKind::FStringEnd, start);
                    self.diagnostics.push(
                        Diagnostic::error("unterminated f-string")
                            .with_label(self.cursor.span_from(start), "f-string starts here"),
                    );
                    return;
                }
                Some(c) if c == quote => {
                    self.cursor.advance();
                    if !text.is_empty() {
                        self.push_token(TokenKind::FStringText(text), start);
                    }
                    self.push_token(TokenKind::FStringEnd, start);
                    return;
                }
                Some('{') => {
                    if self.cursor.peek_at(1) == Some('{') {
                        // Escaped brace {{ -> literal {
                        self.cursor.advance();
                        self.cursor.advance();
                        text.push('{');
                    } else {
                        // Start of expression
                        if !text.is_empty() {
                            self.push_token(TokenKind::FStringText(text.clone()), start);
                            text.clear();
                        }
                        let brace_start = self.cursor.pos();
                        self.cursor.advance(); // consume {
                        self.push_token(TokenKind::FStringExprStart, brace_start);
                        // Push f-string state with brace depth 0 and the quote char
                        self.fstring_stack.push((0, quote));
                        return; // Return to main scan loop to lex the expression
                    }
                }
                Some('}') => {
                    if self.cursor.peek_at(1) == Some('}') {
                        // Escaped brace }} -> literal }
                        self.cursor.advance();
                        self.cursor.advance();
                        text.push('}');
                    } else {
                        // Unexpected }
                        self.cursor.advance();
                        text.push('}');
                    }
                }
                Some('\\') => {
                    self.cursor.advance();
                    match self.cursor.peek() {
                        Some('n') => { self.cursor.advance(); text.push('\n'); }
                        Some('t') => { self.cursor.advance(); text.push('\t'); }
                        Some('\\') => { self.cursor.advance(); text.push('\\'); }
                        Some(c) if c == quote => { self.cursor.advance(); text.push(c); }
                        Some(c) => { self.cursor.advance(); text.push('\\'); text.push(c); }
                        None => { text.push('\\'); }
                    }
                }
                Some(c) => {
                    self.cursor.advance();
                    text.push(c);
                }
            }
        }
    }

    fn push_token(&mut self, kind: TokenKind, start: BytePos) {
        let span = self.cursor.span_from(start);
        self.tokens.push(Token { kind, span });
    }
}
