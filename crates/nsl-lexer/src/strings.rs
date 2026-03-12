use crate::cursor::Cursor;
use crate::token::TokenKind;
use nsl_errors::{BytePos, Diagnostic};

/// Lex a string literal (single-quoted, double-quoted, or triple-quoted).
/// The cursor should be positioned at the opening quote character.
pub fn lex_string(cursor: &mut Cursor, quote: char, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    cursor.advance(); // consume opening quote

    // Check for triple-quoted string
    if cursor.peek() == Some(quote) && cursor.peek_at(1) == Some(quote) {
        cursor.advance(); // second quote
        cursor.advance(); // third quote
        return lex_triple_string(cursor, quote, start, diagnostics);
    }

    // Single-line string
    let mut value = String::new();

    loop {
        match cursor.peek() {
            None | Some('\n') | Some('\r') => {
                diagnostics.push(
                    Diagnostic::error("unterminated string literal")
                        .with_label(cursor.span_from(start), "string starts here"),
                );
                return TokenKind::Error("unterminated string".into());
            }
            Some(c) if c == quote => {
                cursor.advance(); // consume closing quote
                return TokenKind::StringLiteral(value);
            }
            Some('\\') => {
                cursor.advance(); // consume backslash
                match lex_escape(cursor, start, diagnostics) {
                    Some(ch) => value.push(ch),
                    None => {} // error already reported
                }
            }
            Some(c) => {
                cursor.advance();
                value.push(c);
            }
        }
    }
}

fn lex_triple_string(cursor: &mut Cursor, quote: char, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    let mut value = String::new();
    let end_seq = [quote; 3];

    loop {
        match cursor.peek() {
            None => {
                diagnostics.push(
                    Diagnostic::error("unterminated triple-quoted string")
                        .with_label(cursor.span_from(start), "string starts here"),
                );
                return TokenKind::Error("unterminated triple-quoted string".into());
            }
            Some(c) if c == end_seq[0]
                && cursor.peek_at(1) == Some(end_seq[1])
                && cursor.peek_at(2) == Some(end_seq[2]) =>
            {
                cursor.advance();
                cursor.advance();
                cursor.advance();
                return TokenKind::StringLiteral(value);
            }
            Some('\\') => {
                cursor.advance();
                match lex_escape(cursor, start, diagnostics) {
                    Some(ch) => value.push(ch),
                    None => {}
                }
            }
            Some(c) => {
                cursor.advance();
                value.push(c);
            }
        }
    }
}

fn lex_escape(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> Option<char> {
    match cursor.advance() {
        Some('n') => Some('\n'),
        Some('t') => Some('\t'),
        Some('r') => Some('\r'),
        Some('\\') => Some('\\'),
        Some('\'') => Some('\''),
        Some('"') => Some('"'),
        Some('0') => Some('\0'),
        Some('x') => {
            let mut hex = String::new();
            for _ in 0..2 {
                match cursor.peek() {
                    Some(c) if c.is_ascii_hexdigit() => {
                        hex.push(c);
                        cursor.advance();
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error("expected 2 hex digits in \\x escape")
                                .with_label(cursor.span_from(start), "here"),
                        );
                        return None;
                    }
                }
            }
            let code = u32::from_str_radix(&hex, 16).unwrap();
            char::from_u32(code)
        }
        Some('u') => {
            if cursor.eat('{') {
                let mut hex = String::new();
                let mut found_close = false;
                while let Some(c) = cursor.peek() {
                    if c == '}' {
                        cursor.advance();
                        found_close = true;
                        break;
                    }
                    if c.is_ascii_hexdigit() && hex.len() < 6 {
                        hex.push(c);
                        cursor.advance();
                    } else {
                        diagnostics.push(
                            Diagnostic::error("invalid unicode escape")
                                .with_label(cursor.span_from(start), "here"),
                        );
                        return None;
                    }
                }
                if !found_close {
                    diagnostics.push(
                        Diagnostic::error("unterminated \\u{...} escape (missing '}')")
                            .with_label(cursor.span_from(start), "here"),
                    );
                    return None;
                }
                if hex.is_empty() {
                    diagnostics.push(
                        Diagnostic::error("empty \\u{} escape sequence")
                            .with_label(cursor.span_from(start), "here"),
                    );
                    return None;
                }
                let code = u32::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                match char::from_u32(code) {
                    Some(ch) => Some(ch),
                    None => {
                        diagnostics.push(
                            Diagnostic::error(format!("invalid unicode code point: U+{:04X}", code))
                                .with_label(cursor.span_from(start), "here"),
                        );
                        None
                    }
                }
            } else {
                diagnostics.push(
                    Diagnostic::error("expected '{' after \\u")
                        .with_label(cursor.span_from(start), "here"),
                );
                None
            }
        }
        Some(c) => {
            diagnostics.push(
                Diagnostic::error(format!("unknown escape sequence: \\{c}"))
                    .with_label(cursor.span_from(start), "here"),
            );
            Some(c) // recover by using the literal character
        }
        None => {
            diagnostics.push(
                Diagnostic::error("unexpected end of file in escape sequence")
                    .with_label(cursor.span_from(start), "here"),
            );
            None
        }
    }
}
