use crate::cursor::Cursor;
use crate::token::TokenKind;
use nsl_errors::{BytePos, Diagnostic};

/// Lex a number literal starting at the current cursor position.
/// The cursor should be positioned at the first digit.
pub fn lex_number(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    let first = cursor.peek().unwrap_or('0');

    // Check for hex, octal, binary prefixes
    if first == '0' {
        if let Some(next) = cursor.peek_at(1) {
            match next {
                'x' | 'X' => return lex_hex(cursor, start, diagnostics),
                'o' | 'O' => return lex_octal(cursor, start, diagnostics),
                'b' | 'B' => return lex_binary(cursor, start, diagnostics),
                _ => {}
            }
        }
    }

    // Decimal integer or float
    lex_decimal(cursor, start, diagnostics)
}

fn lex_hex(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    cursor.advance(); // '0'
    cursor.advance(); // 'x' or 'X'

    let digits = eat_digits(cursor, |c| c.is_ascii_hexdigit());
    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    if clean.is_empty() {
        diagnostics.push(
            Diagnostic::error("expected hex digits after '0x'")
                .with_label(cursor.span_from(start), "here"),
        );
        return TokenKind::Error("invalid hex literal".into());
    }
    match i64::from_str_radix(&clean, 16) {
        Ok(v) => TokenKind::IntLiteral(v),
        Err(_) => {
            diagnostics.push(
                Diagnostic::error("hex literal too large")
                    .with_label(cursor.span_from(start), "here"),
            );
            TokenKind::Error("hex literal too large".into())
        }
    }
}

fn lex_octal(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    cursor.advance(); // '0'
    cursor.advance(); // 'o' or 'O'

    let digits = eat_digits(cursor, |c| matches!(c, '0'..='7'));
    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    if clean.is_empty() {
        diagnostics.push(
            Diagnostic::error("expected octal digits after '0o'")
                .with_label(cursor.span_from(start), "here"),
        );
        return TokenKind::Error("invalid octal literal".into());
    }
    match i64::from_str_radix(&clean, 8) {
        Ok(v) => TokenKind::IntLiteral(v),
        Err(_) => {
            diagnostics.push(
                Diagnostic::error("octal literal too large")
                    .with_label(cursor.span_from(start), "here"),
            );
            TokenKind::Error("octal literal too large".into())
        }
    }
}

fn lex_binary(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    cursor.advance(); // '0'
    cursor.advance(); // 'b' or 'B'

    let digits = eat_digits(cursor, |c| matches!(c, '0' | '1'));
    let clean: String = digits.chars().filter(|c| *c != '_').collect();
    if clean.is_empty() {
        diagnostics.push(
            Diagnostic::error("expected binary digits after '0b'")
                .with_label(cursor.span_from(start), "here"),
        );
        return TokenKind::Error("invalid binary literal".into());
    }
    match i64::from_str_radix(&clean, 2) {
        Ok(v) => TokenKind::IntLiteral(v),
        Err(_) => {
            diagnostics.push(
                Diagnostic::error("binary literal too large")
                    .with_label(cursor.span_from(start), "here"),
            );
            TokenKind::Error("binary literal too large".into())
        }
    }
}

fn lex_decimal(cursor: &mut Cursor, start: BytePos, diagnostics: &mut Vec<Diagnostic>) -> TokenKind {
    let int_part = eat_digits(cursor, |c| c.is_ascii_digit());
    let mut is_float = false;
    let mut number_str = int_part.to_string();

    // Check for decimal point — but not `..` (range) or `.method` (member access)
    if cursor.peek() == Some('.') {
        match cursor.peek_at(1) {
            Some('.') => {} // `..` range operator, stop here
            Some(c) if c.is_ascii_digit() => {
                is_float = true;
                cursor.advance(); // consume '.'
                number_str.push('.');
                let frac = eat_digits(cursor, |c| c.is_ascii_digit());
                number_str.push_str(frac);
            }
            _ => {} // `.method` — stop at integer
        }
    }

    // Check for exponent — only consume `e`/`E` if followed by digits or sign+digits,
    // otherwise `123e` or `123east` should lex as integer `123` followed by identifier.
    if let Some('e' | 'E') = cursor.peek() {
        let has_exp_digits = match cursor.peek_at(1) {
            Some('+' | '-') => matches!(cursor.peek_at(2), Some('0'..='9')),
            Some('0'..='9') => true,
            _ => false,
        };
        if has_exp_digits {
            is_float = true;
            number_str.push(cursor.advance().unwrap());
            if let Some('+' | '-') = cursor.peek() {
                number_str.push(cursor.advance().unwrap());
            }
            let exp = eat_digits(cursor, |c| c.is_ascii_digit());
            number_str.push_str(exp);
        }
    }

    // Remove underscores for parsing
    let clean: String = number_str.chars().filter(|c| *c != '_').collect();

    if is_float {
        match clean.parse::<f64>() {
            Ok(v) => TokenKind::FloatLiteral(v),
            Err(_) => {
                diagnostics.push(
                    Diagnostic::error("invalid float literal")
                        .with_label(cursor.span_from(start), "here"),
                );
                TokenKind::Error("invalid float literal".into())
            }
        }
    } else {
        match clean.parse::<i64>() {
            Ok(v) => TokenKind::IntLiteral(v),
            Err(_) => {
                diagnostics.push(
                    Diagnostic::error("integer literal too large")
                        .with_label(cursor.span_from(start), "here"),
                );
                TokenKind::Error("integer literal too large".into())
            }
        }
    }
}

fn eat_digits<'a>(cursor: &mut Cursor<'a>, pred: impl Fn(char) -> bool) -> &'a str {
    cursor.eat_while(|c| pred(c) || c == '_')
}
