use std::path::Path;

use nsl_errors::FileId;
use nsl_lexer::{Interner, Token, TokenKind};

#[derive(Debug)]
pub struct FormatResult {
    pub changed: bool,
    pub output: String,
}

pub fn format_source(input: &str) -> Result<FormatResult, String> {
    // Phase 1: Check for mixed indentation (hard fail)
    check_indentation_safety(input)?;

    let mut output = input.to_string();

    // Phase 2: Convert tabs to 4 spaces
    output = normalize_tabs(&output);

    // Phase 3: Trailing whitespace
    output = strip_trailing_whitespace(&output);

    // Phase 4: Max 2 consecutive blank lines
    output = normalize_blank_lines(&output);

    // Phases 5 and 6 work from the token stream and refuse source the
    // lexer rejects: without a reliable token stream the safe reformat is
    // none, and a refusal is visible where a silent no-op is not.

    // Phase 5: Operator spacing (careful with unary minus and comments/strings)
    output = normalize_operators(&output)?;

    // Phase 6: String quotes (single → double, but not in comments or already-double-quoted)
    output = normalize_quotes(&output)?;

    let changed = output != input;
    Ok(FormatResult { changed, output })
}

pub fn format_file(path: &Path, check: bool) -> Result<bool, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("error reading '{}': {}", path.display(), e))?;

    let result = format_source(&content).map_err(|e| format!("{}: {e}", path.display()))?;

    if !result.changed {
        return Ok(false);
    }

    if check {
        // Don't write, just report
        return Ok(true);
    }

    std::fs::write(path, &result.output)
        .map_err(|e| format!("error writing '{}': {}", path.display(), e))?;

    Ok(true)
}

/// Hard-fail if any line has BOTH tabs and spaces as leading whitespace.
fn check_indentation_safety(input: &str) -> Result<(), String> {
    for (i, line) in input.lines().enumerate() {
        let mut saw_tab = false;
        let mut saw_space = false;
        for ch in line.chars() {
            match ch {
                '\t' => saw_tab = true,
                ' ' => saw_space = true,
                _ => break,
            }
        }
        if saw_tab && saw_space {
            return Err(format!(
                "error: ambiguous mixed indentation at line {}. Fix manually before formatting.",
                i + 1
            ));
        }
    }
    Ok(())
}

/// Replace leading tabs with 4 spaces each. Only touches leading whitespace.
fn normalize_tabs(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        let leading_tabs = line.chars().take_while(|&c| c == '\t').count();
        if leading_tabs > 0 {
            for _ in 0..leading_tabs {
                result.push_str("    ");
            }
            result.push_str(&line[leading_tabs..]);
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    // Preserve trailing newline presence/absence
    if !input.is_empty() && !input.ends_with('\n') {
        // Remove the last \n we added
        result.pop();
    }
    result
}

/// Trim trailing spaces and tabs from each line.
fn strip_trailing_whitespace(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for line in input.lines() {
        result.push_str(line.trim_end_matches([' ', '\t']));
        result.push('\n');
    }
    if !input.is_empty() && !input.ends_with('\n') {
        result.pop();
    }
    result
}

/// Collapse runs of 3+ consecutive blank lines to exactly 2 blank lines.
fn normalize_blank_lines(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut consecutive_blank = 0u32;

    for line in input.lines() {
        if line.trim().is_empty() {
            consecutive_blank += 1;
            if consecutive_blank <= 2 {
                result.push('\n');
            }
        } else {
            consecutive_blank = 0;
            result.push_str(line);
            result.push('\n');
        }
    }

    if !input.is_empty() && !input.ends_with('\n') {
        result.pop();
    }
    result
}

/// The token stream phases 5 and 6 work from, or why the lexer rejected
/// `input`.
fn lex(input: &str) -> Result<(Vec<Token>, Interner), String> {
    let mut interner = Interner::new();
    let (tokens, diagnostics) = nsl_lexer::tokenize(input, FileId(0), &mut interner);
    let Some(d) = diagnostics.first() else {
        return Ok((tokens, interner));
    };
    let line = d.labels.first().map(|label| {
        let start = (label.span.start.0 as usize).min(input.len());
        input[..start].matches('\n').count() + 1
    });
    Err(match line {
        Some(n) => format!("error: cannot format, line {n}: {}", d.message),
        None => format!("error: cannot format: {}", d.message),
    })
}

/// Phase 5: one space on each side of a binary operator. Driven by the
/// lexer's token stream, so this phase never touches string and f-string
/// contents, comments, or literals such as `1e-8`, and reads `//` as one
/// operator rather than two. (Phases 2–4 are line-based and still rewrite
/// tabs, trailing whitespace and blank-line runs inside a multi-line
/// string; that predates this phase.)
///
/// The spaced operators are `+ - * / // < > == != <= >= += -= *= /= |> ->`
/// and `=`. A `-` or `*` with no operand before it (`-1`, `*rest`) is unary
/// and left as written, as are `**`, `<`/`>` that open and close a generic
/// argument list (`Tensor<[4], f32>`, recognised by a capitalised or
/// just-declared name glued to the `<`), and `=` inside brackets (`f(x=1)`,
/// `fn f(x: int=1)`), where both spellings are common. No space is ever
/// inserted directly inside a bracket. Anything that is not the gap between
/// two tokens on one line is copied verbatim.
fn normalize_operators(input: &str) -> Result<String, String> {
    let (tokens, interner) = lex(input)?;

    let mut out = String::with_capacity(input.len() + 16);
    let mut copied = 0usize; // bytes of `input` already in `out`
    let mut prev: Option<&Token> = None; // last token on this line, if any
    let mut prev2: Option<&Token> = None; // the one before it
    let mut prev_spaced = false; // was `prev` a spaced operator?
    let mut fstring_depth = 0u32;
    let mut generic_depth = 0u32;
    let mut bracket_depth = 0u32;

    for tok in &tokens {
        let start = tok.span.start.0 as usize;
        let end = tok.span.end.0 as usize;
        match tok.kind {
            TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent | TokenKind::Eof => {
                prev = None;
                prev2 = None;
                prev_spaced = false;
                generic_depth = 0;
                continue;
            }
            TokenKind::DocComment(_) => {
                prev = None;
                prev2 = None;
                prev_spaced = false;
                continue;
            }
            _ => {}
        }

        // Inside an f-string everything is verbatim, including the
        // expressions between its braces.
        let inside_fstring = fstring_depth > 0;
        match tok.kind {
            TokenKind::FStringStart => fstring_depth += 1,
            TokenKind::FStringEnd => fstring_depth = fstring_depth.saturating_sub(1),
            _ => {}
        }

        match tok.kind {
            TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace => {
                bracket_depth += 1
            }
            TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace => {
                bracket_depth = bracket_depth.saturating_sub(1)
            }
            _ => {}
        }

        let prev_ends_operand = prev.is_some_and(|p| ends_operand(&p.kind));
        // `get`, not indexing: an `FStringEnd` after trailing text shares
        // that text's span, so its start precedes `prev`'s end.
        let gap = prev.and_then(|p| input.get(p.span.end.0 as usize..start));

        // `Tensor<...>` / `fn first<T>(...)`: a `<` glued to a capitalised
        // name, or to the name being declared, opens a generic list; its
        // angle brackets are not comparisons.
        let opens_generic = tok.kind == TokenKind::Lt
            && gap == Some("")
            && prev.is_some_and(|p| match &p.kind {
                TokenKind::Ident(sym) => {
                    let declared = prev2.is_some_and(|q| {
                        matches!(
                            q.kind,
                            TokenKind::Fn | TokenKind::Struct | TokenKind::Enum | TokenKind::Trait
                        )
                    });
                    declared
                        || interner
                            .resolve(*sym)
                            .is_some_and(|name| name.starts_with(|c: char| c.is_ascii_uppercase()))
                }
                _ => false,
            });
        let in_generic = generic_depth > 0 || opens_generic;
        if opens_generic {
            generic_depth += 1;
        } else if in_generic && tok.kind == TokenKind::Gt {
            // A bare `<` inside a generic list is a dim bound
            // (`Tensor<[SeqLen < 4096], f32>`), not a nested list, so only
            // `>` moves the depth.
            generic_depth -= 1;
        }

        let spaced = !inside_fstring
            && match tok.kind {
                TokenKind::Minus | TokenKind::Star => prev_ends_operand,
                TokenKind::Lt | TokenKind::Gt => !in_generic,
                TokenKind::Eq => bracket_depth == 0,
                TokenKind::Plus
                | TokenKind::Slash
                | TokenKind::DoubleSlash
                | TokenKind::EqEq
                | TokenKind::NotEq
                | TokenKind::LtEq
                | TokenKind::GtEq
                | TokenKind::PlusEq
                | TokenKind::MinusEq
                | TokenKind::StarEq
                | TokenKind::SlashEq
                | TokenKind::Pipe
                | TokenKind::Arrow => true,
                _ => false,
            };

        // Only the gap between two tokens on one line, made of spaces (or
        // nothing), is ever rewritten: a `\` continuation or a newline
        // inside brackets stays exactly as written.
        let editable = !inside_fstring
            && gap.is_some_and(|g| g.bytes().all(|b| b == b' '))
            && (spaced || prev_spaced)
            && !prev.is_some_and(|p| opens_bracket(&p.kind))
            && !closes_or_punctuates(&tok.kind);

        match prev {
            Some(p) if editable => {
                out.push_str(&input[copied..p.span.end.0 as usize]);
                out.push(' ');
                out.push_str(&input[start..end]);
            }
            _ => out.push_str(&input[copied..end]),
        }
        copied = end;
        prev2 = prev;
        prev = Some(tok);
        prev_spaced = spaced;
    }
    out.push_str(&input[copied..]);
    Ok(out)
}

/// Does a token of this kind end an operand, so that a following `-` or `*`
/// is binary?
fn ends_operand(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Ident(_)
            | TokenKind::IntLiteral(_)
            | TokenKind::FloatLiteral(_)
            | TokenKind::StringLiteral(_)
            | TokenKind::FStringEnd
            | TokenKind::True
            | TokenKind::False
            | TokenKind::None
            | TokenKind::SelfKw
            | TokenKind::RightParen
            | TokenKind::RightBracket
            | TokenKind::RightBrace
    )
}

fn opens_bracket(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace
    )
}

/// Tokens that never get a space put in front of them, whatever precedes.
fn closes_or_punctuates(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::RightParen
            | TokenKind::RightBracket
            | TokenKind::RightBrace
            | TokenKind::Comma
            | TokenKind::Colon
            | TokenKind::Semicolon
            | TokenKind::Dot
    )
}

/// Phase 6: single-quoted strings become double-quoted, keeping the same
/// value: `\'` unescapes and `"` is escaped. Driven by the token stream, so
/// an apostrophe in a comment or inside a double-quoted string is never a
/// quote.
///
/// An f-string's text is not rewritten, so a single-quoted f-string is
/// converted only when its quotes can simply be swapped: no `"` anywhere
/// inside it (in its text it would need an escape; in an interpolation it
/// would end the converted literal early), and no `\'` (the lexer only
/// unescapes the quote in use, so `\'` would become two characters). A
/// triple-quoted string is converted only when its body holds no `"` at
/// all. Nothing inside an f-string's braces is converted, and neither is a
/// nested f-string: the output never leans on the lexer's handling of a
/// quote nested inside the same quote.
fn normalize_quotes(input: &str) -> Result<String, String> {
    let (tokens, _interner) = lex(input)?;

    // (byte range, replacement) edits; an f-string's edits are only known
    // at its end, so they are sorted before they are applied.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut fstring_starts: Vec<&Token> = Vec::new();
    for tok in &tokens {
        let start = tok.span.start.0 as usize;
        let end = tok.span.end.0 as usize;
        match &tok.kind {
            TokenKind::StringLiteral(_) if fstring_starts.is_empty() => {
                let raw = &input[start..end];
                if let Some(converted) = requote(raw) {
                    edits.push((start, end, converted));
                }
            }
            TokenKind::FStringStart => fstring_starts.push(tok),
            TokenKind::FStringEnd => {
                let Some(open) = fstring_starts.pop() else {
                    continue;
                };
                if !fstring_starts.is_empty() {
                    continue; // nested in another f-string
                }
                let open_start = open.span.start.0 as usize;
                let open_end = open.span.end.0 as usize;
                let quote_len = open_end - open_start - 1; // minus the `f`
                let quotes = &input[open_start + 1..open_end];
                let body = &input[open_end..end - quote_len];
                if quotes.starts_with('\'') && !body.contains('"') && !body.contains("\\'") {
                    let dq = "\"".repeat(quote_len);
                    edits.push((open_start + 1, open_end, dq.clone()));
                    edits.push((end - quote_len, end, dq));
                }
            }
            _ => {}
        }
    }
    edits.sort_by_key(|(start, _, _)| *start);

    let mut out = String::with_capacity(input.len());
    let mut copied = 0usize;
    for (start, end, replacement) in edits {
        out.push_str(&input[copied..start]);
        out.push_str(&replacement);
        copied = end;
    }
    out.push_str(&input[copied..]);
    Ok(out)
}

/// The double-quoted spelling of a single-quoted string literal's source
/// text, or `None` if it is already double-quoted or cannot be converted.
fn requote(raw: &str) -> Option<String> {
    if !raw.starts_with('\'') {
        return None;
    }
    const TRIPLE: &str = "'''";
    if let Some(body) = raw.strip_prefix(TRIPLE).and_then(|r| r.strip_suffix(TRIPLE)) {
        if body.contains('"') {
            return None;
        }
        return Some(format!("\"\"\"{body}\"\"\""));
    }
    let body = &raw[1..raw.len() - 1];
    let mut converted = String::with_capacity(body.len() + 2);
    converted.push('"');
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('\'') => converted.push('\''),
                Some(next) => {
                    converted.push('\\');
                    converted.push(next);
                }
                None => converted.push('\\'),
            },
            '"' => converted.push_str("\\\""),
            other => converted.push(other),
        }
    }
    converted.push('"');
    Some(converted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_indentation_safety_ok() {
        assert!(check_indentation_safety("    let x = 1\n    let y = 2\n").is_ok());
        assert!(check_indentation_safety("\tlet x = 1\n\tlet y = 2\n").is_ok());
    }

    #[test]
    fn test_check_indentation_safety_fail() {
        let result = check_indentation_safety("\t    let x = 1\n");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("line 1"));
    }

    #[test]
    fn test_normalize_tabs() {
        let input = "\tlet x = 1\n\t\tlet y = 2\n";
        let output = normalize_tabs(input);
        assert_eq!(output, "    let x = 1\n        let y = 2\n");
    }

    #[test]
    fn test_strip_trailing_whitespace() {
        let input = "let x = 1   \nlet y = 2\t\n";
        let output = strip_trailing_whitespace(input);
        assert_eq!(output, "let x = 1\nlet y = 2\n");
    }

    #[test]
    fn test_normalize_blank_lines() {
        let input = "a\n\n\n\n\nb\n";
        let output = normalize_blank_lines(input);
        assert_eq!(output, "a\n\n\nb\n");
    }

    #[test]
    fn test_normalize_quotes_simple() {
        assert_eq!(normalize_quotes("let s = 'hello'\n").unwrap(), "let s = \"hello\"\n");
    }

    #[test]
    fn test_normalize_quotes_apostrophe_in_comment_preserved() {
        assert_eq!(normalize_quotes("# don't touch this\n").unwrap(), "# don't touch this\n");
        assert_eq!(
            normalize_quotes("let s = \"it's\"  # isn't\n").unwrap(),
            "let s = \"it's\"  # isn't\n"
        );
    }

    #[test]
    fn test_normalize_quotes_keeps_the_value() {
        // `\'` unescapes, `"` gets escaped, other escapes ride along.
        assert_eq!(
            normalize_quotes("let s = 'it\\'s \"q\" \\n \\\\'\n").unwrap(),
            "let s = \"it's \\\"q\\\" \\n \\\\\"\n"
        );
    }

    #[test]
    fn test_normalize_quotes_triple() {
        assert_eq!(
            normalize_quotes("let s = '''a\nb'''\n").unwrap(),
            "let s = \"\"\"a\nb\"\"\"\n"
        );
        // A `"` in the body would need an escape, and the lexer keeps
        // triple-quoted bodies literal, so this one is left alone.
        let input = "let s = '''say \"hi\"'''\n";
        assert_eq!(normalize_quotes(input).unwrap(), input);
    }

    #[test]
    fn test_normalize_quotes_fstrings() {
        assert_eq!(
            normalize_quotes("print(f'{x} = {y}')\n").unwrap(),
            "print(f\"{x} = {y}\")\n"
        );
        // A `"` anywhere inside a single-quoted f-string blocks conversion.
        let nested = "print(f'{d[\"k\"]}')\n";
        assert_eq!(normalize_quotes(nested).unwrap(), nested);
        // A literal inside the braces is never converted: the result would
        // nest a `"` inside `"`.
        let inner = "print(f\"{d['k']}\")\n";
        assert_eq!(normalize_quotes(inner).unwrap(), inner);
    }

    #[test]
    fn test_normalize_quotes_refuses_unlexable_source() {
        let err = normalize_quotes("let s = 'open\n").unwrap_err();
        assert!(err.starts_with("error: cannot format, line 1: "), "{err}");
    }

    #[test]
    fn test_normalize_quotes_fstring_with_inner_literal_converts_only_the_outer() {
        // The outer quotes are swapped; a literal inside the braces is left
        // as written, and the edit order does not depend on token order
        // (the outer edit is only known at the f-string's end).
        assert_eq!(
            normalize_quotes("let k = 'k'\nlet s = f'{d['k']}'\n").unwrap(),
            "let k = \"k\"\nlet s = f\"{d['k']}\"\n"
        );
        assert_eq!(
            normalize_quotes("print(f'{f'{x}'}')\n").unwrap(),
            "print(f\"{f'{x}'}\")\n"
        );
    }

    #[test]
    fn test_normalize_quotes_fstring_with_escaped_quote_is_left_alone() {
        // `\'` is only an escape while the quote is `'`; swapping the quotes
        // would turn it into two characters.
        for input in [
            "let s = f'it\\'s {x}'\n",
            "let s = f'{x}\\''\n",
            "let s = f'''it\\'s'''\n",
            "let s = f'a\\\\\\'b'\n",
        ] {
            assert_eq!(normalize_quotes(input).unwrap(), input, "{input:?}");
        }
    }

    #[test]
    fn test_operator_spacing_basic() {
        assert_eq!(normalize_operators("let x=1+2\n").unwrap(), "let x = 1 + 2\n");
        assert_eq!(normalize_operators("let x   =   1\n").unwrap(), "let x = 1\n");
    }

    #[test]
    fn test_operator_spacing_comparison() {
        assert_eq!(
            normalize_operators("if x==y:\n    pass\n").unwrap(),
            "if x == y:\n    pass\n"
        );
        assert_eq!(
            normalize_operators("if a<b and c>=d:\n    pass\n").unwrap(),
            "if a < b and c >= d:\n    pass\n"
        );
    }

    #[test]
    fn test_operator_spacing_pipe_and_arrow() {
        assert_eq!(normalize_operators("x|>f\n").unwrap(), "x |> f\n");
        assert_eq!(
            normalize_operators("fn f(x: int)->int:\n    return x\n").unwrap(),
            "fn f(x: int) -> int:\n    return x\n"
        );
    }

    #[test]
    fn test_operator_spacing_exponent_untouched() {
        assert_eq!(normalize_operators("x**2\n").unwrap(), "x**2\n");
        assert_eq!(normalize_operators("x ** 2\n").unwrap(), "x ** 2\n");
    }

    #[test]
    fn test_unary_minus_and_star_untouched() {
        assert_eq!(normalize_operators("let x = -1\n").unwrap(), "let x = -1\n");
        assert_eq!(normalize_operators("let x=-1\n").unwrap(), "let x = -1\n");
        assert_eq!(normalize_operators("return -x\n").unwrap(), "return -x\n");
        assert_eq!(normalize_operators("f(-x, *rest)\n").unwrap(), "f(-x, *rest)\n");
        assert_eq!(normalize_operators("a[-1]\n").unwrap(), "a[-1]\n");
        // ...while a binary minus is spaced.
        assert_eq!(normalize_operators("let y = x-1\n").unwrap(), "let y = x - 1\n");
        assert_eq!(normalize_operators("let y = f(x)-1\n").unwrap(), "let y = f(x) - 1\n");
    }

    #[test]
    fn test_exponent_literal_is_one_token() {
        // The pre-token formatter turned `1e-8` into `1e - 8`, which does
        // not parse, and read `//` as two `/`.
        assert_eq!(normalize_operators("let eps = 1e-8\n").unwrap(), "let eps = 1e-8\n");
        assert_eq!(normalize_operators("let lr = 6e-4\n").unwrap(), "let lr = 6e-4\n");
        assert_eq!(normalize_operators("let h = n//2\n").unwrap(), "let h = n // 2\n");
        assert_eq!(normalize_operators("let h = n // 2\n").unwrap(), "let h = n // 2\n");
    }

    #[test]
    fn test_ranges_and_other_operators_verbatim() {
        let closed = "for i in 0..=n:\n    pass\n";
        assert_eq!(normalize_operators(closed).unwrap(), closed);
        let open = "for i in 0..n:\n    pass\n";
        assert_eq!(normalize_operators(open).unwrap(), open);
        assert_eq!(normalize_operators("let m = a@b\n").unwrap(), "let m = a@b\n");
        assert_eq!(normalize_operators("let r = a%b\n").unwrap(), "let r = a%b\n");
    }

    #[test]
    fn test_generics_are_not_comparisons() {
        let input = "fn f(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n";
        assert_eq!(normalize_operators(input).unwrap(), input);
        let bound = "fn g(q: Tensor<[SeqLen < 4096, 64], f32>) -> int:\n    return 1\n";
        assert_eq!(normalize_operators(bound).unwrap(), bound);
        let declared = "fn first<T>(xs: List<T>) -> T:\n    return xs[0]\n";
        assert_eq!(normalize_operators(declared).unwrap(), declared);
        // A comparison against a capitalised name is still spaced when it
        // is written with a space...
        assert_eq!(
            normalize_operators("if N <n:\n    pass\n").unwrap(),
            "if N < n:\n    pass\n"
        );
        // ...and a spaced operator never pushes a space in front of
        // punctuation, so the old formatter's output is a fixed point.
        let legacy = "fn h(x: Tensor < [1], f64 > , y: int) -> Tensor < [1], f64 > :\n    return x\n";
        assert_eq!(normalize_operators(legacy).unwrap(), legacy);
    }

    #[test]
    fn test_keyword_argument_equals_verbatim() {
        assert_eq!(normalize_operators("f(x=1, y = 2)\n").unwrap(), "f(x=1, y = 2)\n");
        let default = "fn f(x: int=1):\n    pass\n";
        assert_eq!(normalize_operators(default).unwrap(), default);
        // Assignment outside brackets is still normalised, whatever is on
        // the left.
        assert_eq!(normalize_operators("xs[i]=1\n").unwrap(), "xs[i] = 1\n");
    }

    #[test]
    fn test_no_space_inside_brackets() {
        assert_eq!(normalize_operators("f(-1)\n").unwrap(), "f(-1)\n");
        assert_eq!(normalize_operators("let t = (a+b)\n").unwrap(), "let t = (a + b)\n");
        assert_eq!(normalize_operators("let t = [a+b,c]\n").unwrap(), "let t = [a + b,c]\n");
    }

    #[test]
    fn test_strings_comments_and_fstrings_verbatim() {
        let input = "let s = \"a+b\"  # x-y\nlet t = f\"{a+b}-{c}\"\nlet u = \"\"\"1+1\n2-2\"\"\"\n";
        assert_eq!(normalize_operators(input).unwrap(), input);
        // The gap before and after a string is still normalised.
        assert_eq!(
            normalize_operators("let s=\"a\"+f\"{b}\"\n").unwrap(),
            "let s = \"a\" + f\"{b}\"\n"
        );
    }

    #[test]
    fn test_line_continuations_verbatim() {
        let input = "let x = a \\\n    + b\nlet y = (a\n    + b)\n";
        assert_eq!(normalize_operators(input).unwrap(), input);
    }

    #[test]
    fn test_operator_spacing_refuses_unlexable_source() {
        let err = normalize_operators("let x = 1\nlet y=1 ! 2\n").unwrap_err();
        assert!(err.starts_with("error: cannot format, line 2: "), "{err}");
        let err = format_source("let y=1 ! 2\n").unwrap_err();
        assert!(err.starts_with("error: cannot format, line 1: "), "{err}");
    }

    #[test]
    fn test_format_source_is_idempotent_on_its_own_output() {
        let input = "let eps=1e-8\nlet h=n//2\nlet s='it\\'s'\nfn f(x: Tensor<[4], f32>)->Tensor<[4], f32>:\n    return x*2\n";
        let once = format_source(input).unwrap();
        assert!(once.changed);
        assert_eq!(
            once.output,
            "let eps = 1e-8\nlet h = n // 2\nlet s = \"it's\"\nfn f(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x * 2\n"
        );
        let twice = format_source(&once.output).unwrap();
        assert!(!twice.changed);
    }
}
