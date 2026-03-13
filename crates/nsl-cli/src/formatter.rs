use std::path::Path;

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

    // Phase 5: Operator spacing (careful with unary minus and comments/strings)
    output = normalize_operators(&output);

    // Phase 6: String quotes (single → double, but not in comments or already-double-quoted)
    output = normalize_quotes(&output);

    let changed = output != input;
    Ok(FormatResult { changed, output })
}

pub fn format_file(path: &Path, check: bool) -> Result<bool, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("error reading '{}': {}", path.display(), e))?;

    let result = format_source(&content)?;

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

/// Ensure single space around binary operators. Skips content inside strings and
/// line comments. Does not touch unary minus, `**`, or operators inside f-string `{...}`.
fn normalize_operators(input: &str) -> String {
    let mut result = String::with_capacity(input.len());

    for line in input.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            // Full-line comment — leave as-is
            result.push_str(line);
        } else {
            result.push_str(&fix_operator_spacing(line));
        }
        result.push('\n');
    }

    if !input.is_empty() && !input.ends_with('\n') {
        result.pop();
    }
    result
}

/// Process a single line's operator spacing. Handles string/comment avoidance.
fn fix_operator_spacing(line: &str) -> String {
    // We walk the line char-by-char building an output buffer.
    // State: inside_string (char = quote char), inside_comment, fstring_depth.
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut out: Vec<char> = Vec::with_capacity(n + 8);

    let mut i = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut in_comment = false;
    let mut in_fstring_interp = 0u32; // nesting depth of `{...}` inside f-strings

    while i < n {
        let ch = chars[i];

        // --- Track comment start ---
        if !in_string && ch == '#' {
            in_comment = true;
        }

        if in_comment {
            out.push(ch);
            i += 1;
            continue;
        }

        // --- Track string boundaries ---
        if !in_string && (ch == '"' || ch == '\'') {
            in_string = true;
            string_char = ch;
            out.push(ch);
            i += 1;
            continue;
        }

        if in_string {
            if ch == '\\' && i + 1 < n {
                // Escape sequence — pass both chars through unchanged
                out.push(ch);
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if ch == '{' && string_char == '"' {
                // Possibly f-string interpolation — track depth but don't format inside
                in_fstring_interp += 1;
                out.push(ch);
                i += 1;
                continue;
            }
            if ch == '}' && in_fstring_interp > 0 {
                in_fstring_interp -= 1;
                out.push(ch);
                i += 1;
                continue;
            }
            if ch == string_char && in_fstring_interp == 0 {
                in_string = false;
            }
            out.push(ch);
            i += 1;
            continue;
        }

        // --- Outside strings and comments: operator spacing ---
        // Multi-char operators first (order matters: check longer sequences first)
        // Operators: |>, ==, !=, <=, >=, +=, -=, *=, /=, **, <, >, +, -, *, /, =

        // Skip ** entirely (exponentiation — don't add spaces)
        if ch == '*' && i + 1 < n && chars[i + 1] == '*' {
            out.push('*');
            out.push('*');
            i += 2;
            continue;
        }

        // Handle `->` (return type arrow) — keep as single token with spaces around
        if ch == '-' && i + 1 < n && chars[i + 1] == '>' {
            ensure_space_before(&mut out);
            out.push('-');
            out.push('>');
            i += 2;
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            if i < n {
                out.push(' ');
            }
            continue;
        }

        // Two-char operators
        let two_char_op: Option<&str> = if i + 1 < n {
            let two = &line[char_byte_pos(line, i)..char_byte_pos(line, i + 2).min(line.len())];
            match two {
                "|>" | "==" | "!=" | "<=" | ">=" | "+=" | "-=" | "*=" | "/=" => Some(two),
                _ => None,
            }
        } else {
            None
        };

        if let Some(op) = two_char_op {
            ensure_space_before(&mut out);
            for c in op.chars() {
                out.push(c);
            }
            // consume both chars, then ensure space after
            i += 2;
            // skip any existing spaces after
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            if i < n && chars[i] != '\n' {
                out.push(' ');
            }
            continue;
        }

        // Single-char binary operators
        if matches!(ch, '+' | '*' | '/' | '<' | '>') {
            ensure_space_before(&mut out);
            out.push(ch);
            i += 1;
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            if i < n {
                out.push(' ');
            }
            continue;
        }

        // `=` — but only as assignment/comparison, not part of ==, !=, <=, >=, +=, -=, *=, /=
        // Already handled two-char operators above, so here `=` is a plain assignment.
        if ch == '=' {
            // Peek ahead — if next is `=`, this is == (but we already handled that above)
            // Peek behind in out — if last non-space is !, <, >, +, -, *, /, that means we're
            // in the middle of a two-char op (shouldn't happen here, but guard anyway)
            let prev_meaningful = last_meaningful_char(&out);
            if matches!(prev_meaningful, Some('!') | Some('<') | Some('>') | Some('+') | Some('-') | Some('*') | Some('/')) {
                // Part of a compound operator already emitted — just push
                out.push(ch);
                i += 1;
                continue;
            }
            ensure_space_before(&mut out);
            out.push(ch);
            i += 1;
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            if i < n {
                out.push(' ');
            }
            continue;
        }

        // `-` — binary or unary?
        if ch == '-' {
            if is_unary_context(&out) {
                // Unary minus — don't add spaces
                out.push(ch);
                i += 1;
                continue;
            }
            // Binary minus
            ensure_space_before(&mut out);
            out.push(ch);
            i += 1;
            while i < n && chars[i] == ' ' {
                i += 1;
            }
            if i < n {
                out.push(' ');
            }
            continue;
        }

        // Regular character
        out.push(ch);
        i += 1;
    }

    out.iter().collect()
}

/// Get the byte position of the i-th char in a string.
fn char_byte_pos(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// Ensure the output buffer ends with exactly one space (trimming extras, adding if absent).
fn ensure_space_before(out: &mut Vec<char>) {
    // Remove trailing spaces
    while out.last() == Some(&' ') {
        out.pop();
    }
    // Add one space if the buffer is non-empty and the last char isn't an open bracket/start
    if let Some(&last) = out.last() {
        if last != '(' && last != '[' && last != '{' {
            out.push(' ');
        }
    }
}

/// Return the last non-space character in the buffer.
fn last_meaningful_char(out: &[char]) -> Option<char> {
    out.iter().rev().find(|&&c| c != ' ').copied()
}

/// Determine if the current position in the output buffer is a unary context.
/// Unary minus is preceded by: start of expression, `=`, `(`, `,`, `[`, `return`, keyword.
fn is_unary_context(out: &[char]) -> bool {
    let lm = last_meaningful_char(out);
    match lm {
        None => true, // start of line
        Some(c) => matches!(c, '=' | '(' | ',' | '[' | '{' | '+' | '-' | '*' | '/' | '<' | '>' | '!'),
    }
}

/// Replace single-quoted strings with double-quoted strings. Skips:
/// - Content inside comments
/// - Apostrophes within words (e.g., `don't`)
/// - Already double-quoted strings
fn normalize_quotes(input: &str) -> String {
    let mut result = String::with_capacity(input.len());

    for line in input.lines() {
        result.push_str(&fix_quotes_in_line(line));
        result.push('\n');
    }

    if !input.is_empty() && !input.ends_with('\n') {
        result.pop();
    }
    result
}

fn fix_quotes_in_line(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut out = Vec::with_capacity(n);

    let mut i = 0;
    let mut in_double_string = false;
    let mut in_comment = false;

    while i < n {
        let ch = chars[i];

        if !in_double_string && ch == '#' {
            in_comment = true;
        }

        if in_comment {
            out.push(ch);
            i += 1;
            continue;
        }

        if in_double_string {
            if ch == '\\' && i + 1 < n {
                out.push(ch);
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if ch == '"' {
                in_double_string = false;
            }
            out.push(ch);
            i += 1;
            continue;
        }

        // Outside any string
        if ch == '"' {
            in_double_string = true;
            out.push(ch);
            i += 1;
            continue;
        }

        if ch == '\'' {
            // Check if this is an apostrophe within a word (e.g., don't, it's)
            // Apostrophe: preceded by a letter AND followed by a letter
            let prev_is_letter = out.last().map(|c| c.is_alphabetic()).unwrap_or(false);
            let next_is_letter = chars.get(i + 1).map(|c| c.is_alphabetic()).unwrap_or(false);
            if prev_is_letter && next_is_letter {
                // Apostrophe in a word — don't touch
                out.push(ch);
                i += 1;
                continue;
            }

            // This is a single-quoted string — convert to double-quoted
            // Collect the string content, watching for the closing quote and escape sequences
            out.push('"');
            i += 1;
            while i < n {
                let sc = chars[i];
                if sc == '\\' && i + 1 < n {
                    out.push(sc);
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                if sc == '"' {
                    // Need to escape a literal double-quote inside the new string
                    out.push('\\');
                    out.push('"');
                    i += 1;
                    continue;
                }
                if sc == '\'' {
                    // Closing quote found
                    out.push('"');
                    i += 1;
                    break;
                }
                out.push(sc);
                i += 1;
            }
            continue;
        }

        out.push(ch);
        i += 1;
    }

    out.iter().collect()
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
        let input = "let s = 'hello'\n";
        let output = normalize_quotes(input);
        assert_eq!(output, "let s = \"hello\"\n");
    }

    #[test]
    fn test_normalize_quotes_apostrophe_preserved() {
        let input = "# don't touch this\n";
        let output = normalize_quotes(input);
        assert_eq!(output, "# don't touch this\n");
    }

    #[test]
    fn test_operator_spacing_basic() {
        let result = fix_operator_spacing("let x=1+2");
        assert_eq!(result, "let x = 1 + 2");
    }

    #[test]
    fn test_operator_spacing_comparison() {
        let result = fix_operator_spacing("if x==y");
        assert_eq!(result, "if x == y");
    }

    #[test]
    fn test_operator_spacing_pipe() {
        let result = fix_operator_spacing("x|>f");
        assert_eq!(result, "x |> f");
    }

    #[test]
    fn test_operator_spacing_exponent_untouched() {
        let result = fix_operator_spacing("x**2");
        assert_eq!(result, "x**2");
    }

    #[test]
    fn test_unary_minus_untouched() {
        let result = fix_operator_spacing("let x = -1");
        // -1 is unary (after `=`)
        assert!(result.contains("-1"));
    }
}
