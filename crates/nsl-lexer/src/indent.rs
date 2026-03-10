use crate::token::{Token, TokenKind};
use nsl_errors::Span;

/// Tracks indentation levels and emits INDENT/DEDENT tokens.
pub struct IndentTracker {
    /// Stack of indentation levels. Always starts with [0].
    stack: Vec<u32>,
    /// Bracket nesting depth. When > 0, indentation is suppressed.
    bracket_depth: u32,
}

impl IndentTracker {
    pub fn new() -> Self {
        Self {
            stack: vec![0],
            bracket_depth: 0,
        }
    }

    pub fn open_bracket(&mut self) {
        self.bracket_depth += 1;
    }

    pub fn close_bracket(&mut self) {
        self.bracket_depth = self.bracket_depth.saturating_sub(1);
    }

    pub fn in_brackets(&self) -> bool {
        self.bracket_depth > 0
    }

    /// Process the indentation at the start of a new logical line.
    /// Returns tokens to emit (Indent, Dedent, or nothing).
    /// Returns `Err(message)` if the indentation level is invalid.
    pub fn process_indent(&mut self, level: u32, span: Span) -> Result<Vec<Token>, String> {
        if self.in_brackets() {
            return Ok(Vec::new());
        }

        let current = *self.stack.last().unwrap();

        if level > current {
            self.stack.push(level);
            Ok(vec![Token {
                kind: TokenKind::Indent,
                span,
            }])
        } else if level == current {
            Ok(Vec::new())
        } else {
            // Dedent: pop until we find a matching level
            let mut tokens = Vec::new();
            while let Some(&top) = self.stack.last() {
                if top == level {
                    break;
                }
                if top < level {
                    return Err(format!(
                        "dedent does not match any outer indentation level (got {level}, stack: {:?})",
                        self.stack
                    ));
                }
                self.stack.pop();
                tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span,
                });
            }

            if *self.stack.last().unwrap() != level {
                return Err(format!(
                    "dedent does not match any outer indentation level (got {level})"
                ));
            }

            Ok(tokens)
        }
    }

    /// Emit remaining DEDENT tokens at EOF.
    pub fn finalize(&mut self, span: Span) -> Vec<Token> {
        let mut tokens = Vec::new();
        while self.stack.len() > 1 {
            self.stack.pop();
            tokens.push(Token {
                kind: TokenKind::Dedent,
                span,
            });
        }
        tokens
    }
}

impl Default for IndentTracker {
    fn default() -> Self {
        Self::new()
    }
}
