use nsl_errors::{BytePos, FileId, Span};

/// Low-level character cursor over source text.
pub struct Cursor<'a> {
    source: &'a str,
    pos: usize,
    file_id: FileId,
}

impl<'a> Cursor<'a> {
    pub fn new(source: &'a str, file_id: FileId) -> Self {
        Self {
            source,
            pos: 0,
            file_id,
        }
    }

    pub fn pos(&self) -> BytePos {
        BytePos(self.pos as u32)
    }

    pub fn file_id(&self) -> FileId {
        self.file_id
    }

    pub fn is_eof(&self) -> bool {
        self.pos >= self.source.len()
    }

    /// Peek at the current character without advancing.
    pub fn peek(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    /// Peek at the character `n` positions ahead.
    pub fn peek_at(&self, n: usize) -> Option<char> {
        self.source[self.pos..].chars().nth(n)
    }

    /// Advance by one character and return it.
    pub fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    /// Advance if the current character matches `expected`.
    pub fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Advance while the predicate holds, returning the consumed slice.
    pub fn eat_while(&mut self, pred: impl Fn(char) -> bool) -> &'a str {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if pred(ch) {
                self.advance();
            } else {
                break;
            }
        }
        &self.source[start..self.pos]
    }

    /// Get a span from `start` to the current position.
    pub fn span_from(&self, start: BytePos) -> Span {
        Span::new(self.file_id, start, self.pos())
    }

    /// Get the remaining source from current position.
    pub fn remaining(&self) -> &'a str {
        &self.source[self.pos..]
    }

    /// Check if remaining source starts with a string.
    pub fn starts_with(&self, s: &str) -> bool {
        self.remaining().starts_with(s)
    }

    /// Skip `n` bytes forward.
    pub fn skip_bytes(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.source.len());
    }
}
