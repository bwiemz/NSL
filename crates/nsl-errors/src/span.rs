use serde::Serialize;

/// A unique identifier for a source file in the compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct FileId(pub usize);

/// A byte offset into a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize)]
pub struct BytePos(pub u32);

impl BytePos {
    pub fn offset(self, n: u32) -> BytePos {
        BytePos(self.0 + n)
    }
}

/// A span of source text: [start, end) byte range within a single file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Span {
    pub file_id: FileId,
    pub start: BytePos,
    pub end: BytePos,
}

impl Span {
    pub const DUMMY: Span = Span {
        file_id: FileId(0),
        start: BytePos(0),
        end: BytePos(0),
    };

    pub fn new(file_id: FileId, start: BytePos, end: BytePos) -> Self {
        Self { file_id, start, end }
    }

    pub fn dummy() -> Self {
        Self::DUMMY
    }

    pub fn merge(self, other: Span) -> Span {
        assert_eq!(self.file_id, other.file_id, "cannot merge spans from different files");
        Span {
            file_id: self.file_id,
            start: std::cmp::min(self.start, other.start),
            end: std::cmp::max(self.end, other.end),
        }
    }

    pub fn len(&self) -> u32 {
        self.end.0.saturating_sub(self.start.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
