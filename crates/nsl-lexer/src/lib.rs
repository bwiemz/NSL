pub mod cursor;
pub mod indent;
pub mod keywords;
pub mod lexer;
pub mod numbers;
pub mod strings;
pub mod token;

pub use token::{Token, TokenKind, Symbol};

use nsl_errors::{Diagnostic, FileId};
use string_interner::StringInterner;

pub type Interner = StringInterner<string_interner::backend::BucketBackend<string_interner::DefaultSymbol>>;

/// Tokenize NSL source code into a stream of tokens.
pub fn tokenize(
    source: &str,
    file_id: FileId,
    interner: &mut Interner,
) -> (Vec<Token>, Vec<Diagnostic>) {
    let lexer = lexer::Lexer::new(source, file_id, interner);
    lexer.tokenize()
}
