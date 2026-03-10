pub mod block;
pub mod decl;
pub mod expr;
pub mod parser;
pub mod pattern;
pub mod pratt;
pub mod stmt;
pub mod types;

use nsl_ast::Module;
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::{Interner, Token};

pub struct ParseResult {
    pub module: Module,
    pub diagnostics: Vec<Diagnostic>,
}

/// Parse a stream of tokens into an AST Module.
pub fn parse(tokens: &[Token], interner: &mut Interner) -> ParseResult {
    let mut p = parser::Parser::new(tokens, interner);
    p.skip_newlines();

    let mut stmts = Vec::new();
    while !p.at(&nsl_lexer::TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&nsl_lexer::TokenKind::Eof) {
            break;
        }
        stmts.push(stmt::parse_stmt(&mut p));
        p.skip_newlines();
    }

    let span = if stmts.is_empty() {
        Span::dummy()
    } else {
        stmts.first().unwrap().span.merge(stmts.last().unwrap().span)
    };

    ParseResult {
        module: Module { stmts, span },
        diagnostics: p.diagnostics,
    }
}
