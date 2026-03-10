use nsl_ast::block::*;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::TokenKind;

use crate::expr::{parse_args, parse_expr};
use crate::parser::Parser;

pub fn parse_train_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'train'

    // Config args: train(model=m, epochs=10, precision=bf16):
    let config = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let args = parse_args(p);
        p.expect(&TokenKind::RightParen);
        args
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut sections = Vec::new();

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        // Check for section headers
        if let TokenKind::Ident(sym) = p.peek().clone() {
            let name = p.interner.resolve(sym).unwrap_or("").to_string();

            match name.as_str() {
                "data" if matches!(p.peek_at(1), &TokenKind::Colon) => {
                    p.advance(); // data
                    p.advance(); // :
                    let block = p.parse_block();
                    sections.push(TrainSection::Data(block.stmts));
                    continue;
                }
                "optimizer" if matches!(p.peek_at(1), &TokenKind::Colon) => {
                    p.advance(); // optimizer
                    p.advance(); // :
                    let expr = parse_expr(p);
                    p.expect_end_of_stmt();
                    sections.push(TrainSection::Optimizer(expr));
                    continue;
                }
                "scheduler" if matches!(p.peek_at(1), &TokenKind::Colon) => {
                    p.advance(); // scheduler
                    p.advance(); // :
                    let expr = parse_expr(p);
                    p.expect_end_of_stmt();
                    sections.push(TrainSection::Scheduler(expr));
                    continue;
                }
                "step" if matches!(p.peek_at(1), &TokenKind::LeftParen) => {
                    p.advance(); // step
                    p.advance(); // (
                    let (param, _) = p.expect_ident();
                    p.expect(&TokenKind::RightParen);
                    p.expect(&TokenKind::Colon);
                    let body = p.parse_block();
                    sections.push(TrainSection::Step { param, body });
                    continue;
                }
                "eval" if matches!(p.peek_at(1), &TokenKind::LeftParen) => {
                    p.advance(); // eval
                    p.advance(); // (
                    let (param, _) = p.expect_ident();
                    p.expect(&TokenKind::RightParen);
                    p.expect(&TokenKind::Colon);
                    let body = p.parse_block();
                    sections.push(TrainSection::Eval { param, body });
                    continue;
                }
                "callbacks" if matches!(p.peek_at(1), &TokenKind::Colon) => {
                    p.advance(); // callbacks
                    p.advance(); // :
                    let callbacks = parse_callbacks(p);
                    sections.push(TrainSection::Callbacks(callbacks));
                    continue;
                }
                "distribute" if matches!(p.peek_at(1), &TokenKind::Colon) => {
                    p.advance(); // distribute
                    p.advance(); // :
                    let expr = parse_expr(p);
                    p.expect_end_of_stmt();
                    sections.push(TrainSection::Distribute(expr));
                    continue;
                }
                _ => {}
            }
        }

        // Generic statement inside train block
        let stmt = crate::stmt::parse_stmt(p);
        sections.push(TrainSection::Stmt(stmt));
    }

    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::TrainBlock(TrainBlock {
            config,
            sections,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_callbacks(p: &mut Parser) -> Vec<CallbackDef> {
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut callbacks = Vec::new();

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        let cb_start = p.current_span();
        let (name, _) = p.expect_ident();
        p.expect(&TokenKind::LeftParen);
        let params = crate::decl::parse_params(p);
        p.expect(&TokenKind::RightParen);
        p.expect(&TokenKind::Colon);
        let body = p.parse_block();

        callbacks.push(CallbackDef {
            name,
            params,
            body: body.clone(),
            span: cb_start.merge(body.span),
        });
    }

    p.eat(&TokenKind::Dedent);
    callbacks
}

pub fn parse_grad_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'grad'

    p.expect(&TokenKind::LeftParen);
    let targets = parse_expr(p);
    p.expect(&TokenKind::RightParen);
    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    let span = start.merge(body.span);
    Stmt {
        kind: StmtKind::GradBlock(GradBlock {
            outputs: None,
            targets,
            body: body.clone(),
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_quant_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'quant'

    let config = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let args = parse_args(p);
        p.expect(&TokenKind::RightParen);
        args
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut stmts = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        stmts.push(crate::stmt::parse_stmt(p));
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::QuantBlock(QuantBlock {
            config,
            body: stmts,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_kernel_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'kernel'

    let (name, _) = p.expect_ident();
    p.expect(&TokenKind::LeftParen);
    let params = crate::decl::parse_params(p);
    p.expect(&TokenKind::RightParen);

    let return_type = if p.eat(&TokenKind::Arrow) {
        Some(crate::types::parse_type(p))
    } else {
        None
    };

    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    let span = start.merge(body.span);
    Stmt {
        kind: StmtKind::KernelDef(KernelDef {
            name,
            params,
            return_type,
            body: body.clone(),
            decorators: Vec::new(),
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_tokenizer_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'tokenizer'

    let (name, _) = p.expect_ident();

    let config = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let args = parse_args(p);
        p.expect(&TokenKind::RightParen);
        args
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut stmts = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        stmts.push(crate::stmt::parse_stmt(p));
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::TokenizerDef(TokenizerDef {
            name,
            config,
            body: stmts,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_dataset_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'dataset'

    let (name, _) = p.expect_ident();

    p.expect(&TokenKind::LeftParen);
    let source = parse_expr(p);
    p.expect(&TokenKind::RightParen);

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut stmts = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        stmts.push(crate::stmt::parse_stmt(p));
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::DatasetDef(DatasetDef {
            name,
            source,
            body: stmts,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}
