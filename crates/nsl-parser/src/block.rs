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
        sections.push(TrainSection::Stmt(Box::new(stmt)));
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

    // Expect 'static' (an identifier, not a keyword)
    let kind = if let TokenKind::Ident(sym) = p.peek().clone() {
        let text = p.interner.resolve(sym).unwrap_or("").to_string();
        if text == "static" {
            p.advance();
            QuantKind::Static
        } else {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error(format!("expected 'static', found '{text}'"))
                    .with_label(p.current_span(), "expected 'static'"),
            );
            p.advance();
            QuantKind::Static
        }
    } else {
        p.diagnostics.push(
            nsl_errors::Diagnostic::error(format!("expected 'static', found {}", p.peek()))
                .with_label(p.current_span(), "expected 'static'"),
        );
        p.advance();
        QuantKind::Static
    };

    // Parse output name
    let (name, _) = p.expect_ident();

    // Expect 'from' keyword
    p.expect(&TokenKind::From);

    // Parse source model name
    let (source, _) = p.expect_ident();

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut default_dtype: Option<QuantDtype> = None;
    let mut default_granularity: Option<QuantGranularity> = None;
    let mut exclude: Vec<String> = Vec::new();
    let mut calibration: Option<CalibrationConfig> = None;

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        if let TokenKind::Ident(sym) = p.peek().clone() {
            let key = p.interner.resolve(sym).unwrap_or("").to_string();
            match key.as_str() {
                "default" => {
                    p.advance(); // consume 'default'
                    p.expect(&TokenKind::Colon);
                    // Parse dtype identifier
                    if let TokenKind::Ident(dtype_sym) = p.peek().clone() {
                        let dtype_text = p.interner.resolve(dtype_sym).unwrap_or("").to_string();
                        match dtype_text.as_str() {
                            "int4" => { p.advance(); default_dtype = Some(QuantDtype::Int4); }
                            "int8" => { p.advance(); default_dtype = Some(QuantDtype::Int8); }
                            _ => {
                                p.diagnostics.push(
                                    nsl_errors::Diagnostic::error(format!(
                                        "expected 'int4' or 'int8', found '{dtype_text}'"
                                    ))
                                    .with_label(p.current_span(), "expected quant dtype"),
                                );
                                p.advance();
                            }
                        }
                    }
                    // Optional comma + granularity
                    if p.eat(&TokenKind::Comma) {
                        default_granularity = Some(parse_quant_granularity(p));
                    }
                    p.expect_end_of_stmt();
                }
                "exclude" => {
                    p.advance(); // consume 'exclude'
                    p.expect(&TokenKind::Colon);
                    p.expect(&TokenKind::LeftBracket);
                    loop {
                        if p.at(&TokenKind::RightBracket) {
                            break;
                        }
                        if let TokenKind::StringLiteral(s) = p.peek().clone() {
                            exclude.push(s);
                            p.advance();
                        } else {
                            p.diagnostics.push(
                                nsl_errors::Diagnostic::error(format!(
                                    "expected string literal in exclude list, found {}",
                                    p.peek()
                                ))
                                .with_label(p.current_span(), "expected string"),
                            );
                            break;
                        }
                        if !p.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    p.expect(&TokenKind::RightBracket);
                    p.expect_end_of_stmt();
                }
                "calibration" => {
                    p.advance(); // consume 'calibration'
                    p.expect(&TokenKind::Colon);
                    p.skip_newlines();
                    p.expect(&TokenKind::Indent);
                    p.skip_newlines();

                    let mut cal_data: Option<nsl_ast::Symbol> = None;
                    let mut cal_samples: i64 = 0;

                    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
                        p.skip_newlines();
                        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
                            break;
                        }
                        if let TokenKind::Ident(cal_sym) = p.peek().clone() {
                            let cal_key = p.interner.resolve(cal_sym).unwrap_or("").to_string();
                            match cal_key.as_str() {
                                "data" => {
                                    p.advance();
                                    p.expect(&TokenKind::Colon);
                                    let (data_sym, _) = p.expect_ident();
                                    cal_data = Some(data_sym);
                                    p.expect_end_of_stmt();
                                }
                                "samples" => {
                                    p.advance();
                                    p.expect(&TokenKind::Colon);
                                    if let TokenKind::IntLiteral(n) = p.peek().clone() {
                                        cal_samples = n;
                                        p.advance();
                                    } else {
                                        p.diagnostics.push(
                                            nsl_errors::Diagnostic::error(format!(
                                                "expected integer literal, found {}",
                                                p.peek()
                                            ))
                                            .with_label(p.current_span(), "expected integer"),
                                        );
                                    }
                                    p.expect_end_of_stmt();
                                }
                                _ => {
                                    p.diagnostics.push(
                                        nsl_errors::Diagnostic::error(format!(
                                            "unexpected calibration key '{cal_key}'"
                                        ))
                                        .with_label(p.current_span(), "unexpected key"),
                                    );
                                    p.advance();
                                }
                            }
                        } else {
                            p.advance();
                        }
                    }
                    p.eat(&TokenKind::Dedent);

                    if let Some(data) = cal_data {
                        calibration = Some(CalibrationConfig {
                            data,
                            samples: cal_samples,
                        });
                    }
                }
                _ => {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error(format!(
                            "unexpected quant config key '{key}'"
                        ))
                        .with_label(p.current_span(), "unexpected key"),
                    );
                    p.advance();
                    p.expect_end_of_stmt();
                }
            }
        } else {
            p.advance();
        }
    }
    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::QuantBlock(QuantBlock {
            kind,
            name,
            source,
            default_dtype,
            default_granularity,
            exclude,
            calibration,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_quant_granularity(p: &mut Parser) -> QuantGranularity {
    if let TokenKind::Ident(sym) = p.peek().clone() {
        let text = p.interner.resolve(sym).unwrap_or("").to_string();
        match text.as_str() {
            "per_tensor" => {
                p.advance();
                QuantGranularity::PerTensor
            }
            "per_channel" => {
                p.advance();
                if p.eat(&TokenKind::LeftParen) {
                    let axis = if let TokenKind::IntLiteral(n) = p.peek().clone() {
                        p.advance();
                        n
                    } else {
                        0
                    };
                    p.expect(&TokenKind::RightParen);
                    QuantGranularity::PerChannel(axis)
                } else {
                    QuantGranularity::PerChannel(0)
                }
            }
            "per_group" => {
                p.advance();
                p.expect(&TokenKind::LeftParen);
                let first = if let TokenKind::IntLiteral(n) = p.peek().clone() {
                    p.advance();
                    n
                } else {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error(format!(
                            "expected group size integer, found {}",
                            p.peek()
                        ))
                        .with_label(p.current_span(), "expected integer"),
                    );
                    128
                };
                // per_group(size) => axis=0, group_size=size
                // per_group(axis, size) => both specified
                let (axis, group_size) = if p.eat(&TokenKind::Comma) {
                    let second = if let TokenKind::IntLiteral(n) = p.peek().clone() {
                        p.advance();
                        n
                    } else {
                        128
                    };
                    (first, second)
                } else {
                    (0, first)
                };
                p.expect(&TokenKind::RightParen);
                QuantGranularity::PerGroup(axis, group_size)
            }
            _ => {
                p.diagnostics.push(
                    nsl_errors::Diagnostic::error(format!(
                        "expected granularity (per_tensor, per_channel, per_group), found '{text}'"
                    ))
                    .with_label(p.current_span(), "expected granularity"),
                );
                p.advance();
                QuantGranularity::PerTensor
            }
        }
    } else {
        p.diagnostics.push(
            nsl_errors::Diagnostic::error(format!(
                "expected granularity identifier, found {}",
                p.peek()
            ))
            .with_label(p.current_span(), "expected granularity"),
        );
        QuantGranularity::PerTensor
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
