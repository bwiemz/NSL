use nsl_ast::block::*;
use nsl_ast::decl::Param;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::types::TypeExpr;
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
                            "awq4" => { p.advance(); default_dtype = Some(QuantDtype::Awq4); }
                            "gptq4" => { p.advance(); default_dtype = Some(QuantDtype::Gptq4); }
                            "gptq8" => { p.advance(); default_dtype = Some(QuantDtype::Gptq8); }
                            _ => {
                                p.diagnostics.push(
                                    nsl_errors::Diagnostic::error(format!(
                                        "expected quant dtype (int4, int8, awq4, gptq4, gptq8), found '{dtype_text}'"
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
    fn parse_key_value_entry(p: &mut Parser) -> KeyValueEntry {
        let start = p.current_span();
        let (key, _) = p.expect_ident();
        p.expect(&TokenKind::Eq);
        let value = parse_expr(p);
        let span = start.merge(value.span);
        KeyValueEntry { key, value, span }
    }

    fn parse_key_value_block(p: &mut Parser) -> Vec<KeyValueEntry> {
        p.skip_newlines();
        p.expect(&TokenKind::Indent);
        p.skip_newlines();

        let mut entries = Vec::new();
        while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
            p.skip_newlines();
            if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
                break;
            }
            if !matches!(p.peek(), TokenKind::Ident(_)) || !matches!(p.peek_at(1), &TokenKind::Eq) {
                p.diagnostics.push(
                    nsl_errors::Diagnostic::error("expected key = value entry")
                        .with_label(p.current_span(), "invalid tokenizer entry"),
                );
                p.synchronize();
                p.eat(&TokenKind::Newline);
                continue;
            }
            let entry = parse_key_value_entry(p);
            p.expect_end_of_stmt();
            entries.push(entry);
            p.skip_newlines();
        }
        p.eat(&TokenKind::Dedent);
        entries
    }

    fn parse_inline_key_values(p: &mut Parser) -> Vec<KeyValueEntry> {
        let mut entries = Vec::new();
        loop {
            let entry = parse_key_value_entry(p);
            entries.push(entry);
            if !p.eat(&TokenKind::Comma) {
                break;
            }
        }
        entries
    }

    fn parse_rule_list(p: &mut Parser) -> Vec<nsl_ast::Symbol> {
        let mut rules = Vec::new();
        loop {
            let (rule, _) = p.expect_ident();
            rules.push(rule);
            if !p.eat(&TokenKind::Comma) {
                break;
            }
        }
        rules
    }

    fn skip_unknown_tokenizer_section(p: &mut Parser) {
        if p.eat(&TokenKind::Newline) {
            p.skip_newlines();
            if p.at(&TokenKind::Indent) {
                let _ = p.parse_block();
                return;
            }
        }
        p.synchronize();
        p.eat(&TokenKind::Newline);
    }

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

        if let TokenKind::Ident(sym) = p.peek().clone() {
            let section_span = p.current_span();
            let section_name = p.interner.resolve(sym).unwrap_or("").to_string();
            p.advance();
            p.expect(&TokenKind::Colon);

            match section_name.as_str() {
                "special_tokens" => {
                    let entries = parse_key_value_block(p);
                    let span = section_span.merge(p.prev_span());
                    stmts.push(TokenizerStmt::SpecialTokens { entries, span });
                }
                "normalize" => {
                    let rules = parse_rule_list(p);
                    p.expect_end_of_stmt();
                    let span = section_span.merge(p.prev_span());
                    stmts.push(TokenizerStmt::Normalize { rules, span });
                }
                "pre_tokenize" => {
                    let rules = parse_rule_list(p);
                    p.expect_end_of_stmt();
                    let span = section_span.merge(p.prev_span());
                    stmts.push(TokenizerStmt::PreTokenize { rules, span });
                }
                "padding" => {
                    let entries = if p.eat(&TokenKind::Newline) {
                        parse_key_value_block(p)
                    } else {
                        let entries = parse_inline_key_values(p);
                        p.expect_end_of_stmt();
                        entries
                    };
                    let span = section_span.merge(p.prev_span());
                    stmts.push(TokenizerStmt::Padding { entries, span });
                }
                "truncation" => {
                    let entries = if p.eat(&TokenKind::Newline) {
                        parse_key_value_block(p)
                    } else {
                        let entries = parse_inline_key_values(p);
                        p.expect_end_of_stmt();
                        entries
                    };
                    let span = section_span.merge(p.prev_span());
                    stmts.push(TokenizerStmt::Truncation { entries, span });
                }
                _ => {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error(format!(
                            "unexpected tokenizer section '{section_name}'"
                        ))
                        .with_label(section_span, "unexpected tokenizer section"),
                    );
                    skip_unknown_tokenizer_section(p);
                }
            }
            continue;
        }

        p.diagnostics.push(
            nsl_errors::Diagnostic::error("expected tokenizer section")
                .with_label(p.current_span(), "invalid tokenizer body entry"),
        );
        p.synchronize();
        p.eat(&TokenKind::Newline);
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
        if !matches!(p.peek(), TokenKind::Ident(_)) || !matches!(p.peek_at(1), &TokenKind::Eq) {
            p.diagnostics.push(
                nsl_errors::Diagnostic::error("expected dataset field assignment")
                    .with_label(p.current_span(), "invalid dataset body entry"),
            );
            p.synchronize();
            p.eat(&TokenKind::Newline);
            continue;
        }

        let start = p.current_span();
        let (key, _) = p.expect_ident();
        p.expect(&TokenKind::Eq);
        let value = parse_expr(p);
        p.expect_end_of_stmt();
        let span = start.merge(value.span);
        stmts.push(KeyValueEntry { key, value, span });
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

pub fn parse_datatype_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'datatype'

    let (name, _) = p.expect_ident();
    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut bits: Option<u8> = None;
    let mut block_size: Option<u32> = None;
    let mut methods: Vec<DatatypeMethod> = Vec::new();
    let mut ptx_blocks: Vec<DatatypePtxBlock> = Vec::new();

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        // Check for @decorator
        if p.at(&TokenKind::At) {
            let dec_start = p.current_span();
            p.advance(); // consume @

            let (dec_sym, _) = p.expect_ident();
            let dec_name = p.interner.resolve(dec_sym.0).unwrap_or("").to_string();

            match dec_name.as_str() {
                "pack" | "unpack" => {
                    let kind = if dec_name == "pack" {
                        DatatypeMethodKind::Pack
                    } else {
                        DatatypeMethodKind::Unpack
                    };
                    let (params, return_type, body) = parse_datatype_method_body(p);
                    methods.push(DatatypeMethod {
                        kind,
                        params,
                        return_type,
                        body,
                        span: dec_start.merge(p.prev_span()),
                    });
                }
                "backward" => {
                    // @backward @pack compound decorator
                    p.expect(&TokenKind::At);
                    let (inner_sym, _) = p.expect_ident();
                    let inner_name = p.interner.resolve(inner_sym.0).unwrap_or("").to_string();
                    if inner_name != "pack" {
                        p.diagnostics.push(
                            nsl_errors::Diagnostic::error("expected @pack after @backward")
                                .with_label(p.prev_span(), "expected @pack"),
                        );
                    }
                    let (params, return_type, body) = parse_datatype_method_body(p);
                    methods.push(DatatypeMethod {
                        kind: DatatypeMethodKind::BackwardPack,
                        params,
                        return_type,
                        body,
                        span: dec_start.merge(p.prev_span()),
                    });
                }
                "pack_ptx" | "unpack_ptx" | "arithmetic_ptx" => {
                    let kind = match dec_name.as_str() {
                        "pack_ptx" => DatatypePtxKind::PackPtx,
                        "unpack_ptx" => DatatypePtxKind::UnpackPtx,
                        "arithmetic_ptx" => DatatypePtxKind::ArithmeticPtx,
                        _ => unreachable!(),
                    };
                    p.expect(&TokenKind::LeftParen);
                    let (_key_sym, _) = p.expect_ident(); // "ptx"
                    p.expect(&TokenKind::Eq);
                    // Read string literal via peek + match
                    let ptx_source = if let TokenKind::StringLiteral(s) = p.peek().clone() {
                        p.advance();
                        s
                    } else {
                        p.diagnostics.push(
                            nsl_errors::Diagnostic::error("expected PTX string literal")
                                .with_label(p.current_span(), "expected string"),
                        );
                        String::new()
                    };
                    p.expect(&TokenKind::RightParen);
                    p.skip_newlines();
                    ptx_blocks.push(DatatypePtxBlock {
                        kind,
                        ptx_source,
                        span: dec_start.merge(p.prev_span()),
                    });
                }
                other => {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error(format!(
                            "unknown datatype decorator @{other}; expected @pack, @unpack, \
                             @backward, @pack_ptx, @unpack_ptx, or @arithmetic_ptx"
                        ))
                        .with_label(dec_start, "unknown decorator"),
                    );
                }
            }
        } else if let TokenKind::Ident(sym) = p.peek().clone() {
            // Metadata: bits: N, block_size: N
            let ident = p.interner.resolve(sym).unwrap_or("").to_string();
            match ident.as_str() {
                "bits" => {
                    p.advance();
                    p.expect(&TokenKind::Colon);
                    if let TokenKind::IntLiteral(v) = p.peek().clone() {
                        bits = Some(v as u8);
                        p.advance();
                    }
                    p.expect_end_of_stmt();
                }
                "block_size" => {
                    p.advance();
                    p.expect(&TokenKind::Colon);
                    if let TokenKind::IntLiteral(v) = p.peek().clone() {
                        block_size = Some(v as u32);
                        p.advance();
                    }
                    p.expect_end_of_stmt();
                }
                _ => {
                    p.diagnostics.push(
                        nsl_errors::Diagnostic::error(format!(
                            "unexpected '{ident}' in datatype block; \
                             expected 'bits', 'block_size', or a @decorator"
                        ))
                        .with_label(p.current_span(), "unexpected"),
                    );
                    p.advance();
                }
            }
        } else {
            p.advance(); // skip unknown token
        }

        p.skip_newlines();
    }

    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::DatatypeDef(DatatypeDef {
            name,
            bits,
            block_size,
            methods,
            ptx_blocks,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_serve_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'serve'

    let (name, _) = p.expect_ident();

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut config = Vec::new();
    let mut endpoints = Vec::new();

    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        // @endpoint decorator followed by fn
        if p.at(&TokenKind::At) {
            let ep = parse_endpoint_def(p);
            endpoints.push(ep);
            continue;
        }

        // fn without @endpoint — still treat as endpoint
        if p.at(&TokenKind::Fn) {
            let ep = parse_endpoint_fn(p);
            endpoints.push(ep);
            continue;
        }

        // Config entry: key [: Type] = expr  OR  key: expr
        let entry = parse_serve_config_entry(p);
        config.push(entry);
    }

    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::ServeBlock(nsl_ast::block::ServeBlock {
            name,
            config,
            endpoints,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

fn parse_serve_config_entry(p: &mut Parser) -> nsl_ast::block::ServeConfigEntry {
    let start = p.current_span();
    let (key, _) = p.expect_ident();

    p.expect(&TokenKind::Colon);

    // Check if next is a type annotation followed by `=`
    let mut type_ann = None;
    if let TokenKind::Ident(_) = p.peek().clone() {
        if matches!(p.peek_at(1), &TokenKind::Eq) {
            type_ann = Some(crate::types::parse_type(p));
            p.expect(&TokenKind::Eq);
        }
    }

    let value = parse_expr(p);
    p.expect_end_of_stmt();

    let span = start.merge(p.prev_span());
    nsl_ast::block::ServeConfigEntry {
        key,
        type_ann,
        value,
        span,
    }
}

fn parse_endpoint_def(p: &mut Parser) -> nsl_ast::block::EndpointDef {
    p.advance(); // consume @
    let (dec_sym, dec_span) = p.expect_ident();
    let dec_str = p.interner.resolve(dec_sym.0).unwrap_or("").to_string();
    if dec_str != "endpoint" {
        p.diagnostics.push(
            nsl_errors::Diagnostic::error(format!("expected @endpoint, got @{dec_str}"))
                .with_label(dec_span, "expected @endpoint"),
        );
    }
    p.skip_newlines();
    parse_endpoint_fn(p)
}

fn parse_endpoint_fn(p: &mut Parser) -> nsl_ast::block::EndpointDef {
    let start = p.current_span();
    p.expect(&TokenKind::Fn);
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

    let span = start.merge(p.prev_span());
    nsl_ast::block::EndpointDef {
        name,
        params,
        return_type,
        body,
        span,
    }
}

/// Parse datatype method body: (params) -> return_type: \n INDENT body DEDENT
fn parse_datatype_method_body(p: &mut Parser) -> (Vec<Param>, Option<TypeExpr>, Vec<Stmt>) {
    p.expect(&TokenKind::LeftParen);
    let params = crate::decl::parse_params(p);
    p.expect(&TokenKind::RightParen);

    let return_type = if p.eat(&TokenKind::Arrow) {
        Some(crate::types::parse_type(p))
    } else {
        None
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);

    let mut body = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        body.push(crate::stmt::parse_stmt(p));
        p.skip_newlines();
    }
    p.eat(&TokenKind::Dedent);

    (params, return_type, body)
}

#[cfg(test)]
mod serve_tests {
    #[test]
    fn parse_serve_block() {
        let source = "serve Inference:\n    max_batch: 32\n    kv_blocks: 2048\n\n    @endpoint\n    fn generate(prompt: str) -> str:\n        return prompt\n";
        let mut interner = nsl_lexer::Interner::new();
        let file_id = nsl_errors::FileId(0);
        let (tokens, _lex_errs) = nsl_lexer::tokenize(source, file_id, &mut interner);
        let result = crate::parse(&tokens, &mut interner);
        assert_eq!(result.module.stmts.len(), 1);
        if let nsl_ast::stmt::StmtKind::ServeBlock(sb) = &result.module.stmts[0].kind {
            assert_eq!(sb.config.len(), 2);
            assert_eq!(sb.endpoints.len(), 1);
        } else {
            panic!("Expected ServeBlock");
        }
    }
}

#[cfg(test)]
mod tokenizer_dataset_tests {
    use nsl_ast::block::TokenizerStmt;

    #[test]
    fn parse_tokenizer_dsl_sections() {
        let source = "tokenizer tok(algorithm=bpe, vocab_size=1024):\n    special_tokens:\n        pad = \"<pad>\"\n        eos = \"<eos>\"\n\n    normalize: nfkc, lowercase\n    pre_tokenize: whitespace, byte_fallback\n\n    padding:\n        side = left\n        pad_to = longest\n\n    truncation:\n        max_length = 128\n        strategy = longest_first\n";
        let mut interner = nsl_lexer::Interner::new();
        let file_id = nsl_errors::FileId(0);
        let (tokens, _lex_errs) = nsl_lexer::tokenize(source, file_id, &mut interner);
        let result = crate::parse(&tokens, &mut interner);

        assert!(result.diagnostics.is_empty(), "unexpected parser diagnostics: {:?}", result.diagnostics);
        assert_eq!(result.module.stmts.len(), 1);

        if let nsl_ast::stmt::StmtKind::TokenizerDef(tok) = &result.module.stmts[0].kind {
            assert_eq!(tok.body.len(), 5);
            assert!(matches!(tok.body[0], TokenizerStmt::SpecialTokens { .. }));
            assert!(matches!(tok.body[1], TokenizerStmt::Normalize { .. }));
            assert!(matches!(tok.body[2], TokenizerStmt::PreTokenize { .. }));
            assert!(matches!(tok.body[3], TokenizerStmt::Padding { .. }));
            assert!(matches!(tok.body[4], TokenizerStmt::Truncation { .. }));
        } else {
            panic!("Expected TokenizerDef");
        }
    }

    #[test]
    fn parse_dataset_field_entries() {
        let source = "dataset train_data(\"demo\"):\n    source = \"data.bin\"\n    sequence_length = 128\n    packing = true\n    pack_separator = 0\n";
        let mut interner = nsl_lexer::Interner::new();
        let file_id = nsl_errors::FileId(0);
        let (tokens, _lex_errs) = nsl_lexer::tokenize(source, file_id, &mut interner);
        let result = crate::parse(&tokens, &mut interner);

        assert!(result.diagnostics.is_empty(), "unexpected parser diagnostics: {:?}", result.diagnostics);
        assert_eq!(result.module.stmts.len(), 1);

        if let nsl_ast::stmt::StmtKind::DatasetDef(ds) = &result.module.stmts[0].kind {
            assert_eq!(ds.body.len(), 4);
        } else {
            panic!("Expected DatasetDef");
        }
    }
}
