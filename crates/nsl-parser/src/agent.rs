//! M56: Agent declaration parser. Mirrors `parse_model_def_stmt`
//! (see decl.rs:56-188) — adapted to emit `StmtKind::AgentDef(AgentDef)`
//! with `AgentMember::{FieldDecl, Method}` members.

use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::decl::{Decorator, FnDef};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::TokenKind;

use crate::decl::{parse_params, parse_type_params};
use crate::expr::{parse_args, parse_expr};
use crate::parser::Parser;
use crate::types::{parse_type, parse_type_no_borrow};

pub fn parse_agent_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'agent'

    let (name, _) = p.expect_ident();

    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    // Constructor parameters (optional)
    let params = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let ps = parse_params(p);
        p.expect(&TokenKind::RightParen);
        ps
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut members = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }

        // Collect decorators (@shared, @auto_device_transfer, etc.)
        let mut decorators = Vec::new();
        while p.at(&TokenKind::At) {
            let dec_start = p.current_span();
            p.advance(); // @
            let mut dec_name = Vec::new();
            let (first, _) = p.expect_ident();
            dec_name.push(first);
            while p.eat(&TokenKind::Dot) {
                let (seg, _) = p.expect_ident();
                dec_name.push(seg);
            }
            let args = if p.at(&TokenKind::LeftParen) {
                p.advance();
                let a = parse_args(p);
                p.expect(&TokenKind::RightParen);
                Some(a)
            } else {
                None
            };
            decorators.push(Decorator {
                name: dec_name,
                args,
                span: dec_start.merge(p.prev_span()),
            });
            p.skip_newlines();
        }

        if p.at(&TokenKind::Fn) || p.at(&TokenKind::Async) {
            let method_start = p.current_span();
            let is_async = p.eat(&TokenKind::Async);
            p.expect(&TokenKind::Fn);
            let (mname, _) = p.expect_ident();
            let mtype_params = if p.at(&TokenKind::Lt) {
                parse_type_params(p)
            } else {
                Vec::new()
            };
            p.expect(&TokenKind::LeftParen);
            let mparams = parse_params(p);
            p.expect(&TokenKind::RightParen);
            let return_type = if p.eat(&TokenKind::Arrow) {
                Some(parse_type_no_borrow(p))
            } else {
                None
            };
            p.expect(&TokenKind::Colon);
            let body = p.parse_block();
            members.push(AgentMember::Method(
                FnDef {
                    name: mname,
                    type_params: mtype_params,
                    effect_params: vec![],
                    params: mparams,
                    return_type,
                    return_effect: None,
                    body: body.clone(),
                    is_async,
                    span: method_start.merge(body.span),
                },
                decorators,
            ));
        } else {
            // Field declaration: name: Type = init_expr
            let member_start = p.current_span();
            let (fname, _) = p.expect_ident();
            p.expect(&TokenKind::Colon);
            let type_ann = parse_type(p);
            let init = if p.eat(&TokenKind::Eq) {
                Some(parse_expr(p))
            } else {
                None
            };
            p.expect_end_of_stmt();
            members.push(AgentMember::FieldDecl {
                name: fname,
                type_ann,
                init,
                decorators,
                span: member_start.merge(p.prev_span()),
            });
            continue; // decorators already consumed
        }
        p.skip_newlines();
    }

    p.eat(&TokenKind::Dedent);

    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::AgentDef(AgentDef {
            name,
            type_params,
            params,
            members,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}
