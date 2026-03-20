use nsl_ast::decl::*;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::TokenKind;

use crate::expr::parse_expr;
use crate::parser::Parser;
use crate::types::parse_type;

pub fn parse_fn_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    let is_async = p.eat(&TokenKind::Async);
    p.expect(&TokenKind::Fn);

    let (name, _) = p.expect_ident();

    // Type parameters: <T, U>
    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    // Parameters: (a: int, b: float = 0.0)
    p.expect(&TokenKind::LeftParen);
    let params = parse_params(p);
    p.expect(&TokenKind::RightParen);

    // Return type: -> Type
    let return_type = if p.eat(&TokenKind::Arrow) {
        Some(parse_type(p))
    } else {
        None
    };

    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    let span = start.merge(body.span);
    Stmt {
        kind: StmtKind::FnDef(FnDef {
            name,
            type_params,
            params,
            return_type,
            body,
            is_async,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_model_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'model'

    let (name, _) = p.expect_ident();

    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    // Constructor parameters
    let params = if p.at(&TokenKind::LeftParen) {
        p.advance();
        let params = parse_params(p);
        p.expect(&TokenKind::RightParen);
        params
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

        // Collect decorators
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
                let args = crate::expr::parse_args(p);
                p.expect(&TokenKind::RightParen);
                Some(args)
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
            // Method — decorators are now preserved in the AST (M53: @real_time support)
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
                Some(parse_type(p))
            } else {
                None
            };
            p.expect(&TokenKind::Colon);
            let body = p.parse_block();
            members.push(ModelMember::Method(FnDef {
                name: mname,
                type_params: mtype_params,
                params: mparams,
                return_type,
                body: body.clone(),
                is_async,
                span: method_start.merge(body.span),
            }, decorators));
        } else {
            // Layer/field declaration: name: Type = init_expr
            let member_start = p.current_span();
            let (lname, _) = p.expect_ident();
            p.expect(&TokenKind::Colon);
            let type_ann = parse_type(p);
            let init = if p.eat(&TokenKind::Eq) {
                Some(parse_expr(p))
            } else {
                None
            };
            p.expect_end_of_stmt();
            members.push(ModelMember::LayerDecl {
                name: lname,
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
        kind: StmtKind::ModelDef(ModelDef {
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

pub fn parse_struct_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'struct'

    let (name, _) = p.expect_ident();
    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut fields = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        let field_start = p.current_span();
        let (fname, _) = p.expect_ident();
        p.expect(&TokenKind::Colon);
        let type_ann = parse_type(p);
        let default = if p.eat(&TokenKind::Eq) {
            Some(parse_expr(p))
        } else {
            None
        };
        p.expect_end_of_stmt();
        fields.push(StructField {
            name: fname,
            type_ann,
            default,
            span: field_start.merge(p.prev_span()),
        });
    }

    p.eat(&TokenKind::Dedent);
    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::StructDef(StructDef {
            name,
            type_params,
            fields,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_enum_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'enum'

    let (name, _) = p.expect_ident();
    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut variants = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        let var_start = p.current_span();
        let (vname, _) = p.expect_ident();

        let fields = if p.at(&TokenKind::LeftParen) {
            p.advance();
            let mut types = Vec::new();
            while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
                types.push(parse_type(p));
                if !p.eat(&TokenKind::Comma) {
                    break;
                }
            }
            p.expect(&TokenKind::RightParen);
            types
        } else {
            Vec::new()
        };

        let value = if p.eat(&TokenKind::Eq) {
            Some(parse_expr(p))
        } else {
            None
        };

        p.expect_end_of_stmt();
        variants.push(EnumVariant {
            name: vname,
            fields,
            value,
            span: var_start.merge(p.prev_span()),
        });
    }

    p.eat(&TokenKind::Dedent);
    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::EnumDef(EnumDef {
            name,
            type_params,
            variants,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_trait_def_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'trait'

    let (name, _) = p.expect_ident();
    let type_params = if p.at(&TokenKind::Lt) {
        parse_type_params(p)
    } else {
        Vec::new()
    };

    p.expect(&TokenKind::Colon);
    p.skip_newlines();
    p.expect(&TokenKind::Indent);
    p.skip_newlines();

    let mut methods = Vec::new();
    while !p.at(&TokenKind::Dedent) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::Dedent) || p.at(&TokenKind::Eof) {
            break;
        }
        let method_start = p.current_span();
        let is_async = p.eat(&TokenKind::Async);
        p.expect(&TokenKind::Fn);
        let (mname, _) = p.expect_ident();
        p.expect(&TokenKind::LeftParen);
        let params = parse_params(p);
        p.expect(&TokenKind::RightParen);
        let return_type = if p.eat(&TokenKind::Arrow) {
            Some(parse_type(p))
        } else {
            None
        };
        // Trait methods may have a body (default impl) or not
        let body = if p.at(&TokenKind::Colon) {
            p.advance();
            p.parse_block()
        } else {
            p.expect_end_of_stmt();
            nsl_ast::stmt::Block {
                stmts: Vec::new(),
                span: method_start,
            }
        };
        methods.push(FnDef {
            name: mname,
            type_params: Vec::new(),
            params,
            return_type,
            body,
            is_async,
            span: method_start.merge(p.prev_span()),
        });
    }

    p.eat(&TokenKind::Dedent);
    let span = start.merge(p.prev_span());
    Stmt {
        kind: StmtKind::TraitDef(TraitDef {
            name,
            type_params,
            methods,
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}

pub fn parse_import_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'import'

    let mut path = Vec::new();
    let (first, _) = p.expect_ident_or_keyword();
    path.push(first);
    while p.eat(&TokenKind::Dot) {
        // Check for glob: import nsl.nn.*
        if p.at(&TokenKind::Star) {
            p.advance();
            p.expect_end_of_stmt();
            return Stmt {
                kind: StmtKind::Import(ImportStmt {
                    path,
                    items: ImportItems::Glob,
                    alias: None,
                    span: start.merge(p.prev_span()),
                }),
                span: start.merge(p.prev_span()),
                id: p.next_node_id(),
            };
        }
        // Check for named imports: import nsl.nn.{Linear, Conv2d}
        if p.at(&TokenKind::LeftBrace) {
            let items = parse_import_items(p);
            p.expect_end_of_stmt();
            return Stmt {
                kind: StmtKind::Import(ImportStmt {
                    path,
                    items: ImportItems::Named(items),
                    alias: None,
                    span: start.merge(p.prev_span()),
                }),
                span: start.merge(p.prev_span()),
                id: p.next_node_id(),
            };
        }
        let (seg, _) = p.expect_ident_or_keyword();
        path.push(seg);
    }

    // Check for alias: import nsl.math as math
    let alias = if p.eat(&TokenKind::As) {
        let (a, _) = p.expect_ident();
        Some(a)
    } else {
        None
    };

    // Simple module import: import nsl.nn  (or import nsl.math as math)
    p.expect_end_of_stmt();
    Stmt {
        kind: StmtKind::Import(ImportStmt {
            path,
            items: ImportItems::Module,
            alias,
            span: start.merge(p.prev_span()),
        }),
        span: start.merge(p.prev_span()),
        id: p.next_node_id(),
    }
}

pub fn parse_from_import_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'from'

    let mut module_path = Vec::new();
    let (first, _) = p.expect_ident_or_keyword();
    module_path.push(first);
    while p.eat(&TokenKind::Dot) {
        let (seg, _) = p.expect_ident_or_keyword();
        module_path.push(seg);
    }

    p.expect(&TokenKind::Import);

    let items = if p.at(&TokenKind::Star) {
        p.advance();
        ImportItems::Glob
    } else if p.at(&TokenKind::LeftBrace) {
        ImportItems::Named(parse_import_items(p))
    } else {
        // from module import Name [, Name2, ...]
        let mut items = Vec::new();
        loop {
            let (name, name_span) = p.expect_ident();
            let alias = if p.eat(&TokenKind::As) {
                let (a, _) = p.expect_ident();
                Some(a)
            } else {
                None
            };
            items.push(ImportItem {
                name,
                alias,
                span: name_span.merge(p.prev_span()),
            });
            if !p.eat(&TokenKind::Comma) {
                break;
            }
        }
        ImportItems::Named(items)
    };

    p.expect_end_of_stmt();
    Stmt {
        kind: StmtKind::FromImport(FromImportStmt {
            module_path,
            items,
            span: start.merge(p.prev_span()),
        }),
        span: start.merge(p.prev_span()),
        id: p.next_node_id(),
    }
}

fn parse_import_items(p: &mut Parser) -> Vec<ImportItem> {
    p.advance(); // consume {
    let mut items = Vec::new();

    while !p.at(&TokenKind::RightBrace) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::RightBrace) {
            break;
        }
        let item_start = p.current_span();
        let (name, _) = p.expect_ident();
        let alias = if p.eat(&TokenKind::As) {
            let (a, _) = p.expect_ident();
            Some(a)
        } else {
            None
        };
        items.push(ImportItem {
            name,
            alias,
            span: item_start.merge(p.prev_span()),
        });
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    p.skip_newlines();
    p.expect(&TokenKind::RightBrace);
    items
}

fn parse_type_params(p: &mut Parser) -> Vec<TypeParam> {
    p.advance(); // consume <
    let mut params = Vec::new();

    while !p.at(&TokenKind::Gt) && !p.at(&TokenKind::Eof) {
        let param_start = p.current_span();
        let (name, _) = p.expect_ident();

        // Parse optional trait bounds: `T: Clone + Debug`
        // Trait bounds are parsed but not yet enforced by the semantic checker.
        let mut bounds = Vec::new();
        if p.eat(&TokenKind::Colon) {
            // Parse first bound (a type expression, e.g. `Clone` or `Comparable<T>`)
            bounds.push(parse_type(p));
            // Parse additional bounds separated by `+`
            while p.eat(&TokenKind::Plus) {
                bounds.push(parse_type(p));
            }
        }

        params.push(TypeParam {
            name,
            bounds,
            span: param_start.merge(p.prev_span()),
        });
        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    p.expect(&TokenKind::Gt);
    params
}

pub fn parse_params(p: &mut Parser) -> Vec<Param> {
    let mut params = Vec::new();

    while !p.at(&TokenKind::RightParen) && !p.at(&TokenKind::Eof) {
        p.skip_newlines();
        if p.at(&TokenKind::RightParen) {
            break;
        }

        let param_start = p.current_span();
        let is_variadic = p.eat(&TokenKind::Star);
        // Handle `self` as a valid parameter name
        let (name, _) = if p.at(&TokenKind::SelfKw) {
            let span = p.advance().span;
            (p.intern("self"), span)
        } else {
            p.expect_ident()
        };
        let type_ann = if p.eat(&TokenKind::Colon) {
            Some(parse_type(p))
        } else {
            None
        };
        let default = if p.eat(&TokenKind::Eq) {
            Some(parse_expr(p))
        } else {
            None
        };

        params.push(Param {
            name,
            type_ann,
            default,
            is_variadic,
            span: param_start.merge(p.prev_span()),
        });

        if !p.eat(&TokenKind::Comma) {
            break;
        }
    }

    p.skip_newlines();
    params
}
