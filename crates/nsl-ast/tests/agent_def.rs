use nsl_ast::agent::{AgentDef, AgentMember};
use nsl_ast::types::{TypeExpr, TypeExprKind};
use nsl_ast::{NodeId, Span, Symbol};

#[test]
fn agent_def_holds_fields_and_methods() {
    let dummy = Span::dummy();

    // Construct a Symbol via string_interner directly — nsl-ast already
    // depends on string_interner; nsl_lexer is not a dep of this crate.
    let mut interner = string_interner::StringInterner::<
        string_interner::backend::BucketBackend<string_interner::DefaultSymbol>,
    >::new();
    let name = Symbol(interner.get_or_intern("Drafter"));

    let def = AgentDef {
        name,
        type_params: vec![],
        params: vec![],
        members: vec![],
        span: dummy,
    };
    assert_eq!(def.members.len(), 0);

    // Verify FieldDecl variant is constructible with a Wildcard type annotation.
    let _ = AgentMember::FieldDecl {
        name,
        type_ann: TypeExpr {
            kind: TypeExprKind::Wildcard,
            span: dummy,
            id: NodeId::dummy(),
        },
        init: None,
        decorators: vec![],
        span: dummy,
    };
}
