use nsl_ast::agent::AgentMember;
use nsl_ast::stmt::StmtKind;
use nsl_errors::FileId;

#[test]
fn parses_agent_with_field_and_method() {
    let src = "agent Drafter:\n    kv_cache: KvCache = empty()\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n";
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(lex_diags.is_empty(), "lex errors: {:?}", lex_diags);

    let result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        result.diagnostics.is_empty(),
        "parse errors: {:?}",
        result.diagnostics
    );

    let stmt = &result.module.stmts[0];
    let agent = match &stmt.kind {
        StmtKind::AgentDef(a) => a,
        other => panic!("expected AgentDef, got {:?}", other),
    };
    assert_eq!(agent.members.len(), 2);
    assert!(matches!(agent.members[0], AgentMember::FieldDecl { .. }));
    assert!(matches!(agent.members[1], AgentMember::Method(..)));
}
