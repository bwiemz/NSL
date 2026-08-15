//! Milestone A: one walk that finds EVERY decorator in a module, with the
//! construct it attaches to.
//!
//! The tree previously had no such walk: each consumer scanned for its own
//! decorator at its own position (`find_cpdt_on_train` matches module-level
//! `Decorated{TrainBlock}` only, the checker branches on stmt-level names,
//! WRGA reads model members), so "which decorators does this program use" had
//! no single answer — which is how `@totally_not_a_real_feature` compiled
//! silently. The namespace gate (nsl-semantic) and the activation reconciler
//! (nsl-cli) both need that answer, so the walk lives here, below both.
//!
//! The generic [`crate::visitor`] walk deliberately skips the member-level
//! decorator vectors (`ModelMember::Method(_, decos)`,
//! `AgentMember::Method(_, decos)`, `LayerDecl/FieldDecl.decorators`,
//! `KernelDef.decorators`) — it visits bodies, not annotations. This visitor
//! overrides `visit_stmt` to pick those up and then delegates to the generic
//! walk for recursion, so any host reachable by the generic walk is reachable
//! here (train-section bodies, fn bodies, nested blocks).

use crate::decl::{Decorator, ModelMember};
use crate::stmt::{Stmt, StmtKind};
use crate::Module;

/// What a decorator is attached to. The variant set mirrors the AST's actual
/// carrier positions, not a wished-for taxonomy: every variant here has a
/// concrete `Vec<Decorator>` in some node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecoratorHost {
    /// `@x` above a top-level or nested `fn`.
    Function,
    /// `@x` above a `model` declaration.
    Model,
    /// `@x` above a `train(...)` block.
    TrainBlock,
    /// `@x` above a `distill(...)` block.
    DistillBlock,
    /// `@x` above a `kernel` definition (the `KernelDef.decorators` field —
    /// distinct from a `Decorated` stmt wrapping the kernel).
    Kernel,
    /// `@x` above an `agent` declaration.
    Agent,
    /// `@x` on a `layer`/field declaration inside a `model`.
    ModelLayer,
    /// `@x` on a method inside a `model`.
    ModelMethod,
    /// `@x` on a field inside an `agent`.
    AgentField,
    /// `@x` on a method inside an `agent`.
    AgentMethod,
    /// `@x` above any other statement (a `let`, an expression, a `grad`
    /// block, ...). The checker's per-name position contracts decide which of
    /// these are legal; the walk only reports where the decorator sat.
    OtherStmt,
}

impl DecoratorHost {
    /// Human-readable position name for diagnostics ("a train block", ...).
    pub fn describe(&self) -> &'static str {
        match self {
            DecoratorHost::Function => "a function",
            DecoratorHost::Model => "a model",
            DecoratorHost::TrainBlock => "a train block",
            DecoratorHost::DistillBlock => "a distill block",
            DecoratorHost::Kernel => "a kernel",
            DecoratorHost::Agent => "an agent",
            DecoratorHost::ModelLayer => "a model layer/field",
            DecoratorHost::ModelMethod => "a model method",
            DecoratorHost::AgentField => "an agent field",
            DecoratorHost::AgentMethod => "an agent method",
            DecoratorHost::OtherStmt => "a statement",
        }
    }
}

/// One decorator occurrence: the node and what it attaches to.
#[derive(Debug, Clone, Copy)]
pub struct DecoratorUse<'a> {
    pub deco: &'a Decorator,
    pub host: DecoratorHost,
}

struct Collector<'a> {
    out: Vec<DecoratorUse<'a>>,
}

/// `Decorated { decorators, stmt }` — classify by the wrapped statement.
fn host_of_inner(inner: &Stmt) -> DecoratorHost {
    match &inner.kind {
        StmtKind::FnDef(_) => DecoratorHost::Function,
        StmtKind::ModelDef(_) => DecoratorHost::Model,
        StmtKind::TrainBlock(_) => DecoratorHost::TrainBlock,
        StmtKind::DistillBlock(_) => DecoratorHost::DistillBlock,
        StmtKind::KernelDef(_) => DecoratorHost::Kernel,
        StmtKind::AgentDef(_) => DecoratorHost::Agent,
        // A doubly-decorated statement: `@a\n@b\nfn ...` parses as nested
        // Decorated in some forms — classify by the innermost non-decorated
        // statement so `@a` and `@b` report the same host.
        StmtKind::Decorated { stmt, .. } => host_of_inner(stmt),
        _ => DecoratorHost::OtherStmt,
    }
}

// Hand recursion rather than the `Visitor` trait: the trait's methods take
// `&Stmt` under an anonymous lifetime, so an implementation cannot store
// `&'a Decorator` borrowed from the module. The recursion mirrors
// `visitor::walk_stmt`'s reach (bodies, sections, nested blocks) and adds the
// member-level decorator vectors the generic walk skips.
impl<'a> Collector<'a> {
    fn collect_stmt(&mut self, stmt: &'a Stmt) {
        match &stmt.kind {
            StmtKind::Decorated { decorators, stmt: inner } => {
                let host = host_of_inner(inner);
                for d in decorators {
                    self.out.push(DecoratorUse { deco: d, host });
                }
                self.collect_stmt(inner);
            }
            StmtKind::ModelDef(m) => {
                for member in &m.members {
                    match member {
                        ModelMember::Method(f, decos) => {
                            for d in decos {
                                self.out.push(DecoratorUse { deco: d, host: DecoratorHost::ModelMethod });
                            }
                            self.collect_block(&f.body);
                        }
                        ModelMember::LayerDecl { decorators, .. } => {
                            for d in decorators {
                                self.out.push(DecoratorUse { deco: d, host: DecoratorHost::ModelLayer });
                            }
                        }
                    }
                }
            }
            StmtKind::AgentDef(a) => {
                for member in &a.members {
                    match member {
                        crate::agent::AgentMember::Method(f, decos) => {
                            for d in decos {
                                self.out.push(DecoratorUse { deco: d, host: DecoratorHost::AgentMethod });
                            }
                            self.collect_block(&f.body);
                        }
                        crate::agent::AgentMember::FieldDecl { decorators, .. } => {
                            for d in decorators {
                                self.out.push(DecoratorUse { deco: d, host: DecoratorHost::AgentField });
                            }
                        }
                    }
                }
            }
            StmtKind::KernelDef(k) => {
                for d in &k.decorators {
                    self.out.push(DecoratorUse { deco: d, host: DecoratorHost::Kernel });
                }
                self.collect_block(&k.body);
            }
            StmtKind::FnDef(f) => self.collect_block(&f.body),
            StmtKind::TrainBlock(t) => self.collect_sections(&t.sections),
            StmtKind::DistillBlock(d) => self.collect_sections(&d.sections),
            StmtKind::GradBlock(g) => self.collect_block(&g.body),
            StmtKind::If { then_block, elif_clauses, else_block, .. } => {
                self.collect_block(then_block);
                for (_, b) in elif_clauses {
                    self.collect_block(b);
                }
                if let Some(b) = else_block {
                    self.collect_block(b);
                }
            }
            StmtKind::For { body, .. }
            | StmtKind::While { body, .. }
            | StmtKind::WhileLet { body, .. } => self.collect_block(body),
            StmtKind::Match { arms, .. } => {
                for arm in arms {
                    self.collect_block(&arm.body);
                }
            }
            StmtKind::ServeBlock(s) => {
                for ep in &s.endpoints {
                    self.collect_block(&ep.body);
                }
            }
            _ => {}
        }
    }

    fn collect_block(&mut self, block: &'a crate::stmt::Block) {
        for s in &block.stmts {
            self.collect_stmt(s);
        }
    }

    fn collect_sections(&mut self, sections: &'a [crate::block::TrainSection]) {
        use crate::block::TrainSection;
        for section in sections {
            match section {
                TrainSection::Data(stmts) => {
                    for s in stmts {
                        self.collect_stmt(s);
                    }
                }
                TrainSection::Step { body, .. } | TrainSection::Eval { body, .. } => {
                    self.collect_block(body);
                }
                TrainSection::Callbacks(cbs) => {
                    for cb in cbs {
                        self.collect_block(&cb.body);
                    }
                }
                TrainSection::Stmt(s) => self.collect_stmt(s),
                TrainSection::Optimizer(_)
                | TrainSection::Scheduler(_)
                | TrainSection::Distribute(_) => {}
            }
        }
    }
}

/// Every decorator in the module, in source order, with its host.
pub fn collect_decorators(module: &Module) -> Vec<DecoratorUse<'_>> {
    let mut c = Collector { out: Vec::new() };
    for stmt in &module.stmts {
        c.collect_stmt(stmt);
    }
    c.out
}
