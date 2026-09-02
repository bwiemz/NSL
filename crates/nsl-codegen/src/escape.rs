//! Parameter escape analysis (roadmap item 1: "alias + escape analysis").
//!
//! # The problem this solves
//!
//! Both user-function calls (`expr/calls.rs`) and model-method calls
//! (`expr/advanced.rs`) compile their arguments with `compile_expr`, NOT
//! `compile_nested_expr`, so an argument that is itself a fresh allocation is
//! never registered as a statement temporary and is never freed. The refusal
//! is deliberate and is documented at both sites:
//!
//! > the callee may ESCAPE a param (member-assign stores the raw pointer with
//! > no retain — `fn set(self, t): self.buf = t`), so freeing a tracked owning
//! > arg at statement end is a use-after-free on the stored field. Owning
//! > call-result args strand instead (status quo).
//!
//! That blanket refusal costs real memory. In `TransformerBlock.forward`
//!
//! ```text
//! let h = x + self.attn.forward(self.attn_norm.forward(x), training)
//! return h + self.ffn.forward(self.ffn_norm.forward(h), training)
//! ```
//!
//! every norm output is a fresh tensor handed straight to a method that only
//! reads it — and every one of them stranded. Measured at Coder-50M `[2,1024]`
//! on an RTX 5070 Ti: ~0.65 GB per forward, with all 17 RMSNorm outputs of a
//! forward still resident.
//!
//! This module replaces the blanket refusal with a decision: a fresh argument
//! may be released at statement end **iff the callee's matching parameter is
//! proven not to escape**.
//!
//! # What "escapes" means here
//!
//! Parameter `p` escapes if the handle passed in can still be reachable after
//! the call returns. In NSL every such store is syntactically visible in the
//! callee body, because a callee body is NSL source:
//!
//!   * it is returned (`return p`, or returned through an alias of `p`),
//!   * it is stored into a field or a container element (`self.buf = p`,
//!     `xs[i] = p`),
//!   * it is placed in an aggregate literal (`[p]`, `(p, q)`, `{"k": p}`),
//!   * it is captured by a lambda,
//!   * it is yielded,
//!   * it is passed to a callee that escapes it in turn.
//!
//! Anything else — arithmetic, matmul, activations, reductions, norms — is a
//! read that produces a fresh result.
//!
//! # Aliasing
//!
//! Escape is tracked over *aliases*, not just the bare symbol, so
//! `let v = p.reshape([...]); return v` correctly escapes `p`. A value aliases
//! `p` when it is `p`, a local previously assigned from an alias of `p`, or the
//! result of an alias-preserving operation on one (`ALIAS_PRESERVING_METHODS`,
//! subscripting, parenthesisation, branch joins). Fresh-result operations
//! (binary/unary ops, non-aliasing method calls, plain calls) break the chain.
//!
//! # Soundness posture
//!
//! Every axis defaults to "escapes":
//!
//!   * an unknown callee (imported model, function pointer, indirect call)
//!     escapes every argument,
//!   * a callee we have no body for escapes every argument,
//!   * the expression walker is an EXHAUSTIVE match over `ExprKind` with no
//!     catch-all, so a new variant is a compile error rather than a silently
//!     missed escape,
//!   * an unmodelled STATEMENT form (`train`/`grad`/`distill`/`quant`/`serve`
//!     bodies, which are compiled against the ENCLOSING state and therefore do
//!     see enclosing parameters) escapes every parameter,
//!   * a call to a name in `RETAINING_BUILTINS` escapes its arguments,
//!   * variadic and arity-mismatched calls escape.
//!
//! Two fixpoints, both monotone:
//!
//!   * ACROSS functions, `param_escapes` starts optimistic (nothing escapes)
//!     and only ever gains escapes, and is consulted only after it settles.
//!   * WITHIN a body, `aliases` uses WEAK updates (a symbol accumulates every
//!     taint it is ever bound to) and the body is re-walked until neither
//!     `aliases` nor `escaped` grows. Weak updates are what make branch joins
//!     and loop-carried aliases correct: a strong update let the
//!     last-walked `if`/`else` arm clobber the other, and let a re-walk of
//!     `let cur = zeros(...)` wipe what the previous pass had learned.
//!
//! Every one of those five bullets and both fixpoint properties exists because
//! an adversarial review found a concrete use-after-free without it; see the
//! regression tests at the bottom of this file, each named for its path.
//!
//! Set `NSL_NO_ESCAPE_ANALYSIS=1` to force the old blanket-refusal behaviour.

use std::collections::{HashMap, HashSet};

use nsl_ast::decl::FnDef;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::Symbol;

/// Tensor methods whose result may alias the receiver's storage. A value
/// derived through one of these keeps the receiver's escape taint.
///
/// `clone`/`to`/`contiguous` are here on purpose: each is documented elsewhere
/// in this crate as possibly handing back the receiver unchanged (a clone can
/// elide to the receiver, `.to()` returns the receiver when already on the
/// target device, `.contiguous()` when already contiguous).
const ALIAS_PRESERVING_METHODS: &[&str] = &[
    "reshape",
    "view",
    "view_as",
    "transpose",
    "permute",
    "squeeze",
    "unsqueeze",
    "expand",
    "expand_as",
    "broadcast_to",
    "flatten",
    "slice",
    "narrow",
    "select",
    "index_select",
    "chunk",
    "split",
    "t",
    "detach",
    "contiguous",
    "clone",
    "to",
    "as_type",
    "astype",
];

/// Free functions and methods that STORE a tensor argument somewhere outliving
/// the call. Anything named here escapes every argument.
///
/// Kept deliberately broad — a false entry costs one stranded tensor, a
/// missing entry costs a use-after-free.
const RETAINING_BUILTINS: &[&str] = &[
    "append",
    "push",
    "insert",
    "extend",
    "add",
    "put",
    "set",
    "store",
    "save",
    "list_push",
    "dict_set",
    "dict_insert",
    "model_save",
    "model_load",
    "checkpoint_save",
    "dataloader",
    "DataLoader",
    "register_buffer",
    "register_parameter",
    "cache_put",
    "kv_cache_store",
    "tape_record",
];

/// Per-function escape result: one flag per DECLARED parameter, in source
/// order. For a model method that includes the leading `self`, which the
/// parser materialises as an ordinary `Param` named `"self"` — see
/// `arg_offset`.
#[derive(Debug, Clone, Default)]
pub struct EscapeInfo {
    pub param_escapes: Vec<bool>,
    /// 1 when the first declared parameter is `self`, else 0. Call sites pass
    /// ARGUMENT indices, which exclude the receiver, so the public query adds
    /// this to line the two up. Getting this wrong silently consults the
    /// neighbouring parameter's flag.
    pub arg_offset: usize,
    /// True when any declared parameter is variadic. Such a callee has no
    /// 1:1 argument-to-parameter mapping, so no argument is ever captive.
    pub variadic: bool,
}

/// True when `def`'s first declared parameter is the RECEIVER.
///
/// Keyed on the entry being a model method (`Model::method`) as well as on the
/// name: `crates/nsl-parser/src/decl.rs` accepts `self` as a parameter name in
/// ANY parameter list, including a top-level `fn`. Testing the name alone gave
/// `fn f(self, t)` an `arg_offset` of 1, so `param_is_captive("f", 0)` read the
/// flag for `t` — a silent off-by-one that could free an escaping argument.
fn has_self_param<R: Fn(Symbol) -> String>(key: &str, def: &FnDef, resolve: &R) -> bool {
    key.contains("::")
        && def
            .params
            .first()
            .is_some_and(|p| resolve(p.name) == "self")
}

/// Escape facts for every function and model method with a visible body.
///
/// Method keys are `"Model::method"`; free-function keys are the bare name.
#[derive(Debug, Clone, Default)]
pub struct EscapeAnalysis {
    infos: HashMap<String, EscapeInfo>,
    enabled: bool,
}

/// Key for a model method.
pub fn method_key(model: &str, method: &str) -> String {
    format!("{model}::{method}")
}

impl EscapeAnalysis {
    /// Disabled analysis: every parameter escapes. Equivalent to the
    /// pre-analysis blanket refusal.
    pub fn disabled() -> Self {
        Self {
            infos: HashMap::new(),
            enabled: false,
        }
    }

    /// True when the parameter receiving ARGUMENT `arg_idx` of `callee` is
    /// PROVEN not to escape, so a fresh argument in that position may be
    /// released at statement end.
    ///
    /// `arg_idx` indexes the call site's argument list, which excludes a
    /// method receiver; the stored `arg_offset` bridges to declared-parameter
    /// indices.
    ///
    /// Returns false for every unknown callee, every out-of-range index, and
    /// whenever the analysis is disabled.
    pub fn param_is_captive(&self, callee: &str, arg_idx: usize) -> bool {
        if !self.enabled {
            return false;
        }
        match self.infos.get(callee) {
            // A variadic callee has no 1:1 argument-to-parameter mapping, so
            // the index would be meaningless. The walker already refuses these
            // (`arity_ok`); mirror it here so the two cannot disagree.
            Some(info) if info.variadic => false,
            Some(info) => info
                .param_escapes
                .get(arg_idx + info.arg_offset)
                .is_some_and(|esc| !esc),
            None => false,
        }
    }

    /// Number of analysed functions — used by tests and diagnostics.
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Iterate the analysed entries (diagnostics: `NSL_DEBUG_ESCAPE=1`).
    pub fn iter(&self) -> impl Iterator<Item = (&String, &EscapeInfo)> {
        self.infos.iter()
    }

    /// Run the analysis to a fixpoint.
    ///
    /// `defs` maps a call-site key (bare name, or `Model::method`) to that
    /// function's AST. `resolve` turns a `Symbol` into its source name.
    pub fn analyze<R>(defs: &HashMap<String, FnDef>, resolve: &R) -> Self
    where
        R: Fn(Symbol) -> String,
    {
        if std::env::var("NSL_NO_ESCAPE_ANALYSIS").as_deref() == Ok("1") {
            return Self::disabled();
        }

        // Optimistic start: assume nothing escapes, then only ever add.
        let mut infos: HashMap<String, EscapeInfo> = defs
            .iter()
            .map(|(k, f)| {
                (
                    k.clone(),
                    EscapeInfo {
                        param_escapes: vec![false; f.params.len()],
                        arg_offset: usize::from(has_self_param(k, f, resolve)),
                        variadic: f.params.iter().any(|p| p.is_variadic),
                    },
                )
            })
            .collect();

        // Monotone fixpoint. Bounded by (#params) flips plus one settling
        // pass, but keep a hard cap so a pathological input cannot spin.
        let cap = defs.values().map(|f| f.params.len()).sum::<usize>() + 2;
        for _ in 0..cap.max(2) {
            let mut changed = false;
            for (key, def) in defs {
                let mut walker = Walker {
                    resolve,
                    defs,
                    infos: &infos,
                    params: def
                        .params
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.name, i))
                        .collect(),
                    aliases: HashMap::new(),
                    escaped: HashSet::new(),
                };
                // Intra-body fixpoint. A store can precede the assignment that
                // taints the stored symbol — the loop-carried
                //     for i in ...:  self.last = cur;  cur = x
                // publishes iteration N-1's value, so `cur` must be seen as
                // aliasing `x` at a statement TEXTUALLY BEFORE the assignment
                // that makes it so.
                //
                // A fixed two passes cannot do this while `aliases` updates
                // are strong: pass 2 re-executes `let cur = zeros([4])` and
                // wipes what pass 1 learned. Updates are therefore WEAK
                // (union, `taint_binding`), which makes `aliases` grow
                // monotonically, so iterating until it stops growing is a real
                // fixpoint. Weak updates only ever over-approximate aliasing,
                // which over-approximates escapes — the safe direction.
                let mut guard = 0;
                loop {
                    let before_aliases: usize =
                        walker.aliases.values().map(|t| t.len()).sum();
                    let before_escaped = walker.escaped.len();
                    walker.walk_block(&def.body);
                    let after_aliases: usize =
                        walker.aliases.values().map(|t| t.len()).sum();
                    guard += 1;
                    if (after_aliases == before_aliases
                        && walker.escaped.len() == before_escaped)
                        || guard > 8
                    {
                        break;
                    }
                }
                let found: Vec<usize> = walker.escaped.iter().copied().collect();
                drop(walker);
                let cur = infos.get_mut(key).expect("key seeded above");
                for idx in found {
                    if idx < cur.param_escapes.len() && !cur.param_escapes[idx] {
                        cur.param_escapes[idx] = true;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Self {
            infos,
            enabled: true,
        }
    }
}

/// Which parameters a value may alias. Empty = aliases nothing (a fresh value).
type Taint = HashSet<usize>;

struct Walker<'a, R: Fn(Symbol) -> String> {
    resolve: &'a R,
    defs: &'a HashMap<String, FnDef>,
    infos: &'a HashMap<String, EscapeInfo>,
    /// Parameter symbol -> declared index.
    params: HashMap<Symbol, usize>,
    /// Local symbol -> the parameters it may alias.
    aliases: HashMap<Symbol, Taint>,
    /// Parameter indices found to escape.
    escaped: HashSet<usize>,
}

impl<R: Fn(Symbol) -> String> Walker<'_, R> {
    fn escape_all(&mut self, t: &Taint) {
        for &i in t {
            self.escaped.insert(i);
        }
    }

    /// WEAK update: a symbol accumulates every taint it is ever bound to,
    /// never loses one.
    ///
    /// A strong update (`insert`, replacing the old taint) is unsound here for
    /// two independent reasons:
    ///   * branches — `if c: v = t` / `else: v = fresh()` walked in sequence
    ///     let the last branch clobber the first, so a later `self.buf = v`
    ///     saw an empty taint and `t` was never marked escaping;
    ///   * loops — the body is walked repeatedly to a fixpoint, and a strong
    ///     update re-executing `let cur = zeros(...)` wipes what the previous
    ///     pass learned, so a loop-carried alias never converges.
    ///
    /// Weak updates only over-approximate aliasing, which over-approximates
    /// escapes: the safe direction (a spurious escape costs a leak; a missed
    /// escape costs a use-after-free).
    fn taint_binding(&mut self, sym: Symbol, t: Taint) {
        if t.is_empty() {
            self.aliases.entry(sym).or_default();
            return;
        }
        self.aliases.entry(sym).or_default().extend(t);
    }

    fn walk_block(&mut self, b: &Block) {
        for s in &b.stmts {
            self.walk_stmt(s);
        }
    }

    fn walk_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::VarDecl {
                pattern,
                value: Some(v),
                ..
            } => {
                let t = self.taint_of(v);
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    self.taint_binding(*sym, t);
                } else if !t.is_empty() {
                    // Destructuring an aliasing value: the parts may alias it
                    // and we do not track them — escape conservatively.
                    self.escape_all(&t);
                }
            }

            StmtKind::Assign { target, value, .. } => {
                let t = self.taint_of(value);
                match &target.kind {
                    ExprKind::Ident(sym) => {
                        // Rebinding a local: it may now ALSO alias whatever the
                        // RHS did (weak — see taint_binding).
                        self.taint_binding(*sym, t);
                    }
                    // `self.buf = p` / `xs[i] = p` — the store outlives the call.
                    ExprKind::MemberAccess { .. } | ExprKind::Subscript { .. } => {
                        self.escape_all(&t);
                    }
                    _ => self.escape_all(&t),
                }
            }

            StmtKind::Return(Some(e)) | StmtKind::Yield(Some(e)) => {
                let t = self.taint_of(e);
                self.escape_all(&t);
            }

            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.visit_reads(condition);
                self.walk_block(then_block);
                for (c, b) in elif_clauses {
                    self.visit_reads(c);
                    self.walk_block(b);
                }
                if let Some(b) = else_block {
                    self.walk_block(b);
                }
            }

            StmtKind::For {
                pattern,
                iterable,
                body,
            } => {
                // Iterating an aliasing container hands out aliasing elements
                // we do not track — escape conservatively.
                let t = self.taint_of(iterable);
                if !t.is_empty() {
                    self.escape_all(&t);
                }
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    // The loop variable is a fresh borrow of a container
                    // element each iteration; it does not alias a parameter
                    // unless the ITERABLE did, which is escaped above.
                    self.aliases.entry(*sym).or_default();
                }
                self.walk_block(body);
            }

            StmtKind::While { condition, body } => {
                self.visit_reads(condition);
                self.walk_block(body);
            }

            StmtKind::WhileLet { pattern, expr, body } => {
                let t = self.taint_of(expr);
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    self.taint_binding(*sym, t);
                } else {
                    self.escape_all(&t);
                }
                self.walk_block(body);
            }

            StmtKind::Match { subject, arms } => {
                let t = self.taint_of(subject);
                // Pattern bindings alias the subject and are not tracked.
                self.escape_all(&t);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.visit_reads(g);
                    }
                    self.walk_block(&arm.body);
                }
            }

            StmtKind::Expr(e) => self.visit_reads(e),

            StmtKind::Decorated { stmt, .. } => self.walk_stmt(stmt),

            // Nested DEFINITIONS introduce their own scope and cannot capture
            // an enclosing parameter, so there is genuinely nothing to taint.
            StmtKind::FnDef(_)
            | StmtKind::ModelDef(_)
            | StmtKind::AgentDef(_)
            | StmtKind::StructDef(_)
            | StmtKind::EnumDef(_)
            | StmtKind::TraitDef(_)
            | StmtKind::KernelDef(_)
            | StmtKind::TokenizerDef(_)
            | StmtKind::DatasetDef(_)
            | StmtKind::DatatypeDef(_)
            | StmtKind::Import(_)
            | StmtKind::FromImport(_)
            | StmtKind::Break
            | StmtKind::Continue
            | StmtKind::Return(None)
            | StmtKind::Yield(None)
            | StmtKind::VarDecl { value: None, .. } => {}

            // Block STATEMENTS are a different matter. `train(...)`,
            // `grad(...)`, `distill(...)`, `quant(...)` and `serve` bodies are
            // compiled against the ENCLOSING FuncState, so they can and do
            // reference the enclosing function's locals and parameters. An
            // earlier version of this arm asserted the opposite and skipped
            // them, which made every escape inside such a block invisible.
            // We do not model their control flow — fail CLOSED: anything they
            // mention escapes.
            _ => {
                // We cannot descend into these bodies (they are typed block
                // structs, not plain `Block`s in a uniform position), and
                // `collect_stmt_idents` does not either — so "parameters
                // mentioned" would UNDER-report. Escape every parameter: the
                // only sound answer for a construct we do not model.
                let all: Taint = self.params.values().copied().collect();
                self.escape_all(&all);
            }
        }
    }

    /// Walk a block and return the taint of its RESULT — the trailing
    /// expression statement, which is what a block/match-arm expression
    /// evaluates to.
    fn block_result_taint(&mut self, b: &Block) -> Taint {
        let mut out = Taint::new();
        for (i, st) in b.stmts.iter().enumerate() {
            let is_last = i + 1 == b.stmts.len();
            if is_last && let StmtKind::Expr(e) = &st.kind {
                out.extend(self.taint_of(e));
                continue;
            }
            self.walk_stmt(st);
        }
        out
    }

    /// Visit an expression purely for its escaping side effects, discarding
    /// the resulting taint (statement position, condition position, ...).
    fn visit_reads(&mut self, e: &Expr) {
        let _ = self.taint_of(e);
    }

    /// Compute which parameters `e` may alias, recording any escapes it causes.
    fn taint_of(&mut self, e: &Expr) -> Taint {
        match &e.kind {
            ExprKind::Ident(sym) => {
                if let Some(&idx) = self.params.get(sym) {
                    let mut t = Taint::new();
                    t.insert(idx);
                    t
                } else {
                    self.aliases.get(sym).cloned().unwrap_or_default()
                }
            }

            ExprKind::Paren(inner) => self.taint_of(inner),

            // Reading a field of an aliasing object yields the FIELD's storage,
            // not the object's, so the chain breaks. The object is only read.
            ExprKind::MemberAccess { object, .. } => {
                self.visit_reads(object);
                Taint::new()
            }

            // Subscripting may produce a view — keep the taint.
            ExprKind::Subscript { object, index } => {
                let t = self.taint_of(object);
                self.visit_subscript(index);
                t
            }

            // Fresh-result operations. Operands are read only.
            ExprKind::BinaryOp { left, right, .. } => {
                self.visit_reads(left);
                self.visit_reads(right);
                Taint::new()
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.visit_reads(operand);
                Taint::new()
            }
            // `x |> f` and `x |> f(a)` lower to a REAL call with the piped
            // value as ARGUMENT 0 (see expr/mod.rs's Pipe lowering, which
            // builds `arg_vals = vec![left_val]` and then appends the RHS
            // call's own args). Treating this like a binary op discarded the
            // piped value's taint AND never consulted the callee's escape
            // facts for it — so `u |> keep(c)`, where `keep` stores its first
            // parameter into a model field, reported `u` captive.
            ExprKind::Pipe { left, right } => {
                let (callee, rest): (&Expr, &[nsl_ast::expr::Arg]) = match &right.kind {
                    ExprKind::Call { callee, args } => (callee.as_ref(), args.as_slice()),
                    // `x |> f` — the RHS is the callee itself.
                    _ => (right.as_ref(), &[]),
                };
                let mut args: Vec<nsl_ast::expr::Arg> = Vec::with_capacity(rest.len() + 1);
                args.push(nsl_ast::expr::Arg {
                    name: None,
                    value: (**left).clone(),
                    span: left.span,
                });
                args.extend(rest.iter().cloned());
                self.taint_of_call(callee, &args)
            }

            ExprKind::Call { callee, args } => self.taint_of_call(callee, args),

            // Aggregates hold references that can outlive the call.
            ExprKind::ListLiteral(items) | ExprKind::TupleLiteral(items) => {
                for it in items {
                    let t = self.taint_of(it);
                    self.escape_all(&t);
                }
                Taint::new()
            }
            ExprKind::DictLiteral(pairs) => {
                for (k, v) in pairs {
                    let tk = self.taint_of(k);
                    self.escape_all(&tk);
                    let tv = self.taint_of(v);
                    self.escape_all(&tv);
                }
                Taint::new()
            }

            // A lambda body can outlive the call frame; anything it mentions
            // escapes. `mentioned_params` is a syntactic over-approximation.
            ExprKind::Lambda { body, .. } => {
                let t = self.mentioned_params_expr(body);
                self.escape_all(&t);
                Taint::new()
            }

            // Branch joins: the result may alias either arm.
            ExprKind::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => {
                self.visit_reads(condition);
                let mut t = self.taint_of(then_expr);
                t.extend(self.taint_of(else_expr));
                t
            }

            // The RESULT of a match/block expression is whatever its arms
            // produce. Returning an empty taint here lost a parameter handed
            // straight back by one arm (`case 0 => t`), so a later
            // `self.buf = v` recorded nothing.
            ExprKind::MatchExpr { subject, arms } => {
                let t = self.taint_of(subject);
                // Pattern bindings alias the subject and are not tracked.
                self.escape_all(&t);
                let mut out = Taint::new();
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.visit_reads(g);
                    }
                    out.extend(self.block_result_taint(&arm.body));
                }
                out
            }

            ExprKind::BlockExpr(b) => self.block_result_taint(b),

            ExprKind::ListComp {
                element,
                generators,
            } => {
                for g in generators {
                    let t = self.mentioned_params_expr(&g.iterable);
                    self.escape_all(&t);
                }
                let t = self.mentioned_params_expr(element);
                self.escape_all(&t);
                Taint::new()
            }

            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.visit_reads(s);
                }
                if let Some(en) = end {
                    self.visit_reads(en);
                }
                Taint::new()
            }

            ExprKind::Await(inner) => {
                let t = self.taint_of(inner);
                // An awaited value crosses a suspension point we do not model.
                self.escape_all(&t);
                Taint::new()
            }

            ExprKind::FString(parts) => {
                for p in parts {
                    if let nsl_ast::expr::FStringPart::Expr(ex) = p {
                        self.visit_reads(ex);
                    }
                }
                Taint::new()
            }

            // Literals, SelfRef and Error carry no operands, so an empty
            // taint is exact for them. Everything else that might be added to
            // ExprKind later must fail CLOSED, matching the documented
            // posture: over-approximate by treating every parameter the
            // expression mentions as aliased AND escaping.
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::NoneLiteral
            | ExprKind::Error => Taint::new(),

            // `self` in a body parses as SelfRef, NOT as `Ident(self_sym)`, so
            // a parameter spelled `self` would otherwise never be tainted.
            ExprKind::SelfRef => self
                .params
                .iter()
                .find(|(sym, _)| (self.resolve)(**sym) == "self")
                .map(|(_, &i)| {
                    let mut t = Taint::new();
                    t.insert(i);
                    t
                })
                .unwrap_or_default(),
            // NO catch-all arm, deliberately. This match is EXHAUSTIVE over
            // `ExprKind`, so adding a variant is a COMPILE ERROR here rather
            // than a silently missed escape — a stronger guarantee than the
            // fail-closed default the module doc describes for the places
            // exhaustiveness is not available. Do not add a `_` arm.
        }
    }

    fn visit_subscript(&mut self, k: &SubscriptKind) {
        match k {
            SubscriptKind::Index(e) => self.visit_reads(e),
            SubscriptKind::Slice { lower, upper, step } => {
                for e in [lower, upper, step].into_iter().flatten() {
                    self.visit_reads(e);
                }
            }
            SubscriptKind::MultiDim(parts) => {
                for p in parts {
                    self.visit_subscript(p);
                }
            }
        }
    }

    fn taint_of_call(&mut self, callee: &Expr, args: &[nsl_ast::expr::Arg]) -> Taint {
        // Resolve the callee to an analysis key and, for method calls, to the
        // receiver expression.
        let (key, method_name, receiver): (Option<String>, Option<String>, Option<&Expr>) =
            match &callee.kind {
                ExprKind::Ident(sym) => (Some((self.resolve)(*sym)), None, None),
                ExprKind::MemberAccess { object, member } => {
                    let m = (self.resolve)(*member);
                    // A method call's key needs the receiver's model type,
                    // which is a codegen-side fact this AST-only pass does not
                    // have. Resolve by unique method name across all analysed
                    // models: unambiguous in practice, and any ambiguity falls
                    // back to "unknown" (escapes).
                    let matches: Vec<&String> = self
                        .defs
                        .keys()
                        .filter(|k| k.ends_with(&format!("::{m}")))
                        .collect();
                    let key = if matches.len() == 1 {
                        Some(matches[0].clone())
                    } else {
                        None
                    };
                    (key, Some(m), Some(object.as_ref()))
                }
                _ => (None, None, None),
            };

        // The receiver is read; if the method is alias-preserving its taint
        // flows to the result.
        let recv_taint = receiver.map(|r| self.taint_of(r)).unwrap_or_default();

        let name_for_blacklist = method_name
            .clone()
            .or_else(|| key.clone())
            .unwrap_or_default();
        let is_retaining = RETAINING_BUILTINS.contains(&name_for_blacklist.as_str());

        // Decide each argument. Argument indices exclude a method receiver,
        // which the parser DOES materialise as `params[0]` named "self", so
        // shift by the same offset the public query uses.
        let known: Option<&EscapeInfo> = key.as_deref().and_then(|k| self.infos.get(k));
        let offset = known.map(|i| i.arg_offset).unwrap_or(0);
        let arity_ok = key
            .as_deref()
            .and_then(|k| self.defs.get(k))
            .is_some_and(|d| {
                d.params.len() == args.len() + offset
                    && !d.params.iter().any(|p| p.is_variadic)
            });

        for (i, a) in args.iter().enumerate() {
            let t = self.taint_of(&a.value);
            if t.is_empty() {
                continue;
            }
            let captive = !is_retaining
                && arity_ok
                && known.is_some_and(|info| {
                    info.param_escapes
                        .get(i + offset)
                        .is_some_and(|esc| !esc)
                });
            if !captive {
                self.escape_all(&t);
            }
        }

        if is_retaining {
            self.escape_all(&recv_taint);
            return Taint::new();
        }

        // Result taint.
        if let Some(m) = &method_name {
            if ALIAS_PRESERVING_METHODS.contains(&m.as_str()) {
                return recv_taint;
            }
            // An unknown method on an aliasing receiver: the receiver is only
            // read (it is `self`, which the callee owns), and the result is a
            // fresh value unless the method is alias-preserving.
            return Taint::new();
        }
        Taint::new()
    }

    /// Syntactic over-approximation: every parameter mentioned anywhere in an
    /// expression, regardless of position. Used where precise tracking is not
    /// worth it (lambda bodies, comprehensions).
    fn mentioned_params_expr(&self, e: &Expr) -> Taint {
        let mut out = Taint::new();
        collect_expr_idents(e, &mut |sym| {
            if let Some(&i) = self.params.get(&sym) {
                out.insert(i);
            }
        });
        out
    }
}

fn collect_block_idents(b: &Block, f: &mut impl FnMut(Symbol)) {
    for s in &b.stmts {
        collect_stmt_idents(s, f);
    }
}

fn collect_stmt_idents(s: &Stmt, f: &mut impl FnMut(Symbol)) {
    match &s.kind {
        StmtKind::VarDecl {
            value: Some(v), ..
        } => collect_expr_idents(v, f),
        StmtKind::Assign { target, value, .. } => {
            collect_expr_idents(target, f);
            collect_expr_idents(value, f);
        }
        StmtKind::Return(Some(e)) | StmtKind::Yield(Some(e)) | StmtKind::Expr(e) => {
            collect_expr_idents(e, f)
        }
        StmtKind::If {
            condition,
            then_block,
            elif_clauses,
            else_block,
        } => {
            collect_expr_idents(condition, f);
            collect_block_idents(then_block, f);
            for (c, b) in elif_clauses {
                collect_expr_idents(c, f);
                collect_block_idents(b, f);
            }
            if let Some(b) = else_block {
                collect_block_idents(b, f);
            }
        }
        StmtKind::For { iterable, body, .. } => {
            collect_expr_idents(iterable, f);
            collect_block_idents(body, f);
        }
        StmtKind::While { condition, body } => {
            collect_expr_idents(condition, f);
            collect_block_idents(body, f);
        }
        StmtKind::WhileLet { expr, body, .. } => {
            collect_expr_idents(expr, f);
            collect_block_idents(body, f);
        }
        StmtKind::Match { subject, arms } => {
            collect_expr_idents(subject, f);
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_expr_idents(g, f);
                }
                collect_block_idents(&a.body, f);
            }
        }
        StmtKind::Decorated { stmt, .. } => collect_stmt_idents(stmt, f),
        _ => {}
    }
}

fn collect_expr_idents(e: &Expr, f: &mut impl FnMut(Symbol)) {
    match &e.kind {
        ExprKind::Ident(sym) => f(*sym),
        ExprKind::Paren(i) | ExprKind::UnaryOp { operand: i, .. } | ExprKind::Await(i) => {
            collect_expr_idents(i, f)
        }
        ExprKind::MemberAccess { object, .. } => collect_expr_idents(object, f),
        ExprKind::Subscript { object, .. } => collect_expr_idents(object, f),
        ExprKind::BinaryOp { left, right, .. } | ExprKind::Pipe { left, right } => {
            collect_expr_idents(left, f);
            collect_expr_idents(right, f);
        }
        ExprKind::Call { callee, args } => {
            collect_expr_idents(callee, f);
            for a in args {
                collect_expr_idents(&a.value, f);
            }
        }
        ExprKind::ListLiteral(xs) | ExprKind::TupleLiteral(xs) => {
            for x in xs {
                collect_expr_idents(x, f);
            }
        }
        ExprKind::DictLiteral(ps) => {
            for (k, v) in ps {
                collect_expr_idents(k, f);
                collect_expr_idents(v, f);
            }
        }
        ExprKind::Lambda { body, .. } => collect_expr_idents(body, f),
        ExprKind::BlockExpr(b) => collect_block_idents(b, f),
        ExprKind::IfExpr {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_expr_idents(condition, f);
            collect_expr_idents(then_expr, f);
            collect_expr_idents(else_expr, f);
        }
        ExprKind::MatchExpr { subject, arms } => {
            collect_expr_idents(subject, f);
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_expr_idents(g, f);
                }
                collect_block_idents(&a.body, f);
            }
        }
        ExprKind::ListComp {
            element,
            generators,
        } => {
            collect_expr_idents(element, f);
            for g in generators {
                collect_expr_idents(&g.iterable, f);
            }
        }
        ExprKind::Range { start, end, .. } => {
            for x in [start, end].into_iter().flatten() {
                collect_expr_idents(x, f);
            }
        }
        ExprKind::FString(parts) => {
            for p in parts {
                if let nsl_ast::expr::FStringPart::Expr(ex) = p {
                    collect_expr_idents(ex, f);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::stmt::StmtKind;
    use nsl_lexer::Interner;

    /// Parse `src` and collect every top-level fn and model method into the
    /// keying scheme `analyze` expects, then run the analysis.
    fn analyze_src(src: &str) -> EscapeAnalysis {
        let mut interner = Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(
            !lex_diags
                .iter()
                .any(|d| matches!(d.level, nsl_errors::Level::Error)),
            "lex errors: {lex_diags:?}"
        );
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            !parsed
                .diagnostics
                .iter()
                .any(|d| matches!(d.level, nsl_errors::Level::Error)),
            "parse errors: {:?}",
            parsed.diagnostics
        );

        let mut defs: HashMap<String, FnDef> = HashMap::new();
        for s in &parsed.module.stmts {
            match &s.kind {
                StmtKind::FnDef(f) => {
                    let name = interner.resolve(f.name.0).unwrap().to_string();
                    defs.insert(name, f.clone());
                }
                StmtKind::ModelDef(md) => {
                    let model = interner.resolve(md.name.0).unwrap().to_string();
                    for m in &md.members {
                        if let nsl_ast::decl::ModelMember::Method(f, _) = m {
                            let meth = interner.resolve(f.name.0).unwrap().to_string();
                            defs.insert(method_key(&model, &meth), f.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        let resolve = |sym: Symbol| {
            interner.resolve(sym.0).unwrap_or("<unknown>").to_string()
        };
        EscapeAnalysis::analyze(&defs, &resolve)
    }

    /// The load-bearing indexing fact: the parser materialises `self` as an
    /// ordinary `Param`, so `param_escapes[0]` describes the RECEIVER, not the
    /// first argument. Call sites pass argument indices. Getting this wrong
    /// silently consults the neighbouring parameter's flag — which is how the
    /// first wiring of this pass looked correct on a fixture where `self`
    /// happened not to escape.
    #[test]
    fn method_arg_indices_skip_the_self_param() {
        let a = analyze_src(
            "model M:\n    buf: Tensor = zeros([2])\n\n    fn stash(self, t: Tensor) -> Tensor:\n        self.buf = t\n        return t\n",
        );
        // `t` is BOTH stored into a field and returned — it must escape, and
        // the query must reach it through arg index 0, not param index 0.
        assert!(
            !a.param_is_captive("M::stash", 0),
            "arg 0 of M::stash escapes (stored into self.buf) but was reported captive \
             — the self-offset is wrong"
        );
    }

    #[test]
    fn read_only_method_param_is_captive() {
        let a = analyze_src(
            "model M:\n    w: Tensor = zeros([2])\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return silu(x @ self.w)\n",
        );
        assert!(
            a.param_is_captive("M::forward", 0),
            "x is only read, so it must be captive"
        );
    }

    #[test]
    fn returned_param_escapes() {
        let a = analyze_src("fn ident(t: Tensor) -> Tensor:\n    return t\n");
        assert!(!a.param_is_captive("ident", 0));
    }

    #[test]
    fn param_returned_through_a_view_escapes() {
        let a = analyze_src(
            "fn v(t: Tensor) -> Tensor:\n    let r = t.reshape([4, 4])\n    return r\n",
        );
        assert!(
            !a.param_is_captive("v", 0),
            "reshape is alias-preserving, so returning the view escapes t"
        );
    }

    #[test]
    fn param_returned_through_a_fresh_op_is_captive() {
        let a = analyze_src("fn f(t: Tensor) -> Tensor:\n    let r = t * 2.0\n    return r\n");
        assert!(
            a.param_is_captive("f", 0),
            "a binary op produces a fresh result, so t itself does not escape"
        );
    }

    #[test]
    fn param_in_aggregate_escapes() {
        let a = analyze_src("fn f(t: Tensor) -> int:\n    let xs = [t]\n    return 1\n");
        assert!(!a.param_is_captive("f", 0));
    }

    #[test]
    fn escape_propagates_through_a_callee() {
        let a = analyze_src(
            "fn sink(t: Tensor) -> Tensor:\n    return t\n\nfn outer(u: Tensor) -> Tensor:\n    return sink(u)\n",
        );
        assert!(!a.param_is_captive("sink", 0));
        assert!(
            !a.param_is_captive("outer", 0),
            "u reaches sink, whose param escapes, so u escapes too"
        );
    }

    #[test]
    fn read_only_callee_keeps_caller_param_captive() {
        let a = analyze_src(
            "fn pure(t: Tensor) -> Tensor:\n    return t * 2.0\n\nfn outer(u: Tensor) -> Tensor:\n    return pure(u)\n",
        );
        assert!(a.param_is_captive("pure", 0));
        assert!(a.param_is_captive("outer", 0));
    }

    #[test]
    fn unknown_callee_is_never_captive() {
        let a = analyze_src("fn f(t: Tensor) -> Tensor:\n    return t * 2.0\n");
        assert!(!a.param_is_captive("not_a_function", 0));
        assert!(!a.param_is_captive("f", 7), "out-of-range index");
    }

    // ── Regression tests for the four missed-escape paths found in adversarial
    // review (2026-07-24). Each was a genuine USE-AFTER-FREE: the parameter was
    // reported captive, so the caller freed a fresh argument that the callee
    // had stored into a field. Field stores take no retain.

    /// S1: `if`/`else` were walked in sequence against one shared alias map
    /// with a STRONG update, so the branch walked LAST clobbered the other.
    #[test]
    fn branch_join_does_not_clobber_the_other_arm() {
        let a = analyze_src(
            "model C:\n    buf: Tensor = zeros([4])\n\n    fn put(self, t: Tensor, flag: bool) -> int:\n        let v = t\n        if flag:\n            v = t\n        else:\n            v = t * 2.0\n        self.buf = v\n        return 0\n",
        );
        assert!(
            !a.param_is_captive("C::put", 0),
            "t reaches self.buf through the THEN arm; the else arm's fresh \
             value must not erase that (strong-update clobber is back)"
        );
    }

    /// S4 (branch-join twin, else-arm form): the arm that hands the parameter
    /// straight back is walked LAST here, the mirror of the then-arm case, so
    /// the two together pin the join in both directions.
    #[test]
    fn branch_join_catches_the_else_arm_too() {
        let a = analyze_src(
            "model C:\n    buf: Tensor = zeros([4])\n\n    fn pick(self, t: Tensor, flag: bool) -> int:\n        let v = zeros([4])\n        if flag:\n            v = zeros([4])\n        else:\n            v = t\n        self.buf = v\n        return 0\n",
        );
        assert!(
            !a.param_is_captive("C::pick", 0),
            "the else arm binds v to the parameter itself"
        );
    }

    /// S3: a store can textually PRECEDE the assignment that makes the stored
    /// symbol alias a parameter. The old fixed two-pass walk could not see it,
    /// because pass 2 re-executed the `let` and wiped what pass 1 learned.
    #[test]
    fn loop_carried_alias_reaches_an_earlier_store() {
        let a = analyze_src(
            "model P:\n    last: Tensor = zeros([4])\n\n    fn run(self, x: Tensor, n: int) -> int:\n        let cur = zeros([4])\n        for i in range(n):\n            self.last = cur\n            cur = x\n        return 0\n",
        );
        assert!(
            !a.param_is_captive("P::run", 0),
            "from iteration 2 onward self.last IS x — the intra-body fixpoint \
             must reach the store that precedes the aliasing assignment"
        );
    }

    /// S2: `x |> f(a)` lowers to `f(x, a)`. Treating it as a binary op threw
    /// away the piped value's taint and never consulted f's escape facts.
    #[test]
    fn pipe_is_treated_as_a_call() {
        let a = analyze_src(
            "model C:\n    buf: Tensor = zeros([4])\n\nfn keep(t: Tensor, c: C) -> int:\n    c.buf = t\n    return 0\n\nfn outer(u: Tensor, c: C) -> int:\n    return u |> keep(c)\n",
        );
        assert!(!a.param_is_captive("keep", 0), "keep stores t into a field");
        assert!(
            !a.param_is_captive("outer", 0),
            "u is piped into keep's escaping parameter 0"
        );
    }

    /// S5: `grad`/`train` block bodies are compiled against the ENCLOSING
    /// state, so they see enclosing parameters. The statement catch-all used to
    /// skip them outright.
    #[test]
    fn block_statement_bodies_are_not_invisible() {
        let a = analyze_src(
            "model C:\n    buf: Tensor = zeros([4])\n\n    fn go(self, t: Tensor) -> int:\n        grad(t):\n            self.buf = t\n        return 0\n",
        );
        assert!(
            !a.param_is_captive("C::go", 0),
            "the store inside the grad block must not be invisible"
        );
    }

    /// S8: the parser accepts `self` as a parameter name in ANY list. Only a
    /// model method actually has a receiver, so a free function must not get
    /// the argument-index offset.
    #[test]
    fn free_function_named_self_param_gets_no_offset() {
        let a = analyze_src(
            "model C:\n    buf: Tensor = zeros([4])\n\nfn f(self: Tensor, t: Tensor, c: C) -> int:\n    c.buf = self\n    return 0\n",
        );
        assert!(
            !a.param_is_captive("f", 0),
            "argument 0 of a FREE function is its first declared parameter, \
             even when that parameter is spelled `self` — it is stored into a \
             field here, so it escapes"
        );
    }

    #[test]
    fn disabled_analysis_reports_nothing_captive() {
        let a = EscapeAnalysis::disabled();
        assert!(!a.param_is_captive("anything", 0));
    }
}
