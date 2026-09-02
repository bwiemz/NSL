//! Dict-local tensor lifetime (the aggregate-lifetime gap, 2026-07-28).
//!
//! A `Dict<Str, Tensor>` produced by a builtin (`topk`) or a compiled NSL
//! function OWNS its stored tensors outright: `nsl_dict_set_str` stores the
//! raw handle, and every read clones (`compile_subscript`'s Dict arm emits
//! `nsl_tensor_clone`), so no borrow of a stored tensor can exist anywhere.
//! Yet codegen only ever emitted `nsl_dict_free_tensor_values` for
//! DataLoader batch dicts — an ordinary dict LOCAL was never freed, so
//! `let r = topk(scaled, k)` stranded the dict plus both stored tensors on
//! every call (`stdlib/nsl/inference/sampling.nsl`: 2 VRAM blocks per call
//! on GPU inputs; ~79 MB/call resident heap growth in a CPU `topk(x, 1M)`
//! loop).
//!
//! This module is the conservative usage scan that decides which dict
//! locals the return-local sweep may free with
//! `nsl_dict_free_tensor_values`. A symbol qualifies only when EVERY
//! appearance in the function body is one of:
//!
//!   - its single TOP-LEVEL `let` binding, whose RHS is a call to a
//!     `FRESH_DICT_BUILTINS` entry (fresh owning dict by construction;
//!     top-level so the binding dominates every later return — a
//!     conditionally-bound slot would hand garbage to the free);
//!   - a subscript READ `r[...]` (lowered to `nsl_dict_get_str` + clone —
//!     touches the dict, takes nothing out of it).
//!
//! Anything else vetoes the symbol: a bare identifier use (call argument,
//! return value, RHS of another binding, element of a literal), a
//! reassignment, ANY pattern binding of the name outside its own decl
//! (shadowing re-declarations, loop patterns, comprehension generators,
//! match-expression arms), a subscript WRITE (`r[k] = t` would store a
//! handle the dict does not own), a locally-bound callee name (the
//! lambda-shadow guard), or the presence of an opaque block
//! (`train`/`grad`/`distill`/`quant`/`serve` compile through their own
//! lowerings this walker does not model — their presence clears every
//! candidate). Unknown statement shapes keep the DEFAULT walk, which
//! vetoes on any bare identifier appearance inside them — fail-closed: a
//! missed veto is a leak-shaped bug turned use-after-free, so the walk
//! must only ever ALLOW shapes it fully understands. Review on 7a0f83d9
//! proved the direction twice (HIGH-1 comprehension-generator shadowing,
//! HIGH-2 lambda-returned dict aliases — both runtime crashes, both now
//! vetoed and gated).

use std::collections::{HashMap, HashSet};

use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::pattern::{Pattern, PatternKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::visitor::{walk_expr, walk_pattern, walk_stmt, Visitor};
use nsl_ast::{NodeId, Symbol};
use nsl_semantic::types::Type;

/// Builtins whose call result is a FRESH dict owning its tensor values.
/// Pass 1 admits candidates only for these callees: admitting arbitrary
/// Dict-typed calls let a LAMBDA returning a captured dict mint two
/// "owning" candidates for one dict — a double free (review HIGH-2 on
/// 7a0f83d9, confirmed crash). User fns and model methods cannot return
/// dicts at all (checker error, probed), so the builtin list is the whole
/// fresh-dict surface. Per-runtime verification:
/// - topk → nsl_tensor_topk: fresh dict, values/indices created rc=1 and
///   stored raw (sampling.rs).
/// - load_safetensors → nsl_safetensors_load: fresh dict; every stored
///   tensor is a fresh `allocate_f32_tensor` the dict solely owns
///   (safetensors_io.rs; #427-review-verified, gated by
///   `load_safetensors_dict_is_swept_at_exit`).
const FRESH_DICT_BUILTINS: &[&str] = &["topk", "load_safetensors"];

/// Usage-veto walker. `candidates` holds the top-level dict-local bindings
/// (symbol → its VarDecl stmt id so the binding itself is not treated as a
/// shadowing re-declaration); `vetoed` accumulates disqualified symbols.
struct DictLocalScan<'a> {
    candidates: &'a HashMap<Symbol, NodeId>,
    vetoed: HashSet<Symbol>,
    opaque_block_seen: bool,
    /// Every symbol bound by any pattern or declaration the walk sees.
    /// A candidate whose CALLEE name is locally bound is vetoed after the
    /// walk — `let topk = |y| d` would otherwise defeat the
    /// `FRESH_DICT_BUILTINS` name check (the lambda-shadow variant of
    /// review HIGH-2).
    bound_anywhere: HashSet<Symbol>,
}

impl DictLocalScan<'_> {
    fn is_live_candidate(&self, sym: Symbol) -> bool {
        self.candidates.contains_key(&sym) && !self.vetoed.contains(&sym)
    }
}

impl Visitor for DictLocalScan<'_> {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            // Opaque bodies compile against the same FuncState through their
            // own lowerings — clear the whole scan.
            StmtKind::TrainBlock(_)
            | StmtKind::GradBlock(_)
            | StmtKind::DistillBlock(_)
            | StmtKind::QuantBlock(_)
            | StmtKind::ServeBlock(_) => {
                self.opaque_block_seen = true;
                walk_stmt(self, stmt);
            }
            // A re-declaration of a candidate anywhere but its own binding
            // site is a shadow/rebind — the swept slot would hold only the
            // last value while earlier ones strand, and a NESTED (loop/if)
            // re-declaration can leave the slot conditionally stale.
            StmtKind::VarDecl { pattern, value, .. } => {
                let mut bound = HashSet::new();
                collect_pattern_syms(pattern, &mut bound);
                for sym in bound {
                    self.bound_anywhere.insert(sym);
                    if self.candidates.contains_key(&sym)
                        && self.candidates[&sym] != stmt.id
                    {
                        self.vetoed.insert(sym);
                    }
                }
                // Visit ONLY the RHS: the default walk would hand this
                // pattern to `visit_pattern`, whose blanket veto below
                // would disqualify a candidate at its own binding site.
                if let Some(v) = value {
                    self.visit_expr(v);
                }
            }
            StmtKind::Assign { target, .. } => {
                // `r = ...` rebinding and `r[k] = ...` storing both veto —
                // handled here because the expression walk below deliberately
                // ALLOWS a candidate as a subscript object (the read shape).
                match &target.kind {
                    ExprKind::Ident(sym) => {
                        self.bound_anywhere.insert(*sym);
                        if self.is_live_candidate(*sym) {
                            self.vetoed.insert(*sym);
                        }
                    }
                    ExprKind::Subscript { object, .. } => {
                        if let ExprKind::Ident(sym) = &object.kind
                            && self.is_live_candidate(*sym)
                        {
                            self.vetoed.insert(*sym);
                        }
                    }
                    _ => {}
                }
                walk_stmt(self, stmt);
            }
            // Loop / match patterns that bind a candidate name shadow it.
            // These patterns are NOT visited by walk_stmt, so this arm is
            // their only recording site — the `bound_anywhere` inserts also
            // keep the lambda-shadow guard sound if callee resolution ever
            // starts consulting loop-bound locals (re-review LOW on
            // 4a790ad0).
            StmtKind::For { pattern, .. } | StmtKind::WhileLet { pattern, .. } => {
                let mut bound = HashSet::new();
                collect_pattern_syms(pattern, &mut bound);
                for sym in bound {
                    self.bound_anywhere.insert(sym);
                    if self.candidates.contains_key(&sym) {
                        self.vetoed.insert(sym);
                    }
                }
                walk_stmt(self, stmt);
            }
            StmtKind::Match { arms, .. } => {
                for arm in arms {
                    let mut bound = HashSet::new();
                    collect_pattern_syms(&arm.pattern, &mut bound);
                    for sym in bound {
                        self.bound_anywhere.insert(sym);
                        if self.candidates.contains_key(&sym) {
                            self.vetoed.insert(sym);
                        }
                    }
                }
                walk_stmt(self, stmt);
            }
            _ => walk_stmt(self, stmt),
        }
    }

    /// ANY pattern binding of a candidate name outside its own VarDecl —
    /// comprehension generators, match-EXPRESSION arms, loop patterns,
    /// lambda params — re-points the symbol's `state.variables` slot at a
    /// non-dict value (`compile_comp_level` inserts the loop element var
    /// over the same symbol), and the sweep would then free garbage
    /// (review HIGH-1 on 7a0f83d9: a `[0 for r in ...]` generator —
    /// confirmed crash). The candidate's own binding never reaches this
    /// override: its VarDecl arm visits only the RHS.
    fn visit_pattern(&mut self, pat: &Pattern) {
        let mut bound = HashSet::new();
        collect_pattern_syms(pat, &mut bound);
        for sym in bound {
            self.bound_anywhere.insert(sym);
            if self.candidates.contains_key(&sym) {
                self.vetoed.insert(sym);
            }
        }
        walk_pattern(self, pat);
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            // The ONE allowed expression shape: candidate as the OBJECT of a
            // subscript READ. Walk the index normally; skip the object ident.
            ExprKind::Subscript { object, index }
                if matches!(&**index, SubscriptKind::Index(_))
                    && matches!(&object.kind,
                                ExprKind::Ident(s) if self.is_live_candidate(*s)) =>
            {
                let SubscriptKind::Index(idx) = &**index else {
                    unreachable!("guarded by the match arm condition above");
                };
                self.visit_expr(idx);
            }
            // Any other bare appearance — argument, return value, alias RHS,
            // literal element, non-Index subscript — disqualifies.
            ExprKind::Ident(sym) => {
                if self.candidates.contains_key(sym) {
                    self.vetoed.insert(*sym);
                }
            }
            _ => walk_expr(self, expr),
        }
    }
}

fn collect_pattern_syms(pattern: &nsl_ast::pattern::Pattern, out: &mut HashSet<Symbol>) {
    match &pattern.kind {
        PatternKind::Ident(s) | PatternKind::Rest(Some(s)) => {
            out.insert(*s);
        }
        PatternKind::Tuple(ps) | PatternKind::List(ps) | PatternKind::Or(ps) => {
            for p in ps {
                collect_pattern_syms(p, out);
            }
        }
        PatternKind::Constructor { args, .. } => {
            for p in args {
                collect_pattern_syms(p, out);
            }
        }
        PatternKind::Struct { fields, .. } => {
            for f in fields {
                if let Some(p) = &f.pattern {
                    collect_pattern_syms(p, out);
                }
            }
        }
        PatternKind::Guarded { pattern, .. } | PatternKind::Typed { pattern, .. } => {
            collect_pattern_syms(pattern, out);
        }
        PatternKind::Literal(_) | PatternKind::Wildcard | PatternKind::Rest(None) => {}
    }
}

/// The scan's output: which dict locals the return sweep frees at exit,
/// and which of those additionally get a per-rebind clear because they
/// are bound inside a top-level loop body (the loop lowering predeclares
/// their slot to zero; `compile_var_decl` frees the previous iteration's
/// dict before the rebind).
#[derive(Default)]
pub(crate) struct DictSweepPlan {
    pub(crate) at_exit: HashSet<Symbol>,
    pub(crate) loop_rebind: HashSet<Symbol>,
}

/// Records every identifier occurrence — including subscript objects and
/// pattern bindings, which `DictLocalScan` deliberately exempts. Used for
/// the outside-the-loop mention veto on loop-body candidates: a
/// subscript READ after the loop would touch a slot that is zero when
/// the loop ran no iterations, and `nsl_dict_get_str(0)` is a null
/// dereference.
#[derive(Default)]
struct MentionScan {
    mentioned: HashSet<Symbol>,
}

impl Visitor for MentionScan {
    fn visit_expr(&mut self, expr: &Expr) {
        if let ExprKind::Ident(sym) = &expr.kind {
            self.mentioned.insert(*sym);
        }
        walk_expr(self, expr);
    }
    fn visit_pattern(&mut self, pat: &Pattern) {
        collect_pattern_syms(pat, &mut self.mentioned);
        walk_pattern(self, pat);
    }
}

/// Entry point: scan `body` and return the dict-local sweep plan.
/// `node_type` resolves an expression id to its checked semantic type
/// (the compiler's `node_type`, passed as a closure to keep this module
/// independent of `Compiler`).
pub(crate) fn collect_sweepable_dict_locals(
    body: &Block,
    node_type: impl Fn(NodeId) -> Type,
    resolve_sym: impl Fn(Symbol) -> String,
) -> DictSweepPlan {
    // Pass 1: `let <ident> = <fresh-dict builtin call>` bindings whose
    // checked type is Dict<_, Tensor-valued>. The callee must be an Ident
    // naming a FRESH_DICT_BUILTINS entry — see the const above for why
    // arbitrary Dict-typed calls (lambdas) are unsafe to admit. Admitted
    // from two positions only, both chosen so the (pre)declared slot
    // dominates every return that can observe it:
    //   - the body's top level;
    //   - the DIRECT body of a TOP-LEVEL `for`/`while` (the loop-rebind
    //     pool; the loop lowering zero-predeclares these slots).
    let admit = |stmt: &Stmt| -> Option<(Symbol, NodeId, Symbol)> {
        let StmtKind::VarDecl { pattern, value, .. } = &stmt.kind else {
            return None;
        };
        let PatternKind::Ident(sym) = &pattern.kind else {
            return None;
        };
        let value = value.as_ref()?;
        let ExprKind::Call { callee, .. } = &value.kind else {
            return None;
        };
        let ExprKind::Ident(callee_sym) = &callee.kind else {
            return None;
        };
        if !FRESH_DICT_BUILTINS.contains(&resolve_sym(*callee_sym).as_str()) {
            return None;
        }
        let is_tensor_dict = matches!(
            node_type(value.id),
            Type::Dict(_, ref v) if v.is_tensor()
        );
        if !is_tensor_dict {
            return None;
        }
        Some((*sym, stmt.id, *callee_sym))
    };

    let mut candidates: HashMap<Symbol, NodeId> = HashMap::new();
    let mut candidate_callees: HashMap<Symbol, Symbol> = HashMap::new();
    // sym → index of its owning top-level loop stmt (loop-rebind pool).
    let mut loop_owner: HashMap<Symbol, usize> = HashMap::new();
    let insert = |sym: Symbol,
                      id: NodeId,
                      callee: Symbol,
                      owner: Option<usize>,
                      candidates: &mut HashMap<Symbol, NodeId>,
                      candidate_callees: &mut HashMap<Symbol, Symbol>,
                      loop_owner: &mut HashMap<Symbol, usize>| {
        // A candidate declared twice (any mix of positions) is a rebind —
        // drop it outright rather than keeping either binding.
        if candidates.insert(sym, id).is_some() {
            candidates.remove(&sym);
            candidate_callees.remove(&sym);
            loop_owner.remove(&sym);
        } else {
            candidate_callees.insert(sym, callee);
            if let Some(ix) = owner {
                loop_owner.insert(sym, ix);
            }
        }
    };
    for (ix, stmt) in body.stmts.iter().enumerate() {
        if let Some((sym, id, callee)) = admit(stmt) {
            insert(
                sym,
                id,
                callee,
                None,
                &mut candidates,
                &mut candidate_callees,
                &mut loop_owner,
            );
            continue;
        }
        let (StmtKind::For { body: loop_body, .. } | StmtKind::While { body: loop_body, .. }) =
            &stmt.kind
        else {
            continue;
        };
        for inner in &loop_body.stmts {
            if let Some((sym, id, callee)) = admit(inner) {
                insert(
                    sym,
                    id,
                    callee,
                    Some(ix),
                    &mut candidates,
                    &mut candidate_callees,
                    &mut loop_owner,
                );
            }
        }
    }
    if candidates.is_empty() {
        return DictSweepPlan::default();
    }

    // Pass 2: the usage veto walk over the whole body.
    let mut scan = DictLocalScan {
        candidates: &candidates,
        vetoed: HashSet::new(),
        opaque_block_seen: false,
        bound_anywhere: HashSet::new(),
    };
    for stmt in &body.stmts {
        scan.visit_stmt(stmt);
    }
    if scan.opaque_block_seen {
        return DictSweepPlan::default();
    }
    let mut vetoed = scan.vetoed;
    let bound_anywhere = scan.bound_anywhere;

    // Loop-rebind pool only: veto on ANY mention outside the owning loop
    // statement. Even the otherwise-allowed subscript read is unsafe
    // there — with zero iterations the predeclared slot is still 0 and
    // `nsl_dict_get_str(0)` is a null dereference.
    for (sym, owner_ix) in &loop_owner {
        if vetoed.contains(sym) {
            continue;
        }
        let mut outside = MentionScan::default();
        for (ix, stmt) in body.stmts.iter().enumerate() {
            if ix != *owner_ix {
                outside.visit_stmt(stmt);
            }
        }
        if outside.mentioned.contains(sym) {
            vetoed.insert(*sym);
        }
    }

    let admitted: HashSet<Symbol> = candidates
        .into_keys()
        .filter(|sym| !vetoed.contains(sym))
        // Lambda-shadow guard: if the CALLEE name is bound anywhere in
        // this body, the call may not be the builtin at all.
        .filter(|sym| {
            candidate_callees
                .get(sym)
                .is_none_or(|callee| !bound_anywhere.contains(callee))
        })
        .collect();
    let loop_rebind: HashSet<Symbol> = admitted
        .iter()
        .copied()
        .filter(|sym| loop_owner.contains_key(sym))
        .collect();
    DictSweepPlan {
        at_exit: admitted,
        loop_rebind,
    }
}
