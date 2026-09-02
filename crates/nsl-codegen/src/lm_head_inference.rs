//! Item 4 — compiler inference for the fused LM head.
//!
//! # What this closes
//!
//! The fused linear-CE kernel (`fused_linear_ce.rs`) deletes the
//! `[batch*seq, vocab]` logits surface and its gradient. It is worth ~8.5%
//! step time on Coder-50M. Until now the ONLY way to get it was to hand-write
//!
//! ```text
//!   @fused_lm_ce(enabled=true, vocab_size=49152, hidden_size=512,
//!                batch_size=1, seq_len=1024)
//!   train(model=m, ...):
//! ```
//!
//! onto the train block — four numbers the compiler already knows, restated by
//! hand, in the model source. `models/benchmarks/mfu_bench.py` measured the
//! 8.5% by *generating* that decorator into a throwaway program; no production
//! pretraining script carries it. A win that only exists in a benchmark
//! harness is not in the production profile.
//!
//! This module supplies the missing half: the **row count**. The other three
//! numbers come from the tape and the weight's declared dims (see
//! `source_ad.rs`'s `known_dims`), but `batch_size * seq_len` is a property of
//! the *data pipeline*, not of the model, so it has to be read off the
//! `DataLoader` construction.
//!
//! # Why the DataLoader, and why unanimity
//!
//! `DataLoader(tokens, batch_size=B, seq_len=S, ...)` already requires every
//! keyword argument to be a compile-time constant (`expr/calls.rs` rejects a
//! non-literal outright), and the batch it yields is exactly `[B, S]`
//! (`dataloader.rs:544`). So `rows = B * S` is a genuine compile-time fact,
//! not an assumption — *provided* we can tell which loader feeds the train
//! block.
//!
//! Deciding that precisely would mean reproducing `state.cleanup
//! .dataloader_vars`' control-flow merge, which is exactly the kind of
//! second implementation that drifts. Instead this scan is deliberately
//! **unanimous-or-nothing**: every `DataLoader` in the compilation unit must
//! agree on `(batch_size, seq_len)`. A pretraining program has one loader, so
//! it proves; a program with two differently-shaped loaders declines and runs
//! the composite path. Over-collecting loaders (the walk descends into
//! branches that codegen would reject anyway) can only ever move the answer
//! toward "decline", never toward a wrong row count.
//!
//! # `drop_last` is load-bearing, not a detail
//!
//! With `drop_last = false` the final batch is SHORT. `rows` is baked into the
//! op at compile time, so a short batch would size the kernel's workspace and
//! grid for more rows than exist. The runtime does catch it —
//! `nsl_fused_lce_targets_i64_alloc` refuses when the label count disagrees
//! with the pinned `rows` — but catching it in the last step of an eight-hour
//! run is not the same as not inferring it. An unset `drop_last` defaults to
//! `true` in `expr/calls.rs`, and that default is mirrored here.

use std::collections::BTreeSet;

/// Item 4: `Passthrough` name prefix for a compile-time-constant shape list.
///
/// The dims follow as decimal, comma-separated: `const_shape:1,1024,49152`.
/// Encoded in the op name rather than as a new `PrimalOp` variant because the
/// op has no gradient, no inputs, and no saved tensors — a variant would add
/// an arm to every exhaustive `match` over `PrimalOp` (AD rules, saved-tensor
/// analysis, the arena, the lowerer, the pass registry's op tables) to say
/// "nothing" five times.
pub const CONST_SHAPE_PREFIX: &str = "const_shape:";

/// Build the `Passthrough` name for a proven shape.
pub fn const_shape_name(dims: &[i64]) -> String {
    let mut s = String::from(CONST_SHAPE_PREFIX);
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&d.to_string());
    }
    s
}

/// Parse the dims back out of a `const_shape:` op name.
///
/// A malformed entry yields an EMPTY shape, which lowers to an empty list —
/// visibly wrong at the first `ls[0]` rather than quietly plausible. Only
/// [`const_shape_name`] writes these names, so a parse failure means the two
/// have drifted, which `const_shape_name_round_trips` pins.
pub fn parse_const_shape_dims(name: &str) -> Vec<i64> {
    let Some(rest) = name.strip_prefix(CONST_SHAPE_PREFIX) else {
        return Vec::new();
    };
    if rest.is_empty() {
        return Vec::new();
    }
    let parsed: Option<Vec<i64>> = rest.split(',').map(|p| p.parse::<i64>().ok()).collect();
    parsed.unwrap_or_default()
}

use nsl_ast::visitor::{walk_stmt, Visitor};
use nsl_ast::{expr::ExprKind, stmt::Stmt, Module};
use nsl_lexer::Interner;

/// How aggressively the compiler may install a fused LM head that the user
/// did not ask for by name.
///
/// This is the "flag that authorizes the compiler rewrite" — the rewrite is
/// never silent-by-default, because it changes which kernel computes the loss
/// and (for `is_large` heads) which reduction order the dW accumulation uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LmHeadFusion {
    /// Never infer. An explicit `@fused_lm_ce` decorator still works.
    #[default]
    Off,
    /// Infer where the chain and the shapes can be proven; fall back to the
    /// composite path (with a `[fused-lm-ce]` note on stderr) where they
    /// cannot. This is what `--pretrain-optimized` selects.
    Auto,
    /// Infer, and make an unprovable chain a hard compile error. For CI and
    /// for anyone who would rather not discover at step 40,000 that the run
    /// has been paying for the logits surface all along.
    Require,
}

impl LmHeadFusion {
    /// Parse the `--fuse-lm-head` value. `None` for an unrecognised token —
    /// the CLI turns that into a clap error listing the three spellings.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "auto" => Some(Self::Auto),
            "require" => Some(Self::Require),
            _ => None,
        }
    }

    /// The spelling `--fuse-lm-head` accepts, for diagnostics that suggest a
    /// different mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::Require => "require",
        }
    }

    pub fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// The row-count facts a proven scan yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoaderFacts {
    pub batch_size: u32,
    pub seq_len: u32,
}

impl LoaderFacts {
    pub fn rows(self) -> u64 {
        self.batch_size as u64 * self.seq_len as u64
    }
}

/// Result of scanning a compilation unit for its training row count.
#[derive(Debug, Clone, PartialEq)]
pub enum LoaderScan {
    /// Every `DataLoader` agrees, and none of them can yield a short batch.
    Proven(LoaderFacts),
    /// Carries the reason, which is printed verbatim in the decline note so
    /// the user can see what to change.
    Unproven(String),
}

impl LoaderScan {
    pub fn facts(&self) -> Option<LoaderFacts> {
        match self {
            Self::Proven(f) => Some(*f),
            Self::Unproven(_) => None,
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Proven(_) => None,
            Self::Unproven(r) => Some(r),
        }
    }
}

/// One `DataLoader(...)` construction as the scan saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LoaderSite {
    /// `None` when the keyword was absent or not an integer literal.
    batch_size: Option<u32>,
    seq_len: Option<u32>,
    /// Mirrors `expr/calls.rs`: absent means `true`.
    drop_last: bool,
}

struct LoaderCollector<'a> {
    interner: &'a Interner,
    sites: Vec<LoaderSite>,
}

impl Visitor for LoaderCollector<'_> {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &nsl_ast::expr::Expr) {
        if let ExprKind::Call { callee, args } = &expr.kind
            && let ExprKind::Ident(sym) = &callee.kind
            && self.interner.resolve(sym.0) == Some("DataLoader")
        {
            let mut site = LoaderSite {
                batch_size: None,
                seq_len: None,
                drop_last: true,
            };
            for arg in args {
                let Some(name) = arg.name else { continue };
                // A negative or oversized literal is not a usable
                // dimension; leaving the slot `None` declines rather
                // than wrapping into a plausible-looking u32.
                let as_u32 = || match &arg.value.kind {
                    ExprKind::IntLiteral(v) if *v > 0 => u32::try_from(*v).ok(),
                    _ => None,
                };
                match self.interner.resolve(name.0) {
                    Some("batch_size") => site.batch_size = as_u32(),
                    Some("seq_len") => site.seq_len = as_u32(),
                    Some("drop_last") => {
                        // Anything that is not a literal `true` — a
                        // literal `false`, or a non-literal that
                        // `expr/calls.rs` will reject anyway — leaves
                        // the short-batch risk unrefuted.
                        site.drop_last =
                            matches!(&arg.value.kind, ExprKind::BoolLiteral(true));
                    }
                    _ => {}
                }
            }
            self.sites.push(site);
        }
        nsl_ast::visitor::walk_expr(self, expr);
    }
}

/// Scan a whole compilation unit for the training row count.
///
/// See the module docs for why this is unanimous-or-nothing.
pub fn scan_dataloaders(module: &Module, interner: &Interner) -> LoaderScan {
    let mut c = LoaderCollector {
        interner,
        sites: Vec::new(),
    };
    c.visit_module(module);

    if c.sites.is_empty() {
        return LoaderScan::Unproven(
            "no DataLoader(...) in this compilation unit, so the compiler cannot \
             prove the token-row count the fused kernel bakes in"
                .to_string(),
        );
    }

    // Report the FIRST unprovable site rather than a count: the user has to
    // go edit one line, and naming which fact is missing is what points at it.
    for site in &c.sites {
        if !site.drop_last {
            return LoaderScan::Unproven(
                "a DataLoader sets drop_last=false, so the final batch is short \
                 and batch_size * seq_len is not a constant row count"
                    .to_string(),
            );
        }
        if site.batch_size.is_none() {
            return LoaderScan::Unproven(
                "a DataLoader does not give batch_size as a positive integer literal"
                    .to_string(),
            );
        }
        if site.seq_len.is_none() {
            return LoaderScan::Unproven(
                "a DataLoader does not give seq_len as a positive integer literal"
                    .to_string(),
            );
        }
    }

    // `batch_size`/`seq_len` are both `Some` for every site by here.
    let shapes: BTreeSet<(u32, u32)> = c
        .sites
        .iter()
        .map(|s| {
            (
                s.batch_size.expect("checked above"),
                s.seq_len.expect("checked above"),
            )
        })
        .collect();
    if shapes.len() > 1 {
        let listed = shapes
            .iter()
            .map(|(b, s)| format!("{b}x{s}"))
            .collect::<Vec<_>>()
            .join(", ");
        return LoaderScan::Unproven(format!(
            "this compilation unit builds DataLoaders of differing shapes ({listed}); \
             the compiler will not guess which one feeds the train block"
        ));
    }

    let (batch_size, seq_len) = *shapes.iter().next().expect("non-empty");
    // `rows` is multiplied out into a u64 for the workspace sizing, but the op
    // carries u32 fields. A product that does not fit is not a shape any real
    // LM head has, and silently truncating it would mis-size the grid.
    if (batch_size as u64) * (seq_len as u64) > u32::MAX as u64 {
        return LoaderScan::Unproven(format!(
            "batch_size * seq_len ({batch_size} * {seq_len}) overflows the u32 row \
             count the fused op carries"
        ));
    }
    LoaderScan::Proven(LoaderFacts {
        batch_size,
        seq_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::FileId;

    fn scan(src: &str) -> LoaderScan {
        let mut interner = Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex: {lex_diags:?}");
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(parsed.diagnostics.is_empty(), "parse: {:?}", parsed.diagnostics);
        scan_dataloaders(&parsed.module, &interner)
    }

    #[test]
    fn a_single_literal_loader_proves_the_row_count() {
        let s = scan("let l = DataLoader(t, batch_size=2, seq_len=1024, shuffle=false)\n");
        assert_eq!(
            s,
            LoaderScan::Proven(LoaderFacts {
                batch_size: 2,
                seq_len: 1024
            })
        );
        assert_eq!(s.facts().unwrap().rows(), 2048);
    }

    #[test]
    fn drop_last_true_is_the_unset_default() {
        // `expr/calls.rs` inserts drop_last=true when the keyword is absent;
        // if that default ever flips, this test and the scan must move
        // together — which is the point of pinning it here.
        assert!(matches!(
            scan("let l = DataLoader(t, batch_size=1, seq_len=8)\n"),
            LoaderScan::Proven(_)
        ));
        assert!(matches!(
            scan("let l = DataLoader(t, batch_size=1, seq_len=8, drop_last=true)\n"),
            LoaderScan::Proven(_)
        ));
    }

    #[test]
    fn drop_last_false_is_unproven() {
        let s = scan("let l = DataLoader(t, batch_size=1, seq_len=8, drop_last=false)\n");
        assert!(s.reason().unwrap().contains("drop_last=false"), "{s:?}");
    }

    #[test]
    fn no_loader_at_all_is_unproven() {
        let s = scan("let x = 1\n");
        assert!(s.reason().unwrap().contains("no DataLoader"), "{s:?}");
    }

    #[test]
    fn a_missing_dimension_is_unproven_and_names_which_one() {
        assert!(scan("let l = DataLoader(t, seq_len=8)\n")
            .reason()
            .unwrap()
            .contains("batch_size"));
        assert!(scan("let l = DataLoader(t, batch_size=4)\n")
            .reason()
            .unwrap()
            .contains("seq_len"));
    }

    #[test]
    fn a_non_positive_dimension_is_unproven() {
        // 0 rows would divide by zero in the mean reduction; a negative
        // literal is not representable. Both must decline rather than
        // wrap or produce a degenerate grid.
        assert!(scan("let l = DataLoader(t, batch_size=0, seq_len=8)\n")
            .reason()
            .is_some());
        assert!(scan("let l = DataLoader(t, batch_size=-1, seq_len=8)\n")
            .reason()
            .is_some());
    }

    #[test]
    fn two_agreeing_loaders_still_prove() {
        // Train + eval loaders of the same shape are the common case and
        // must not decline — unanimity is about disagreement, not count.
        let s = scan(
            "let tr = DataLoader(a, batch_size=2, seq_len=64)\n\
             let ev = DataLoader(b, batch_size=2, seq_len=64)\n",
        );
        assert!(matches!(s, LoaderScan::Proven(_)), "{s:?}");
    }

    #[test]
    fn two_disagreeing_loaders_decline_and_list_both() {
        let s = scan(
            "let tr = DataLoader(a, batch_size=2, seq_len=64)\n\
             let ev = DataLoader(b, batch_size=8, seq_len=64)\n",
        );
        let r = s.reason().unwrap();
        assert!(r.contains("2x64") && r.contains("8x64"), "{r}");
    }

    #[test]
    fn a_loader_nested_in_a_branch_is_still_collected() {
        // The walk must not stop at straight-line scope: a loader codegen
        // would reject still counts as disagreement here, and declining is
        // the safe direction.
        let s = scan(
            "let tr = DataLoader(a, batch_size=2, seq_len=64)\n\
             if flag:\n    let ev = DataLoader(b, batch_size=9, seq_len=64)\n",
        );
        assert!(s.reason().unwrap().contains("9x64"), "{s:?}");
    }

    #[test]
    fn a_loader_inside_a_train_data_section_is_collected() {
        // TrainSection::Data is a statement list the plain block walk does
        // not reach; `walk_stmt` handles it, and a loader declared there is
        // exactly the one feeding the block.
        let s = scan(
            "train(model=m, epochs=1):\n\
             \x20   data:\n\
             \x20       let l = DataLoader(t, batch_size=3, seq_len=32)\n\
             \x20   step(batch):\n\
             \x20       let loss = 0\n",
        );
        assert_eq!(
            s,
            LoaderScan::Proven(LoaderFacts {
                batch_size: 3,
                seq_len: 32
            })
        );
    }

    #[test]
    fn const_shape_name_round_trips() {
        for dims in [vec![1i64, 1024, 49152], vec![2048, 4096], vec![7]] {
            assert_eq!(parse_const_shape_dims(&const_shape_name(&dims)), dims);
        }
    }

    #[test]
    fn a_malformed_const_shape_parses_to_nothing() {
        // Empty is the loud answer: `ls[0]` on an empty list fails at the
        // first use, where a partial parse would produce a plausible-looking
        // wrong shape and mis-size the flatten.
        assert!(parse_const_shape_dims("const_shape:").is_empty());
        assert!(parse_const_shape_dims("const_shape:1,x,3").is_empty());
        assert!(parse_const_shape_dims("shape").is_empty());
    }

    #[test]
    fn mode_parsing_round_trips_and_rejects_junk() {
        for m in [LmHeadFusion::Off, LmHeadFusion::Auto, LmHeadFusion::Require] {
            assert_eq!(LmHeadFusion::parse(m.as_str()), Some(m));
        }
        assert_eq!(LmHeadFusion::parse("on"), None);
        assert_eq!(LmHeadFusion::parse(""), None);
        assert!(!LmHeadFusion::default().is_on());
    }
}
