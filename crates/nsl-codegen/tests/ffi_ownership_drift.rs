//! Drift gates binding the THREE ownership authorities to each other
//! (2026-07-29). The recurring bug class of the ownership campaign was
//! silent divergence between:
//!
//!   1. the owning-ref Ident ALLOWLIST (`expr_result_is_owned_temporary`,
//!      nsl-codegen/src/expr/mod.rs) — three separate leak cycles came
//!      from names missing here (`scaled_dot_product_attention`, then
//!      `rmsnorm`/`dropout`, then `lt_scalar`/`multinomial`), and one
//!      entry (`"slice"`) was dead;
//!   2. the FFI ownership TABLE (`ffi_ownership.rs`) — carried a
//!      documented-wrong `nsl_tensor_slice` borrow entry and a dead
//!      `nsl_tensor_concat` key (the extern is `nsl_tensor_cat`);
//!   3. the RUNTIME's actual `pub extern "C"` inventory.
//!
//! These gates make each divergence a test failure:
//!   - the allowlist parsed from source must equal EXPECTED's keys
//!     exactly (add a name without classifying it here → red);
//!   - every mapped name's FFI must be in the table as OwnedNewResult
//!     (allowlist says owning, table must agree);
//!   - every table key must name a real runtime extern (dead entries →
//!     red).
//!
//! Full emission-site registration through `register_ffi_result_
//! ownership` (making the table the single runtime authority) requires
//! threading `state` through `compile_call_by_name` — queued as the v2
//! refactor; these bindings close the drift class first.
//!
//! Known limits, stated plainly: the name→FFI edge in EXPECTED is
//! hand-asserted, not derived from dispatch — a wrong claim naming a
//! real owned-classified extern would pass (the tanh miswiring was
//! caught only because its extern did not exist at all). And a new
//! dispatch arm added WITHOUT an allowlist entry — the original leak
//! class — is invisible to all three bindings. Both are the v2
//! refactor's job.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use nsl_codegen::ffi_ownership::{ffi_ownership_kind, ffi_ownership_table_keys, FfiOwnershipKind};

enum Expected {
    /// The allowlist name dispatches 1:1 to this runtime FFI, which must
    /// be classified OwnedNewResult in the table.
    Ffi(&'static str),
    /// No single FFI produces the result; the reason documents why the
    /// allowlist entry is still sound.
    Exception(&'static str),
}

/// One entry per allowlist name. Kept in the allowlist's textual order
/// where practical; the assertion is set-based.
fn expected() -> BTreeMap<&'static str, Expected> {
    use Expected::*;
    let mut m = BTreeMap::new();
    // Unary/scalar math family
    m.insert("exp", Ffi("nsl_tensor_exp"));
    m.insert("log", Ffi("nsl_tensor_log"));
    m.insert("sqrt", Ffi("nsl_tensor_sqrt"));
    m.insert("abs", Ffi("nsl_tensor_abs"));
    m.insert("sign", Ffi("nsl_tensor_sign"));
    m.insert("neg", Ffi("nsl_tensor_neg"));
    // Creation intrinsics
    m.insert("zeros", Ffi("nsl_tensor_zeros"));
    m.insert("ones", Ffi("nsl_tensor_ones"));
    m.insert("full", Ffi("nsl_tensor_full"));
    m.insert("rand", Ffi("nsl_tensor_rand"));
    m.insert("randn", Ffi("nsl_tensor_randn"));
    m.insert("arange", Ffi("nsl_tensor_arange"));
    m.insert("zeros_like", Ffi("nsl_tensor_zeros_like"));
    // Activations / softmax family
    m.insert("relu", Ffi("nsl_tensor_relu"));
    m.insert("gelu", Ffi("nsl_tensor_gelu"));
    m.insert("silu", Ffi("nsl_tensor_silu"));
    m.insert("sigmoid", Ffi("nsl_tensor_sigmoid"));
    m.insert("tanh", Ffi("nsl_tensor_tanh_act"));
    m.insert("softmax", Ffi("nsl_tensor_softmax"));
    m.insert("log_softmax", Ffi("nsl_tensor_logsoftmax"));
    m.insert("tensor_sin", Ffi("nsl_tensor_sin"));
    m.insert("tensor_cos", Ffi("nsl_tensor_cos"));
    m.insert("rotate_half", Ffi("nsl_tensor_rotate_half"));
    // Reductions
    m.insert("mean", Ffi("nsl_tensor_mean"));
    m.insert("sum", Ffi("nsl_tensor_sum"));
    // Attention family: lowered as multi-call chains whose final call
    // (fresh matmul / flash kernel output) owns the result — no single
    // FFI to bind. Verified in the allowlist's own comment block.
    m.insert(
        "scaled_dot_product_attention",
        Exception("chain lowering; result is the final fresh matmul/flash output"),
    );
    m.insert(
        "scaled_dot_product_attention_masked",
        Exception("chain lowering; result is the final fresh matmul/flash output"),
    );
    m.insert(
        "scaled_dot_product_attention_packed",
        Exception("chain lowering; result is the final fresh matmul/flash output"),
    );
    // Shape/view + materialise family
    m.insert("contiguous", Ffi("nsl_tensor_contiguous"));
    m.insert("unsqueeze", Ffi("nsl_tensor_unsqueeze"));
    m.insert("tensor_slice", Ffi("nsl_tensor_slice"));
    m.insert("stack", Ffi("nsl_tensor_stack"));
    m.insert("tensor_cat", Ffi("nsl_tensor_cat"));
    m.insert("cumsum", Ffi("nsl_tensor_cumsum"));
    m.insert("argmax", Ffi("nsl_tensor_argmax"));
    m.insert("causal_mask", Ffi("nsl_tensor_causal_mask"));
    // Stdlib return-position builtins (PR #424)
    m.insert("rmsnorm", Ffi("nsl_tensor_rmsnorm"));
    m.insert("dropout", Ffi("nsl_tensor_dropout"));
    m.insert("layernorm", Ffi("nsl_tensor_layernorm"));
    m.insert("bias_add", Ffi("nsl_tensor_bias_add"));
    m.insert("gather", Ffi("nsl_tensor_gather"));
    m.insert("embedding_lookup", Ffi("nsl_tensor_embedding_lookup"));
    // Sampling family (PR #426)
    m.insert("lt_scalar", Ffi("nsl_tensor_lt_scalar"));
    m.insert("multinomial", Ffi("nsl_tensor_multinomial"));
    m
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Extract the Ident-allowlist names from the `return matches!(` block in
/// expr/mod.rs. Comment lines are skipped BEFORE quote extraction — the
/// block's comments quote names (`a "slice" entry here is dead`) that
/// must not count.
fn parse_allowlist() -> BTreeSet<String> {
    let src = std::fs::read_to_string(manifest_dir().join("src/expr/mod.rs"))
        .expect("read expr/mod.rs");
    let start = src
        .find("return matches!(")
        .expect("allowlist matches! block not found — if it moved, update this gate");
    let block = &src[start..];
    let mut names = BTreeSet::new();
    // Line-based walk with a line-exact terminator: a comment inside the
    // block contains the SUBSTRING `");` (a quoted spelling note), which
    // a naive find(");") mistakes for the end.
    for line in block.lines() {
        let t = line.trim();
        if t == ");" {
            break;
        }
        if t.starts_with("//") {
            continue;
        }
        let mut rest = t;
        while let Some(q1) = rest.find('"') {
            let after = &rest[q1 + 1..];
            let Some(q2) = after.find('"') else { break };
            names.insert(after[..q2].to_string());
            rest = &after[q2 + 1..];
        }
    }
    assert!(
        names.len() > 20,
        "allowlist extraction found only {} names — parser drift, fix this gate",
        names.len()
    );
    names
}

/// Collect every `pub extern "C" fn NAME` in nsl-runtime.
fn runtime_externs() -> BTreeSet<String> {
    fn walk(dir: &Path, out: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("read runtime src dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let src = std::fs::read_to_string(&path).expect("read runtime source");
                for chunk in src.split("pub extern \"C\" fn ").skip(1) {
                    let name: String = chunk
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.insert(name);
                    }
                }
            }
        }
    }
    let root = manifest_dir().join("../nsl-runtime/src");
    let mut out = BTreeSet::new();
    walk(&root, &mut out);
    assert!(
        out.len() > 100,
        "runtime extern extraction found only {} names — parser drift",
        out.len()
    );
    out
}

/// Binding 1: the allowlist and EXPECTED must be the same set — a new
/// allowlist name must be classified here (and thus in the table), and a
/// removed one must be cleaned up here.
#[test]
fn allowlist_names_are_exactly_the_classified_set() {
    let allow = parse_allowlist();
    let exp: BTreeSet<String> = expected().keys().map(|s| s.to_string()).collect();
    let missing: Vec<_> = allow.difference(&exp).collect();
    let stale: Vec<_> = exp.difference(&allow).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "allowlist/classification drift.\nIn allowlist but unclassified here: {missing:?}\n\
         Classified here but gone from the allowlist: {stale:?}\n\
         Every owning-ref allowlist name needs a mapped FFI (verified\n\
         per-runtime) or a documented Exception in this gate's EXPECTED."
    );
}

/// Binding 2: every mapped FFI is in the ownership table as
/// OwnedNewResult — the table and the allowlist must agree that these
/// results are fresh owning references.
#[test]
fn mapped_ffis_are_classified_owned_in_the_table() {
    let mut wrong = Vec::new();
    for (name, exp) in expected() {
        match exp {
            Expected::Ffi(ffi) => match ffi_ownership_kind(ffi) {
                Some(FfiOwnershipKind::OwnedNewResult) => {}
                other => wrong.push(format!("{name} -> {ffi}: {other:?}")),
            },
            Expected::Exception(reason) => {
                assert!(
                    !reason.is_empty(),
                    "{name} is excepted with no documented reason"
                );
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "allowlist names whose FFI the ownership table does not classify \
         OwnedNewResult (the allowlist frees these results — a disagreeing \
         table entry is either a leak or a UAF depending on direction):\n{}",
        wrong.join("\n")
    );
}

/// Binding 3: every table key names a REAL runtime extern. A dead key
/// (`nsl_tensor_concat` vs the actual `nsl_tensor_cat`) classifies
/// nothing while reading as coverage.
#[test]
fn every_table_key_is_a_runtime_extern() {
    let externs = runtime_externs();
    let dead: Vec<_> = ffi_ownership_table_keys()
        .into_iter()
        .filter(|k| !externs.contains(*k))
        .collect();
    assert!(
        dead.is_empty(),
        "ffi_ownership table keys that name no runtime `pub extern \"C\"` \
         fn (dead entries): {dead:?}"
    );
}
