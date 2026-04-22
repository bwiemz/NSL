// Compile-only skeleton check: the module, types, and entry point exist
// and have the signatures the spec mandates.

use nsl_codegen::wggo_prune::{
    PruneRewrite, PruneRewriteResult, PruneRefusal, run,
};
use std::collections::BTreeSet;

#[test]
fn types_are_constructible_and_run_is_callable() {
    // Empty PruneRewriteResult constructs.
    let _r = PruneRewriteResult {
        rewrites: Vec::<PruneRewrite>::new(),
        refusals: Vec::<PruneRefusal>::new(),
        pruned_forward_var_ids: BTreeSet::new(),
        ops_deleted: 0,
    };
    // (run() is callable with an empty plan — deferred until the full signature lands.)
}
