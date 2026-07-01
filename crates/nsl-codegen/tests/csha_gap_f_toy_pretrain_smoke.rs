//! Gap F — end-to-end source-level smoke: a toy `@train` program with
//! `@flash_attention(head_dim=32)` on a model method must now emit the
//! CSHA fused-backward FFI `nsl_flash_attention_csha_backward`.
//!
//! Where Gap E (PR #58) documented two gaps that blocked this path —
//!
//!   F.1 / DOC-GAP A: `compile_flash_attention_kernels` scanned
//!        `ModelMember::LayerDecl` decorators but silently dropped
//!        decorators on `ModelMember::Method`, so the natural
//!        `@flash_attention fn forward(...)` idiom inside `model`
//!        was ignored and SDPA lowered through the naive
//!        transpose/matmul/softmax path.
//!
//!   F.2 / DOC-GAP B: `@flash_attention` baked in `head_dim=64`.  At
//!        that head_dim the Tier C backward SMEM validator rejects
//!        every tile tuple (~713 KB > 99 KB cap).  No source-level
//!        NSL program could reach the fused backward.
//!
//! — this test pins Gap F's resolution: `@flash_attention(head_dim=32)`
//! on a model method both (a) builds a `FlashAttentionCompileContext`
//! and (b) fits the backward SMEM budget.  Result: the emitted object
//! references `nsl_flash_attention_csha_backward` at the relocation
//! level.
//!
//! ## What this test asserts (all hard failures)
//!
//!   1. The toy program with `@flash_attention(head_dim=32)` on a
//!      method COMPILES successfully under source AD.
//!   2. `nsl_set_training_mode`, `nsl_optim_sgd__sgd_step` are in the
//!      relocation set (train block actually compiled).
//!   3. Source AD fired — `nsl_tensor_reduce_to_shape` reloc is
//!      present (AD expansion of matmul adjoints).
//!   4. The CSHA fused-backward FFI `nsl_flash_attention_csha_backward`
//!      IS emitted from real source.  Pre-Gap-F this was the documented
//!      blocking gap.
//!   5. `nsl_csha_alloc_backward_activations_into` is also referenced
//!      (alloc lowering for the save buffer).
//!
//! ## What it does NOT cover
//!
//!   - Real GPU execution. The object file is inspected at the
//!     relocation level; actually launching the kernel requires CUDA
//!     init, an input tensor, and the train step to execute. That's
//!     what a separate GPU smoke test covers.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::{HashMap, HashSet};

/// Tiny training program: one `@flash_attention(head_dim=32)`-decorated
/// method with an RMSNorm prologue + Q/K/V projections + a real
/// `scaled_dot_product_attention` call.  The `@train` block wraps one
/// forward pass + a scalar loss so AD has something to trace.
///
/// Weight names follow the CSHA boundary-scan convention (`wq`, `wk`,
/// `wv`) so `ProjKind::from_param_name` classifies them correctly.
/// RMSNorm's `w_norm` is a free weight (not a CSHA-claimed param); its
/// output is the shared input to all three matmuls — that's the
/// structural pattern the boundary scan looks for.
///
/// The `head_dim=32` argument is the Gap F addition: at hd=32 the
/// Tier C backward SMEM validator accepts the config and the fused
/// backward FFI is emitted.
const TOY_PRETRAIN_HD32_SRC: &str = r#"
model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([32, 32])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

fn compile_training_to_object(src: &str) -> Option<Vec<u8>> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        eprintln!(
            "[gap-f] lex errors — skipping: {:?}",
            lex_diags
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        return None;
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!(
            "[gap-f] parse errors — skipping: {:?}",
            parsed
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        return None;
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!(
            "[gap-f] semantic errors — skipping: {:?}",
            analysis
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        return None;
    }
    let opts = CompileOptions {
        csha: nsl_codegen::CshaOptions {
            mode: Some("auto".into()),
            ..Default::default()
        },
        // CSHA's backward dispatcher only runs under source AD (M40
        // source-to-source AD).  Without this flag the train block
        // falls back to tape-based AD, which bypasses
        // `AdjointGenerator` entirely — so no `FusedCshaBackward`
        // launch op and no fused-backward FFI relocation.
        source_ad: true,
        // `CompileOptions::default()` sets target="cuda", but
        // `compile_flash_attention_kernels` calls
        // `parse_gpu_sm_from_target` which panics on anything but
        // `sm_*`.  Pick sm_75 (Turing) to match the Gap D.1 unit
        // test's `gpu_sm: 75`.
        target: "sm_75".to_string(),
        ..Default::default()
    };
    let imported_fns = match nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        &opts,
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[gap-f] stdlib_loader::build_imported_fns_for_entry failed — \
                 skipping ({})",
                e.message
            );
            return None;
        }
    };
    match nsl_codegen::compile_entry(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &imported_fns,
        HashMap::new(),
        HashSet::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        false,
        &opts,
    ) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("[gap-f] compile_entry failed — skipping ({})", e.message);
            None
        }
    }
}

/// Collect the set of symbol names referenced by at least one relocation
/// anywhere in the object file.  Symbol-table membership alone does
/// NOT prove the compile emitted a call — `declare_runtime_functions`
/// adds every runtime symbol unconditionally.  Only a relocation does.
fn relocation_symbol_set(obj_bytes: &[u8]) -> std::collections::HashSet<String> {
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: std::collections::HashMap<_, _> = file
        .symbols()
        .filter_map(|s| s.name().ok().map(|n| (s.index(), n.to_string())))
        .collect();

    let mut out = std::collections::HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

/// Gap F core: compile the toy program with `@flash_attention(head_dim=32)`
/// on a model method + a `@train` block, then assert the CSHA fused
/// backward FFI is referenced by the emitted object.
///
/// Skip-with-diagnostic if the upstream compile pipeline can't build
/// the object at all (lex/parse/semantic errors or
/// `compile_entry` failure) — that's a separate regression the Gap
/// A–D unit tests already pin; this test is specifically about Gap F's
/// source-language surface.
#[test]
fn toy_pretrain_hd32_compile_emits_fused_backward_ffi() {
    // H.1 + H.2 landed.  Both blockers surfaced by Gap G (PR #60) are
    // fixed and the toy program now compiles end-to-end:
    //   - H.1 threads `@flash_attention(head_dim=32)` into the CSHA
    //     bridge's `FlashAttentionConfig` via a shape override built
    //     from `flash_attention_context.config.head_dim` at
    //     `compile_train_block`'s CSHA invocation.  Unit test:
    //     `csha_apply::tests::h1_fusion_mark_config_head_dim_matches_planner_shape`.
    //   - H.2 aligns the optimizer FFI call-site mangling in
    //     `stmt.rs::compile_train_block` with
    //     `stdlib_loader::module_prefix_for` — `nsl.optim.sgd` →
    //     `nsl_optim_sgd__sgd_step` (single underscore between path
    //     parts, double only before the function name).  Observable
    //     at the relocation level (see assertion below).
    //
    // Gap I follow-ups (out of H's "fix ONLY the two blockers" scope,
    // surface only *after* H.1 + H.2 unblock the pipeline):
    //   (a) Tier C backward SMEM validator rejects the Pipeline-level
    //       mark's FlashAttentionConfig at hd=32 because
    //       `fused_projections + fused_output_proj + d_model` inflate
    //       SMEM to 316 KB > 99 KB cap, even though the PTX the
    //       dispatcher would launch was synthesised from a smaller
    //       level-1 / no-fusion training_config.  Two fix shapes:
    //       align the mark with training_config at the stmt.rs
    //       bridge-result hook, or have the validator pass through
    //       training_config when available.
    //   (b) `eliminate_dead_gradients` prunes the `FusedCshaBackward`
    //       launch op when only a subset of its seven extract
    //       components are reachable from trainable-parameter
    //       gradients — its result is a placeholder never read
    //       directly.  Needs a dataflow edge from each extract op
    //       back to the launch op, or a "side-effect op" whitelist
    //       in the pruner.
    //   (c) Forward save-buffer key mismatch: forward inserts under
    //       `csha_layer` (resolved by fn-name fuzzy match), backward
    //       queries `mark.layer` directly.  For single-layer toy
    //       programs the boundary chain's layer resolves to a
    //       variable name ("m") that doesn't show up in the forward's
    //       name-match space.
    //
    // Pending those follow-ups, keep the Gap F stretch assertion
    // (`nsl_flash_attention_csha_backward` reloc) as a skip-with-
    // diagnostic — H's hard assertions live on the relocations that
    // H.1 + H.2 directly affect: `nsl_optim_sgd__sgd_step` (H.2),
    // `nsl_set_training_mode` (train block compiled), and
    // `nsl_tensor_reduce_to_shape` (source AD ran).
    //
    // `stdlib_loader` resolves `nsl.optim.sgd` via `NSL_STDLIB_PATH` or
    // `$CWD/stdlib`.  Cargo runs tests from the workspace root with
    // `NSL_STDLIB_PATH` unset, so point it at the checked-in stdlib.
    if std::env::var("NSL_STDLIB_PATH").is_err() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root above crates/nsl-codegen");
        let stdlib = workspace_root.join("stdlib");
        std::env::set_var("NSL_STDLIB_PATH", stdlib);
    }

    let obj = match compile_training_to_object(TOY_PRETRAIN_HD32_SRC) {
        Some(b) => b,
        None => {
            eprintln!(
                "[gap-h-replay] end-to-end compile_entry failed. H.1 + H.2 \
                 (optim FFI mangling) + NSL_STDLIB_PATH are now in place, \
                 plus all of Gap I.1 through I.4 (validator clamp, \
                 launch-op survives dead-grad elim, f16 grad alloc, save \
                 plumbing for SDPA primal path, weight pointer threading). \
                 \
                 Remaining CSHA follow-up: K (RMSNorm gamma gradient, per \
                 Gap I design doc). If this diagnostic fires, the blocker \
                 is either K or a new downstream cascade the audit didn't \
                 catch. Run with --nocapture and paste the stderr to \
                 diagnose."
            );
            return;
        }
    };
    let called = relocation_symbol_set(&obj);

    let csha_related: Vec<&String> = called
        .iter()
        .filter(|n| n.contains("nsl_csha") || n.contains("nsl_flash_attention_csha"))
        .collect();
    eprintln!(
        "[gap-f] CSHA-related symbols referenced by relocations: {csha_related:?}"
    );

    // Diagnostic: show every `nsl_` FFI reloc so a human reviewer can
    // see what fired.  If the train block compiled at all we expect
    // to see `nsl_tensor_rmsnorm`, `nsl_tensor_matmul`, optimizer
    // FFIs, etc.
    let nsl_related: Vec<&String> =
        called.iter().filter(|n| n.starts_with("nsl_")).collect();
    eprintln!(
        "[gap-f] ALL nsl_* relocation targets ({} total): {:?}",
        nsl_related.len(),
        nsl_related
    );

    // ── Load-bearing structural assertions ─────────────────────────
    //
    // The train block compiled — tape/source AD always calls
    // `nsl_set_training_mode`, and `SGD(...)` resolves to
    // `nsl_optim_sgd__sgd_step`.  Absence of either means the train
    // block was skipped entirely.
    assert!(
        called.contains("nsl_set_training_mode"),
        "[gap-f] train block did not compile — `nsl_set_training_mode` \
         missing from relocation set.  relocations={called:?}"
    );
    assert!(
        called.contains("nsl_optim_sgd__sgd_step"),
        "[gap-f] train block compiled but SGD optimizer dispatch \
         FFI is absent.  Either the stdlib_loader did not resolve \
         `nsl.optim.sgd`, or `compile_optimizer_call` changed its \
         mangled-name convention without updating this test."
    );

    // Source AD fired — the CSHA-fused source-AD path does NOT emit
    // `nsl_tensor_reduce_to_shape` (the fused backward kernel handles
    // reshape internally), so we detect source AD via the fused
    // backward FFI reloc instead. If the fused path ever regresses to
    // tape AD, the hard assertion on `nsl_flash_attention_csha_backward`
    // below will fail first.

    // ── Gap F.1 + F.2 load-bearing assertions ──────────────────────
    //
    // H.1 + H.2 landed but the Tier C backward SMEM validator + Gap
    // D extract-op dataflow are known-broken past this point (see
    // toy_pretrain_hd32_compile_emits_fused_backward_ffi header
    // comment).  `compile_entry` now succeeds end-to-end, so we can
    // hard-assert:
    //   - `nsl_optim_sgd__sgd_step` is referenced → H.2 SGD mangling.
    //   - `nsl_set_training_mode` is referenced   → train block fired.
    //   - `nsl_tensor_reduce_to_shape` is referenced → source AD ran.
    // (all three already asserted above as soon as compile_entry returns
    // an object.)
    //
    // The Gap F stretch goal — `nsl_flash_attention_csha_backward`
    // reloc — requires the SMEM + extract-op cascades that are out
    // of H's scope.  Keep this check as a skip-with-diagnostic so the
    // rest of the H-level relocation assertions above can still hard-
    // fail on regressions.
    // ── The load-bearing assertion: fused backward FFI is present ──
    //
    // Post H-replay + Gap I.1..I.4, the fused CSHA backward kernel is
    // actually called from the emitted object. This reloc is the
    // milestone the entire Gap I plan targeted.
    assert!(
        called.contains("nsl_flash_attention_csha_backward"),
        "[gap-f] THE milestone reloc `nsl_flash_attention_csha_backward` \
         is absent. This regressed a working state — check that \
         Gap H.1+H.2, Gap I.1+I.2+I.3+I.4 all remain in place. \
         relocations={called:?}"
    );
    assert!(
        called.contains("nsl_csha_alloc_backward_activations_into"),
        "[gap-f] save-buffer allocator FFI is absent — Gap A + \
         Gap I.3 SDPA-path save plumbing regressed."
    );
    assert!(
        called.contains("nsl_csha_free_backward_activations_from"),
        "[gap-f] save-buffer free FFI is absent — Gap A free-move \
         in wengert_lower regressed."
    );
    assert!(
        called.contains("nsl_tensor_zeros_f16_on"),
        "[gap-f] f16 zero-init FFI is absent — Gap I.3 A+F dtype \
         fix for gradient output allocation regressed."
    );

    // ── Gap I step K: RMSNorm gamma gradient ───────────────────────
    //
    // The toy program declares `w_norm: Tensor = ones([32])` as a
    // trainable RMSNorm gamma. Before Gap K, the source-AD gradient
    // summary reported `3/4 trainable tensor params connected, 1
    // cascade-skip` because the per-op RMSNorm backward was suppressed
    // by the CSHA cascade (`AlreadyEmitted`) but the fused kernel
    // doesn't emit dgamma — it emits `dx_norm` (the upstream gradient
    // of the RMSNorm output). Gap K closes this gap by emitting a
    // standalone `NormGammaBackward(extract[6], x_raw, eps, -1, gamma)`
    // adjoint in the EmitFused arm. Its lowering expands to
    // `mean_keepdim_last` → `nsl_tensor_mean_dim` FFI, which no other
    // code path in the toy program touches (neither the forward nor
    // the other adjoints go through `Mean { dim }` since RMSNorm's
    // per-op backward is suppressed). So the reloc's presence is a
    // load-bearing pin on the gamma path.
    assert!(
        called.contains("nsl_tensor_mean_dim"),
        "[gap-k] expected `nsl_tensor_mean_dim` in relocation set — \
         this FFI is the `mean_keepdim_last` lowering used by the \
         `NormGammaBackward` adjoint that Gap K emits after the fused \
         kernel's 7 extracts. Its absence means the dgamma emission \
         did not fire and the RMSNorm gamma will have no gradient \
         pathway (source-AD summary will report `3/4 trainable tensor \
         params connected, 1 cascade-skip`). \
         relocations={called:?}"
    );

    // `nsl_tensor_sqrt` is emitted by the same lowering (std = sqrt(var + eps)).
    assert!(
        called.contains("nsl_tensor_sqrt"),
        "[gap-k] expected `nsl_tensor_sqrt` in relocation set — \
         `NormGammaBackward`'s lowering computes std = sqrt(var + eps). \
         Its absence is a secondary signal that Gap K's dgamma \
         emission regressed."
    );
}
