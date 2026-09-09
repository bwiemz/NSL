//! Section 6e of the train block's source-AD arm: the Milestone C·p2
//! transient-memory arena projection. Over the *final* forward+adjoint
//! tape (post-CCR splice, post-adjoint-last-use frees) it derives the
//! Stage-2A element hints (semantic-typed shapes, initializer-derived
//! model-field dims, adjoints mirrored from their unique primal), renders
//! the `--memory-report` / `NSL_ARENA_REPORT=1` arena report through the
//! M36 interference/BFD engine (`transient_arena.rs`), and under
//! `--transient-arena` computes the Stage-2B placement, publishing it on
//! the compiler and declaring the arena and its slot geometry to the
//! runtime (`nsl_arena_init` / `nsl_arena_declare_slot`).
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 573 lines,
//! 5 inputs ([`TransientArenaInputs`]) plus the function builder.
//! Returns the element hints, which the driver's CSLA schedule precompute
//! shares. The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`)
//! pin the arena declarations on the `--transient-arena` fixture.

use std::collections::HashMap;

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::wengert::VarId;

/// Every binding of `compile_train_block_inner` the arena projection
/// reads; names are the driver's.
pub(crate) struct TransientArenaInputs<'a> {
    /// The final adjoint tape (post-CCR splice and last-use frees).
    pub(crate) adjoint: &'a crate::wengert::WengertList,
    /// Whether the CSLA window schedule is active (its calibration shares the element hints).
    pub(crate) csla_active: bool,
    /// The forward tape the adjoint was generated from.
    pub(crate) effective_primal: &'a crate::wengert::WengertList,
    /// The forward extractor (var nodes, named parameters, known dims).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The adjoint generator (primal → adjoint VarId map).
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
}

impl Compiler<'_> {
    /// Project the transient-memory arena over the final tape (see the
    /// module header) and return the Stage-2A element hints.
    pub(crate) fn emit_transient_arena_projection(
        &mut self,
        builder: &mut FunctionBuilder,
        inputs: TransientArenaInputs<'_>,
    ) -> Result<HashMap<VarId, u64>, CodegenError> {
        let TransientArenaInputs {
            adjoint,
            csla_active,
            effective_primal,
            extractor,
            generator,
        } = inputs;

        // 6e. Milestone C·p2: transient-memory arena projection.
        // Reuses the M36 interference/BFD engine (transient_arena.rs)
        // over the *final* forward+adjoint tape — post-CCR-splice and
        // post-adjoint-last-use-frees, so FreeTensor markers bound each
        // interval exactly. This is the backward+forward transient
        // surface the M36 slab planner (AST, forward-only) never sees.
        // Pure analysis; no codegen change. Gated by --memory-report or
        // NSL_ARENA_REPORT=1.
        // Stage-2A shape hints, shared by the arena report and the
        // CSLA layerwise calibration below. Three SOUND layers, in
        // priority order:
        //   1. semantic-typed shapes (annotated Tensor<[..]> values
        //      — near-empty on today's corpus since model fields are
        //      typed from annotations only);
        //   2. initializer-derived model-field dims for trainable
        //      params (the unique-field-name bridge; covers the
        //      dominant `randn([64, 128]) * 0.15` idiom);
        //   3. each sized primal mirrored onto its adjoint
        //      accumulator (an adjoint has its primal's shape by
        //      construction) — this is what sizes the backward's
        //      gradient transients, the surface the arena exists
        //      to place.
        // Symbolic/computed dims stay unsized; nothing is guessed.
        let arena_place_on = self.compile_options.memory.transient_arena;
        let arena_report_on = self.compile_options.memory.report
            || arena_place_on
            || std::env::var("NSL_ARENA_REPORT").ok().as_deref() == Some("1");
        let elem_hints: HashMap<VarId, u64> =
            if arena_report_on || csla_active {
                let mut hints = crate::profiling::captures::elem_hints_from_var_nodes(
                    extractor.var_nodes(),
                    self.type_map,
                );
                let field_elems = self.models.unique_field_elems();
                for (name, vid) in extractor.named_param_var_ids() {
                    if hints.contains_key(vid) {
                        continue;
                    }
                    let leaf = name.rsplit('.').next().unwrap_or(name);
                    if let Some(&e) = field_elems.get(leaf) {
                        hints.insert(*vid, e);
                    }
                }
                // Mirror ONLY adjoint accumulators with a UNIQUE
                // primal preimage. Add/Sub rules alias the OUTPUT's
                // adjoint onto both operands (Identity — no reduce
                // op), so a shared accumulator carries the OUTPUT's
                // shape and mirroring it would mis-size a broadcast
                // operand's grad. Reducing rules emit a dedicated
                // reduce_to_shape var per operand, which is exactly
                // what a unique preimage certifies.
                let mut preimage: std::collections::HashMap<crate::wengert::VarId, u32> =
                    Default::default();
                for a in generator.adjoint_vars_map().values() {
                    *preimage.entry(*a).or_default() += 1;
                }
                let mirrored: Vec<(crate::wengert::VarId, u64)> = generator
                    .adjoint_vars_map()
                    .iter()
                    .filter(|(_, a)| preimage.get(a) == Some(&1))
                    .filter_map(|(p, a)| hints.get(p).map(|&e| (*a, e)))
                    .collect();
                for (a, e) in mirrored {
                    hints.entry(a).or_insert(e);
                }
                hints
            } else {
                Default::default()
            };

        if arena_report_on {
            // Stage-2A: partially quantified — sized transients get
            // real bytes (a lower bound on the full arena), the rest
            // still report as concurrency-only. Param-gradient
            // accumulators and the loss ESCAPE the tape (read by the
            // optimizer emission / callbacks, never by a tape op) —
            // without the escape pin, last-use liveness gives them
            // point intervals and BFD time-shares every gradient in
            // one slot, an illegal aliasing that fakes the savings.
            // Stage-2B assigns offsets.
            let param_vids: std::collections::HashSet<crate::wengert::VarId> = extractor
                .named_param_var_ids()
                .iter()
                .map(|(_, v)| *v)
                .collect();
            let mut tape_escaping: std::collections::HashSet<crate::wengert::VarId> = generator
                .adjoint_vars_map()
                .iter()
                .filter(|(p, _)| param_vids.contains(p))
                .map(|(_, a)| *a)
                .collect();
            tape_escaping.insert(effective_primal.output);
            // Stage-2B: sizes the hint bridge cannot reach. Shapes
            // first — dims propagate through matmul, which is where
            // the numel-only pass stopped (it sized 0 of 1321
            // transients on coder50m: every backward elementwise
            // chain sits downstream of a matmul, and a matmul's
            // output numel is a function of the SHAPES). The numel
            // pass still runs last, as a fallback for values whose
            // dims die at a runtime-shaped op but whose count
            // survives.
            let mut dim_seeds: std::collections::HashMap<
                crate::wengert::VarId,
                Vec<i64>,
            > = crate::profiling::captures::dim_hints_from_var_nodes(
                extractor.var_nodes(),
                self.type_map,
            );
            // Model-type-resolved per-var dims FIRST: a field named
            // `weight` exists in every Linear/Embedding module, so
            // the bare-leaf-name bridge below drops it as ambiguous
            // while this map has each var's correct dims.
            for (vid, d) in extractor.known_param_dims() {
                dim_seeds.entry(*vid).or_insert_with(|| d.clone());
            }
            let field_dims = self.models.unique_field_dims();
            for (name, vid) in extractor.named_param_var_ids() {
                if dim_seeds.contains_key(vid) {
                    continue;
                }
                let leaf = name.rsplit('.').next().unwrap_or(name);
                if let Some(d) = field_dims.get(leaf) {
                    dim_seeds.insert(*vid, d.clone());
                }
            }
            // Item 4's DataLoader proof seeds the batch fields: a
            // Proven scan certifies every batch is exactly
            // [batch_size, seq_len] (unanimous loaders, drop_last
            // required, short batches padded by the runtime), and
            // those fields are the entry point the whole forward
            // chain hangs off. Only fields the runtime emits at
            // [B, S], and only reads of a step-input dict (an Input
            // leaf) — a user-built dict proves nothing.
            let arena_debug =
                std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1");
            if let Some(facts) = self.lm_head_loader_scan.facts() {
                let input_leaves: std::collections::HashSet<
                    crate::wengert::VarId,
                > = effective_primal
                    .ops
                    .iter()
                    .filter(|o| {
                        matches!(o.op, crate::wengert::PrimalOp::Input(_))
                    })
                    .map(|o| o.result)
                    .collect();
                for op in &effective_primal.ops {
                    let crate::wengert::PrimalOp::Passthrough(n) = &op.op else {
                        continue;
                    };
                    let Some(field) = n.strip_prefix("dict_get:") else {
                        continue;
                    };
                    let eligible = matches!(
                        field,
                        "input_ids" | "labels" | "segment_ids" | "position_ids"
                    ) && op.inputs.len() == 1
                        && input_leaves.contains(&op.inputs[0]);
                    if arena_debug {
                        nsl_log::nsl_log!(INFO, "arena-debug", 
                            "[arena-debug] dict_get:{field} v{} inputs={:?} \
                             leaf={} -> seed {}",
                            op.result,
                            op.inputs,
                            op.inputs
                                .first()
                                .is_some_and(|i| input_leaves.contains(i)),
                            eligible,
                        );
                    }
                    if eligible {
                        dim_seeds.entry(op.result).or_insert_with(|| {
                            vec![facts.batch_size as i64, facts.seq_len as i64]
                        });
                    }
                }
            } else if arena_debug {
                nsl_log::nsl_log!(INFO, "arena-debug", 
                    "[arena-debug] loader scan unproven: {:?}",
                    self.lm_head_loader_scan.reason()
                );
            }
            // Same unique-preimage rule as the elems mirror above:
            // an adjoint has its primal's shape by construction, but
            // only a dedicated accumulator certifies WHICH primal.
            let mirror_pairs: Vec<(
                crate::wengert::VarId,
                crate::wengert::VarId,
            )> = {
                let mut preimage: std::collections::HashMap<
                    crate::wengert::VarId,
                    u32,
                > = Default::default();
                for a in generator.adjoint_vars_map().values() {
                    *preimage.entry(*a).or_default() += 1;
                }
                generator.adjoint_vars_map()
                    .iter()
                    .filter(|(_, a)| preimage.get(a) == Some(&1))
                    .map(|(p, a)| (*p, *a))
                    .collect()
            };
            let n_dim_seeds = dim_seeds.len();
            let scalar_seeds = extractor.known_param_scalar_values();
            let size_info = crate::transient_arena::propagate_size_info(
                effective_primal,
                adjoint,
                &|v| dim_seeds.get(&v).cloned(),
                &|v| scalar_seeds.get(&v).copied(),
                &mirror_pairs,
            );
            let shape_elems: std::collections::HashMap<
                crate::wengert::VarId,
                u64,
            > = size_info
                .iter()
                .filter_map(|(v, si)| si.numel().map(|n| (*v, n)))
                .collect();
            let propagated = crate::transient_arena::propagate_elems(
                effective_primal,
                adjoint,
                &|v| {
                    shape_elems
                        .get(&v)
                        .copied()
                        .or_else(|| elem_hints.get(&v).copied())
                },
            );
            // Provenance split. "Sized nothing" reads completely
            // differently when the seeds are empty vs when the
            // propagation stopped early — and the fixes differ too.
            nsl_log::nsl_log!(INFO, "arena", 
                "[arena] element counts: {} dim seed(s) + {} numel \
                 hint(s) -> {} shape-propagated -> {} sized, of {} \
                 tape value(s)",
                n_dim_seeds,
                elem_hints.len(),
                shape_elems.len(),
                propagated.len(),
                effective_primal.ops.len() + adjoint.ops.len(),
            );
            // Milestone C: SCHEDULED. The pass scans BOTH lists
            // (birth/death liveness over the concatenated
            // [forward; adjoint] positions) but the scheduler retains
            // ONE digest per (epoch, pass) — the adjoint is the one
            // digested, because every admitted placement lives there
            // (`admit` refuses NotBackward) and it is the list the
            // arena-consuming lowering walks. A forward-tape splice
            // between here and that lowering would evade this digest;
            // none exists today (fuse_swiglu_gate_backward runs
            // BEFORE this analyze, CCR's splices before that), and
            // the honest fix for a second scanned list is a
            // scheduler API that digests both — deliberately not
            // built for a window with no known mutator.
            let sched = self.passes.scheduler();
            let arena = sched
                .schedule("MemoryPlanner", Some(adjoint), || {
                    crate::transient_arena::analyze(
                        effective_primal,
                        adjoint,
                        &|v| propagated.get(&v).copied(),
                        &tape_escaping,
                        4, // GPU f32 training dtype width
                    )
                })
                .map_err(CodegenError::new)?
                .finish(&self.bus)
                .map_err(CodegenError::new)?;
            nsl_log::nsl_log!(INFO, "arena", "[arena]\n{}", arena.render_report("  "));

            // ── Stage-2B: placement ──────────────────────────
            //
            // Admission is deliberately narrow (see `admit`), so the
            // interesting number is usually how much was REFUSED and
            // to which rule. Printing that is the difference between
            // "the arena placed nothing because the model has no
            // eligible temporaries" and "the arena placed nothing
            // because a rule is broken", which otherwise look
            // identical from outside.
            if arena_place_on {
                let (ok, refused) = crate::transient_arena::admit(
                    &arena,
                    effective_primal,
                    adjoint,
                    &tape_escaping,
                    &size_info,
                );
                let (placements, payload) =
                    crate::transient_arena::pack(&arena, &ok);
                let mut by_reason: std::collections::BTreeMap<&str, usize> =
                    Default::default();
                for (_, r) in &refused {
                    *by_reason.entry(match r {
                        crate::transient_arena::RefusedBecause::NotBackward =>
                            "forward region",
                        crate::transient_arena::RefusedBecause::Unsized =>
                            "no static size",
                        crate::transient_arena::RefusedBecause::SavedForBackward =>
                            "saved for backward",
                        crate::transient_arena::RefusedBecause::EscapesTape =>
                            "escapes the tape",
                        crate::transient_arena::RefusedBecause::Aliasing =>
                            "may alias an input",
                        crate::transient_arena::RefusedBecause::NotSingleAllocation =>
                            "not a proven single allocation",
                        crate::transient_arena::RefusedBecause::InPlaceReuse =>
                            "in-place reuse (input dies here)",
                        crate::transient_arena::RefusedBecause::RuntimePathVaries =>
                            "runtime path varies (broadcast/view operand)",
                    }).or_default() += 1;
                }
                nsl_log::nsl_log!(INFO, "arena", 
                    "[arena] placement: {} of {} transient(s) admitted, \
                     {:.2} MiB payload in {} slot(s)",
                    placements.len(),
                    arena.transients.len(),
                    payload as f64 / 1048576.0,
                    placements.len(),
                );
                for (reason, n) in &by_reason {
                    nsl_log::nsl_log!(WARN, "arena", "[arena]   refused {n:>5} — {reason}");
                }
                // Which op kinds cost the coverage. Unsized = a
                // propagation rule is missing or a seed never
                // reached it; NotSingleAllocation = sized but the
                // allowlist excludes its producer. The two have
                // completely different fixes, per-op-kind counts
                // are what tells them apart.
                if std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1") {
                    let mut by_kind: std::collections::BTreeMap<String, usize> =
                        Default::default();
                    for (v, r) in &refused {
                        use crate::transient_arena::RefusedBecause as R;
                        if !matches!(r, R::Unsized | R::NotSingleAllocation) {
                            continue;
                        }
                        let producer = adjoint
                            .ops
                            .iter()
                            .chain(effective_primal.ops.iter())
                            .find(|o| o.result == *v);
                        let kind = match producer.map(|o| &o.op) {
                            Some(crate::wengert::PrimalOp::Passthrough(n)) => {
                                format!(
                                    "Passthrough:{}",
                                    n.split(':').next().unwrap_or(n)
                                )
                            }
                            Some(other) => {
                                let d = format!("{other:?}");
                                d.split([' ', '{', '('])
                                    .next()
                                    .unwrap_or("?")
                                    .to_string()
                            }
                            None => "<no producing op>".to_string(),
                        };
                        *by_kind
                            .entry(format!("{kind} [{r:?}]"))
                            .or_default() += 1;
                    }
                    let mut rows: Vec<_> = by_kind.into_iter().collect();
                    rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                    for (kind, n) in rows {
                        nsl_log::nsl_log!(INFO, "arena-debug", "[arena-debug]   {n:>5} x {kind}");
                    }
                    // The forward stall is invisible above (admit
                    // refuses forward transients NotBackward before
                    // Unsized ever fires), but an unsized forward
                    // value starves everything downstream of it in
                    // the backward too.
                    let mut fwd_unsized: std::collections::BTreeMap<String, usize> =
                        Default::default();
                    for t in &arena.transients {
                        if t.region != crate::transient_arena::Region::Forward
                            || t.elems.is_some()
                        {
                            continue;
                        }
                        if let Some(o) =
                            effective_primal.ops.iter().find(|o| o.result == t.var)
                        {
                            let kind = match &o.op {
                                crate::wengert::PrimalOp::Passthrough(n) => {
                                    format!(
                                        "Passthrough:{}",
                                        n.split(':').next().unwrap_or(n)
                                    )
                                }
                                other => {
                                    let d = format!("{other:?}");
                                    d.split([' ', '{', '('])
                                        .next()
                                        .unwrap_or("?")
                                        .to_string()
                                }
                            };
                            *fwd_unsized.entry(kind).or_default() += 1;
                        }
                    }
                    let mut rows: Vec<_> = fwd_unsized.into_iter().collect();
                    rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                    for (kind, n) in rows {
                        nsl_log::nsl_log!(INFO, "arena-debug", "[arena-debug]   fwd unsized {n:>5} x {kind}");
                    }
                    // The FIRST stalls in tape order — everything
                    // after the first is usually just downstream
                    // starvation.
                    let mut shown = 0;
                    for (i, o) in effective_primal.ops.iter().enumerate() {
                        if shown >= 15 {
                            break;
                        }
                        if shape_elems.contains_key(&o.result)
                            || matches!(
                                o.op,
                                crate::wengert::PrimalOp::Input(_)
                                    | crate::wengert::PrimalOp::Param(_)
                                    | crate::wengert::PrimalOp::Constant(_)
                                    | crate::wengert::PrimalOp::FreeTensor
                            )
                        {
                            continue;
                        }
                        // Const-lattice ops (non-tensor results)
                        // are not stalls; their state is invisible
                        // to shape_elems by design.
                        if let crate::wengert::PrimalOp::Passthrough(n) = &o.op
                            && matches!(
                                n.as_str(),
                                "shape" | "subscript" | "int" | "float" | "list"
                                    | "ndim" | "item"
                            )
                        {
                            continue;
                        }
                        let ins: Vec<String> = o
                            .inputs
                            .iter()
                            .map(|v| match size_info.get(v) {
                                Some(si) => format!("v{v}:{si:?}"),
                                None => format!("v{v}:?"),
                            })
                            .collect();
                        let kind = match &o.op {
                            crate::wengert::PrimalOp::Passthrough(n) => {
                                format!("Passthrough:{n}")
                            }
                            other => format!("{other:?}"),
                        };
                        nsl_log::nsl_log!(INFO, "arena-debug", 
                            "[arena-debug]   stall #{i} v{} {} <- [{}]",
                            o.result,
                            kind.chars().take(60).collect::<String>(),
                            ins.join(", ")
                        );
                        shown += 1;
                    }
                    // Where dims get LOST (result Numel): the
                    // dims-loss point poisons everything downstream
                    // into numel-land even when counts survive.
                    let mut shown = 0;
                    for (i, o) in effective_primal.ops.iter().enumerate() {
                        if shown >= 12 {
                            break;
                        }
                        if !matches!(
                            size_info.get(&o.result),
                            Some(crate::transient_arena::SizeInfo::Numel(_))
                        ) {
                            continue;
                        }
                        let ins: Vec<String> = o
                            .inputs
                            .iter()
                            .map(|v| match size_info.get(v) {
                                Some(si) => format!("v{v}:{si:?}"),
                                None => format!("v{v}:?"),
                            })
                            .collect();
                        let kind = match &o.op {
                            crate::wengert::PrimalOp::Passthrough(n) => {
                                format!("Passthrough:{n}")
                            }
                            other => format!("{other:?}"),
                        };
                        nsl_log::nsl_log!(INFO, "arena-debug", 
                            "[arena-debug]   numel #{i} v{} {} <- [{}]",
                            o.result,
                            kind.chars().take(60).collect::<String>(),
                            ins.join(", ")
                        );
                        shown += 1;
                    }
                }
                if std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1") {
                    for p in &placements {
                        let kind = adjoint
                            .ops
                            .iter()
                            .find(|o| o.result == p.var)
                            .map(|o| match &o.op {
                                crate::wengert::PrimalOp::Passthrough(n) => {
                                    format!("Passthrough:{n}")
                                }
                                other => format!("{other:?}")
                                    .split([' ', '{', '('])
                                    .next()
                                    .unwrap_or("?")
                                    .to_string(),
                            })
                            .unwrap_or_else(|| "<none>".into());
                        let inputs_desc = adjoint
                            .ops
                            .iter()
                            .find(|o| o.result == p.var)
                            .map(|o| {
                                o.inputs
                                    .iter()
                                    .map(|v| {
                                        let pk = adjoint
                                            .ops
                                            .iter()
                                            .chain(effective_primal.ops.iter())
                                            .find(|q| q.result == *v)
                                            .map(|q| match &q.op {
                                                crate::wengert::PrimalOp::Passthrough(n) => n.clone(),
                                                other => format!("{other:?}")
                                                    .split([' ', '{', '('])
                                                    .next()
                                                    .unwrap_or("?")
                                                    .to_string(),
                                            })
                                            .unwrap_or_else(|| "<leaf?>".into());
                                        format!("v{v}<{pk}>{:?}", size_info.get(v))
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or_default();
                        nsl_log::nsl_log!(INFO, "arena-debug", 
                            "[arena-debug] slot {} v{} {} B {} <- {}",
                            p.slot_index, p.var, p.bytes, kind, inputs_desc
                        );
                    }
                }
                self.arena_placements =
                    placements.iter().map(|p| (p.var, *p)).collect();
                if !placements.is_empty() {
                    let total = builder.ins().iconst(cl_types::I64, payload as i64);
                    let nslots =
                        builder.ins().iconst(cl_types::I64, placements.len() as i64);
                    self.compile_call_by_name(
                        builder, "nsl_arena_init", &[total, nslots])?;
                    // Slot geometry, in dense order, so the runtime
                    // can verify the INTERIOR red zones — without it
                    // only the arena's outermost guards are
                    // checkable and a slot-k overrun into slot k+1
                    // goes unseen.
                    for p in &placements {
                        let off =
                            builder.ins().iconst(cl_types::I64, p.offset as i64);
                        let bytes =
                            builder.ins().iconst(cl_types::I64, p.bytes as i64);
                        self.compile_call_by_name(
                            builder,
                            "nsl_arena_declare_slot",
                            &[off, bytes],
                        )?;
                    }
                }
            }
        }

        Ok(elem_hints)
    }
}
