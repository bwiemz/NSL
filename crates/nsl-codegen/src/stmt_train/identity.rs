//! The train block's checkpoint-identity emission: the resolved
//! train/optimizer/scheduler record (item 4) and the full-state resume
//! load (Milestone B / item 8).
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1).
//! The record renderer is a pure function of the resolved configuration
//! — which is why it moved first: the `k=v,k=v` string it produces is what
//! `nsl_runtime::train_config_record` diffs on resume, and its format was
//! pinned only end-to-end (`train_config_resume_gate.rs`). It now has unit
//! tests, including one that checks every key it renders against the
//! runtime's `MOMENT_KEYS` / `TRAJECTORY_KEYS`: a key in neither class is
//! silently unguarded on resume, which is the drift this file exists to
//! make impossible.
//!
//! The resume load emits one runtime call and two bindings the batch loop
//! and the checkpoint save reuse (`has_dataloader`, `checkpoint_dl_handle`);
//! the train-block CLIF snapshots pin it on the `checkpoint_load` fixtures.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::param_roles::NoDecayScope;
use nsl_semantic::optim_config::ResolvedScheduler;

/// The resolved values the record is rendered from — the bindings
/// `compile_train_block_inner` derives from `resolve_optim_config` and the
/// train header, under their names there.
#[derive(Clone, Copy)]
pub(crate) struct TrainConfigRecordInputs<'a> {
    pub(crate) optimizer_name: &'a str,
    pub(crate) lr_value: f64,
    pub(crate) grad_accumulation_steps: i64,
    /// `f64::MAX` = no clipping.
    pub(crate) grad_clip: f64,
    pub(crate) weight_decay_value: f64,
    pub(crate) beta1_value: f64,
    pub(crate) beta2_value: f64,
    pub(crate) eps_value: f64,
    pub(crate) momentum_value: f64,
    pub(crate) dampening_value: f64,
    pub(crate) nesterov_value: bool,
    pub(crate) ns_steps_value: f64,
    pub(crate) adamw_lr_value: Option<f64>,
    pub(crate) no_decay_scope: &'a NoDecayScope,
    pub(crate) scheduler: &'a Option<ResolvedScheduler>,
}

/// Render the train/optimizer/scheduler record (item 4): one fixed-order
/// `k=v,k=v` string of the RESOLVED config. Values are compile-time
/// constants the runtime cannot recover. Floats render via Display
/// (shortest round-trip, digits/dot only for validated positive config
/// values); the one user-text field (no_decay roles) is allowlisted to
/// [a-z0-9_-] — the #519 lesson: allowlist user text inside structured
/// containers.
pub(crate) fn render_train_config_record(inputs: &TrainConfigRecordInputs<'_>) -> String {
    let TrainConfigRecordInputs {
        optimizer_name,
        lr_value,
        grad_accumulation_steps,
        grad_clip,
        weight_decay_value,
        beta1_value,
        beta2_value,
        eps_value,
        momentum_value,
        dampening_value,
        nesterov_value,
        ns_steps_value,
        adamw_lr_value,
        no_decay_scope,
        scheduler,
    } = *inputs;
    let clip_s = if grad_clip == f64::MAX {
        "none".to_string()
    } else {
        grad_clip.to_string()
    };
    let adamw_lr_s = adamw_lr_value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "none".to_string());
    let mut nd: Vec<String> = no_decay_scope
        .static_roles
        .iter()
        .map(|r| {
            r.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect()
        })
        .collect();
    if no_decay_scope.exempt_non_rank2 {
        nd.push("vector".to_string());
    }
    nd.sort();
    let nd_s = if nd.is_empty() { "none".to_string() } else { nd.join("+") };
    let (sched_s, sched_params): (&str, Vec<f64>) = match &scheduler {
        None => ("none", Vec::new()),
        Some(sch) => {
            use nsl_semantic::optim_config::ResolvedScheduler as RS;
            let ps = match sch {
                RS::ConstantLr => Vec::new(),
                RS::StepLr { step_size, gamma } => vec![*step_size, *gamma],
                RS::ExponentialLr { gamma } => vec![*gamma],
                RS::LinearDecay { total_steps, end_factor } => {
                    vec![*total_steps, *end_factor]
                }
                RS::CosineAnneal { t_max, eta_min } => vec![*t_max, *eta_min],
                RS::WarmupCosine { warmup_steps, total_steps, min_lr } => {
                    vec![*warmup_steps, *total_steps, *min_lr]
                }
                RS::OneCycle { max_lr, total_steps, pct_start } => {
                    vec![*max_lr, *total_steps, *pct_start]
                }
            };
            (sch.fn_name(), ps)
        }
    };
    let mut rec = format!(
        "opt={optimizer_name},lr={lr_value},accum={grad_accumulation_steps},\
clip={clip_s},wd={weight_decay_value},beta1={beta1_value},beta2={beta2_value},\
eps={eps_value},momentum={momentum_value},dampening={dampening_value},\
nesterov={},ns_steps={ns_steps_value},adamw_lr={adamw_lr_s},no_decay={nd_s},\
sched={sched_s}",
        nesterov_value as u8,
    );
    for (i, v) in sched_params.iter().enumerate() {
        rec.push_str(&format!(",sp{}={v}", i + 1));
    }
    rec
}

impl Compiler<'_> {
    /// Render + install the resolved record for checkpoint identity. At
    /// train-block ENTRY, per block (a module can hold several), before the
    /// resume load reads it as the LIVE side and before any save writes it
    /// into the sidecar.
    pub(crate) fn emit_train_config_record(
        &mut self,
        builder: &mut FunctionBuilder,
        inputs: &TrainConfigRecordInputs<'_>,
    ) -> Result<(), CodegenError> {
        let rec = render_train_config_record(inputs);
        self.intern_string(&rec)?;
        let rec_ptr = self.compile_string_literal(builder, &rec)?;
        let rec_len = builder.ins().iconst(cl_types::I64, rec.len() as i64);
        self.compile_call_by_name(
            builder,
            "nsl_set_train_config_record",
            &[rec_ptr, rec_len],
        )?;
        Ok(())
    }

    /// Milestone B: the full-state resume load. Returns the DataLoader
    /// binding the batch loop iterates (`None` = this train block has no
    /// loader) and the handle the checkpoint records — bound ONCE so the
    /// two cannot drift.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_checkpoint_resume(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &FuncState,
        checkpoint_load_path: &Option<String>,
        epochs: i64,
        param_list: Value,
        state_list_1: Value,
        state_list_2: Value,
        step_count_var: Variable,
    ) -> Result<(Option<Value>, Value), CodegenError> {
        // Milestone B: full-state resume. Emitted AFTER moment allocation and
        // BEFORE the first register belt: θ and m/v are all still plain f32
        // device tensors here, so the loader's H2D path covers every arm —
        // including bf16-sr and --weight-stream, whose registration happens
        // at step-body top and will quantize/evict the LOADED values. The
        // restored micro-batch counter seeds step_count_var so bias
        // correction, the scheduler, and checkpoint cadence all continue
        // instead of re-warming.
        //
        // Item 8: the DataLoader handle travels with the call so the runtime
        // can restore the data position too (and refuse a resume whose corpus
        // or geometry drifted). 0 = this train block has no loader, which the
        // sidecar records and cross-checks — a loader-less checkpoint resumed
        // into a loader run (or the reverse) is a silently different training
        // stream, not a continuation.
        // Bound ONCE here and reused by the batch loop below, so the handle
        // the checkpoint records and the handle the loop iterates are the
        // same value by construction — two independent `.last()` reads could
        // drift and silently record a position from a different loader.
        let has_dataloader = state.cleanup.dataloader_vars.last().copied();
        let checkpoint_dl_handle = has_dataloader
            .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));
        if let Some(load_path) = checkpoint_load_path.clone() {
            self.intern_string(&load_path)?;
            let path_val = self.compile_string_literal(builder, &load_path)?;
            let path_len = builder.ins().iconst(cl_types::I64, load_path.len() as i64);
            let epochs_val_for_resume = builder.ins().iconst(cl_types::I64, epochs);
            let restored = self.compile_call_by_name(
                builder,
                "nsl_train_checkpoint_load",
                &[
                    path_val,
                    path_len,
                    param_list,
                    state_list_1,
                    state_list_2,
                    checkpoint_dl_handle,
                    epochs_val_for_resume,
                ],
            )?;
            builder.def_var(step_count_var, restored);
        }
        Ok((has_dataloader, checkpoint_dl_handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(no_decay: &'a NoDecayScope, sched: &'a Option<ResolvedScheduler>) -> TrainConfigRecordInputs<'a> {
        TrainConfigRecordInputs {
            optimizer_name: "adamw",
            lr_value: 0.0001,
            grad_accumulation_steps: 4,
            grad_clip: 1.0,
            weight_decay_value: 0.1,
            beta1_value: 0.9,
            beta2_value: 0.95,
            eps_value: 0.00000001,
            momentum_value: 0.0,
            dampening_value: 0.0,
            nesterov_value: false,
            ns_steps_value: 5.0,
            adamw_lr_value: None,
            no_decay_scope: no_decay,
            scheduler: sched,
        }
    }

    /// The exact string is the contract: the runtime parses it by key and
    /// the sidecar carries it verbatim, so a renamed or reordered key is a
    /// resume refusal against every existing checkpoint.
    #[test]
    fn the_record_format_is_pinned() {
        let nd = NoDecayScope { static_roles: vec![], exempt_non_rank2: false };
        let sched = Some(ResolvedScheduler::WarmupCosine {
            warmup_steps: 200.0,
            total_steps: 2048.0,
            min_lr: 0.00003,
        });
        assert_eq!(
            render_train_config_record(&inputs(&nd, &sched)),
            "opt=adamw,lr=0.0001,accum=4,clip=1,wd=0.1,beta1=0.9,beta2=0.95,\
             eps=0.00000001,momentum=0,dampening=0,nesterov=0,ns_steps=5,\
             adamw_lr=none,no_decay=none,sched=warmup_cosine,sp1=200,sp2=2048,sp3=0.00003"
        );
        // No clip and no scheduler render as their sentinels, never as an
        // absent key (absent-on-both is "no difference" downstream).
        let mut i = inputs(&nd, &None);
        i.grad_clip = f64::MAX;
        let rec = render_train_config_record(&i);
        assert!(rec.contains(",clip=none,") && rec.ends_with(",sched=none"), "{rec}");
    }

    /// no_decay roles are user text inside a structured container: they
    /// are allowlisted, lowercased, sorted, and `vector` rides with them.
    #[test]
    fn no_decay_roles_are_allowlisted_sorted_and_joined() {
        let nd = NoDecayScope {
            static_roles: vec!["Head".to_string(), "emb,x=y".to_string()],
            exempt_non_rank2: true,
        };
        let rec = render_train_config_record(&inputs(&nd, &None));
        let field = rec.split(',').find(|f| f.starts_with("no_decay=")).unwrap();
        assert_eq!(field, "no_decay=emb_x_y+head+vector");
        assert!(rec.split(',').all(|f| f.matches('=').count() == 1), "{rec}");
    }

    /// Every key the renderer can emit is in one of the runtime's two
    /// resume classes. A key in neither is silently unguarded — the drift
    /// `train_config_record.rs` documents and this pins from the emitting
    /// side. Exercised with the widest scheduler (three parameters).
    #[test]
    fn every_rendered_key_is_classified_by_the_runtime() {
        use nsl_runtime::train_config_record::{MOMENT_KEYS, TRAJECTORY_KEYS};
        let nd = NoDecayScope { static_roles: vec![], exempt_non_rank2: false };
        let sched = Some(ResolvedScheduler::OneCycle {
            max_lr: 0.001,
            total_steps: 100.0,
            pct_start: 0.3,
        });
        let rec = render_train_config_record(&inputs(&nd, &sched));
        for kv in rec.split(',') {
            let key = kv.split('=').next().unwrap();
            assert!(
                MOMENT_KEYS.contains(&key) || TRAJECTORY_KEYS.contains(&key),
                "key `{key}` is rendered but in neither resume class: {rec}"
            );
        }
        assert_eq!(rec.split(',').count(), 18, "{rec}");
    }
}
