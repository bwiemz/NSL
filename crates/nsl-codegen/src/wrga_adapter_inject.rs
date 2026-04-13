//! WRGA Milestone B.2 Task 2b: codegen pass that materialises `@adapter`
//! decorators as synthesized model fields + strict initialisers + forward
//! rewrite.
//!
//! Runs once per compile, after `wrga::run` produces the `WrgaPlan` (and the
//! plan has been stashed on `compiler.last_wrga_plan`), and before the model
//! struct / forward pass are lowered.  This is the B.2 observation surface:
//! it only decides synthesized field names and init strategies.  B.3 will
//! collapse the unfused update `y = x @ W + ((x @ A) @ B) * (alpha / rank)`
//! into a single MMA epilogue.

use crate::wrga::WrgaPlan;
use crate::AdapterKind;

/// Kind of initialiser used for a synthesized adapter field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitKind {
    /// `nsl_tensor_zeros` — strict zero, NEVER default init. Required for
    /// LoRA-B so that the adapted model equals the base model at step 0.
    Zeros,
    /// `nsl_tensor_ones` — IA³'s identity-preserving initialiser.
    Ones,
    /// Kaiming-uniform init — used for LoRA-A (breaks symmetry) and the
    /// random half of GatedLoRA.
    KaimingUniform,
}

/// Per-field initialisation directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitStrategy {
    pub field_name: String,
    pub kind: InitKind,
}

/// Per-site adapter materialisation decision produced by this pass.
#[derive(Debug, Clone)]
pub struct AdapterSite {
    /// Stable site identifier, e.g. `m_w__lora`. Derived from the target
    /// parameter name with dots replaced by underscores plus an adapter-kind
    /// suffix.
    pub site_id: String,
    pub kind: AdapterKind,
    pub target_param: String,
    pub rank: i64,
    pub alpha: i64,
    pub synthesized_fields: Vec<String>,
}

/// Result of the inject pass.
pub struct AdapterInjectResult {
    pub sites: Vec<AdapterSite>,
}

/// Run the injection pass.
///
/// Walks every placement in `plan` that carries a `decorator_kind` (i.e. was
/// driven by an explicit `@adapter(...)` decorator), decides the synthesized
/// fields + init strategies, writes them back onto the placement so later
/// codegen stages (and tests) can observe them, and returns the aggregate
/// site list.
pub fn run(plan: &mut WrgaPlan) -> AdapterInjectResult {
    let mut sites = Vec::new();
    for placement in plan.placements.iter_mut() {
        let Some(kind) = placement.decorator_kind else {
            continue;
        };
        let rank = placement
            .suggested_rank
            .max(1)
            .min(i64::MAX as usize) as i64;
        let rank = rank.max(1);
        let alpha = placement.alpha.unwrap_or(rank).max(1);
        let site_id = site_id_for(&placement.name, kind);

        let (fields, inits) = match kind {
            AdapterKind::Lora => {
                let a = format!("lora_A_{site_id}");
                let b = format!("lora_B_{site_id}");
                (
                    vec![a.clone(), b.clone()],
                    vec![
                        InitStrategy {
                            field_name: a,
                            kind: InitKind::KaimingUniform,
                        },
                        InitStrategy {
                            field_name: b,
                            kind: InitKind::Zeros,
                        },
                    ],
                )
            }
            AdapterKind::Ia3 => {
                let s = format!("ia3_scale_{site_id}");
                (
                    vec![s.clone()],
                    vec![InitStrategy {
                        field_name: s,
                        kind: InitKind::Ones,
                    }],
                )
            }
            AdapterKind::GatedLora => {
                let a = format!("lora_A_{site_id}");
                let b = format!("lora_B_{site_id}");
                let g = format!("gate_{site_id}");
                (
                    vec![a.clone(), b.clone(), g.clone()],
                    vec![
                        InitStrategy {
                            field_name: a,
                            kind: InitKind::KaimingUniform,
                        },
                        InitStrategy {
                            field_name: b,
                            kind: InitKind::Zeros,
                        },
                        InitStrategy {
                            field_name: g,
                            kind: InitKind::Zeros,
                        },
                    ],
                )
            }
        };

        placement.synthesized_fields = fields.clone();
        placement.init_strategies = inits;

        sites.push(AdapterSite {
            site_id,
            kind,
            target_param: placement.name.clone(),
            rank,
            alpha,
            synthesized_fields: fields,
        });
    }
    AdapterInjectResult { sites }
}

/// Build a stable site identifier from the target parameter path + kind.
fn site_id_for(target: &str, kind: AdapterKind) -> String {
    let safe = target.replace('.', "_");
    let suffix = match kind {
        AdapterKind::Lora => "lora",
        AdapterKind::Ia3 => "ia3",
        AdapterKind::GatedLora => "gatedlora",
    };
    format!("{safe}__{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn site_id_replaces_dots_and_adds_kind_suffix() {
        assert_eq!(
            site_id_for("blocks.6.wq", AdapterKind::Lora),
            "blocks_6_wq__lora",
        );
        assert_eq!(site_id_for("m.w", AdapterKind::Ia3), "m_w__ia3");
        assert_eq!(
            site_id_for("m.w", AdapterKind::GatedLora),
            "m_w__gatedlora",
        );
    }
}
