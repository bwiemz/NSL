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
    /// Input dimension (k_in) of the target weight. Zero when resolution
    /// failed — callers MUST treat zero as "skip materialisation" and a
    /// diagnostic will already have been emitted.
    pub input_dim: u32,
    /// Output dimension (d_out) of the target weight. Zero when resolution
    /// failed (see `input_dim`).
    pub output_dim: u32,
    /// Resolved model-class name that owns the target field (e.g. `"Toy"`).
    /// Empty when the target could not be resolved to a known model.
    pub target_model: String,
    /// Target field name on the model struct (e.g. `"w"` for `"m.w"`).
    /// Empty on resolution failure.
    pub target_field: String,
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
            input_dim: 0,
            output_dim: 0,
            target_model: String::new(),
            target_field: String::new(),
        });
    }
    AdapterInjectResult { sites }
}

/// Parse a tensor type string like `"Tensor<[32, 16], f32>"` into its
/// `(output_dim, input_dim)` shape pair.  Returns None if the string is
/// not a 2-D tensor type annotation.
///
/// Convention: NSL weights are declared as `Tensor<[out, in], dtype>` so
/// the first dim is the output channel count and the second is the input.
pub(crate) fn parse_tensor_2d_shape(type_str: &str) -> Option<(u32, u32)> {
    let open = type_str.find('[')?;
    let close = type_str.find(']')?;
    if close <= open {
        return None;
    }
    let inner = &type_str[open + 1..close];
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        return None;
    }
    let out: u32 = parts[0].parse().ok()?;
    let inp: u32 = parts[1].parse().ok()?;
    Some((out, inp))
}

/// Resolve `(input_dim, output_dim)` for an adapter target weight.
///
/// Returns `Some((input_dim, output_dim))` when resolution succeeds;
/// `None` when the model/field is absent or the type isn't a 2-D tensor.
/// On `None`, the caller must emit a diagnostic — never ship `(0, 0)`
/// silently onto an `AdapterSite`.
pub(crate) fn resolve_dims_for_target(
    model_name: &str,
    field_name: &str,
    field_types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> Option<(u32, u32)> {
    let field_map = field_types.get(model_name)?;
    let type_str = field_map.get(field_name)?;
    let (out, inp) = parse_tensor_2d_shape(type_str)?;
    Some((inp, out))
}

/// Wrapper over [`run`] that additionally resolves real `input_dim` /
/// `output_dim` values on each produced `AdapterSite` by consulting the
/// compiler's `model_field_types` registry.  On any resolution failure an
/// `eprintln!` diagnostic is emitted and the site's dims remain zero so
/// downstream stages skip materialisation.
pub fn run_with_compiler(
    plan: &mut WrgaPlan,
    compiler: &crate::compiler::Compiler,
) -> AdapterInjectResult {
    let mut result = run(plan);
    for site in result.sites.iter_mut() {
        let target = site.target_param.clone();
        let (var_name, field_name) = match target.split_once('.') {
            Some(p) => p,
            None => {
                eprintln!(
                    "[wrga] @adapter target '{}' is not a `model.field` form; skipping",
                    target
                );
                continue;
            }
        };

        // B.2.1 Task 5.5: search `model_tensor_field_shapes` (the
        // shape-bearing map populated by `collect_models` for Tensor
        // fields) before falling back to `model_field_types`.
        let tensor_shapes = &compiler.models.model_tensor_field_shapes;
        let field_types = &compiler.models.model_field_types;
        let mut candidates: Vec<&String> = tensor_shapes
            .iter()
            .filter(|(_, fields)| fields.contains_key(field_name))
            .map(|(name, _)| name)
            .collect();
        if candidates.is_empty() {
            candidates = field_types
                .iter()
                .filter(|(_, fields)| fields.contains_key(field_name))
                .map(|(name, _)| name)
                .collect();
        }
        if candidates.is_empty() {
            eprintln!(
                "[wrga] @adapter target '{}': field '{}' not found in any known model; \
                 skipping adapter materialisation (dims remain 0)",
                target, field_name
            );
            continue;
        }
        if candidates.len() > 1 {
            eprintln!(
                "[wrga] @adapter target '{}': field '{}' ambiguous across models {:?}; \
                 using first; follow-up: thread let-binding type map",
                target, field_name, candidates
            );
        }
        let model_name = candidates.remove(0).clone();
        // B.2.1 Task 3: submodel-target detection. If the target field's type
        // string does not start with `"Tensor<"`, it's a submodel reference
        // (e.g. a nested model), which is out of scope for B.2.1.
        // Look up the type string from the shape map first (for Tensor
        // fields populated by collect_models from the init expr), then
        // fall back to model_field_types (for sub-model references).
        let type_str = tensor_shapes
            .get(&model_name)
            .and_then(|m| m.get(field_name))
            .cloned()
            .or_else(|| {
                field_types
                    .get(&model_name)
                    .and_then(|m| m.get(field_name))
                    .cloned()
            })
            .unwrap_or_default();
        if !type_str.trim_start().starts_with("Tensor<") {
            eprintln!(
                "[wrga] @adapter target '{}': targets a submodel, not a weight tensor; \
                 adapt the submodel's inner weight directly (e.g., \"{}.weight\") \
                 or use a submodel-level decorator (not yet supported)",
                target, target,
            );
            continue;
        }
        match resolve_dims_for_target(&model_name, field_name, tensor_shapes)
            .or_else(|| resolve_dims_for_target(&model_name, field_name, field_types))
        {
            Some((inp, out)) => {
                site.input_dim = inp;
                site.output_dim = out;
                site.target_model = model_name.clone();
                site.target_field = field_name.to_string();
            }
            None => {
                eprintln!(
                    "[wrga] @adapter target '{}': type string for '{}.{}' \
                     isn't a 2-D tensor; skipping",
                    target, model_name, field_name
                );
            }
        }
        let _ = var_name; // reserved for later let-binding-type-map resolution
    }
    result
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
    fn dims_for_resolves_from_field_types_registry() {
        use std::collections::HashMap;
        let mut inner = HashMap::new();
        inner.insert("w".to_string(), "Tensor<[32, 16], f32>".to_string());
        let mut outer = HashMap::new();
        outer.insert("Toy".to_string(), inner);
        let (input_dim, output_dim) =
            resolve_dims_for_target("Toy", "w", &outer).expect("dims must resolve");
        assert_eq!(input_dim, 16, "input_dim = k_in (second dim of Tensor<[out, in]>)");
        assert_eq!(output_dim, 32, "output_dim = d_out (first dim)");
    }

    #[test]
    fn dims_for_returns_none_on_missing_model() {
        use std::collections::HashMap;
        let mut inner = HashMap::new();
        inner.insert("w".to_string(), "Tensor<[32, 16], f32>".to_string());
        let mut outer = HashMap::new();
        outer.insert("Toy".to_string(), inner);
        assert!(
            resolve_dims_for_target("Unknown", "w", &outer).is_none(),
            "unknown model must return None (caller emits diagnostic)",
        );
    }

    #[test]
    fn dims_for_returns_none_on_missing_field() {
        use std::collections::HashMap;
        let mut inner = HashMap::new();
        inner.insert("w".to_string(), "Tensor<[32, 16], f32>".to_string());
        let mut outer = HashMap::new();
        outer.insert("Toy".to_string(), inner);
        assert!(resolve_dims_for_target("Toy", "nonexistent", &outer).is_none());
    }

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
