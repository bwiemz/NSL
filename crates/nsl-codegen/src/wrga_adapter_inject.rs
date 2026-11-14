//! WRGA: decide, for each `@adapter(...)` target, which model weight it
//! adapts, how that weight is used, and which tensors to synthesize for it.
//!
//! A target is written `Class.field`. [`resolve_sites`] finds the class,
//! checks that the field is a weight tensor, and looks through the class's
//! methods for the matmul the adapter attaches to. Anything it cannot
//! resolve is a compile error: an adapter that silently attaches nowhere (or
//! to another class) trains nothing, or the wrong thing.
//!
//! The update itself is `y = x @ W + ((x @ A) @ B) * (alpha / rank)` for
//! LoRA, written by `wrga_adapter_rewrite`.

use crate::error::CodegenError;
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
    /// A standard-normal draw scaled by `1 / sqrt(fan_in)` — used for LoRA-A
    /// (breaks symmetry) and the random half of GatedLoRA.
    ScaledNormal,
}

/// Per-field initialisation directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitStrategy {
    pub field_name: String,
    pub kind: InitKind,
}

/// How a model's methods use an adapted weight. It decides which axis of the
/// weight is the input width and which the output width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightUse {
    /// `x @ self.w`: the weight is `[in, out]`.
    Direct,
    /// `x @ self.w.transpose(0, 1)`, the form of the standard library's
    /// `Linear`: the weight is `[out, in]`.
    Transposed,
}

impl WeightUse {
    /// The weight's axis that holds the input width.
    pub fn in_axis(self) -> i64 {
        match self {
            WeightUse::Direct => 0,
            WeightUse::Transposed => 1,
        }
    }

    /// The weight's axis that holds the output width.
    pub fn out_axis(self) -> i64 {
        1 - self.in_axis()
    }
}

/// Per-site adapter materialisation decision produced by this pass.
#[derive(Debug, Clone)]
pub struct AdapterSite {
    /// Stable site identifier, e.g. `Toy_w__lora`. Derived from the target
    /// with dots replaced by underscores plus an adapter-kind suffix.
    pub site_id: String,
    pub kind: AdapterKind,
    pub target_param: String,
    pub rank: i64,
    pub alpha: i64,
    pub synthesized_fields: Vec<String>,
    /// Input width of the adapted matmul when the weight's shape is a
    /// literal in the model definition, 0 otherwise. The adapter tensors are
    /// sized from the weight itself when the model is constructed
    /// (`wrga_adapter_init`), so 0 does not mean "no adapter"; only the
    /// fused kernel, which is synthesized per shape, needs the static value.
    pub input_dim: u32,
    /// Output width, under the same rule as `input_dim`.
    pub output_dim: u32,
    /// The model class that owns the target field (e.g. `"Toy"`).
    pub target_model: String,
    /// Target field name on the model struct (e.g. `"w"`).
    pub target_field: String,
    /// Which matmul form the model's methods use the weight in.
    pub weight_use: WeightUse,
    /// Fusion decision for this site, populated after the fusion pass runs.
    /// The rewrite emits the single-call fused form only for
    /// `EpilogueFusedLora` / `EpilogueFusedGatedLora` / `ActivationFusedIa3`
    /// and only when the fused path is opted in and a kernel was synthesized
    /// for the site's shape; everything else is the ordinary operations.
    pub fusion_decision: Option<crate::wrga_fusion::FusionTarget>,
}

impl AdapterSite {
    /// The number of parameters the adapter adds to ONE instance of its
    /// model, when the adapted weight's shape is known at compile time:
    /// `rank · (in + out)` for LoRA, the same plus an `[out]` gate for gated
    /// LoRA, and `out` for IA³.
    pub fn param_count(&self) -> Option<usize> {
        if self.input_dim == 0 || self.output_dim == 0 {
            return None;
        }
        let (n_in, n_out) = (self.input_dim as usize, self.output_dim as usize);
        let rank = self.rank.max(1) as usize;
        Some(match self.kind {
            AdapterKind::Lora => rank * (n_in + n_out),
            AdapterKind::GatedLora => rank * (n_in + n_out) + n_out,
            AdapterKind::Ia3 => n_out,
        })
    }
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

        let (fields, inits) = synthesized_for(kind, &site_id);

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
            weight_use: WeightUse::Direct,
            fusion_decision: None,
        });
    }
    AdapterInjectResult { sites }
}

/// The tensors an adapter of `kind` adds for the site `site_id`, with the
/// initialiser of each, in side-table slot order.
pub(crate) fn synthesized_for(kind: AdapterKind, site_id: &str) -> (Vec<String>, Vec<InitStrategy>) {
    let named = |prefix: &str, kind: InitKind| InitStrategy {
        field_name: format!("{prefix}_{site_id}"),
        kind,
    };
    let inits = match kind {
        AdapterKind::Lora => vec![
            named("lora_A", InitKind::ScaledNormal),
            named("lora_B", InitKind::Zeros),
        ],
        AdapterKind::Ia3 => vec![named("ia3_scale", InitKind::Ones)],
        AdapterKind::GatedLora => vec![
            named("lora_A", InitKind::ScaledNormal),
            named("lora_B", InitKind::Zeros),
            named("gate", InitKind::Zeros),
        ],
    };
    (inits.iter().map(|i| i.field_name.clone()).collect(), inits)
}

/// The model classes this compile knows that declare a field `field`, sorted.
fn models_with_field(compiler: &crate::compiler::Compiler<'_>, field: &str) -> Vec<String> {
    let mut out: Vec<String> = compiler
        .models
        .model_method_bodies
        .keys()
        .filter(|m| {
            compiler
                .types
                .struct_layouts
                .get(*m)
                .is_some_and(|l| l.fields.iter().any(|f| f.name == field))
        })
        .cloned()
        .collect();
    out.sort();
    out
}

/// Which targets a compile is responsible for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetScope {
    /// The module that carries the `@adapter` decorators: every target must
    /// resolve, to a model defined here or imported.
    AllTargets,
    /// A module the decorated module imports. It materialises the adapters
    /// of the models it defines, and leaves every other target to the module
    /// that defines that model; the decorated module reports a target that
    /// resolves nowhere.
    OwnModelsOnly,
}

/// Resolve every `@adapter` target of `plan` against the compiler's models.
///
/// A target is `Class.field`. The class is taken as written: with models
/// `Aaa` and `Zzz` both owning a field `w`, `Zzz.w` used to attach the
/// adapter to `Aaa` (the first class, alphabetically, with such a field) and
/// say so only at INFO level. A target that names no model, no field, a
/// field that is not a 2-D weight, or a weight no matmul of the model uses,
/// is an error: each of those used to leave the adapter tensors allocated
/// and handed to the optimizer while the forward pass never read them.
pub fn resolve_sites(
    plan: &mut WrgaPlan,
    compiler: &crate::compiler::Compiler<'_>,
    scope: TargetScope,
) -> Result<AdapterInjectResult, CodegenError> {
    use crate::context::FieldKind;

    let mut result = run(plan);
    let mut resolved: Vec<AdapterSite> = Vec::with_capacity(result.sites.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for mut site in result.sites.drain(..) {
        let target = site.target_param.clone();
        let fail = |why: String| -> CodegenError {
            CodegenError::new(format!("@adapter target '{target}': {why}"))
        };
        let Some((class, field)) = target.split_once('.') else {
            return Err(fail(
                "a target is written `Class.field`, the model class and the weight field \
                 to adapt (for example \"Linear.w\")"
                    .to_string(),
            ));
        };
        if field.contains('.') {
            return Err(fail(format!(
                "a target is written `Class.field` and applies to every instance of the \
                 class; a path to one instance is not supported. Name the class that \
                 declares '{}'",
                field.rsplit('.').next().unwrap_or(field)
            )));
        }
        let defined_here = compiler.types.struct_layouts.contains_key(class)
            && !compiler.models.imported_model_names.contains(class);
        if scope == TargetScope::OwnModelsOnly && !defined_here {
            continue;
        }
        // One adapter per weight. A second one on the same `Class.field`
        // (in another @adapter, or listed twice in one) was accepted: both
        // got the same tensor names, and the forward rewrite took the first
        // site it matched, so the second was allocated, saved and handed to
        // the optimizer without ever entering the forward pass.
        if !seen.insert(target.clone()) {
            return Err(fail(
                "it is named more than once in the program's @adapter targets (in two \
                 @adapter decorators, or twice in one); a weight takes one adapter, so remove \
                 the repeat"
                    .to_string(),
            ));
        }
        let Some(layout) = compiler.types.struct_layouts.get(class) else {
            let owners = models_with_field(compiler, field);
            let hint = if owners.is_empty() {
                format!("and no model has a field '{field}'")
            } else {
                format!("models with a field '{field}': {}", owners.join(", "))
            };
            return Err(fail(format!(
                "there is no model named '{class}' (the part before the dot is the model \
                 class, not a variable); {hint}"
            )));
        };
        let Some(struct_field) = layout.fields.iter().find(|f| f.name == field) else {
            let tensors: Vec<&str> = layout
                .fields
                .iter()
                .filter(|f| f.kind == FieldKind::Tensor)
                .map(|f| f.name.as_str())
                .collect();
            return Err(fail(format!(
                "model '{class}' has no field '{field}'; its tensor fields are: {}",
                if tensors.is_empty() { "(none)".to_string() } else { tensors.join(", ") }
            )));
        };
        match struct_field.kind {
            FieldKind::Tensor => {}
            FieldKind::Nested => {
                let inner = compiler
                    .models
                    .model_field_types
                    .get(class)
                    .and_then(|m| m.get(field))
                    .cloned()
                    .unwrap_or_default();
                return Err(CodegenError::new(submodel_target_hint(&target, &inner)));
            }
            FieldKind::Buffer | FieldKind::Scalar => {
                return Err(fail(format!(
                    "'{class}.{field}' is not a weight tensor; an adapter attaches to a \
                     2-D `Tensor` field used as the right operand of a matmul"
                )));
            }
        }
        if let Some(rank) = compiler
            .models
            .model_field_ranks
            .get(class)
            .and_then(|m| m.get(field))
            && *rank != 2
        {
            return Err(fail(format!(
                "'{class}.{field}' is a {rank}-D tensor; an adapter attaches to a 2-D weight"
            )));
        }

        let uses = crate::wrga_adapter_rewrite::count_weight_uses(compiler, class, field);
        let weight_use = match (uses.direct, uses.transposed) {
            (0, 0) => {
                return Err(fail(format!(
                    "no method of '{class}' computes `x @ self.{field}` or \
                     `x @ self.{field}.transpose(0, 1)`, the two forms an adapter attaches \
                     to, so the adapter would be allocated and trained without ever \
                     entering the forward pass"
                )));
            }
            (_, 0) => WeightUse::Direct,
            (0, _) => WeightUse::Transposed,
            _ => {
                return Err(fail(format!(
                    "'{class}' uses `self.{field}` both as `x @ self.{field}` and as \
                     `x @ self.{field}.transpose(0, 1)`; one adapter cannot serve both, \
                     because its input and output widths would swap between them"
                )));
            }
        };

        // Static widths, when the weight's shape is a literal in a model this
        // module defines. `model_tensor_field_shapes` holds `Tensor<[d0, d1],
        // f32>` for exactly those fields.
        if let Some((d0, d1)) = compiler
            .models
            .model_tensor_field_shapes
            .get(class)
            .and_then(|m| m.get(field))
            .and_then(|t| parse_tensor_2d_shape(t))
        {
            (site.input_dim, site.output_dim) = match weight_use {
                WeightUse::Direct => (d0, d1),
                WeightUse::Transposed => (d1, d0),
            };
        }
        site.target_model = class.to_string();
        site.target_field = field.to_string();
        site.weight_use = weight_use;
        resolved.push(site);
    }
    result.sites = resolved;
    Ok(result)
}

/// Parse a tensor type string like `"Tensor<[32, 16], f32>"` into its two
/// dimensions, in declaration order. Returns None if the string is not a
/// 2-D tensor type annotation.
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
    let d0: u32 = parts[0].parse().ok()?;
    let d1: u32 = parts[1].parse().ok()?;
    Some((d0, d1))
}

/// Build the diagnostic for an `@adapter` target that names a submodel
/// field instead of a weight tensor.
///
/// Only 2-segment `<Class>.<field>` targets resolve, so the hint must steer
/// users to that form, naming the inner model's weight field; a 3-segment
/// path such as `"m.sub.weight"` can never work.
pub(crate) fn submodel_target_hint(target: &str, inner_class: &str) -> String {
    let class = if inner_class.trim().is_empty() {
        "<InnerClass>"
    } else {
        inner_class.trim()
    };
    format!(
        "@adapter target '{target}': targets a submodel, not a weight tensor; \
         the target resolver only supports 2-segment `<model>.<field>` paths, so \
         adapt the inner model's weight with the 2-segment form \"<InnerClass>.<field>\" \
         (e.g., \"{class}.<weight_field>\"); \
         submodel-level decorators are not yet supported"
    )
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
    fn a_2d_shape_string_parses_in_declaration_order() {
        assert_eq!(parse_tensor_2d_shape("Tensor<[32, 16], f32>"), Some((32, 16)));
        assert_eq!(parse_tensor_2d_shape("Tensor<[4, 6, 8], f32>"), None);
        assert_eq!(parse_tensor_2d_shape("Linear"), None);
    }

    #[test]
    fn the_input_axis_follows_the_matmul_form() {
        // `x @ w` contracts x's last axis with w's first; `x @ w.transpose(0,
        // 1)` with w's second.
        assert_eq!((WeightUse::Direct.in_axis(), WeightUse::Direct.out_axis()), (0, 1));
        assert_eq!((WeightUse::Transposed.in_axis(), WeightUse::Transposed.out_axis()), (1, 0));
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

    #[test]
    fn submodel_hint_suggests_working_two_segment_form() {
        let hint = submodel_target_hint("m.sub", "Inner");
        assert!(
            hint.contains("\"<InnerClass>.<field>\""),
            "hint must name the 2-segment <InnerClass>.<field> form: {hint}",
        );
        assert!(
            hint.contains("2-segment"),
            "hint must explain the resolver's 2-segment limit: {hint}",
        );
        assert!(
            hint.contains("\"Inner.<weight_field>\""),
            "hint must show a concrete example using the inner class name: {hint}",
        );
        assert!(
            !hint.contains("m.sub.weight"),
            "hint must NOT suggest the 3-segment '<target>.weight' form — the \
             resolver splits once on '.' so that form can never resolve: {hint}",
        );
        assert!(
            hint.contains("not yet supported"),
            "hint must still refuse loudly on submodel-level decorators: {hint}",
        );
    }

    #[test]
    fn submodel_hint_tolerates_empty_inner_class() {
        let hint = submodel_target_hint("m.sub", "  ");
        assert!(
            hint.contains("\"<InnerClass>.<weight_field>\""),
            "empty inner class must fall back to a placeholder: {hint}",
        );
    }

}
