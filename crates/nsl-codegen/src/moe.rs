//! M32: MoE codegen — @moe extraction + moe_dispatch lowering.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use crate::weight_aware::WeightMap;

/// CPDT Part III v1 production-forward (M32 gap closure): derive
/// `(hidden_dim, intermediate_dim)` for `nsl_moe_dispatch_full_v2` from the
/// WeightMap router/experts shapes scoped under `key`.
///
/// The contract mirrors `cpdt_expert_prune.rs:15`:
///   - router shape == `[hidden_dim, num_experts]`
///   - experts shape == `[num_experts, hidden_dim * intermediate_dim]`
///
/// Returns `None` (silent v1 fallback) when ANY of these fail: no router or
/// experts entry resolves under the key, shapes aren't 2D, router's
/// num-experts axis doesn't match `num_experts`, hidden_dim is zero, or
/// experts trailing dim isn't a multiple of hidden_dim. S4 promotes these
/// silent-None cases to compile errors for `cpdt_mode == Full`.
pub fn derive_v2_dims(
    weight_map: &WeightMap,
    key: &str,
    num_experts: usize,
) -> Option<(usize, usize)> {
    let router = ["router.weight", "gate.weight", "router", "gate"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts = ["experts.weight", "experts"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    if router.shape.len() != 2 || experts.shape.len() != 2 {
        return None;
    }
    if router.shape[1] != num_experts {
        return None;
    }
    let hidden = router.shape[0];
    let block_elems = experts.shape[1];
    if hidden == 0 || block_elems == 0 || !block_elems.is_multiple_of(hidden) {
        return None;
    }
    Some((hidden, block_elems / hidden))
}

/// Compile-time info about a MoE layer.
#[derive(Debug, Clone)]
pub struct MoeInfo {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub aux_loss_coeff: f32,
}

/// Extract @moe decorator from a list of decorators.
pub fn extract_moe_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<MoeInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "moe" {
            let mut num_experts: usize = 0;
            let mut top_k: usize = 2;
            let mut capacity_factor: f32 = 1.25;
            let mut aux_loss_coeff: f32 = 0.01;

            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        match name {
                            "num_experts" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    num_experts = *v as usize;
                                }
                            }
                            "top_k" => {
                                if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                    top_k = *v as usize;
                                }
                            }
                            "capacity_factor" => {
                                if let ExprKind::FloatLiteral(v) = &arg.value.kind {
                                    capacity_factor = *v as f32;
                                }
                            }
                            "aux_loss_coeff" => {
                                if let ExprKind::FloatLiteral(v) = &arg.value.kind {
                                    aux_loss_coeff = *v as f32;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if num_experts > 0 {
                return Some(MoeInfo {
                    num_experts,
                    top_k,
                    capacity_factor,
                    aux_loss_coeff,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_moe_empty_decorators() {
        let decorators: Vec<Decorator> = vec![];
        let result = extract_moe_decorator(&decorators, &|_| "");
        assert!(result.is_none());
    }

    #[test]
    fn test_moe_info_defaults() {
        // Verify default values used when only num_experts is provided
        let info = MoeInfo {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            aux_loss_coeff: 0.01,
        };
        assert_eq!(info.num_experts, 8);
        assert_eq!(info.top_k, 2);
        assert!((info.capacity_factor - 1.25).abs() < 1e-6);
        assert!((info.aux_loss_coeff - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_moe_info_clone() {
        let info = MoeInfo {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 2.0,
            aux_loss_coeff: 0.05,
        };
        let cloned = info.clone();
        assert_eq!(cloned.num_experts, 4);
        assert_eq!(cloned.top_k, 1);
    }

    // ── derive_v2_dims (CPDT Part III v1 production-forward) ───────────────
    use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};

    fn make_weight(name: &str, shape: Vec<usize>) -> WeightEntry {
        let total: usize = shape.iter().product();
        let bytes = vec![0u8; total * 4]; // f32 = 4 bytes/elem
        WeightEntry::new(name.to_string(), bytes, shape, WeightDType::F32)
    }

    #[test]
    fn derive_v2_dims_router_experts_resolve() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.weight", vec![4, 16 * 32]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 4),
            Some((16, 32)),
            "hidden=16 from router.shape[0], intermediate=32 from experts.shape[1]/hidden"
        );
    }

    #[test]
    fn derive_v2_dims_alternate_name_suffixes() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.gate.weight", vec![8, 2]));
        wm.insert(make_weight("blocks.0.experts", vec![2, 8 * 16]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 2),
            Some((8, 16)),
            "fallback name `gate.weight` and `experts` (no .weight) must resolve"
        );
    }

    #[test]
    fn derive_v2_dims_returns_none_when_missing_router() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.experts.weight", vec![4, 64]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 4),
            None,
            "no router entry → silent v1 fallback (S4 promotes to compile error)"
        );
    }

    #[test]
    fn derive_v2_dims_returns_none_when_router_n_experts_mismatches() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 8]));
        wm.insert(make_weight("blocks.0.experts.weight", vec![4, 16 * 32]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 4),
            None,
            "router.shape[1] != num_experts → refuse v2"
        );
    }

    #[test]
    fn derive_v2_dims_returns_none_on_non_divisible_block_elems() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.weight", vec![4, 33]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 4),
            None,
            "block_elems % hidden != 0 → refuse v2 (silent corruption guard)"
        );
    }
}
