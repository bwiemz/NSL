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
    // experts.shape[0] must also match num_experts. This is independent
    // of the router check because router and experts could be sourced
    // from different stages (e.g. only one was pruned, leaving them out
    // of sync). Without this guard, a mismatch silently mis-indexes the
    // packed expert blocks at runtime.
    if experts.shape[0] != num_experts {
        return None;
    }
    let hidden = router.shape[0];
    let block_elems = experts.shape[1];
    if hidden == 0 || block_elems == 0 || !block_elems.is_multiple_of(hidden) {
        return None;
    }
    Some((hidden, block_elems / hidden))
}

/// CPDT Part III v2.3 codegen lowering for `nsl_moe_dispatch_full_v3`:
/// derive `(hidden_dim, intermediate_dim)` from the WeightMap by
/// resolving BOTH the up- and down-projection expert blocks.
///
/// Layout contract (mirrors v2 packed convention, extended for two
/// matmuls):
///   - router               shape == [hidden_dim, num_experts]
///   - experts up-proj     shape == [num_experts, hidden_dim * intermediate_dim]
///   - experts down-proj   shape == [num_experts, intermediate_dim * hidden_dim]
///
/// Name resolution suffixes mirror cpdt_expert_prune's resilience to
/// alternate naming (some safetensors bundles use `gate.weight` instead
/// of `router.weight`; `experts.weight` is the v2 single-matmul name).
/// For v3 the up/down keys MUST be distinguished: NSL's packed-layout
/// convention uses `experts.up.weight` / `experts.down.weight` (with
/// the `.weight` suffix optional for both). NOTE: raw HF Mixtral
/// safetensors use a DIFFERENT layout (per-expert `experts.{e}.w1` /
/// `w2` / `w3` for SwiGLU) — those bundles need to be packed into NSL's
/// 2D `[num_experts, hidden * intermediate]` row-major layout by an
/// ingestion pre-step before this helper will resolve them. v2.4 may
/// add a raw-HF ingestion path; for v2.3 the packed layout is the
/// contract.
///
/// Returns `None` (silent v2 fallback at the call site) when any of:
///   - no router entry resolves under the key
///   - no experts-up entry resolves under the key
///   - no experts-down entry resolves under the key
///   - shapes aren't 2D
///   - router.shape[1] != num_experts
///   - experts_up.shape[0] != num_experts OR experts_down.shape[0] != num_experts
///   - hidden_dim is zero
///   - experts_up trailing dim isn't a multiple of hidden_dim
///   - the derived intermediate_dim from up doesn't match the derived
///     intermediate_dim from down
///
/// The call site promotes `None` to a hard compile error under
/// `cpdt_mode == Full` (same shape as S4 from v1's production-forward
/// landing).
pub fn derive_v3_dims(
    weight_map: &WeightMap,
    key: &str,
    num_experts: usize,
) -> Option<(usize, usize)> {
    let router = ["router.weight", "gate.weight", "router", "gate"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts_up = ["experts.up.weight", "experts.up"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts_down = ["experts.down.weight", "experts.down"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    if router.shape.len() != 2
        || experts_up.shape.len() != 2
        || experts_down.shape.len() != 2
    {
        return None;
    }
    if router.shape[1] != num_experts {
        return None;
    }
    if experts_up.shape[0] != num_experts || experts_down.shape[0] != num_experts {
        return None;
    }
    let hidden = router.shape[0];
    let up_block_elems = experts_up.shape[1];
    let down_block_elems = experts_down.shape[1];
    if hidden == 0
        || up_block_elems == 0
        || down_block_elems == 0
        || !up_block_elems.is_multiple_of(hidden)
        || !down_block_elems.is_multiple_of(hidden)
    {
        return None;
    }
    let intermediate_from_up = up_block_elems / hidden;
    let intermediate_from_down = down_block_elems / hidden;
    if intermediate_from_up != intermediate_from_down {
        // Up and down disagree on the inner dim — silent corruption
        // hazard at runtime. Refuse v3.
        return None;
    }
    Some((hidden, intermediate_from_up))
}

/// CPDT Part III v2.4: activation selector for the v3 paper-faithful
/// MoE FFN. Matches the runtime `activation_kind` enum exactly so
/// `as i64` produces the FFI value with no translation table.
///
/// Values are pinned by their numeric repr — DO NOT renumber. The
/// runtime FFI at `crates/nsl-runtime/src/moe/ffi.rs::nsl_moe_dispatch_full_v3`
/// branches on these literal integers.
#[repr(i64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeActivation {
    /// No activation (identity) — test-only; production should use
    /// a real nonlinearity.
    Identity = 0,
    /// SiLU = `x * sigmoid(x)`. Mixtral / DeepSeek MoE default.
    Silu = 1,
    /// GELU (tanh approximation, matches `torch.gelu(approximate='tanh')`).
    Gelu = 2,
    /// ReLU = `max(0, x)`.
    Relu = 3,
}

impl MoeActivation {
    /// Parse a source-level activation name (the value of the
    /// `@moe(activation="…")` kwarg). Case-insensitive on the kind
    /// string itself. Returns None for unknown names — callers must
    /// surface this as a decorator-validation error rather than
    /// silently fall back to a default.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "identity" | "none" => Some(Self::Identity),
            "silu" | "swish" => Some(Self::Silu),
            "gelu" => Some(Self::Gelu),
            "relu" => Some(Self::Relu),
            _ => None,
        }
    }
}

impl Default for MoeActivation {
    fn default() -> Self {
        Self::Silu
    }
}

/// Compile-time info about a MoE layer.
#[derive(Debug, Clone)]
pub struct MoeInfo {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub aux_loss_coeff: f32,
    /// CPDT Part III v2.4: activation for the v3 (paper-faithful FFN)
    /// lowering path. Defaults to SiLU. Source: `@moe(activation="…")`
    /// decorator kwarg. Unread by the v1/v2 lowering paths.
    pub activation: MoeActivation,
}

/// Extract @moe decorator from a list of decorators.
///
/// The `activation` kwarg (v2.4) is INVALID-VALUE-FATAL: if the kwarg
/// is present but its string value doesn't resolve via
/// `MoeActivation::from_str`, this returns `Err` so the build fails
/// loudly. Silent fallback to the default would mask a typo'd
/// `@moe(activation="gleu")` and produce production output that
/// doesn't match what the source code asked for.
///
/// Missing `activation` kwarg → `MoeActivation::default()` (= SiLU),
/// matching the v2.3 hardcoded behavior for back-compat.
pub fn extract_moe_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Result<Option<MoeInfo>, String> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "moe" {
            let mut num_experts: usize = 0;
            let mut top_k: usize = 2;
            let mut capacity_factor: f32 = 1.25;
            let mut aux_loss_coeff: f32 = 0.01;
            let mut activation: MoeActivation = MoeActivation::default();

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
                            "activation" => {
                                // INVALID-VALUE-FATAL: typo guard. A
                                // silent default would mask
                                // `@moe(activation="gleu")` and ship
                                // production output that doesn't match
                                // source intent.
                                if let ExprKind::StringLiteral(s) = &arg.value.kind {
                                    activation = MoeActivation::from_str(s).ok_or_else(|| format!(
                                        "@moe: unknown activation \"{}\". Supported (v2.4): \
                                         silu/swish (default), gelu, relu, identity/none. \
                                         SwiGLU is a v2.5 deferral (needs a third gate-projection weight).",
                                        s
                                    ))?;
                                } else {
                                    return Err(format!(
                                        "@moe: activation kwarg must be a string literal \
                                         (e.g. activation=\"silu\"), got non-string expression"
                                    ));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if num_experts > 0 {
                return Ok(Some(MoeInfo {
                    num_experts,
                    top_k,
                    capacity_factor,
                    aux_loss_coeff,
                    activation,
                }));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_moe_empty_decorators() {
        let decorators: Vec<Decorator> = vec![];
        let result = extract_moe_decorator(&decorators, &|_| "");
        assert!(matches!(result, Ok(None)), "empty decorators should resolve to Ok(None)");
    }

    #[test]
    fn test_moe_info_defaults() {
        // Verify default values used when only num_experts is provided
        let info = MoeInfo {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            aux_loss_coeff: 0.01,
            activation: MoeActivation::Silu,
        };
        assert_eq!(info.num_experts, 8);
        assert_eq!(info.top_k, 2);
        assert!((info.capacity_factor - 1.25).abs() < 1e-6);
        assert!((info.aux_loss_coeff - 0.01).abs() < 1e-6);
        assert_eq!(info.activation, MoeActivation::Silu);
    }

    #[test]
    fn test_moe_info_clone() {
        let info = MoeInfo {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 2.0,
            aux_loss_coeff: 0.05,
            activation: MoeActivation::Gelu,
        };
        let cloned = info.clone();
        assert_eq!(cloned.num_experts, 4);
        assert_eq!(cloned.top_k, 1);
        assert_eq!(cloned.activation, MoeActivation::Gelu);
    }

    // ── MoeActivation parser (CPDT Part III v2.4) ──────────────────────────

    #[test]
    fn moe_activation_from_str_canonical_names() {
        assert_eq!(MoeActivation::from_str("silu"), Some(MoeActivation::Silu));
        assert_eq!(MoeActivation::from_str("gelu"), Some(MoeActivation::Gelu));
        assert_eq!(MoeActivation::from_str("relu"), Some(MoeActivation::Relu));
        assert_eq!(MoeActivation::from_str("identity"), Some(MoeActivation::Identity));
    }

    #[test]
    fn moe_activation_from_str_aliases_and_case() {
        // `swish` is an alias for `silu` (older literature). `none` is
        // the explicit identity alias. Case-insensitive.
        assert_eq!(MoeActivation::from_str("swish"), Some(MoeActivation::Silu));
        assert_eq!(MoeActivation::from_str("none"), Some(MoeActivation::Identity));
        assert_eq!(MoeActivation::from_str("SiLU"), Some(MoeActivation::Silu));
        assert_eq!(MoeActivation::from_str("ReLU"), Some(MoeActivation::Relu));
    }

    #[test]
    fn moe_activation_from_str_rejects_unknown() {
        // Common typos that the INVALID-VALUE-FATAL guard must catch.
        // The parser refuses; collection.rs surfaces the error as a
        // build failure rather than silent default.
        assert_eq!(MoeActivation::from_str("gleu"), None);
        assert_eq!(MoeActivation::from_str("swiglu"), None, "SwiGLU needs a 3rd weight tensor — v2.5 deferral");
        assert_eq!(MoeActivation::from_str(""), None);
        assert_eq!(MoeActivation::from_str("relu6"), None);
    }

    #[test]
    fn moe_activation_default_is_silu() {
        assert_eq!(MoeActivation::default(), MoeActivation::Silu,
            "Mixtral/DeepSeek default must be SiLU; v2.3 hardcoded it and the default keeps back-compat.");
    }

    #[test]
    fn moe_activation_repr_matches_ffi_contract() {
        // The runtime FFI's activation_kind switch hardcodes
        // 0/1/2/3 in nsl-runtime/src/moe/ffi.rs. If anyone renumbers
        // this enum, the FFI dispatch silently mis-selects. Pin it.
        assert_eq!(MoeActivation::Identity as i64, 0);
        assert_eq!(MoeActivation::Silu as i64, 1);
        assert_eq!(MoeActivation::Gelu as i64, 2);
        assert_eq!(MoeActivation::Relu as i64, 3);
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
    fn derive_v2_dims_returns_none_when_experts_n_axis_mismatches() {
        // Router and num_experts agree, but experts.shape[0] disagrees.
        // Without this guard, a partially-pruned WeightMap would
        // silently mis-index the packed expert blocks at runtime.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.weight", vec![6, 16 * 32]));
        assert_eq!(
            derive_v2_dims(&wm, "blocks.0", 4),
            None,
            "experts.shape[0] != num_experts → refuse v2 (router/experts out of sync)"
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

    // ── derive_v3_dims (CPDT Part III v2.3 paper-faithful FFN lowering) ───
    #[test]
    fn derive_v3_dims_resolves_with_up_and_down_present() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 4),
            Some((16, 32)),
            "hidden=16 from router.shape[0], intermediate=32 from experts.up.shape[1] / hidden"
        );
    }

    #[test]
    fn derive_v3_dims_resolves_alternate_name_suffixes() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.gate.weight", vec![8, 2]));
        wm.insert(make_weight("blocks.0.experts.up", vec![2, 8 * 16]));
        wm.insert(make_weight("blocks.0.experts.down", vec![2, 16 * 8]));
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 2),
            Some((8, 16)),
            "fallback names without `.weight` suffix must resolve"
        );
    }

    #[test]
    fn derive_v3_dims_returns_none_when_only_v2_single_experts_present() {
        // v2 single-matmul layout: just experts.weight, no up/down split.
        // v3 must refuse — caller falls through to v2 emission.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.weight", vec![4, 16 * 32]));
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 4),
            None,
            "v2 single-matmul layout has no up/down split → derive_v3_dims must refuse"
        );
    }

    #[test]
    fn derive_v3_dims_returns_none_when_only_up_present() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        // No experts.down entry.
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 4),
            None,
            "missing experts.down → refuse v3"
        );
    }

    #[test]
    fn derive_v3_dims_returns_none_when_up_and_down_disagree_on_intermediate() {
        // up implies intermediate=32 (16*32 / 16), down implies
        // intermediate=64 (16*64 / 16). Silent-corruption hazard if not
        // caught — v3 would still compute but with mis-aligned trailing
        // axis, producing plausible-looking garbage.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 16 * 64]));
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 4),
            None,
            "up-derived intermediate != down-derived intermediate → refuse v3 (silent-corruption guard)"
        );
    }

    #[test]
    fn derive_v3_dims_returns_none_when_router_n_experts_mismatches() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 8]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v3_dims(&wm, "blocks.0", 4),
            None,
            "router.shape[1] != num_experts → refuse v3"
        );
    }
}
