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

/// CPDT Part III v2.5 codegen lowering for `nsl_moe_dispatch_full_v4`:
/// derive `(hidden_dim, intermediate_dim)` from the WeightMap by
/// resolving router + gate + up + down expert blocks.
///
/// Layout contract (extends v3 with a gate projection):
///   - router               shape == [hidden_dim, num_experts]
///   - experts gate-proj   shape == [num_experts, hidden_dim * intermediate_dim]
///   - experts up-proj     shape == [num_experts, hidden_dim * intermediate_dim]
///   - experts down-proj   shape == [num_experts, intermediate_dim * hidden_dim]
///
/// Name resolution suffixes: NSL's packed-layout convention uses
/// `experts.gate.weight` / `experts.up.weight` / `experts.down.weight`
/// (with `.weight` suffix optional for all three). Raw HF Mixtral
/// safetensors use per-expert `experts.{e}.w1/w2/w3` (w1=gate, w3=up,
/// w2=down for SwiGLU) — those bundles need a packing pre-step before
/// this helper resolves them. v2.next raw-HF ingestion is deferred.
///
/// Returns `None` (caller produces a hard CodegenError under
/// `moe_dispatch_swiglu` — there is no v3-fallback semantics from this
/// site) when any of:
///   - no router entry resolves
///   - no experts.gate / experts.up / experts.down entry resolves
///   - shapes aren't 2D
///   - router.shape[1] != num_experts
///   - any of {gate, up, down}.shape[0] != num_experts
///   - hidden_dim is zero
///   - gate / up / down trailing dims aren't multiples of hidden_dim
///   - the derived intermediate_dim from gate, up, and down don't all
///     agree (silent-corruption guard)
pub fn derive_v4_dims(
    weight_map: &WeightMap,
    key: &str,
    num_experts: usize,
) -> Option<(usize, usize)> {
    let router = ["router.weight", "gate.weight", "router", "gate"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts_gate = ["experts.gate.weight", "experts.gate"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts_up = ["experts.up.weight", "experts.up"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    let experts_down = ["experts.down.weight", "experts.down"]
        .iter()
        .find_map(|s| weight_map.get(&format!("{key}.{s}")))?;
    if router.shape.len() != 2
        || experts_gate.shape.len() != 2
        || experts_up.shape.len() != 2
        || experts_down.shape.len() != 2
    {
        return None;
    }
    if router.shape[1] != num_experts {
        return None;
    }
    if experts_gate.shape[0] != num_experts
        || experts_up.shape[0] != num_experts
        || experts_down.shape[0] != num_experts
    {
        return None;
    }
    let hidden = router.shape[0];
    let gate_block_elems = experts_gate.shape[1];
    let up_block_elems = experts_up.shape[1];
    let down_block_elems = experts_down.shape[1];
    if hidden == 0
        || gate_block_elems == 0
        || up_block_elems == 0
        || down_block_elems == 0
        || !gate_block_elems.is_multiple_of(hidden)
        || !up_block_elems.is_multiple_of(hidden)
        || !down_block_elems.is_multiple_of(hidden)
    {
        return None;
    }
    let intermediate_from_gate = gate_block_elems / hidden;
    let intermediate_from_up = up_block_elems / hidden;
    let intermediate_from_down = down_block_elems / hidden;
    // All three projections must agree on the inner dim. If gate and
    // up disagree, the silu(gate) * up multiply is mis-aligned. If
    // down disagrees, the down-matmul reads garbage. Refuse v4.
    if intermediate_from_gate != intermediate_from_up
        || intermediate_from_up != intermediate_from_down
    {
        return None;
    }
    Some((hidden, intermediate_from_gate))
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
    /// CPDT Part III v2.4/v2.8: activation for the v3 (paper-faithful
    /// FFN) and v4 (SwiGLU/GeGLU/ReGLU) lowering paths. Defaults to SiLU
    /// (kind=1 → SiLU on v3, SwiGLU on v4). Identity is REFUSED on v4
    /// at the moe_dispatch_swiglu codegen boundary (see
    /// `expr/calls.rs`); on v3 it is the explicit "no activation"
    /// choice. v1/v2 lowering paths ignore this field. Source:
    /// `@moe(activation="…")` decorator kwarg.
    pub activation: MoeActivation,
    /// CPDT Part III v2.7: optional WeightMap key prefix override.
    /// When `None` (the default), the v3/v4 lowering arms key into the
    /// WeightMap under the moe_configs key (`<model_name>.<field_name>`).
    /// When `Some("foo.bar")`, the lowering arms key under `foo.bar`
    /// instead — so the user can declare `@moe(weight_prefix=
    /// "model.layers.0.block_sparse_moe")` against a real HF Mixtral
    /// safetensors file (whose key segments contain `.` characters
    /// that NSL field names cannot). Source: `@moe(weight_prefix="…")`
    /// decorator kwarg.
    pub weight_prefix: Option<String>,
}

/// CPDT Part III v2.10 — validate the shape of a `weight_prefix` string
/// at decorator-parse time.
///
/// The v3/v4 WeightMap lookup composes the prefix with a suffix via
/// `format!("{prefix}.{suffix}")` — e.g.,
/// `format!("{weight_prefix}.router.weight")`. A malformed prefix
/// silently produces a key that no real safetensors file uses, so the
/// lookup just returns None and the user sees a `derive_v4_dims`-not-
/// resolved error far downstream from the actual mistake. Refuse loudly
/// at the source so the diagnostic points at the typo.
///
/// Rejected patterns (each names the offending substring):
///   - leading `.` (e.g. `".hf.layer"`) → composes to `..router.weight`
///   - trailing `.` (e.g. `"hf.layer."`) → composes to `hf.layer..router.weight`
///   - consecutive dots `..` (e.g. `"hf..layer"`) → composes with `..` in the middle
///   - any ASCII whitespace (` `, `\t`, `\n`, `\r`, etc.) — safetensors
///     keys do not contain whitespace; almost certainly a copy-paste
///     artifact from a config file
///
/// Accepted patterns (the documented HF/Mixtral forms):
///   - alphanumeric + underscores + single dots, no leading/trailing dot
///     (e.g., `"model.layers.0.block_sparse_moe"`, `"transformer.h.3.ffn"`)
///   - bare names with no dots (e.g., `"my_layer"`) — degenerate but valid
///
/// Returns Ok(()) on accept, Err(msg) on reject. The empty-string check
/// stays in the kwarg parser (the v2.7 ordering convention) and is NOT
/// duplicated here.
pub fn validate_weight_prefix(s: &str) -> Result<(), String> {
    debug_assert!(
        !s.is_empty(),
        "validate_weight_prefix called on empty string — caller should refuse empty first",
    );
    if s.starts_with('.') {
        return Err(format!(
            "@moe: weight_prefix cannot start with '.' (got \"{s}\"). \
             Resolution would compose to `..router.weight`. Drop the leading dot."
        ));
    }
    if s.ends_with('.') {
        // v2.10 fix F1 (LOW adversarial review): the parenthetical
        // narration is input-agnostic. For `"hf."` the composed key is
        // `hf..router.weight` (double-dot); for `"hf..."` it would be
        // `hf....router.weight` (four dots). "Empty segment" describes
        // both cases correctly without claiming a specific dot count.
        return Err(format!(
            "@moe: weight_prefix cannot end with '.' (got \"{s}\"). \
             Resolution would compose to `{s}.router.weight` (empty segment between \
             the prefix tail and `router`). Drop the trailing dot."
        ));
    }
    if s.contains("..") {
        return Err(format!(
            "@moe: weight_prefix cannot contain consecutive dots '..' (got \"{s}\"). \
             Each dot must separate non-empty path segments — `model.layers.0` is valid, \
             `model..layers` is not."
        ));
    }
    if let Some(ws) = s.chars().find(|c| c.is_ascii_whitespace()) {
        return Err(format!(
            "@moe: weight_prefix cannot contain whitespace (found {ws:?} in \"{s}\"). \
             Safetensors keys never contain whitespace — this is almost certainly a \
             stray space or newline copy-pasted from a config file."
        ));
    }
    Ok(())
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
            let mut weight_prefix: Option<String> = None;

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
                            "weight_prefix" => {
                                // CPDT Part III v2.7 — explicit
                                // WeightMap-key prefix for the v3/v4
                                // lowering. Required when the user's
                                // safetensors key segments contain `.`
                                // (e.g., HF Mixtral `model.layers.0.
                                // block_sparse_moe`) because NSL field
                                // names cannot. Empty-string is REJECTED
                                // here — None and "" are NOT equivalent:
                                // a Some("") would look up under
                                // `.router.weight` (with leading dot),
                                // which is never a real safetensors
                                // key. Better to refuse loudly than
                                // silently produce that lookup.
                                if let ExprKind::StringLiteral(s) = &arg.value.kind {
                                    if s.is_empty() {
                                        return Err(
                                            "@moe: weight_prefix must be a non-empty string \
                                             (e.g. weight_prefix=\"model.layers.0.block_sparse_moe\")"
                                                .to_string(),
                                        );
                                    }
                                    // CPDT Part III v2.10 — refuse malformed
                                    // prefixes at parse time so the diagnostic
                                    // points at the typo, not at the
                                    // downstream `derive_v4_dims` failure.
                                    validate_weight_prefix(s)?;
                                    weight_prefix = Some(s.clone());
                                } else {
                                    return Err(format!(
                                        "@moe: weight_prefix kwarg must be a string literal \
                                         (e.g. weight_prefix=\"model.layers.0.block_sparse_moe\"), \
                                         got non-string expression"
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
                    weight_prefix,
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
            weight_prefix: None,
        };
        assert_eq!(info.num_experts, 8);
        assert_eq!(info.top_k, 2);
        assert!((info.capacity_factor - 1.25).abs() < 1e-6);
        assert!((info.aux_loss_coeff - 0.01).abs() < 1e-6);
        assert_eq!(info.activation, MoeActivation::Silu);
        assert_eq!(info.weight_prefix, None);
    }

    #[test]
    fn test_moe_info_clone() {
        let info = MoeInfo {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 2.0,
            aux_loss_coeff: 0.05,
            activation: MoeActivation::Gelu,
            weight_prefix: Some("model.layers.0.block_sparse_moe".to_string()),
        };
        let cloned = info.clone();
        assert_eq!(cloned.num_experts, 4);
        assert_eq!(cloned.top_k, 1);
        assert_eq!(cloned.activation, MoeActivation::Gelu);
        assert_eq!(
            cloned.weight_prefix.as_deref(),
            Some("model.layers.0.block_sparse_moe")
        );
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

    // CPDT Part III v2.7 — @moe(weight_prefix=…) parsing is covered
    // end-to-end by the CLI integration test
    // crates/nsl-cli/tests/cpdt_part3_v2_7_hf_mixtral_pack.rs. The
    // empty-string and non-string-literal refusal arms are pinned by
    // that test plus the test_moe_info_clone test above (which exercises
    // weight_prefix: Some(...) in the struct surface).

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

    // ── derive_v4_dims (CPDT Part III v2.5 SwiGLU lowering) ────────────────

    #[test]
    fn derive_v4_dims_resolves_with_gate_up_down_present() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.gate.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 4),
            Some((16, 32)),
            "hidden=16 from router.shape[0]; intermediate=32 shared across gate/up/down"
        );
    }

    #[test]
    fn derive_v4_dims_resolves_alternate_name_suffixes() {
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.gate.weight", vec![8, 2]));
        wm.insert(make_weight("blocks.0.experts.gate", vec![2, 8 * 16]));
        wm.insert(make_weight("blocks.0.experts.up", vec![2, 8 * 16]));
        wm.insert(make_weight("blocks.0.experts.down", vec![2, 16 * 8]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 2),
            Some((8, 16)),
            "names without `.weight` suffix must resolve"
        );
    }

    #[test]
    fn derive_v4_dims_returns_none_when_v3_layout_missing_gate() {
        // v3 packed layout: just experts.up + experts.down, no gate.
        // v4 must refuse — caller produces a hard CodegenError.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 4),
            None,
            "v3 layout (no gate) must NOT resolve as v4"
        );
    }

    #[test]
    fn derive_v4_dims_returns_none_when_gate_disagrees_on_intermediate() {
        // gate implies intermediate=64 but up implies intermediate=32.
        // Silent corruption hazard at runtime — refuse v4.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.gate.weight", vec![4, 16 * 64]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 4),
            None,
            "gate-derived intermediate != up-derived intermediate → refuse v4"
        );
    }

    #[test]
    fn derive_v4_dims_returns_none_when_down_disagrees_on_intermediate() {
        // gate and up agree (intermediate=32) but down implies 64.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.gate.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 64 * 16]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 4),
            None,
            "down-derived intermediate != gate/up → refuse v4"
        );
    }

    #[test]
    fn derive_v4_dims_returns_none_when_gate_shape_0_mismatches_num_experts() {
        // gate.shape[0] = 6 but num_experts = 4. Silent mis-indexing
        // hazard at runtime if not caught.
        let mut wm = WeightMap::default();
        wm.insert(make_weight("blocks.0.router.weight", vec![16, 4]));
        wm.insert(make_weight("blocks.0.experts.gate.weight", vec![6, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.up.weight", vec![4, 16 * 32]));
        wm.insert(make_weight("blocks.0.experts.down.weight", vec![4, 32 * 16]));
        assert_eq!(
            derive_v4_dims(&wm, "blocks.0", 4),
            None,
            "gate.shape[0] != num_experts → refuse v4"
        );
    }

    // ── validate_weight_prefix (CPDT Part III v2.10) ─────────────────────
    //
    // Pins each of the 4 refusal kinds (leading dot, trailing dot,
    // consecutive dots, whitespace) plus happy paths. A regression that
    // dropped any single check would fail exactly one test, naming the
    // dropped case.

    #[test]
    fn validate_weight_prefix_accepts_canonical_hf_form() {
        // The documented v2.7 form — alphanumeric + underscores + single
        // dots separating non-empty segments.
        assert!(validate_weight_prefix("model.layers.0.block_sparse_moe").is_ok());
        assert!(validate_weight_prefix("transformer.h.3.ffn").is_ok());
        assert!(validate_weight_prefix("encoder.block.5.layer.1.mlp").is_ok());
    }

    #[test]
    fn validate_weight_prefix_accepts_bare_name_no_dots() {
        // Degenerate but valid — a single segment with no dots. The
        // resolver still composes `bare.router.weight` correctly.
        assert!(validate_weight_prefix("hf_layer").is_ok());
        assert!(validate_weight_prefix("layer0").is_ok());
        assert!(validate_weight_prefix("a").is_ok());
    }

    #[test]
    fn validate_weight_prefix_rejects_leading_dot() {
        let err = validate_weight_prefix(".hf.layer").unwrap_err();
        assert!(err.contains("cannot start with '.'"), "msg was: {err}");
        assert!(err.contains(".hf.layer"), "msg must echo the offending prefix");
        // Even a single leading dot is rejected (no special case for
        // "single-char prefix").
        assert!(validate_weight_prefix(".x").is_err());
        assert!(validate_weight_prefix(".").is_err());
    }

    #[test]
    fn validate_weight_prefix_rejects_trailing_dot() {
        let err = validate_weight_prefix("hf.layer.").unwrap_err();
        assert!(err.contains("cannot end with '.'"), "msg was: {err}");
        assert!(err.contains("hf.layer."), "msg must echo the offending prefix");
        // A bare "." would hit the leading-dot check first; that's fine
        // — both refusals are correct for that degenerate input.
    }

    #[test]
    fn validate_weight_prefix_rejects_consecutive_dots() {
        // v2.10 fix F4 (LOW adversarial review): also pin the
        // offending-prefix echo so a regression that drops `(got "{s}")`
        // is caught.
        let err = validate_weight_prefix("model..layers.0").unwrap_err();
        assert!(err.contains("consecutive dots"), "msg was: {err}");
        assert!(err.contains("model..layers.0"), "msg must echo prefix: {err}");
        // Even longer runs are rejected (the `..` substring check covers
        // any run of length ≥ 2).
        assert!(validate_weight_prefix("a...b").is_err());
        // Embedded in the middle is rejected too.
        assert!(validate_weight_prefix("transformer.h..3.ffn").is_err());
    }

    #[test]
    fn validate_weight_prefix_rejects_whitespace() {
        // Common copy-paste hazards: stray space, tab, newline. v2.10
        // fix F4 (LOW adversarial review): also pin the offending-prefix
        // echo so a regression that drops `(got "{s}")` is caught.
        let err_space = validate_weight_prefix("model. layers.0").unwrap_err();
        assert!(err_space.contains("whitespace"), "msg was: {err_space}");
        assert!(err_space.contains("model. layers.0"), "msg must echo prefix: {err_space}");
        assert!(validate_weight_prefix("model.\tlayers").is_err());
        assert!(validate_weight_prefix("model.\nlayers").is_err());
        // Trailing space (also caught by whitespace before trailing-dot).
        assert!(validate_weight_prefix("model.layers ").is_err());
        // Leading space.
        assert!(validate_weight_prefix(" model.layers").is_err());
    }

    #[test]
    fn validate_weight_prefix_refusal_ordering_pins_leading_dot_first() {
        // Multiple violations: leading-dot AND trailing-dot AND
        // consecutive dots AND whitespace. Leading-dot is checked first
        // (the resolver's compose order would fail there first), so the
        // refusal must name THAT case — a regression that changed check
        // order would surface as a different message.
        let err = validate_weight_prefix(". .layer..").unwrap_err();
        assert!(
            err.contains("cannot start with '.'"),
            "leading-dot must win over other violations; msg was: {err}"
        );
    }
}
