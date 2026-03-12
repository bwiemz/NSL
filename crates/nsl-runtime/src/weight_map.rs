// M18b: Name mapping conventions + transpose rules

use std::collections::HashMap;

#[cfg(feature = "interop")]
use regex_lite::Regex;

// ─────────────────────────────────────────────────────────────────────────────
// Standard HuggingFace → NSL name mappings
// ─────────────────────────────────────────────────────────────────────────────

const NAME_MAPPINGS: &[(&str, &str)] = &[
    ("self_attn", "attention"),
    ("mlp.gate_proj", "ffn.gate"),
    ("mlp.up_proj", "ffn.up"),
    ("mlp.down_proj", "ffn.down"),
    ("embed_tokens", "embedding"),
    ("lm_head", "head"),
    ("input_layernorm", "norm1"),
    ("post_attention_layernorm", "norm2"),
    ("model.", ""),  // strip common "model." prefix
];

// ─────────────────────────────────────────────────────────────────────────────
// map_hf_name: HuggingFace name → NSL field path
// ─────────────────────────────────────────────────────────────────────────────

/// Map a HuggingFace parameter name to an NSL model field path.
///
/// Applies user overrides first, then standard substitutions, then converts
/// `layers.N.` patterns to `layers[N].` using regex.
pub fn map_hf_name(hf_name: &str, overrides: &HashMap<String, String>) -> String {
    // 1. User overrides take precedence
    if let Some(mapped) = overrides.get(hf_name) {
        return mapped.clone();
    }

    // 2. Apply standard substitutions sequentially
    let mut result = hf_name.to_string();
    for &(from, to) in NAME_MAPPINGS {
        result = result.replace(from, to);
    }

    // 3. Convert layers.N. → layers[N].
    #[cfg(feature = "interop")]
    {
        let re = Regex::new(r"layers\.(\d+)\.").unwrap();
        result = re.replace_all(&result, "layers[$1].").to_string();
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// needs_transpose: whether a weight tensor should be transposed on load
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if the weight at `field_name` in `layer_type` should be
/// transposed (i.e., it is a Linear weight, not bias/embedding/norm).
pub fn needs_transpose(field_name: &str, layer_type: &str) -> bool {
    // Never transpose biases
    if field_name.ends_with(".bias") || field_name == "bias" {
        return false;
    }

    // Never transpose embedding tables or norm weights/scales
    let layer_lower = layer_type.to_lowercase();
    if layer_lower.contains("embedding")
        || layer_lower.contains("layernorm")
        || layer_lower.contains("rmsnorm")
    {
        return false;
    }

    // Also detect norm by common weight names
    let name_lower = field_name.to_lowercase();
    if name_lower.contains("norm") && name_lower.ends_with(".weight") {
        return false;
    }
    if name_lower.contains("embedding") {
        return false;
    }

    // Linear weights get transposed (PyTorch stores them as [out, in])
    if layer_lower.contains("linear") {
        return true;
    }

    // Heuristic: if the layer type is unknown but the field is "weight", assume Linear
    if field_name.ends_with(".weight") || field_name == "weight" {
        return true;
    }

    false
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_mapping: check all model params have matching weights
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that model parameters can be resolved to weight dict entries.
///
/// Returns `(matched, unmapped, unused)` where:
/// - `matched`: param names that have a matching weight
/// - `unmapped`: param names with no weight found
/// - `unused`: weight names not matched to any param
pub fn validate_mapping(
    model_params: &[String],
    weight_names: &[String],
    overrides: &HashMap<String, String>,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let weight_set: HashMap<&str, ()> = weight_names.iter().map(|n| (n.as_str(), ())).collect();
    let mut matched = Vec::new();
    let mut unmapped = Vec::new();
    let mut used_weights = std::collections::HashSet::new();

    for param in model_params {
        let mapped = map_hf_name(param, overrides);
        if weight_set.contains_key(mapped.as_str()) {
            matched.push(param.clone());
            used_weights.insert(mapped.clone());
        } else {
            unmapped.push(param.clone());
        }
    }

    let unused: Vec<String> = weight_names
        .iter()
        .filter(|w| !used_weights.contains(*w))
        .cloned()
        .collect();

    (matched, unmapped, unused)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn no_overrides() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn test_map_hf_name_basic() {
        let result = map_hf_name("model.layers.0.self_attn.q_proj.weight", &no_overrides());
        assert_eq!(result, "layers[0].attention.q_proj.weight");
    }

    #[test]
    fn test_map_hf_name_embedding() {
        let result = map_hf_name("model.embed_tokens.weight", &no_overrides());
        assert_eq!(result, "embedding.weight");
    }

    #[test]
    fn test_map_hf_name_with_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "model.embed_tokens.weight".to_string(),
            "my_custom_embedding.weight".to_string(),
        );
        let result = map_hf_name("model.embed_tokens.weight", &overrides);
        assert_eq!(result, "my_custom_embedding.weight");
    }

    #[test]
    fn test_layer_index_conversion() {
        let result = map_hf_name("model.layers.5.mlp.gate_proj.weight", &no_overrides());
        assert_eq!(result, "layers[5].ffn.gate.weight");
    }

    #[test]
    fn test_needs_transpose_linear() {
        assert!(needs_transpose("weight", "Linear"));
    }

    #[test]
    fn test_no_transpose_embedding() {
        assert!(!needs_transpose("weight", "Embedding"));
        assert!(!needs_transpose("weight", "LayerNorm"));
        assert!(!needs_transpose("weight", "RMSNorm"));
    }

    #[test]
    fn test_no_transpose_bias() {
        assert!(!needs_transpose("bias", "Linear"));
        assert!(!needs_transpose("output.bias", "Linear"));
    }

    #[test]
    fn test_map_hf_name_lm_head() {
        let result = map_hf_name("lm_head.weight", &no_overrides());
        assert_eq!(result, "head.weight");
    }

    #[test]
    fn test_map_hf_name_layernorm() {
        let result = map_hf_name("model.layers.2.input_layernorm.weight", &no_overrides());
        assert_eq!(result, "layers[2].norm1.weight");
    }

    #[test]
    fn test_validate_mapping_basic() {
        let params = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
        ];
        let weights = vec![
            "embedding.weight".to_string(),
            "layers[0].attention.q_proj.weight".to_string(),
        ];
        let (matched, unmapped, unused) = validate_mapping(&params, &weights, &no_overrides());
        assert_eq!(matched.len(), 2);
        assert!(unmapped.is_empty());
        assert!(unused.is_empty());
    }

    #[test]
    fn test_validate_mapping_unmapped() {
        let params = vec!["unknown.param".to_string()];
        let weights = vec!["some.weight".to_string()];
        let (matched, unmapped, unused) = validate_mapping(&params, &weights, &no_overrides());
        assert!(matched.is_empty());
        assert_eq!(unmapped.len(), 1);
        assert_eq!(unused.len(), 1);
    }
}
