// M18b: HuggingFace Hub download + weight loading

use std::collections::HashMap;
use std::ffi::c_char;

use crate::tensor::NslTensor;

// ─────────────────────────────────────────────────────────────────────────────
// ParamMeta: compiler-generated offset table entry
// ─────────────────────────────────────────────────────────────────────────────

/// Compiler-generated descriptor for a single model parameter field.
/// The compiler emits an array of these as a static table, passed to
/// `nsl_hf_load` so the runtime can map weight tensors into the model struct.
#[repr(C)]
pub struct ParamMeta {
    /// Null-terminated field path (e.g., "layers[0].attention.q_proj.weight")
    pub name_ptr: *const c_char,
    /// Byte offset into the model struct where the tensor pointer lives
    pub offset: i64,
    /// Expected shape dims (0 = no compile-time shape constraint)
    pub shape_ptr: *const i64,
    /// Number of shape dimensions
    pub ndim: i64,
    /// 1 = transpose the weight before storing, 0 = store as-is
    pub transpose: i64,
}

// SAFETY: ParamMeta contains raw pointers but is only used from a single
// thread in the context of nsl_hf_load.
unsafe impl Send for ParamMeta {}
unsafe impl Sync for ParamMeta {}

// ─────────────────────────────────────────────────────────────────────────────
// transpose_2d: transpose a [M, N] f32 tensor to [N, M]
// ─────────────────────────────────────────────────────────────────────────────

/// Transpose a 2D f32 tensor (shape [M, N]) to shape [N, M].
/// Returns a new tensor pointer; the original is not modified.
pub fn transpose_2d(tensor_ptr: i64) -> i64 {
    crate::tensor::nsl_tensor_transpose(tensor_ptr, 0, 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// load_weights_from_dict: core mapping logic (testable without network)
// ─────────────────────────────────────────────────────────────────────────────

/// For each ParamMeta entry, find the matching weight in `dict_ptr` (an NslDict
/// of name → tensor pointer), validate shape if constraints are set, optionally
/// transpose, and write the tensor pointer at `model_ptr + offset`.
///
/// This function is the testable core — it does not require network access.
pub fn load_weights_from_dict(model_ptr: i64, metadata: &[ParamMeta], dict_ptr: i64) {
    let overrides: HashMap<String, String> = HashMap::new();

    for meta in metadata {
        // Resolve field path string
        let field_name = unsafe {
            std::ffi::CStr::from_ptr(meta.name_ptr)
                .to_str()
                .unwrap_or("[bad utf8]")
                .to_owned()
        };

        // Look up in dict — try field_name directly, then try mapping through HF conventions
        let weight_key = field_name.clone();

        // Check if the key exists in the dict
        let key_ptr = crate::safetensors_io::alloc_c_string(&weight_key);
        let contains = crate::dict::nsl_dict_contains(dict_ptr, key_ptr as i64);
        let tensor_ptr = if contains != 0 {
            crate::dict::nsl_dict_get_str(dict_ptr, key_ptr as i64)
        } else {
            // Try reverse-mapping: maybe dict keys are HF-style, meta names are NSL-style
            // Search dict keys for one that maps to our field_name
            let keys_list_ptr = crate::dict::nsl_dict_keys(dict_ptr);
            let keys_list = crate::list::NslList::from_ptr(keys_list_ptr);
            let mut found_ptr: i64 = 0;
            for i in 0..keys_list.len as usize {
                let key_i64 = unsafe { *keys_list.data.add(i) };
                let key_str = unsafe {
                    std::ffi::CStr::from_ptr(key_i64 as *const c_char)
                        .to_str()
                        .unwrap_or("")
                        .to_owned()
                };
                let mapped = crate::weight_map::map_hf_name(&key_str, &overrides);
                if mapped == field_name {
                    found_ptr = crate::dict::nsl_dict_get_str(dict_ptr, key_i64);
                    break;
                }
            }
            if found_ptr == 0 {
                eprintln!(
                    "[nsl] hf_load error: no weight found for model field '{}'",
                    field_name
                );
                eprintln!("[nsl] available weights in safetensors file:");
                let keys_list = crate::dict::nsl_dict_keys(dict_ptr);
                let n_keys = crate::list::nsl_list_len(keys_list);
                for k in 0..n_keys {
                    let k_ptr = crate::list::nsl_list_get(keys_list, k);
                    let k_str = unsafe { std::ffi::CStr::from_ptr(k_ptr as *const std::ffi::c_char) }
                        .to_str()
                        .unwrap_or("?");
                    eprintln!("  - {}", k_str);
                }
                std::process::abort();
            }
            found_ptr
        };

        // Validate shape if compile-time constraints are present
        if !meta.shape_ptr.is_null() && meta.ndim > 0 {
            let tensor = NslTensor::from_ptr(tensor_ptr);
            if tensor.ndim != meta.ndim {
                eprintln!(
                    "[nsl] hf_load: shape mismatch for '{}': expected ndim={}, got ndim={}",
                    field_name, meta.ndim, tensor.ndim
                );
                std::process::abort();
            }
            for d in 0..meta.ndim as usize {
                let expected = unsafe { *meta.shape_ptr.add(d) };
                if expected == 0 {
                    continue; // 0 = no constraint for this dim
                }
                let actual = unsafe { *tensor.shape.add(d) };
                if actual != expected {
                    eprintln!(
                        "[nsl] hf_load: shape mismatch for '{}' dim {}: expected {}, got {}",
                        field_name, d, expected, actual
                    );
                    std::process::abort();
                }
            }
        }

        // Optionally transpose
        let final_ptr = if meta.transpose != 0 {
            transpose_2d(tensor_ptr)
        } else {
            crate::tensor::nsl_tensor_clone(tensor_ptr)
        };

        // Write tensor pointer into model struct at the specified byte offset
        let slot = (model_ptr + meta.offset) as *mut i64;
        unsafe { *slot = final_ptr };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// nsl_hf_load: FFI entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Download a HuggingFace model's `model.safetensors` (or `pytorch_model.bin`
/// converted to safetensors) and load its weights into a compiled NSL model struct.
///
/// Arguments:
/// - `model_ptr`: pointer to the model struct (must be valid for write)
/// - `metadata_ptr`: pointer to array of `ParamMeta` descriptors
/// - `metadata_len`: number of entries in the metadata array
/// - `repo_id_ptr`: pointer to UTF-8 repo ID string (e.g., "meta-llama/Llama-2-7b-hf")
/// - `repo_id_len`: byte length of the repo ID string
/// - `device`: 0 = CPU, 1+ = CUDA device ID
///
/// Returns 0 on success, aborts on error.
#[no_mangle]
pub extern "C" fn nsl_hf_load(
    model_ptr: i64,
    metadata_ptr: i64,
    metadata_len: i64,
    repo_id_ptr: i64,
    repo_id_len: i64,
    device: i64,
) -> i64 {
    #[cfg(feature = "interop")]
    {
        use hf_hub::api::sync::Api;

        let repo_id = unsafe {
            let slice =
                std::slice::from_raw_parts(repo_id_ptr as *const u8, repo_id_len as usize);
            std::str::from_utf8(slice).unwrap_or_else(|_| {
                eprintln!("[nsl] hf_load: repo_id is not valid UTF-8");
                std::process::abort();
            })
        };

        // Build the hf-hub API client and download model.safetensors
        let api = Api::new().unwrap_or_else(|e| {
            eprintln!("[nsl] hf_load: failed to create hf-hub Api: {}", e);
            std::process::abort();
        });

        let model_repo = api.model(repo_id.to_string());

        let dict_ptr = match model_repo.get("model.safetensors") {
            Ok(local_path) => {
                // Single-file model
                let path_str = local_path.to_str().unwrap_or_else(|| {
                    eprintln!("[nsl] hf_load: downloaded path is not valid UTF-8");
                    std::process::abort();
                });
                crate::safetensors_io::nsl_safetensors_load(
                    path_str.as_ptr() as i64,
                    path_str.len() as i64,
                    device,
                )
            }
            Err(_single_err) => {
                // Fallback: try sharded model via index JSON
                let index_path = model_repo
                    .get("model.safetensors.index.json")
                    .unwrap_or_else(|e| {
                        eprintln!(
                            "[nsl] hf_load: no model.safetensors or index found for '{}': {}",
                            repo_id, e
                        );
                        std::process::abort();
                    });

                let index_content = std::fs::read_to_string(&index_path).unwrap_or_else(|e| {
                    eprintln!("[nsl] hf_load: failed to read index JSON: {}", e);
                    std::process::abort();
                });

                let index_json: serde_json::Value =
                    serde_json::from_str(&index_content).unwrap_or_else(|e| {
                        eprintln!("[nsl] hf_load: failed to parse index JSON: {}", e);
                        std::process::abort();
                    });

                // Extract unique shard filenames from weight_map values
                let weight_map = index_json
                    .get("weight_map")
                    .and_then(|v| v.as_object())
                    .unwrap_or_else(|| {
                        eprintln!("[nsl] hf_load: index JSON missing 'weight_map' object");
                        std::process::abort();
                    });

                let mut shard_names: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                    .collect();
                shard_names.sort();
                shard_names.dedup();

                eprintln!(
                    "[nsl] hf_load: sharded model detected for '{}', downloading {} shard(s)...",
                    repo_id,
                    shard_names.len()
                );

                // Create a merged dict from all shards
                let merged_dict = crate::dict::nsl_dict_new();

                for shard_name in &shard_names {
                    eprintln!("[nsl] hf_load: downloading shard '{}'...", shard_name);
                    let shard_path =
                        model_repo.get(shard_name).unwrap_or_else(|e| {
                            eprintln!(
                                "[nsl] hf_load: failed to download shard '{}': {}",
                                shard_name, e
                            );
                            std::process::abort();
                        });

                    let shard_path_str = shard_path.to_str().unwrap_or_else(|| {
                        eprintln!("[nsl] hf_load: shard path is not valid UTF-8");
                        std::process::abort();
                    });

                    let shard_dict = crate::safetensors_io::nsl_safetensors_load(
                        shard_path_str.as_ptr() as i64,
                        shard_path_str.len() as i64,
                        device,
                    );

                    // Merge shard tensors into the combined dict
                    let keys_list_ptr = crate::dict::nsl_dict_keys(shard_dict);
                    let keys_list = crate::list::NslList::from_ptr(keys_list_ptr);
                    for i in 0..keys_list.len as usize {
                        let key_i64 = unsafe { *keys_list.data.add(i) };
                        let tensor_ptr =
                            crate::dict::nsl_dict_get_str(shard_dict, key_i64);
                        crate::dict::nsl_dict_set_str(merged_dict, key_i64, tensor_ptr);
                    }
                    // Free the per-shard dict structure (values moved to merged_dict)
                    crate::dict::nsl_dict_free(shard_dict);
                }

                eprintln!(
                    "[nsl] hf_load: loaded {} tensors from {} shard(s)",
                    crate::dict::nsl_dict_len(merged_dict),
                    shard_names.len()
                );

                merged_dict
            }
        };

        // Map weights into model struct using the metadata table
        let metadata_slice = unsafe {
            std::slice::from_raw_parts(metadata_ptr as *const ParamMeta, metadata_len as usize)
        };

        load_weights_from_dict(model_ptr, metadata_slice, dict_ptr);

        0 // success
    }

    #[cfg(not(feature = "interop"))]
    {
        let _ = (model_ptr, metadata_ptr, metadata_len, repo_id_ptr, repo_id_len, device);
        eprintln!("[nsl] nsl_hf_load: requires 'interop' feature");
        std::process::abort();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_io::{alloc_c_string, allocate_f32_tensor};

    /// A mock model struct with two tensor pointer fields.
    #[repr(C)]
    struct MockModel {
        field_a: i64,
        field_b: i64,
    }

    #[test]
    fn test_load_weights_into_struct() {
        // Create two small tensors
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape_a = [2usize, 2];
        let tensor_a = allocate_f32_tensor(&data_a, &shape_a, 2, 4, 0);
        let tensor_a_ptr = Box::into_raw(Box::new(tensor_a)) as i64;

        let data_b = vec![5.0f32, 6.0, 7.0];
        let shape_b = [3usize];
        let tensor_b = allocate_f32_tensor(&data_b, &shape_b, 1, 3, 0);
        let tensor_b_ptr = Box::into_raw(Box::new(tensor_b)) as i64;

        // Build a dict with two entries: "field_a" and "field_b"
        let dict_ptr = crate::dict::nsl_dict_new();
        let key_a = alloc_c_string("field_a") as i64;
        let key_b = alloc_c_string("field_b") as i64;
        crate::dict::nsl_dict_set_str(dict_ptr, key_a, tensor_a_ptr);
        crate::dict::nsl_dict_set_str(dict_ptr, key_b, tensor_b_ptr);

        // Build ParamMeta entries for field_a (offset 0) and field_b (offset 8)
        let name_a = alloc_c_string("field_a") as *const c_char;
        let name_b = alloc_c_string("field_b") as *const c_char;

        let metadata = vec![
            ParamMeta {
                name_ptr: name_a,
                offset: 0,
                shape_ptr: std::ptr::null(),
                ndim: 0,
                transpose: 0,
            },
            ParamMeta {
                name_ptr: name_b,
                offset: std::mem::size_of::<i64>() as i64,
                shape_ptr: std::ptr::null(),
                ndim: 0,
                transpose: 0,
            },
        ];

        // Allocate a MockModel and call load_weights_from_dict
        let mut model = MockModel {
            field_a: 0,
            field_b: 0,
        };
        let model_ptr = &mut model as *mut MockModel as i64;

        load_weights_from_dict(model_ptr, &metadata, dict_ptr);

        // Both fields should now hold non-zero tensor pointers
        assert_ne!(model.field_a, 0, "field_a should be a non-zero tensor pointer");
        assert_ne!(model.field_b, 0, "field_b should be a non-zero tensor pointer");

        // Verify the tensor data is accessible
        let ta = NslTensor::from_ptr(model.field_a);
        assert_eq!(ta.len, 4);
        let v0 = unsafe { *ta.data_f32().add(0) };
        assert!((v0 - 1.0f32).abs() < 1e-5, "field_a[0] should be 1.0, got {}", v0);

        let tb = NslTensor::from_ptr(model.field_b);
        assert_eq!(tb.len, 3);
        let v0b = unsafe { *tb.data_f32().add(0) };
        assert!((v0b - 5.0f32).abs() < 1e-5, "field_b[0] should be 5.0, got {}", v0b);
    }

    #[test]
    fn test_load_weights_with_transpose() {
        // Create a [2, 3] tensor to transpose to [3, 2]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let tensor = allocate_f32_tensor(&data, &shape, 2, 6, 0);
        let tensor_ptr = Box::into_raw(Box::new(tensor)) as i64;

        let dict_ptr = crate::dict::nsl_dict_new();
        let key = alloc_c_string("weight") as i64;
        crate::dict::nsl_dict_set_str(dict_ptr, key, tensor_ptr);

        let name = alloc_c_string("weight") as *const c_char;
        let metadata = vec![ParamMeta {
            name_ptr: name,
            offset: 0,
            shape_ptr: std::ptr::null(),
            ndim: 0,
            transpose: 1, // request transpose
        }];

        let mut slot: i64 = 0;
        let model_ptr = &mut slot as *mut i64 as i64;
        load_weights_from_dict(model_ptr, &metadata, dict_ptr);

        assert_ne!(slot, 0, "transposed tensor pointer should be non-zero");
        let t = NslTensor::from_ptr(slot);
        // Shape should be [3, 2] after transpose
        let d0 = unsafe { *t.shape.add(0) };
        let d1 = unsafe { *t.shape.add(1) };
        assert_eq!(d0, 3, "transposed dim0 should be 3");
        assert_eq!(d1, 2, "transposed dim1 should be 2");
    }

    #[test]
    fn test_load_weights_hf_name_mapping() {
        // Dict uses HF-style name, meta uses NSL-style name
        let data = vec![1.0f32; 4];
        let shape = [2usize, 2];
        let tensor = allocate_f32_tensor(&data, &shape, 2, 4, 0);
        let tensor_ptr = Box::into_raw(Box::new(tensor)) as i64;

        let dict_ptr = crate::dict::nsl_dict_new();
        // HF-style key: "model.embed_tokens.weight"
        let hf_key = alloc_c_string("model.embed_tokens.weight") as i64;
        crate::dict::nsl_dict_set_str(dict_ptr, hf_key, tensor_ptr);

        // Meta uses NSL-style name: "embedding.weight"
        let nsl_name = alloc_c_string("embedding.weight") as *const c_char;
        let metadata = vec![ParamMeta {
            name_ptr: nsl_name,
            offset: 0,
            shape_ptr: std::ptr::null(),
            ndim: 0,
            transpose: 0,
        }];

        let mut slot: i64 = 0;
        let model_ptr = &mut slot as *mut i64 as i64;
        load_weights_from_dict(model_ptr, &metadata, dict_ptr);

        assert_ne!(slot, 0, "embedding.weight should be found via HF name mapping");
    }
}
