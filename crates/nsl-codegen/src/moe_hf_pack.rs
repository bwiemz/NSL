//! CPDT Part III v2.6 — Hugging Face Mixtral safetensors → NSL packed
//! MoE weight layout conversion.
//!
//! # Background
//!
//! Hugging Face Mixtral-style MoE safetensors store the per-expert
//! up / gate / down projections as N SEPARATE tensors under names of
//! the form `<prefix>.experts.{e}.w1.weight` / `.w2.weight` /
//! `.w3.weight`, with each tensor shaped `[out_features, in_features]`
//! (PyTorch's `nn.Linear.weight` convention — output dim first). The
//! HF-Mixtral expert convention is:
//!
//!   - `w1` = gate projection, `[intermediate, hidden]`
//!   - `w3` = up projection,   `[intermediate, hidden]`
//!   - `w2` = down projection, `[hidden, intermediate]`
//!
//! NSL's v4 `nsl_moe_dispatch_full_v4` FFI (CPDT Part III v2.5) instead
//! expects ONE packed tensor per projection, shaped
//! `[n_experts, in*out]` row-major, where the inner k-loop reads
//! `[in_dim, out_dim]` row-major (i.e., logical `W[in][out]`). That
//! convention is the per-expert matmul inner-loop layout — `token @ W`
//! reads `token[k] * W[k, j]` with `W[k, j]` at offset `k*out + j`.
//!
//! Converting HF → NSL packed therefore requires TWO operations per
//! per-expert tensor:
//!   1. **Transpose** `[out, in]` (HF) → `[in, out]` (NSL inner-loop).
//!   2. **Pack along expert axis** N transposed tensors → one
//!      `[n_experts, in*out]` tensor (concatenation of flattened
//!      `[in*out]` blocks).
//!
//! Both operations are PURE BYTE manipulation — no dtype conversion
//! happens. FP16 stays FP16, F32 stays F32, etc. Mixed-dtype experts
//! within a single MoE layer are rejected (silent dtype reinterpretation
//! across the packed tensor would corrupt the runtime read).
//!
//! # Scope (v2.6)
//!
//! This module ships the byte-level packing PRIMITIVE only. It does
//! NOT:
//!   - Auto-detect HF patterns at `WeightMap::load` time. (v2.next)
//!   - Add a CLI flag for opt-in invocation. (v2.next)
//!   - Add a source-level `@moe(weight_prefix="…")` kwarg that lets
//!     the v4 lowering look up packed entries under a non-NSL prefix.
//!     (v2.next, paired with auto-detection.)
//!
//! Today the primitive is reachable by direct library call only. The
//! v2.next auto-detect cycle wires it into the WeightMap construction
//! path so end users with a real Mixtral safetensors file can build
//! without manual repacking. Decoupling the byte transform from the
//! source-level naming question keeps v2.6 small enough to fit the
//! adversarial-review cadence and lets v2.7 land the wiring without
//! re-litigating the transform's correctness.
//!
//! # Refusal contract
//!
//! `pack_hf_mixtral_experts` returns `Err(PackError::…)` on any of:
//!   - `num_experts == 0`
//!   - missing any of `<hf_prefix>.experts.{e}.w{1,2,3}.weight` for
//!     `e ∈ 0..num_experts`
//!   - **extra** per-expert entries at index ≥ `num_experts` — silent
//!     half-pack hazard (caller said N experts but file holds N+M)
//!   - inconsistent per-expert shape (e.g., `e=0` is `[2, 3]` and
//!     `e=1` is `[2, 4]`) — silent corruption hazard at the
//!     packed-tensor stride
//!   - per-expert tensor not 2-D
//!   - per-expert tensor `data.len() != shape.iter().product() * dtype.byte_width()`
//!     — silent truncation hazard inside `transpose_2d_bytes`
//!   - mixed dtype across experts within a single projection — same
//!     silent-corruption class
//!   - the packed target name (e.g., `<target_prefix>.experts.gate.weight`)
//!     already exists in the WeightMap — refuses to silently overwrite
//!
//! ## Atomicity
//!
//! All refusals leave the WeightMap UNCHANGED (atomic semantics).
//! Internally this is achieved by a strict two-phase commit:
//!   - Phase 1 validates ALL three projections (Gate, Up, Down) and
//!     all `3 * num_experts` per-expert entries up-front, building a
//!     `Vec<ProjectionPlan>` of validated shape/dtype/source-keys
//!     descriptors. Any error returns early here, BEFORE any mutation.
//!   - Phase 2 is infallible: it transposes + packs the bytes,
//!     inserts the 3 packed entries, and removes the consumed HF
//!     sources. Phase 2 cannot fail because Phase 1 already proved
//!     every input is valid.
//!
//! Without atomicity, a Gate-succeeds-then-Up-fails sequence would
//! leave the packed Gate entry resident + the per-expert Gate sources
//! gone, blocking caller recovery (a retry after fixing the Up tensor
//! would trip `TargetAlreadyExists` on the orphan Gate). The two-phase
//! design closes that footgun for v2.7's auto-detect cycle.
//!
//! On success the per-expert source entries are REMOVED from the
//! WeightMap (the packed tensor is the canonical representation
//! downstream consumes), and three new entries are inserted under the
//! target prefix.

use crate::weight_aware::{WeightEntry, WeightMap};

/// The three projections an HF Mixtral expert block stores.
///
/// `repr` is load-bearing for the per-projection arrays in
/// `PackOutcome` and for the inner-loop dispatch.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfProjection {
    /// HF `w1` — gate projection. Stored `[intermediate, hidden]`,
    /// transposed to `[hidden, intermediate]` row-major in NSL packed.
    Gate = 0,
    /// HF `w3` — up projection. Same shape + transpose as gate.
    Up = 1,
    /// HF `w2` — down projection. Stored `[hidden, intermediate]`,
    /// transposed to `[intermediate, hidden]` row-major in NSL packed.
    Down = 2,
}

impl HfProjection {
    /// HF safetensors suffix (e.g. `w1` for `Gate`).
    pub fn hf_suffix(self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Up => "w3",
            Self::Down => "w2",
        }
    }

    /// NSL packed suffix under the target prefix.
    pub fn nsl_suffix(self) -> &'static str {
        match self {
            Self::Gate => "experts.gate.weight",
            Self::Up => "experts.up.weight",
            Self::Down => "experts.down.weight",
        }
    }

    pub const ALL: [HfProjection; 3] = [Self::Gate, Self::Up, Self::Down];
}

/// Outcome of a successful packing call. Useful for telemetry +
/// regression-pinning the names that disappeared / appeared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackOutcome {
    /// `num_experts` parameter the call was made with — pinned in the
    /// outcome so a future caller can verify it matches the source-level
    /// `@moe(num_experts=N)` declaration after the fact.
    pub num_experts: usize,
    /// NSL packed names that were inserted into the WeightMap (3 entries).
    pub inserted_names: Vec<String>,
    /// HF per-expert names that were removed from the WeightMap (3 * num_experts).
    pub removed_names: Vec<String>,
}

/// Failure modes for `pack_hf_mixtral_experts`. Distinct variants so a
/// caller can distinguish "wrong layout" (probably a configuration
/// error) from "silent-corruption hazard" (must refuse loudly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// `num_experts == 0` — meaningless input.
    ZeroNumExperts,
    /// Expected key `<prefix>.experts.{e}.{w*}.weight` not present in
    /// the WeightMap.
    MissingExpert {
        projection: HfProjection,
        expert_index: usize,
        expected_key: String,
    },
    /// A per-expert tensor was not 2-D — HF's `nn.Linear.weight` is
    /// always 2-D, so anything else means the user pointed at the
    /// wrong prefix or the file isn't an HF Mixtral export.
    NotTwoDimensional {
        projection: HfProjection,
        expert_index: usize,
        actual_ndim: usize,
        key: String,
    },
    /// Per-expert shapes within a single projection disagree.
    /// Refusing here prevents producing a packed tensor with a
    /// silently-wrong stride.
    ShapeMismatch {
        projection: HfProjection,
        expected: Vec<usize>,
        actual: Vec<usize>,
        offending_expert: usize,
    },
    /// Mixed dtype across experts within a single projection — would
    /// corrupt the packed tensor's interpretation.
    DtypeMismatch {
        projection: HfProjection,
        expected_dtype: String,
        actual_dtype: String,
        offending_expert: usize,
    },
    /// The packed target name already exists in the WeightMap. Refusing
    /// rather than overwriting prevents a silently-stale collision with
    /// pre-existing NSL packed entries.
    TargetAlreadyExists { name: String },
    /// `data.len()` of a per-expert entry does not equal
    /// `shape.iter().product() * dtype.byte_width()`. Without this guard,
    /// `transpose_2d_bytes` would silently truncate oversize input in
    /// release builds (the `debug_assert!` is elided) or panic with a
    /// generic slice-bounds message on undersize input. Surfacing the
    /// mismatch here keeps the loud-refusal convention symmetric across
    /// every malformed-input class.
    DataLengthMismatch {
        projection: HfProjection,
        expert_index: usize,
        key: String,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    /// More HF source entries exist at indices ≥ `num_experts` than the
    /// caller declared. Refusing rather than silently dropping the extras
    /// catches a misconfigured `@moe(num_experts=N)` against an N+M
    /// checkpoint. The orphaned entries would otherwise leak memory and
    /// confuse downstream weight-aware passes.
    ExtraExpertsPresent {
        hf_prefix: String,
        declared_num_experts: usize,
        found_extra_index: usize,
        offending_key: String,
    },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroNumExperts => write!(f, "pack_hf_mixtral_experts: num_experts == 0"),
            Self::MissingExpert { projection, expert_index, expected_key } => write!(
                f,
                "pack_hf_mixtral_experts: missing {} (w{}) for expert {}: expected key '{}'",
                projection.nsl_suffix(),
                match projection {
                    HfProjection::Gate => 1,
                    HfProjection::Up => 3,
                    HfProjection::Down => 2,
                },
                expert_index,
                expected_key,
            ),
            Self::NotTwoDimensional { projection, expert_index, actual_ndim, key } => write!(
                f,
                "pack_hf_mixtral_experts: {} expert {} ({}) is {}-D, expected 2-D",
                projection.nsl_suffix(),
                expert_index,
                key,
                actual_ndim,
            ),
            Self::ShapeMismatch { projection, expected, actual, offending_expert } => write!(
                f,
                "pack_hf_mixtral_experts: {} shape mismatch — expert 0 is {:?}, expert {} is {:?}",
                projection.nsl_suffix(),
                expected,
                offending_expert,
                actual,
            ),
            Self::DtypeMismatch { projection, expected_dtype, actual_dtype, offending_expert } => {
                write!(
                    f,
                    "pack_hf_mixtral_experts: {} dtype mismatch — expert 0 is {}, expert {} is {}",
                    projection.nsl_suffix(),
                    expected_dtype,
                    offending_expert,
                    actual_dtype,
                )
            }
            Self::TargetAlreadyExists { name } => write!(
                f,
                "pack_hf_mixtral_experts: target name '{}' already exists in the WeightMap — refusing to overwrite",
                name,
            ),
            Self::DataLengthMismatch {
                projection,
                expert_index,
                key,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "pack_hf_mixtral_experts: {} expert {} ({}) has data.len() = {}, expected {} bytes for declared shape + dtype",
                projection.nsl_suffix(),
                expert_index,
                key,
                actual_bytes,
                expected_bytes,
            ),
            Self::ExtraExpertsPresent {
                hf_prefix,
                declared_num_experts,
                found_extra_index,
                offending_key,
            } => write!(
                f,
                "pack_hf_mixtral_experts: caller declared {} experts under prefix '{}' but the WeightMap contains an entry at expert index {} ({}) — refuse rather than silently leave orphans",
                declared_num_experts,
                hf_prefix,
                found_extra_index,
                offending_key,
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// Transpose a 2-D byte buffer with logical shape `[rows, cols]` into
/// logical shape `[cols, rows]`, both row-major, with element width
/// `element_bytes`. Operates purely on bytes — no dtype awareness, no
/// numeric conversion. The output `Vec<u8>` length equals the input
/// length.
///
/// This is the inner loop of HF → NSL transpose. Kept separate so the
/// transpose is testable independently of the multi-expert orchestration.
pub fn transpose_2d_bytes(
    input: &[u8],
    rows: usize,
    cols: usize,
    element_bytes: usize,
) -> Vec<u8> {
    debug_assert_eq!(
        input.len(),
        rows * cols * element_bytes,
        "transpose_2d_bytes: input length must equal rows * cols * element_bytes"
    );
    let mut out = vec![0u8; rows * cols * element_bytes];
    for r in 0..rows {
        for c in 0..cols {
            let in_off = (r * cols + c) * element_bytes;
            let out_off = (c * rows + r) * element_bytes;
            out[out_off..out_off + element_bytes]
                .copy_from_slice(&input[in_off..in_off + element_bytes]);
        }
    }
    out
}

/// Pack HF Mixtral per-expert MoE weights into NSL's packed convention.
///
/// See the module-level docstring for the layout contract. On success
/// the WeightMap is mutated: the 3 * num_experts per-expert HF entries
/// are removed, and 3 new packed entries are inserted under
/// `<target_prefix>`.
///
/// `hf_prefix` and `target_prefix` may be the same string (the common
/// case when the HF prefix is also the desired moe_configs key).
/// Per-projection plan built by Phase 1 (validation) and consumed by
/// Phase 2 (mutation). Carrying it as an intermediate value is what
/// makes the two-phase commit possible — Phase 2 reads only validated
/// data and cannot fail.
struct ProjectionPlan {
    projection: HfProjection,
    out_dim: usize,
    in_dim: usize,
    dtype: crate::weight_aware::WeightDType,
    element_bytes: usize,
    /// `num_experts` long: the HF source key for each expert. Stored
    /// so Phase 2 doesn't re-derive them via `format!` (the formatted
    /// string IS the lookup key, so reusing the Phase 1 strings keeps
    /// the two phases self-consistent under any future renaming).
    source_keys: Vec<String>,
}

/// Phase 1: validate ALL three projections + all `3 * num_experts` per-expert
/// entries up-front. Returns a vector of plans (Gate, Up, Down order) on
/// success. On error the WeightMap is unmodified — Phase 1 only reads.
fn validate_all_projections(
    weight_map: &WeightMap,
    hf_prefix: &str,
    num_experts: usize,
) -> Result<Vec<ProjectionPlan>, PackError> {
    let mut plans: Vec<ProjectionPlan> = Vec::with_capacity(3);

    for proj in HfProjection::ALL {
        let mut expected_shape: Option<Vec<usize>> = None;
        let mut expected_dtype: Option<crate::weight_aware::WeightDType> = None;
        let mut source_keys: Vec<String> = Vec::with_capacity(num_experts);

        for e in 0..num_experts {
            let key = format!(
                "{}.experts.{}.{}.weight",
                hf_prefix,
                e,
                proj.hf_suffix(),
            );
            let Some(entry) = weight_map.get(&key) else {
                return Err(PackError::MissingExpert {
                    projection: proj,
                    expert_index: e,
                    expected_key: key,
                });
            };
            if entry.shape.len() != 2 {
                return Err(PackError::NotTwoDimensional {
                    projection: proj,
                    expert_index: e,
                    actual_ndim: entry.shape.len(),
                    key,
                });
            }
            match (&expected_shape, &expected_dtype) {
                (None, None) => {
                    expected_shape = Some(entry.shape.clone());
                    expected_dtype = Some(entry.dtype);
                }
                (Some(sh), Some(dt)) => {
                    if entry.shape != *sh {
                        return Err(PackError::ShapeMismatch {
                            projection: proj,
                            expected: sh.clone(),
                            actual: entry.shape.clone(),
                            offending_expert: e,
                        });
                    }
                    if entry.dtype != *dt {
                        return Err(PackError::DtypeMismatch {
                            projection: proj,
                            expected_dtype: format!("{:?}", dt),
                            actual_dtype: format!("{:?}", entry.dtype),
                            offending_expert: e,
                        });
                    }
                }
                _ => unreachable!("shape and dtype are always set together"),
            }
            // Validate the data buffer length matches the declared
            // shape + dtype before Phase 2's transpose_2d_bytes (which
            // would otherwise silently truncate oversize input in
            // release builds).
            let expected_bytes =
                entry.shape.iter().product::<usize>() * entry.dtype.byte_width();
            if entry.data.len() != expected_bytes {
                return Err(PackError::DataLengthMismatch {
                    projection: proj,
                    expert_index: e,
                    key,
                    expected_bytes,
                    actual_bytes: entry.data.len(),
                });
            }
            source_keys.push(key);
        }

        // Refuse if MORE per-expert entries exist than declared. A
        // user mistyping `@moe(num_experts=4)` against an 8-expert
        // checkpoint would otherwise silently get a 4-expert build
        // with the other 4 leaking as orphans. We scan only the
        // immediate next index (num_experts) because finding even one
        // extra is enough to refuse — the user's input is misconfigured.
        let extra_key = format!(
            "{}.experts.{}.{}.weight",
            hf_prefix,
            num_experts,
            proj.hf_suffix(),
        );
        if weight_map.get(&extra_key).is_some() {
            return Err(PackError::ExtraExpertsPresent {
                hf_prefix: hf_prefix.to_string(),
                declared_num_experts: num_experts,
                found_extra_index: num_experts,
                offending_key: extra_key,
            });
        }

        let shape = expected_shape.expect("first-expert pass sets expected_shape");
        let dtype = expected_dtype.expect("first-expert pass sets expected_dtype");
        plans.push(ProjectionPlan {
            projection: proj,
            out_dim: shape[0], // HF stores [out, in]
            in_dim: shape[1],
            dtype,
            element_bytes: dtype.byte_width(),
            source_keys,
        });
    }

    Ok(plans)
}

pub fn pack_hf_mixtral_experts(
    weight_map: &mut WeightMap,
    hf_prefix: &str,
    target_prefix: &str,
    num_experts: usize,
) -> Result<PackOutcome, PackError> {
    if num_experts == 0 {
        return Err(PackError::ZeroNumExperts);
    }

    // Check for target collisions across ALL three projections FIRST.
    // We pre-check all three (not just Gate) so a regression in the
    // Up/Down branches surfaces in tests, and so the early-exit doesn't
    // depend on alphabetic projection order.
    for proj in HfProjection::ALL {
        let target_name = format!("{}.{}", target_prefix, proj.nsl_suffix());
        if weight_map.get(&target_name).is_some() {
            return Err(PackError::TargetAlreadyExists { name: target_name });
        }
    }

    // PHASE 1 — validate every projection + every expert up-front.
    // Any error returns BEFORE the WeightMap is touched, so refusals
    // are atomic.
    let plans = validate_all_projections(weight_map, hf_prefix, num_experts)?;

    // PHASE 2 — infallible. Build all 3 packed buffers locally first,
    // THEN apply the inserts + removes in one batch. Splitting
    // build-from-mutate ensures even an unexpected panic during a
    // later projection's transpose can't leave a half-mutated map.
    let mut staged_inserts: Vec<WeightEntry> = Vec::with_capacity(3);
    let mut inserted_names: Vec<String> = Vec::with_capacity(3);
    let mut removed_names: Vec<String> = Vec::with_capacity(3 * num_experts);

    for plan in &plans {
        let elements_per_expert = plan.out_dim * plan.in_dim;
        let bytes_per_expert = elements_per_expert * plan.element_bytes;
        let mut packed = vec![0u8; num_experts * bytes_per_expert];
        for (e, key) in plan.source_keys.iter().enumerate() {
            let entry = weight_map
                .get(key)
                .expect("Phase 1 validated presence and the WeightMap has not been mutated since");
            let transposed =
                transpose_2d_bytes(&entry.data, plan.out_dim, plan.in_dim, plan.element_bytes);
            let dst_off = e * bytes_per_expert;
            packed[dst_off..dst_off + bytes_per_expert].copy_from_slice(&transposed);
        }
        let packed_name = format!("{}.{}", target_prefix, plan.projection.nsl_suffix());
        let packed_shape = vec![num_experts, plan.in_dim * plan.out_dim];
        staged_inserts.push(WeightEntry::new(
            packed_name.clone(),
            packed,
            packed_shape,
            plan.dtype,
        ));
        inserted_names.push(packed_name);
    }

    // Apply: inserts + removes. Both directions are infallible — the
    // staged inserts are valid WeightEntries, the source keys are
    // known-present from Phase 1.
    for entry in staged_inserts {
        weight_map.insert(entry);
    }
    for plan in &plans {
        for key in &plan.source_keys {
            weight_map.remove(key);
            removed_names.push(key.clone());
        }
    }

    Ok(PackOutcome {
        num_experts,
        inserted_names,
        removed_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};

    fn make_f32_entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let num_elements = shape.iter().product::<usize>();
        assert_eq!(vals.len(), num_elements, "shape vs vals mismatch");
        let mut data = Vec::with_capacity(num_elements * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape,
            dtype: WeightDType::F32,
            num_elements,
            sparsity: None,
            eliminated: false,
        }
    }

    fn read_f32(entry: &WeightEntry) -> Vec<f32> {
        entry
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    // ── transpose_2d_bytes ────────────────────────────────────────────

    #[test]
    fn transpose_2d_bytes_swaps_2x3_to_3x2_element_by_element() {
        // Input row-major [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] (rows=2, cols=3)
        // Expected output row-major [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        // (rows=3, cols=2 → rows=cols_in, cols=rows_in).
        let vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut bytes = Vec::with_capacity(24);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = transpose_2d_bytes(&bytes, 2, 3, 4);
        assert_eq!(out.len(), 24);
        let out_vals: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out_vals, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_2d_bytes_square_2x2_is_diagonal_invariant() {
        // [[a, b], [c, d]] → [[a, c], [b, d]] — diagonal elements
        // (a, d) preserved at same offsets.
        let vals = [11.0_f32, 22.0, 33.0, 44.0];
        let mut bytes = Vec::with_capacity(16);
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = transpose_2d_bytes(&bytes, 2, 2, 4);
        let out_vals: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out_vals, vec![11.0, 33.0, 22.0, 44.0]);
    }

    #[test]
    fn transpose_2d_bytes_f16_width_works_at_2_bytes() {
        // Use raw u16 patterns to verify the byte-level transpose
        // doesn't depend on f32 alignment.
        let bytes: Vec<u8> = vec![
            0x01, 0x02, // (0, 0)
            0x03, 0x04, // (0, 1)
            0x05, 0x06, // (1, 0)
            0x07, 0x08, // (1, 1)
        ];
        let out = transpose_2d_bytes(&bytes, 2, 2, 2);
        // After transpose: (0,0) (1,0) (0,1) (1,1)
        assert_eq!(out, vec![0x01, 0x02, 0x05, 0x06, 0x03, 0x04, 0x07, 0x08]);
    }

    // ── pack_hf_mixtral_experts: happy paths ──────────────────────────

    #[test]
    fn pack_hf_mixtral_experts_two_experts_packs_correctly() {
        // 2 experts, hidden=2, intermediate=3.
        // HF layout: w1=[3,2], w3=[3,2], w2=[2,3].
        // NSL packed: gate/up=[2, 2*3]=[2, 6], down=[2, 3*2]=[2, 6].
        let mut wm = WeightMap::new_for_test();
        // Expert 0
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![3, 2],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 3],
            &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ));
        // Expert 1
        wm.insert(make_f32_entry(
            "moe0.experts.1.w1.weight",
            vec![3, 2],
            &[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w3.weight",
            vec![3, 2],
            &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w2.weight",
            vec![2, 3],
            &[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ));

        let outcome =
            pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).expect("pack ok");

        assert_eq!(outcome.num_experts, 2);
        assert_eq!(outcome.inserted_names.len(), 3);
        assert_eq!(outcome.removed_names.len(), 6);

        let gate = wm.get("moe0.experts.gate.weight").expect("gate present");
        assert_eq!(gate.shape, vec![2, 6]);
        // Expert 0 gate: HF [3, 2] = [[1,2],[3,4],[5,6]] →
        // transpose [2, 3] = [[1,3,5],[2,4,6]] → flat [1,3,5,2,4,6].
        // Expert 1 gate: HF [3, 2] = [[19,20],[21,22],[23,24]] →
        // transpose [2, 3] = [[19,21,23],[20,22,24]] →
        // flat [19,21,23,20,22,24].
        let gate_vals = read_f32(gate);
        assert_eq!(
            gate_vals,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 19.0, 21.0, 23.0, 20.0, 22.0, 24.0]
        );

        let up = wm.get("moe0.experts.up.weight").expect("up present");
        let up_vals = read_f32(up);
        assert_eq!(
            up_vals,
            vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0, 25.0, 27.0, 29.0, 26.0, 28.0, 30.0]
        );

        let down = wm.get("moe0.experts.down.weight").expect("down present");
        // Expert 0 down: HF [2, 3] = [[13,14,15],[16,17,18]] →
        // transpose [3, 2] = [[13,16],[14,17],[15,18]] → flat
        // [13,16,14,17,15,18].
        let down_vals = read_f32(down);
        assert_eq!(
            down_vals,
            vec![13.0, 16.0, 14.0, 17.0, 15.0, 18.0, 31.0, 34.0, 32.0, 35.0, 33.0, 36.0]
        );

        // Per-expert sources removed.
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                let key = format!("moe0.experts.{}.{}.weight", e, proj);
                assert!(wm.get(&key).is_none(), "{} should be removed", key);
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_supports_distinct_hf_and_target_prefixes() {
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry(
            "hf.experts.0.w1.weight",
            vec![2, 2],
            &[1.0, 2.0, 3.0, 4.0],
        ));
        wm.insert(make_f32_entry(
            "hf.experts.0.w3.weight",
            vec![2, 2],
            &[5.0, 6.0, 7.0, 8.0],
        ));
        wm.insert(make_f32_entry(
            "hf.experts.0.w2.weight",
            vec![2, 2],
            &[9.0, 10.0, 11.0, 12.0],
        ));

        let outcome =
            pack_hf_mixtral_experts(&mut wm, "hf", "nsl_target", 1).expect("pack ok");

        // Packed under nsl_target, not hf.
        assert!(wm.get("nsl_target.experts.gate.weight").is_some());
        assert!(wm.get("hf.experts.gate.weight").is_none());
        // HF sources removed.
        assert!(wm.get("hf.experts.0.w1.weight").is_none());
        assert_eq!(outcome.inserted_names[0], "nsl_target.experts.gate.weight");
    }

    // ── refusal paths ────────────────────────────────────────────────

    #[test]
    fn pack_hf_mixtral_experts_refuses_zero_num_experts() {
        let mut wm = WeightMap::new_for_test();
        assert_eq!(
            pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 0),
            Err(PackError::ZeroNumExperts)
        );
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_missing_expert() {
        let mut wm = WeightMap::new_for_test();
        // Only expert 0 present; expert 1 missing.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::MissingExpert { expert_index, projection, .. } => {
                assert_eq!(expert_index, 1);
                assert_eq!(projection, HfProjection::Gate);
            }
            other => panic!("expected MissingExpert, got {:?}", other),
        }
        // WeightMap is unmodified on error.
        assert!(wm.get("moe0.experts.0.w1.weight").is_some());
        assert!(wm.get("moe0.experts.gate.weight").is_none());
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_shape_mismatch() {
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![3, 2],
            &[1.0; 6],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![3, 2],
            &[1.0; 6],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 3],
            &[1.0; 6],
        ));
        // Expert 1 has a DIFFERENT w1 shape — silent-corruption hazard.
        wm.insert(make_f32_entry(
            "moe0.experts.1.w1.weight",
            vec![3, 4],
            &[1.0; 12],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w3.weight",
            vec![3, 2],
            &[1.0; 6],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w2.weight",
            vec![2, 3],
            &[1.0; 6],
        ));
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ShapeMismatch { projection, expected, actual, offending_expert } => {
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(expected, vec![3, 2]);
                assert_eq!(actual, vec![3, 4]);
                assert_eq!(offending_expert, 1);
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_not_two_dimensional() {
        let mut wm = WeightMap::new_for_test();
        // 1-D tensor under what should be a 2-D weight key.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![4],
            &[1.0, 2.0, 3.0, 4.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 1).unwrap_err();
        match err {
            PackError::NotTwoDimensional { actual_ndim, projection, expert_index, .. } => {
                assert_eq!(actual_ndim, 1);
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(expert_index, 0);
            }
            other => panic!("expected NotTwoDimensional, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_target_already_exists() {
        let mut wm = WeightMap::new_for_test();
        // Pre-existing packed entry under the target.
        wm.insert(make_f32_entry(
            "moe0.experts.gate.weight",
            vec![1, 4],
            &[99.0; 4],
        ));
        // Plus a valid set of HF sources.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(
            matches!(err, PackError::TargetAlreadyExists { ref name } if name == "moe0.experts.gate.weight"),
            "expected TargetAlreadyExists, got {:?}",
            err,
        );
        // WeightMap is unmodified — pre-existing packed entry still
        // present, HF sources still present (not removed).
        assert!(wm.get("moe0.experts.0.w1.weight").is_some());
        let pre = wm.get("moe0.experts.gate.weight").unwrap();
        assert_eq!(read_f32(pre), vec![99.0, 99.0, 99.0, 99.0]);
    }

    // ── v2.6 adversarial-review fixes — additional refusal coverage ────

    #[test]
    fn pack_hf_mixtral_experts_refuses_dtype_mismatch() {
        // v2.6 adversarial-review fix F5: the DtypeMismatch refusal
        // branch exists in code (the highest-stakes silent-corruption
        // hazard of the enumerated set) but had no unit test. A
        // refactor that consolidated the per-projection inequality
        // checks could drop the dtype guard and ship without CI
        // catching it. Pin it explicitly.
        let mut wm = WeightMap::new_for_test();
        // Expert 0: F32 throughout.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![2, 2],
            &[1.0, 2.0, 3.0, 4.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[5.0, 6.0, 7.0, 8.0],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[9.0, 10.0, 11.0, 12.0],
        ));
        // Expert 1: F16 w1 (dtype mismatch on Gate); other tensors F32.
        let f16_bytes = vec![0u8; 2 * 2 * 2]; // 4 f16 elements = 8 bytes
        wm.insert(WeightEntry::new(
            "moe0.experts.1.w1.weight".to_string(),
            f16_bytes,
            vec![2, 2],
            WeightDType::F16,
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));

        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::DtypeMismatch {
                projection,
                offending_expert,
                ..
            } => {
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(offending_expert, 1);
            }
            other => panic!("expected DtypeMismatch, got {:?}", other),
        }
        // WeightMap is unmodified — no packed entries appeared.
        assert!(wm.get("moe0.experts.gate.weight").is_none());
        assert!(wm.get("moe0.experts.up.weight").is_none());
        assert!(wm.get("moe0.experts.down.weight").is_none());
        // All 6 HF source entries still present.
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                assert!(
                    wm.get(&format!("moe0.experts.{}.{}.weight", e, proj)).is_some(),
                    "moe0.experts.{}.{}.weight should still be present after DtypeMismatch refusal",
                    e,
                    proj,
                );
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_extra_experts_present() {
        // v2.6 adversarial-review fix F4: a caller passing
        // num_experts=2 against a 3-expert HF checkpoint would
        // otherwise silently pack 2 + leave expert 2's HF sources
        // orphaned in the WeightMap. Refuse loudly.
        let mut wm = WeightMap::new_for_test();
        for e in 0..3 {
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.weight", e),
                vec![2, 2],
                &[1.0; 4],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.weight", e),
                vec![2, 2],
                &[1.0; 4],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w2.weight", e),
                vec![2, 2],
                &[1.0; 4],
            ));
        }
        // Caller declares 2 experts but expert 2 is present.
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ExtraExpertsPresent {
                declared_num_experts,
                found_extra_index,
                ..
            } => {
                assert_eq!(declared_num_experts, 2);
                assert_eq!(found_extra_index, 2);
            }
            other => panic!("expected ExtraExpertsPresent, got {:?}", other),
        }
        // WeightMap is unmodified — all 9 HF source entries still present,
        // no packed entries.
        assert!(wm.get("moe0.experts.gate.weight").is_none());
        assert!(wm.get("moe0.experts.2.w1.weight").is_some());
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_data_length_mismatch() {
        // v2.6 adversarial-review fix F2: a malformed WeightEntry
        // (data.len() != shape.product() * dtype.bytes) would either
        // silently truncate inside transpose_2d_bytes (oversize, release
        // build) or panic with a generic slice-bounds error (undersize).
        // Surface the mismatch loudly at the load boundary instead.
        let mut wm = WeightMap::new_for_test();
        // shape [2, 2] declares 16 bytes of F32 data; provide only 4.
        wm.insert(WeightEntry::new(
            "moe0.experts.0.w1.weight".to_string(),
            vec![0u8; 4], // truncated buffer
            vec![2, 2],
            WeightDType::F32,
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 1).unwrap_err();
        match err {
            PackError::DataLengthMismatch {
                projection,
                expert_index,
                expected_bytes,
                actual_bytes,
                ..
            } => {
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(expert_index, 0);
                assert_eq!(expected_bytes, 16);
                assert_eq!(actual_bytes, 4);
            }
            other => panic!("expected DataLengthMismatch, got {:?}", other),
        }
        // WeightMap is unmodified.
        assert!(wm.get("moe0.experts.gate.weight").is_none());
        assert!(wm.get("moe0.experts.0.w1.weight").is_some());
    }

    #[test]
    fn pack_hf_mixtral_experts_atomicity_holds_when_up_projection_fails() {
        // v2.6 adversarial-review fix F1/F3/F6 (the atomicity bug):
        // before the two-phase commit refactor, Gate's insert + source
        // removes happened BEFORE Up's validation ran. If Up failed,
        // the WeightMap was left half-packed (gate.weight present, w1
        // sources gone) — blocking caller recovery because a retry
        // would trip TargetAlreadyExists on the orphan Gate.
        //
        // This test constructs a fixture where w1 (Gate) AND w2 (Down)
        // are consistent across experts but w3 (Up) is mismatched, then
        // asserts that the refusal leaves EVERY part of the WeightMap
        // unmodified: no packed entries appear, every HF source
        // remains.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.weight", e),
                vec![2, 2],
                &[(e as f32) * 0.1; 4],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w2.weight", e),
                vec![2, 2],
                &[(e as f32) * 0.2; 4],
            ));
        }
        // Expert 0 w3 = [2,2], expert 1 w3 = [3,3] — shape mismatch on Up.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[0.5; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.1.w3.weight",
            vec![3, 3],
            &[0.5; 9],
        ));

        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ShapeMismatch { projection, offending_expert, .. } => {
                assert_eq!(projection, HfProjection::Up);
                assert_eq!(offending_expert, 1);
            }
            other => panic!("expected ShapeMismatch on Up, got {:?}", other),
        }
        // Atomicity: NO packed entries appeared.
        assert!(wm.get("moe0.experts.gate.weight").is_none(),
            "atomicity violation: gate packed entry leaked despite Up failing");
        assert!(wm.get("moe0.experts.up.weight").is_none());
        assert!(wm.get("moe0.experts.down.weight").is_none());
        // Atomicity: every HF source entry still present.
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                assert!(
                    wm.get(&format!("moe0.experts.{}.{}.weight", e, proj)).is_some(),
                    "atomicity violation: moe0.experts.{}.{}.weight was removed before Up failure",
                    e,
                    proj,
                );
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_atomicity_holds_when_down_projection_fails() {
        // Mirror of the above test, but with the mismatch on w2 (Down,
        // the THIRD projection). Both Gate and Up validate successfully;
        // Down fails. Verifies the two-phase commit applies symmetrically
        // — a refactor that re-introduced per-projection mutation would
        // leave gate + up packed and w1 + w3 sources gone.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.weight", e),
                vec![2, 2],
                &[1.0; 4],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.weight", e),
                vec![2, 2],
                &[1.0; 4],
            ));
        }
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        // Expert 1 w2 is 1-D (not 2-D) — triggers NotTwoDimensional on Down.
        wm.insert(make_f32_entry(
            "moe0.experts.1.w2.weight",
            vec![4],
            &[1.0; 4],
        ));

        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::NotTwoDimensional { projection, expert_index, actual_ndim, .. } => {
                assert_eq!(projection, HfProjection::Down);
                assert_eq!(expert_index, 1);
                assert_eq!(actual_ndim, 1);
            }
            other => panic!("expected NotTwoDimensional on Down, got {:?}", other),
        }
        // Atomicity assertions: NO packed entries, every HF source still present.
        for nsl in ["gate", "up", "down"] {
            assert!(
                wm.get(&format!("moe0.experts.{}.weight", nsl)).is_none(),
                "atomicity violation: {} packed entry leaked despite Down failing",
                nsl,
            );
        }
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                assert!(
                    wm.get(&format!("moe0.experts.{}.{}.weight", e, proj)).is_some(),
                    "atomicity violation: moe0.experts.{}.{}.weight was removed before Down failure",
                    e,
                    proj,
                );
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_target_collision_on_up_not_just_gate() {
        // The original target-collision test pre-inserted on the
        // "experts.gate.weight" key, which is the FIRST projection
        // checked. If the pre-check loop ever regressed to short-
        // circuit only on Gate, an Up-only or Down-only collision
        // would still need to be caught. Pin it explicitly.
        let mut wm = WeightMap::new_for_test();
        // Pre-existing collision on the Up target only.
        wm.insert(make_f32_entry(
            "moe0.experts.up.weight",
            vec![1, 4],
            &[99.0; 4],
        ));
        // Valid HF sources for 1 expert.
        wm.insert(make_f32_entry(
            "moe0.experts.0.w1.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w3.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 2],
            &[1.0; 4],
        ));

        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(
            matches!(err, PackError::TargetAlreadyExists { ref name } if name == "moe0.experts.up.weight"),
            "expected TargetAlreadyExists on up.weight, got {:?}",
            err,
        );
    }
}
