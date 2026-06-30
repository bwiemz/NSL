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

    /// CPDT Part III v2.15: HF safetensors bias suffix (e.g. `w1.bias`
    /// for `Gate`). Symmetric to `hf_suffix()`. The canonical
    /// `nn.Linear` convention is `<module>.weight` + `<module>.bias`, so
    /// each HF MoE expert's per-projection bias key (when biases are
    /// enabled at training time) is the matching `w{1,2,3}.bias`
    /// suffix.
    pub fn hf_bias_suffix(self) -> &'static str {
        match self {
            Self::Gate => "w1.bias",
            Self::Up => "w3.bias",
            Self::Down => "w2.bias",
        }
    }

    /// CPDT Part III v2.16: alternative short-form HF bias suffix
    /// (e.g. `b1` for `Gate`). Some early Mixtral exports — notably
    /// pre-1.0 community converters that flattened the `nn.Linear`
    /// to a custom `Linear` struct with a `bias: Parameter` field of
    /// rank 1 — emitted bias keys as `<prefix>.experts.{e}.b{1,2,3}`
    /// without the `.bias` suffix. v2.16 accepts BOTH forms (the
    /// canonical `.w{N}.bias` and the short `.b{N}`) so users
    /// importing converted Mixtral checkpoints don't have to rename
    /// keys before loading.
    ///
    /// Within a single block, the bias auto-pack pins ONE form per
    /// projection per block. Mixing forms (e.g. expert 0 uses `.w1.bias`
    /// while expert 1 uses `.b1`) is REFUSED as `MixedBiasSuffixForms`
    /// because the typical cause is a partially-converted checkpoint.
    pub fn hf_bias_suffix_short(self) -> &'static str {
        match self {
            Self::Gate => "b1",
            Self::Up => "b3",
            Self::Down => "b2",
        }
    }

    /// CPDT Part III v2.16: returns BOTH bias-suffix variants
    /// (canonical `.w{N}.bias` first, short `.b{N}` second) for
    /// caller-side iteration. Order matters: the canonical form is
    /// probed first so a checkpoint with both present resolves to
    /// `.w{N}.bias` (the FormCanonical variant of `BiasSuffixForm`).
    pub fn hf_bias_suffixes_all(self) -> [&'static str; 2] {
        [self.hf_bias_suffix(), self.hf_bias_suffix_short()]
    }

    /// CPDT Part III v2.15: NSL packed bias suffix under the target
    /// prefix. Matches the names `detect_v4_biases` (and the v3
    /// detector) resolve.
    pub fn nsl_bias_suffix(self) -> &'static str {
        match self {
            Self::Gate => "experts.gate.bias",
            Self::Up => "experts.up.bias",
            Self::Down => "experts.down.bias",
        }
    }

    pub const ALL: [HfProjection; 3] = [Self::Gate, Self::Up, Self::Down];
}

/// CPDT Part III v2.16: which of the two HF bias suffix conventions a
/// per-expert bias key uses. Pinned per (projection, expert) at
/// detection time; the per-block uniformity invariant is enforced by
/// `validate_all_bias_projections`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasSuffixForm {
    /// Canonical `.w{N}.bias` (matches `nn.Linear.bias` convention).
    Canonical,
    /// Short `.b{N}` (matches early Mixtral converter convention).
    Short,
}

impl BiasSuffixForm {
    pub fn label(self) -> &'static str {
        match self {
            Self::Canonical => "canonical (.w{N}.bias)",
            Self::Short => "short (.b{N})",
        }
    }
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
    /// CPDT Part III v2.15: HF per-expert bias keys exist for SOME
    /// (projection, expert) slots but not all `3 * num_experts`. Bias
    /// auto-pack is all-or-nothing per block — mixing partial bias
    /// presence with the v4 lowering would silently drop the missing
    /// directions at runtime. The `present` and `missing` vectors
    /// enumerate the offending slots so the user can repair their
    /// checkpoint.
    PartialBiasBundle {
        hf_prefix: String,
        present: Vec<String>,
        missing: Vec<String>,
    },
    /// CPDT Part III v2.15: per-expert bias keys exist at indices
    /// ≥ `num_experts`. Symmetric to `ExtraExpertsPresent` but for
    /// bias keys. The weight-pack's extra-experts scan would have
    /// caught a matching orphan `w*.weight`, but a checkpoint shipping
    /// extra biases WITHOUT extra weights (rare but possible — a
    /// half-pruned checkpoint) needs its own loud refusal.
    ///
    /// v2.16-C (closes v2.15 review F15 LOW deferral): the variant
    /// now carries `found_extra_indices: Vec<usize>` (sorted +
    /// deduplicated) instead of a single `found_extra_index`. Users
    /// recovering from a half-pruned checkpoint can fix every orphan
    /// in one pass instead of rebuilding once per orphan.
    ExtraBiasesPresent {
        hf_prefix: String,
        declared_num_experts: usize,
        found_extra_indices: Vec<usize>,
        offending_key: String,
    },
    /// CPDT Part III v2.15 review fix F6 (IMPORTANT adversarial review):
    /// a per-expert bias declares `num_elements == 0` (e.g., a `[0]`
    /// shape or a `[N, 0]` packed shape). Phase 2 would silently
    /// allocate an empty buffer and write a `[num_experts, 0]` packed
    /// entry, which `detect_v4_biases` would later reject with a
    /// cryptic "0 elements; expected N * D" diagnostic. Refusing at
    /// pack time gives the user a clear actionable error pointing at
    /// the bias source.
    ZeroBiasDim {
        hf_prefix: String,
        projection: HfProjection,
        offending_key: String,
    },
    /// CPDT Part III v2.15 review fix F7 (IMPORTANT adversarial review):
    /// the three projections within a single block have INCONSISTENT
    /// dtypes (e.g., gate F32, up F32, down F16). Within-projection
    /// dtype consistency is already enforced by `DtypeMismatch`, but
    /// the v4 runtime FFI (`nsl_moe_dispatch_full_v4`) requires
    /// uniform dtype across all three projections AND tokens —
    /// mixed dtypes would silently fail at the FFI's dtype-equality
    /// guard, returning 0 with no actionable diagnostic. Refusing at
    /// pack time is far more actionable.
    MixedProjectionDtypes {
        hf_prefix: String,
        gate_dtype: String,
        up_dtype: String,
        down_dtype: String,
    },
    /// CPDT Part III v2.15 review fix F4 (IMPORTANT adversarial review):
    /// `bias_dim * element_bytes` or `num_experts * bytes_per_expert`
    /// overflows `usize`. Realistic checkpoints never trigger this
    /// (would require num_experts in the billions), but an adversarial
    /// safetensors with synthetic huge shapes would otherwise wrap to
    /// a small allocation in release builds and trip a generic
    /// copy_from_slice OOB panic. Surfacing the overflow as a typed
    /// error matches the loud-refusal convention.
    BiasByteOverflow {
        hf_prefix: String,
        projection: HfProjection,
        bias_dim: usize,
        num_experts: usize,
        element_bytes: usize,
    },
    /// CPDT Part III v2.16: a single block mixes the canonical
    /// `.w{N}.bias` suffix with the short `.b{N}` form across
    /// (projection, expert) slots. The typical cause is a
    /// partially-converted checkpoint where some keys were renamed
    /// and others weren't. The pack refuses rather than silently
    /// picking one form per slot because the inconsistency usually
    /// indicates a stale conversion script.
    MixedBiasSuffixForms {
        hf_prefix: String,
        canonical_keys: Vec<String>,
        short_keys: Vec<String>,
    },
    /// CPDT Part III v2.16 (closes v2.15 review F2 HIGH deferral):
    /// the orchestrator detected per-expert bias keys (canonical or
    /// short form) under a prefix where NO per-expert weight keys
    /// exist. A v4 dispatch needs both weights and biases; the user's
    /// checkpoint is malformed for the v4 lowering. Surface the
    /// cause at load time with an actionable diagnostic instead of
    /// letting `derive_v4_dims` fail later with a less obvious
    /// "router missing" diagnostic.
    BiasesWithoutWeights {
        hf_prefix: String,
        bias_keys: Vec<String>,
    },
    /// CPDT Part III v2.16 review fix F1 (IMPORTANT adversarial review):
    /// a single (projection, expert) slot has BOTH the canonical
    /// `.w{N}.bias` AND the short `.b{N}` form present. The pre-v2.16
    /// behavior was to silently prefer canonical and leave the short
    /// form as an orphan. The adversarial review surfaced this 3 times
    /// (same-slot duplicates with mismatched shapes/dtypes would be
    /// silently undetected). Refuse loudly: the typical cause is a
    /// half-completed conversion script that wrote new keys without
    /// removing the originals, and a clean refusal points the user
    /// at the cause.
    SameSlotBothBiasSuffixForms {
        hf_prefix: String,
        projection: HfProjection,
        expert_index: usize,
        canonical_key: String,
        short_key: String,
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
            Self::PartialBiasBundle { hf_prefix, present, missing } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' has partial HF bias bundle. present={:?}, missing={:?}. v4 bias auto-pack is all-or-nothing per block — either add the missing per-expert bias keys (one of `.w{{1,2,3}}.bias` per expert) or remove the present ones before loading.",
                hf_prefix, present, missing,
            ),
            Self::ExtraBiasesPresent {
                hf_prefix,
                declared_num_experts,
                found_extra_indices,
                offending_key,
            } => write!(
                f,
                "pack_hf_mixtral_biases: caller declared {} experts under prefix '{}' but the WeightMap contains orphan bias entries at expert indices {:?}. Representative key (smallest orphan index, canonical form): {}. Remove ALL orphans before rebuilding — refuse rather than silently leave them.",
                declared_num_experts,
                hf_prefix,
                found_extra_indices,
                offending_key,
            ),
            Self::ZeroBiasDim { hf_prefix, projection, offending_key } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' projection {} bias ({}) has 0 elements. Zero-length biases are nonsensical and would later trip a confusing diagnostic at `detect_v4_biases`. Either remove the empty bias or fix the source checkpoint.",
                hf_prefix,
                projection.nsl_bias_suffix(),
                offending_key,
            ),
            Self::MixedProjectionDtypes {
                hf_prefix,
                gate_dtype,
                up_dtype,
                down_dtype,
            } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' has mismatched bias dtypes across projections (gate={}, up={}, down={}). The v4 FFI requires uniform dtype across all three projections (matching `tokens.dtype`); mixed dtypes would silently fail at the runtime dtype-equality gate. Re-export the checkpoint with consistent bias dtypes.",
                hf_prefix, gate_dtype, up_dtype, down_dtype,
            ),
            Self::BiasByteOverflow {
                hf_prefix,
                projection,
                bias_dim,
                num_experts,
                element_bytes,
            } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' projection {} byte-count derivation overflows usize (bias_dim={}, num_experts={}, element_bytes={}). The checkpoint shape is pathologically large — re-export with realistic dims.",
                hf_prefix,
                projection.nsl_bias_suffix(),
                bias_dim,
                num_experts,
                element_bytes,
            ),
            Self::MixedBiasSuffixForms { hf_prefix, canonical_keys, short_keys } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' mixes the canonical `.w{{N}}.bias` suffix and the short `.b{{N}}` suffix across (projection, expert) slots. canonical_keys={:?}, short_keys={:?}. Pick ONE convention for the whole block — the typical cause is a partially-converted checkpoint where a rename script ran on some keys but not others.",
                hf_prefix, canonical_keys, short_keys,
            ),
            Self::BiasesWithoutWeights { hf_prefix, bias_keys } => write!(
                f,
                "HF Mixtral auto-pack: prefix '{}' has per-expert bias keys but NO per-expert weight keys ({:?} found, no `.experts.{{e}}.w{{1,2,3}}.weight` companions). A v4 dispatch needs both weights and biases — the checkpoint is malformed for v4 lowering. Either add the missing weight keys or remove the orphan biases.",
                hf_prefix, bias_keys,
            ),
            Self::SameSlotBothBiasSuffixForms {
                hf_prefix,
                projection,
                expert_index,
                canonical_key,
                short_key,
            } => write!(
                f,
                "pack_hf_mixtral_biases: prefix '{}' has BOTH the canonical `{}` and the short `{}` at the same (projection={}, expert={}) slot. The typical cause is a conversion script that wrote new keys without removing the originals. Pick ONE form for every slot — leaving duplicates would silently orphan the form that wasn't picked, hiding shape/dtype mismatches between the two.",
                hf_prefix,
                canonical_key,
                short_key,
                projection.nsl_bias_suffix(),
                expert_index,
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

    // Unified extra-experts scan. The v2.6 implementation probed only
    // the immediate next index (`num_experts`) per projection, which
    // silently MISSED gapped sets like {0, 1, 3} (expert 2 missing
    // entirely, expert 3 fully present). Scanning the full WeightMap
    // for any `<hf_prefix>.experts.<N>.w[123].weight` where
    // N >= num_experts catches every leak — pinned by the v2.7
    // adversarial-review F1 fix.
    let prefix_search = format!("{}.experts.", hf_prefix);
    let mut max_extra: Option<usize> = None;
    for name in weight_map.names() {
        let Some(rest) = name.strip_prefix(&prefix_search) else {
            continue;
        };
        let Some((idx_str, tail)) = rest.split_once('.') else {
            continue;
        };
        if !matches!(tail, "w1.weight" | "w2.weight" | "w3.weight") {
            continue;
        }
        let Ok(idx) = idx_str.parse::<usize>() else {
            continue;
        };
        if idx >= num_experts {
            max_extra = Some(max_extra.map_or(idx, |m| m.max(idx)));
        }
    }
    if let Some(idx) = max_extra {
        // Report the first w-suffix's key. Picking deterministically
        // matters for stable error messages.
        let offending_key = format!("{}{}.w1.weight", prefix_search, idx);
        return Err(PackError::ExtraExpertsPresent {
            hf_prefix: hf_prefix.to_string(),
            declared_num_experts: num_experts,
            found_extra_index: idx,
            offending_key,
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

// ─────────────────────────────────────────────────────────────────────────
// CPDT Part III v2.15 — HF bias auto-pack
// ─────────────────────────────────────────────────────────────────────────
//
// Symmetric to the v2.6 weight pack: detect + transform per-expert HF
// bias keys (`<prefix>.experts.{e}.w{1,2,3}.bias`) into NSL packed
// convention (`<prefix>.experts.{gate,up,down}.bias`) so the v4 bias
// detector (`detect_v4_biases` in `moe.rs`) resolves them.
//
// Layout transform is pure concatenation along a new expert axis:
//   - Per-expert HF bias is 1-D `[D]` (matching the `nn.Linear.bias`
//     convention: D = out_features of the parent weight).
//   - NSL packed bias is 2-D `[num_experts, D]`, row-major.
//   - NO transpose — biases are 1-D, so the byte ordering is just
//     `expert0_bias || expert1_bias || …` concatenated.
//
// The `D` per projection follows the parent weight's `out_dim`:
//   - Gate (w1.bias):  D = intermediate_dim
//   - Up   (w3.bias):  D = intermediate_dim
//   - Down (w2.bias):  D = hidden_dim
//
// Refusal contract (all-or-nothing per block):
//   1. NO bias keys for any (projection, expert) slot → `Ok(None)` (no-op)
//   2. ALL `3 * num_experts` bias keys present + consistent → pack
//   3. Anything else → `Err(PartialBiasBundle | …)` — loud refusal.
//      The v4 lowering's `detect_v4_biases` is itself all-or-nothing
//      (gate + up + down all present, or none); auto-pack must honor
//      the same contract or risk silently dropping biases at runtime.
//
// Scope: `.w{1,2,3}.bias` only (canonical PyTorch nn.Linear convention).
// The dot-prefixed `.b{1,2,3}` form some early Mixtral exports used is
// deferred to a future cycle. v3 (2-projection) bias auto-pack is not
// reachable today because the weight detector requires all 3
// `.w{1,2,3}.weight` keys (only the v4 path is auto-packable).

/// Per-projection bias plan built by Phase 1 and consumed by Phase 2.
/// Mirrors `ProjectionPlan` but for 1-D biases — no `in_dim`, no
/// transpose. `bias_dim` is the parent weight's `out_dim` (intermediate
/// for gate/up, hidden for down).
struct BiasProjectionPlan {
    projection: HfProjection,
    bias_dim: usize,
    dtype: crate::weight_aware::WeightDType,
    element_bytes: usize,
    /// `num_experts` long: the HF source key for each expert. Stored
    /// from Phase 1 so Phase 2 doesn't re-derive via `format!`.
    source_keys: Vec<String>,
}

/// CPDT Part III v2.16: resolve a per-expert bias key by probing
/// BOTH suffix forms (canonical `.w{N}.bias` and short `.b{N}`).
///
/// v2.16 review fix F1 (IMPORTANT): probes BOTH forms in one pass
/// and REFUSES if both are present at the same slot — the original
/// "canonical-wins, short-orphan-stays" behavior silently masked
/// shape/dtype mismatches between the duplicates and left
/// inconsistent state in the WeightMap. The adversarial review
/// surfaced this same hazard from 3 distinct angles, so the fix
/// closes it loudly.
///
/// Returns:
///   - `Ok(None)` — neither form present at this slot
///   - `Ok(Some(canonical_key, Canonical))` — only canonical present
///   - `Ok(Some(short_key, Short))` — only short present
///   - `Err(SameSlotBothBiasSuffixForms)` — both present (refuse)
fn resolve_bias_key(
    weight_map: &WeightMap,
    hf_prefix: &str,
    expert_index: usize,
    projection: HfProjection,
) -> Result<Option<(String, BiasSuffixForm)>, PackError> {
    let canonical = format!(
        "{}.experts.{}.{}",
        hf_prefix,
        expert_index,
        projection.hf_bias_suffix(),
    );
    let short = format!(
        "{}.experts.{}.{}",
        hf_prefix,
        expert_index,
        projection.hf_bias_suffix_short(),
    );
    let canonical_present = weight_map.get(&canonical).is_some();
    let short_present = weight_map.get(&short).is_some();
    match (canonical_present, short_present) {
        (false, false) => Ok(None),
        (true, false) => Ok(Some((canonical, BiasSuffixForm::Canonical))),
        (false, true) => Ok(Some((short, BiasSuffixForm::Short))),
        (true, true) => Err(PackError::SameSlotBothBiasSuffixForms {
            hf_prefix: hf_prefix.to_string(),
            projection,
            expert_index,
            canonical_key: canonical,
            short_key: short,
        }),
    }
}

/// Validate every per-expert bias key for the three projections.
/// Returns `Ok(None)` when NO bias keys exist at all (the clean no-bias
/// case). Returns `Ok(Some(plans))` when ALL `3 * num_experts` keys
/// exist + are consistent. Returns `Err(...)` for any partial,
/// shape-mismatched, dtype-mismatched, data-length-mismatched, or
/// orphan-extra case.
fn validate_all_bias_projections(
    weight_map: &WeightMap,
    hf_prefix: &str,
    num_experts: usize,
) -> Result<Option<Vec<BiasProjectionPlan>>, PackError> {
    // First pass: probe BOTH suffix forms per (projection, expert)
    // slot. v2.16 accepts canonical `.w{N}.bias` and short `.b{N}`,
    // but the per-block uniformity invariant requires that EVERY
    // resolved slot use the SAME form — mixing is refused.
    let mut present_keys: Vec<String> = Vec::new();
    let mut missing_keys: Vec<String> = Vec::new();
    let mut canonical_keys: Vec<String> = Vec::new();
    let mut short_keys: Vec<String> = Vec::new();
    for proj in HfProjection::ALL {
        for e in 0..num_experts {
            // resolve_bias_key may return Err for same-slot duplicates
            // (v2.16 review fix F1); propagate as an early refusal so
            // the diagnostic points exactly at the offending slot.
            match resolve_bias_key(weight_map, hf_prefix, e, proj)? {
                Some((key, BiasSuffixForm::Canonical)) => {
                    canonical_keys.push(key.clone());
                    present_keys.push(key);
                }
                Some((key, BiasSuffixForm::Short)) => {
                    short_keys.push(key.clone());
                    present_keys.push(key);
                }
                None => {
                    // Report the canonical form as the "missing" key
                    // because that's the convention we recommend.
                    missing_keys.push(format!(
                        "{}.experts.{}.{}",
                        hf_prefix,
                        e,
                        proj.hf_bias_suffix(),
                    ));
                }
            }
        }
    }

    if present_keys.is_empty() {
        // Clean no-bias case — no-op for this block. The weight pack
        // already happened (or is about to); the user's compilation
        // proceeds with the v2.5/v2.7 no-bias 5-arg path.
        return Ok(None);
    }
    if !missing_keys.is_empty() {
        // Partial bundle. Refuse loudly with both lists so the user
        // can pinpoint what's missing.
        return Err(PackError::PartialBiasBundle {
            hf_prefix: hf_prefix.to_string(),
            present: present_keys,
            missing: missing_keys,
        });
    }
    // v2.16 invariant: per-block suffix uniformity. Mixed forms
    // almost always indicate a partially-converted checkpoint.
    if !canonical_keys.is_empty() && !short_keys.is_empty() {
        return Err(PackError::MixedBiasSuffixForms {
            hf_prefix: hf_prefix.to_string(),
            canonical_keys,
            short_keys,
        });
    }

    // All `3 * num_experts` bias keys present. Validate per-projection
    // shape + dtype + data-length consistency, mirroring the weight
    // pack's per-projection pass. v2.16: per-slot key resolution
    // goes through `resolve_bias_key` so the same canonical-vs-short
    // probe runs here as in the all-keys-present scan above.
    let mut plans: Vec<BiasProjectionPlan> = Vec::with_capacity(3);
    for proj in HfProjection::ALL {
        let mut expected_num_elements: Option<usize> = None;
        let mut expected_dtype: Option<crate::weight_aware::WeightDType> = None;
        let mut source_keys: Vec<String> = Vec::with_capacity(num_experts);

        for e in 0..num_experts {
            // v2.16 review fix F1: the early-pass above has already
            // validated that no slot trips SameSlotBothBiasSuffixForms,
            // so `?` here is safe — re-running resolve_bias_key would
            // only re-discover known-good state.
            let (key, _form) = resolve_bias_key(weight_map, hf_prefix, e, proj)?
                .expect("all-keys-present scan above validated existence");
            // Phase 1 above proved presence; double-checking here would
            // mask a future regression in the all-or-nothing scan.
            let entry = weight_map
                .get(&key)
                .expect("resolve_bias_key returned a key that was just looked up");
            // v2.15 review fix F5 (IMPORTANT): defensive invariant
            // assertion that `num_elements` matches `shape.product()`.
            // The WeightEntry constructor computes num_elements from
            // shape, so this is true by construction — but a future
            // refactor (e.g., adding a builder that lets callers
            // override num_elements without re-deriving from shape)
            // would silently break Phase 2's slice copy. Catching the
            // drift here in debug builds prevents the silent
            // truncation hazard.
            debug_assert_eq!(
                entry.num_elements,
                entry.shape.iter().product::<usize>(),
                "WeightEntry invariant: num_elements must equal shape.product() for key {}",
                key,
            );
            // A bias entry is conventionally 1-D `[D]`, but a checkpoint
            // shipped as `[1, D]` or `[D, 1]` has the same flat element
            // count and is also accepted at the FFI layer. Match the
            // detector's tolerance (`num_elements` only) here so a
            // user's auto-packed bias keys resolve identically to a
            // hand-packed bundle.
            match (expected_num_elements, expected_dtype) {
                (None, None) => {
                    expected_num_elements = Some(entry.num_elements);
                    expected_dtype = Some(entry.dtype);
                }
                (Some(n), Some(dt)) => {
                    if entry.num_elements != n {
                        return Err(PackError::ShapeMismatch {
                            projection: proj,
                            expected: vec![n],
                            actual: vec![entry.num_elements],
                            offending_expert: e,
                        });
                    }
                    if entry.dtype != dt {
                        return Err(PackError::DtypeMismatch {
                            projection: proj,
                            expected_dtype: format!("{:?}", dt),
                            actual_dtype: format!("{:?}", entry.dtype),
                            offending_expert: e,
                        });
                    }
                }
                _ => unreachable!("num_elements and dtype always set together"),
            }
            let expected_bytes = entry.num_elements * entry.dtype.byte_width();
            if entry.data.len() != expected_bytes {
                return Err(PackError::DataLengthMismatch {
                    projection: proj,
                    expert_index: e,
                    key: key.clone(),
                    expected_bytes,
                    actual_bytes: entry.data.len(),
                });
            }
            source_keys.push(key);
        }

        let dim = expected_num_elements.expect("first-expert pass sets expected_num_elements");
        let dt = expected_dtype.expect("first-expert pass sets expected_dtype");
        // v2.15 review fix F6 (IMPORTANT): refuse zero-element biases
        // up-front. The pack would otherwise produce a `[num_experts,
        // 0]` packed entry that `detect_v4_biases` later rejects with
        // a less actionable diagnostic — surface the cause at its
        // origin instead.
        if dim == 0 {
            return Err(PackError::ZeroBiasDim {
                hf_prefix: hf_prefix.to_string(),
                projection: proj,
                offending_key: source_keys[0].clone(),
            });
        }
        // v2.15 review fix F4 (IMPORTANT): refuse on overflow in the
        // per-expert byte count derivation. Phase 2 multiplies
        // `bias_dim * element_bytes` then `num_experts * ...` to
        // allocate the packed buffer; wrapping multiplication in
        // release builds would silently undersize the Vec and panic
        // (or, with very large num_experts, allocate a tiny buffer
        // and copy_from_slice would OOB on the first expert).
        let bytes_per_expert = dim.checked_mul(dt.byte_width()).ok_or(
            PackError::BiasByteOverflow {
                hf_prefix: hf_prefix.to_string(),
                projection: proj,
                bias_dim: dim,
                num_experts,
                element_bytes: dt.byte_width(),
            },
        )?;
        num_experts.checked_mul(bytes_per_expert).ok_or(
            PackError::BiasByteOverflow {
                hf_prefix: hf_prefix.to_string(),
                projection: proj,
                bias_dim: dim,
                num_experts,
                element_bytes: dt.byte_width(),
            },
        )?;
        plans.push(BiasProjectionPlan {
            projection: proj,
            bias_dim: dim,
            dtype: dt,
            element_bytes: dt.byte_width(),
            source_keys,
        });
    }

    // v2.15 review fix F7 (IMPORTANT): cross-projection dtype
    // consistency check. Within-projection consistency is enforced
    // above (DtypeMismatch); cross-projection consistency must also
    // hold because `nsl_moe_dispatch_full_v4` requires uniform dtype
    // across all three bias arrays AND tokens. Refusing at pack time
    // produces an actionable diagnostic instead of the FFI's silent
    // `return 0` at the dtype-equality gate.
    if !plans.is_empty() {
        let gate_dt = plans[0].dtype;
        let up_dt = plans[1].dtype;
        let down_dt = plans[2].dtype;
        if gate_dt != up_dt || up_dt != down_dt {
            return Err(PackError::MixedProjectionDtypes {
                hf_prefix: hf_prefix.to_string(),
                gate_dtype: format!("{:?}", gate_dt),
                up_dtype: format!("{:?}", up_dt),
                down_dtype: format!("{:?}", down_dt),
            });
        }
    }

    // Extra-bias scan symmetric to the weight pack's extra-experts
    // scan. A checkpoint with bias keys at index >= num_experts (a
    // half-pruned bundle the weight scan didn't catch because the
    // matching `.w*.weight` was deleted but `.w*.bias` wasn't) MUST be
    // refused — those orphans would otherwise leak and confuse
    // downstream passes that walk the WeightMap looking for keys
    // under `<prefix>.experts.*`.
    //
    // v2.16-A: the scan also recognizes the short `.b{N}` form
    // alongside the canonical `.w{N}.bias` form.
    // v2.16-C: the scan accumulates ALL orphan indices (sorted +
    // deduplicated) instead of reporting only the maximum. Users
    // recovering from a half-pruned checkpoint can fix every orphan
    // in one pass instead of rebuilding repeatedly.
    let prefix_search = format!("{}.experts.", hf_prefix);
    let mut extra_indices: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();
    for name in weight_map.names() {
        let Some(rest) = name.strip_prefix(&prefix_search) else {
            continue;
        };
        let Some((idx_str, tail)) = rest.split_once('.') else {
            continue;
        };
        if !matches!(
            tail,
            "w1.bias" | "w2.bias" | "w3.bias" | "b1" | "b2" | "b3",
        ) {
            continue;
        }
        let Ok(idx) = idx_str.parse::<usize>() else {
            continue;
        };
        if idx >= num_experts {
            extra_indices.insert(idx);
        }
    }
    if !extra_indices.is_empty() {
        let found_extra_indices: Vec<usize> = extra_indices.into_iter().collect();
        let first_idx = found_extra_indices[0];
        // Report the canonical form for the FIRST orphan as the
        // representative `offending_key`. The full list is in
        // `found_extra_indices` for downstream consumers.
        let offending_key = format!("{}{}.w1.bias", prefix_search, first_idx);
        return Err(PackError::ExtraBiasesPresent {
            hf_prefix: hf_prefix.to_string(),
            declared_num_experts: num_experts,
            found_extra_indices,
            offending_key,
        });
    }

    Ok(Some(plans))
}

/// Pack HF Mixtral per-expert MoE biases into NSL's packed convention.
///
/// See the section docstring above for the layout contract. On success
/// the WeightMap is mutated: the `3 * num_experts` per-expert HF bias
/// entries are removed, and 3 new packed bias entries are inserted
/// under `<target_prefix>`.
///
/// Returns:
///   - `Ok(None)` when the WeightMap holds no per-expert bias keys for
///     any (projection, expert) slot under `hf_prefix`. The WeightMap
///     is left untouched — this is the clean "no biases on this block"
///     no-op path.
///   - `Ok(Some(outcome))` when all `3 * num_experts` HF bias keys
///     existed and got packed. The outcome enumerates the inserted +
///     removed names for telemetry.
///   - `Err(PackError::…)` for any refusal. Refusals are atomic
///     (two-phase commit pattern, same as `pack_hf_mixtral_experts`):
///     Phase 1 (`validate_all_bias_projections`) only reads, Phase 2
///     stages all packed buffers locally then applies inserts + removes
///     in one batch.
///
/// # Scope: what this function does and does NOT validate
///
/// v2.15 review fix F8 (IMPORTANT): the pack's job is to TRANSFORM HF
/// key names into NSL key names without numerics validation. It does
/// validate:
///   - Per-projection shape + dtype consistency across experts
///   - All-or-nothing per (projection, expert) slot presence
///   - Cross-projection dtype consistency (gate/up/down all same dtype)
///   - Non-zero bias dim
///   - Byte-count overflow safety (`checked_mul` guards)
///   - Per-projection target name collisions
///   - Orphan bias keys at expert index >= num_experts
///
/// It does NOT validate:
///   - Per-bias element count vs the parent WEIGHT's `out_dim` (gate
///     bias should have `intermediate_dim` elements, down bias should
///     have `hidden_dim` elements). That validation happens at
///     codegen via [`crate::moe::detect_v4_biases`], which has access
///     to the dims derived from `derive_v4_dims`. A user who packs a
///     bundle with bias_dim mismatched against weight_out_dim will
///     get an error at the v4 lowering, not at pack time.
///
/// This split mirrors the v2.6 weight pack's "pack is permissive,
/// lowering is strict" principle — duplicating the dim derivation
/// here would invite drift between the two validators.
///
/// # Scope: what the AUTO-DETECTOR does NOT find
///
/// v2.15 review fix F2 (HIGH): the orchestrator
/// `pack_all_detected_hf_mixtral_blocks` discovers blocks via the
/// weight-driven `detect_hf_mixtral_blocks` scan (which looks for
/// `.w{1,2,3}.weight` keys). A checkpoint with HF BIASES but no HF
/// WEIGHTS under the same prefix is NOT detected as a block, so
/// `pack_hf_mixtral_biases` never runs and the biases stay orphaned.
/// This is acceptable today because such a checkpoint is malformed
/// for v4 dispatch (no weights → `derive_v4_dims` fails first), but
/// a future cycle could extend the detector to also scan for bias
/// keys for completeness.
pub fn pack_hf_mixtral_biases(
    weight_map: &mut WeightMap,
    hf_prefix: &str,
    target_prefix: &str,
    num_experts: usize,
) -> Result<Option<PackOutcome>, PackError> {
    if num_experts == 0 {
        return Err(PackError::ZeroNumExperts);
    }

    // Pre-check target name collisions on all 3 bias slots BEFORE the
    // validation pass, mirroring the weight pack's pre-check. Same
    // rationale: a regression that consolidated the collision checks
    // would otherwise surface in surprising places.
    for proj in HfProjection::ALL {
        let target_name = format!("{}.{}", target_prefix, proj.nsl_bias_suffix());
        if weight_map.get(&target_name).is_some() {
            return Err(PackError::TargetAlreadyExists { name: target_name });
        }
    }

    // PHASE 1 — validate. Returns Ok(None) if no biases anywhere, which
    // we pass through directly: nothing to pack.
    let Some(plans) = validate_all_bias_projections(weight_map, hf_prefix, num_experts)?
    else {
        return Ok(None);
    };

    // PHASE 2 — infallible. Build packed buffers locally, then apply
    // inserts + removes atomically.
    let mut staged_inserts: Vec<WeightEntry> = Vec::with_capacity(3);
    let mut inserted_names: Vec<String> = Vec::with_capacity(3);
    let mut removed_names: Vec<String> = Vec::with_capacity(3 * num_experts);

    for plan in &plans {
        let bytes_per_expert = plan.bias_dim * plan.element_bytes;
        let mut packed = vec![0u8; num_experts * bytes_per_expert];
        for (e, key) in plan.source_keys.iter().enumerate() {
            let entry = weight_map
                .get(key)
                .expect("Phase 1 validated presence; WeightMap unchanged since");
            // v2.15 review fix F14 (LOW): Phase 1's DataLengthMismatch
            // check enforces `entry.data.len() == entry.num_elements *
            // byte_width()` EXACTLY for every shape — 1-D `[D]`,
            // 2-D `[1, D]`, 2-D `[D, 1]` — so by the time this code
            // runs, `entry.data.len()` equals `bytes_per_expert`
            // (because num_elements is invariant to layout and
            // bias_dim was set from the same num_elements). The
            // `[..bytes_per_expert]` slice is therefore a no-op range
            // check that protects against future regressions in the
            // Phase 1 contract. No padding semantics exist.
            let dst_off = e * bytes_per_expert;
            packed[dst_off..dst_off + bytes_per_expert]
                .copy_from_slice(&entry.data[..bytes_per_expert]);
        }
        let packed_name = format!("{}.{}", target_prefix, plan.projection.nsl_bias_suffix());
        let packed_shape = vec![num_experts, plan.bias_dim];
        staged_inserts.push(WeightEntry::new(
            packed_name.clone(),
            packed,
            packed_shape,
            plan.dtype,
        ));
        inserted_names.push(packed_name);
    }

    for entry in staged_inserts {
        weight_map.insert(entry);
    }
    for plan in &plans {
        for key in &plan.source_keys {
            weight_map.remove(key);
            removed_names.push(key.clone());
        }
    }

    Ok(Some(PackOutcome {
        num_experts,
        inserted_names,
        removed_names,
    }))
}

// ─────────────────────────────────────────────────────────────────────────
// CPDT Part III v2.7 — auto-detection
// ─────────────────────────────────────────────────────────────────────────

/// One detected HF Mixtral MoE block in a WeightMap. Produced by
/// [`detect_hf_mixtral_blocks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedHfMixtralBlock {
    /// HF prefix, e.g. `model.layers.0.block_sparse_moe`.
    pub hf_prefix: String,
    /// Highest expert index + 1. The detector requires the contiguous
    /// run `experts.{0..num_experts}.w{1,2,3}.weight` to all be
    /// present; gaps cause the block to be rejected from the
    /// detected set (the pack call would refuse with MissingExpert
    /// anyway).
    pub num_experts: usize,
}

/// Scan a `WeightMap` for HF Mixtral MoE blocks. A block is detected
/// when every key in the contiguous run `<prefix>.experts.{0..N}.w{1,
/// 2,3}.weight` is present for some `N ≥ 1` AND no `<prefix>.experts.N.
/// w*.weight` exists (i.e., the run is exact, not truncated to a
/// caller's `num_experts`). The detector does NOT validate per-expert
/// shapes / dtypes — that's the packing primitive's job. A detected
/// block is just "a contiguous run of HF-shaped names that the user
/// probably wants packed."
///
/// Detection is read-only. The returned `DetectedHfMixtralBlock`s are
/// sorted by `hf_prefix` for deterministic iteration order under
/// downstream consumers.
///
/// Gotchas pinned by the v2.7 adversarial review:
/// - If a prefix has w1/w2/w3 for experts {0, 1, 3} (i.e., expert 2
///   missing entirely, expert 3 fully present), the detector reports
///   `num_experts=2` (the largest contiguous run starting at 0). The
///   caller's pack invocation then trips `ExtraExpertsPresent` because
///   `validate_all_projections`'s unified scan (post-v2.7-F1 fix)
///   finds the orphan at index 3 even though it isn't the immediate
///   next index. That's the right behavior — the user fixes their
///   safetensors, not their NSL source.
/// - Already-packed entries (`<prefix>.experts.gate.weight` etc.) are
///   ignored by detection (the parser requires a NUMERIC expert index;
///   `gate`/`up`/`down` fail the `parse::<usize>()` arm).
/// - The detector requires a NON-EMPTY prefix segment before
///   `.experts.` (real HF Mixtral safetensors always include a
///   `model.layers.X.block_sparse_moe` style prefix; bare
///   `experts.0.w1.weight` keys are silently skipped). Documented per
///   the v2.7 adversarial-review F3 fix.
/// - The detector tolerates ANY (non-empty) name segments before the
///   trailing `.experts.{N}.w{1,2,3}.weight` suffix —
///   `model.layers.0.block_sparse_moe`, `transformer.h.0.moe`,
///   `experts_layer_0` all detect identically.
pub fn detect_hf_mixtral_blocks(weight_map: &WeightMap) -> Vec<DetectedHfMixtralBlock> {
    use std::collections::BTreeMap;
    // For each candidate prefix, tracks max(expert_index_seen) and
    // whether ALL three w-suffixes were observed for every index in
    // 0..=max_index. We accumulate into a per-(prefix, expert_index)
    // bitmask (bit 0 = w1 seen, bit 1 = w2 seen, bit 2 = w3 seen) and
    // post-process to find the largest contiguous run.
    let mut per_prefix: BTreeMap<String, BTreeMap<usize, u8>> = BTreeMap::new();

    for name in weight_map.names() {
        // Look for the `.experts.<N>.w<1|2|3>.weight` suffix; the prefix
        // is everything to the left of `.experts.`.
        let Some((prefix, rest)) = name.rsplit_once(".experts.") else {
            continue;
        };
        // `rest` must look like `<digits>.w[123].weight`.
        let Some((idx_str, tail)) = rest.split_once('.') else {
            continue;
        };
        let Ok(expert_index) = idx_str.parse::<usize>() else {
            // Not a numeric expert index — already-packed entries
            // (`experts.gate.weight`, `experts.up.weight`,
            // `experts.down.weight`) take this branch.
            continue;
        };
        let suffix_bit = match tail {
            "w1.weight" => 1u8 << 0,
            "w2.weight" => 1u8 << 1,
            "w3.weight" => 1u8 << 2,
            _ => continue, // Bias tensor or other HF artifact.
        };
        let by_prefix = per_prefix.entry(prefix.to_string()).or_default();
        let entry_bits = by_prefix.entry(expert_index).or_insert(0);
        *entry_bits |= suffix_bit;
    }

    let mut detected: Vec<DetectedHfMixtralBlock> = Vec::new();
    for (prefix, by_index) in &per_prefix {
        // Largest N such that experts 0..N are all complete (all 3
        // w-suffixes seen). Stops at the first missing-or-incomplete
        // index. If expert 0 itself is incomplete, the block is not
        // detected.
        const ALL_THREE: u8 = 0b111;
        let mut n_complete = 0usize;
        loop {
            match by_index.get(&n_complete) {
                Some(&bits) if bits == ALL_THREE => {
                    n_complete += 1;
                }
                _ => break,
            }
        }
        if n_complete >= 1 {
            detected.push(DetectedHfMixtralBlock {
                hf_prefix: prefix.clone(),
                num_experts: n_complete,
            });
        }
    }
    detected
}

/// Outcome of a `pack_all_detected_hf_mixtral_blocks` call: which
/// blocks were detected, which packs succeeded, and which failed (with
/// the per-block `PackError`).
#[derive(Debug, Clone)]
pub struct AutoPackOutcome {
    /// Successfully packed blocks (in detection order). Each entry's
    /// `hf_prefix` equals the value used as both `hf_prefix` and
    /// `target_prefix` in the underlying `pack_hf_mixtral_experts`
    /// call — the auto-pack always rewrites in place.
    pub packed: Vec<(DetectedHfMixtralBlock, PackOutcome)>,
    /// Blocks the detector found but the packer refused. The auto-pack
    /// path does NOT short-circuit on the first error — it tries every
    /// detected block and reports all failures, so a user with one bad
    /// MoE block (e.g., dtype mismatch on layer 5) still gets the
    /// other layers packed.
    pub failed: Vec<(DetectedHfMixtralBlock, PackError)>,
    /// CPDT Part III v2.15: blocks whose weight pack succeeded AND
    /// whose per-expert HF bias keys (`.w{1,2,3}.bias`) also packed
    /// successfully into the NSL `.experts.{gate,up,down}.bias`
    /// convention. Blocks with no bias keys do NOT appear here (the
    /// bias pack returned `Ok(None)` — no-op). The wire-up at
    /// `entry_points.rs` logs this list separately from `packed` so
    /// users can see bias auto-pack firing in stderr.
    pub bias_packed: Vec<(DetectedHfMixtralBlock, PackOutcome)>,
    /// CPDT Part III v2.15: blocks whose weight pack succeeded but
    /// whose bias pack refused (e.g., partial-bias bundle). Surfaced
    /// alongside `failed` by the entry-points wrapper as a hard build
    /// error — the v4 lowering would otherwise silently drop biases.
    /// Note: a bias-pack failure does NOT roll back the successful
    /// weight pack for that block; per-pass atomicity is preserved
    /// (each pass is independently atomic), but cross-pass atomicity
    /// is not (matching v2.7's cross-block independence).
    pub bias_failed: Vec<(DetectedHfMixtralBlock, PackError)>,
}

/// Scan `weight_map` for HF Mixtral blocks (via
/// [`detect_hf_mixtral_blocks`]) and run `pack_hf_mixtral_experts` on
/// each. Returns an [`AutoPackOutcome`] enumerating successes + per-
/// block failures.
///
/// Each pack call is independent — a failure on one block does NOT
/// roll back another block's packed entries. Per-block atomicity is
/// already provided by `pack_hf_mixtral_experts`'s two-phase commit
/// (v2.6); cross-block atomicity is intentionally NOT provided because
/// a partial result is still useful (you can build a model that uses
/// the successful layers and inspect the failures).
/// CPDT Part III v2.16-B: scan for per-expert HF bias keys at
/// prefixes that have NO matching per-expert weight keys. Returns a
/// sorted list of `(hf_prefix, bias_keys)` pairs for orphan prefixes.
/// `detected_weight_prefixes` is the set of prefixes the weight
/// detector already found — those are EXCLUDED from the orphan scan
/// since their bias keys will be picked up by the per-block bias
/// pack downstream.
///
/// Recognizes BOTH bias suffix forms: canonical `.w{1,2,3}.bias` and
/// short `.b{1,2,3}` (v2.16-A).
fn find_bias_only_prefixes(
    weight_map: &WeightMap,
    detected_weight_prefixes: &std::collections::HashSet<String>,
) -> Vec<(String, Vec<String>)> {
    use std::collections::BTreeMap;
    let mut per_prefix: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for name in weight_map.names() {
        // Bias key shape: `<prefix>.experts.<idx>.<bias-suffix>`. The
        // `<prefix>` is everything to the left of `.experts.`. We
        // match canonical `.w{1,2,3}.bias` AND short `.b{1,2,3}`.
        let Some((prefix, rest)) = name.rsplit_once(".experts.") else {
            continue;
        };
        let Some((idx_str, tail)) = rest.split_once('.') else {
            continue;
        };
        // Skip non-numeric expert indices (e.g., already-packed
        // `experts.gate.bias`).
        if idx_str.parse::<usize>().is_err() {
            continue;
        }
        let is_bias = matches!(
            tail,
            "w1.bias" | "w2.bias" | "w3.bias" | "b1" | "b2" | "b3",
        );
        if !is_bias {
            continue;
        }
        // Skip prefixes the weight detector found — their biases will
        // be packed by the per-block bias pass.
        if detected_weight_prefixes.contains(prefix) {
            continue;
        }
        per_prefix
            .entry(prefix.to_string())
            .or_default()
            .push(name.to_string());
    }

    per_prefix
        .into_iter()
        .map(|(prefix, mut keys)| {
            keys.sort();
            (prefix, keys)
        })
        .collect()
}

pub fn pack_all_detected_hf_mixtral_blocks(weight_map: &mut WeightMap) -> AutoPackOutcome {
    let detected = detect_hf_mixtral_blocks(weight_map);
    let mut packed = Vec::new();
    let mut failed = Vec::new();
    let mut bias_packed = Vec::new();
    let mut bias_failed = Vec::new();

    // CPDT Part III v2.16-B (closes v2.15 review F2 HIGH deferral):
    // BEFORE running the per-block pack, scan for per-expert HF bias
    // keys at prefixes WITHOUT matching per-expert weight keys. Those
    // bias-only-no-weights prefixes are NOT detected as blocks by
    // `detect_hf_mixtral_blocks` (which requires all 3 weight keys),
    // so the bias pack never runs for them and the orphans stay in
    // the WeightMap. The user's compilation would later fail with a
    // less obvious "router missing" or `derive_v4_dims` error.
    //
    // Surface the orphans HERE as `BiasesWithoutWeights` so the user
    // gets an actionable diagnostic at load time pointing at the
    // actual problem (missing weights for the bias-bearing block).
    let detected_prefixes: std::collections::HashSet<String> = detected
        .iter()
        .map(|b| b.hf_prefix.clone())
        .collect();
    for (orphan_prefix, bias_keys) in find_bias_only_prefixes(weight_map, &detected_prefixes) {
        // Synthetic DetectedHfMixtralBlock — num_experts=0 is a
        // sentinel meaning "the orchestrator never derived this from
        // weights." The error message in `BiasesWithoutWeights`
        // carries the actionable detail.
        let synthetic_block = DetectedHfMixtralBlock {
            hf_prefix: orphan_prefix.clone(),
            num_experts: 0,
        };
        bias_failed.push((
            synthetic_block,
            PackError::BiasesWithoutWeights {
                hf_prefix: orphan_prefix,
                bias_keys,
            },
        ));
    }

    for block in detected {
        // Rewrite in place — target prefix == HF prefix. The v2.7
        // source kwarg `@moe(weight_prefix="…")` lets users point the
        // v4 lowering at the HF prefix directly.
        match pack_hf_mixtral_experts(
            weight_map,
            &block.hf_prefix,
            &block.hf_prefix,
            block.num_experts,
        ) {
            Ok(outcome) => {
                packed.push((block.clone(), outcome));
                // CPDT Part III v2.15 — bias auto-pack. Only attempted
                // when the weight pack succeeded (a weight-pack failure
                // already surfaces as a hard build error at the
                // entry-points wrapper). The bias pack is independent
                // of the weight pack's atomicity: Ok(None) is the
                // no-bias no-op, Ok(Some) is a successful per-block
                // packing, Err is an all-or-nothing refusal that
                // entry_points.rs surfaces alongside weight-pack
                // failures.
                match pack_hf_mixtral_biases(
                    weight_map,
                    &block.hf_prefix,
                    &block.hf_prefix,
                    block.num_experts,
                ) {
                    Ok(Some(bias_outcome)) => {
                        bias_packed.push((block, bias_outcome));
                    }
                    Ok(None) => {
                        // No biases on this block — clean no-op. Don't
                        // record; the absence in `bias_packed` itself
                        // is the signal that this block has no
                        // biases.
                    }
                    Err(err) => {
                        bias_failed.push((block, err));
                    }
                }
            }
            Err(err) => failed.push((block, err)),
        }
    }
    AutoPackOutcome { packed, failed, bias_packed, bias_failed }
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

    /// CPDT Part III v2.17: BF16-typed test fixture (parallel to
    /// make_f32_entry). Real HF Mixtral checkpoints ship in BF16; the
    /// byte-agnostic pack chain MUST produce a packed entry that
    /// preserves the BF16 dtype + correct byte content.
    fn make_bf16_entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let num_elements = shape.iter().product::<usize>();
        assert_eq!(vals.len(), num_elements, "shape vs vals mismatch");
        let mut data = Vec::with_capacity(num_elements * 2);
        for v in vals {
            data.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape,
            dtype: WeightDType::BF16,
            num_elements,
            sparsity: None,
            eliminated: false,
        }
    }

    fn read_bf16(entry: &WeightEntry) -> Vec<f32> {
        entry
            .data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect()
    }

    /// CPDT Part III v2.17: F16-typed test fixture (parallel to
    /// make_f32_entry). Some HF checkpoints use F16 (older Llama,
    /// converted GPT variants). The pack must handle it identically
    /// to BF16 since both have byte_width=2.
    fn make_f16_entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let num_elements = shape.iter().product::<usize>();
        assert_eq!(vals.len(), num_elements, "shape vs vals mismatch");
        let mut data = Vec::with_capacity(num_elements * 2);
        for v in vals {
            data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape,
            dtype: WeightDType::F16,
            num_elements,
            sparsity: None,
            eliminated: false,
        }
    }

    fn read_f16(entry: &WeightEntry) -> Vec<f32> {
        entry
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
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

    // ── v2.7 auto-detection tests ─────────────────────────────────────

    fn insert_hf_block(wm: &mut WeightMap, prefix: &str, num_experts: usize) {
        for e in 0..num_experts {
            for w in ["w1", "w2", "w3"] {
                let (rows, cols) = if w == "w2" { (2, 3) } else { (3, 2) };
                wm.insert(make_f32_entry(
                    &format!("{}.experts.{}.{}.weight", prefix, e, w),
                    vec![rows, cols],
                    &[1.0_f32; 6],
                ));
            }
        }
    }

    #[test]
    fn detect_hf_mixtral_blocks_finds_single_complete_block() {
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "model.layers.0.block_sparse_moe", 2);
        let found = detect_hf_mixtral_blocks(&wm);
        assert_eq!(
            found,
            vec![DetectedHfMixtralBlock {
                hf_prefix: "model.layers.0.block_sparse_moe".to_string(),
                num_experts: 2,
            }]
        );
    }

    #[test]
    fn detect_hf_mixtral_blocks_finds_multiple_blocks_sorted_by_prefix() {
        // Two MoE layers, one with 4 experts, one with 2.
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "model.layers.1.block_sparse_moe", 2);
        insert_hf_block(&mut wm, "model.layers.0.block_sparse_moe", 4);
        let found = detect_hf_mixtral_blocks(&wm);
        // BTreeMap-backed sort: layers.0 before layers.1.
        assert_eq!(
            found,
            vec![
                DetectedHfMixtralBlock {
                    hf_prefix: "model.layers.0.block_sparse_moe".to_string(),
                    num_experts: 4,
                },
                DetectedHfMixtralBlock {
                    hf_prefix: "model.layers.1.block_sparse_moe".to_string(),
                    num_experts: 2,
                },
            ]
        );
    }

    #[test]
    fn detect_hf_mixtral_blocks_skips_block_with_incomplete_expert_0() {
        // Only w1 + w2 present for expert 0 — w3 missing. The block
        // is incomplete and must NOT be detected.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.0.w1.weight", vec![3, 2], &[1.0; 6]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.weight", vec![2, 3], &[1.0; 6]));
        let found = detect_hf_mixtral_blocks(&wm);
        assert!(found.is_empty(), "incomplete expert 0 must not detect");
    }

    #[test]
    fn detect_hf_mixtral_blocks_stops_at_first_incomplete_index() {
        // Experts 0+1 complete, expert 2 partial. Detector reports
        // num_experts=2. Caller's pack invocation will then trip
        // ExtraExpertsPresent on expert 2 — correct behavior, the
        // user has a malformed safetensors.
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "moe0", 2);
        // Add a partial expert 2 (only w1).
        wm.insert(make_f32_entry("moe0.experts.2.w1.weight", vec![3, 2], &[1.0; 6]));
        let found = detect_hf_mixtral_blocks(&wm);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].num_experts, 2);
    }

    #[test]
    fn detect_hf_mixtral_blocks_ignores_already_packed_entries() {
        // A WeightMap that's already been packed (e.g., from a prior
        // CLI run with the same file) must NOT detect a new block —
        // the packed `experts.gate.weight` / `experts.up.weight` etc.
        // are NOT numeric expert indices.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry(
            "moe0.experts.gate.weight",
            vec![2, 12],
            &[1.0; 24],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.up.weight",
            vec![2, 12],
            &[1.0; 24],
        ));
        wm.insert(make_f32_entry(
            "moe0.experts.down.weight",
            vec![2, 12],
            &[1.0; 24],
        ));
        let found = detect_hf_mixtral_blocks(&wm);
        assert!(found.is_empty(), "already-packed entries must not detect as HF");
    }

    #[test]
    fn detect_hf_mixtral_blocks_ignores_unrelated_keys() {
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("router.weight", vec![2, 2], &[1.0; 4]));
        wm.insert(make_f32_entry(
            "model.embed_tokens.weight",
            vec![2, 2],
            &[1.0; 4],
        ));
        // No `.experts.<N>.w[123].weight` keys at all.
        let found = detect_hf_mixtral_blocks(&wm);
        assert!(found.is_empty());
    }

    // ── pack_all_detected_hf_mixtral_blocks orchestrator ──────────────

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_packs_one_block_in_place() {
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "moe0", 2);
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1);
        assert!(outcome.failed.is_empty());
        assert_eq!(outcome.packed[0].0.hf_prefix, "moe0");
        // Packed entries appeared under HF prefix.
        assert!(wm.get("moe0.experts.gate.weight").is_some());
        assert!(wm.get("moe0.experts.up.weight").is_some());
        assert!(wm.get("moe0.experts.down.weight").is_some());
        // HF sources removed.
        assert!(wm.get("moe0.experts.0.w1.weight").is_none());
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_independent_failure_does_not_block_other_blocks() {
        // Two MoE blocks. The first has a dtype mismatch (corrupted);
        // the second is clean. The auto-pack must successfully pack
        // the second and report the first as failed — cross-block
        // failures must not roll back successful blocks.
        let mut wm = WeightMap::new_for_test();
        // Block 0: corrupted (expert 1's w1 has a different dtype).
        wm.insert(make_f32_entry("bad.experts.0.w1.weight", vec![2, 2], &[1.0; 4]));
        wm.insert(make_f32_entry("bad.experts.0.w2.weight", vec![2, 2], &[1.0; 4]));
        wm.insert(make_f32_entry("bad.experts.0.w3.weight", vec![2, 2], &[1.0; 4]));
        let bad_w1_f16 = vec![0u8; 2 * 2 * 2];
        wm.insert(WeightEntry::new(
            "bad.experts.1.w1.weight".to_string(),
            bad_w1_f16,
            vec![2, 2],
            WeightDType::F16,
        ));
        wm.insert(make_f32_entry("bad.experts.1.w2.weight", vec![2, 2], &[1.0; 4]));
        wm.insert(make_f32_entry("bad.experts.1.w3.weight", vec![2, 2], &[1.0; 4]));
        // Block 1: clean.
        insert_hf_block(&mut wm, "good", 2);

        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1, "the good block must pack");
        assert_eq!(outcome.failed.len(), 1, "the bad block must fail");
        assert_eq!(outcome.packed[0].0.hf_prefix, "good");
        assert_eq!(outcome.failed[0].0.hf_prefix, "bad");
        assert!(matches!(outcome.failed[0].1, PackError::DtypeMismatch { .. }));
        // Good block is packed; bad block's HF sources are still
        // present (atomic refusal per v2.6's two-phase commit).
        assert!(wm.get("good.experts.gate.weight").is_some());
        assert!(wm.get("bad.experts.0.w1.weight").is_some());
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_noop_on_empty_or_already_packed() {
        let mut wm = WeightMap::new_for_test();
        // Already-packed; detector returns empty; orchestrator no-ops.
        wm.insert(make_f32_entry(
            "moe0.experts.gate.weight",
            vec![2, 12],
            &[1.0; 24],
        ));
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert!(outcome.packed.is_empty());
        assert!(outcome.failed.is_empty());
        // Pre-existing packed entry is untouched.
        assert!(wm.get("moe0.experts.gate.weight").is_some());
    }

    // ── v2.7 adversarial-review fixes ─────────────────────────────────

    #[test]
    fn pack_hf_mixtral_experts_refuses_gapped_experts_with_orphan_at_higher_index() {
        // v2.7 adversarial-review fix F1 (HIGH): pre-fix, a checkpoint
        // with experts {0, 1, 3} (expert 2 entirely missing, expert 3
        // fully present) would silently half-pack — the per-projection
        // probe only checked index `num_experts` (= 2), and since
        // expert 2 was MISSING, no refusal fired. Expert 3's three
        // tensors leaked as orphans.
        //
        // The unified scan catches ANY index >= num_experts. Pin it
        // explicitly with the gapped case.
        let mut wm = WeightMap::new_for_test();
        // Experts 0 + 1 complete.
        for e in [0_usize, 1_usize] {
            for w in ["w1", "w2", "w3"] {
                let (rows, cols) = if w == "w2" { (2, 3) } else { (3, 2) };
                wm.insert(make_f32_entry(
                    &format!("moe0.experts.{}.{}.weight", e, w),
                    vec![rows, cols],
                    &[1.0_f32; 6],
                ));
            }
        }
        // Expert 2 ENTIRELY MISSING; expert 3 fully present.
        for w in ["w1", "w2", "w3"] {
            let (rows, cols) = if w == "w2" { (2, 3) } else { (3, 2) };
            wm.insert(make_f32_entry(
                &format!("moe0.experts.3.{}.weight", w),
                vec![rows, cols],
                &[1.0_f32; 6],
            ));
        }

        // detect_hf_mixtral_blocks reports num_experts=2 (the largest
        // contiguous run starting at 0). The packer must then refuse
        // because expert 3 is an orphan.
        let detected = detect_hf_mixtral_blocks(&wm);
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].num_experts, 2);

        let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ExtraExpertsPresent {
                declared_num_experts,
                found_extra_index,
                ..
            } => {
                assert_eq!(declared_num_experts, 2);
                assert_eq!(found_extra_index, 3, "must find the orphan at index 3, NOT only probe index 2");
            }
            other => panic!("expected ExtraExpertsPresent at index 3, got {:?}", other),
        }
        // WeightMap atomically unmodified — orphan still present, no
        // packed entries appeared.
        assert!(wm.get("moe0.experts.gate.weight").is_none());
        assert!(wm.get("moe0.experts.3.w1.weight").is_some());
    }

    #[test]
    fn pack_hf_mixtral_experts_refuses_orphan_at_arbitrary_higher_index() {
        // Same F1 fix, broader scope: orphans at index 7 with
        // num_experts=2 must also be caught.
        let mut wm = WeightMap::new_for_test();
        for e in [0_usize, 1_usize] {
            for w in ["w1", "w2", "w3"] {
                let (rows, cols) = if w == "w2" { (2, 3) } else { (3, 2) };
                wm.insert(make_f32_entry(
                    &format!("hf.experts.{}.{}.weight", e, w),
                    vec![rows, cols],
                    &[1.0_f32; 6],
                ));
            }
        }
        // Far-away orphan.
        wm.insert(make_f32_entry(
            "hf.experts.7.w1.weight",
            vec![3, 2],
            &[1.0_f32; 6],
        ));

        let err = pack_hf_mixtral_experts(&mut wm, "hf", "hf", 2).unwrap_err();
        match err {
            PackError::ExtraExpertsPresent { found_extra_index, .. } => {
                assert_eq!(found_extra_index, 7);
            }
            other => panic!("expected ExtraExpertsPresent at index 7, got {:?}", other),
        }
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_mixed_hf_and_already_packed_reports_failure() {
        // v2.7 adversarial-review fix F2 (IMPORTANT): a WeightMap with
        // BOTH HF per-expert keys AND already-packed entries under the
        // SAME prefix is a corrupted/half-aborted state. The auto-pack
        // detector finds the HF block; the packer immediately trips
        // `TargetAlreadyExists` on the pre-existing packed entry. The
        // failure is reported via auto_pack.failed — and the wrapper
        // in entry_points.rs then SURFACES the failure as a hard
        // CodegenError (F2 fix) rather than silently letting the build
        // continue against stale shadow data.
        let mut wm = WeightMap::new_for_test();
        // Pre-existing packed gate entry.
        wm.insert(make_f32_entry(
            "moe0.experts.gate.weight",
            vec![2, 12],
            &[99.0; 24],
        ));
        // Plus a fresh HF expert under the same prefix.
        for w in ["w1", "w2", "w3"] {
            let (rows, cols) = if w == "w2" { (2, 3) } else { (3, 2) };
            wm.insert(make_f32_entry(
                &format!("moe0.experts.0.{}.weight", w),
                vec![rows, cols],
                &[1.0; 6],
            ));
        }
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert!(
            outcome.packed.is_empty(),
            "mixed-state block must not pack: {:?}",
            outcome.packed,
        );
        assert_eq!(outcome.failed.len(), 1);
        assert!(matches!(
            outcome.failed[0].1,
            PackError::TargetAlreadyExists { .. }
        ));
        // The stale packed entry is preserved (atomic refusal).
        assert!(wm.get("moe0.experts.gate.weight").is_some());
        assert!(wm.get("moe0.experts.0.w1.weight").is_some());
    }

    // ── v2.15 — HF bias auto-pack tests ───────────────────────────────

    /// Helper: insert per-expert HF biases for ALL three projections.
    /// `intermediate` is the gate/up bias dim; `hidden` is the down
    /// bias dim. Matches `nn.Linear.bias` shape `[out_features]` per
    /// projection.
    fn insert_hf_biases(
        wm: &mut WeightMap,
        prefix: &str,
        num_experts: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        for e in 0..num_experts {
            // w1.bias (gate) shape [intermediate]
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.w1.bias", prefix, e),
                vec![intermediate],
                &vec![1.0_f32 + e as f32; intermediate],
            ));
            // w3.bias (up) shape [intermediate]
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.w3.bias", prefix, e),
                vec![intermediate],
                &vec![10.0_f32 + e as f32; intermediate],
            ));
            // w2.bias (down) shape [hidden]
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.w2.bias", prefix, e),
                vec![hidden],
                &vec![100.0_f32 + e as f32; hidden],
            ));
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_two_experts_packs_correctly() {
        // 2 experts, hidden=2, intermediate=3. Weights are present too
        // but this test exercises the bias pack in isolation (skip the
        // orchestrator). After packing:
        //   - gate.bias / up.bias shape [2, 3] (intermediate dim)
        //   - down.bias shape [2, 2] (hidden dim)
        let mut wm = WeightMap::new_for_test();
        insert_hf_biases(&mut wm, "moe0", 2, 2, 3);

        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("pack ok")
            .expect("biases present, must produce Some");

        assert_eq!(outcome.num_experts, 2);
        assert_eq!(outcome.inserted_names.len(), 3);
        assert_eq!(outcome.removed_names.len(), 6);

        let gate = wm.get("moe0.experts.gate.bias").expect("gate.bias present");
        assert_eq!(gate.shape, vec![2, 3]);
        // Expert 0 gate.bias = [1, 1, 1]; expert 1 = [2, 2, 2].
        let gate_vals = read_f32(gate);
        assert_eq!(gate_vals, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let up = wm.get("moe0.experts.up.bias").expect("up.bias present");
        assert_eq!(up.shape, vec![2, 3]);
        let up_vals = read_f32(up);
        assert_eq!(up_vals, vec![10.0, 10.0, 10.0, 11.0, 11.0, 11.0]);

        let down = wm.get("moe0.experts.down.bias").expect("down.bias present");
        assert_eq!(down.shape, vec![2, 2]);
        let down_vals = read_f32(down);
        assert_eq!(down_vals, vec![100.0, 100.0, 101.0, 101.0]);

        // Per-expert HF bias sources removed.
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                let key = format!("moe0.experts.{}.{}.bias", e, proj);
                assert!(wm.get(&key).is_none(), "{} should be removed", key);
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_no_biases_present_returns_ok_none() {
        // Clean no-bias case — no per-expert .w*.bias keys exist. The
        // pack returns Ok(None) and leaves the WeightMap untouched.
        let mut wm = WeightMap::new_for_test();
        // Insert only weights (no biases) so the orchestrator's
        // weight-pack would still succeed for this block.
        insert_hf_block(&mut wm, "moe0", 2);
        let names_before: Vec<String> = wm.names().map(|s| s.to_string()).collect();
        let result = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("pack ok");
        assert!(result.is_none(), "no biases present → Ok(None) expected");
        let names_after: Vec<String> = wm.names().map(|s| s.to_string()).collect();
        assert_eq!(names_before, names_after, "WeightMap must be untouched");
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_partial_bundle_missing_one_expert() {
        // 2 experts; expert 0 has all 3 bias suffixes, expert 1 only
        // has w1.bias + w3.bias (missing w2.bias). Partial bundle —
        // refuse loudly.
        let mut wm = WeightMap::new_for_test();
        // Full expert 0.
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        // Partial expert 1 — w2.bias missing.
        wm.insert(make_f32_entry("moe0.experts.1.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.1.w3.bias", vec![3], &[1.0; 3]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::PartialBiasBundle { present, missing, .. } => {
                assert_eq!(present.len(), 5);
                assert_eq!(missing.len(), 1);
                assert!(missing.iter().any(|s| s == "moe0.experts.1.w2.bias"));
            }
            other => panic!("expected PartialBiasBundle, got {:?}", other),
        }
        // Atomicity — every bias source preserved, no packed entries.
        assert!(wm.get("moe0.experts.0.w1.bias").is_some());
        assert!(wm.get("moe0.experts.gate.bias").is_none());
        assert!(wm.get("moe0.experts.up.bias").is_none());
        assert!(wm.get("moe0.experts.down.bias").is_none());
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_partial_bundle_missing_full_projection() {
        // Partial-by-projection: ALL experts have w1.bias + w3.bias but
        // NO expert has w2.bias. The downstream `detect_v4_biases`
        // requires all 3 directions; auto-pack must refuse to avoid
        // silently dropping the down bias at runtime.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![3],
                &[1.0; 3],
            ));
        }
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::PartialBiasBundle { present, missing, .. } => {
                assert_eq!(present.len(), 4);
                assert_eq!(missing.len(), 2);
                // Both missing entries are w2.bias.
                assert!(missing.iter().all(|s| s.ends_with("w2.bias")));
            }
            other => panic!("expected PartialBiasBundle, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_shape_mismatch() {
        // Expert 0 w1.bias is [3]; expert 1 w1.bias is [4]. Mismatched
        // bias dim on Gate — silent-corruption hazard (packed buffer
        // would have wrong stride).
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        wm.insert(make_f32_entry("moe0.experts.1.w1.bias", vec![4], &[1.0; 4]));
        wm.insert(make_f32_entry("moe0.experts.1.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.1.w2.bias", vec![2], &[1.0; 2]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ShapeMismatch { projection, expected, actual, offending_expert } => {
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(expected, vec![3]);
                assert_eq!(actual, vec![4]);
                assert_eq!(offending_expert, 1);
            }
            other => panic!("expected ShapeMismatch on Gate, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_dtype_mismatch() {
        // Expert 0 biases F32; expert 1 w3.bias F16 (same element
        // count, different dtype). Refuse — silent reinterpretation
        // across the packed tensor would corrupt the runtime read.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        wm.insert(make_f32_entry("moe0.experts.1.w1.bias", vec![3], &[1.0; 3]));
        // F16 expert-1 w3.bias — 3 elements * 2 bytes = 6 bytes.
        wm.insert(WeightEntry::new(
            "moe0.experts.1.w3.bias".to_string(),
            vec![0u8; 6],
            vec![3],
            WeightDType::F16,
        ));
        wm.insert(make_f32_entry("moe0.experts.1.w2.bias", vec![2], &[1.0; 2]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::DtypeMismatch { projection, offending_expert, .. } => {
                assert_eq!(projection, HfProjection::Up);
                assert_eq!(offending_expert, 1);
            }
            other => panic!("expected DtypeMismatch on Up, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_target_collision() {
        // Pre-existing gate.bias under target prefix — refuse loudly
        // rather than overwriting.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.gate.bias", vec![1, 3], &[99.0; 3]));
        // Plus a valid HF bias bundle for 1 expert.
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(
            matches!(err, PackError::TargetAlreadyExists { ref name } if name == "moe0.experts.gate.bias"),
            "expected TargetAlreadyExists on gate.bias, got {:?}",
            err,
        );
        // Stale packed entry preserved; HF sources preserved.
        assert!(wm.get("moe0.experts.0.w1.bias").is_some());
        let pre = wm.get("moe0.experts.gate.bias").unwrap();
        assert_eq!(read_f32(pre), vec![99.0, 99.0, 99.0]);
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_extra_biases_at_higher_index() {
        // Caller declares 2 experts; valid bias bundles for experts
        // 0+1; expert 5 has w1.bias only (orphan). Refuse loudly via
        // the unified extra-bias scan — partial expert 5 would also
        // trip PartialBiasBundle, but for HIGH indices the orphan
        // semantics are clearer than "partial bundle".
        let mut wm = WeightMap::new_for_test();
        insert_hf_biases(&mut wm, "moe0", 2, 2, 3);
        // Orphan at index 5.
        wm.insert(make_f32_entry("moe0.experts.5.w1.bias", vec![3], &[1.0; 3]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ExtraBiasesPresent { found_extra_indices, .. } => {
                assert_eq!(found_extra_indices, vec![5]);
            }
            other => panic!("expected ExtraBiasesPresent at [5], got {:?}", other),
        }
        // Atomicity: orphan preserved, no packed entries.
        assert!(wm.get("moe0.experts.5.w1.bias").is_some());
        assert!(wm.get("moe0.experts.gate.bias").is_none());
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_zero_num_experts() {
        let mut wm = WeightMap::new_for_test();
        assert!(matches!(
            pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 0),
            Err(PackError::ZeroNumExperts)
        ));
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_packs_weights_and_biases_in_place() {
        // End-to-end through the orchestrator: a block with both
        // weights AND biases gets both packed in a single call, and
        // the resulting WeightMap looks exactly like a hand-packed
        // bundle.
        let mut wm = WeightMap::new_for_test();
        // hidden=2, intermediate=3, 2 experts. insert_hf_block uses
        // these exact dims for weights (w1/w3 = [3,2] = [intermediate,
        // hidden]; w2 = [2,3] = [hidden, intermediate]).
        insert_hf_block(&mut wm, "moe0", 2);
        insert_hf_biases(&mut wm, "moe0", 2, 2, 3);

        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1, "weight pack succeeded");
        assert_eq!(outcome.bias_packed.len(), 1, "bias pack succeeded");
        assert!(outcome.failed.is_empty());
        assert!(outcome.bias_failed.is_empty());

        // Both weight + bias packed entries exist under the NSL
        // convention.
        assert!(wm.get("moe0.experts.gate.weight").is_some());
        assert!(wm.get("moe0.experts.up.weight").is_some());
        assert!(wm.get("moe0.experts.down.weight").is_some());
        assert!(wm.get("moe0.experts.gate.bias").is_some());
        assert!(wm.get("moe0.experts.up.bias").is_some());
        assert!(wm.get("moe0.experts.down.bias").is_some());

        // All HF sources (weights + biases) removed.
        for proj in ["w1", "w2", "w3"] {
            for e in 0..2 {
                assert!(
                    wm.get(&format!("moe0.experts.{}.{}.weight", e, proj)).is_none(),
                    "{} weight should be removed", proj,
                );
                assert!(
                    wm.get(&format!("moe0.experts.{}.{}.bias", e, proj)).is_none(),
                    "{} bias should be removed", proj,
                );
            }
        }
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_packs_weights_skipping_biases_when_absent() {
        // Block has weights only (no biases) — the orchestrator packs
        // weights successfully and the bias pass returns Ok(None) so
        // nothing is recorded in bias_packed. Verifies the no-op path
        // doesn't pollute the outcome.
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "moe0", 2);
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1);
        assert!(outcome.bias_packed.is_empty(), "no biases → bias_packed empty");
        assert!(outcome.failed.is_empty());
        assert!(outcome.bias_failed.is_empty());
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_partial_bias_bundle_surfaces_in_bias_failed() {
        // Block has weights + a PARTIAL bias bundle (only w1.bias for
        // expert 0). The orchestrator packs weights successfully but
        // the bias pass refuses with PartialBiasBundle, which lands
        // in bias_failed (and entry_points.rs turns this into a hard
        // build error).
        let mut wm = WeightMap::new_for_test();
        insert_hf_block(&mut wm, "moe0", 2);
        // Only one bias key out of 6 — clear partial state.
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1, "weight pack must succeed");
        assert!(outcome.bias_packed.is_empty(), "bias pack failed, nothing packed");
        assert_eq!(outcome.bias_failed.len(), 1, "partial-bias-bundle surfaced");
        assert!(matches!(
            outcome.bias_failed[0].1,
            PackError::PartialBiasBundle { .. }
        ));
        // Weight pack DID apply (per-pass atomicity, not cross-pass).
        // Source bias key for expert 0 preserved — the PartialBiasBundle
        // refusal didn't touch it.
        assert!(wm.get("moe0.experts.gate.weight").is_some());
        assert!(wm.get("moe0.experts.0.w1.bias").is_some());
    }

    // ── v2.15 review fixes — additional refusal coverage ──────────────

    #[test]
    fn pack_hf_mixtral_biases_refuses_target_collision_on_up_not_just_gate() {
        // v2.15 review fix F1 (HIGH): the pre-check loop covers all 3
        // bias slots (gate/up/down), but the original test suite only
        // exercised the Gate slot. A regression that consolidated the
        // pre-check to only Gate would let an Up or Down collision
        // proceed to Phase 2's `weight_map.insert()`, which silently
        // overwrites the stale entry. Pin Up explicitly.
        let mut wm = WeightMap::new_for_test();
        // Pre-existing collision on the Up bias target only.
        wm.insert(make_f32_entry("moe0.experts.up.bias", vec![1, 3], &[99.0; 3]));
        // Valid HF bias bundle for 1 expert.
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(
            matches!(err, PackError::TargetAlreadyExists { ref name } if name == "moe0.experts.up.bias"),
            "expected TargetAlreadyExists on up.bias, got {:?}",
            err,
        );
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_target_collision_on_down_not_just_gate() {
        // v2.15 review fix F1 (HIGH): symmetric test for the Down slot.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.down.bias", vec![1, 2], &[99.0; 2]));
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(
            matches!(err, PackError::TargetAlreadyExists { ref name } if name == "moe0.experts.down.bias"),
            "expected TargetAlreadyExists on down.bias, got {:?}",
            err,
        );
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_zero_bias_dim() {
        // v2.15 review fix F6 (IMPORTANT): a [0]-shaped bias for all
        // experts trips the ZeroBiasDim refusal at Phase 1, surfacing
        // the cause at the bias source rather than as a less actionable
        // "0 elements; expected N" diagnostic from detect_v4_biases
        // later.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![0],
                &[],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w2.bias", e),
                vec![2],
                &[1.0; 2],
            ));
        }
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ZeroBiasDim { projection, offending_key, .. } => {
                assert_eq!(projection, HfProjection::Gate);
                assert!(offending_key.ends_with("w1.bias"));
            }
            other => panic!("expected ZeroBiasDim on Gate, got {:?}", other),
        }
        // Atomicity: no packed entries appeared.
        assert!(wm.get("moe0.experts.gate.bias").is_none());
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_mixed_projection_dtypes() {
        // v2.15 review fix F7 (IMPORTANT): gate F32, up F32, down F16
        // is internally consistent within each projection (every
        // expert's bias for that projection has the same dtype), but
        // the cross-projection mismatch would silently fail the
        // runtime FFI's dtype-equality gate (return 0 with no clear
        // diagnostic). Refusing here at pack time gives a clear error.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            // Gate + Up: F32 (3 elements per bias).
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            // Down: F16 (2 elements per bias = 4 bytes).
            wm.insert(WeightEntry::new(
                format!("moe0.experts.{}.w2.bias", e),
                vec![0u8; 4],
                vec![2],
                WeightDType::F16,
            ));
        }
        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::MixedProjectionDtypes { gate_dtype, up_dtype, down_dtype, .. } => {
                assert_eq!(gate_dtype, "F32");
                assert_eq!(up_dtype, "F32");
                assert_eq!(down_dtype, "F16");
            }
            other => panic!("expected MixedProjectionDtypes, got {:?}", other),
        }
        // Atomicity: no packed entries appeared.
        assert!(wm.get("moe0.experts.gate.bias").is_none());
        assert!(wm.get("moe0.experts.up.bias").is_none());
        assert!(wm.get("moe0.experts.down.bias").is_none());
    }

    // ── v2.16 — short-form suffix + bias-only detection + multi-orphan ─

    /// v2.16-A helper: insert biases under the SHORT `.b{N}` suffix
    /// form for a single block (parallel to insert_hf_biases which
    /// uses the canonical `.w{N}.bias` form).
    fn insert_hf_biases_short_form(
        wm: &mut WeightMap,
        prefix: &str,
        num_experts: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        for e in 0..num_experts {
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.b1", prefix, e),
                vec![intermediate],
                &vec![1.0_f32 + e as f32; intermediate],
            ));
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.b3", prefix, e),
                vec![intermediate],
                &vec![10.0_f32 + e as f32; intermediate],
            ));
            wm.insert(make_f32_entry(
                &format!("{}.experts.{}.b2", prefix, e),
                vec![hidden],
                &vec![100.0_f32 + e as f32; hidden],
            ));
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_accepts_short_form_suffix() {
        // v2.16-A: a checkpoint shipping biases under `.b{1,2,3}`
        // resolves identically to the canonical `.w{1,2,3}.bias`
        // form. Numerical content must be byte-identical to the
        // canonical happy-path test.
        let mut wm = WeightMap::new_for_test();
        insert_hf_biases_short_form(&mut wm, "moe0", 2, 2, 3);

        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("pack ok")
            .expect("biases present");

        assert_eq!(outcome.num_experts, 2);
        let gate = wm.get("moe0.experts.gate.bias").expect("gate.bias present");
        assert_eq!(gate.shape, vec![2, 3]);
        assert_eq!(read_f32(gate), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let up = wm.get("moe0.experts.up.bias").expect("up.bias present");
        assert_eq!(read_f32(up), vec![10.0, 10.0, 10.0, 11.0, 11.0, 11.0]);
        let down = wm.get("moe0.experts.down.bias").expect("down.bias present");
        assert_eq!(read_f32(down), vec![100.0, 100.0, 101.0, 101.0]);

        // Short-form HF sources removed.
        for short in ["b1", "b2", "b3"] {
            for e in 0..2 {
                let key = format!("moe0.experts.{}.{}", e, short);
                assert!(wm.get(&key).is_none(), "{} should be removed", key);
            }
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_mixed_canonical_and_short_suffix_forms() {
        // v2.16-A invariant: per-block uniformity. Expert 0 uses
        // canonical `.w1.bias`, expert 1 uses short `.b1`. Refuse —
        // a partially-converted checkpoint is the typical cause.
        let mut wm = WeightMap::new_for_test();
        // Expert 0: canonical
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[1.0; 2]));
        // Expert 1: short
        wm.insert(make_f32_entry("moe0.experts.1.b1", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.1.b3", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.1.b2", vec![2], &[1.0; 2]));

        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::MixedBiasSuffixForms { canonical_keys, short_keys, .. } => {
                assert_eq!(canonical_keys.len(), 3, "expert 0 contributes 3 canonical");
                assert_eq!(short_keys.len(), 3, "expert 1 contributes 3 short");
            }
            other => panic!("expected MixedBiasSuffixForms, got {:?}", other),
        }
        // Atomicity: every source preserved, no packed entries.
        assert!(wm.get("moe0.experts.0.w1.bias").is_some());
        assert!(wm.get("moe0.experts.1.b1").is_some());
        assert!(wm.get("moe0.experts.gate.bias").is_none());
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_same_slot_both_forms_present() {
        // v2.16 review fix F1 (IMPORTANT adversarial review): if a
        // single slot has BOTH canonical `.w{N}.bias` and short `.b{N}`
        // forms, the pre-fix behavior was to silently prefer canonical
        // and leave the short form as an orphan. The adversarial review
        // surfaced this 3 times as an actionability gap (mismatched
        // shapes/dtypes between the two duplicates would also be
        // silently undetected). Refuse loudly with
        // SameSlotBothBiasSuffixForms pointing at the offending slot.
        let mut wm = WeightMap::new_for_test();
        // Expert 0 has BOTH canonical w1.bias AND short b1 — same
        // projection, same expert. Different values to exercise the
        // "would have silently picked one" hazard.
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[7.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[7.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[7.0; 2]));
        wm.insert(make_f32_entry("moe0.experts.0.b1", vec![3], &[99.0; 3]));

        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 1).unwrap_err();
        match err {
            PackError::SameSlotBothBiasSuffixForms {
                projection,
                expert_index,
                canonical_key,
                short_key,
                ..
            } => {
                assert_eq!(projection, HfProjection::Gate);
                assert_eq!(expert_index, 0);
                assert_eq!(canonical_key, "moe0.experts.0.w1.bias");
                assert_eq!(short_key, "moe0.experts.0.b1");
            }
            other => panic!("expected SameSlotBothBiasSuffixForms, got {:?}", other),
        }
        // Atomicity — both forms still present, no packed entries.
        assert!(wm.get("moe0.experts.0.w1.bias").is_some());
        assert!(wm.get("moe0.experts.0.b1").is_some());
        assert!(wm.get("moe0.experts.gate.bias").is_none());
    }

    #[test]
    fn pack_hf_mixtral_biases_refuses_same_slot_both_forms_with_mismatched_dtype() {
        // v2.16 review fix F1 explicit dtype-mismatch scenario:
        // canonical w1.bias is F32, short b1 is F16. Pre-fix, the
        // canonical (F32) would have been picked and the F16 b1 would
        // have stayed as an orphan with NO dtype validation. Now both
        // duplicates trip the refusal, exposing the inconsistency.
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_f32_entry("moe0.experts.0.w1.bias", vec![3], &[7.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w3.bias", vec![3], &[7.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.0.w2.bias", vec![2], &[7.0; 2]));
        // F16 b1 — 3 elements * 2 bytes = 6 bytes.
        wm.insert(WeightEntry::new(
            "moe0.experts.0.b1".to_string(),
            vec![0u8; 6],
            vec![3],
            WeightDType::F16,
        ));

        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 1).unwrap_err();
        assert!(matches!(err, PackError::SameSlotBothBiasSuffixForms { .. }));
    }

    #[test]
    fn pack_hf_mixtral_biases_canonical_and_short_produce_byte_identical_packed() {
        // v2.16 review fix F3 (LOW): the suffix-form aliasing contract
        // says canonical and short produce byte-identical packed
        // entries for the same numerical values. Lock that in by
        // packing the same numerical content twice (once per form)
        // and comparing the resulting packed entries byte-for-byte.
        let canonical_packed: Vec<u8> = {
            let mut wm = WeightMap::new_for_test();
            insert_hf_biases(&mut wm, "moe0", 2, 2, 3);
            pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
                .expect("canonical pack ok")
                .expect("biases present");
            wm.get("moe0.experts.gate.bias").unwrap().data.clone()
        };
        let short_packed: Vec<u8> = {
            let mut wm = WeightMap::new_for_test();
            insert_hf_biases_short_form(&mut wm, "moe0", 2, 2, 3);
            pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
                .expect("short pack ok")
                .expect("biases present");
            wm.get("moe0.experts.gate.bias").unwrap().data.clone()
        };
        assert_eq!(
            canonical_packed, short_packed,
            "canonical vs short pack must produce byte-identical packed gate.bias",
        );
    }

    #[test]
    fn pack_hf_mixtral_biases_enumerates_all_extra_bias_orphan_indices() {
        // v2.16-C: orphans at indices 3, 5, 7 (num_experts=2) all
        // appear in `found_extra_indices`, sorted + deduplicated.
        // Pre-v2.16 the scan only reported the max (idx 7); users
        // had to rebuild after each fix.
        let mut wm = WeightMap::new_for_test();
        insert_hf_biases(&mut wm, "moe0", 2, 2, 3);
        // Multiple orphans at different indices.
        wm.insert(make_f32_entry("moe0.experts.3.w1.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.5.w3.bias", vec![3], &[1.0; 3]));
        wm.insert(make_f32_entry("moe0.experts.7.w2.bias", vec![2], &[1.0; 2]));
        // Also include a short-form orphan to confirm both forms
        // contribute to the scan.
        wm.insert(make_f32_entry("moe0.experts.9.b1", vec![3], &[1.0; 3]));

        let err = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2).unwrap_err();
        match err {
            PackError::ExtraBiasesPresent { found_extra_indices, .. } => {
                assert_eq!(
                    found_extra_indices,
                    vec![3, 5, 7, 9],
                    "all orphan indices must be enumerated, sorted",
                );
            }
            other => panic!("expected ExtraBiasesPresent, got {:?}", other),
        }
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_surfaces_bias_only_orphan_prefix() {
        // v2.16-B: a WeightMap with bias keys at a prefix but NO
        // matching weight keys. The detector returns empty for this
        // prefix; the orchestrator's pre-scan must surface
        // BiasesWithoutWeights into bias_failed.
        let mut wm = WeightMap::new_for_test();
        // Real HF block A — fully weights, no biases.
        insert_hf_block(&mut wm, "good", 2);
        // Orphan block B — biases only, no weights.
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("bias_only.experts.{}.w1.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("bias_only.experts.{}.w3.bias", e),
                vec![3],
                &[1.0; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("bias_only.experts.{}.w2.bias", e),
                vec![2],
                &[1.0; 2],
            ));
        }

        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        // Good block packs cleanly.
        assert_eq!(outcome.packed.len(), 1);
        assert_eq!(outcome.packed[0].0.hf_prefix, "good");
        // Bias-only block surfaces in bias_failed.
        assert_eq!(outcome.bias_failed.len(), 1);
        assert_eq!(outcome.bias_failed[0].0.hf_prefix, "bias_only");
        assert!(matches!(
            outcome.bias_failed[0].1,
            PackError::BiasesWithoutWeights { .. }
        ));
        // Orphan bias keys are preserved (not silently dropped).
        assert!(wm.get("bias_only.experts.0.w1.bias").is_some());
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_orphan_prefix_short_form_also_detected() {
        // v2.16-A + v2.16-B composition: orphan biases use the short
        // `.b{N}` form. find_bias_only_prefixes must catch them too.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f32_entry(
                &format!("orphan.experts.{}.b1", e),
                vec![3],
                &[1.0; 3],
            ));
        }
        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.bias_failed.len(), 1);
        match &outcome.bias_failed[0].1 {
            PackError::BiasesWithoutWeights { bias_keys, .. } => {
                assert!(bias_keys.iter().any(|k| k.ends_with(".b1")));
            }
            other => panic!("expected BiasesWithoutWeights, got {:?}", other),
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_accepts_2d_packed_bias_shape() {
        // Some checkpoints ship per-expert bias as `[1, D]` instead of
        // `[D]` (a leftover from a packing pre-script). Match the v4
        // detector's tolerance (num_elements-only) by accepting both
        // layouts. The packed output should be byte-identical.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            // 2-D `[1, 3]` w1.bias instead of 1-D `[3]`.
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![1, 3],
                &[1.0_f32 + e as f32; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![1, 3],
                &[10.0_f32 + e as f32; 3],
            ));
            wm.insert(make_f32_entry(
                &format!("moe0.experts.{}.w2.bias", e),
                vec![1, 2],
                &[100.0_f32 + e as f32; 2],
            ));
        }
        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("pack ok")
            .expect("biases present");
        let gate = wm.get("moe0.experts.gate.bias").expect("gate.bias present");
        // Packed shape is canonicalized to [num_experts, bias_dim],
        // independent of source shape.
        assert_eq!(gate.shape, vec![2, 3]);
        let gate_vals = read_f32(gate);
        assert_eq!(gate_vals, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(outcome.num_experts, 2);
    }

    // ── v2.17 — BF16/F16 dtype coverage ──────────────────────────────
    //
    // The byte-agnostic pack chain SHOULD handle BF16/F16 cleanly by
    // design (transpose_2d_bytes works in raw bytes; element_bytes
    // derives from WeightDType::byte_width). v2.17 pins this with
    // happy-path tests covering both dtypes for weight + bias pack +
    // orchestrator. Real HF Mixtral checkpoints ship in BF16; if the
    // chain regressed to F32-assumption (e.g., a hard-coded 4 in a
    // byte-offset computation), these tests catch it.

    #[test]
    fn pack_hf_mixtral_experts_bf16_two_experts_round_trips() {
        // 2 experts, hidden=2, intermediate=3. BF16 entries throughout.
        // The transpose semantics should be byte-identical to the F32
        // happy-path test (modulo the 2-byte element width).
        let mut wm = WeightMap::new_for_test();
        wm.insert(make_bf16_entry(
            "moe0.experts.0.w1.weight",
            vec![3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        wm.insert(make_bf16_entry(
            "moe0.experts.0.w3.weight",
            vec![3, 2],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ));
        wm.insert(make_bf16_entry(
            "moe0.experts.0.w2.weight",
            vec![2, 3],
            &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ));
        wm.insert(make_bf16_entry(
            "moe0.experts.1.w1.weight",
            vec![3, 2],
            &[19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        ));
        wm.insert(make_bf16_entry(
            "moe0.experts.1.w3.weight",
            vec![3, 2],
            &[25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
        ));
        wm.insert(make_bf16_entry(
            "moe0.experts.1.w2.weight",
            vec![2, 3],
            &[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        ));

        let outcome = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2)
            .expect("BF16 weight pack ok");
        assert_eq!(outcome.num_experts, 2);

        let gate = wm.get("moe0.experts.gate.weight").expect("gate present");
        // Packed dtype MUST be BF16 (preserved from source).
        assert_eq!(gate.dtype, WeightDType::BF16);
        assert_eq!(gate.shape, vec![2, 6]);
        assert_eq!(gate.data.len(), 2 * 6 * 2, "BF16 = 2 bytes/element");
        // Transpose semantics: HF [3, 2] = [[1,2],[3,4],[5,6]] →
        // [2, 3] = [[1,3,5],[2,4,6]] → flat [1,3,5,2,4,6].
        // BF16 is lossless for small integer values 1..30.
        let gate_vals = read_bf16(gate);
        assert_eq!(
            gate_vals,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 19.0, 21.0, 23.0, 20.0, 22.0, 24.0]
        );
    }

    #[test]
    fn pack_hf_mixtral_experts_f16_two_experts_round_trips() {
        // Same as BF16 case but with F16. Both dtypes have byte_width=2
        // so the byte-level pack should be identical modulo the actual
        // bit pattern of each element. Small integers are lossless in
        // F16 up to 2048, so we can use exact equality.
        let mut wm = WeightMap::new_for_test();
        for (e, base) in [(0_usize, 0.0_f32), (1, 100.0)] {
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w1.weight", e),
                vec![3, 2],
                &[base + 1.0, base + 2.0, base + 3.0, base + 4.0, base + 5.0, base + 6.0],
            ));
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w3.weight", e),
                vec![3, 2],
                &[base + 7.0; 6],
            ));
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w2.weight", e),
                vec![2, 3],
                &[base + 8.0; 6],
            ));
        }

        let outcome = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", 2)
            .expect("F16 weight pack ok");
        assert_eq!(outcome.num_experts, 2);

        let gate = wm.get("moe0.experts.gate.weight").expect("gate present");
        assert_eq!(gate.dtype, WeightDType::F16);
        assert_eq!(gate.shape, vec![2, 6]);
        assert_eq!(gate.data.len(), 2 * 6 * 2);
        let gate_vals = read_f16(gate);
        // Expert 0: HF [3,2] [[1,2],[3,4],[5,6]] → packed [1,3,5,2,4,6]
        // Expert 1: HF [3,2] [[101,102],[103,104],[105,106]] → packed
        //   [101,103,105,102,104,106]
        assert_eq!(
            gate_vals,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 101.0, 103.0, 105.0, 102.0, 104.0, 106.0]
        );
    }

    #[test]
    fn pack_hf_mixtral_biases_bf16_round_trips() {
        // BF16 happy-path bias pack. The pack copy is pure
        // concatenation (no transpose for 1-D biases), so the BF16
        // bytes flow through unchanged.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![3],
                &[1.0 + e as f32, 2.0 + e as f32, 3.0 + e as f32],
            ));
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![3],
                &[10.0 + e as f32; 3],
            ));
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.w2.bias", e),
                vec![2],
                &[100.0 + e as f32; 2],
            ));
        }

        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("BF16 bias pack ok")
            .expect("biases present");
        assert_eq!(outcome.num_experts, 2);

        let gate = wm.get("moe0.experts.gate.bias").expect("gate.bias present");
        assert_eq!(gate.dtype, WeightDType::BF16);
        assert_eq!(gate.shape, vec![2, 3]);
        let gate_vals = read_bf16(gate);
        assert_eq!(gate_vals, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);

        let down = wm.get("moe0.experts.down.bias").expect("down.bias present");
        assert_eq!(down.dtype, WeightDType::BF16);
        assert_eq!(down.shape, vec![2, 2]);
        assert_eq!(read_bf16(down), vec![100.0, 100.0, 101.0, 101.0]);
    }

    #[test]
    fn pack_hf_mixtral_biases_f16_round_trips() {
        // F16 happy-path bias pack. Lossless for small integers.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w1.bias", e),
                vec![3],
                &[1.0 + e as f32; 3],
            ));
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w3.bias", e),
                vec![3],
                &[10.0 + e as f32; 3],
            ));
            wm.insert(make_f16_entry(
                &format!("moe0.experts.{}.w2.bias", e),
                vec![2],
                &[100.0 + e as f32; 2],
            ));
        }
        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("F16 bias pack ok")
            .expect("biases present");
        assert_eq!(outcome.num_experts, 2);
        let up = wm.get("moe0.experts.up.bias").expect("up.bias present");
        assert_eq!(up.dtype, WeightDType::F16);
        assert_eq!(read_f16(up), vec![10.0, 10.0, 10.0, 11.0, 11.0, 11.0]);
    }

    #[test]
    fn pack_all_detected_hf_mixtral_blocks_bf16_orchestrator_end_to_end() {
        // End-to-end through the orchestrator with BF16 weights AND
        // BF16 biases. Pins that the entire HF→NSL auto-pack pipeline
        // is dtype-clean at the BF16 width — what real Mixtral
        // checkpoints actually ship.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w1.weight", e),
                vec![3, 2],
                &[1.0 + (e as f32) * 10.0; 6],
            ));
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w3.weight", e),
                vec![3, 2],
                &[2.0 + (e as f32) * 10.0; 6],
            ));
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w2.weight", e),
                vec![2, 3],
                &[3.0 + (e as f32) * 10.0; 6],
            ));
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w1.bias", e),
                vec![3],
                &[4.0 + (e as f32) * 10.0; 3],
            ));
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w3.bias", e),
                vec![3],
                &[5.0 + (e as f32) * 10.0; 3],
            ));
            wm.insert(make_bf16_entry(
                &format!("Block.experts.{}.w2.bias", e),
                vec![2],
                &[6.0 + (e as f32) * 10.0; 2],
            ));
        }

        let outcome = pack_all_detected_hf_mixtral_blocks(&mut wm);
        assert_eq!(outcome.packed.len(), 1);
        assert_eq!(outcome.bias_packed.len(), 1);
        assert!(outcome.failed.is_empty());
        assert!(outcome.bias_failed.is_empty());

        // All 6 packed entries exist with BF16 dtype.
        for nsl_name in [
            "Block.experts.gate.weight",
            "Block.experts.up.weight",
            "Block.experts.down.weight",
            "Block.experts.gate.bias",
            "Block.experts.up.bias",
            "Block.experts.down.bias",
        ] {
            let entry = wm.get(nsl_name).unwrap_or_else(|| panic!("{} missing", nsl_name));
            assert_eq!(
                entry.dtype,
                WeightDType::BF16,
                "{} must preserve BF16 dtype",
                nsl_name,
            );
        }
    }

    #[test]
    fn pack_hf_mixtral_biases_short_form_bf16_round_trips() {
        // v2.16-A composition with v2.17: short `.b{N}` form ALSO
        // works at BF16. The pack should produce a BF16 packed entry
        // identical to the canonical-form BF16 case.
        let mut wm = WeightMap::new_for_test();
        for e in 0..2 {
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.b1", e),
                vec![3],
                &[1.0 + e as f32; 3],
            ));
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.b3", e),
                vec![3],
                &[10.0 + e as f32; 3],
            ));
            wm.insert(make_bf16_entry(
                &format!("moe0.experts.{}.b2", e),
                vec![2],
                &[100.0 + e as f32; 2],
            ));
        }
        let outcome = pack_hf_mixtral_biases(&mut wm, "moe0", "moe0", 2)
            .expect("short-form BF16 bias pack ok")
            .expect("biases present");
        assert_eq!(outcome.num_experts, 2);
        let gate = wm.get("moe0.experts.gate.bias").expect("gate.bias present");
        assert_eq!(gate.dtype, WeightDType::BF16);
        assert_eq!(read_bf16(gate), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
}
