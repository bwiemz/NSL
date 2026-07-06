//! CFIE — persistent decode kernel + GPU-side continuous-batching
//! scheduler.
//!
//! Paper §6 describes a single kernel that, once launched, retains
//! the SMs and loops internally until `<EOS>` — one kernel launch per
//! layer per decode step instead of 15-30.  Paper §6 assumes a static
//! batch; Gemini's review insists (correctly) that a production
//! inference engine must accept new requests mid-stream.
//!
//! This module PLANS the missing piece (the runtime that executes it
//! lives in `nsl-runtime::cfie`):
//!
//!   * a **request ring buffer** the host pushes new requests into (the
//!     runtime ships a lock-free SPSC ring, though the shipped consumer
//!     is still host-side — see `nsl-runtime::cfie`),
//!   * a **scheduler descriptor** that, at the top of every decode loop
//!     iteration, is meant to integrate pending requests into the active
//!     batch (GPU-side on-device dequeue is the intended endpoint),
//!   * a **feasibility analysis** that decides which CSHA fusion
//!     level the persistent kernel can sustain on the target GPU.

use serde::Serialize;

/// Which CSHA fusion level the persistent kernel is running at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FusionLevel {
    /// No intra-block fusion — one kernel per op (baseline).
    None,
    /// Norm prologue + RoPE epilogue folded into projections.
    Level1,
    /// Projection pipelined with attention (producer/consumer warps).
    Level2,
    /// Full block fusion: attention + FFN in one persistent kernel.
    Level3,
}

impl FusionLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            FusionLevel::None => "none",
            FusionLevel::Level1 => "level1",
            FusionLevel::Level2 => "level2",
            FusionLevel::Level3 => "level3",
        }
    }
}

/// Fixed attributes of the target GPU the planner needs.
#[derive(Debug, Clone, Copy)]
pub struct GpuBudget {
    /// Per-SM SMEM budget in bytes (H100: 228 KB, A100: 164 KB).
    pub smem_per_sm: u32,
    /// Number of SMs on the GPU (H100: 132, A100: 108).
    pub num_sms: u32,
    /// Kernel launch overhead in microseconds — used to quantify the
    /// savings of the persistent-kernel model.
    pub kernel_launch_us: f64,
}

impl Default for GpuBudget {
    fn default() -> Self {
        Self {
            smem_per_sm: 228 * 1024,
            num_sms: 132,
            kernel_launch_us: 5.0,
        }
    }
}

/// Build the planner budget from a `gpu_specs` database entry — the
/// single place the GpuSpec -> GpuBudget field mapping lives (audit
/// gap G17: hand-built budgets in the serve wiring drifted from the
/// database).  `l1_cache_kb` is the unified L1/SMEM carveout per SM.
pub fn gpu_budget_from_spec(spec: &crate::gpu_specs::GpuSpec) -> GpuBudget {
    GpuBudget {
        smem_per_sm: spec.l1_cache_kb * 1024,
        num_sms: spec.num_sms,
        kernel_launch_us: spec.kernel_launch_overhead_ns as f64 / 1000.0,
    }
}

/// Model shape the persistent kernel wraps.
#[derive(Debug, Clone, Copy)]
pub struct PersistentModel {
    pub d_model: u32,
    pub head_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_ff: u32,
    pub dtype_bytes: u32,
}

/// Host → device ring-buffer descriptor (the request queue; the runtime
/// ring is a lock-free SPSC structure, popped host-side today).
#[derive(Debug, Clone, Serialize)]
pub struct RingBuffer {
    /// Capacity in slots.  Must be a power of two so the GPU-side
    /// scheduler can mask the index without a modulo.
    pub capacity: u32,
    /// Bytes per slot (pointer + sequence id + length + metadata).
    pub slot_bytes: u32,
    /// CPU-side pinned-memory footprint in bytes.
    pub pinned_bytes: u64,
}

/// Scheduler metadata baked into the persistent kernel.
#[derive(Debug, Clone, Serialize)]
pub struct Scheduler {
    pub ring_buffer: RingBuffer,
    /// Maximum active sequences the persistent kernel tracks.
    pub max_active: u32,
    /// Bytes reserved in SMEM for the per-CTA scheduler state (head /
    /// tail pointers, per-slot status flags).
    pub smem_scheduler_bytes: u32,
    /// True when the scheduler dequeues requests at the top of every
    /// decode loop iteration.  Always `true` for CFIE; retained so
    /// future variants (e.g., warmup-only ingestion) are expressible.
    pub continuous_batching: bool,
}

/// Complete persistent-kernel plan.
#[derive(Debug, Clone, Serialize)]
pub struct PersistentPlan {
    pub fusion: FusionLevel,
    pub smem_used_bytes: u32,
    pub smem_budget_bytes: u32,
    pub scheduler: Scheduler,
    /// Per-layer kernel launches the non-persistent baseline would
    /// have used (N norm + Q/K/V + RoPE + attention + O + FFN gate/up/
    /// down + 2× residual adds ≈ 12).
    pub baseline_launches_per_layer: u32,
    /// Launches the persistent kernel issues per layer.
    pub persistent_launches_per_layer: u32,
    /// Microseconds saved per decode step vs the baseline.
    pub us_saved_per_step: f64,
    /// Rationale string for the report.
    pub rationale: String,
}

impl PersistentPlan {
    pub fn launch_reduction(&self) -> f64 {
        if self.baseline_launches_per_layer == 0 {
            return 0.0;
        }
        1.0 - (self.persistent_launches_per_layer as f64
            / self.baseline_launches_per_layer as f64)
    }

    /// Bytes fitting ratio — useful when debugging SMEM budgets.
    pub fn smem_utilization(&self) -> f64 {
        if self.smem_budget_bytes == 0 {
            return 0.0;
        }
        self.smem_used_bytes as f64 / self.smem_budget_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// SMEM modelling
// ---------------------------------------------------------------------------

fn l1_smem_bytes(model: &PersistentModel) -> u32 {
    // Level 1: one norm scratchpad + one tile of Q/K/V.
    let norm = 4 * 1024;
    let tile = 64 * model.head_dim * model.dtype_bytes;
    (norm + tile)
}

fn l2_smem_bytes(model: &PersistentModel) -> u32 {
    // Level 2: producer + consumer tiles + partial weight tile.
    let tile = 64 * model.head_dim * model.dtype_bytes;
    let weight_slice = model.head_dim * model.d_model * model.dtype_bytes;
    (2 * tile + weight_slice)
}

/// FFN staging tile width for the Level-3 residency model — matches
/// the decode-block kernel's per-tile silu(gate)*up chunk.
const L3_FFN_TILE: u32 = 128;

fn l3_smem_bytes(model: &PersistentModel) -> u32 {
    // Level 3: full block fusion for single-token decode.  Paper §6's
    // worked example: d_model=4096 (7B models) needs ~128 KB — feasible
    // on H100 (228 KB/SM) but not on older GPUs.  The audit flagged the
    // old formula for ignoring d_model entirely; this models the actual
    // Level-3 residency of the ONE-CTA-per-layer decode block:
    //   * x + x_normed residual-stream buffers: 2 * d_model * 4 (f32),
    //   * attention workspace, one concurrent head (the single CTA
    //     walks Q heads sequentially): the decode-attention SMEM shape
    //     head_dim*4 (q row) + 512 (score tile) + 8 (rescale + l),
    //   * q + attn_out staging for ALL heads (kept resident across the
    //     sequential head walk): 2 * n_heads * head_dim * 4 — the
    //     decode-block kernel allocates exactly this; omitting it made
    //     the planner claim Level 3 for shapes whose emitted kernel
    //     exceeds the serve wiring's 48 KB static-.shared gate
    //     (adversarial-review finding),
    //   * FFN staging: gate/up tiles of width min(T, d_ff) at T=128
    //     (2 * tile * 4) + the d_model-wide down-projection accumulator
    //     (d_model * 4).
    // Both d_model (3x, dominant) and d_ff (up to the tile cap) scale
    // the footprint.  Note this can undercut l1/l2, which stage 64-row
    // batching tiles / pipelined weight slices — the deepest-first
    // cascade in `choose_fusion` still picks the cheapest-deepest level.
    let x_buffers = 2 * model.d_model * 4;
    let attn_workspace = model.head_dim * 4 + 512 + 8;
    let qo_buffers = 2 * model.n_heads * model.head_dim * 4;
    let ffn_tile = model.d_ff.min(L3_FFN_TILE);
    let ffn_staging = 2 * ffn_tile * 4 + model.d_model * 4;
    x_buffers + attn_workspace + qo_buffers + ffn_staging
}

fn scheduler_smem_bytes(max_active: u32) -> u32 {
    // 16 bytes/slot overhead (2× u32 pointers + 1× u64 flags).
    16 * max_active.max(1) + 32 /* head/tail/signal */
}

fn next_pow2(n: u32) -> u32 {
    let mut x = 1u32;
    while x < n && x < u32::MAX / 2 {
        x <<= 1;
    }
    x
}

/// Pick the deepest fusion level that fits in the per-SM SMEM budget
/// alongside the scheduler overhead.
pub fn choose_fusion(model: &PersistentModel, budget: &GpuBudget, max_active: u32) -> FusionLevel {
    let sched = scheduler_smem_bytes(max_active);
    let available = budget.smem_per_sm.saturating_sub(sched);
    if l3_smem_bytes(model) <= available {
        FusionLevel::Level3
    } else if l2_smem_bytes(model) <= available {
        FusionLevel::Level2
    } else if l1_smem_bytes(model) <= available {
        FusionLevel::Level1
    } else {
        FusionLevel::None
    }
}

/// Build the persistent-kernel plan + the GPU-side scheduler.
pub fn plan(model: &PersistentModel, budget: &GpuBudget, max_active: u32) -> PersistentPlan {
    let max_active = max_active.max(1);
    let ring_capacity = next_pow2(max_active).max(16);
    let slot_bytes = 64; // pointer + id + len + pad
    let ring = RingBuffer {
        capacity: ring_capacity,
        slot_bytes,
        pinned_bytes: (ring_capacity as u64) * (slot_bytes as u64),
    };
    let sched_smem = scheduler_smem_bytes(max_active);
    let scheduler = Scheduler {
        ring_buffer: ring,
        max_active,
        smem_scheduler_bytes: sched_smem,
        continuous_batching: true,
    };

    let fusion = choose_fusion(model, budget, max_active);
    let smem_used = match fusion {
        FusionLevel::None => 0,
        FusionLevel::Level1 => l1_smem_bytes(model),
        FusionLevel::Level2 => l2_smem_bytes(model),
        FusionLevel::Level3 => l3_smem_bytes(model),
    } + sched_smem;

    // Kernel launch accounting.
    let baseline = 12u32; // see doc-comment
    let persistent = match fusion {
        FusionLevel::None => baseline,
        FusionLevel::Level1 => 6,
        FusionLevel::Level2 => 2,
        FusionLevel::Level3 => 1,
    };
    let saved_per_layer = baseline.saturating_sub(persistent) as f64 * budget.kernel_launch_us;
    let saved_per_step = saved_per_layer * model.n_layers as f64;

    let rationale = format!(
        "fusion={} — {} launches/layer vs baseline {}, scheduler reserves {} B SMEM for {} active sequences",
        fusion.as_str(),
        persistent,
        baseline,
        sched_smem,
        max_active
    );

    PersistentPlan {
        fusion,
        smem_used_bytes: smem_used,
        smem_budget_bytes: budget.smem_per_sm,
        scheduler,
        baseline_launches_per_layer: baseline,
        persistent_launches_per_layer: persistent,
        us_saved_per_step: saved_per_step,
        rationale,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn nslcoder_model() -> PersistentModel {
        PersistentModel {
            d_model: 512,
            head_dim: 64,
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 4,
            d_ff: 1408,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn h100_fits_level3_for_small_model() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 32);
        assert_eq!(p.fusion, FusionLevel::Level3);
        assert!(p.persistent_launches_per_layer < p.baseline_launches_per_layer);
    }

    #[test]
    fn tiny_smem_falls_back_to_none() {
        let budget = GpuBudget {
            smem_per_sm: 512, // absurdly small
            num_sms: 1,
            kernel_launch_us: 5.0,
        };
        let p = plan(&nslcoder_model(), &budget, 4);
        assert_eq!(p.fusion, FusionLevel::None);
        assert_eq!(
            p.persistent_launches_per_layer,
            p.baseline_launches_per_layer
        );
    }

    #[test]
    fn ring_capacity_is_power_of_two() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 33);
        assert!(p.scheduler.ring_buffer.capacity.is_power_of_two());
        assert!(p.scheduler.ring_buffer.capacity >= 64);
    }

    #[test]
    fn continuous_batching_is_always_enabled() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 16);
        assert!(p.scheduler.continuous_batching);
    }

    #[test]
    fn launch_reduction_improves_with_higher_fusion() {
        // The Level-3 residency model (single-token decode block) can
        // undercut Level 1's 64-row batching tile for small d_model, so
        // forcing a Level-1 pick needs a model where l3 > l1: d_model
        // dominates l3 (3 * d_model * 4) but never enters l1.
        let mut big = nslcoder_model();
        big.d_model = 8192; // l3 ~= 100 KB; l1 stays 12 KB
        assert!(l3_smem_bytes(&big) > l1_smem_bytes(&big));

        let l1_budget = GpuBudget {
            smem_per_sm: l1_smem_bytes(&big) + scheduler_smem_bytes(8) + 1_000,
            num_sms: 64,
            kernel_launch_us: 5.0,
        };
        let p1 = plan(&big, &l1_budget, 8);
        assert_eq!(p1.fusion, FusionLevel::Level1);

        let l3_budget = GpuBudget::default();
        let p3 = plan(&big, &l3_budget, 8);
        assert_eq!(p3.fusion, FusionLevel::Level3);
        assert!(p3.launch_reduction() > p1.launch_reduction());
    }

    #[test]
    fn l3_smem_scales_with_d_model_and_d_ff() {
        // The audit's gap-#G17 complaint: the old formula ignored
        // d_model.  Both axes must move the footprint.
        let base = nslcoder_model();
        let mut wider = base;
        wider.d_model = base.d_model * 2;
        assert!(l3_smem_bytes(&wider) > l3_smem_bytes(&base));

        let mut tiny_ff = base;
        tiny_ff.d_ff = 32; // below the 128-wide staging tile
        assert!(l3_smem_bytes(&tiny_ff) < l3_smem_bytes(&base));
    }

    #[test]
    fn gpu_budget_from_spec_maps_database_fields() {
        let h100 = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        let b = gpu_budget_from_spec(h100);
        assert_eq!(b.smem_per_sm, 256 * 1024);
        assert_eq!(b.num_sms, 132);
        assert!((b.kernel_launch_us - 5.0).abs() < 1e-9);
    }

    #[test]
    fn saved_microseconds_scale_with_layers() {
        let mut model = nslcoder_model();
        model.n_layers = 32;
        let p32 = plan(&model, &GpuBudget::default(), 16);
        model.n_layers = 8;
        let p8 = plan(&model, &GpuBudget::default(), 16);
        assert!(p32.us_saved_per_step > p8.us_saved_per_step);
    }

    #[test]
    fn rationale_contains_fusion_level() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 16);
        assert!(p.rationale.contains(p.fusion.as_str()));
    }

    #[test]
    fn scheduler_reserves_smem() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 64);
        assert!(p.scheduler.smem_scheduler_bytes > 0);
    }

    #[test]
    fn smem_utilization_in_range() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 16);
        let util = p.smem_utilization();
        assert!(util >= 0.0 && util <= 1.0);
    }

    #[test]
    fn zero_active_is_clamped_to_one() {
        let p = plan(&nslcoder_model(), &GpuBudget::default(), 0);
        assert!(p.scheduler.max_active >= 1);
    }
}
