# Tier B SMEM Probe — Findings

**Probe spec:** §2 of `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
**Sweep:** 3 static-sizes {256, 1024, 4096} × 3 dynamic-sizes {16 KB, 64 KB, 96 KB} × 2 architectures {sm_80, sm_120} = 18 configurations.

## 2026-05-12 — initial run

**Hardware / toolkit at run time:**
- Primary GPU: RTX 5070 Ti (sm_120, Blackwell)
- Secondary GPU (if available): TODO model and sm
- CUDA toolkit: 13.x
- NVIDIA driver: TODO version from nvidia-smi
- Host OS: Windows 11

**Sweep result** (paste from `/c/tmp/tier_b_probe.log` after running on hardware):

```text
sm_<sm>  N=<N>  M=<M>  <outcome>
...
```

**Outcome row selected from spec §2.4 five-outcome decision matrix:** TODO row name

**Decision:** TODO text per the selected row

**Rationale notes:** TODO free-text observations.

## Re-run triggers

The probe is re-run when any of:

- **CUDA toolkit major version bump** (e.g., 13.x → 14.x). Driver behavior on mixed `.shared` allocations may shift.
- **New target architecture added to NSL's supported matrix** (e.g., sm_130 when it ships). Probe sweeps the new architecture.
- **Production deployment surface reports `CUDA_ERROR_ILLEGAL_ADDRESS`** on a Tier B kernel that previously passed CI. Indicates the probe's prior outcome no longer holds.

## Re-run log

(Future dated entries appended here. Each entry includes: date, trigger reason, hardware/toolkit, sweep result, outcome row, any decision changes.)
