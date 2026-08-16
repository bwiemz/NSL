# Milestone B — 1B@2048 as a stable first-class single-GPU workload (2026-08-16)

RTX PRO 4500 Blackwell 32 GB (31.39 GiB per `cuMemGetInfo`), CUDA 13.x.
Canonical workload: `models/coder1b/pretrain_1b2048.nsl` — microbatch 2,
seq 2048, grad_accumulation 2 (8,192 tokens per optimizer step), AdamW
lr 3e-4. Every number below was produced and ASSERTED by
`models/benchmarks/endurance_1b.py` in one invocation; raw streams in
`b2048_trajectories_2026_08_16.json`, full artifacts under
`target/b2048_endurance/` (per-phase stdout/stderr, `ws_decision.json`).

## Exit criteria, each with its witness

| criterion | result | witness |
|---|---|---|
| fused LM-CE always proven active | **PASS** | `--fuse-lm-head require` (unfusable = compile error) + `[lm-head-fusion] inferred: vocab=49152 hidden=2048` + `[csla] fused-ce tape-carry: 1 slots` at compile; **244 fwd + 244 bwd** fused-kernel launches counted at runtime (`fused_lce_launch_count`) |
| microbatch 2 with several GiB margin | **PASS** | peak allocated **27.44 GiB** of 31.39 → **3.95 GiB** allocator margin (driver-level lower bound 1.5–2.6 GiB depending on ambient desktop). Was 28.94 GiB / ~2.4 GiB before this campaign |
| resident-weight policy makes a modeled decision | **PASS** | machine-readable record (`NSL_WS_DECISION_JSON`): pin-all — 144/144 params, reserve 15.7 GiB, free-at-decision 26.0 GiB, must_free 0 — validated per run against the measured at-peak transient need (17.1 GiB ≤ 26.0). On the SR-BF16 arm the policy is BYPASSED by design (bf16-sr owns residency); the harness asserts the bypass and the SR posture instead |
| true peak allocations attributable | **PASS** | per-surface at-global-peak decomposition (sums to the peak): weights 2.43 + optim_m 4.00 + optim_v 4.00 + activations 17.01 GiB — plus the NEW per-op "Top contexts at peak" table (matmul 5.75, add 4.26, mul 3.25, silu 2.00, rmsnorm 1.03 GiB, `srbf16_widen_view` 232 MB self-named) |
| no unexpected PCIe weight traffic | **PASS** | the streaming stack's own byte counters (new): **0 B** h2d / 0 B d2h; global steady-state PCIe **1.7 MiB/step** max over 242 intervals (the DataLoader batch) |
| reference F32 + SR-BF16 trajectories banked | **PASS** | 244 micro-batch losses each, same schedule/tokens. F32 (`+--optim-state-offload`, byte-preserving staging): last-20 mean **8.83**, min 7.78, peak 21.12 GiB. SR-BF16: last-20 mean **9.62**, min 7.87. The gap is a real observation banked for the record — SR storage rounding at a 122-step horizon on the synthetic tiled stream |
| clean checkpoint/restart | **PASS** | `train(checkpoint_save=…, checkpoint_every=50)` → θ `.nslm` + `NSLO` `.optim` sidecar (m/v + step counter), paired atomic renames + model-signature pairing check. Resume: **losses printed-precision identical to the uninterrupted parent for the first 6 micro-batches** (rel Δ 0.0), max 3.1e-3 after 44 (TF32/atomics accumulation). CPU gate `train_checkpoint_gate.rs` proves byte-identical resume where the environment is deterministic |
| 100+ step endurance, flat VRAM high-water | **PASS** | **122 optimizer steps** (244 micro-batches); `gpu_peak_bytes()` stream flat at 29,466,117,120 B from micro-batch 73 through 244 — byte equality, not a trend |

## What moved the margin (28.94 → 27.44 GiB)

**Dropout p=0 elision (−1.50 GiB), which is also a semantics fix.** The 1B
model configures dropout 0.0, but every call site passes
`self._dropout_p.item()` — a non-literal the source-AD extractor could not
resolve, so its `.unwrap_or(0.1)` default ran REAL p=0.1 dropout: output +
mask per call, force-saved (RNG is not replayable), multiplied per
micro-batch by the CSLA window. The resolver now follows the ctor-folded
config scalar through the `.item()` passthrough (value-preserving
whitelist — a constructor's `inputs[0]` is its shape list, so a naive walk
reads p=1.0) and elides the call at p=0. `dropout_f32` vanished from the
peak table.

`--fuse-rmsnorm-backward` is in the canonical arm (launch/backward win;
peak unchanged — the peak is end-of-forward). `--checkpoint-stride 2` was
measured and REJECTED for the canonical arm (peak +0.66 GiB at mb2).

## Defect surfaced, deferred deliberately

Source-AD dropout backward multiplies the upstream gradient by the **p
argument**, not the runtime RNG mask (never tape-carried) — every
non-elided dropout under `--source-ad` trains with gradients decorrelated
from the forward mask. Pre-existing; hidden for p=0 models by the elision.
coder50m/coder500m ship `DROPOUT = 0.1`, so a refusal would red-line every
pipeline over them — the lowering now WARNS once, loudly, and the fix
(tape-carry the mask, or flip the configs to 0.0 with re-baselines) is its
own campaign.

## Reproduce

```
cargo build --release --features cuda --bin nsl
python3 models/benchmarks/endurance_1b.py \
  --nsl target/release/nsl \
  --arm layerwise_srbf16_fusedce_rn --f32-arm layerwise_f32_fusedce_rn_oso
```

An OOM or any failed criterion is a hard exit; `ALL MILESTONE B ASSERTIONS
PASSED` is the green line.
