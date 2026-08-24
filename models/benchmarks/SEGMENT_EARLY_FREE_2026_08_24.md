# Item 11 — per-segment forward early-free

**Hardware:** RTX PRO 4500 Blackwell (32 GB, 31.39 usable), CUDA 13.x.
**Probe:** `models/coder1b/mem_probe_32step.nsl` — reproduces the full-epoch
production allocator peak **to the byte** (22,073,520,640 B, measured against
`PROD1B_VALIDATION_2026_08_19.md` EC1), so every number below is a
minutes-scale measurement of the production peak.

## EC1 — `--checkpoint-blocks` never reduced the FORWARD peak

CCR classifies each block segment's interiors as recompute victims — and
freed them in ONE `FreeTensor` list lowered **after the whole forward**
(`build_early_free_list`, stmt.rs). At end-of-forward, where the 1B global
peak sits, every segment's interiors were still live. Checkpointing was
reducing the backward's activation wall and never the forward's.

The fix lowers one mini-list per segment at that segment's end, via the same
sliced `compile_wengert_ops_range` emission the weight-streaming forward
already uses. Eligibility is unchanged from the post-forward list: interior
means no primal consumer at or past `segment_end`, the adjoint reads
recompute clones, and the force-saved classes never enter
`per_segment_recompute`. The free moves earlier; nothing readable changes.

## EC2 — measured at 1B

| configuration | PEAK_BYTES | PEAK_ACTIVATIONS | outcome |
| --- | --- | --- | --- |
| offload, pre-fix | 22,073,520,640 (20.56 GiB) | 17,761,423,872 (16.54 GiB) | completes |
| offload, **fix** | **8,242,689,024 (7.68 GiB)** | **3,930,592,256 (3.66 GiB)** | completes |
| resident, pre-fix | — | — | **OOM in forward** (32 MB request, 122 MB free) |
| resident, **fix** | **21,166,157,824 (19.71 GiB)** | 3,968,348,160 (3.70 GiB) | **completes** |

Witness line both fixed runs: `[ccr] per-segment early-free: 928 interior
value(s) freed across 16 segment(s)`. "Resident" = no `--optim-state-offload`:
f32 AdamW m/v/m_partial all on-device (4.00 GiB each, non-zero PEAK_OPTIM_*
as the witness), weights 4.02 GiB, two-phase grad clip planned. The retry the
roadmap item asked for — fully resident production 1B with grad clipping and
f32 optimizer state — goes from OOM to ~11.7 GiB of headroom.

The item's revised target was ~3–4 GiB off the activation peak; measured is
**12.88 GiB** (16.54 → 3.66).

## EC3 — value-neutrality, proven not argued

* 50M `det_train_check` under `--deterministic --checkpoint-blocks`, one
  binary, kill switch `NSL_CCR_SEGMENT_FREE=0` vs default: loss streams
  **bit-identical**; witness 464 interiors / 8 segments on the default arm,
  absent on the kill-switch arm.
* `ccr_checkpoint_parity`: CPU and GPU arms green, including the
  `--deterministic` GPU arm and the production-tolerance arm.
* `ccr_periodic_stride_gate` 5/5 — bit-exact across strides, covering the
  kept-anchors (coalesced-segment) path through the new emission.
* `ccr_checkpoint_activation_parity` green.

## EC4 — what is NOT gated, and why (read before trusting coverage)

`ccr_segment_early_free_gate` pins the witness and kill-switch bit-parity. It
does NOT carry a GPU memory assertion: three fixture attempts each tripped a
different pre-existing runtime quirk before the feature was exercised (host-
resident input ⇒ CPU gradients on a GPU run; `zeros().to(cuda)` loss target ⇒
`data_f64()` abort; 2D-logits cross_entropy ⇒ `nsl_tensor_compare` abort —
all reproduce with the kill switch set; see the bug ledger). The memory
number is pinned by THIS record + the committed probe, and item 12's
re-profile re-measures it. A synthetic memory fixture on the production
input pattern is welcome future work; it was not going to be hand-debugged
into existence through a fourth runtime quirk here.

## EC5 — the debugging detour, recorded because it cost an hour

A "hang" reproduced 4-of-4 in the feature × `--deterministic` cell and
vanished under instrumentation. It was never real: the gate binary
(`CARGO_BIN_EXE_nsl`) is a DEBUG build, the first fixture's host-resident
input put part of the backward on one host thread (quirk 1 above), and the
"reproductions" ran concurrently with 19–50-minute CPU-saturating test jobs
started by the same investigation. Uncontended release runs of every arm
complete in seconds. Two rules restated: time it in RELEASE before calling it
a hang, and check what else the machine is running before believing 4-of-4.

## Composition boundaries

Weight-streaming keeps its own (post-forward) free discipline; CSLA is
excluded (its retention contract is the bulk-free's). Both documented at the
plan-construction site; both fall back to exactly the pre-fix behaviour.
