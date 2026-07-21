# Post-#405 1B pipeline benchmark (P0 item 3)

Cumulative-arm measurement of the streaming/overlap stack that #403/#404/#405
landed, on the **committed** 1B program (`models/coder1b/pretrain_cert.nsl`,
16-layer d2048 GQA, ~1.07B params) at the demo config: batch=2, seq=512,
grad_accum=8, **4 optimizer windows (32 micro-batches, 32,768 tokens)**,
`--deterministic`, synthetic learnable token stream. RTX 5070 Ti 16 GiB,
release build, 2026-07-20. Runner: `models/benchmarks/p0_campaign.py bench`;
raw logs + per-arm `result.json` under `models/benchmarks/p0_logs/`.

Base flags (every arm): `--source-ad --checkpoint-blocks --layerwise-accum
--optim-state-offload --weight-stream`.

| arm | + flags | micro-batches | train wall s | tok/s | alloc peak GB | smi peak GB | util % | compile+init s | uploads/evicts | pack up/ev | prefetch | async_wb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A baseline stream | — | 32 | 359 | **91** | **9.10** | 15.45 | 85 | 0.5+26 | 5184/5184 | 0/0 | 0 | 0 |
| B + arena | `--stream-arena` | 32 | 370 | 88 | 9.34 | 15.41 | 84 | 0.5+26 | 5184/5184 | 576/576 | 0 | 0 |
| C + prefetch | `--stream-prefetch` | 32 | 369 | 89 | 9.59 | 15.29 | 84 | 0.5+26 | 5184/5184 | 576/576 | 64 | 0 |
| D + async writeback | `--stream-async-writeback` | 32 | 367 | 89 | 10.08 | 15.33 | 84 | 0.5+26 | 5184/5184 | 576/576 | 64 | 64 |
| E + checkpoint-stride | `--checkpoint-stride auto` | 32 | 368 | 89 | 10.08 | 15.39 | 83 | 0.5+26 | 5184/5184 | 576/576 | 64 | 64 |
| F + fused RMSNorm bwd (all) | `--fuse-rmsnorm-backward` | 32 | 367 | 89 | 10.08 | 15.27 | 84 | 0.5+26 | 5184/5184 | 576/576 | 64 | 64 |

*tok/s = trained tokens / loss-stream wall (compile, init and teardown
excluded). `uploads/evicts` are LOGICAL per-param transfer requests; in
arena mode the actual CUDA transfer calls are the `pack` counters — the
**5184 → 576 (9×) transfer-call reduction** the P3 arenas were built for.*

## Findings

1. **Correctness across the stack**: arms A–E produce **bit-identical**
   loss streams (all 32 micro-batch losses equal to full f32 precision).
   Arm F (fused RMSNorm backward) tracks within its documented f32
   tolerance — first divergence in the 7th decimal, final losses
   10.605582 vs 10.605580 (~2e-7 relative). The whole post-#405 overlap
   stack preserves the training computation.
2. **Throughput is compute-bound at this config — the overlap stack does
   not pay for itself at 1B/batch-2**: baseline streaming is the fastest
   arm (91 tok/s); arenas+prefetch+async-wb land at 88–89 tok/s (−2–3%).
   GPU utilization is 83–85% *in every arm* — PCIe transfers were already
   effectively hidden behind compute at this shape, so cutting transfer
   CALLS 9× (pack counters) buys no wall-clock, while the arena/event
   machinery adds a small constant cost. The P3 gains measured on the
   small FFN/GQA fixtures (transfer-count-dominated) do not transfer to
   this compute-dominated shape. Larger batches (more PCIe pressure per
   window) or PCIe-starved hosts are where B–D should win; re-measure
   there before defaulting the flags on.
3. **Memory cost of the overlap machinery**: allocator peak rises
   9.10 → 10.08 GB across B–D (arena slots +0.24, prefetch double-buffer
   +0.25, async-writeback staging +0.49). nvidia-smi peak stays ~15.3–15.5
   GB in all arms (the card is committed either way).
4. **The checkpoint-stride lever is inert on this stack** (measured, not
   assumed): `auto` falls back to stride 1 at 1B — the model's symbolic
   shapes leave nothing to price ("no static tensor sizes available") —
   and explicit `--checkpoint-stride 2` **refuses** under
   `--layerwise-accum` (SDPA backward side-bands cannot cross replay
   ranges). Arm E ≡ D functionally. Wiring static-shape hints (or
   range-crossing side-bands) is the unlock; until then the stride lever
   only exists off the layerwise path.
5. **Fixed costs**: compile 0.5 s; model init + HtoD + streaming
   registration ≈ 26 s; host-pinned mirrors are by construction the
   streamed weight bytes (144 registered params ≈ 3.86 GB) plus bounded
   staging slots. (The runner's VmHWM column reads the CLI driver
   process, not the trainee child — treat the counter-derived pinned
   figure as authoritative.)
6. **PCIe**: nvidia-smi dmon burst peaks of 18–28 GB/s rx during upload
   phases — near gen4 x16 line rate, confirming transfers saturate the
   link *when they run*; they are simply not on the critical path at this
   shape.

## Relation to the #402 reference numbers

#402's "4.00 GB allocator peak / 5.6 GB driver / 1,728 transfers over four
windows" was measured on the fused-CE scratch variant at a different
config; it is not directly comparable to the committed program measured
here (9.10 GB / 15.4 GB / 5,184 logical transfers over four windows, CE
logits chain included). The comparable statement post-#405: at the demo
config the streaming 1B trains in 9.1–10.1 GB allocator peak with
bit-exact parity across every overlap feature, and the arena stack turns
5,184 transfer calls into 576.
