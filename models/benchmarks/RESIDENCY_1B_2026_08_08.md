# The 1B residency result (2026-08-08)

RTX PRO 4500 Blackwell **32 GB**, `matrix_bench.py`, arms interleaved
round-robin over 3 rounds, timing pass uninstrumented. Same harness and same
`1b @ seq 2048, microbatch 1` cell as
[MATRIX_2026_08_06.md](MATRIX_2026_08_06.md) — this document does not replace
that one, it corrects one fact in it.

## The fact this replaces

MATRIX_2026_08_06 recorded 1B@2048 at **901.9 ms/step, 2,271 tok/s, 17.7%
MFU**, and noted in passing that the layerwise arms "stream ~3.7 GiB/step
H2D — weight streaming's cost, and the reason it fits".

The second half of that sentence is wrong. Weight streaming is not why 1B
fits on this card, and the 3.7 GiB/step was buying almost nothing.

## Measured

The comparison that judges the policy is **`layerwise` vs `layerwise_policy`**:
identical flags, `NSL_WS_RESIDENT` off vs on. `layerwise_resident` drops
`--weight-stream` entirely and is the reference for what residency is worth
intrinsically — it does **not** exercise the pinned code path.

| arm | flags beyond `--source-ad` | policy | ms/step | tokens/s | MFU% | peak alloc MB |
|---|---|---|---|---|---|---|
| `layerwise` | `--checkpoint-blocks --layerwise-accum --weight-stream` | off | 912.0 | 2,246 | 17.5 (tf32) | 12,304 |
| **`layerwise_policy`** | **same flags** | **on** | **765.2** | **2,676** | — | — |
| `layerwise_resident` | `--checkpoint-blocks --layerwise-accum` | n/a | 768.9 | 2,663 | 20.8 (tf32) | 13,239 |
| `layerwise_resident_srbf16` | `… --weight-stream --param-dtype bf16-sr` | n/a² | 775.9 | 2,639 | 9.0 (bf16)¹ | 12,304 |

Per-round, to show the gaps are not noise. The two policy states were
measured in separate processes (the policy is a code change, not a flag the
old binary understood), so compare medians across runs and within-round
inside each run:

- policy **off**: 914.6 / 912.0 / 916.0 → median 912.0
- policy **on**: 765.2 / 766.0 / 743.3 → median 765.2
- flag absent, same run as policy-on: 787.1 / 764.8 / 732.4

Within-arm spread is under 1% in the policy-off run; the policy-on run drifts
downward across rounds (all arms fastest in round 3), which is why the
harness interleaves and why `layerwise_policy` ≈ `layerwise_resident`
*within* each round is the load-bearing observation: **765.2 vs 764.8 in
round 2**, i.e. the pinned path costs nothing measurable over not passing the
flag at all. Reproducing this table needs a binary with the policy; the
`layerwise` arm now pins `NSL_WS_RESIDENT=0` so the streaming baseline stays
obtainable from the harness on any binary.

¹ MFU is quoted against the arm's own roofline, so the bf16 figure has a
different denominator and is **not** comparable to the tf32 ones. Compare
`tokens/s`.

² bf16-sr dispatches to its own residency backend *before* the policy is
consulted, so the policy neither helps nor hinders it — its mirrors were
already device-resident.

## Direct confirmation at 1B (not inferred from timing)

Re-running the built `layerwise` arm — `--weight-stream` still passed — with
`NSL_WS_COUNTER=1`:

```
[weight-stream] uploads: 0 evicts: 0 writeback: 0 registered: 144 ptr_moves: 0 ...
[weight-stream] residency: pinned=144 of 144 param(s) pinned_mib=3712 streamed_mib=0
```

**3,712 MiB is the ~3.7 GiB/step the 2026-08-06 matrix measured**: the
streamed set was the whole parameter surface, re-uploaded every step. All 144
parameters now pin, and uploads / evicts / writebacks / pointer-moves are all
zero. The run's `[gpu-mem]` peak reads `alloc=13239MB` — the *resident* arm's
footprint, not the streaming arm's 12,304 MB, which is the same fact from the
memory side.

(That capture predates the fix for the blank audit tail, which is why the
`free_at_decision`/`reserve`/`must_free` fields are missing from the line
above — see the CHANGELOG. Current builds print them.)

## What it establishes

- **Not streaming is worth 15.7% of step time — +18.6% tokens/s** (2,246 →
  2,663) and takes MFU from 17.5% to 20.8%, the highest in the matrix.
- **Streaming was buying under 1 GiB of peak device memory** (12,304 →
  13,239 MB alloc, +935 MB). It is not what makes 1B fit. The peak is set by
  the CSLA *window backward*, which re-uploads and holds most of θ anyway;
  streaming only thins the forward segments, and pays 3.7 GiB/step of PCIe
  for the privilege. Peak driver memory resident is 14.7 GB of a **32.6 GB**
  card — less than half, with ~18 GB unused while the old configuration
  moved the whole parameter set across PCIe every step.
- **The streaming stack was correct for the hardware it was designed on.**
  `models/coder1b/pretrain_layerwise_fit.nsl`'s header records the 16 GiB
  experiments in detail: weights 3.9 + m/v 7.7 GiB left nothing for
  activations, and D2b weight streaming is what made 1B train at all. That
  reasoning did not survive the card change, and nothing in the compiler
  noticed, because streaming was an unconditional consequence of the flag
  rather than a decision (`ws_active = compile_options.weight_stream`; when
  set, every streamable parameter streams — there was no capacity model, no
  device query, nowhere that "does it fit?" was asked).
- **bf16-sr does not pay for itself at 1B once residency is equalized.**
  This is the first controlled comparison of it: `layerwise_resident` and
  `layerwise_resident_srbf16` run the same CSLA schedule at the same
  `grad_accumulation=2`, both fully device-resident, differing only in
  parameter dtype — and bf16-sr is ~1% slower on tokens/s (2,639 vs 2,663)
  while halving the weight surface. The 500M ladder in MATRIX_2026_08_06
  (fp32 405 → srbf16 239 ms) is **not** evidence against this: its `fp32`
  and `bf16` arms run `--source-ad` alone, with no CSLA schedule and no
  accumulation, so that comparison measured the schedule, exactly as that
  document's own note says. bf16-sr remains the right tool when the weight
  surface is the binding constraint; it is not a throughput win by itself.

## The fix

Residency is now a decision instead of a consequence — see the
`[weight-stream] residency:` line on any streaming run, and the CHANGELOG
entry for the policy. The modelling point: at registration every parameter
is *already* device-resident, so streaming does not acquire memory, it
**frees** θ and buys it back over PCIe every step. The question is therefore
"how many bytes must be surrendered so the activation working set still
fits", not "can I afford to keep this". Parameters stream until that debt is
paid; the rest stay resident with no host mirror at all.

Reproduce with:

```
python3 models/benchmarks/matrix_bench.py --cells 1b@2048 --rounds 3
```

Raw data: `target/residency_logs/matrix_results.json`.
