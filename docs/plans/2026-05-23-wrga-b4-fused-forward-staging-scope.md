# WRGA B.4 -- Fused-Forward Staging (RESOLVED 2026-05-23: fused stays opt-in, unfused is the default everywhere, staging rewrite NOT pursued)

**Status: RESOLVED 2026-05-23 by measurement. The fused GatedLoRA forward kernel
loses to unfused cuBLAS at every measured m -- there is no regime where it wins,
so there is no niche to optimize the staging for.** The scoping framing below
(options A-D, "is the staging rewrite worth doing") is preserved for the record,
but the answer is: the staging rewrite (option A) is NOT pursued, because even if
it eliminated the lane-0 tax it would only *maybe* reach parity in a regime
(small-m) that measurement shows fused does not actually win.

This supersedes the initial scoping recommendation in this same document (which
leaned "ship D + do A"). The measurement overturned the premise that motivated A.

---

## The measurement that resolved it

`C:\tmp\wrga_msweep2.py` (forward-only fused crashes -- see segfault below -- so
the fused forward kernel was timed via the working train-block path reading
`adapter_fused`; cuBLAS x@W was timed via a no-adapter forward). RTX 5070 Ti,
CUDA 13.2, dim=4096, rank=16:

| m | fused fwd kernel | cuBLAS x@W | fused / cuBLAS |
|---|---|---|---|
| 1 | 1,285 us | 84 us | **15.2x slower** |
| 16 | 3,554 us | 92 us | 38.6x slower |
| 64 | 6,222 us | 97 us | 64.1x slower |
| 256 | 17,828 us | 276 us | 64.5x slower |
| 1024 | 66,586 us | 985 us | 67.6x slower |

The comparison is **unfair in fused's favor**: it pits the fused *full* GatedLoRA
forward (x@W + adapter + gate) against cuBLAS doing **x@W alone**. Even doing less
work, cuBLAS wins 15x at m=1. Against the *full* unfused forward (x@W + two tiny
adapter GEMVs + sigmoid + gate ~= ~300 us at m=1), fused (1,285 us) still loses
~4x. **There is no crossover at any measured m.**

## The overturned premise (why this matters for the record)

The B.3.2 resolution scheduled a hypothetical small-m niche for the fused kernel,
reasoned structurally: "at small-m, fusion's savings (per-matmul cuBLAS launch
overhead + 3-matmul HBM round-trip elimination) dominate, so fusion wins." That
argument was **structurally coherent but rested on an unmeasured quantity: the
fused kernel's own cost.**

The measurement shows the fused kernel costs 1,285 us at m=1 -- an order of
magnitude more than the ~200 us of launch/round-trip savings fusion offers there.
The specific error: the lane-0 staging tax scales with the number of blocks, which
is `m_tiles x n_tiles`. At small-m the *m-tile* count drops, but the *n-tile*
count stays large (n=4096 -> 512 n-blocks even at m=1). So the staging tax does
**not** scale down with m the way "small-m -> few blocks -> cheap" assumed; it
scales with n, which stays large. The fused kernel's own overhead dwarfs the
fusion savings at every m.

This is the same failure mode as the 106x B.3.2 trigger (a confident claim resting
on an unmeasured/broken substrate, overturned by measurement) -- applied one level
up, to the structural argument for the small-m niche itself. Recorded as a
re-proposal gate below.

## Resolution (the decision)

1. **Fused stays opt-in (`NSL_WRGA_FUSED_CUDA=1`); unfused cuBLAS is the default
   everywhere.** There is no crossover, so there is no m-threshold to set -- the
   "D" dispatch stopgap collapses to "always unfused." Just don't put the fused
   path on any default code path.
2. **The staging rewrite (option A) is NOT pursued.** It is a speculative bet: it
   *might* bring m=1 fused from 1,285 us toward ~40 us and finally beat unfused at
   small-m, but that is unproven, requires the full staging rewrite, and the win
   would be marginal against an unfused path that already works at all m. Do not
   invest in it on the strength of a structural argument.
3. **B.3.2 (fused backward) stays deferred** -- and now the fused *forward* does
   not justify itself either, so the entire fused-adapter-kernel approach is
   questionable without a much deeper rewrite (multi-warp + tensor cores), which
   is a different, larger project not justified by any measured need.
4. **Forward-only @adapter segfault is fixed separately** (robustness, stands
   alone -- see below).

## A-gate (the only condition under which the staging rewrite is revisited)

Option A is revisited ONLY if BOTH hold:
- a concrete deployment need for small-m adapter inference emerges with a perf bar
  the unfused cuBLAS path cannot meet, AND
- a prototype of A is built and **re-measured** to actually beat unfused at the
  target m before any further investment.

Never pursue A on a structural argument alone. The structural argument for the
small-m niche has already been measured and falsified once.

## Re-proposal gate (institutional -- read before re-proposing the fused niche)

A future "let's pursue the fused small-m niche / it obviously wins at small-m"
proposal MUST engage the measurement above, not re-derive the structural premise.
The premise ("fusion wins at small-m because launch/round-trip savings dominate")
is **measured false**: the fused kernel's own cost (lane-0 staging x n-block count,
which scales with n not m, so stays large at small-m) dwarfs the savings by ~10x
at m=1. To reopen this, overturn the measurement (e.g., with a different kernel
design -- multi-warp + tensor cores -- prototyped and re-measured), do not
re-derive the Option-1 argument.

---

## The forward-only @adapter segfault (separate robustness bug)

**Symptom:** a forward-only program using `@adapter(...)` on a CUDA target
(`--target cuda_sm80`) segfaults at runtime (Windows access violation
0xC0000005 / exit 139). Reproduced at all m, with and without
`NSL_WRGA_FUSED_CUDA=1`.

**Root cause:** the adapter side-table is materialized by
`emit_adapter_init_sidetable`, which is called **only inside train-block
compilation** (`crates/nsl-codegen/src/stmt.rs:4417`). A forward-only program has
no train block, so the side-table base pointer stays null (constructor zero-init,
`wrga_adapter_init.rs:16`). The adapter rewrite (fired on sm>=80,
`wrga_adapter_rewrite.rs`) then emits field accesses to `self.lora_A_<site>` etc.,
which dereference the null side-table base **before** reaching the fused FFI's
null-guard (`fused_adapter.rs:744`). So the existing null-guard never sees these
pointers; the crash is upstream in the field-access read of a null side-table.

**Note:** this is broader than the fused path -- it is any forward-only @adapter
on sm>=80, because the rewrite fires on the target SM regardless of the fused env
var. It also confirms there is **no inference-only adapter materialization path**
today: adapters only materialize inside a train block (B.2.1 side-table). Building
one is explicitly out of scope under this resolution (we are not investing in the
small-m inference niche).

**Fix (robustness, minimal):** guard the side-table-base read so an unmaterialized
(null-base) adapter field load yields a null tensor (0) instead of dereferencing
the null base, and make the adapter FFI return the base `x @ W` forward (with a
one-time stderr warning) when adapter pointers are null -- "fall back to base
forward, warn, don't crash." This converts a hard crash into a correct-base-forward
+ diagnostic, independent of the optimization decision.

---

## Considered options (preserved for the record -- A/B/C not pursued; D collapsed)

The fused kernel is **single-warp-per-tile** (`wrga_kernel_helpers.rs:5`); all four
tile-staging helpers gate global->SMEM loads on lane 0 (`%tid_x == 0` at
`wrga_kernel_helpers.rs:211/299/366/427`), leaving 31/32 lanes idle -- ~131k
serialized lane-0 loads at k=4096. The options considered:

- **A -- lane-distributed staging** (distribute each tile's 128 b32 elements across
  the 32 lanes). NOT pursued: even if it removed the lane-0 tax, measurement shows
  fused has no winning regime to optimize for.
- **B -- cp.async double-buffering.** NOT pursued (follow-on to A, which is not
  pursued).
- **C -- multi-warp-per-tile + tensor cores.** This is the only design that could
  plausibly beat cuBLAS, but it is a near-rewrite and a different project, not
  justified by any measured need.
- **D -- m-aware dispatch stopgap.** Collapsed to "always unfused" (no crossover).

The institutional principle still holds: scope a mechanism to where it has a
structural advantage and concede where it can't compete. The measurement moved the
boundary to "fused wins nowhere," so the principle yields: keep it opt-in, default
unfused everywhere, do not invest in making it competitive.
