# PCA Tier B B.1.5-3 skip_ratio=0 investigation findings

**Status:** RESOLVED
**Date:** 2026-05-13
**Owner:** bwiemz
**Branch:** `worktree-feat-pca-tier-b15-and-b2`
**Base HEAD at start:** `f3c1bb4f`
**Related spec:** `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
**Blocks unblocked:** B1.5-6 (M2 / M6 measurement + keep/revert decision)

---

## 1. What surfaced

On `worktree-feat-pca-tier-b15-and-b2` HEAD `f3c1bb4f`, the bench binary's
`--tier-b=on` runs against the gate fixture reported `skip_ratio=0` for every
iteration, even though:

- The kernel ran (exit 0, no `CUDA_ERROR_*`).
- The PTX synthesis emitted 16 `@%p_writeback_TB st.global.u8 [%dec_slot_TB], %dec_val_TB` sites
  (verified by dumping `NSL_BENCH_DUMP_PTX=1`).
- The bit-identical M3 parity assertion passed — Tier-B-on and Tier-B-off produced
  byte-identical `O` tensors.
- The skip-decisions HBM buffer was allocated, the kernel-arg slot was wired,
  and the bench binary read it back via `cuMemcpyDtoH`.

Reading back the 16384-byte HBM buffer after the kernel returned showed every
byte `0x00`. Expected: ~50 % of the bytes set to `0x01` (skipped tile) for the
50 %-sparsity gate fixture.

This blocked B1.5-6 (the M2 / M6 measurement + keep/revert decision per spec §3 /
§8) — the M2 threshold needs `skip_ratio`.

### Reproduction

```bash
cargo build -p nsl-codegen --features "cuda debug_kernel_instrumentation" \
    --bin bench --release

NSL_BENCH_PRINT_DECISIONS=1 ./target/release/bench \
    --fixture gate_4096 --tier-b on --seed 42 --iterations 100
# → tier_b_bench_result:fixture=gate_4096:...:skip_ratio=0:...
# → [bench] skip_decisions buffer: total=16384 bytes, n_skip(==1)=0, n_keep(==0)=16384, n_other=0
```

## 2. Failure-mode enumeration

The investigation tested four failure cases (α / β / γ / δ) in the order
prescribed by the brief.

### 2.1 Case α — predicate never fires

**Hypothesis:** the kernel's `setp.lt.u16` / `setp.gt.u16` evaluates `false` for
every `(qt, kvt)` pair. Sub-causes: range-table values all-zero, addresses
wrong, operand order wrong.

### 2.2 Case β — writeback predicate fails

**Hypothesis:** the `@%p_writeback_TB st.global.u8` predicate is wrong.
Sub-causes: `owning_warp` math, `warp_id` derivation, lane-0 check.

### 2.3 Case γ — HBM index mismatch

**Hypothesis:** the writeback writes to wrong slot OR bench reads wrong slot.

### 2.4 Case δ — HBM byte parsing

**Hypothesis:** kernel writes correct slots but bench reads them wrong
(byte-width / endianness / alignment).

## 3. Test results per case (evidence)

### 3.1 δ first (deepest cheap check)

Augmented the bench's `NSL_BENCH_PRINT_DECISIONS` diagnostic to dump the first
128 bytes raw and the offset of the first non-zero byte. Output:

```
[bench] no non-zero bytes in entire 16384 byte buffer
```

**Result:** all bytes `0x00`. Either the kernel writes correct slots but the
value is always `0` (no skips), or no slot is ever written (and `cuMemsetD8`'s
initial zeros remain). δ as "bench parses wrong" is **ruled out** — bench reads
were inspected raw, not parsed.

### 3.2 Bisecting writeback predicate vs slot-pointer reachability

Injected diagnostic probe (`pca_tilerange::emit_skip_decision_writeback`):

```ptx
mov.u16 %dec_probe_TB, 254;                       ; 0xFE
setp.eq.u32 %p_probe_TB, %tid_local_TB, 0;
mov.u64 %probe_slot_TB, %dec_buf_TB;              ; slot 0 of the buffer
@%p_probe_TB st.global.u8 [%probe_slot_TB], %dec_probe_TB;
```

Output:

```
[bench] first 128 bytes: fe 00 00 00 ...
[bench] first non-zero byte at offset 0 = 0xfe
```

**Result:** the unconditional thread-0 store wrote `0xFE` to slot 0
successfully. The buffer is reachable, the kernel-arg slot is correctly wired
through to `[skip_decisions_ptr]`. **The HBM index, the buffer pointer, and the
`st.global.u8` instruction itself are functional.** γ is ruled out (kernel can
write to the buffer; the issue is upstream of slot computation).

### 3.3 β — writeback predicate

Injected second probe writing `0xAA` under `%p_warp_owner_TB` alone (no lane
filter) and `0xCC` under `%p_writeback_TB`:

```ptx
@%p_writeback_TB st.global.u8 [%dec_slot_TB], %dec_probe_cc_TB;  ; 0xCC
@%p_warp_owner_TB st.global.u8 [%dec_slot_TB], %dec_probe_aa_TB; ; 0xAA
```

Output: **no `0xAA` and no `0xCC` anywhere**. So `%p_warp_owner_TB` (the
`warp_id == owning_warp` half of the writeback gate) never fires. Plausible
sub-causes from this alone: the predicate logic is wrong; OR the entire writeback
code is upstream-skipped via control-flow.

### 3.4 Bisecting again with unconditional `st.global.u8 [%dec_slot_TB]`

Injected third probe replacing the conditional store with an unconditional one:

```ptx
mov.u16 %dec_probe_bb_TB, 187;                              ; 0xBB
st.global.u8 [%dec_slot_TB], %dec_probe_bb_TB;              ; no predicate
@%p_writeback_TB st.global.u8 [%dec_slot_TB], %dec_val_TB;
```

Output: **79–99 `0xBB` bytes** scattered across the first ~64 byte positions of
the buffer (out of an expected 4096 reachable slots given the buggy slot-index
formula, see §4). Crucially, `%p_warp_owner_TB` does fire after all — the
"all-128-threads-write-0xBB" pattern is observed at the slots `%dec_slot_TB`
addresses, but the post-conditional `%dec_val_TB` store overwrites most of them
with `0` (the "kept" decision).

**Conclusion: `%p_writeback_TB` fires fine. β is not the actual fault.** The
fault is upstream: the skip decision `%dec_val_TB` is always 0 because
`%p_skip_TB` is always false. That is case **α**.

### 3.5 α — predicate operand-order vs addressing inspection

Manual inspection of `pca_tilerange.rs::emit_skip_predicate` against the dumped
PTX (`/c/tmp/gate.ptx`):

```ptx
shl.b32 %tile_byte_TB, 0, 1;                  ; qt = 0 immediate → tile_byte = 0
cvt.u64.u32 %addr_TB, %tile_byte_TB;          ; addr_TB = 0  (DEAD STORE)
cvta.shared.u64 %addr_TB, shmem;              ; addr_TB = shmem_base  (OVERWRITES!)
add.u64 %addr_TB, %addr_TB, 50176;            ; addr_TB = shmem_base + qtile_min_offset
ld.shared.u16 %qmin_TB, [%addr_TB];           ; reads qtile_min[0] always
```

**Found the bug:** the `cvt.u64.u32 %addr_TB, %tile_byte_TB` widens the per-tile
byte offset into the SAME register `%addr_TB` that the very next line —
`cvta.shared.u64 %addr_TB, shmem` — overwrites with the SMEM base. The per-tile
offset is computed but **never added** to the final address. Net effect: the
predicate always reads slot 0 of each of the four range tables (qtile_min,
qtile_max, kvtile_min, kvtile_max), regardless of the runtime `qt` and `kvt`.

For the gate fixture (segment-masked causal, 8 segments × 512 tokens, block 64),
slot 0 of each range table is segment 0. So:
- qmin = qmax = 0
- kvmin = kvmax = 0
- `p_lt = (0 < 0) = false`
- `p_gt = (0 > 0) = false`
- `p_skip = false` (for every `(qt, kvt)` evaluation, on every kernel
  invocation)

The four `setp` operand orders are correct as written; the addressing was
broken upstream of them.

**α confirmed as root cause.**

### 3.6 Second wiring bug (also α)

While reading the predicate emission, a second wiring issue surfaced. The qt
operand passed in from `s_compute::emit` is `&q_tile_iter.to_string()`:

```rust
crate::pca_tilerange::emit_skip_predicate(
    ptx, config, seq_len,
    &q_tile_iter.to_string(), // ← compile-time literal 0..15
    "%r_kvt_ord_TB",
    range_table_base,
    &skip_label,
);
```

But `q_tile_iter` is the **within-CTA** inner iter ordinal (`0..block_q/4 - 1`
= 0..15 for the gate fixture), **not** the global q-tile index. Each CTA
handles ONE global q-tile per the launch contract `grid_x = num_q_tiles = 64`;
the correct global q-tile ordinal for this CTA is `%bid_x` (`blockIdx.x`).

Effects of using `q_tile_iter` (independent of the address-overwrite bug):

- Only range-table slots 0..15 are ever consulted — entries for q-tiles 16..63
  are dead.
- Every CTA queries the same q-tile range; a CTA processing q-tile 50 still
  asks "is q-tile 0..15 disjoint from kvt?", producing nonsense answers.

Even with the address-overwrite bug fixed, this would still produce incorrect
skip decisions for `bid_x >= 16` (and incorrect / inconsistent ones for
`bid_x < 16`). Fixing both is required.

### 3.7 Cases γ / δ — recap

- **γ** (HBM index mismatch): ruled out by the §3.2 unconditional-tid-0 probe.
  The slot pointer arithmetic and the bench-side reader agree on layout.
- **δ** (HBM byte parsing): ruled out by §3.1 — the bench's parser reads raw
  bytes; the buffer is genuinely all-zero post-launch when the bug is present.

## 4. Root cause

**Case α fires.** Two coupled wiring bugs in the Tier B skip-predicate
emission:

1. **PTX address-overwrite** in `pca_tilerange.rs::emit_skip_predicate`. The
   per-tile byte offset is computed into `%addr_TB` and then immediately
   overwritten by `cvta.shared.u64 %addr_TB, shmem`. Net effect: the four range
   tables are always indexed at slot 0, regardless of the runtime qt / kvt
   operands. The predicate is constant-false for any fixture whose slot 0 sees
   the same segment ID in both q and kv axes (true for every supported
   fixture).

2. **Wrong qt operand passed in from the call site** in
   `flash_attention_v2/phases/forward/s_compute.rs`. The compile-time literal
   `q_tile_iter` (within-CTA inner iter ordinal 0..15) was passed instead of
   the per-CTA global q-tile index `%bid_x` (0..63). Even if (1) were fixed,
   this would produce incorrect skip decisions for `bid_x >= 16` and
   inconsistent ones below.

Neither bug is in the spec-pinned algorithm (`disjoint = (qmax<kvmin) ||
(qmin>kvmax)`); both are in the emitter's wiring. Brief constraint "fix the
wiring, not the algorithm" is satisfied.

## 5. Fix

### 5.1 `crates/nsl-codegen/src/pca_tilerange.rs::emit_skip_predicate`

Introduce a separate widened register `%tile_off_wide_TB` to hold the per-tile
byte offset across the `cvta.shared.u64 %addr_TB, shmem` instruction, and
explicitly add it to the address after the table-base addend:

```diff
-    ptx.push_str("    .reg .u64  %addr_TB;\n");
+    ptx.push_str("    .reg .u64  %addr_TB, %tile_off_wide_TB;\n");
     ...
     ptx.push_str(&format!("    shl.b32 %tile_byte_TB, {qt_reg}, 1;\n"));
-    ptx.push_str("    cvt.u64.u32 %addr_TB, %tile_byte_TB;\n");
+    ptx.push_str("    cvt.u64.u32 %tile_off_wide_TB, %tile_byte_TB;\n");
+    ptx.push_str("    cvta.shared.u64 %addr_TB, shmem;\n");
     ptx.push_str(&format!(
-        "    cvta.shared.u64 %addr_TB, shmem;\n    add.u64 %addr_TB, %addr_TB, {};\n",
+        "    add.u64 %addr_TB, %addr_TB, {};\n",
         addrs.qtile_min
     ));
+    ptx.push_str("    add.u64 %addr_TB, %addr_TB, %tile_off_wide_TB;\n");
     ptx.push_str("    ld.shared.u16 %qmin_TB, [%addr_TB];\n");
```

Same pattern applied to all four range-table loads (qtile_min / qtile_max /
kvtile_min / kvtile_max). Long-form comment in the source documents the bug
so future readers don't reintroduce it.

### 5.2 `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs`

Pass `%bid_x` as the qt operand instead of `q_tile_iter`:

```diff
     crate::pca_tilerange::emit_skip_predicate(
         ptx,
         config,
         seq_len,
-        &q_tile_iter.to_string(), // qt is compile-time; emit as immediate
+        "%bid_x", // global q-tile index for THIS CTA (grid_x = num_q_tiles)
         "%r_kvt_ord_TB",
         range_table_base,
         &skip_label,
     );
```

Inline comment updated to document the prior-correct `q_tile_iter` was wrong
for the kernel's per-CTA q-tile contract.

### 5.3 Snapshot / SASS-baseline updates

The PTX change cascades to four snapshot regenerations (each a routine
re-baseline, not a behavioural-snapshot change):

- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap`
- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_64_64_64.snap`
- `pca_tier_b_predicate_isolation__predicate_4k_block64.snap`
- `pca_tier_b_sass_baselines.rs` baselines `5144 → 5200` (32×32×32) and
  `14784 → 14888` (64×64×64) — exactly the additional `add.u64 %addr_TB, %addr_TB, %tile_off_wide_TB`
  + `cvt.u64.u32 %tile_off_wide_TB, ...` per range-table load × four range tables
  × N predicate sites.

### 5.4 Diagnostic dump in the bench's skip-ratio readback

`crates/nsl-codegen/src/bin/bench/launch.rs` — when
`NSL_BENCH_PRINT_DECISIONS=1`, also dump the first 128 raw bytes of the
HBM-readback buffer and the offset of the first non-zero byte. Kept in tree
as the most direct future-debug aid for this layer (it surfaced the
unsanitized-zero-buffer signal that distinguishes "writeback never fires" from
"writeback fires but always writes 0").

## 6. JIT-amortization verification

CUDA events bracket each individual `cuLaunchKernel`, and `cuModuleLoadDataEx`
(where JIT happens) is **outside** the timed loop in `time_kernel_launches`.
So the JIT cost is amortized to ZERO across the timed iterations — at any
`--iterations N` ≥ 1 the per-iter measurement is post-JIT.

Empirical verification on RTX 5070 Ti / CUDA 13.2 / driver 591.86:

| `--iterations` | tier-b=off median_us | tier-b=on median_us | ratio |
|---|---:|---:|---:|
| 1 | 38675.84 | 9458.69 | 0.245 |
| 5 | 38542.27 | 9840.19 | 0.255 |
| 100 | 38466.98 | 9820.86 | 0.255 |

All three N values lie within ±1.5 % of each other for both modes. **JIT is
fully amortized at N=1**; the default `--iterations 100` is conservative and
not the source of the previously-noted timing anomalies. The pre-fix
"1000× slowdown" referenced in the brief was a misread of the pre-fix
behaviour: Tier-B-on was 38 ms ≈ Tier-B-off because the predicate's always-
false output meant Tier B's added preamble + per-tile predicate work was paid
without recovering any savings. Post-fix Tier-B-on is 4.0 × faster than
Tier-B-off at the gate fixture, with `skip_ratio=0.875`.

## 7. Re-verified skip_ratio after fix

```
tier_b_bench_result:fixture=gate_4096:tier_b=on:median_us=9260.319709777832:n=100:skip_ratio=0.875:seed=42
```

The 87.5 % skip ratio matches the gate fixture's segment shape: 8 segments
of 512 tokens each × block 64 yields 7 of every 8 (qt, kvt) tile pairs in
different segments → skip; 1 of 8 in same segment → keep. Slot pattern in
the buffer (first 128 bytes):

```
00 00 00 00 00 00 00 00  ← q-tile 0, kv-tiles 0..7  (segment 0 ↔ 0, KEEP)
01 01 01 01 01 01 01 01  ← q-tile 0, kv-tiles 8..15 (segment 0 ↔ 1, SKIP)
01 01 01 01 01 01 01 01
01 01 01 01 01 01 01 01
01 01 01 01 01 01 01 01
01 01 01 01 01 01 01 01
01 01 01 01 01 01 01 01
01 01 01 01 01 01 01 01  ← end of q-tile 0 (kv-tiles 56..63, segment 0 ↔ 7)
00 00 00 00 00 00 00 00  ← q-tile 1, kv-tiles 0..7  (segment 0 ↔ 0, KEEP)
01 01 ...
```

Pattern matches the analytical expectation exactly. n_skip = 14336,
n_keep = 2048, n_other = 0; 14336 / 16384 = 0.875.

## 8. Gate-criteria status

1. `cargo test -p nsl-codegen --features "cuda debug_kernel_instrumentation" --test pca_tier_b_m3_parity --release` → **PASS** (10/10 bit-identical Tier-B-on vs Tier-B-off).
2. Smoke test: `bench --fixture gate_4096 --tier-b on --seed 42 --iterations 100` → **PASS** (`skip_ratio=0.875 > 0`).
3. Findings doc (this file) committed.
4. Code fix committed.
5. JIT-amortization verification table in §6.

## 9. Out of scope / follow-ups noted

- The 8-segment generation in `launch.rs::generate_segment_mask` ignores the
  fixture name and steers only by `target_sparsity`. `parity_5` (single_doc,
  `target_sparsity=0.0`) thus produces a 2-segment mask, not a single segment,
  so the reported `skip_ratio=0.5` for `parity_5` is the bench-generator's
  pattern, not the fixture's "single_doc" intent. Out of scope here (the M3
  parity assertion is byte-equality on the same input pattern; both Tier-B-on
  and Tier-B-off see the same generated mask). Future fixture work could
  thread the parity fixture's actual `PackingFixture` segment_ids through to
  the bench harness.
- Stage B.2 (backward) inherits the same emitter via §7.4 of the design spec.
  When V-B.2-predicate runs (per spec §7.3), it must verify backward's
  `qt` operand source — backward likely also uses `%bid_x` for the q-tile
  index, but if FA-2 backward uses KV-outer / Q-inner iteration order, the
  per-tile operand wiring may differ.
- The redundant 16× per-CTA emission of the predicate (once per `q_tile_iter`
  inner iter) is wasteful — the predicate's qt is now `%bid_x` (uniform across
  q_tile_iter values within a CTA), so all 16 emissions compute the same
  result. Hoisting the predicate out of the per-q_iter loop is a
  measurement-time optimization candidate; the wiring is functionally correct
  as-is.
