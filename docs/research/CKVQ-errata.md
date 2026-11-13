# Compiler-Known Value Quantization (CKVQ) — Errata

**Companion to** `CKVQ.pdf`. **Date:** 2026-11-13.

This file lists the statements in the CKVQ proposal (April 2026) that should
not be relied on, and says what is true instead. The entries come from a
November 2026 review that checked the proposal against its own arithmetic,
against the published literature, and against the code in this repository.
Each entry says how it was checked. Where a point rests on a search-result
summary or a secondary source, or was not re-checked against the original,
the entry says so. Section numbers are the PDF's.

---

## Part 1 — Errors in the proposal

### E1 — Per-page scales throw away what pre-RoPE quantization preserves

**Proposal.** §2.1: "Before RoPE, outliers are concentrated in a small set of
channels", with `format=per_channel_2bit` in the transformation. §2.3:
"Per-page is an order of magnitude better metadata overhead at comparable
quantization quality." §5.2, invariant I-29: "A page has one scale per head;
all tokens within a page share the scale."

**Correction.** Innovation 1 quantizes keys before RoPE (rotary position
embedding) because their outliers then sit in a few channels, which a
per-channel scale can absorb. Innovation 3 and I-29 then mandate one scale
for all 128 channels of a head, which discards exactly that structure. The
two cancel.

A numeric check on synthetic keys (one 32-token page, 128 channels, 6 of
them at 12× magnitude, asymmetric min/max uniform quantizer) gives the
following root-mean-square error in the attention logits, whose own
root-mean-square value is 2.44:

| Scale granularity      | 2-bit | 4-bit |
| ---------------------- | ----- | ----- |
| Per-channel, pre-RoPE  | 1.04  | 0.23  |
| Per-channel, post-RoPE | 1.28  | 0.24  |
| Per-page, pre-RoPE     | 6.36  | 1.07  |
| Per-page, post-RoPE    | 13.4  | 1.05  |

Before RoPE, the per-page error is about five to six times the per-channel
error (6.36/1.04 = 6.1; 1.07/0.23 = 4.7), and at 2 bits it is larger than
the logits themselves. "Comparable quantization quality" is unsupported.

**Source.** Numeric check.

### E2 — Keys stored pre-RoPE cannot be fed to a tensor-core multiply as stored

**Proposal.** §2.5: "on Blackwell, the attention kernel uses NVFP4 MMA paths
directly on the quantized KV values, avoiding the dequantize step entirely.
This is a 2× throughput improvement over dequant-then-MMA paths."

**Correction.** KV is the attention key/value cache, NVFP4 is NVIDIA's 4-bit
floating-point format, and MMA is the tensor-core matrix-multiply
instruction. Under innovation 1 the stored keys have not been rotated, so on
every read they must be dequantized and rotated before the dot product with
the query. They cannot go into an MMA as stored, so innovations 1 and 5 are
incompatible for keys. The NVIDIA post that the proposal cites itself says
values are "dequantized from NVFP4 to FP8 before attention". The 2× figure
has no source.

**Source.**
<https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/>

### E3 — The 2-bit format clips its largest positive value by construction

**Proposal.** §3.1: "2-bit integer per element, symmetric signed (values in
{-2, -1, 0, 1} × scale)". §4.4:
`K_scale = compute_absmax(K_proj) / 2^(bit_budget - 1)`.

**Correction.** At 2 bits the scale is absmax/2, so a positive element at
the maximum magnitude maps to code +2, which the level set does not contain.
It clamps to +1 and reads back as absmax/2: with absmax 1.0, a value of 1.0
becomes 0.5. This breaks the proposal's own correctness test 4 in §6.4, "no
saturated values except identified outliers". The level set is also not
symmetric, since −2 has no positive counterpart. KIVI and KVQuant use
asymmetric or non-uniform levels.

**Source.** Numeric check.

### E4 — Outlier storage is specified three incompatible ways, and one halves the ratio

**Proposal.** §2.4: "Memory per outlier: 4 bytes (2 for index, 2 for value).
At 2 outliers × 128 head_dim per page, outlier storage is 1 KB per page —
comparable to the quantized value storage." §3.1: "each as (index: u8,
value: f16). Budget: 4k bytes per page", and "Compression ratio vs FP16: ~8×
for 32-token pages at head_dim=128." §4.2: "typically <2% of compressed data
size".

**Correction.**

- §2.4 stores whole outlier tokens: 2 × 128 × 4 bytes = 1,024 bytes, the
  same size as the 1,024 bytes of 2-bit data, which is about 100% overhead.
  The page is then about 2,048 bytes against 8,192 bytes in FP16, a ratio of
  about 4×, not ~8×.
- §3.1 stores single elements, but a `u8` index cannot address the 4,096
  elements of a page (32 × 128), and `u8` plus `f16` is 3 bytes, not 4.
- §4.2's "<2%" holds for §3.1's reading (2 + 8 + 1 = 11 bytes on 1,024,
  about 1.1%) and not for §2.4's.
- Two single elements out of 4,096 is 0.05% of the page. KVQuant's abstract
  specifies per-vector dense-and-sparse outlier handling, a much larger
  budget.

**Source.** Recomputed from the proposal's own figures; KVQuant abstract,
<https://arxiv.org/abs/2401.18079>.

### E5 — The headline ratios do not follow from each other

**Proposal.** Abstract: "4-8× KV cache memory reduction at under 0.5%
accuracy loss; ... 5-10× context extension at fixed hardware". §3.3 (NVFP4):
"Compression ratio vs FP16: 4×."

**Correction.** At fixed hardware the context can grow by at most the factor
by which the cache shrinks, so a 4–8× reduction cannot give a 5–10×
extension. For NVFP4, §3.3's own layout is 4 bits per value plus 1 byte of
scale per 16 elements, which is 4.5 bits per element and a ratio of
16/4.5 = 3.56×, not 4×.

**Source.** Recomputed from the proposal's own figures.

### E6 — Attention-sink tokens are not magnitude outliers

**Proposal.** §2.4: "attention-sink tokens are always outliers in KV
magnitude, so they land in the outlier list and are preserved at full
precision."

**Correction.** This is the wrong way round. KVSink reports that sink tokens
have significantly smaller query, key and value norms, and that sinks also
occur beyond the first positions. An outlier list selected by a magnitude
threshold will not catch them. KVQuant keeps the first token in FP16 because
of its sensitivity, not its magnitude (this last point not re-checked
against the paper).

**Source.** <https://arxiv.org/html/2508.04257v1>

### E7 — The key/value bit asymmetry is backwards and is not KIVI's

**Proposal.** §3.5: "The asymmetric middle-layer choice (2-bit keys, 4-bit
values) comes from KIVI's observation that keys tolerate lower bit budgets
than values for typical LLMs." §2.2: "GQA layers: each group's K/V shared
across multiple Q; more aggressive compression OK (KIVI observation)."

**Correction.** KIVI's abstract says only that keys should be quantized per
channel and values per token. KVTuner finds that the key cache is the more
sensitive of the two, so keys need at least as many bits as values, the
opposite of §3.5. The remark about GQA (grouped-query attention) layers is
not in KIVI's abstract, and it could not be verified in the paper.

**Source.** <https://arxiv.org/abs/2402.02750> (KIVI);
<https://proceedings.mlr.press/v267/li25dd.html> (KVTuner).

### E8 — "The dequant is fully hidden" contradicts the figures beside it

**Proposal.** §4.3: "The dequant cost is typically 10-20 cycles per element
at CKVQ-4bit, with memory latency 200-500 cycles per cache line — so the
dequant is fully hidden."

**Correction.** A 64-byte cache line holds 128 four-bit values. At 10–20
cycles per element that is 1,280–2,560 cycles of dequantization per line,
several times the 200–500 cycles of latency it is said to hide behind.

**Source.** Recomputed from the proposal's own figures.

### E9 — A per-page scale is not known while the page is being filled

**Proposal.** §4.4: "Compute per-page scale (page may hold up to 32 new
tokens being filled)".

**Correction.** During decoding, tokens arrive one at a time, so a page's
scale is unknown until the page is full. The proposal neither keeps recent
tokens at full precision until then nor says how tokens already stored are
re-quantized when the scale changes. KIVI keeps such a full-precision
residual (not re-checked against the paper).

### E10 — The KV-size table mixes models with and without GQA

**Proposal.** §1.1 table: Llama-3-8B at 8K is "4 GB" and at 128K is "64 GB";
Llama-3-70B at 128K is "40 GB (GQA)". Text below it: "approximately 328 GB —
across four H100s just for cache."

**Correction.** By the section's own formula, 4 GB and 64 GB are what 32
layers × 32 heads × 128 dimensions give in FP16, which ignores GQA.
Llama-3-8B has 8 KV heads, so the figures are 1.0 GiB at 8K
(2 × 32 × 8 × 128 × 8,192 × 2 bytes) and 16 GiB at 128K. The 70B rows do
apply GQA, so the table is inconsistent with itself. And 328 GB does not fit
on four 80 GB H100s (328/80 = 4.1); it needs five.

**Source.** Recomputed from the proposal's own formula.

### E11 — The NSLCoder-50M demonstration needs no quantization

**Proposal.** §6.3: "NSLCoder-50M at 100K context on RTX 5070 Ti 16GB."

**Correction.** With the model's real configuration (8 layers, 4 KV heads,
head dimension 64, so 8,192 bytes per token in FP16), 100,000 tokens of KV
cache are 819 MB (0.76 GiB). That fits on a 16 GB card uncompressed, so the
demonstration would show nothing about CKVQ.

**Source.** Recomputed; model dimensions read from
`models/coder50m/model.nsl`.

### E12 — Smaller slips

- Reference 3 gives the title "TurboQuant: Fast Vector Quantization via
  Random Rotation and Residual Compression". The paper is "TurboQuant:
  Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et
  al.; <https://arxiv.org/abs/2504.19874>,
  <https://iclr.cc/virtual/2026/poster/10006985>).
- §1.2 describes TurboQuant as "PolarQuant + QJL; 6× reduction". Its method
  is a random rotation, per-coordinate scalar quantizers, and a 1-bit QJL
  step on the residual. The "6×" comes from a blog framing; the abstract
  claims quality neutrality at 3.5 bits. (This item is from a search-result
  summary, not from the paper itself.)
- §7: "Multi-Latent Attention compresses KV to a latent representation
  shared across layers". DeepSeek's Multi-head Latent Attention is a
  per-layer low-rank scheme (not re-checked against the source).
- §2.5: "Ampere / Hopper (SM_80, SM_90): ... E4M3 FP8 for keys (M35
  instructions available)". No FP8 instruction support on Ampere could be
  verified.
- §1.2: "CKVQ aims at all five simultaneously" follows a list of four.
- Layer 0 is given three different precisions: "bit budget typically 2 for
  keys, 2 for values" in §2.2; 8-bit for the first few tokens and 4-bit for
  the rest in §3.5; and FP16 in the companion CFIE proposal (`CFIE.pdf`,
  §2.5).

Checked and found correct: KVQuant is NeurIPS 2024 and its abstract reports
pre-RoPE per-channel key quantization, under 0.1 perplexity degradation at
3 bits, and 1M context on one A100-80GB (<https://arxiv.org/abs/2401.18079>); KIVI
is ICML 2024; TurboQuant exists and is ICLR 2026; the NVIDIA NVFP4 KV-cache
post exists and says "<1% accuracy loss"; and the §2.3 arithmetic (1,024
bytes, 0.2%, 6%) is right.

---

## Part 2 — Superseded by later work

- **KVTuner** (ICML 2025,
  <https://proceedings.mlr.press/v267/li25dd.html>). Layer-wise key and
  value bit pairs have to be searched per model, and keys need at least as
  many bits as values. This removes the "symbolic, no calibration" premise
  of §2.2 and inverts the §3.5 table.
- **KVSink** (August 2025, <https://arxiv.org/html/2508.04257v1>). Sink
  tokens have to be predicted, not detected by magnitude, and they are not
  only at position 0. §2.4's outlier list does not handle them.
- **TurboQuant** (<https://arxiv.org/abs/2504.19874>). It already offers a
  data-oblivious route, with no calibration, at about 3.5 bits (figure from
  a search-result summary). CKVQ's claim of novelty against it is weak.
- **CommVQ** (<https://arxiv.org/abs/2506.18879>). A codebook that commutes
  with RoPE lets decoding fold into the attention computation, reported at
  1–2 bits. This is the published answer to the conflict in E2.
- **SQuat** (<https://arxiv.org/html/2503.24358v2>). A calibration-free
  method that keeps quantization residuals orthogonal to the query subspace.
  It is a further calibration-free method that the proposal does not
  mention.
- **NVFP4 KV cache in SGLang**
  (<https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache>). The format is
  now integrated in a serving engine, with about 26–30% higher decode
  throughput at peak batch (figure from a search-result summary); its 50%
  memory saving is relative to an FP8 cache. Native FP4 tensor-core
  instructions exist on datacenter Blackwell (sm_100a/103a) only
  (<https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/blackwell_functionality.html>);
  consumer Blackwell, which includes the project's RTX 5070 Ti, is reported
  not to expose them
  (<https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607>).

---

## Part 3 — What the code implements

CKVQ is not implemented. Outside the PDF the name appears only in a glossary
entry (`docs/wiki/Glossary.md`). There is no GEBA (Gradient-Envelope Bit
Allocation) code, no pre-RoPE quantization, no per-page scale or outlier
list, no 2-bit, 4-bit or NVFP4 format, no decision in WGGO (Wengert Graph
Global Optimization, NSL's joint optimisation pass), no page-structure
extension, and no pass over the compiler's mid-level IR.

Two existing pieces are the nearest, and cover almost none of it:

- **CFIE per-layer KV precision** — `crates/nsl-codegen/src/cfie_kv_quant.rs`
  and `crates/nsl-codegen/src/cfie_kv_quant_ptx.rs`, enabled by the
  serve-block key `kv_quant`. It chooses FP16 or INT8 per layer, from a
  positional prior or from the largest singular value of the key and value
  projection weights, and emits read-side attention kernels that dequantize
  INT8 with one scalar scale per layer and per K or V, passed at launch.
  Keys are stored after RoPE. There is no quantize-on-store kernel, nothing
  launches the read kernels outside tests, and the scheme is not
  per-channel, per-token or per-page.
- **`@kv_compress`** — `crates/nsl-runtime/src/kv_compress/`. CPU routines
  for INT8 per head or per token, INT4 per 64-element group, and an FP8
  cast, plus sliding-window and H2O eviction managers. The decorator's
  scheme is recorded on the paged-KV manager
  (`crates/nsl-runtime/src/paged_kv/manager.rs`) "for use by future
  compression-aware eviction logic" and is never read, and
  `nsl_kv_quantize_and_store` has no caller, so the cache stays
  uncompressed.

At the one point where code and proposal meet, the code is the better guide:
CFIE's middle precision band keeps keys in FP16 and values in INT8, giving
keys the higher precision (E7).

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Keys: per-channel scale and zero-point, pre-RoPE, fixed by a short offline
   calibration. Values: per-token groups. Use asymmetric min/max levels.
2. Give keys at least as many bits as values, searched per model as KVTuner
   does. Drop the per-page scalar scale.
3. Drop the claim of NVFP4 MMA directly on stored keys; keep NVFP4 only as
   storage with dequantization to FP8, on datacenter Blackwell.
4. Keep a small FP16 window of recent tokens, so a page is quantized once,
   when it is full. Handle sink tokens by prediction, not by magnitude.
5. Measure perplexity change on coder50m at 4 and 2 bits against FP16, and
   attention-logit error per layer, before making any accuracy claim.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code. CKVQ itself is not implemented, so there is no
CKVQ code to be wrong; this part lists what the review found wrong in the two
nearest pieces named in Part 3, as distinct from the proposal's own errors in
Part 1. Fixed items are described in `CHANGELOG.md` under "Unreleased"; the
open list is as of 2026-11-13. Each item ends with how it was established:
reproduced by running, numeric check, or by reading the code only.

### Fixed

None of the defects below has been fixed.

### Open, in the adjacent code

- **`nsl_kv_quantize_and_store` cannot append to a block.** Its token offset
  is ignored: scales are computed over the new tokens only and written from
  index 0, replacing the block's earlier scales, so earlier tokens would be
  dequantized with a scale they were not stored with. Scales should be kept
  per absolute token, or fixed in advance. It has no caller today.
  `crates/nsl-runtime/src/kv_compress/ffi.rs`. By reading the code only.
- **The CFIE INT8 read kernels take one scale per layer and per K or V, and
  nothing says when it is fixed.** A scale chosen from early tokens clips
  later ones; a scale updated later invalidates stored codes. On synthetic
  keys with outlier channels, one scale gives 13.6% mean relative error on
  ordinary channels against 1.1% on outlier channels. Nothing writes INT8
  KV. `crates/nsl-codegen/src/cfie_kv_quant_ptx.rs`. Numeric check.
- **`@kv_compress(method="quantize", ...)` is accepted, reported as active,
  and does nothing.** Compile time and run time both log "KV cache
  compression"; the cache stays uncompressed. It should be refused or wired.
  `compile_call` in `crates/nsl-codegen/src/expr/calls.rs`;
  `nsl_kv_cache_init` and `set_compression` in
  `crates/nsl-runtime/src/paged_kv/manager.rs`. By reading the code only.
- **With two differing `@kv_compress` decorators the policy used is
  arbitrary.** It is taken as the first value of a hash map. `compile_call`
  in `crates/nsl-codegen/src/expr/calls.rs`. By reading the code only.
- **INT4 and per-token INT8 quantization index past a 128-entry scale
  array and panic.** INT4 on a block of 8 heads × 16 tokens × 128 dims needs
  256 groups; per-token INT8 fails for a block above 128 tokens (CFIE's
  default block is 256). The unit tests use 16 elements.
  `quantize_int4_per_group`, `quantize_int8_per_token` in
  `crates/nsl-runtime/src/kv_compress/quantize.rs`. By reading the code only.
- **The reported compression ratio ignores per-block metadata.**
  `KvBlockQuantMeta` is 1,028 bytes per K or V block. For 4 heads × 16
  tokens × 64 dims, against FP16, INT8 is 8192/5124 = 1.60× (not 2×) and
  INT4 is 8192/3076 = 2.66× (not 4×). `bytes_per_element` in
  `crates/nsl-runtime/src/kv_compress/mod.rs`. Numeric check.
- **The INT8 read-kernel GPU test cannot see a wrong scale.** It compares
  the kernel with a CPU dequantization that uses the same scale.
  `crates/nsl-codegen/tests/cfie_kv_quant_gpu_parity.rs`. By reading the
  code only.
- **The H2O and sliding-window managers have no callers.** H2O also evicts
  a block only when every token in it is in the lowest-score set, which
  scattered scores almost never produce, and it does not track blocks it
  has already evicted. `check_eviction` in
  `crates/nsl-runtime/src/kv_compress/h2o.rs`. By reading the code only.
