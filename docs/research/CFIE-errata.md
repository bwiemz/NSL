# Compile-Fused Inference Engine (CFIE) — Errata

**Companion to** `CFIE.pdf`. **Date:** 2026-11-13.

This file lists the statements in the CFIE proposal (April 2026) that should
not be relied on, and says what is true instead. The entries come from a
November 2026 review that checked the proposal against its own arithmetic,
against the published literature, and against the code in this repository
that implements it. Each entry says how it was checked. Where a point rests
on a secondary source, or was not re-checked against the original, the entry
says so. Section numbers are the PDF's.

---

## Part 1 — Errors in the proposal

### E1 — The speculative accept/reject rule does not preserve the target distribution

**Proposal.** §2.3, "Compiled Speculative Step":
`draft_token[k] = argmax(draft_logits)`,
`draft_probs[k] = softmax(draft_logits)[draft_token[k]]`, accept
`if random() < target_prob / draft_probs[k]`, otherwise
`correction_token = sample(target_logits[k] - draft_probs)`.

**Correction.** Speculative decoding reproduces the target model's
distribution only if a drafted token x is accepted with probability
min(1, p(x)/q(x)), where p is the target distribution and q is the
distribution the draft token was actually drawn from, and a rejected position
is resampled from max(0, p − q) normalised over the whole vocabulary. The
pseudo-code gets both halves wrong.

- The draft is greedy (argmax), so q is a point mass and q(x*) = 1 for the
  drafted token x*. Dividing by the softmax probability of x*, which is below
  1, accepts the draft too often. With target p = [0.3, 0.7] and draft softmax
  [0.6, 0.4], the draft proposes token 0 and the stated test accepts it with
  probability 0.3/0.6 = 0.5, so token 0 is emitted at least half the time
  against a target of 0.3. With p(x*) = 0.45 and a draft softmax value of
  0.40 the ratio exceeds 1 and the draft token is always emitted.
- The correction step subtracts probabilities from logits, which does not
  define a distribution.

With q(x*) = 1 the rule reproduces the target exactly: the total-variation
distance was about 1e-17 on five random 8-token cases. With the softmax
probability as the denominator, and the residual read as the target with the
drafted token's probability reduced by the draft probability, the distance on
the same cases was 0.17 to 0.58. The shipped code followed the pseudo-code
on this point until November 2026; it now uses the correct rule (Part 3).

**Source.** Numeric check: exact enumeration of the rule's output
distribution.

### E2 — A one-CTA decode-block kernel cannot deliver the claimed speedup

**Proposal.** §2.4: "Persistent Decode Block Kernel (1 CTA, 1 kernel launch
per layer)".

**Correction.** A CTA (cooperative thread array, one CUDA thread block) runs
on a single SM (streaming multiprocessor): 1 of 132 on an H100, 1 of 70 on an
RTX 5070 Ti. Decode at batch 1 is a sequence of matrix-vector products over
all the model's weights and is limited by memory bandwidth. Putting a whole
layer in one CTA saves tens of microseconds of launch overhead and gives up
nearly all of the GPU's memory parallelism. For the 50M model, which reads
about 102 MB of weights per token, the estimate is milliseconds per token for
the one-CTA design against roughly 0.1 ms at the bandwidth limit with all SMs
in use. The kernel is also not persistent in the usual sense: it is launched
once per layer per token and exits.

**Source.** Reasoning from the design. The per-token figures are estimates,
not measurements, and the SM counts were not re-checked against a source.

### E3 — The launch-overhead figures contradict each other

**Proposal.** §2.4: "~500-1000 kernel launches per token. At ~5μs per launch,
this is 2.5-5ms of pure launch overhead", then "32 kernel launches instead of
~1000. At 5μs per launch: 160μs saved per token." §1: "the per-token overhead
budget is ~100-500μs". §3 build report: "Kernel launches per token: 34 (vs
~500 baseline)" and "Estimated decode latency: 82μs/token (vs 210μs
baseline)".

**Correction.**

- 160 µs is the cost of the 32 launches that remain (32 × 5 µs), not the
  saving. On the proposal's own figures the saving is (1000 − 32) × 5 µs =
  4,840 µs, about 4.8 ms.
- §1's whole per-token overhead budget, 100–500 µs, is 5 to 50 times smaller
  than the 2.5–5 ms of launch overhead that §2.4 then claims
  (2,500/500 = 5; 5,000/100 = 50).
- In the §3 report, 34 launches at 5 µs each are 170 µs, more than the 82 µs
  it estimates for the whole token, and a 500-launch baseline is 2,500 µs,
  not 210 µs.
- The "~500" baseline is the 32-layer figure applied to NSLCoder-50M, which
  has 8 layers. At the proposal's own 15–30 launches per layer that model
  needs 120–240.

**Source.** Recomputed from the proposal's own figures.

### E4 — The KV-cache sizes are wrong and disagree with each other

**Proposal.** §2.1: "Per-token KV: 4 KV heads × 128 head_dim × 2 (K+V) × 8
layers × 2 bytes = 16,384 bytes" and "Max tokens: 15,486 MB / 16 KB ≈ 968,000
tokens". §2.5 table (NSLCoder-50M, seq=2048): "FP16 uniform 128 MB" and
"~72 MB (10% FP16 + 90% INT8)".

**Correction.** KV here is the attention key/value cache.

- NSLCoder-50M does not have a head dimension of 128.
  `models/coder50m/model.nsl` sets `HEAD_DIM = 64` with 8 heads and 4 KV
  heads, so the per-token size is 4 × 64 × 2 × 8 × 2 = 8,192 bytes, half the
  stated figure.
- "15,486 MB / 16 KB ≈ 968,000" mixes decimal and binary units. Computed
  consistently in binary units it is 15,486 × 2^20 / 16,384 = 991,104.
- At the proposal's 16,384 bytes per token, a 2,048-token sequence is
  33.5 MB (32 MiB), not 128 MB. 128 MiB corresponds to a batch of 4, which
  the table does not state.
- "~72 MB (10% FP16 + 90% INT8)" matches neither the layer list directly
  above it nor the §3 build report. On the table's own scale (128 MB all
  FP16, 64 MB all INT8, so 8 MB per FP16 tensor and 4 MB per INT8 tensor
  over 16 K and V tensors), the layer list puts 5 of 16 tensors in FP16,
  which is 5 × 8 + 11 × 4 = 84 MB. The build report's "layers 0,7=FP16,
  layers 1-6=INT8" puts 4 of 16 in FP16, which is 80 MB.

**Source.** Recomputed from the proposal's own figures; model dimensions read
from `models/coder50m/model.nsl`.

### E5 — The top-p loop drops the token that crosses the threshold

**Proposal.** §2.2, Phase 3: `cumsum += prob / sum_exp`,
`if cumsum > top_p: break`, and only then `candidates.push((prob, idx))`.

**Correction.** The loop breaks before pushing the token whose probability
takes the cumulative sum past `top_p`. The nucleus is the smallest set of
tokens whose mass is at least `top_p`, so that token belongs in it. As
written, if the most likely token alone exceeds `top_p`, the candidate set is
empty. The shipped sampler keeps the crossing token (Part 3).

### E6 — The tree variant does not say how it stays lossless

**Proposal.** §2.3: "the top-2 tokens at each position are explored (tree
width=2, depth=K)".

**Correction.** Taking the top-2 tokens is deterministic candidate selection,
not sampling from a draft distribution. Single-draft accept/reject over such
a tree preserves the target distribution only when the target itself is
greedy. A stochastic target needs a multi-draft verification rule of the
kind used by SpecInfer and SpecTr (not re-checked against those papers).

### E7 — A JSON Schema is not a DFA in general

**Proposal.** §2.2: "The DFA transition table is embedded as a compile-time
constant in the kernel". §3 build report: "Grammar DFA: 47 states, 892
transitions, fused into sampling kernel".

**Correction.** A DFA (deterministic finite automaton) cannot express a JSON
Schema that is recursive or allows unbounded nesting; those need a pushdown
automaton. "47 states, 892 transitions" is plausible only for an automaton
over bytes. A table over the model's 49,152 tokens has far more live
transitions.

### E8 — Level-3 feasibility is stated three ways that do not agree

**Proposal.** §2.4: "For single-token decode with d_model ≤ 2048, the entire
transformer block can be a persistent kernel", and "For d_model=4096 (7B
models), Level 3 requires ~128KB SMEM — feasible on H100 (228KB per SM)". §3
build report, for an H100 target: "L3 infeasible at d=512 on H100".

**Correction.** Level 3 is the full-block fusion level of CSHA
(Compiler-Synthesized Holistic Attention, NSL's attention-fusion pass), and
SMEM is per-CTA shared memory. The first statement limits Level 3 to
d_model ≤ 2048, the second allows it at 4096 on an H100, and the third rules
it out at 512 on the same GPU. They cannot all hold. The 228 KB figure itself
is right: an H100 has 228 KB of shared memory per SM, of which one thread
block can use 227 KB.

**Source.** <https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html>
for the shared-memory figure.

### E9 — The fused-sampling memory saving is about 2% of the step's traffic

**Proposal.** §2.2: "49152 × 4 bytes × 6 HBM round-trips = 1.1 MB of wasted
HBM traffic per token."

**Correction.** HBM is the GPU's main memory, and the LM head is the final
projection onto the vocabulary. The arithmetic is right (49,152 × 4 × 6 =
1,179,648 bytes, about 1.18 MB), but the same step has to read the LM-head
weights, about 50 MB for this model (49,152 × 512 × 2 bytes = 50.3 MB). The
traffic removed is roughly 2% of what the step reads, so the saving is
overstated.

**Source.** Recomputed from the proposal's own figures.

### E10 — Prior systems are misdescribed, and one is misdated

- §2.1: "vLLM's PagedAttention allocates KV cache in variable-sized blocks at
  runtime." The blocks are fixed-size, 16 tokens by default and configurable.
  Source (secondary):
  <https://www.mintlify.com/vllm-project/vllm/concepts/paged-attention>.
- §1 table: "PagedAttention block table lookup ~5-15μs per request". The
  proposal gives no source, and the figure could not be verified. A
  block-table lookup inside a kernel is an indexed load, which costs
  nanoseconds.
- §4 table: "Runtime Python interpreter (Outlines/lm-format-enforcer)" as the
  grammar baseline was already out of date in April 2026 (Part 2).
- Abstract: "runtime overhead that vLLM, TensorRT-LLM, and SGLang cannot
  avoid". FlashInfer explicitly keeps compatibility with CUDA Graphs
  (<https://arxiv.org/abs/2501.01005>). TensorRT fuses operations when it
  builds an engine, and CUDA Graphs remove per-launch cost (these two points
  not re-checked against a source).
- References: SGLang is dated "2025". It is an arXiv paper from December 2023
  that appeared at NeurIPS 2024 (not re-checked against the source).

The citations for PagedAttention (SOSP 2023), speculative decoding
(Leviathan et al., ICML 2023) and FlashInfer (2025) were found correct.

---

## Part 2 — Superseded by later work

- **FlashSampling** (arXiv 2603.15854, March 2026,
  <https://arxiv.org/abs/2603.15854>). It predates the proposal and is not
  cited. It fuses exact sampling into the LM-head matmul with the Gumbel-max
  trick, keeping one candidate per vocabulary tile and never materialising
  logits, and reports up to 19% lower time per output token in vLLM.
  Optimization 2 is therefore not novel, and Gumbel-argmax is a parallel
  reduction that samples the full distribution exactly, which a serial top-k
  merge does not.
- **Structured-decoding baselines.** vLLM's default backend is `auto`, which
  tries XGrammar first and falls back to guidance
  (<https://docs.vllm.ai/en/stable/features/structured_outputs/>; the
  ordering is from a forum answer, a secondary source:
  <https://discuss.vllm.ai/t/general-questions-on-structured-output-backend/1444>).
  SGLang defaults to XGrammar, with llguidance and Outlines optional
  (<https://docs.sglang.ai/advanced_features/structured_outputs.html>).
  These are native engines, and XGrammar and llguidance accept EBNF and
  context-free grammars, which a baked DFA cannot express. The comparison
  target for §4 is a native mask engine, not a Python interpreter.
- **Blackwell.** Datacenter Blackwell (sm_100a/103a) has the `tcgen05.mma`
  tensor-core instruction with block-scaled FP4 and FP8 kinds
  (<https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/blackwell_functionality.html>).
  Consumer Blackwell (sm_120/121, which includes the project's RTX 5070 Ti)
  is reported not to expose it
  (<https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607>).
  CFIE's kernels target the sm_70 floor with scalar loads, so they use none
  of this.
- **KVTuner** (ICML 2025,
  <https://proceedings.mlr.press/v267/li25dd.html>). It finds that per-layer
  KV sensitivity is specific to the model and has to be searched offline.
  §2.5's positional prior (first and last layers FP16, middle layers INT8)
  and its weight-spectrum heuristic have no support there.

---

## Part 3 — What the code implements

CFIE is implemented in part. It is switched on by any of the serve-block keys
`kv_layout`, `kv_quant`, `sampling:` with `fused`, or `speculative:` (any one
activates full mode), by the command-line flags
`--cfie [full|sampling|off]` and `--cfie-report PATH`, or by the decorator
`@cfie(mode=, target=)`. Serve-block parsing is in
`crates/nsl-codegen/src/cfie_serve.rs`, kernel emission is wired in
`crates/nsl-codegen/src/serve.rs`, and the runtime driver is
`crates/nsl-runtime/src/cfie/engine.rs` (`nsl_cfie_generate`,
`nsl_cfie_decode_step`, `nsl_cfie_speculative_generate`).

A build carries two kernels, the decode block and the fused sampler, and two
more for a draft model when `speculative:` is configured. Every kernel a
build registers is launched by a generation driver. The engine serves one
sequence at a time, and each kernel runs on one thread block.

No shipped model uses CFIE: `models/coder50m/generate.nsl` re-runs the full
forward pass for every token, with no KV cache, and samples with the standard
library's `sample_top_k`. CFIE binds only checkpoints with HF-Llama weight
names, stored `[out][in]`; a checkpoint saved by an NSL model has other
names and the other layout, and there is no exporter. The CFIE GPU tests are
marked `#[ignore]` and run on synthetic toy weights. The repository contains
no measurement of CFIE's speed, and the build report prints none.

The codegen files named below are under `crates/nsl-codegen/src/`.

- **Static KV layout (§2.1; key `kv_layout`) — implemented, narrower than
  proposed.** `cfie_kv_plan.rs` plans a contiguous half-precision pool laid
  out as `[layers][2][slots × per_slot][kv_heads][head_dim]`, with a slot
  allocator on the host, when `max_batch` sequences of `max_seq` tokens fit
  the GPU's memory. When they do not, the plan is paged and no decode
  kernel is emitted; `kv_layout: "static"` is then a compile error. There
  is no bump allocator.
- **Fused decode-sample (§2.2; sub-block `sampling:`) — implemented.**
  `cfie_fused_sample.rs` and `cfie_sample_ptx.rs` emit one CTA of 128
  threads. Top-k is a serial replace-minimum on one thread, k is at most 64
  (a larger `top_k` is a compile error), and the vocabulary size must be a
  multiple of 128. Temperature, top-k and top-p are fixed at compile time,
  and the `params` argument of `generate()` is ignored, which is
  documented. A non-zero temperature must come with a `top_k`; `top_p` is
  optional. Each token has its own random draw, keyed by `sampling: seed:`,
  the request's index in the process and the token's position. There is no
  repetition penalty. The code differs from the proposal in two places and
  is right in both: its nucleus rule keeps the crossing token (E5), and its
  softmax maximum is taken over the candidates rather than over all logits.
- **Compiled speculative decoding (§2.3; sub-block `speculative:`) —
  implemented for a greedy draft model, with no speedup.** A draft model
  compiled into the binary (`draft_weights`) proposes K tokens greedily
  (`cfie_spec_sampler_ptx.rs` is its sampler). The target is then stepped
  through the same positions, one at a time, and its own fused sampler
  draws each token; a drawn token that equals the drafted one confirms the
  next drafted input, and the first that differs ends the round
  (`speculative_rounds` in the engine). For a greedy draft this is the
  correct rule of E1 — the drafted token is emitted with the target's
  probability of it, and otherwise the token comes from the target with
  the drafted one removed — so the output is exactly what plain decoding
  emits. The proposal's single batched verification pass does not exist,
  so a round costs K draft steps on top of one target step per token:
  speculation is slower than plain decoding, and the build report says so.
  The `tree`, `medusa` and `lookahead` methods, `tree_width`, a draft
  temperature and a `speculative:` section with no draft model are compile
  errors.
- **Persistent decode kernel (§2.4) — partial.** `cfie_persistent.rs` and
  `cfie_persistent_ptx.rs` emit a one-CTA whole-layer kernel, and only when
  its static shared-memory footprint is at most 48 KB (a model of d_model
  4096 does not fit, and `generate()` is then a compile error). Fusion
  levels 1 and 2 are planner verdicts with no kernel. The kernel is
  launched once per layer per token and exits; there is no on-device
  scheduler and no request queue. Its rotary embedding pairs element j of
  a head with element j + head_dim/2, as `stdlib/nsl/nn/rope.nsl` and
  HF-Llama checkpoints do. The one-CTA design is the proposal's, and it is
  the problem (E2).
- **Per-layer KV precision (§2.5; key `kv_quant`) — advisory only.** The KV
  pool is half precision in every layer. `cfie_kv_quant.rs` computes a
  per-layer ranking that `kv_quant: "auto"` prints in the build report,
  labelled as not applied; the ranking is relative to the model's other
  layers and is not a measure of sensitivity. `kv_quant: "uniform_int8"`
  is a compile error: no kernel writes or reads INT8 KV.
- **CSHA decode fusion (§2.6) — subsumed** by the persistent decode kernel.
- **Grammar-constrained decoding (§2.2, §3; sub-block `grammar:`) — not
  available.** A `grammar:` section is a compile error. `cfie_grammar.rs`
  and `cfie_grammar_ptx.rs` can build a byte-level DFA from a JSON Schema
  (`schema_convert.rs`), project it onto tokens, minimise it and bake one
  token bitmask per state, and the sampler kernel can apply such a mask,
  but no driver advances the automaton from token to token or handles
  end-of-sequence, so none of it is reachable from a serve block. The
  baked masks are capped at 16 × 2^20 cells, about 340 states at a
  49,152-token vocabulary.
- **WGGO integration (§5) — report-only.** WGGO (Wengert Graph Global
  Optimization) is NSL's joint optimisation pass. `wggo_cfie.rs` is
  advisory, produces a report only, and is off by default.
- **Build report (§3) — implemented, with different content.** It states
  which kernels the build carries or why one is missing, the launches per
  token the decode loop makes (layers + 1) beside a count of the
  operations an unfused pipeline would launch, and that speed is not
  measured. It prints no latency, throughput or speedup.

Three older, separate paths cover some of the same ground:
`@speculative(...)` with the `speculative_decode(...)` intrinsic,
`@kv_compress(...)`, and the `nsl_grammar_*` runtime functions.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. In §2.3, use q(x*) = 1 for a greedy draft and resample from
   max(0, p − q), with the sampler's temperature, top-k and top-p (the code
   now does). Then batch the K verification positions into one launch:
   without that, speculation costs time.
2. Replace the one-CTA block with a few multi-CTA kernels whose loads are
   coalesced along weight rows (estimated 10–50× at 50M scale; not measured).
3. Use Gumbel-max sampling, as FlashSampling does, in place of serial top-k.
4. Step the grammar automaton on each sampled token and allow end-of-sequence
   only in accepting states; for general schemas, a pushdown or Earley engine.
5. Before any speed claim, test CFIE's logits against the NSL model's own
   forward pass, then measure tokens per second and redo §1, §2.4 and §3.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code. This part lists what it found wrong in the
implementation, as distinct from the proposal's own errors in Part 1. Fixed
items are described in `CHANGELOG.md` under "Unreleased"; the open list is as
of 2026-11-13. Each open item ends with how it was established: reproduced by
running, numeric check, or by reading the code only. Below, "the engine" is
`crates/nsl-runtime/src/cfie/engine.rs`, and file names given without a
directory are under `crates/nsl-codegen/src/`.

### Fixed

- The standard library's `sample_top_k(logits, k, 0.0)` and
  `sample_top_p(logits, p, 0.0)` divided by a zero temperature and aborted in
  the multinomial draw. Temperature 0 is now greedy
  (`stdlib/nsl/inference/sampling.nsl`; test
  `crates/nsl-cli/tests/sampling_temperature_zero_gate.rs`).
- Every token was drawn at the same quantile. `generate()` passed seed 0 and
  every step reused it, so the sampler's uniform draw was 0.0528 at each
  step: "temperature 0.7, top-p 0.9" picked the top token whenever it held
  5.3% of the kept mass. A request now has a key made from the configured
  seed (`sampling: seed:`, default 0) and its index in the process, and the
  draw at each position is seeded from the key and the position
  (`request_key`, `step_seed`, `nsl_cfie_decode_step` in the engine).
  `crates/nsl-codegen/tests/cfie_sampling_distribution.rs` runs the sampler
  kernel for 3,000 positions and compares the token counts with the exact
  probabilities; `cfie_speculative_generate_gpu_e2e.rs` does the same on the
  GPU through `nsl_cfie_generate`.
- Speculative decoding did not sample the target distribution. The draft is
  greedy, yet a token was accepted when r < p_target/p_draft with p_draft
  the draft's softmax value: target [0.3, 0.7] with draft softmax
  [0.6, 0.4] gave [0.5, 0.5]. The rejection kernel is gone. The target's
  own sampler now draws the token at each drafted position, and the round
  continues while the draw equals the draft (`speculative_rounds` in the
  engine), which for a greedy draft is the exact rule of E1. The engine's
  unit tests enumerate the emitted token over a grid of quantiles and find
  each token at its target probability, and check that the output is plain
  decoding's for every draft tried.
- Speculative verification ignored the sampler's settings: acceptance and
  correction used a softmax at temperature 1 with no top-k or top-p, so
  with `temperature: 0.0` the target's argmax was emitted with probability
  0.44 or 0.69 where it should be 1. Verification is now the configured
  sampler itself; the kernel that wrote the temperature-1 row is gone. The
  GPU test requires speculative output to equal plain decoding for a
  greedy target and for a temperature-0.7, top-k, top-p target, with a
  draft that mispredicts.
- A grammar restricted every token to the start state's token set: the
  driver always passed state 0 while the baked mask was bound, so
  `{"type":"boolean"}` yielded an endless run of `true`/`false` tokens. A
  `grammar:` section is now a compile error that says the automaton is not
  stepped (`validate_serve_config` in `crates/nsl-semantic/src/cfie.rs`,
  `prepare` in `cfie_serve.rs`).
- A grammar state with no allowed token emitted token id 0: every candidate
  was masked to minus infinity, the probabilities became NaN, and the walk
  fell back to the first candidate slot. The sampler kernel now writes an
  id outside the vocabulary (`NO_CANDIDATE` in `cfie_sample_ptx.rs`) when
  it has no candidate, which also covers a model whose output is NaN, and
  the decode loop treats that id as a failure.
- The decode block's RoPE was not the convention NSL models use. It rotated
  adjacent pairs (2p, 2p+1); `stdlib/nsl/nn/rope.nsl` and HF-Llama
  checkpoints rotate pairs (j, j + d/2), and the same q and k gave
  different attention. The kernel and its CPU reference now use the
  half-split pairing (`rope_pairs` in `cfie_persistent_ptx.rs`), and
  `cfie_persistent_kir_equivalence.rs` checks the kernel's rotated keys
  against values NumPy computed from the standard library's formula.
- Weight binding compared element counts, so a matrix stored `[in][out]`
  bound silently and was read transposed. Rank and every dimension are
  compared now, and a transposed matrix is refused and named as such
  (`check_weight_shape` in the engine). What is still missing is under
  Open.
- `norm_eps` did not reach the sampler kernels: the fused sampler and the
  draft sampler baked 1e-5 for the final norm. Both take the configured
  value (`FusedSampleKernelConfig::norm_eps`,
  `SpecSamplerConfig::norm_eps`).
- The KV-precision planner's score was presented as a sensitivity and
  applied. It is min-max normalised across layers, so some layer always
  scores 0 and some layer 1 whatever the real sensitivity. Nothing applies
  the plan any more; `kv_quant: "auto"` prints it in the build report as an
  advisory ranking, with that caveat in the line. A unit test gives the
  same plan to a model whose most prominent layer exceeds the others by
  0.2% and to one where it exceeds them 200-fold (`cfie_kv_quant.rs`).
- The JSON-schema grammar was not JSON: integers allowed leading zeros
  (`00`), strings were printable ASCII with no escapes, every property was
  required, and whitespace was an unbounded run of spaces. Integers and
  numbers now follow the JSON number grammar, strings take JSON's escapes
  and well-formed UTF-8, `required` decides which properties are
  mandatory, and at most one space is allowed where whitespace may go
  (`schema_convert.rs`). Its tests enumerate what the grammar accepts and
  parse every string with `serde_json`. Members are accepted in one order
  only (the property names sorted), which is narrower than JSON Schema.
- Common configurations built a binary whose `generate()` always refused:
  `kv_layout: "static"` with no `kv_quant` key selected read-only quantised
  kernels and excluded the decode block; d_model 4096 exceeds the 48 KB
  shared-memory limit; `top_k: 100` skipped the sampler with only a note.
  The first now builds the decode path. In the other cases, and whenever
  the decode block or the sampler is not in the build, a `generate()` call
  is a compile error that names the missing kernel and the reason
  (`run_cfie_for_serve` in `serve.rs`; `cfie_serve_end_to_end.rs`).
- `static_with_bump` sized the pool for the full envelope it had found too
  big (a 137 GB pool on an 80 GB card in one build) and said a GPU-side
  bump allocator would handle it; none existed. The layout kind is gone.
  An envelope that does not fit is a paged plan with no decode kernel, and
  `kv_layout: "static"` with such an envelope is a compile error that
  gives both sizes (`plan` in `cfie_kv_plan.rs`).
- Sampling keys were filled in or dropped without notice: a `sampling:`
  section with only `temperature` got top_k = 50 and top_p = 0.9, and the
  `speculative:` key `temperature` was stored and never read. A non-zero
  temperature without `top_k` is now an error, a missing `top_p` means no
  nucleus filter, and a non-zero `speculative: temperature`, `tree_width`,
  the `tree`, `medusa` and `lookahead` methods, the `draft` key, and a
  `speculative:` section with no `draft_weights` are errors. With no
  `sampling:` section the defaults still apply, and the report names them
  as defaults.
- The build report's launch counts were placeholders: a hard-coded baseline
  of 500 and "407 (vs 500 baseline)" for an 8-layer model that makes 9
  launches per token. The report now prints the decode loop's count
  (layers + 1) and, beside it, the planner's count for an unfused pipeline
  (12 per layer + 6), labelled as a count of operations
  (`kernel_launches_per_token` in `cfie.rs`).
- The report described a batched mode that does not exist ("Continuous
  batching: on", "Estimated throughput (batch=64)") and estimated latency
  by dividing by the whole GPU's bandwidth. Those lines and the cost model
  behind them (`cfie_cost.rs`) are removed. The report says the engine
  serves one sequence at a time and that speed is not measured.
- The report contradicted itself on the decode block ("registered at serve
  init" two lines above "decode-block/spec kernels NOT registered"). Each
  kernel now has one line: emitted and registered, or not emitted and why.
- The report promised a speculative speedup ("expected speedup=1.50×", from
  1 + K·p/(1 + K) with a placeholder p = 0.6) that the driver cannot give.
  The figure is removed; the speculative section states the launches per
  round and that speculation is slower than plain decoding.
- Prefill ran the full LM-head sampler on every prompt token and discarded
  the result. Only the last prompt position runs the sampler now
  (`target_prefill` in the engine).
- The speculative GPU test could not detect the acceptance defect: its CPU
  reference applied the same rule as the kernel. That reference is deleted
  with the kernel. `cfie_speculative_generate_gpu_e2e.rs` now compares
  decoded token counts with exact probabilities and requires speculative
  output to equal plain decoding.
- Three kernel kinds were emitted and never launched, and a request ring
  was initialised and never consumed. The tree-mask verification kernel
  and the per-layer INT8/FP16 read kernels are deleted with their launch
  functions and tests; the request ring and its `nsl_cfie_ring_*` functions
  are deleted. The standalone decode-attention kernel is no longer emitted
  into any binary or registered (`nsl_cfie_launch_decode_attn` is deleted);
  its builder remains in `cfie_decode_attention.rs` because the decode
  block is assembled from its sections, and its interpreter and assembler
  tests remain as the tests of those sections.

In the older paths:

- `@speculative` draft sampling was not random: the draft sampler derived
  its "random" number from the step index alone (0.6322, 0.9495, 0.2376 for
  steps 0, 1, 2 of every request). It now draws from the runtime's seeded
  generator (`stochastic_sample` in
  `crates/nsl-runtime/src/speculative/draft.rs`).
- `@speculative` verification was seeded by the first draft token id. It
  takes a seed from the runtime's generator
  (`nsl_speculative_decode_step`). A unit test drafts from one distribution,
  verifies against another and checks 20,000 emitted tokens against the
  target's probabilities.
- `@speculative` tree verification accepted any token whose target
  probability exceeded 0.01 at a non-zero temperature and never resampled.
  `nsl_speculative_verify_tree` now refuses a non-zero temperature;
  greedy tree verification is unchanged.
- `nsl_grammar_init` built an automaton that masked every token. It now
  refuses: it is given no transitions to build from. A grammar is installed
  from a real table with `install_grammar` in
  `crates/nsl-runtime/src/grammar.rs`.
- Two sampling tests asserted nothing (`tests/m19_topk_test.nsl`,
  `tests/test_sampling.nsl`) and no harness ran them. They are replaced by
  `crates/nsl-cli/tests/sampling_distribution_gate.rs`, which checks the
  standard library samplers' token counts against exact probabilities and
  that a seed reproduces a draw.

### Open

- **An NSL-trained checkpoint cannot be served.** Binding accepts HF-Llama
  weight names and reads matrices `[out][in]`. An NSL checkpoint
  (`[in, out]`, names like `blocks.i.attn.wq`, tied head) is refused by
  name, and there is no exporter. A square matrix stored `[in][out]` under
  an HF-Llama name has the expected shape and cannot be detected.
  CFIE's logits have never been compared with an NSL model's own forward
  pass; that comparison is the test that would close this.
  `resolve_and_upload_model` (the engine). By reading the code only.
- **Decode kernels read weights uncoalesced.** Each of 128 threads walks its
  own weight row with scalar half-precision loads, so a warp's 32 loads hit
  32 rows; estimated, not measured, at milliseconds per token for the 50M
  model. This is the one-CTA design of E2. `dot_rows`
  (`cfie_persistent_ptx.rs`). By reading the code only.
- **Speculative decoding is slower than plain decoding.** The target
  verifies one position per step, so a draft adds work and saves none. It
  needs a batched verification pass (Part 4, item 1). The output is
  correct, and the build report states the cost.
  `nsl_cfie_speculative_generate` (the engine). By reading the code only.
- **The WGGO advisory model still credits speculation with a speedup.**
  `wggo_cfie.rs` divides a layer's cost by 1 + K·p/(1 + K) when it
  considers speculative decoding. The model is advisory and off by
  default. By reading the code only.
