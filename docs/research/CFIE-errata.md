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
the same cases was 0.17 to 0.58. The shipped code follows the pseudo-code on
this point (Part 3).

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
`kv_layout`, `kv_quant`, `sampling:` with `fused`, `speculative:` or
`grammar:` (any one activates full mode), by the command-line flags
`--cfie [full|sampling|off]` and `--cfie-report PATH`, or by the decorator
`@cfie(mode=, target=)`. Serve-block parsing is in
`crates/nsl-codegen/src/cfie_serve.rs`, kernel emission is wired in
`crates/nsl-codegen/src/serve.rs`, and the runtime driver is
`crates/nsl-runtime/src/cfie/engine.rs` (`nsl_cfie_generate`,
`nsl_cfie_decode_step`).

No shipped model uses CFIE: `models/coder50m/generate.nsl` re-runs the full
forward pass for every token, with no KV cache, and samples with the standard
library's `sample_top_k`. The CFIE GPU tests are marked `#[ignore]`, run on
synthetic toy weights, and compare against CPU mirrors of the same algorithm.
The repository contains no measurement of CFIE's speed.

The codegen files named below are under `crates/nsl-codegen/src/`.

- **Static KV layout (§2.1; key `kv_layout`) — implemented, narrower than
  proposed.** `cfie_kv_plan.rs` plans a contiguous half-precision pool laid
  out as `[layers][2][slots × per_slot][kv_heads][head_dim]`, with a slot
  allocator on the host. The GPU-side bump allocator that the
  `static_with_bump` layout names does not exist; it is a descriptor that
  nothing consumes. A standalone decode-attention kernel
  (`cfie_decode_attention.rs`) is registered and never launched.
- **Fused decode-sample (§2.2; sub-block `sampling:`) — implemented.**
  `cfie_fused_sample.rs` and `cfie_sample_ptx.rs` emit one CTA of 128
  threads. Top-k is a serial replace-minimum on one thread, k is at most 64,
  and the vocabulary size must be a multiple of 128. Temperature, top-k and
  top-p are fixed at compile time, and the `params` argument of `generate()`
  is ignored, which is documented. There is no repetition penalty. The code
  differs from the proposal in two places and is right in both: its nucleus
  rule keeps the crossing token (E5), and its softmax maximum is taken over
  the candidates rather than over all logits including grammar-masked ones.
- **Compiled speculative decoding (§2.3; sub-block `speculative:`) —
  partial.** `cfie_speculative.rs`, `cfie_speculative_ptx.rs` and
  `cfie_spec_sampler_ptx.rs` implement a linear greedy draft only. The
  target verifies with K sequential single-token passes, not the proposal's
  single batched forward. A tree-mask verification kernel is emitted and
  registered and never launched; the `tree`, `medusa` and `lookahead`
  methods are refused when `draft_weights` is set and are plan-only
  otherwise. The keys `speculative.temperature`, `min_acceptance_rate` and
  `share_kv` are parsed and never read. On the acceptance rule neither side
  is right: the code implements the proposal's rule (E1).
- **Persistent decode kernel (§2.4) — partial.** `cfie_persistent.rs` and
  `cfie_persistent_ptx.rs` emit a one-CTA whole-layer kernel, and only when
  its static shared-memory footprint is at most 48 KB. Fusion levels 1 and 2
  are plan-only. There is no on-device scheduler: the request ring is
  initialised and never consumed, and the binary serves one sequence at a
  time. The one-CTA design is the proposal's, and it is the problem (E2).
- **Per-layer KV precision (§2.5; key `kv_quant`) — partial, read side
  only.** `cfie_kv_quant.rs` plans a per-layer FP16-or-INT8 choice and
  `cfie_kv_quant_ptx.rs` emits the matching read kernels. No kernel writes
  INT8 KV, and nothing launches the read kernels outside tests. The INT8
  scale is a launch-time parameter rather than a baked constant, which is
  the better choice.
- **CSHA decode fusion (§2.6) — subsumed** by the persistent decode kernel.
- **Grammar-constrained decoding (§2.2, §3; sub-block `grammar:`) —
  partial.** `cfie_grammar.rs` and `cfie_grammar_ptx.rs` build a byte-level
  DFA, project it onto tokens, minimise it, and bake one token bitmask per
  state into the kernel module; the sampler applies the mask. The transition
  table is not shipped to the runtime and the automaton's state stays at 0,
  so the grammar is not advanced from token to token. The baked masks are
  capped at 16 × 2^20 cells, about 340 states at a 49,152-token vocabulary.
- **WGGO integration (§5) — report-only.** WGGO (Wengert Graph Global
  Optimization) is NSL's joint optimisation pass. `wggo_cfie.rs` is
  advisory, produces a report only, and is off by default.

Three older, separate paths cover some of the same ground:
`@speculative(...)` with the `speculative_decode(...)` intrinsic,
`@kv_compress(...)`, and the `nsl_grammar_*` runtime functions.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Fix §2.3 first: use q(x*) = 1 for a greedy draft and resample from
   max(0, p − q); give verification the same temperature, top-k and top-p
   as the sampler; batch the K verification positions into one launch.
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

Nothing else below has been fixed.

### Open

- **Every token is drawn at the same quantile.** `generate()` passes seed 0
  and every step reuses it, so the sampler's uniform draw is 0.0528 at each
  step: "temperature 0.7, top-p 0.9" picks the top token whenever it holds
  5.3% of the kept mass. Each request and position should get its own draw.
  `compile_call` (`expr/calls.rs`); `nsl_cfie_generate` and
  `nsl_cfie_decode_step` (the engine); `build` (`cfie_sample_ptx.rs`).
  Numeric check.
- **Speculative decoding does not sample the target distribution.** The
  draft is greedy, yet a token is accepted when r < p_target/p_draft with
  p_draft the draft's softmax value. Target [0.3, 0.7] with draft softmax
  [0.6, 0.4] gives [0.5, 0.5]; p_draft = 1 gives the target (E1).
  `build_draft_sample` (`cfie_spec_sampler_ptx.rs`), `build_rejection`
  (`cfie_speculative_ptx.rs`). Numeric check.
- **Speculative verification ignores the sampler's settings.** Acceptance
  and correction use a softmax at temperature 1 with no top-k or top-p. With
  `temperature: 0.0` and target softmax [0.55, 0.45], the target's argmax is
  emitted with probability 0.44 against draft softmax [0.2, 0.8] and 0.69
  against [0.8, 0.2]; it should be 1. `build_verify_probs`
  (`cfie_spec_sampler_ptx.rs`). Numeric check (greedy case only).
- **A grammar restricts every token to the start state's token set.** The
  driver always passes state 0 (its comment says "unconstrained") while the
  baked mask is bound, so `{"type":"boolean"}` yields an endless run of
  `true`/`false` tokens; nothing handles end-of-sequence. It should step
  the automaton per token, or refuse at compile time. `nsl_cfie_generate`,
  `nsl_cfie_engine_finalize` (the engine). By reading the code only.
- **A grammar state with no allowed token emits token id 0.** Every
  candidate is masked to minus infinity, the probabilities become NaN, and
  the walk falls back to the first candidate slot. An empty mask row should
  be an error. `build` (`cfie_sample_ptx.rs`). By reading the code only.
- **The decode block's RoPE is not the convention NSL models use.** It
  rotates adjacent pairs (2p, 2p+1); `stdlib/nsl/nn/rope.nsl` rotates pairs
  (i, i + d/2). The same q and k give different attention probabilities,
  equal only after permuting the projection rows. The GPU tests' CPU
  reference uses the same adjacent-pair rule, so they pass. `rope_pairs`
  (`cfie_persistent_ptx.rs`). Numeric check.
- **Weight binding checks names and element counts, not layout.** Only
  HF-Llama names are accepted and `[out][in]` is assumed, so a square
  `[in][out]` matrix binds silently. An NSL checkpoint (`[in, out]`, names
  like `blocks.i.attn.wq`, tied head) is refused by name, and there is no
  exporter. `resolve_and_upload_model`, `upload_named` (the engine). By
  reading the code only.
- **`norm_eps` does not reach the sampler kernels.** The decode block takes
  the configured epsilon; the fused sampler and the speculative sampler
  kernels bake 1e-5 for the final norm. `RMS_EPS` (`cfie_sample_ptx.rs`,
  `cfie_spec_sampler_ptx.rs`). By reading the code only.
- **The KV-precision planner's score is relative, not absolute.** The
  largest singular values are min-max normalised across layers, so some
  layer always scores 0 and some layer 1 whatever the real sensitivity.
  `min_max_normalize` (`cfie_kv_quant.rs`). By reading the code only.
- **The JSON-schema grammar is not JSON.** Integers allow leading zeros
  (`00`); strings are printable ASCII with no escapes; every property is
  required, in map-iteration order; whitespace is an unbounded run of
  spaces. `integer_element`, `string_element`, `convert_object`
  (`schema_convert.rs`). By reading the code only.
- **Common configurations build a binary whose `generate()` always
  refuses.** `kv_layout: "static"` with no `kv_quant` key selects the
  quantised kernels, which exclude the decode block; d_model 4096 exceeds
  the 48 KB gate; `top_k: 100` skips the sampler with only a note. It should
  be a compile error. `run_cfie_for_serve` (`serve.rs`). Reproduced by
  running (the builds); the run-time refusal by reading the code only.
- **`static_with_bump` sizes the pool for the full envelope it found too
  big.** The plan says a GPU-side bump allocator handles fragmentation; none
  exists, and direct-index kernels are emitted for max_batch × max_seq: a
  137 GB pool on an 80 GB card in the case built. `plan`
  (`cfie_kv_plan.rs`), `run_cfie_for_serve` (`serve.rs`). Reproduced by
  running (the build); the pool formula by reading the code only.
- **Sampling keys are filled in or dropped without notice.** A `sampling:`
  section with only `temperature` gets top_k = 50 and top_p = 0.9. The
  `speculative:` key `temperature` is stored and never read; the draft is
  always greedy. `sampling_params`, `extract_speculative`
  (`cfie_serve.rs`). By reading the code only.
- **The build report's launch counts are placeholders.** The baseline is a
  hard-coded 500. An 8-layer example prints "407 (vs 500 baseline)" and
  "2069.4us/token", of which 2,035 µs is 407 × 5 µs; the real path makes
  n_layers + 1 = 9 launches, and the planner's own baseline is 102. `run`
  (`cfie.rs`), `estimate` (`cfie_cost.rs`). Reproduced by running.
- **The report describes a batched mode that does not exist.** It prints
  "Continuous batching: on" and "Estimated throughput (batch=64)" for a
  binary that serves one sequence at a time, and its memory term divides by
  the whole GPU's peak bandwidth although the kernels are single-CTA.
  `render_report` (`cfie.rs`), `estimate` (`cfie_cost.rs`). Reproduced by
  running.
- **The report contradicts itself on the decode block.** For the shipped
  `examples/cfie_serve_static.nsl` it prints that the persistent decode
  block is "registered at serve init" two lines above "decode-block/spec
  kernels NOT registered". `render_report` (`cfie.rs`). Reproduced by
  running.
- **The report promises a speculative speedup the driver cannot give.** A
  round runs K draft steps, K sequential full target steps and a serial
  vocabulary walk for at most K + 1 tokens: slower than plain decoding. The
  report prints "expected speedup=1.50×" from 1 + K·p/(1 + K) with a
  placeholder p = 0.6. `nsl_cfie_speculative_generate` (the engine),
  `expected_speedup` (`cfie_speculative.rs`). By reading the code only.
- **Decode kernels read weights uncoalesced, and prefill samples.** Each of
  128 threads walks its own weight row with scalar half-precision loads, so
  a warp's 32 loads hit 32 rows; estimated, not measured, at milliseconds
  per token for the 50M model. Prefill also runs the full LM-head sampler on
  every prompt token and discards the result. `dot_rows`
  (`cfie_persistent_ptx.rs`), `nsl_cfie_generate`. By reading the code only.
- **The speculative GPU test cannot detect the acceptance defect.** Its CPU
  reference applies the same rule as the kernel, so a draft that differs
  from the target still passes. `cpu_reference_reject`
  (`cfie_speculative_ptx.rs`), used by
  `crates/nsl-codegen/tests/cfie_speculative_generate_gpu_e2e.rs`. By reading
  the code only.
- **Three kernel kinds are emitted and never launched.** The standalone
  decode-attention kernel, the tree-mask verification kernel and the
  per-layer INT8/FP16 read kernels are launched only from tests; the request
  ring is initialised and never consumed. `nsl_cfie_launch_decode_attn`,
  `nsl_cfie_launch_spec_verify`, `nsl_cfie_launch_quant_attn` (the engine).
  By reading the code only.

#### Older paths in scope

These predate CFIE and are separate from it (Part 3, last paragraph).

- **`@speculative` draft sampling is not random.** The draft sampler derives
  its "random" number from the step index alone: 0.6322, 0.9495, 0.2376 for
  steps 0, 1, 2 of every request. `stochastic_sample` in
  `crates/nsl-runtime/src/speculative/draft.rs`. Numeric check.
- **`@speculative` verification is seeded by the first draft token id.**
  `nsl_speculative_decode_step` in
  `crates/nsl-runtime/src/speculative/ffi.rs`. By reading the code only.
- **`@speculative` tree verification is not rejection sampling.** At a
  non-zero temperature it accepts any token whose target probability
  exceeds 0.01 and never resamples. `nsl_speculative_verify_tree` in
  `crates/nsl-runtime/src/speculative/ffi.rs`. By reading the code only.
- **`nsl_grammar_init` builds an automaton that masks every token.** It has
  no transitions and all-zero masks, so `nsl_grammar_apply_mask` masks the
  whole vocabulary. It has no production caller.
  `crates/nsl-runtime/src/grammar.rs`. By reading the code only.
- **Two sampling tests assert nothing.** `tests/m19_topk_test.nsl` prints
  "deterministic RNG: PASS" without comparing its two seeded draws, and
  `tests/test_sampling.nsl` prints its samples and "PASS" with no check. By
  reading the code only.
