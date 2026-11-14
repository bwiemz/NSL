# Compile-Planned Distributed Training (CPDT) — Errata

**Companion to** `CPDT Research.pdf`. **Date:** 2026-11-13.

The proposal was reviewed in November 2026 against its own arithmetic, the
papers and libraries it cites, and the CPDT code in this repository. This
file lists the statements in the PDF that should not be relied on and says
what is true instead. Arithmetic was recomputed from the proposal's own
figures. External facts were checked against the linked source unless the
entry says otherwise. Statements about the code describe the tree on the date
above.

---

## Part 1 — Errors in the proposal

### E1 — An FP16 second moment is zero for gradients of ordinary size

**Proposal.** §3.2, fused-optimizer pseudocode:
`v_5_fp16 = β₂ × v_5_fp16 + (1-β₂) × (g_5² ).to_fp16()`. The §3.1 tier table
gives the Low tier the same format, "INT8 m, FP16 v".

**Correction.** Adam's second moment `v` accumulates (1−β₂)·g². The smallest
positive FP16 value is 2⁻²⁴ ≈ 5.96e-8, and anything at or below half of it,
2.98e-8, rounds to zero. With the usual β₂ = 0.999 and g = 1e-3 the increment
is 1e-9, which is stored as zero. A `v` that is zero can only leave zero if
0.001·g² > 2.98e-8, that is if |g| > 5.5e-3. A model with a global gradient
norm of about 1 and P parameters has a root-mean-square element gradient of
1/√P: 1.4e-4 for 50M parameters and 1.2e-5 for 7B. Almost every element is
below the threshold, so its `v` is identically zero.

Where `v` is representable there is a second problem. FP16's relative spacing
(2⁻¹¹ to 2⁻¹⁰) is about the size of 1−β₂ (≈ 2⁻¹⁰), so round-to-nearest stalls
the moving average. In simulation this biases the step size by a few percent
in FP16's normal range and by more than 20% in its subnormal range.

**Source.** Numeric check: IEEE half-precision limits, and a simulation of
the update with round-to-nearest FP16 storage.

### E2 — No mix of the four precision tiers can average 1.3 bytes per parameter

**Proposal.** §3.1, memory comparison: CPDT at "(1.3 bytes/param average)"
against "INT8 Adam (bitsandbytes uniform)" at "97.5 MB (2 bytes/param)",
followed by "The additional 30% savings over uniform INT8 comes from using
INT8 for BOTH m and v on low-sensitivity parameters (FFN middle layers),
which bitsandbytes doesn't support".

**Correction.** The cheapest tier in the table directly above is "INT8 m,
INT8 v" at 2 bytes. Every plan built from that table therefore averages at
least 2 bytes per parameter and uses at least as much memory as uniform 8-bit
bitsandbytes. The explanation also contradicts the comparison row: uniform
INT8 Adam at 2 bytes per parameter already is INT8 for both `m` and `v`.

### E3 — The sample plan's figures do not follow from its own inputs

**Proposal.** §6.2, the sample `--cpdt-plan` output for "NSLCoder-50M (48.8M
params)".

**Correction.** Line by line:

- "embed (49K vocab × 512): FP32 m+v, stochastic rounding (2.0 MB)".
  49,000 × 512 = 25.1M parameters, and 25.1M × 8 bytes = 201 MB. Nothing is
  rounded at FP32, so "stochastic rounding" means nothing on this line.
- "blocks.0-1 (attention+FFN): FP16 m, FP32 v (1.8 MB each)". The other
  23.7M parameters over 8 blocks are about 2.96M per block, and 2.96M × 6
  bytes = 17.8 MB. Every block row is ten times too low; the 3-byte rows
  ("0.9 MB each") are 8.9 MB.
- "Total optimizer memory: 14.6 MB / GPU (vs 48.8 MB uniform FP32)". The rows
  as printed sum to 12.8 MB, not 14.6. Uniform FP32 is 390 MB by the §3.1
  table (48.8M × 8 bytes).
- "1.2 GB (model)". 48.8M parameters at FP32 are 195 MB.
- "Effective: ZeRO-1 equivalent with intra-node gradient reduction". With
  s_p = 1, s_g = 8, s_os = 8 the gradients are sharded as well as the
  optimizer state, which is ZeRO-2 by the standard definition.

Recomputed with the plan's own tiers, the optimizer state is about 307 MB
(6.3 bytes per parameter). That is a 21% saving over FP32, and about three
times the 97.6 MB of uniform 8-bit bitsandbytes.

**Source.** Recomputed from the proposal's own figures.

### E4 — The communication model charges plain data-parallel training nothing

**Proposal.** §2, communication volume per step:
`backward_reducescatter = Σ |W_l| × dtype_bytes × (s_g > 1)`, with the same
form for the forward and optimizer all-gathers.

**Correction.** With all three sharding factors equal to 1 the total is zero,
but plain distributed data-parallel training (DDP) must all-reduce its
gradients every step. The paper that defines ZeRO (Zero Redundancy Optimizer)
gives communication volumes of 2Ψ for baseline data parallelism, 2Ψ with
optimizer state and gradients sharded (ZeRO-2), and 3Ψ with parameters
sharded too (ZeRO-3), where Ψ is the model size. ZeRO-2 never costs more
communication than DDP; the formula makes DDP free and ZeRO-2 expensive.

Three things in the cost model of AMSP, the system the proposal compares
itself with, are also missing:

- Parameter sharding costs two all-gathers and a reduce-scatter per
  micro-batch (T_p = M·L·Σ[2·AG + RS]), not one all-gather per step.
- There is no micro-batch factor M.
- There is no per-operation latency α. AMSP's cost of a collective is
  (p−1)(α + v/(w·p)), and α is what makes fine-grained sharding expensive at
  scale.

**Source.** ZeRO: <https://ar5iv.labs.arxiv.org/html/1910.02054>. AMSP:
<https://arxiv.org/abs/2311.00257> (PDF).

### E5 — The capacity-factor formula counts top_k / n_experts twice

**Proposal.** §4.1:
`capacity_factor = max(1.0, roofline_slack(expert_ffn) × top_k / n_experts)`,
and "Higher capacity factor better — fill the idle ALUs".

**Correction.** In a mixture-of-experts (MoE) layer, expert capacity is
tokens × top_k / n_experts × capacity_factor. The ratio top_k / n_experts is
already applied outside the factor, so putting it inside counts it twice. For
top-2 routing over 8 experts the formula returns 1.0 unless the slack exceeds
4. The capacity factor exists to absorb routing imbalance, not to use spare
compute. With balanced routing the load on one expert is roughly Binomial
with mean μ = T·k/E, which gives a factor of about 1 + z/√μ for a margin of z
standard deviations. A factor above 1 only pads buffers; it does not "fill
the idle ALUs".

### E6 — A router column's norm does not measure how often the expert is selected

**Proposal.** §4.2: `router_affinity(e) = ‖router_weights[:, e]‖₂`, with the
comment "Low affinity → this expert is rarely selected".

**Correction.** Selection depends on how an expert's logit compares with the
others, not on its magnitude. A zero column gives a logit of 0, which wins
whenever the other logits are negative. In a numeric counterexample, a column
of norm 1e-4 is the top-1 choice for 100% of tokens when the other columns
point against the mean hidden state. Pruning that expert changes the output
for every token.

**Source.** Numeric check.

### E7 — The sensitivity score has no derivation

**Proposal.** §3.1: the score multiplies by `gradient_magnitude_est(θ)`
("estimated gradient scale from weight analysis") and divides by
`element_count(θ)` ("larger parameters tolerate more noise (averaging
effect)"). The tier table has a row "Very low (provably stable via WCET)".

**Correction.** The proposal gives no derivation for the formula, and three
of its parts do not hold up:

- Gradient scale cannot be read from the weights.
- Each element has its own `v`, so a tensor's size does not average its
  quantization error.
- WCET is worst-case execution time. It is unrelated to numerical stability.

### E8 — AMSP reports 52% MFU and a 1.56× throughput gain, not "56% MFU improvement"

**Proposal.** §1: per-component sharding "outperforms uniform ZeRO stages by
up to 56% MFU improvement"; the system is called "AMSP/Lins". Reference 4:
"Tianwei, Z., et al."

**Correction.** The AMSP abstract reports "up to 52% Model FLOPs Utilization"
(MFU) on 1024 GPUs, and a 1.56 times improvement in training throughput over
MiCS and ZeRO++. The first author is Qiaoling Chen; Tianwei Zhang is the
tenth of eleven authors. "Lins" could not be verified as a system name: it
appears only as a figure file name in the PDF's metadata. The statement in §8
that "AMSP requires a communication profiler" is correct; the profiler runs
offline, once per cluster.

**Source.** <https://arxiv.org/abs/2311.00257> (abstract and PDF).

### E9 — AsyncHZP does not show what it is cited for

**Proposal.** §1: "Recent work (AMSP/Lins, AsyncHZP) shows that"
per-component sharding outperforms uniform ZeRO stages.

**Correction.** AsyncHZP (arXiv 2510.20111, ByteDance Seed, October 2025)
compares against tuned N-D parallelism, not against uniform ZeRO stages, and
the speedups in its tables are 10–20%. It says group sizes "can be
arbitrarily specified" and gives no method for choosing them. It does not
mention AMSP.

**Source.** <https://arxiv.org/html/2510.20111v1>.

### E10 — Reference 6 has the wrong title, and the paper anticipates novelty claim 2

**Proposal.** References: 'DeepSpeed Team. "DeepCompile: Compiler-Level
Optimizations for Distributed Training." 2025.' §8, item 2: "NSL generates
communication calls as part of the compiled backward IR."

**Correction.** The paper is "DeepCompile: A Compiler-Driven Approach to
Optimizing Distributed Deep Learning Training" (Tanaka et al., arXiv
2504.09983, April 2025). It does compiler-inserted prefetching, selective
unsharding and offloading for ZeRO-3, and reports up to 1.28× over ZeRO-3 and
1.54× over FSDP (PyTorch's Fully Sharded Data Parallel). That anticipates §8
item 2. Items 1 and 4 need rewording against it.

**Source.** A search-result summary of the arXiv listing
(<https://arxiv.org/abs/2504.09983>). The full text was not read.

### E11 — bitsandbytes has no stochastic rounding, and Q-Adam-mini keeps `v` in FP32

**Proposal.** §3.3: "Q-Adam-mini discovered that embedding layers need
stochastic rounding for INT8 stability. In bitsandbytes, this is a runtime
flag."

**Correction.** bitsandbytes has no stochastic rounding. Its remedy for
embeddings is the `StableEmbedding` layer together with 32-bit optimizer
state. Q-Adam-mini (ICML 2025) quantizes only `m` to INT8 and keeps `v` in
FP32, in Adam-mini's blockwise form, where `v` is under 1% of memory. It is
evidence against quantizing `v`, not support for the "INT8 m, INT8 v" tier.
The proposal's other statements about bitsandbytes are correct: the
4096-element threshold (`min_8bit_size`) and the manual per-parameter
overrides.

**Source.** <https://huggingface.co/docs/bitsandbytes/en/optimizers>,
<https://arxiv.org/abs/2110.02861>, <https://icml.cc/virtual/2025/51750>.

### E12 — The bandwidth constants are a bidirectional total and a per-node total

**Proposal.** §6.2: "NVLink 900GB/s intra, IB 100GB/s inter". §2.2:
"intra-node bandwidth is 10-20× inter-node".

**Correction.**

- 900 GB/s is NVIDIA's bidirectional total for NVLink on an H100; one
  direction is 450 GB/s. Measured all-reduce bus bandwidth with NCCL (NVIDIA's
  collective communications library) on 8×H100 is roughly 350–450 GB/s.
- 100 GB/s matches AMSP's testbed per node, not per GPU: 4× HDR InfiniBand is
  800 Gb/s in total, shared by 8 GPUs, which is 12.5 GB/s per GPU. A DGX H100
  with 400 Gb/s NDR per GPU gives 50 GB/s per GPU.
- The proposal's own two constants give a ratio of 9×, not 10–20×.

**Source.** The NVLink, NCCL and DGX figures come from secondary sources and
are approximate. The testbed is from the AMSP paper
(<https://arxiv.org/abs/2311.00257>). The ratio is recomputed from the
proposal's own figures.

### E13 — Smaller slips

- §2: "~200ms for a full search across all feasible configurations". The
  search is over "≤64 combinations" of closed-form evaluations; the
  implemented search reports 0.10 ms.
- §1 says the space has "thousands of possible configurations"; §2 says
  "≤64 combinations".
- §3.2: "1 cycle (reinterpret)" for FP16 → FP32. That step is a numeric
  conversion, not a reinterpretation of bits.
- §6.2: "MFU: 67% (vs 48% naive DDP)" for a 50M-parameter model on 8×H100
  could not be verified, and is implausible.

---

## Part 2 — Superseded by later work

- **Adam-mini** (<https://arxiv.org/abs/2406.16793>) and **Q-Adam-mini**
  (<https://icml.cc/virtual/2025/51750>). One second-moment scalar per block
  removes over 99.9% of `v`, which is about half of Adam's state, with `v`
  still in FP32. INT8 `m` on top makes the saving about 8×. This beats every
  tier in the proposal and avoids quantizing `v` at all.
- **Muon.** NSL already ships this optimizer. It keeps one state tensor, so
  half of the optimizer-precision problem disappears for matrices routed to
  it. The runtime already stores Muon momentum in BF16 with stochastic
  rounding (`nsl_muon_state_sr_store`).
- **DeepCompile** (<https://arxiv.org/abs/2504.09983>) and **AsyncHZP**
  (<https://arxiv.org/html/2510.20111v1>). These are the baselines to beat.
  Both put per-operation latency and background-thread overlap at the centre
  of the design; the proposal has neither.
- **Hardware.** GB200 NVL72 racks put 72 GPUs in one NVLink domain at
  1.8 TB/s per GPU (NVLink 5), and Quantum-X800 InfiniBand runs at 800 Gb/s
  (from search-result summaries of NVIDIA pages, not the pages themselves).
  The proposal's reference cluster, and the constants the implementation
  hard-codes from it (8 GPUs per node, an 80 GiB budget, 900 and 100 GB/s),
  describe a 2023 DGX H100.
- **MoE practice.** DeepSeek-V3 (<https://arxiv.org/abs/2412.19437>) uses
  sigmoid gating, a per-expert bias adjusted online in place of an auxiliary
  loss, and one shared plus 256 routed experts with 8 active, and reports no
  token dropping in training. Mixtral-style and MegaBlocks training is also
  dropless (not re-checked against the source). A capacity factor with token
  dropping, which §4.1 plans for, is the 2021 Switch/GShard design.
- **FSDP2** (`fully_shard`, per-parameter sharding) has replaced the FSDP the
  proposal cites (not re-checked against the source).

---

## Part 3 — What the code implements

How it is switched on:

- CPDT: `--cpdt [full|zero_only|off]` with `--cpdt-num-gpus N` (required) and
  optionally `--cpdt-intra-bw`, `--cpdt-inter-bw`, `--cpdt-memory-budget`,
  `--cpdt-prune-experts`, `--cpdt-report` and `-w/--weights`; or
  `@cpdt(...)` on a train block. The cargo feature `experimental-cpdt` is on
  by default.
- ZeRO itself is separate: `--zero-stage 1|2|3`, `--zero-elementwise`,
  `--devices N`, `--collectives sim|sim-gpu|nccl`.
- Reduced-precision moments are also reached through
  `--wggo-moment-precision` and `--wggo-memory-budget`.

Component by component:

- **ZeRO configuration search (§2) — report-only.**
  `crates/nsl-codegen/src/cpdt_zero.rs` is a cost model whose result goes only
  to `CpdtPlan::render_report`, which labels it as a recommendation that is
  not applied. The stage that code generation uses comes from `--zero-stage`.
  The cost model no longer repeats E4: plain data-parallel training is
  charged its all-reduce, sharded parameters two all-gathers and a
  reduce-scatter per micro-batch, every collective a fixed latency, and no
  communication is treated as hidden under compute. The latency (50 μs per
  collective) is an assumed figure, not a measurement.
- **Per-layer overlapped communication schedule (§2.1) — report-only.**
  `crates/nsl-codegen/src/cpdt_comm.rs` builds a list, and its header says
  "ADVISORY / COST-MODEL ARTIFACT — NOT lowered". The `nsl_cpdt_*` functions
  in `crates/nsl-runtime/src/cpdt/ffi.rs` are identity copies that code
  generation never emits. The code is honest about this; the proposal's
  central claim is unimplemented.
- **ZeRO runtime — implemented, separately from CPDT.**
  `crates/nsl-runtime/src/zero.rs` does whole-tensor ownership, an all-reduce
  (stage 1) or reduce-scatter (stage 2), a per-owner broadcast, and stage 3
  with optional elementwise sharding. It is synchronous, with no overlap. It
  is a sound mechanism of its own, the same inter-tensor scheme AMSP
  describes, and CPDT does not plan it.
- **Per-layer mesh assignment (§2.2) — absent.** There is one global
  (s_p, s_g, s_os).
- **Sensitivity score (§3.1) — partial.**
  `crates/nsl-codegen/src/cpdt_sensitivity.rs` computes rms(weights) ×
  position / element count. The spectral factor is absent; embedding, norm,
  first and last layers are forced to High, where "last" is the model's own
  last block. Neither side is validated. In effect the tier is a function of
  tensor size and position.
- **Precision tiers to storage (§3.1) — partial, and deliberately different.**
  `first_moment_storage` and `second_moment_storage` in
  `crates/nsl-codegen/src/cpdt_precision_exec.rs` store `m` as BF16 for any
  reduced tier and `v` as FP32 always. INT8 is never stored. A reduced tier
  saves 2 of the 8 moment bytes per element. The code is right, for the
  reason in E1.
- **Fused quantized optimizer kernel (§3.2) — implemented differently.**
  `crates/nsl-codegen/src/cpdt_optim.rs` emits a descriptive operation list
  that nothing consumes. The working path, in
  `crates/nsl-codegen/src/stmt_fase.rs`, casts the moments to FP32, runs the
  existing step and casts back. It needs the Deferred mode of FASE (NSL's
  fused gradient-accumulation and optimizer step), which means AdamW with
  `grad_accumulation ≥ 2`, and it is refused under Muon, ZeRO-3 and
  `bf16-sr`. The implementation is simpler than the proposal and
  structurally fine.
- **INT8 blockwise quantization (§3.2) — not wired.**
  `crates/nsl-runtime/src/tensor/int8_blockwise.rs` is a linear absmax code
  with block size 64, CPU only, whose stochastic rounding is seeded by the
  caller. The proposal's "blockwise dynamic quantization", a non-linear
  code, is the right one; a linear INT8 code is unsound for `v`, and the
  module says so.
- **Stochastic rounding for embeddings (§3.3) — inert.** The flag is set
  only on embeddings, which are forced to FP32, and acts only on INT8, which
  is never emitted.
- **MoE capacity factor (§4.1) — partial.**
  `apply_capacity_overrides` in `crates/nsl-codegen/src/cpdt_moe_capacity.rs`
  may raise `@moe(capacity_factor)` to the roofline value under
  `--cpdt full` and never lowers it, but the roofline slack it uses is a
  constant 0.0, so that value is 1.0 for every layer. Neither side is right
  (E5).
- **Dead-expert pruning (§4.2) — implemented, opt-in.**
  `crates/nsl-codegen/src/cpdt_expert_prune.rs` slices the router and
  expert weights where the column norm is below 0.01, but only under
  `--cpdt-prune-experts`; without the flag it reports the experts the
  criterion picks out and leaves them. The mechanism is real; the criterion
  is wrong (E6), which is why it is no longer applied unasked.
- **Expert placement (§4.3) and joint solver (§5) — present, never run on
  MoE.** `crates/nsl-codegen/src/cpdt_expert.rs` and
  `crates/nsl-codegen/src/cpdt_joint.rs` exist, but `invoke_cpdt_if_enabled`
  in `crates/nsl-codegen/src/stmt_pass_bridges.rs` passes them no MoE shape
  and no router. The joint loop recomputes the same deterministic ZeRO
  search and stops at iteration 2.
- **MoE training (assumed throughout §4) — absent.** Dispatch is CPU,
  forward-only, and has no autodiff rule. `@moe(aux_loss_coeff=…)` with a
  non-zero value is refused at compile time, and `nsl_moe_aux_loss` is never
  emitted. MoE is inference-only.
- **`@cpdt(target_memory, precision)` (§6.1) — refused** with a diagnostic
  ("has no consumer"), which is the correct behaviour.

No shipped model uses CPDT. Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Keep `v` in FP32 in every tier, as the code now does. A 16-bit `v` needs
   BF16 with stochastic rounding (simulated step error within about ±5%).
2. Replace the tier table with Adam-mini's blockwise `v` plus 8-bit or BF16
   `m` (about 8×), and the sensitivity score with a measured or size rule.
3. Rewrite the ZeRO cost model: charge DDP its all-reduce, add per-collective
   latency, and give no overlap credit while the runtime is synchronous.
4. Have the planner set the ZeRO stage; its report is now labelled "not
   applied".
5. Make MoE trainable and dropless before planning capacity or placement.
6. Restate the novelty claims against DeepCompile and AsyncHZP.

---

## Part 5 — Implementation defects found in the same review

The review also ran the CPDT code. This part lists what it found wrong in the
implementation, as distinct from the proposal's own errors in Part 1. The
fixed items are described in `CHANGELOG.md` under "Unreleased". The open list
is as of 2026-11-13.

### Fixed

- Reduced-precision optimizer moments (`--cpdt full --weights`,
  `--wggo-moment-precision`, `--wggo-memory-budget`) stored Adam's second
  moment `v` as FP16, where the per-step increment for a gradient of 1e-3 is
  below half the smallest subnormal: `v` stayed zero and the step became
  `m_hat / eps`. A constant gradient of 1e-3 moved a weight from 1.0 to −0.85
  in 200 AdamW steps, where FP32 moments reach +0.82. Now `v` is always
  stored FP32 and a reduced first moment is stored BF16
  (`first_moment_storage` and `second_moment_storage` in
  `crates/nsl-codegen/src/cpdt_precision_exec.rs`), and the same run ends at
  0.8158. The test is
  `reduced_moments_track_full_precision_at_a_realistic_gradient_size` in
  `crates/nsl-cli/tests/cpdt_moment_precision_gate.rs`.
- The plan's byte total counted each tier's nominal bytes (1 for INT8) while
  2 were stored. It now counts stored bytes (`stored_moment_bytes`): a
  reduced tier saves 2 of the 8 moment bytes per element.
- MoE expert capacity was tokens / experts × factor whatever `top_k` was, so
  balanced top-2 routing at factor 1.0 dropped half the assignments and
  later tokens came out as zeros. Capacity now includes `top_k` (`route_topk`
  in `crates/nsl-runtime/src/moe/router.rs`).
- `--cpdt full` replaced a user's `@moe(capacity_factor)` with the roofline
  value, which is 1.0 for every layer because the slack passed in is a
  constant 0.0. The pass may now only raise a layer's factor
  (`apply_capacity_overrides` in
  `crates/nsl-codegen/src/cpdt_moe_capacity.rs`).
- Under `--cpdt full` with a weights file, an MoE expert whose router column
  has norm below 0.01 was sliced out of the router and expert weights. Such
  an expert can be the one selected most often (E6), so pruning it changes
  the model's output. The pass now only reports those experts; slicing them
  needs `--cpdt-prune-experts` (`run_moe_prune_pass` in
  `crates/nsl-codegen/src/cpdt_expert_prune.rs`).
- When an MoE expert was over capacity, assignments were dropped in token
  order, so later positions lost first, and a token that kept only one of
  its experts kept that expert's weight without renormalisation. An expert
  now keeps the assignments with the highest routing probability, and each
  token's surviving weights are renormalised to sum to one (`route_topk` in
  `crates/nsl-runtime/src/moe/router.rs`).
- The precision planner assumed every model has 8 layers: on a 12-block
  model block 7 was taken for the last layer and kept FP32, and block 11,
  the real last layer, was reduced. The count now comes from the train
  block's own parameters (`train_block_layer_count` in
  `crates/nsl-codegen/src/stmt_pass_bridges.rs`), and a plan that reduces no
  moment no longer reports precision as active.
- `@moe(aux_loss_coeff=…)` was accepted and had no effect. A non-zero
  coefficient is now a compile-time error that says no auxiliary loss is
  applied (`validate_moe_decorator` in `crates/nsl-semantic/src/moe.rs`).
  MoE dispatch still has no backward rule (Part 3).
- With `--cpdt full --weights`, a parameter whose name in the weights file
  carries a leading prefix, as in `model.blocks.0.w`, was not matched to the
  train block's parameter and silently kept FP32 moments. Such a name now
  joins when it is the only one ending in the parameter's path, and the
  weights-file tensors that join nothing are listed in a warning
  (`precision_for_path` and `unjoined_plan_names` in
  `crates/nsl-codegen/src/cpdt_precision_exec.rs`).
- An inference-only MoE program compiled with `--cpdt full` failed with
  "--cpdt: owner CPDT recorded no disposition", although the MoE passes had
  run. They now record what they did (`record_moe_disposition` in
  `crates/nsl-codegen/src/cpdt_moe_capacity.rs`).
- In a debug build, a weights file naming a layer numbered at or above the
  assumed count of 8 would panic on an assertion in `position_criticality`
  (`crates/nsl-codegen/src/cpdt_sensitivity.rs`). The assertion is removed;
  such a layer is scored as an interior one.
- `--cpdt-report` for a 33,344-parameter model printed "params=12.98 MB";
  the parameters are 0.13 MB. The size came from WGGO's cost table for a
  built-in layer. WGGO's plan now carries the model's own bytes (see the
  WGGO errata), and the report prints 0.13 MB.
- The ZeRO cost model charged plain data-parallel training no communication,
  omitted the second all-gather for sharded parameters, the micro-batch
  factor and per-collective latency, and counted communication only where it
  exceeded total compute time, so every configuration tied and the tie went
  to the most sharding. It now charges each configuration its collectives
  (2Ψ, 2Ψ and 3Ψ per micro-batch for data-parallel, gradient-sharded and
  parameter-sharded, as in the ZeRO paper), a fixed latency per collective,
  and no overlap (`comm_per_step` and `evaluate_config` in
  `crates/nsl-codegen/src/cpdt_zero.rs`). On a cluster where everything
  fits, the recommendation is now plain data-parallel.
- The planner's memory budget was a constant 80 GiB per GPU. It is now set
  with `--cpdt-memory-budget <GiB>` (default 80), and the report prints it.
- The ZeRO stage printed by `--cpdt-report` is not applied, and the report
  did not say so. The section is now headed "recommendation only, not
  applied" and states which stage the compiled program runs, from
  `--zero-stage` (`CpdtPlan::render_report` in
  `crates/nsl-codegen/src/cpdt.rs`).
- The comment on `nsl_zero_reduce_grads` in `crates/nsl-runtime/src/zero.rs`
  said the path "is currently DEAD in production". It now describes where
  code generation emits the call.
- The test `stochastic_rounding_is_unbiased_in_mean` in
  `crates/nsl-runtime/src/tensor/int8_blockwise.rs` could not fail:
  `nsl_tensor_quant_int8_blockwise` reseeded its generator from the element
  count on every call, so its 64 trials were identical, and its tolerance
  exceeded the error of any rounding rule. The `stochastic` argument is now
  the seed, and the test (`stochastic_rounding_is_unbiased_over_seeds`)
  requires the mean of a value of 10.25 quantization steps over 4000 seeds
  to be within 0.05 step of 10.25, where round-to-nearest gives 10.
- The expert placement planner (`crates/nsl-codegen/src/cpdt_expert.rs`),
  which no production path runs, charged a balanced partition no dispatch
  and halved `top_k` in integer arithmetic for the hybrid, so balanced
  always won at zero cost. A partition is now charged the dispatch of the
  `(G−1)/G` of assignments whose expert is on another GPU, and the hybrid
  half of the full dispatch.

### Open

- An inference-only program that dispatches through `moe_dispatch`, built
  with `--cpdt full --weights`, now compiles and then stops at run time with
  "invalid tensor handle 0x2". The compile used to be refused as inert, which
  hid it; the gate for that case (`cpdt_model_facts_gate`) checks the compile
  only. Reproduced by running. Right: find the handle that is passed before
  it is created, or refuse the configuration at compile time again.
- The INT8 blockwise code, which is not wired into the optimizer, is a linear
  absmax quantizer, and that is unsound for `v`: with a five-decade spread
  inside a block the small elements round to zero and the median step is 33
  times too large, with or without stochastic rounding. Where:
  `crates/nsl-runtime/src/tensor/int8_blockwise.rs`. Established: numeric
  check (a simulation of the same algorithm, not a run of this code). The
  module now states this and nothing stores `v` in the format; replacing it
  with a non-linear code (the dynamic map bitsandbytes uses) needs that
  code's table reproduced and checked against the reference implementation,
  which was not available.
