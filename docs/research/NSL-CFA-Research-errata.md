# CFA (Compile-Fused Alignment) — Errata

**Companion to** `NSL-CFA-Research.md.pdf`. **Date:** 2026-11-13.

This file lists the statements in the CFA proposal that should not be relied
on, and what is true instead. The proposal was written in April 2026 and
checked in November 2026 against its own arithmetic, the literature, and the
repository. The arithmetic below was recomputed; external facts were checked
against the linked source unless the entry says otherwise; and statements
about the code describe the repository on the date above. The PDF itself is
unchanged.

Abbreviations: DPO is Direct Preference Optimization, GRPO is Group Relative
Policy Optimization, SimPO is Simple Preference Optimization, and KTO and
DAPO are the methods of the proposal's references 5 and 3. CoW is
copy-on-write, LoRA is low-rank adaptation, SFT is supervised fine-tuning, AD
is automatic differentiation, CE is cross-entropy, GEMM is a general matrix
multiply, PPO is Proximal Policy Optimization, and TRL is Hugging Face's
Transformer Reinforcement Learning library.

---

## Part 1 — Errors in the proposal

### E1 — A copy-on-write reference saves nothing under full fine-tuning

**Proposal.** §2.1: "typically only 10-30% of layers have diverged", so "The
CoW scheme saves 70-90% of the reference model's memory cost in early
training". §2.3 lists "W_lm_head (shared between policy and ref if CoW not
diverged)".

**Correction.** Copy-on-write shares buffers that are identical, not buffers
that are close. One AdamW step with weight decay changes every element of
every trainable tensor, so every layer diverges at step 1 and the reference
costs a full copy from then on, the LM head included. The remark that memory
is tightest in early training "because the model hasn't settled yet" has no
content: memory use does not depend on the training phase. The only case with
a real saving is LoRA or frozen layers, where plain sharing of the untouched
weights is enough. For DPO, the 28 GB of §1.1 has a one-model alternative the
proposal never mentions: TRL's `precompute_ref_log_probs`, with which "the
reference model does not need to be kept in memory".

**Source.** Logic; https://huggingface.co/docs/trl/main/en/dpo_trainer for
the precompute option.

### E2 — The KTO loss in §2.6 is not the published loss

**Proposal.** §2.6, `kto_loss`: "let kl = policy_logprob - ref_logprob", then
"return -log_sigmoid(β * kl)" for preferred examples and
"return -log_sigmoid(-β * kl)" otherwise.

**Correction.** The paper defines r = log(π_θ / π_ref) and a reference point
z₀ = KL(π_θ ‖ π_ref), which is estimated from mismatched pairs, clamped at 0,
and carries no gradient. Its value function is v = λ_D · σ(β(r − z₀)) for
desirable examples and v = λ_U · σ(β(z₀ − r)) for undesirable ones, and the
loss is L = E[λ_y − v]. The proposal's version differs in three ways: it uses
log σ where the paper uses σ, it drops z₀, and it drops the λ weights. It
also names the log-ratio `kl`, which is the quantity the paper calls r.

**Source.** https://arxiv.org/html/2402.01306

### E3 — The DPO gradient in §2.3 has the wrong sign

**Proposal.** §2.3, kernel pseudo-code: "d_loss = β × sigmoid(-reward_diff)",
then "grad_policy_hidden_chosen = d_loss × d_logprob_d_hidden(...)" and
"grad_policy_hidden_rejected = -d_loss × d_logprob_d_hidden(...)".

**Correction.** With u = β((log π(y_w) − log π_ref(y_w)) − (log π(y_l) −
log π_ref(y_l))) and L = −log σ(u):

    ∂L/∂log π(y_w) = −β · σ(−u)        ∂L/∂log π(y_l) = +β · σ(−u)

The proposal has the two signs swapped, which is gradient ascent on the loss.
The magnitude β · σ(−u) is right. On random logits with β = 0.1, automatic
differentiation of the loss gives −0.0689 for the chosen term where the
pseudo-code gives +0.0689. The loss formula itself matches Rafailov et al.,
in the form the TRL documentation gives it.

**Source.** Numeric check against PyTorch autograd;
https://huggingface.co/docs/trl/main/en/dpo_trainer for the loss formula.

### E4 — The rollout speedup is against sequential generation, and its figures disagree

**Proposal.** The abstract claims decode kernels "8× faster than vLLM". The
§2.4 table gives "~128 seconds" for K = 8 with PyTorch `generate()` against
"~16 seconds", and puts launches per token at "~3,780 (32-layer model)". The
text of §2.4 says `generate()` "launches ~500 kernels per token per layer"
and offers "1 kernel launch per layer, not 30". §2.4 and §5 give the speedup
as "8-25×", and §2.4 states that this makes total GRPO training 3-5× faster.

**Correction.**

- The 128 seconds is 8 × 16 s, that is, the eight rollouts generated one
  after another. Every framework batches the K rollouts, so the baseline is
  not one anyone runs.
- The abstract states its 8× against vLLM, but the table's 8× (16 / 2 per
  rollout, 128 / 16 in total) is against the `generate()` column. No figure
  for vLLM is given anywhere. The 25× is 128 / 5, the speculative row.
- The three launch counts do not agree: 500 per token per layer is 16,000
  per token for 32 layers; 3,780 per token is 118 per layer; and 30 per layer
  is 960 per token.
- Sampling at 2 µs instead of 30 µs saves 28 µs × 512 = 14 ms of a 16 s
  rollout.
- Granting the rollout speedup on an 80% share, Amdahl's law gives
  1 / (0.2 + 0.8 / 8) = 3.3× end to end at 8× and 4.3× at 25×. The ceiling
  for an infinitely fast rollout is 5×, so the top of the stated 3-5× range
  cannot be reached.

No measurement in the proposal supports the 8× figure.

**Source.** Recomputed from the proposal's own figures.

### E5 — The build report's memory figures are inconsistent

**Proposal.** §3 build report for "NSLCoder-50M (48.8M params)":
"Policy optimizer (CPDT mixed): 37 MB", "Fused log-prob kernel: logits never
materialized (saves 1.5 GB)", "Standard PyTorch TRL DPO: ~3.8 GB" and
"Memory reduction: 3.2-3.5× smaller", built with "--target 8xH100".

**Correction.**

- 37 MB for 48.8M parameters is 0.76 bytes per parameter for two AdamW
  moments. In FP32 they take 48.8M × 8 = 390 MB.
- There is no FP32 master-weights line (48.8M × 4 = 195 MB). The CPKD
  proposal's build report has one.
- With both, the starting total is 97.6 + 390.4 + 195.2 + 128 + 800 =
  1,611 MB, about 1.6 GB, and the ratio to 3.8 GB is about 2.4×, not
  3.2-3.5×.
- 800 MB of the reported 1.1 GB is CUDA context, which PyTorch pays too.
- The 1.5 GB of logits is the batch-1 figure: four tensors of
  2048 × 49152 × 4 bytes = 384 MiB each. The example sets `batch_size = 4`,
  which gives 6 GiB, more than the 3.8 GB quoted for the whole of TRL.
- The target is eight H100s for a 50M-parameter model.

**Source.** Recomputed from the proposal's own figures.

### E6 — In-process reward execution is unsafe and contradicts the code it cites

**Proposal.** §2.5 cites "NSL's coder-rl design" as the existing reward
ladder, then moves it into the training process with "no subprocess spawning,
no IPC overhead", calling `nsl_execute()` among others "all as function calls
within the compiled binary, not as subprocess invocations".

**Correction.** The NSL runtime aborts the process on many errors by design,
the gather bounds check being one example. Running model-generated code in
the trainer's own process would let that code kill or hang the trainer. The
coder-rl material the section cites specifies the opposite:
`models/coder-rl/reward.nsl` and
`docs/plans/2026-03-28-nsl-coder-rl-design.md` both describe subprocess
calls.

### E7 — DAPO is misdescribed

**Proposal.** §1.1 lists DAPO with "2 (policy + frozen reference)" models at
28 GB. The §3 GRPO example sets "clip = [0.8, 1.2]", commented "DAPO-style
clip range", together with "beta = 0.04", commented "KL penalty", and
"token_level = true".

**Correction.** DAPO removes the KL term and has no reference model, so it is
a one-model method. Its clip parameters are ε_low = 0.2 and ε_high = 0.28,
giving [0.8, 1.28]; [0.8, 1.2] is the range of plain PPO. A KL coefficient
of 0.04 is the GRPO setting of DeepSeekMath, and DAPO has no KL term to
apply it to. The `dapo_loss` function in §2.6 is itself a correct token-level
clipped surrogate.

**Source.** https://arxiv.org/html/2503.14476 for DAPO;
https://arxiv.org/html/2402.03300 for DeepSeekMath's β.

### E8 — Existing frameworks already generate rollouts with a production inference engine

**Proposal.** §1.2: "current alignment frameworks use unoptimized generation
paths", glossed as HuggingFace `model.generate()` and not production
inference engines.

**Correction.** OpenRLHF describes itself as "built upon Ray, vLLM, DeepSpeed
and HuggingFace Transformers", and its README pairs the 80% generation figure
with vLLM as the remedy. TRL's GRPO trainer has `use_vllm`, in colocate and
server modes. The 80% figure itself is real.

**Source.** https://arxiv.org/abs/2405.11143,
https://github.com/OpenRLHF/OpenRLHF,
https://huggingface.co/docs/trl/main/en/grpo_trainer

### E9 — Fused alignment losses that never materialise logits already exist

**Proposal.** §5 gives the PyTorch (TRL/OpenRLHF) log-probability path as
"Full [B,S,V] logits → log_softmax" and then a gather, and §7 concludes "No
existing alignment framework can exploit all four of these observations
simultaneously".

**Correction.** The fused-loss observation is already exploited. Liger-Kernel
ships fused-linear DPO, ORPO, CPO, SimPO, KTO and GRPO losses; its v0.5.0
release of 2024-12-11 announced "80% more memory efficient post training
losses (DPO, ORPO, CPO, etc)", and TRL exposes them through
`use_liger_kernel`. With E8, two of the four observations were available in
existing frameworks before the proposal was written.

**Source.** https://github.com/linkedin/Liger-Kernel,
https://linkedin.github.io/Liger-Kernel

### E10 — The reference model saves no activations in any framework

**Proposal.** §1.3: "4 full forward passes per DPO step!". The §5 table has a
row stating that in PyTorch the reference backward keeps saved activations as
wasted memory, where CFA's is absent from the binary.

**Correction.** Every framework runs the reference model without gradients,
so no reference activations are saved and there is nothing for CFA to
remove. The four passes are four sequences' worth of computation however they
are batched, and CFA's fused kernel does not reduce that.

### E11 — Smaller slips

- §1.1, "A single A100-80GB barely fits this". The figure it refers to is
  28 + 20 = 48 GB, which fits with 32 GB to spare.
- §1.2, "GRPO generates K=8-16". DeepSeekMath used a group size of 64
  (https://arxiv.org/html/2402.03300); DAPO used 16
  (https://arxiv.org/html/2503.14476).
- §7, "CFA completes the eleventh and final paper", above a table of ten
  rows. The CPKD proposal's conclusion says "nine papers" over a different
  list.

**Checked and found correct:** the DPO loss; the form of the SimPO loss (the
paper's title and venue were confirmed at https://arxiv.org/abs/2405.14734,
the equation was not re-checked against it); the online log-sum-exp
pseudo-code of §2.2; 192 KiB of logits per token at vocabulary 49152;
384 MiB per sample and 1.5 GiB per tensor at batch 4; 32 KiB of log-probs;
and that GSPO exists (https://arxiv.org/abs/2507.18071). The five-stage
reward ladder of §2.5 matches the header of `models/coder-rl/reward.nsl`.

---

## Part 2 — Superseded by later work

All of the following predates April 2026.

- **TRL's GRPO trainer**
  (https://huggingface.co/docs/trl/main/en/grpo_trainer). It defaults to
  `beta=0.0`, in which case "the reference model is not loaded"; it defaults
  to `loss_type='dapo'`; and it offers sequence-level importance ratios
  (GSPO, https://arxiv.org/abs/2507.18071). The proposal's row saying GRPO
  needs two models no longer holds, and copy-on-write has no user there.
- **"Understanding R1-Zero-Like Training" (Dr. GRPO)**
  (https://arxiv.org/abs/2503.20783). It shows a length bias from the
  1/|o_i| scaling of the GRPO loss, and TRL drops that scaling for this
  reason. A GRPO design should not include it.
- **TRL's DPO trainer**
  (https://huggingface.co/docs/trl/main/en/dpo_trainer). It has 16 loss
  types, reference log-prob precompute, Liger fused losses, and TR-DPO
  reference syncing. The DPO baseline of §5 describes none of this.
- **Liger-Kernel** (https://github.com/linkedin/Liger-Kernel). Fused,
  chunked losses exist for DPO, SimPO, KTO and GRPO. A fused loss kernel has
  to be compared with these, not with materialised logits.

---

## Part 3 — What the code implements

Nothing in CFA is implemented. There is no `cfa_*.rs` file, no `align`
keyword or block, no `frozen_copy`, no `--align` flag, no DPO, SimPO, KTO or
GRPO loss in the standard library, no `log_sigmoid` builtin, and no flag,
decorator or test for any of it.

§2.5 of the proposal cites the coder-rl design. What `models/coder-rl/`
holds:

- `train_grpo.nsl` is an SFT script. GRPO appears only in comments, which
  defer it to a Rust "reward server" (`grpo_server`) and a
  `collect_rejections.nsl`; neither exists in the tree. The comments say
  per-token log-probs are available from an unreduced `cross_entropy`, but
  the standard library's `cross_entropy` has no unreduced mode.
- `reward.nsl` is constants and comments ("In a real implementation, these
  would call the NSL CLI as subprocesses"). Beyond the five stages the
  proposal quotes, it has a `REWARD_RUN_OK` of 0.85 and per-statement partial
  credit.
- `tasks.nsl` is comments only.
- `model.nsl` defines a model of 11.31M parameters by count (its banner says
  10.1M and its header 12M) with vocabulary 4096. The proposal's
  "NSLCoder-50M (48.8M params)" at vocabulary 49152 is a different model.
- The design document, `docs/plans/2026-03-28-nsl-coder-rl-design.md` §5,
  normalises advantages as GRPO does, but its loss
  `L = -sum(A_i * log(pi(code_i | prompt)))` is REINFORCE with a group
  baseline: no ratio, no clip, no per-token mean. It is gradient-equivalent
  to GRPO only for a single on-policy update. Its β is 0.05 against the
  paper's 0.04. Its batch of 32 prompts × 8 candidates at sequence 2048 and
  vocabulary 4096 implies 8.6 GB of FP32 logits before anything else, against
  its claim of fitting in 16 GB.

§2.6's promise that a user writes the loss as an NSL function and source AD
does the rest does not hold today. Under `--source-ad` a user-defined loss
function is not extracted, in a `grad` block or a `train` block, and the
compiler falls back to tape AD. Under tape AD a loss written as a function
now gets its gradient; it used to get all zeros (Part 5, "Fixed"). The same
mathematics written inline in a train-block step body trains correctly in
both modes.

`docs/wiki/Glossary.md` used to define CFA as "Compiler-Fused Attention" and
to send FlashAttention readers to this PDF, which is about alignment. It now
gives the proposal's own title and says none of it has been built.

Defects found in this code are listed in Part 5.

---

## Part 4 — Recommended direction

1. Drop copy-on-write. For DPO, precompute reference log-probs once per
   example; for LoRA, share the base weights.
2. Replace the three kernels with one: a per-row-weighted fused linear CE on
   the existing GEMM-chunked path. DPO, SimPO, KTO and clipped policy
   gradient then need only host-side scalar maths on per-sequence sums.
3. Take GRPO defaults from current practice: β = 0, no 1/|o_i|, clip
   0.2/0.28, optional sequence-level ratio, skip zero-variance groups.
4. Run rewards in a sandboxed subprocess with time and memory limits.
5. Correct the KTO loss, the DPO sign and the DAPO rows, and measure rollout
   speed against batched vLLM before stating a figure.

---

## Part 5 — Implementation defects found in the same review

The review also ran the code. CFA itself is not implemented, so there is no
CFA code to be wrong; this part lists what the review found wrong in the
nearest code, as distinct from the proposal's own errors in Part 1: the
automatic differentiation (AD) a hand-written alignment loss would go
through, and the files under `models/coder-rl/`. Fixed items are described
in `CHANGELOG.md` under "Unreleased"; the open list is as of 2026-11-13. Each
open item ends with how it was established: reproduced by running, numeric
check, or by reading the code only.

### Fixed

- A loss written as a user function returned a correct value and an all-zero
  gradient under tape AD whenever it nested two or more elementwise ops on a
  `let`-bound value (`neg(log(u))`, `log(sigmoid(u))`). The fused elementwise
  call records nothing on the autodiff tape, and functions are compiled
  without knowing a tape may be live. Model methods had the same hole. The
  fused call now runs as its separate, recorded ops whenever a tape is
  recording (`fused_chain_needs_tape` in `crates/nsl-runtime/src/cpu.rs`;
  test `crates/nsl-cli/tests/tape_fused_chain_gate.rs`), so the DPO and SimPO
  losses written as functions now get non-zero gradients under tape AD.

### Open, in the adjacent code

- **`train_grpo.nsl`'s schedule does not fit its run.** At batch 8 the
  172-sequence corpus gives 21 steps per epoch and 210 steps in 10 epochs,
  against `warmup_steps=100, total_steps=1000`: warm-up is 48% of the run and
  the rate ends near 2.9e-4 of 3e-4. Unlike `train_sft.nsl` it also trains
  with dropout on, weight decay 0.01, and data read from `data/`. By reading
  the code only; the corpus is not in the tree and 172 is `train_sft.nsl`'s.
- **Source AD does not extract a user-defined loss function.** Under
  `--source-ad`, in a `grad` block or a `train` block, the compiler warns of
  an "unrecognized FFI callee" and falls back to tape AD; a `distill` block
  refuses the tape instead. It should inline the function or refuse loudly.
  `extract_expr_inner` in `crates/nsl-codegen/src/source_ad.rs`. Reproduced
  by running.
- **`train_sft.nsl`'s comments disagree with its code.** They say
  "batch_size=4" with 43 steps per epoch and a 20-step warm-up; the code
  uses batch 2 and `warmup_steps=40`. By reading the code only.
- **The reward ladder in `reward.nsl` can be gamed.** A valid program that
  ignores the prompt earns 0.85 with no test, or 0.8 with a failing one,
  while an attempt that fails to compile earns at most 0.6; partial credit
  pays 0.02 per valid statement before a parse error. The compiler checks
  validity, not whether the task was done. Nothing runs this ladder yet. By
  reading the code only.
