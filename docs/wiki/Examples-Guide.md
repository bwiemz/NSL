<!-- owner: @bwiemz -->

# Examples Guide

Annotated reading order for [`examples/`](../../examples/). Each entry tells you what the file teaches, which lines to read first, and where to go next.

## Recommended reading order

### 1. `hello.nsl`

**What it teaches:** `let`/`const` bindings, typed variables, function definitions, control flow (`if`/`elif`/`else`), `for` loops, and `struct` declarations — the complete NSL syntax surface in one short file.

**Key lines:** 3–5 (`let`/`const` declarations), 7–9 (function definition with typed params and return type), 14–20 (if/elif/else), 22–25 (`for` loop over a list), 28–31 (struct definition).

**Next:** `m11_model_basic.nsl` — once you can read basic NSL, learn the `model` keyword.

[`examples/hello.nsl`](../../examples/hello.nsl)

---

### 2. `m11_model_basic.nsl`

**What it teaches:** Defining a `model` block with fields and methods, instantiating it, and calling methods — the minimal pattern for every NSL model.

**Key lines:** 3 (`model Counter(start: int):`), 4 (field declaration with initializer), 6–7 (`fn get` method returning a field), 10–11 (`fn increment` mutating `self.value`), 13–16 (instantiation and method calls).

**Next:** `m12_grad_basic.nsl` — add automatic differentiation to a model.

[`examples/m11_model_basic.nsl`](../../examples/m11_model_basic.nsl)

---

### 3. `m12_grad_basic.nsl`

**What it teaches:** Using the `grad` keyword to compute gradients — the destructured `(loss, grads)` return, what the trailing expression inside the `grad` block means, and how the tape sees element-wise operations and reductions.

**Key lines:** 1–4 (inline derivation — forward / backward math in closed form), 6–7 (tensor setup with `ones`), 9–11 (`grad(w): ... y.sum()` — the complete gradient expression), 13–14 (printing loss and grads to confirm shapes).

**Next:** `m14_adam_scheduler.nsl` — plug gradients into a full training loop.

[`examples/m12_grad_basic.nsl`](../../examples/m12_grad_basic.nsl)

---

### 4. `m14_adam_scheduler.nsl`

**What it teaches:** The `train` block DSL — declaring an optimizer, attaching a learning-rate scheduler, writing a `step` callback that computes loss, and the `on_step` hook for logging.

**Key lines:** 3–8 (model with a `forward` method), 13–20 (`train(model=m, epochs=10):` — optimizer, scheduler, step, callbacks).

**Next:** `gpt2.nsl` — the GPT-2 architecture, trained end to end on the CPU.

[`examples/m14_adam_scheduler.nsl`](../../examples/m14_adam_scheduler.nsl)

---

### 5. `gpt2.nsl`

**What it teaches:** The GPT-2 architecture written out in NSL and trained end to end on the CPU: learned token and position embeddings, pre-LayerNorm decoder blocks of causal multi-head self-attention (one fused `c_attn` projection, `scaled_dot_product_attention` with the causal mask) and a 4x GELU MLP, a final LayerNorm and an LM head tied to the token embedding, with GPT-2's initialisation. Nested `model` blocks (`CausalSelfAttention`, `GPT2MLP`, `GPT2Block`) and a `[GPT2Block; N]` model array; an in-memory `DataLoader`; a `train` block with AdamW, warmup-cosine, gradient clipping and an `on_step` callback. The sizes are a toy configuration (the header says which constants make it GPT-2 small); `nsl run examples/gpt2.nsl --target cpu` trains for 60 steps in seconds and the loss falls from about ln(64) to below 0.5.

**Key lines:** the `CausalSelfAttention` forward (splitting the fused projection with `tensor_slice`, heads by `reshape`/`transpose`), the `GPT2` model (tied head), and the `train` block.

**Next:** `codeforge_nano.nsl` — a real proof-of-concept trained model with a byte-level tokenizer.

[`examples/gpt2.nsl`](../../examples/gpt2.nsl)

---

### 6. `codeforge_nano.nsl`

**What it teaches:** A self-contained mini language model (single-head attention, SwiGLU FFN, RMSNorm) written as a realistic NSL proof of concept, including a byte-level tokenizer, `tensor_slice` for sequence preparation, and inline documentation of current NSL limitations (no GQA/RoPE/causal mask yet).

**Key lines:** 5–17 (architecture overview in comments — fast way to understand what's in the file), 22–29 (hyperparameter `const` declarations), 31–35 (byte tokenizer + encoding), 37–38 (input/target slice construction).

**Next:** Subsystem demos below for CSHA, WRGA, and CPDT.

[`examples/codeforge_nano.nsl`](../../examples/codeforge_nano.nsl)

---

## Subsystem demos

Reference examples for specific subsystems. Read the corresponding wiki page first.

### CSHA (fused attention) — `csha_toy_pretrain_hd32.nsl`

Read [Optimization-Passes § CSHA](Optimization-Passes.md#csha) first.

**Start here:** read [`csha_toy_pretrain_hd32.nsl`](../../examples/csha_toy_pretrain_hd32.nsl). It needs a CUDA GPU: `nsl run --target cuda --source-ad examples/csha_toy_pretrain_hd32.nsl`.

**What to look at:** The `model TinyAttn` block shows the minimal `@flash_attention(head_dim=32)` method pattern that the CSHA boundary scan matches: RMSNorm, three projections and `scaled_dot_product_attention` with nothing in between. Weight names `wq`/`wk`/`wv` are intentional — `ProjKind::from_param_name` depends on them. The program trains one SGD step and prints its loss and statistics of each weight's update; `crates/nsl-cli/tests/csha_toy_pretrain_gate.rs` compares them with PyTorch. Adding `--csha` is a compile error for this program: the fused backward it would train through is withheld until its gradients match a reference.

[`examples/csha_toy_pretrain_hd32.nsl`](../../examples/csha_toy_pretrain_hd32.nsl)

---

### WRGA (LoRA / IA³ / GatedLoRA fusion)

No dedicated `.nsl` example file exists for WRGA. The behavior is covered by the CLI integration tests:

- **[`wrga_adapter_runtime_equivalence.rs`](../../crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs)** — primary reference: verifies that fused LoRA/IA³/GatedLoRA adapter materialisation produces the same output as the unfused path. Start here.
- **[`wrga_gatedlora_backward_trigger.rs`](../../crates/nsl-cli/tests/wrga_gatedlora_backward_trigger.rs)** — exercises the backward-trigger measurement (tests the 2.5× threshold condition that schedules the fused backward kernel).
- **[`wrga_report_cli.rs`](../../crates/nsl-cli/tests/wrga_report_cli.rs)** — exercises `--wrga-report` CLI output.

---

### CPDT (weight-aware compilation)

No dedicated `.nsl` example file exists for CPDT. The behavior is covered by the CLI integration tests:

- **[`cpdt_weights_cli.rs`](../../crates/nsl-cli/tests/cpdt_weights_cli.rs)** — primary reference: exercises `load_safetensors(...)` auto-detect, the four-case decision table in `nsl build`, and the `validate(wm, applied)` layer-prefix validation.
- **[`cpdt_cli.rs`](../../crates/nsl-cli/tests/cpdt_cli.rs)** — covers the `@cpdt(weight_aware=false)` runtime opt-out and the one-`@cpdt`-per-program semantic enforcement.

---

## Not in this guide

`examples/` contains 60+ files including milestone-specific smoke tests (`m10_*.nsl`, `m13_*.nsl`, `m15_*.nsl`, `m17_*.nsl`, etc.) — these aren't in the reading order because they're compiler-regression artifacts, not learning material. If you're debugging a specific milestone's behavior (e.g., shape-checking errors, import resolution, activation functions), those are the files to reach for. `ls examples/m<N>_*.nsl` to find them.

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
