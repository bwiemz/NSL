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

**Next:** `gpt2.nsl` — a production-scale model that uses every feature above.

[`examples/m14_adam_scheduler.nsl`](../../examples/m14_adam_scheduler.nsl)

---

### 5. `gpt2.nsl`

**What it teaches:** A complete GPT-2 (124 M) pipeline including imports from `nsl.nn`, `nsl.optim`, `nsl.data`, `nsl.quant`, and `nsl.export`; a `CONFIG` dict; nested `model` blocks (GPT2Block, full GPT2); quantization via `gptq`; and ONNX export — everything from a blank file to a deployable checkpoint.

**Key lines:** 1–13 (import surface — good reference for which stdlib modules exist), 17–31 (CONFIG dict), 36–40 (opening of `model GPT2Block` showing `LayerNorm` + `Linear` composition).

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

**Start here:** read [`csha_toy_pretrain_hd32.nsl`](../../examples/csha_toy_pretrain_hd32.nsl) as reference material — it requires a GPU with sufficient shared memory to run.

**What to look at:** The file documents the two compiler gaps it closes (DOC-GAP A: scanner now descends into `ModelMember::Method`; DOC-GAP B: `head_dim=N` is now a decorator parameter). The `model TinyAttn` block (lines 25–38) shows the minimal `@flash_attention(head_dim=32)` method pattern. Weight names `wq`/`wk`/`wv` are intentional — `ProjKind::from_param_name` depends on them for boundary scanning.

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
