# M14: Training DSL Implementation Design

> **Date:** 2026-03-10
> **Status:** Approved
> **Depends on:** M9 (runtime), M10 (types), M11 (models), M12 (autodiff), M13 (imports)

**Goal:** Implement the `train` block end-to-end — from parsing (already done) through semantic validation and codegen — plus NSL standard library modules for optimizers, schedulers, and loss functions, and a simple .nslm checkpoint format.

**Architecture:** The compiler generates the training loop structure (epoch iteration, implicit grad wrapping, gradient accumulation, clipping, optimizer dispatch, scheduler updates, callbacks, evaluation). The NSL stdlib provides pure-function optimizers, schedulers, and loss functions. New Rust runtime primitives support in-place tensor mutation and element-wise math ops.

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implicit grad wrapping | Yes | Spec shows step bodies assign to loss without explicit grad — compiler wraps automatically |
| Checkpoint format | Simple binary .nslm | Custom flat format: magic + JSON header + raw f64 data. Safetensors compat deferred to M18 |
| Optimizer implementation | Pure NSL functions | Follows "if it composing primitives into ML abstractions, it is NSL" principle. Users can read/modify/extend |
| Optimizer state | In-place mutation via copy_data | Buffers allocated once at training start, mutated in-place. No allocation per step |
| Loss scaling | Scale loss before backward | Numerically safer for future fp16 support. loss = loss * (1/accumulate) before tape_backward |
| Dimensional reductions | sum(dim, keepdim) | Required for correct batched cross-entropy. Global reduction when dim=-1 (backward compat) |
| Mixed precision | Parse + validate only | Full bf16/fp16 requires dtype infrastructure (M17). Stub with warning for M14 |
| Optimizer state serialization | Not included | .nslm saves weights only. Compiler warns if model.load() precedes train block |

---

## Architecture Overview

```
Compiler generates:              Stdlib provides (NSL):         Runtime adds (Rust):
---------------------           ----------------------         ---------------------
- Epoch/step loop               - nsl.optim.SGD                - zeros_like(tensor)
- Implicit grad wrapping        - nsl.optim.Adam               - sqrt(tensor) elem-wise
- Grad accumulation counter     - nsl.optim.AdamW              - pow(tensor, scalar)
- Loss scaling (1/accum)        - nsl.optim.Lion               - abs(tensor) elem-wise
- Grad clipping call            - nsl.optim.Muon               - sign(tensor) elem-wise
- Optimizer step dispatch       - nsl.optim.SOAP               - exp(tensor) elem-wise
- Scheduler LR update           - nsl.optim.schedulers.*       - log(tensor) elem-wise
- Callback invocation           - nsl.nn.cross_entropy         - copy_data(dst, src)
- Evaluation block execution    - nsl.nn.mse_loss              - add_inplace(dst, src)
- Checkpoint save/load calls    - nsl.nn.l1_loss               - zero_inplace(tensor)
- Param group resolution        - nsl.nn.bce_loss              - clip_grad_norm(list, max)
                                                               - reduce_max(t, dim, keepdim)
                                                               - gather(t, dim, indices)
                                                               - clamp(t, min, max)
                                                               - cos(f64), floor(f64)
                                                               - model_save / model_load
                                                               - sum/mean dim+keepdim upgrade
```

---

## Train Block Codegen — Desugared Output

A train block like:

```nsl
train(model=m, epochs=10, accumulate=4, clip_grad_norm=1.0):
    optimizer: AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler: WarmupCosine(warmup_steps=100, total_steps=1000, min_lr=1e-5)
    step(batch):
        let logits = model.forward(x)
        loss = mse_loss(logits, y)
    callbacks:
        on_step(step, loss):
            if step % 100 == 0:
                print(loss)
    eval(epoch):
        print(epoch)
```

Desugars to:

```
# 1. Extract model params (compiler knows model fields)
params = [m.w, m.b, ...]

# 2. Create optimizer state (allocated once, never reallocated)
adam_m = [zeros_like(p) for p in params]
adam_v = [zeros_like(p) for p in params]

# 3. Init scheduler state
lr = 3e-4
step_count = 0

# 4. Epoch loop
for epoch in 0..10:
    accum_grads = [zeros_like(p) for p in params]
    running_loss = 0.0

    # 5. Accumulation loop
    for micro in 0..4:
        tape_start(params)
        # --- user step body ---
        let logits = model.forward(x)
        loss = mse_loss(logits, y)
        # --- end ---
        loss = loss * 0.25                  # Scale BEFORE backward
        running_loss += loss.item()
        grads = tape_backward(loss, params)
        tape_stop()
        # In-place accumulate
        for i in 0..len(params):
            tensor_add_inplace(accum_grads[i], grads[param_name_i])

    # 6. Gradient clipping
    clip_grad_norm(accum_grads, 1.0)

    # 7. Optimizer step (all mutations in-place via copy_data)
    for i in 0..len(params):
        adamw_step(params[i], accum_grads[i], adam_m[i], adam_v[i],
                   lr, 0.9, 0.95, 1e-8, param_wd[i], step_count + 1)

    # 8. Zero accumulators for next step
    for i in 0..len(params):
        tensor_zero_inplace(accum_grads[i])

    # 9. Scheduler update
    lr = warmup_cosine(3e-4, step_count, 100, 1000, 1e-5)
    step_count += 1

    # 10. Callbacks (averaged loss)
    on_step(step_count, running_loss)

    # 11. Evaluation
    print(epoch)
```

**Key memory safety properties:**
- Optimizer state buffers (adam_m, adam_v) allocated once, mutated in-place
- Gradient accumulators allocated once per epoch, zeroed in-place between steps
- Model parameters mutated in-place via copy_data — model struct pointers remain valid
- Loss scaled before backward to prevent fp16 overflow (future-proofing)
- Callbacks receive averaged loss across micro-batches
- Intermediate tensors from expressions freed after consumption (compiler emits tensor_release)

**Data handling:** No DataLoader yet (future milestone). Step body runs once per epoch by default. If data section specifies count=N, step runs N times per epoch.

---

## Stdlib Optimizers (Pure NSL)

All optimizers are void functions that mutate params and state in-place via copy_data.

### SGD (stdlib/nsl/optim/sgd.nsl)

```nsl
fn sgd_step(param: Tensor, grad: Tensor, velocity: Tensor,
            lr: float, momentum: float, dampening: float,
            weight_decay: float, nesterov: bool):
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    if momentum > 0.0:
        copy_data(velocity, momentum * velocity + (1.0 - dampening) * grad)
        if nesterov:
            let update = grad + momentum * velocity
        else:
            let update = velocity
        copy_data(param, param - lr * update)
    else:
        copy_data(param, param - lr * grad)
```
State: 1 buffer (velocity) per param.

### Adam (stdlib/nsl/optim/adam.nsl)

```nsl
fn adam_step(param: Tensor, grad: Tensor, m: Tensor, v: Tensor,
            lr: float, beta1: float, beta2: float, eps: float,
            weight_decay: float, t: float):
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    copy_data(m, beta1 * m + (1.0 - beta1) * grad)
    copy_data(v, beta2 * v + (1.0 - beta2) * grad * grad)
    let m_hat = m * (1.0 / (1.0 - pow(beta1, t)))
    let v_hat = v * (1.0 / (1.0 - pow(beta2, t)))
    copy_data(param, param - lr * m_hat / (sqrt(v_hat) + eps))
```
State: 2 buffers (m, v) per param.

### AdamW (stdlib/nsl/optim/adamw.nsl)

```nsl
fn adamw_step(param: Tensor, grad: Tensor, m: Tensor, v: Tensor,
              lr: float, beta1: float, beta2: float, eps: float,
              weight_decay: float, t: float):
    copy_data(m, beta1 * m + (1.0 - beta1) * grad)
    copy_data(v, beta2 * v + (1.0 - beta2) * grad * grad)
    let m_hat = m * (1.0 / (1.0 - pow(beta1, t)))
    let v_hat = v * (1.0 / (1.0 - pow(beta2, t)))
    let step_update = lr * m_hat / (sqrt(v_hat) + eps)
    # Single write: decoupled weight decay + gradient update combined
    if weight_decay > 0.0:
        copy_data(param, param * (1.0 - lr * weight_decay) - step_update)
    else:
        copy_data(param, param - step_update)
```
State: 2 buffers per param.

### Lion (stdlib/nsl/optim/lion.nsl)

```nsl
fn lion_step(param: Tensor, grad: Tensor, m: Tensor,
             lr: float, beta1: float, beta2: float,
             weight_decay: float):
    let update = sign(beta1 * m + (1.0 - beta1) * grad)
    if weight_decay > 0.0:
        copy_data(param, param * (1.0 - lr * weight_decay) - lr * update)
    else:
        copy_data(param, param - lr * update)
    copy_data(m, beta2 * m + (1.0 - beta2) * grad)
```
State: 1 buffer (m) per param.

### Muon (stdlib/nsl/optim/muon.nsl)

```nsl
fn muon_step(param: Tensor, grad: Tensor, velocity: Tensor,
             lr: float, momentum: float, nesterov: bool):
    copy_data(velocity, momentum * velocity + grad)
    if nesterov:
        let update = grad + momentum * velocity
    else:
        let update = velocity
    copy_data(param, param - lr * update)
```
State: 1 buffer per param. (Full Newton-Schulz orthogonalization deferred.)

### SOAP (stdlib/nsl/optim/soap.nsl)

```nsl
fn soap_step(param: Tensor, grad: Tensor, m: Tensor, v: Tensor,
             lr: float, beta1: float, beta2: float, eps: float):
    copy_data(m, beta1 * m + (1.0 - beta1) * grad)
    copy_data(v, beta2 * v + (1.0 - beta2) * grad * grad)
    let update = m / (sqrt(v) + eps)
    copy_data(param, param - lr * update)
```
State: 2 buffers per param. (Diagonal approximation; full Shampoo preconditioning deferred.)

---

## Stdlib Schedulers (stdlib/nsl/optim/schedulers.nsl)

Pure scalar functions: (base_lr, step, config) -> current_lr. No state, no tensors.

```nsl
fn constant_lr(base_lr: float, step: float) -> float:
    return base_lr

fn step_lr(base_lr: float, step: float, step_size: float, gamma: float) -> float:
    return base_lr * pow(gamma, floor(step / step_size))

fn exponential_lr(base_lr: float, step: float, gamma: float) -> float:
    return base_lr * pow(gamma, step)

fn linear_decay(base_lr: float, step: float, total_steps: float,
                end_factor: float) -> float:
    if step >= total_steps:
        return base_lr * end_factor
    let progress = step / total_steps
    return base_lr * (1.0 - progress * (1.0 - end_factor))

fn cosine_anneal(base_lr: float, step: float, t_max: float,
                 eta_min: float) -> float:
    if step >= t_max:
        return eta_min
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + cos(3.14159265 * step / t_max))

fn warmup_cosine(base_lr: float, step: float, warmup_steps: float,
                 total_steps: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    let progress = (step - warmup_steps) / (total_steps - warmup_steps)
    if progress >= 1.0:
        return min_lr
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + cos(3.14159265 * progress))

fn one_cycle(base_lr: float, step: float, max_lr: float,
             total_steps: float, pct_start: float) -> float:
    let warmup_end = total_steps * pct_start
    if step < warmup_end:
        return base_lr + (max_lr - base_lr) * (step / warmup_end)
    let progress = (step - warmup_end) / (total_steps - warmup_end)
    return max_lr * 0.5 * (1.0 + cos(3.14159265 * progress))
```

Runtime scalar ops needed: cos(f64) -> f64, floor(f64) -> f64.

---

## Stdlib Loss Functions (stdlib/nsl/nn/losses.nsl)

Loss functions run INSIDE grad blocks — all tensor ops must be tape-compatible.

```nsl
fn mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    let diff = pred - target
    return mean(diff * diff)

fn l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return mean(abs(pred - target))

fn cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # logits: [batch, classes], targets: [batch] (integer class indices)
    let max_val = reduce_max(logits, dim=1, keepdim=true)   # [batch, 1]
    let shifted = logits - max_val                           # [batch, classes]
    let exp_sum = sum(exp(shifted), dim=1, keepdim=true)     # [batch, 1]
    let log_probs = shifted - log(exp_sum)                   # [batch, classes]
    let nll = neg(gather(log_probs, dim=1, indices=targets)) # [batch]
    return mean(nll)                                         # scalar

fn bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    let eps_val = 1e-7
    let clamped = clamp(pred, eps_val, 1.0 - eps_val)
    return neg(mean(target * log(clamped) + (1.0 - target) * log(1.0 - clamped)))
```

---

## Runtime Additions

### New Element-wise Ops (with autodiff tape support)

| Op | Forward | Backward (tape) | Saves |
|----|---------|-----------------|-------|
| nsl_tensor_exp(a) | element-wise exp | grad * saved_output | output |
| nsl_tensor_log(a) | element-wise ln | grad / saved_input | input |
| nsl_tensor_sqrt(a) | element-wise sqrt | grad / (2 * saved_output) | output |
| nsl_tensor_abs(a) | element-wise abs | grad * sign(input) | input |
| nsl_tensor_sign(a) | element-wise sign | zero (non-differentiable) | nothing |
| nsl_tensor_clamp(a, min, max) | element-wise clamp | grad where unclamped, 0 where clamped | input + bounds |
| nsl_tensor_gather(a, dim, idx) | index selection along dim | scatter grad to indices | indices + dim |
| nsl_tensor_reduce_max(a, dim, keepdim) | max along dim | grad to argmax positions | input + argmax |

### Upgraded Existing Ops

| Op | Change |
|----|--------|
| nsl_tensor_sum | Add dim: i64 and keepdim: bool params. dim=-1 means global (backward compat) |
| nsl_tensor_mean | Add dim: i64 and keepdim: bool params. dim=-1 means global (backward compat) |

### New In-place / Mutation Ops (NOT taped)

| Op | Purpose |
|----|---------|
| nsl_tensor_copy_data(dst, src) | Memcpy src data into dst. Asserts same shape. Asserts contiguous. NOT taped |
| nsl_tensor_add_inplace(dst, src) | dst.data[i] += src.data[i]. For gradient accumulation |
| nsl_tensor_zero_inplace(t) | Zero all data. For resetting accumulators |
| nsl_tensor_zeros_like(t) | Allocate new tensor with same shape, filled with zeros |

### New Scalar Ops

| Op | Purpose |
|----|---------|
| nsl_cos(f64) -> f64 | For cosine schedulers |
| nsl_floor(f64) -> f64 | For StepLR scheduler |

### Gradient Clipping

nsl_clip_grad_norm(grad_list: *NslList, max_norm: f64) — computes global gradient norm across all tensors in list, scales all gradients if norm exceeds max_norm.

### Checkpoint I/O

nsl_model_save(path, param_names: *NslList, param_tensors: *NslList):
- Asserts all tensors are contiguous (errors with clear message if not — no silent corruption)
- Writes little-endian f64 data
- Format: "NSLM" magic (4B) + version u32 (4B) + header_size u64 (8B) + JSON header + 64-byte aligned padding + raw tensor data

nsl_model_load(path, param_tensors: *NslList):
- Reads .nslm file, validates magic/version
- Writes data into existing param tensors via memcpy (model struct stays valid)

---

## Limitations (Documented)

- **Checkpoint saves weights only** — optimizer state (momentum buffers) is NOT serialized. Training resumes with fresh optimizer state. Compiler emits warning if model.load() precedes train block.
- **Mixed precision** — precision=bf16 parses but emits warning. All computation stays f64. Full support in M17.
- **Muon/SOAP** — Simplified implementations (momentum-only Muon, diagonal SOAP). Full algorithms require matrix decomposition ops.
- **Data loading** — No DataLoader. Step body runs once per epoch. data: count=N repeats step N times per epoch.
- **Non-contiguous tensors** — model_save errors on non-contiguous tensors. No auto-contiguous clone yet.

---

## Parameter Groups

Compiler resolves glob patterns against model param names at compile time:

```nsl
optimizer: AdamW(lr=6e-4, groups=[
    {params: "*.weight", weight_decay: 0.1},
    {params: "*.bias", weight_decay: 0.0}
])
```

Each param gets tagged with its group overrides. The codegen emits per-param hyperparameter arrays:

```
param_wd = [0.1, 0.0, 0.1, 0.0, ...]  # resolved at compile time
for i in 0..len(params):
    adamw_step(params[i], grads[i], m[i], v[i], lr, ..., param_wd[i], t)
```

model.params(filter=...) is NOT a runtime call — the compiler statically resolves it.

---

## Semantic Validation (checker.rs)

check_train_block() must validate:
1. model= config is required and references a defined model variable
2. epochs= is required and is a positive integer
3. optimizer: section is required, references a known optimizer type
4. scheduler: is optional, references a known scheduler type
5. step(batch): section is required, body must assign to loss
6. accumulate= if present, must be positive integer
7. clip_grad_norm= if present, must be positive float
8. precision= if present and not f64/fp32, emit warning
9. eval(epoch): is optional
10. callbacks: is optional, callback names must be on_step or on_epoch
11. groups= patterns validated against model known param names

---

## Testing Strategy

### Unit Tests (Rust)
- All new runtime ops with shape verification
- Dimensional reductions: sum(dim=1), mean(dim=0), reduce_max(dim=1, keepdim=true)
- Autodiff backward passes for all new tape ops
- Checkpoint save/load roundtrip

### Integration Tests (NSL)
- m14_sgd_basic.nsl — SGD on linear regression, verify loss decreases
- m14_adam_scheduler.nsl — AdamW + WarmupCosine, verify training works
- m14_cross_entropy.nsl — cross_entropy on batched logits
- m14_checkpoint.nsl — save/load roundtrip, verify predictions match
- m14_grad_accum.nsl — gradient accumulation (accumulate=4)
- m14_param_groups.nsl — per-param weight decay
- m14_all_optimizers.nsl — all 6 optimizers produce decreasing loss
- m14_all_schedulers.nsl — all 7 schedulers produce expected LR curves
- m14_callbacks.nsl — on_step callback fires with correct loss
