# NeuralScript: Training & Automatic Differentiation

## Overview

NSL implements training as a language feature, not a library. The `grad` keyword, `train` block DSL, and autodiff system are all built into the compiler and runtime, enabling compile-time optimization of training loops that Python frameworks cannot achieve.

---

## Automatic Differentiation

### The `grad` Keyword
```
let loss = model.forward(x, y)
let grads = grad(loss)    # reverse-mode autodiff
```

`grad` is a **keyword**, not a function call. The compiler emits tape recording instructions around the forward pass and backward pass computation after `grad()`.

### Tape-Based Reverse Mode
The primary autodiff implementation uses a global tape:

1. **Forward pass**: Each differentiable operation appends a `TapeOp` to the global tape
2. **`grad()` call**: Triggers backward pass — walks tape in reverse, applying adjoint rules
3. **Gradient accumulation**: Gradients accumulated per-parameter via pointer-keyed hashmap

**Tape operations** (defined in `nsl-runtime/src/autodiff/`):
- Arithmetic: Add, Sub, Mul, Div, MatMul
- Activations: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax
- Normalization: LayerNorm, RMSNorm
- Shape: Slice, Concat, Reshape, Transpose
- Other: Dropout, Embedding lookup

**Critical implementation detail**: Tape ops store raw i64 pointers to tensors. These pointers are used **only as hashmap keys** for gradient accumulation — they must never be dereferenced because the original tensors may have been freed. Shape information is stored separately in the tape op.

### Source-to-Source AD (M40)
An alternative to tape-based AD that transforms the forward computation into an explicit backward function:

**File**: `source_ad.rs`, `wengert.rs`

1. **Wengert extraction**: Converts forward pass AST into a Wengert list (sequence of elementary operations)
2. **Adjoint generation**: Walks Wengert ops in reverse, applying AD rules
3. **Dead code elimination**: Removes unused adjoint variables

Advantages over tape-based:
- No runtime tape overhead
- Enables compiler optimizations on the backward pass
- Better memory planning (backward graph known at compile time)

### Decorators
```
@no_grad
fn inference_only(model, x):
    return model.forward(x)    # no tape recording

@checkpoint
fn memory_efficient_layer(x):
    ...    # recompute forward during backward instead of caching activations

@backward
fn custom_backward(x, grad_output):
    return custom_gradient(x, grad_output)
```

---

## Training DSL

### Train Block
```
train my_training:
    model = Transformer(vocab_size=32000, hidden=512, layers=6)
    optimizer = AdamW(lr=3e-4, weight_decay=0.1)
    scheduler = cosine_anneal(min_lr=1e-5, warmup_steps=1000)
    loss_fn = cross_entropy
    data = DataLoader("train.bin", batch_size=32, shuffle=true)
    epochs = 10

    step:
        let logits = model.forward(batch.input)
        let loss = loss_fn(logits, batch.target)
        return loss

    on_epoch(epoch, avg_loss):
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    on_step(step, loss):
        if step % 100 == 0:
            print(f"Step {step}: {loss:.4f}")
```

The `train` block is a **declarative DSL** that compiles to an optimized training loop:

### What the Compiler Generates
1. **Epoch loop**: Iterates over data for specified epochs
2. **Implicit tape management**: `tape_start()` before forward, `backward()` after loss, `tape_stop()` after gradient step
3. **Per-parameter optimizer dispatch**: Applies the chosen optimizer to each `Param` in the model
4. **Scheduler injection**: Automatically injects base_lr and step count into scheduler calls
5. **Auto-import**: Compiler automatically imports required optimizer/scheduler stdlib modules

### Train Block Components

| Component | Purpose | Required |
|-----------|---------|----------|
| `model` | The model to train | Yes |
| `optimizer` | Optimizer instance | Yes |
| `loss_fn` | Loss function | Yes |
| `data` | DataLoader for training data | Yes |
| `epochs` | Number of training epochs | Yes |
| `scheduler` | Learning rate scheduler | No |
| `step` | Custom forward/loss computation | No (default: forward + loss_fn) |
| `on_epoch` | Per-epoch callback | No |
| `on_step` | Per-step callback | No |

---

## Optimizers (stdlib, 6 implementations)

All optimizers are written in NSL and compiled alongside user code.

### SGD
```
optimizer = SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
```
Classic stochastic gradient descent with optional momentum and weight decay.

### Adam
```
optimizer = Adam(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
```
Adaptive moment estimation with first and second moment tracking.

### AdamW
```
optimizer = AdamW(lr=3e-4, beta1=0.9, beta2=0.999, weight_decay=0.1)
```
Adam with decoupled weight decay (correct L2 regularization).

### Lion
```
optimizer = Lion(lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=1e-2)
```
Sign-based optimizer using interpolation between current and EMA gradients.

### Muon
```
optimizer = Muon(lr=0.02, momentum=0.95)
```
Momentum-based optimizer with orthogonalization.

### SOAP
```
optimizer = SOAP(lr=3e-4, beta1=0.95, beta2=0.95, shampoo_beta=0.9)
```
Shampoo-based optimizer with preconditioning.

### Implementation Notes
- **In-place updates**: Optimizer state updated via `copy_data()`, buffers allocated once via `zeros_like()`
- **`grad` is a keyword**: Optimizer stdlib code uses `gradient` for the gradient parameter name (not `grad`)
- **`model` is a keyword**: Parser required fix for `model=m` in train config (variable named `model`)
- **`pow()` not available**: Use `**` operator (`beta1 ** t` not `pow(beta1, t)`)

---

## Learning Rate Schedulers (stdlib, 7 implementations)

```
scheduler = cosine_anneal(min_lr=1e-5, warmup_steps=1000)
scheduler = warmup_cosine(warmup_steps=500, min_lr=0.0)
scheduler = one_cycle(max_lr=3e-4, total_steps=10000)
```

| Scheduler | Behavior |
|-----------|----------|
| `constant_lr` | Fixed learning rate |
| `step_lr` | Multiply by gamma every N steps |
| `exponential_lr` | Multiply by gamma each step |
| `linear_decay` | Linear decay from base_lr to min_lr |
| `cosine_anneal` | Cosine annealing with optional warmup |
| `warmup_cosine` | Linear warmup then cosine decay |
| `one_cycle` | Ramp up to max_lr then decay (super-convergence) |

---

## Loss Functions (stdlib)

```
let loss = cross_entropy(logits, targets)
let loss = mse_loss(predictions, targets)
let loss = l1_loss(predictions, targets)
let loss = bce_loss(predictions, targets)  # binary cross-entropy
```

All loss functions return scalar tensors compatible with `grad()`.

---

## Data Pipeline

### DataLoader (stdlib)
```
import nsl.data.loader

let dl = DataLoader("train.bin", batch_size=32, shuffle=true)
for batch in dl:
    let x = batch.input
    let y = batch.target
```

- Batching with configurable batch size
- Optional shuffling per epoch
- Binary data format support

### Tokenization
```
import nsl.tokenize.tokenizer

let tok = Tokenizer("tokenizer.json")
let ids = tok.encode("Hello world")
let text = tok.decode(ids)
```

BPE tokenizer powered by HuggingFace `tokenizers` crate (Rust, no Python).

---

## Model Checkpointing

### Save/Load Format (.nslm)
```
# Save
save_checkpoint(model, "checkpoint.nslm")

# Load
load_checkpoint(model, "checkpoint.nslm")
```

Binary format:
- **Magic**: `NSLM` (4 bytes)
- **Header**: JSON metadata (parameter names, shapes, dtypes)
- **Data**: 64-byte aligned f64 arrays for each parameter

No pickle, no Python — safe, portable, and fast.

---

## Tensor Operations (Runtime)

The runtime (`nsl-runtime`) provides all tensor operations via C ABI:

### Creation
`zeros`, `ones`, `rand`, `randn`, `empty`, `full`, `arange`, `eye`

### Arithmetic
`+`, `-`, `*`, `/`, `@` (matmul), `**` (power) — with broadcasting

### Reductions
`sum`, `mean`, `max`, `min`, `argmax`

### Shape Operations
`reshape`, `transpose`, `clone`, `expand`, `slice`, `cat`, `stack`, `squeeze`, `unsqueeze`, `gather`, `scatter`

### Activations
`relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `elu`

### Math
`exp`, `log`, `sqrt`, `abs`, `sign`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`

All operations support both CPU (f64) and GPU (f32) with automatic device dispatch.
