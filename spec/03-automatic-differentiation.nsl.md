# Section 3 — Automatic Differentiation

## Design Rationale

In PyTorch, autodiff is a runtime system (`autograd`) that dynamically builds computation
graphs. This is flexible but opaque — users can't inspect the graph at compile time, and
errors surface late. In NSL, `grad` is a **language keyword** that tells the compiler to
instrument a block for differentiation. The compiler can then verify differentiability,
optimize the backward pass, and apply gradient checkpointing — all before the code runs.

## Grad Keyword Semantics

```ebnf
grad_block      ::= 'grad' '(' grad_target ')' ':' INDENT diff_block DEDENT
grad_target     ::= IDENT (',' IDENT)*                 # parameters to differentiate w.r.t.
diff_block      ::= statement*                          # must end with a scalar expression
grad_result     ::= 'grad' '(' expr ',' wrt '=' target ')'  # inline gradient computation

annotation      ::= '@no_grad' | '@checkpoint' | '@backward' | '@custom_vjp'
```

### Rules:

1. **`grad` blocks** must produce a **scalar** output (the loss). Gradients are computed
   w.r.t. all `Param` tensors reachable from the computation, unless restricted with `wrt=`.

2. **`@no_grad`** suppresses gradient tracking for a block — used for inference, evaluation,
   or non-differentiable operations (like argmax).

3. **`@checkpoint`** tells the compiler to discard intermediate activations and recompute
   them during the backward pass — trades compute for memory.

4. **`@backward`** lets users define custom backward passes for non-standard operations.

5. **Higher-order gradients** are supported by nesting `grad` calls. The compiler builds
   a second-order computation graph automatically.

6. **Gradient flow** is tracked through the type system: `Param` types always participate
   in gradients; `Buffer` and regular `Tensor` types do not.

## 5 Annotated Examples

### Example 1: Scalar Loss — Standard Training Gradient

```nsl
# The most common pattern: compute loss, get gradients, update parameters.
# 'grad' instruments the block and computes dL/d(param) for all Param tensors.

model my_model:
    w: Param<[784, 10], fp32> = init.xavier([784, 10])
    b: Param<[10], fp32> = init.zeros([10])

    fn forward(x: Tensor<[batch, 784], fp32>) -> Tensor<[batch, 10], fp32>:
        return x @ self.w + self.b @broadcast

# Computing gradients
let x = load_batch()            # [32, 784]
let y = load_labels()           # [32, 10] one-hot

# 'grad' block: everything inside is traced for differentiation
let loss, grads = grad(my_model.params()):
    let logits = my_model.forward(x)
    let loss = cross_entropy(logits, y)
    loss    # last expression is the scalar to differentiate

# grads is a dict-like structure: {param_name: gradient_tensor}
# grads.w has shape [784, 10], same as w
# grads.b has shape [10], same as b

# Apply gradients manually (or use an optimizer)
my_model.w -= 0.01 * grads.w
my_model.b -= 0.01 * grads.b
```

### Example 2: Custom Backward Pass

```nsl
# Sometimes the automatic backward is numerically unstable or inefficient.
# @backward lets you define the gradient computation manually.
# The forward function returns the output; the backward receives the upstream
# gradient (grad_output) and returns gradients w.r.t. each input.

@backward
fn stable_softmax(x: Tensor<[batch, vocab], fp32>) -> Tensor<[batch, vocab], fp32>:
    ## Forward pass
    let shifted = x - x.max(dim=-1, keepdim=true)
    let exp_x = shifted.exp()
    let sum_exp = exp_x.sum(dim=-1, keepdim=true)
    return exp_x / sum_exp

    ## Custom backward: receives grad_output, returns grad_input
    backward(grad_output: Tensor<[batch, vocab], fp32>,
             output: Tensor<[batch, vocab], fp32>) -> Tensor<[batch, vocab], fp32>:
        # Numerically stable softmax backward
        let s = output
        let ds = grad_output
        let sum_term = (ds * s).sum(dim=-1, keepdim=true)
        return s * (ds - sum_term)

# Usage — the custom backward is automatically used during grad computation
let loss, grads = grad(model.params()):
    let probs = stable_softmax(logits)
    let loss = -log(probs.gather(dim=-1, index=targets)).mean()
    loss
```

### Example 3: Second-Order Gradients (Hessian-Vector Product)

```nsl
# Higher-order gradients are computed by nesting grad calls.
# This is useful for meta-learning (MAML), Hessian-free optimization,
# and regularization techniques that penalize gradient magnitude.

fn hessian_vector_product(
    model: MyModel,
    x: Tensor<[batch, features], fp32>,
    y: Tensor<[batch], int32>,
    v: dict<str, Tensor>          # the vector to multiply with the Hessian
) -> dict<str, Tensor>:

    # First-order gradient
    let loss, first_grads = grad(model.params()):
        let out = model.forward(x)
        cross_entropy(out, y)

    # Compute dot product of gradients with vector v
    let grad_dot_v = sum(
        (g * v_i).sum()
        for (g, v_i) in zip(first_grads.values(), v.values())
    )

    # Second-order gradient: differentiate (grad . v) w.r.t. params
    let _, hvp = grad(model.params()):
        grad_dot_v

    return hvp    # This is H @ v, the Hessian-vector product

# Example: gradient penalty regularization
fn gradient_penalty_loss(
    model: MyModel,
    x: Tensor<[batch, features], fp32>,
    y: Tensor<[batch], int32>,
    lambda_gp: f32 = 10.0
) -> Tensor<[], fp32>:

    let loss, grads = grad(model.params()):
        let out = model.forward(x)
        cross_entropy(out, y)

    # Penalize the norm of the gradient — requires second-order grad
    let grad_norm = sum(g.norm() ** 2 for g in grads.values()).sqrt()

    # This second grad call differentiates through the first grad call
    let total_loss, _ = grad(model.params()):
        loss + lambda_gp * (grad_norm - 1.0) ** 2

    return total_loss
```

### Example 4: Gradient Clipping

```nsl
# Gradient clipping is a first-class operation on gradient dictionaries.
# NSL provides built-in clipping functions that operate on the grad result.

fn train_step(model: MyModel, batch: Batch, optimizer: AdamW) -> f32:
    let loss, grads = grad(model.params()):
        let logits = model.forward(batch.input_ids)
        cross_entropy(logits, batch.labels)

    # Clip gradients by global norm (most common for transformers)
    let clipped_grads = grads.clip_norm(max_norm=1.0)

    # Alternative: clip by value
    # let clipped_grads = grads.clip_value(min=-0.5, max=0.5)

    # Alternative: clip by individual parameter norm
    # let clipped_grads = grads.clip_norm_per_param(max_norm=1.0)

    # Report clipping statistics
    let grad_norm = grads.global_norm()
    if grad_norm > 1.0:
        log.warn(f"gradient clipped: {grad_norm:.4f} -> 1.0")

    # Apply clipped gradients
    optimizer.step(clipped_grads)
    return loss.item()
```

### Example 5: Gradient Accumulation with Checkpointing

```nsl
# Gradient accumulation simulates larger batch sizes on limited memory.
# @checkpoint saves memory by recomputing activations during backward.

const ACCUMULATION_STEPS = 8
const MICRO_BATCH = 4
# Effective batch size = MICRO_BATCH * ACCUMULATION_STEPS = 32

fn train_epoch(model: TransformerLM, dataloader: DataLoader, optimizer: AdamW):
    let accumulated_grads = GradAccumulator(model.params())

    for (step, micro_batch) in dataloader.enumerate():
        # @checkpoint: recompute forward activations during backward
        # Saves ~60% memory for transformer models
        let loss, grads = grad(model.params()):
            @checkpoint
            let hidden = model.encoder(micro_batch.input_ids)
            @checkpoint
            let logits = model.decoder(hidden)
            cross_entropy(logits, micro_batch.labels) / ACCUMULATION_STEPS

        # Accumulate gradients
        accumulated_grads.add(grads)

        # Only step the optimizer every ACCUMULATION_STEPS
        if (step + 1) % ACCUMULATION_STEPS == 0:
            let final_grads = accumulated_grads.finalize()
            let clipped = final_grads.clip_norm(max_norm=1.0)
            optimizer.step(clipped)
            optimizer.zero_grad()
            accumulated_grads.reset()

            log.info(f"step {step // ACCUMULATION_STEPS}: loss={loss.item():.4f}")
```

## @no_grad Annotation

```nsl
# @no_grad suppresses gradient tracking — used for inference and evaluation.
# Any Param tensors accessed inside @no_grad are treated as frozen constants.

@no_grad
fn evaluate(model: MyModel, test_loader: DataLoader) -> f32:
    let total_correct = 0
    let total_samples = 0
    for batch in test_loader:
        let logits = model.forward(batch.inputs)
        let preds = logits.argmax(dim=-1)
        total_correct += (preds == batch.labels).sum().item()
        total_samples += batch.labels.shape[0]
    return total_correct / total_samples

# @no_grad can also wrap individual expressions
fn inference(model: MyModel, x: Tensor) -> Tensor:
    return @no_grad model.forward(x)    # inline no_grad
```

## How the Compiler Handles grad

1. **Graph construction**: When the compiler sees a `grad` block, it generates two versions
   of the code — a forward pass and a backward pass. The backward pass is derived by applying
   the chain rule to each operation in reverse order.

2. **Differentiability checking**: The compiler verifies that all operations in the `grad`
   block are differentiable. Non-differentiable ops (like `argmax`, `round`, integer casts)
   produce compile warnings unless wrapped in `@no_grad` or `straight_through()`.

3. **Memory planning**: The compiler determines which intermediate tensors must be retained
   for the backward pass and which can be freed. `@checkpoint` annotations override this —
   marked tensors are freed and recomputed.

4. **Fusion**: The compiler can fuse forward and backward kernels when beneficial
   (e.g., fused softmax + cross-entropy backward).

## Design Tensions & Tradeoffs

1. **Static vs Dynamic graphs**: NSL's `grad` is fundamentally a static graph system — the
   compiler analyzes the `grad` block at compile time. For dynamic control flow inside `grad`
   (e.g., `if` statements that depend on tensor values), the compiler generates all branches
   and selects at runtime. This limits expressiveness compared to PyTorch's fully dynamic
   autograd but enables much better optimization.

2. **Scalar output requirement**: Requiring `grad` blocks to produce a scalar simplifies the
   API enormously (no need for `jacobian`, `vjp`, `jvp` as separate concepts). For Jacobians,
   users can call `grad` in a loop or use `nsl.autograd.jacobian()` which does this internally.

3. **@checkpoint granularity**: Too few checkpoints = OOM. Too many = slow recomputation.
   The compiler can auto-suggest checkpoint placement based on memory budget analysis, but
   manual placement gives users control. Future: `@checkpoint(budget=8GB)` for automatic
   placement within a memory budget.
