# Section 5 — Training Loop & Optimization

## Design Rationale

PyTorch training loops are boilerplate-heavy: 40+ lines of repetitive code for `zero_grad`,
`forward`, `loss`, `backward`, `clip`, `step`, `scheduler.step`, `logging`, `checkpointing` —
repeated in every project. NSL's `train` block is a first-class construct that encapsulates
the entire training pipeline with declarative configuration. The compiler can optimize the
entire loop holistically (e.g., fusing optimizer steps with gradient computation, overlapping
data loading with computation).

## Train Block Grammar

```ebnf
train_block     ::= 'train' '(' train_config ')' ':' INDENT train_body DEDENT
train_config    ::= config_entry (',' config_entry)*
config_entry    ::= IDENT '=' expression
train_body      ::= (train_stmt NEWLINE)*
train_stmt      ::= 'data' ':' data_config
                   | 'optimizer' ':' optim_config
                   | 'scheduler' ':' sched_config
                   | 'loss' ':' loss_def
                   | 'step' ':' INDENT step_body DEDENT
                   | 'callbacks' ':' callback_list
                   | 'eval' ':' INDENT eval_body DEDENT
                   | statement

optim_config    ::= optim_type '(' optim_params ')'
optim_type      ::= 'Adam' | 'AdamW' | 'SGD' | 'Lion' | 'Muon' | 'SOAP'
                   | 'Adagrad' | 'RMSProp' | 'LARS' | 'LAMB'

sched_config    ::= sched_type '(' sched_params ')'
sched_type      ::= 'Cosine' | 'WarmupCosine' | 'LinearDecay' | 'OneCycle'
                   | 'ConstantLR' | 'StepLR' | 'ExponentialLR'

# Distributed training annotations
distribute_ann  ::= 'distribute' '(' dist_config ')'
dist_config     ::= 'strategy' '=' dist_strategy (',' dist_params)*
dist_strategy   ::= 'ddp' | 'fsdp' | 'pipeline' | 'tensor_parallel' | 'zero3'
```

## Built-in Optimizers

| Optimizer | Constructor                                                          | Description                              |
|-----------|----------------------------------------------------------------------|------------------------------------------|
| `Adam`    | `Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`     | Standard Adam                            |
| `AdamW`   | `AdamW(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`   | Adam with decoupled weight decay         |
| `SGD`     | `SGD(lr=0.01, momentum=0.9, nesterov=false)`                         | Stochastic gradient descent              |
| `Lion`    | `Lion(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)`                | Google Brain's Lion optimizer            |
| `Muon`    | `Muon(lr=0.02, momentum=0.95, nesterov=true, ns_steps=5)`           | Muon optimizer with Nesterov momentum    |
| `SOAP`    | `SOAP(lr=1e-3, betas=(0.95, 0.95), shampoo_beta=0.95)`              | SOAP preconditioned optimizer            |

## Built-in Schedulers

| Scheduler       | Constructor                                                    | Description                              |
|-----------------|----------------------------------------------------------------|------------------------------------------|
| `Cosine`        | `Cosine(T_max, eta_min=0.0)`                                  | Cosine annealing                         |
| `WarmupCosine`  | `WarmupCosine(warmup_steps, total_steps, min_lr=0.0)`          | Linear warmup then cosine decay          |
| `LinearDecay`   | `LinearDecay(total_steps, end_factor=0.0)`                     | Linear learning rate decay               |
| `OneCycle`      | `OneCycle(max_lr, total_steps, pct_start=0.3)`                 | 1cycle policy (super-convergence)        |
| `StepLR`        | `StepLR(step_size, gamma=0.1)`                                 | Decay by gamma every step_size epochs    |
| `ConstantLR`    | `ConstantLR()`                                                 | No scheduling (constant LR)             |

## 5 Complete Training Loop Examples

### Example 1: Simple Image Classifier Training

```nsl
## Train a CNN image classifier on CIFAR-10 with standard settings.
## This demonstrates the minimal train block syntax.

model CNN:
    conv1: Conv2d = Conv2d(3, 32, kernel=3, padding=1)
    conv2: Conv2d = Conv2d(32, 64, kernel=3, padding=1)
    pool: MaxPool2d = MaxPool2d(2)
    fc1: Linear = Linear(64 * 8 * 8, 256)
    fc2: Linear = Linear(256, 10)

    fn forward(x: Tensor<[batch, 3, 32, 32], fp32>) -> Tensor<[batch, 10], fp32>:
        let h = self.conv1(x) |> relu |> self.pool
        let h = self.conv2(h) |> relu |> self.pool
        let h = h.flatten(start_dim=1)
        return self.fc1(h) |> relu |> self.fc2

let cnn = CNN().to(cuda)
let dataset = nsl.data.CIFAR10(root="./data", train=true)

train(model=cnn, epochs=20, precision=fp32):
    # Data configuration
    data:
        source = dataset
        batch_size = 128
        shuffle = true
        num_workers = 4

    # Optimizer and scheduler
    optimizer: SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler: StepLR(step_size=7, gamma=0.1)

    # Training step — called for each batch
    step(batch):
        let (images, labels) = batch
        let logits = model.forward(images)
        loss = cross_entropy(logits, labels)

    # Evaluation — called at end of each epoch
    eval(epoch):
        let test_data = nsl.data.CIFAR10(root="./data", train=false)
        let accuracy = evaluate_accuracy(model, test_data)
        log.metric("test_accuracy", accuracy)
        if accuracy > 0.93:
            model.save(f"best_cnn_epoch{epoch}.nslm")
```

### Example 2: LLM Pre-training with Mixed Precision and Gradient Accumulation

```nsl
## Pre-train a transformer language model with full production settings:
## BF16 mixed precision, gradient accumulation, cosine scheduler with warmup,
## gradient clipping, and periodic checkpointing.

let llm = TransformerDecoder(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072
).to(cuda)

let corpus = dataset("webtext"):
    source = nsl.data.MemoryMapped("data/openwebtext.bin")
    sequence_length = 2048
    packing = true    # pack multiple documents into one sequence

train(
    model=llm,
    epochs=1,
    precision=bf16,                    # automatic mixed precision
    grad_scaler=auto,                   # dynamic loss scaling for bf16
    accumulate=8,                       # 8 micro-batches per optimizer step
    clip_grad_norm=1.0                  # global gradient norm clipping
):
    data:
        source = corpus
        batch_size = 4                  # micro-batch size (effective = 4 * 8 = 32)
        num_workers = 8
        prefetch = 2

    optimizer: AdamW(
        lr=6e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        # Per-parameter group: no weight decay on biases and norms
        groups=[
            {params: model.params(filter="*.weight"), weight_decay: 0.1},
            {params: model.params(filter="*.bias|*.norm*"), weight_decay: 0.0}
        ]
    )

    scheduler: WarmupCosine(
        warmup_steps=2000,
        total_steps=100000,
        min_lr=6e-5                     # 10x reduction at end
    )

    step(batch):
        let logits = model.forward(batch.input_ids)
        # Shift logits and labels for next-token prediction
        let shift_logits = logits[:, :-1, :]
        let shift_labels = batch.input_ids[:, 1:]
        loss = cross_entropy(shift_logits.flatten(0, 1), shift_labels.flatten())

    callbacks:
        on_step(step, loss):
            if step % 100 == 0:
                log.metric("train/loss", loss)
                log.metric("train/lr", scheduler.get_lr())
                log.metric("train/grad_norm", grads.global_norm())

        on_step(step, _):
            if step % 5000 == 0:
                model.save(f"checkpoints/step_{step}.nslm")

    eval(epoch):
        let val_loss = evaluate_loss(model, val_data)
        log.metric("val/loss", val_loss)
        log.metric("val/perplexity", val_loss.exp())
```

### Example 3: Fine-tuning with LoRA

```nsl
## Fine-tune a pre-trained model using Low-Rank Adaptation (LoRA).
## Only LoRA parameters are trained — base model is frozen.

import nsl.nn.lora.{LoRA, apply_lora}

# Load pre-trained model and freeze it
let base_model = TransformerDecoder.load("llama-7b.nslm")
base_model.freeze()    # all parameters become non-trainable

# Apply LoRA adapters to attention projections
let lora_model = apply_lora(base_model,
    target_modules=["wq", "wk", "wv", "wo"],
    rank=16,
    alpha=32,
    dropout=0.05
)

# Only LoRA parameters are trainable
print(f"trainable params: {lora_model.trainable_count()}")  # ~4M vs 7B total

let finetune_data = dataset("instruct"):
    source = nsl.data.JSONL("data/alpaca.jsonl")
    format = InstructFormat(
        instruction_key="instruction",
        input_key="input",
        output_key="output"
    )
    max_length = 2048

train(model=lora_model, epochs=3, precision=bf16):
    data:
        source = finetune_data
        batch_size = 8
        shuffle = true

    optimizer: AdamW(lr=2e-4, weight_decay=0.0)    # no weight decay for LoRA
    scheduler: WarmupCosine(warmup_steps=100, total_steps=3000)

    step(batch):
        let logits = model.forward(batch.input_ids)
        # Only compute loss on output tokens (mask instruction tokens)
        loss = cross_entropy(logits, batch.labels, ignore_index=-100)

    eval(epoch):
        let score = evaluate_on_benchmark(model, "mmlu")
        log.metric("mmlu_score", score)

# Save only the LoRA weights (small file)
lora_model.save_adapter("lora_alpaca.nslm")

# Merge LoRA weights into base model for inference
let merged = lora_model.merge()
merged.save("llama-7b-alpaca-merged.nslm")
```

### Example 4: Distributed Training with FSDP

```nsl
## Multi-GPU training with Fully Sharded Data Parallel (FSDP).
## The model is sharded across GPUs, each GPU processes a slice of data.

let model = TransformerDecoder(
    vocab_size=50257, d_model=4096, n_layers=32, n_heads=32, d_ff=11008
)

# Distribute across 8 GPUs with FSDP
distribute(strategy=fsdp, ranks=8):
    # FSDP configuration
    sharding = full             # fully shard parameters, gradients, and optimizer states
    mixed_precision = bf16       # communication in bf16
    cpu_offload = false          # keep everything on GPU
    activation_checkpointing = true
    wrap_policy = transformer_layer(TransformerBlock)   # shard at block boundaries

train(
    model=model,
    epochs=1,
    precision=bf16,
    accumulate=4,
    clip_grad_norm=1.0
):
    data:
        source = corpus
        batch_size = 2          # per-GPU micro-batch
        num_workers = 4
        # Automatically uses DistributedSampler — each GPU gets different data
        distributed = true

    optimizer: AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler: WarmupCosine(warmup_steps=1000, total_steps=50000)

    step(batch):
        let logits = model.forward(batch.input_ids)
        loss = cross_entropy(logits[:, :-1, :].flatten(0, 1),
                             batch.input_ids[:, 1:].flatten())

    callbacks:
        on_step(step, loss):
            # Only log from rank 0
            if dist.rank() == 0 and step % 100 == 0:
                log.metric("train/loss", loss)

        on_step(step, _):
            if step % 5000 == 0:
                # FSDP-aware checkpoint: gathers full state dict
                model.save(f"checkpoints/step_{step}.nslm", distributed=true)
```

### Example 5: Reinforcement Learning from Human Feedback (RLHF)

```nsl
## RLHF training with PPO. Demonstrates:
## - Multiple models in one training loop
## - Custom step logic with reward model
## - KL divergence penalty against reference model

let policy = TransformerDecoder.load("sft_model.nslm").to(cuda)
let ref_model = TransformerDecoder.load("sft_model.nslm").to(cuda)
ref_model.freeze()
let reward_model = RewardModel.load("reward_model.nslm").to(cuda)
reward_model.freeze()

let prompts = dataset("prompts"):
    source = nsl.data.JSONL("data/prompts.jsonl")

train(model=policy, epochs=1, precision=bf16):
    data:
        source = prompts
        batch_size = 16
        shuffle = true

    optimizer: AdamW(lr=1e-6, weight_decay=0.0)
    scheduler: ConstantLR()

    step(batch):
        # 1. Generate completions from policy
        let completions = @no_grad policy.generate(
            batch.prompt_ids, max_tokens=256, temperature=0.7
        )

        # 2. Score with reward model
        let rewards = @no_grad reward_model.score(completions)

        # 3. Compute KL penalty against reference model
        let policy_logprobs = policy.log_probs(completions)
        let ref_logprobs = @no_grad ref_model.log_probs(completions)
        let kl_penalty = (policy_logprobs - ref_logprobs).mean()

        # 4. PPO objective
        let advantages = compute_advantages(rewards, kl_penalty, gamma=1.0, lam=0.95)
        let ratio = (policy_logprobs - old_logprobs).exp()
        let clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
        let ppo_loss = -min(ratio * advantages, clipped * advantages).mean()

        # 5. Combined loss
        loss = ppo_loss + 0.1 * kl_penalty

    callbacks:
        on_step(step, loss):
            if step % 10 == 0:
                let mean_reward = rewards.mean().item()
                log.metric("rlhf/reward", mean_reward)
                log.metric("rlhf/kl", kl_penalty.item())
                log.metric("rlhf/loss", loss.item())
```

## Train Block Compiler Optimizations

The compiler can optimize the `train` block holistically:

1. **Overlapped data loading**: Data for step N+1 is loaded while step N computes
2. **Fused optimizer**: Gradient clipping + optimizer step are fused into one kernel
3. **Communication overlap**: In distributed training, gradient all-reduce overlaps with backward computation
4. **Memory planning**: The compiler knows the entire training loop and can plan memory allocation to avoid fragmentation
5. **Automatic mixed precision**: The compiler inserts cast operations and handles loss scaling automatically when `precision=bf16`

## Design Tensions & Tradeoffs

1. **DSL vs Code**: The `train` block is a DSL — it constrains how training loops look. Custom
   training algorithms (like RLHF or GAN training with multiple models) push against this
   structure. Mitigation: the `step` block is arbitrary NSL code, giving full flexibility
   where it matters most. Only the outer structure (data, optimizer, scheduler) is declarative.

2. **Implicit vs Explicit**: `train(accumulate=8)` hides the accumulation loop. Users who
   need to customize accumulation behavior (e.g., different scaling per accumulation step)
   must drop to the manual `grad` + `GradAccumulator` API from Section 3.

3. **Parameter groups**: Per-parameter optimizer settings (different LR for different layers)
   use a filter pattern syntax. This is less flexible than PyTorch's arbitrary param groups
   but covers 95% of use cases with far less code.
