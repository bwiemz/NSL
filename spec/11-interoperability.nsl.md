# Section 11 — Interoperability

## Design Rationale

No ML language can succeed in isolation — the PyTorch/HuggingFace ecosystem has thousands
of pre-trained models, datasets, and utilities. NSL must be a first-class citizen in this
ecosystem from day one. The interop layer provides seamless bridging: NSL models can be
wrapped for PyTorch consumption, PyTorch models can be imported into NSL, and HuggingFace
weights load natively. The goal is zero-friction migration: start using NSL for new code
while keeping existing PyTorch code running.

## Interop API Design

### Core Principles

1. **Zero-copy where possible**: Tensor data is shared between NSL and PyTorch via the
   DLPack standard — no copying unless dtype conversion is required.
2. **Type-safe bridging**: When importing a PyTorch model, NSL infers the types from the
   model's parameter shapes and annotates them. Users can refine these annotations.
3. **Bidirectional**: NSL → PyTorch and PyTorch → NSL are equally supported.
4. **Gradients flow across boundaries**: A hybrid NSL+PyTorch computation graph supports
   backpropagation across the boundary.

## 4 Bridging Examples

### Example 1: Wrapping an NSL Model for PyTorch Consumption

```nsl
## Use @torch annotation to make any NSL model usable as a PyTorch nn.Module.
## This is useful for integrating NSL models into existing PyTorch training pipelines,
## evaluation frameworks (lm-eval-harness), or serving infrastructure.

import nsl.compat.{to_torch}

# Define a model in NSL
model NSLTransformer(vocab_size: int, d_model: int, n_layers: int, n_heads: int):
    embed: Embedding = Embedding(vocab_size, d_model)
    layers: list<TransformerBlock> = [
        TransformerBlock(d_model, n_heads, d_model * 4)
        for _ in 0..n_layers
    ]
    norm: RMSNorm = RMSNorm(d_model)
    lm_head: Linear = Linear(d_model, vocab_size, bias=false)
        @tie_weights(self.embed.weight)

    fn forward(input_ids: Tensor<[batch, seq], int32>) -> Tensor<[batch, seq, vocab_size], bf16>:
        let h = self.embed(input_ids)
        for layer in self.layers:
            let (h, _) = layer.forward(h)
        return self.lm_head(self.norm(h))

# Instantiate and wrap for PyTorch
let nsl_model = NSLTransformer(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12
).to(cuda)

# Convert to PyTorch nn.Module
let torch_model = to_torch(nsl_model)

# Now usable in any PyTorch code:
# >>> import torch
# >>> output = torch_model(torch.tensor([[1, 2, 3]]).cuda())
# >>> loss = torch.nn.functional.cross_entropy(output.view(-1, 50257), labels.view(-1))
# >>> loss.backward()  # gradients flow through NSL model

# The wrapped model supports:
# - .parameters() — returns PyTorch Parameter objects
# - .state_dict() / .load_state_dict() — standard PyTorch serialization
# - .train() / .eval() — mode switching
# - .to(device) — device transfer
# - torch.save() / torch.load() — checkpoint saving

# Serve with PyTorch's TorchServe or vLLM
# >>> torch.jit.save(torch.jit.trace(torch_model, example_input), "model.pt")
```

### Example 2: Importing a PyTorch Model into NSL

```nsl
## Import an existing PyTorch nn.Module into NSL for further optimization,
## quantization, or integration into an NSL training pipeline.

import nsl.compat.{from_torch}

# Option A: Import from a Python module
let pytorch_model = from_torch(
    py.call("torchvision.models.resnet50", pretrained=true)
)

# The imported model has inferred NSL types:
# pytorch_model.conv1: Conv2d(3, 64, kernel=7, stride=2, padding=3)
# pytorch_model.bn1: BatchNorm(64)
# pytorch_model.fc: Linear(2048, 1000)
print(nsl.bench.summary(pytorch_model, input_shape=[1, 3, 224, 224]))

# Option B: Import from a checkpoint file
let gpt2 = from_torch(
    checkpoint="models/gpt2-pytorch.pt",
    # Provide the model architecture for type checking
    architecture=TransformerDecoder(
        vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072
    )
)

# Now use NSL features on the imported model:

# 1. Quantize with NSL's quantization DSL
let quantized = quant(scheme=int4, mode=static, granularity=per_group(size=128)):
    model = gpt2
    calibration_data = calib_data

# 2. Benchmark with NSL's profiler
let perf = nsl.bench.benchmark(gpt2, input_shape=[1, 2048])
print(f"throughput: {perf.tokens_per_second:.0f} tok/s")

# 3. Export to ONNX
nsl.export.to_onnx(quantized, input_shape=[1, 2048], output="gpt2_int4.onnx")
```

### Example 3: Loading HuggingFace Models and Tokenizers

```nsl
## Load any model from HuggingFace Hub directly into NSL.
## Supports all common architectures: LLaMA, Mistral, GPT-NeoX, Falcon, etc.

import nsl.compat.{from_hf, tokenizer_from_hf}

# Load model and tokenizer from HuggingFace
let model = from_hf("meta-llama/Llama-2-7b-hf")
let tokenizer = tokenizer_from_hf("meta-llama/Llama-2-7b-hf")

# The model is now a native NSL model with full type information:
# model: TransformerDecoder<vocab=32000, d=4096, layers=32, heads=32>
print(f"parameters: {nsl.bench.count_params(model):,}")   # 6,738,415,616
print(f"model dtype: {model.dtype}")                       # bf16

# Use NSL's generation
let prompt = "The future of programming languages is"
let input_ids = tokenizer.encode(prompt)

@no_grad
let output = model.generate(input_ids, max_tokens=100, temperature=0.7)
print(tokenizer.decode(output[0]))

# Fine-tune with NSL's training DSL
train(model=model, epochs=1, precision=bf16):
    data:
        source = my_finetune_data
        batch_size = 4
    optimizer: AdamW(lr=2e-5)
    step(batch):
        let logits = model.forward(batch.input_ids)
        loss = cross_entropy(logits, batch.labels)

# Save as NSL checkpoint (no pickle — safe, fast)
model.save("llama2-finetuned.nslm")

# Or save as safetensors for HuggingFace compatibility
nsl.compat.save_safetensors(model.state_dict(), "model.safetensors")
```

### Example 4: Hybrid NSL + Python Pipeline

```nsl
## Use Python libraries for tasks NSL doesn't cover yet (e.g., web scraping,
## plotting, specialized data processing) while keeping the ML pipeline in NSL.

import nsl.compat.{py}

# Call Python's matplotlib for visualization
fn plot_training_curve(losses: list<f32>, output: str):
    py.call("matplotlib.pyplot.figure", figsize=(10, 6))
    py.call("matplotlib.pyplot.plot", losses)
    py.call("matplotlib.pyplot.xlabel", "Step")
    py.call("matplotlib.pyplot.ylabel", "Loss")
    py.call("matplotlib.pyplot.title", "Training Loss")
    py.call("matplotlib.pyplot.savefig", output)
    py.call("matplotlib.pyplot.close")

# Use Python's requests for downloading data
fn download_dataset(url: str, output: str):
    let response = py.call("requests.get", url, stream=true)
    py.call("builtins.open", output, "wb") |> |f| {
        for chunk in py.call("iter", response.iter_content, chunk_size=8192):
            f.write(chunk)
    }

# Use wandb for experiment tracking
fn init_wandb(config: dict):
    py.call("wandb.init", project="nsl-training", config=config)

fn log_wandb(metrics: dict<str, f32>, step: int):
    py.call("wandb.log", metrics, step=step)

# Use HuggingFace evaluate for benchmarking
fn evaluate_model(model: Model, dataset_name: str) -> dict:
    let evaluator = py.call("evaluate.load", dataset_name)
    # Convert NSL model to PyTorch for the evaluator
    let torch_model = nsl.compat.to_torch(model)
    return py.call("evaluator.compute",
                   model_or_pipeline=torch_model,
                   data=dataset_name)

# Practical example: training with wandb logging and matplotlib visualization
let losses = []
train(model=my_model, epochs=10, precision=bf16):
    data:
        source = train_data
        batch_size = 32
    optimizer: AdamW(lr=1e-4)
    step(batch):
        let logits = model.forward(batch.input_ids)
        loss = cross_entropy(logits, batch.labels)
    callbacks:
        on_step(step, loss):
            losses.append(loss.item())
            if step % 100 == 0:
                log_wandb({"loss": loss.item()}, step)

# Plot the training curve after training
plot_training_curve(losses, "training_curve.png")
```

## Design Tensions & Tradeoffs

1. **Python FFI overhead**: Calling Python functions from NSL incurs Python interpreter
   overhead. This is acceptable for non-performance-critical tasks (logging, plotting,
   data downloading) but should not be used in hot loops. The compiler warns if `py.call`
   is used inside a training step.

2. **Type safety across boundaries**: PyTorch tensors are dynamically typed; NSL tensors
   are statically typed. At the boundary, NSL inserts runtime shape/dtype checks. These
   checks can be disabled with `@unchecked` for performance-critical paths.

3. **Gradient flow across boundaries**: Supporting backpropagation through NSL→PyTorch→NSL
   requires maintaining compatible autograd graphs. NSL uses DLPack and custom autograd
   functions to bridge the two gradient systems. This adds complexity but is essential for
   gradual migration.

4. **Model architecture inference**: When importing a PyTorch model with `from_torch()`,
   NSL infers the architecture by tracing. Some PyTorch patterns (dynamic control flow,
   custom autograd functions) may not trace correctly. Users can provide an explicit
   architecture declaration as a fallback.
