# NeuralScript (NSL)

A statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads.

NSL compiles to native code via Cranelift with zero Python or C++ dependencies. The entire stack is Rust (compiler + runtime) and NSL (standard library).

## Why NSL?

- **Python-familiar syntax** with indentation-based blocks, `let`/`const`, `fn`, `model`
- **Compile-time tensor shape checking** — catch dimension mismatches before running
- **Native autodiff** — `grad` is a keyword, not a library call
- **Declarative training** — `train` blocks replace boilerplate training loops
- **GPU/CUDA native** — `kernel` keyword for custom GPU ops, `.to(cuda)` for device transfer
- **No GIL** — just `nsl run model.nsl`

## Quick Example

```python
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model=m, epochs=5):
    optimizer: SGD(lr=0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
```

Output:
```
4.0
3.6864
3.39738624
3.131031158784
2.8855583159353344
```

### GPU Kernel Example

```python
kernel vec_add(a, b, c):
    let i = thread_id()
    c[i] = a[i] + b[i]

let a = full([1024], 1.0).to(cuda)
let b = full([1024], 2.0).to(cuda)
let c = zeros([1024]).to(cuda)
vec_add(a, b, c, grid=4, block=256)
let result = c.to(cpu)  # each element = 3.0
```

## Features

### Core Language
- Indentation-based syntax (no braces)
- `let` (mutable), `const` (immutable) bindings
- `int`, `float`, `bool`, `str`, `list`, `dict` types
- `fn` functions with type annotations
- `for`/`while` loops, `if`/`elif`/`else`
- Pattern matching, tuple destructuring
- Module system with `from ... import ...`

### Tensors
- `Tensor` type with shape-aware operations
- Creation: `zeros`, `ones`, `rand`, `full`, `arange`
- Arithmetic: `+`, `-`, `*`, `/`, `@` (matmul)
- Reductions: `sum`, `mean` (global or with `dim`/`keepdim`)
- Element-wise: `exp`, `log`, `sqrt`, `abs`, `sign`, `clamp`
- Shape ops: `reshape`, `transpose`, `clone`

### Models
- `model` keyword for defining neural network architectures
- Named parameters as typed fields
- Methods with `self` access
- Constructor with default values

### Automatic Differentiation
- `grad(params):` block records a computation tape
- Tape-based reverse-mode autodiff
- `@no_grad` decorator to exclude functions from tape
- Supports: add, sub, mul, div, matmul, exp, log, sqrt, abs, clamp, sum, mean, reduce_max, gather
- Broadcasting: NumPy-style broadcast for elementwise ops

### Neural Network Layers
- **Activation functions**: relu, gelu, silu, sigmoid, tanh, softmax
- **Layers**: Linear, Embedding, MLP, Attention, Conv2d, MaxPool2d
- **Normalization**: LayerNorm, RMSNorm
- **Regularization**: Dropout (training-mode aware)
- **Weight init**: randn for Kaiming-style initialization

### Training DSL
- `train(model=m, epochs=N):` declarative training blocks
- **6 optimizers**: SGD, Adam, AdamW, Lion, Muon, SOAP
- **7 schedulers**: constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
- **4 loss functions**: mse_loss, l1_loss, cross_entropy, bce_loss
- Implicit gradient computation (no manual `grad` blocks needed)
- Callbacks: `on_step(step, loss)`, `on_epoch(epoch, loss)`
- `.nslm` checkpoint format for model serialization

### Tokenization
- Byte-level tokenizer (no training needed)
- BPE tokenizer via HuggingFace `tokenizers` crate
- Encode text → token ID tensors, decode back to text
- Batch encoding with padding and attention masks

### GPU/CUDA
- `kernel` keyword for user-defined GPU kernels (compiled to PTX)
- `.to(cuda)` / `.to(cpu)` for transparent device transfer
- GPU intrinsics: `thread_id()`, `block_id()`, `block_dim()`, `sync_threads()`
- Kernel launch with explicit grid/block: `my_kernel(a, b, c, grid=4, block=256)`
- 15 built-in GPU PTX kernels for tensor ops (add, mul, matmul, activations, etc.)
- CUDA Unified Memory for zero-copy host/device access
- Automatic f64↔f32 dtype conversion on device transfer
- Optional: build with `--features cuda` to enable GPU support

### Quantization
- `quant static` block for declarative weight quantization
- `QuantizedTensor` type with packed INT4 (2 values/byte) and INT8 storage
- Per-tensor, per-channel, and per-group granularity
- Asymmetric affine quantization (weight-only RTN)
- Mixed-precision matmul (`nsl_qtensor_matmul_mixed`)
- Model monomorphization: compiler synthesizes quantized model type
- Glob-based `exclude` patterns to keep specific layers in full precision

### Test Framework
- `@test` decorator marks functions as test cases
- `assert_eq`, `assert_close` for value and tensor assertions
- `nsl test` CLI with process isolation (each test in separate process)
- Filter tests with `--filter`

## Building

Requires Rust (stable).

```bash
cargo build -p nsl-cli
```

## Usage

```bash
# Run an NSL program
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl

# Run with verbose output
cargo run -p nsl-cli -- run --dump-ir examples/hello.nsl
```

## Project Structure

```
crates/
  nsl-lexer/       Tokenizer (indentation-aware)
  nsl-ast/         Abstract syntax tree definitions
  nsl-parser/      Recursive descent parser
  nsl-semantic/    Type checking, shape inference, name resolution
  nsl-codegen/     Cranelift IR generation and native compilation
  nsl-runtime/     Rust static library linked into every NSL binary
  nsl-cli/         Command-line interface (nsl run, nsl build)
  nsl-errors/      Shared error types

stdlib/nsl/        Standard library (written in NSL)
  math.nsl         Basic math utilities
  nn/layers.nsl    Linear, Embedding, MLP
  nn/norms.nsl     LayerNorm, RMSNorm
  nn/activations.nsl  Activation function wrappers
  nn/attention.nsl Dot-product attention
  nn/dropout.nsl   Dropout layer
  nn/losses.nsl    Loss functions (mse, l1, cross_entropy, bce)
  optim/           Optimizers (sgd, adam, adamw, lion, muon, soap)
  optim/schedulers.nsl  Learning rate schedulers
  tokenize/        Tokenizer wrappers
  quant/ops.nsl    Quantization wrappers

spec/              Language specification (13 chapters)
examples/          Example programs and integration tests
tests/expected/    Expected output for integration tests
docs/plans/        Design documents and implementation plans
```

## Testing

```bash
# Run unit tests
cargo test --workspace

# Run a specific integration test
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl

# Run NSL test suite
cargo run -p nsl-cli -- test tests/m15_test.nsl

# Run end-to-end language model demo
cargo run -p nsl-cli -- run examples/m15_tiny_lm.nsl

# Run quantization example
cargo run -p nsl-cli -- run examples/m16_quantize.nsl

# Run GPU kernel test (requires CUDA)
cargo run -p nsl-cli --features cuda -- run tests/m17_kernel_test.nsl

# Run GPU compute test (requires CUDA)
cargo run -p nsl-cli --features cuda -- run tests/m17_gpu_training_test.nsl
```

## Development Status

NSL is in active development. Current milestone progress:

| Milestone | Description | Status |
|-----------|-------------|--------|
| M9 | Rust runtime + tensor foundation | Complete |
| M10 | Compile-time tensor shape checking | Complete |
| M11 | `model` keyword + codegen | Complete |
| M12 | `grad` keyword + tape-based autodiff | Complete |
| M13 | Import system + multi-file compilation | Complete |
| M14 | Training DSL + optimizers + schedulers | Complete |
| M15 | NN stdlib + tokenization + test framework | Complete |
| M16 | Quantization foundations (`quant static` block) | Complete |
| M17 | GPU/CUDA + `kernel` keyword | Complete |
| M18a | Transformer foundations (tensor ops + model composition) | Complete |
| M18b | Interop (safetensors, HuggingFace loading, ONNX export) | Complete |
| M19 | Data pipeline + inference sampling | Planned |
| M20 | v0.1 release (full GPT-2 pipeline) | Planned |
| M21 | Advanced quantization (FP8, activation quant) | Planned |
| M22 | Algorithmic quantization (QAT, GPTQ, AWQ) | Planned |

## License

APACHE 2.0
