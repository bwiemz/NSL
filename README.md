# NeuralScript (NSL)

A statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads.

NSL compiles to native code via Cranelift with zero Python or C++ dependencies. The entire stack is Rust (compiler + runtime) and NSL (standard library).

## Why NSL?

- **Python-familiar syntax** with indentation-based blocks, `let`/`const`, `fn`, `model`
- **Compile-time tensor shape checking** — catch dimension mismatches before running
- **Native autodiff** — `grad` is a keyword, not a library call
- **Declarative training** — `train` blocks replace boilerplate training loops
- **No GIL, no CUDA toolkit required** — just `nsl run model.nsl`

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

### Training DSL
- `train(model=m, epochs=N):` declarative training blocks
- **6 optimizers**: SGD, Adam, AdamW, Lion, Muon, SOAP
- **7 schedulers**: constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
- **4 loss functions**: mse_loss, l1_loss, cross_entropy, bce_loss
- Implicit gradient computation (no manual `grad` blocks needed)
- Callbacks: `on_step(step, loss)`, `on_epoch(epoch, loss)`
- `.nslm` checkpoint format for model serialization

## Building

Requires Rust (stable). No other dependencies.

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
  nn/losses.nsl    Loss functions (mse, l1, cross_entropy, bce)
  optim/           Optimizers (sgd, adam, adamw, lion, muon, soap)
  optim/schedulers.nsl  Learning rate schedulers

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
| M15 | Data pipeline + tokenization | Planned |
| M16 | Quantization (`quant` keyword) | Planned |
| M17 | GPU/CUDA + `kernel` keyword | Planned |
| M18 | Interop (PyTorch, HuggingFace, ONNX) | Planned |
| M19-20 | Package ecosystem + v0.1 release | Planned |

## License

MIT
