# NeuralScript (NSL)

A statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads.

NSL compiles to native code via Cranelift with zero Python or C++ dependencies. The entire stack is Rust (compiler + runtime) and NSL (standard library).

## Why NSL?

- **Python-familiar syntax** with indentation-based blocks, `let`/`const`, `fn`, `model`
- **Compile-time tensor shape checking** — catch dimension mismatches before running
- **Native autodiff** — `grad` is a keyword, not a library call
- **Declarative training** — `train` blocks replace boilerplate training loops
- **GPU/CUDA native** — `kernel` keyword for custom GPU ops, `.to(cuda)` for device transfer
- **No GIL, no runtime** — just `nsl run model.nsl`

## Installation

### From GitHub Releases

Download the latest release from [GitHub Releases](https://github.com/bwiemz/NSL/releases).

```bash
# Linux/macOS
tar xzf nsl-v0.9.0-<target>.tar.gz
export PATH="$PWD/nsl-v0.9.0-<target>/bin:$PATH"
```

> **Important:** Keep `nsl` binary alongside the `lib/` directory — the compiler needs `lib/libnsl_runtime.a` and `lib/stdlib/`.

### Prerequisites

- **Linux/macOS:** C compiler (`gcc` or `clang`) — usually pre-installed
- **Windows:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (MSVC `link.exe`)
- **GPU (optional):** NVIDIA CUDA Toolkit

### From Source

```bash
git clone https://github.com/bwiemz/NSL.git
cd NSL
cargo build --release -p nsl-cli
```

## Quick Start

```bash
nsl init myproject
cd myproject
nsl run main.nsl
```

## Tutorial: Build a Transformer from Scratch

### 1. Define a Model

```python
from nsl.nn.norms import RMSNorm
from nsl.nn.losses import cross_entropy

model MLP:
    w1: Tensor = randn([512, 1408]) * full([1], 0.02)
    w2: Tensor = randn([1408, 512]) * full([1], 0.02)
    norm: RMSNorm = RMSNorm(512)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.norm.forward(x)
        let gate = silu(h @ self.w1)
        return gate @ self.w2
```

### 2. Train It

```python
from nsl.nn.losses import mse_loss

let m = MLP()
let x = randn([32, 512])
let y = randn([32, 512])

train(model=m, epochs=100):
    optimizer: AdamW(lr=0.001, weight_decay=0.01)
    scheduler: cosine_anneal(min_lr=0.0001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            if step % 10 == 0:
                print(loss)
```

### 3. Save and Load

```python
model_save(m, "checkpoint.nslm")
# ... later ...
model_load(m, "checkpoint.nslm")
```

### 4. Run on GPU

```python
let m = MLP()
let x = randn([32, 512]).to(cuda)
let pred = m.forward(x)
```

### 5. Custom GPU Kernels

```python
kernel vec_add(a, b, c):
    let i = thread_id()
    c[i] = a[i] + b[i]

let a = full([1024], 1.0).to(cuda)
let b = full([1024], 2.0).to(cuda)
let c = zeros([1024]).to(cuda)
vec_add(a, b, c, grid=4, block=256)
```

### 6. Pretrain on Real Data

```python
from model import NSLCoder
from nsl.nn.losses import cross_entropy

let m = NSLCoder()
let tokens = load_mmap("data/tokens.bin", 3)
let loader = DataLoader(tokens, batch_size=32, seq_len=1024, shuffle=true)

for batch in loader:
    let logits = m.forward_train(batch.input_ids, true)
    let loss = cross_entropy(logits, batch.labels)
    print(loss)
```

## CLI Reference

```bash
nsl run file.nsl                    # Run a program
nsl build file.nsl                  # Compile to native executable
nsl check file.nsl                  # Type-check without running
nsl fmt file.nsl                    # Format code
nsl test tests/*.nsl                # Run test suite

# Compile-time analysis
nsl check --perf file.nsl           # Roofline performance analysis
nsl check --nan-analysis file.nsl   # NaN/Inf risk detection
nsl check --deterministic file.nsl  # Determinism verification

# GPU build
nsl build file.nsl --features cuda  # Enable GPU support
nsl run file.nsl --target cuda      # Run with CUDA backend
```

## Project Structure

```text
crates/
  nsl-lexer/       Tokenizer (indentation-aware)
  nsl-ast/         Abstract syntax tree definitions
  nsl-parser/      Recursive descent parser
  nsl-semantic/    Type checking, shape inference, name resolution
  nsl-codegen/     Cranelift IR generation and native compilation
  nsl-runtime/     Rust static library linked into every NSL binary
  nsl-cli/         Command-line interface

stdlib/nsl/        Standard library (written in NSL)
  nn/              Neural network layers, norms, losses, attention
  optim/           Optimizers and learning rate schedulers
  tokenize/        Tokenizer wrappers

spec/              Language specification (13 chapters)
examples/          Example programs and integration tests
models/            Reference model implementations
benchmarks/        Performance benchmarks
docs/              Design documents, plans, and summaries
```

## Benchmarks

All benchmarks run on CPU (AMD 9800X3D, 64GB RAM). GPU benchmarks require `--features cuda`.

### Operator Fusion (M31)

Chains of elementwise ops are fused into a single loop — zero intermediate tensor allocations.

| Chain | Fused | Unfused | Speedup |
|-------|-------|---------|---------|
| `sigmoid(relu(a + b))` on [1000,1000] | 3.45 ms, 1 alloc, 4 MB | 8.67 ms, 2 allocs, 8 MB | **2.5x** |
| `sigmoid(tanh(relu(a + b)))` on [256,512] | 32.3 ms, 20 allocs | 52.7 ms, 40 allocs | **1.6x** |

Fusion is automatic and preserves numerical correctness. Disable with `--disable-fusion` for differential testing.

### DataLoader Throughput (M19)

| Config | Throughput |
|--------|-----------|
| batch=1, seq=1024 | 270K batches/sec, **277M tokens/sec** |
| batch=32, seq=1024 | 9.3K batches/sec, **306M tokens/sec** |

The DataLoader reads pre-tokenized u16 data via zero-copy mmap. Causal attention masks are generated inside the model's GQA layer, not in the DataLoader.

### Roofline Cost Model (M37)

Per-op analysis of the NSLCoder-50M forward pass (H100-SXM target):

| Operation | GFLOPs | AI (FLOP/byte) | Bound |
|-----------|--------|-----------------|-------|
| Q/K/V projections | 0.27-0.54 | 73-102 | Compute |
| Attention QK^T | 1.07 | 28.4 | Compute |
| **Softmax** | 0.04 | **0.625** | **Memory** |
| FFN matmuls | 1.48 | 137.4 | Compute |
| **LM head** | **51.5** | 169.5 | **Compute** |

Full forward pass: **117.5 GFLOPs**, overall AI = 63.9 FLOP/byte. Memory-bound at batch=1 (softmax and elementwise ops dominate); compute-bound at batch=32.

### Training Correctness

```
$ nsl run examples/m14_sgd_basic.nsl
4.0                    # epoch 1: mse_loss = 4.0 (correct for ones([2,1]) weights)
3.6863999366760254     # epoch 2: gradient descent working
3.397386074066162      # epoch 3: loss decreasing
3.1310312747955322     # epoch 4
2.8855583667755127     # epoch 5
```

All 1,420 tests pass across 7 crates.

### Recommended Training Config (RTX 5070 Ti, 16GB VRAM)

```python
# models/coder50m/config.nsl
const PRETRAIN_BATCH_SIZE = 32     # ~83% VRAM utilization, compute-bound
const PRETRAIN_LR = 0.0003
const PRETRAIN_WARMUP = 3000
const PRETRAIN_GRAD_CLIP = 1.0
```

## Testing

```bash
cargo test --workspace              # 1,420+ unit and integration tests
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl  # Training demo
cargo run -p nsl-cli -- test tests/m15_test.nsl          # NSL test suite

# Run benchmarks
cargo run -p nsl-cli --release -- run benchmarks/bench_fusion_metrics.nsl
cargo run -p nsl-cli --release -- run benchmarks/bench_roofline.nsl
```

## Documentation

- **[SPECIFICATION.md](SPECIFICATION.md)** — Full feature reference, architecture, per-op roofline analysis
- **[spec/](spec/)** — Formal language specification (13 chapters)
- **[docs/summaries/](docs/summaries/)** — Condensed technical summaries for each subsystem
- **[docs/plans/](docs/plans/)** — Roadmap and future milestone designs

## Contributing

NSL is structured as a standard Rust workspace. To get started:

```bash
git clone https://github.com/bwiemz/NSL.git
cd NSL
cargo build                         # Build all crates
cargo test --workspace              # Verify everything passes
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl  # Run a training example
```

Key entry points for contributors:
- **Add a tensor op:** [crates/nsl-runtime/src/tensor/](crates/nsl-runtime/src/tensor/) — implement the op, add to [builtins.rs](crates/nsl-codegen/src/builtins.rs)
- **Add an AD rule:** [crates/nsl-runtime/src/autodiff/backward.rs](crates/nsl-runtime/src/autodiff/backward.rs) — add a `TapeOp` variant and backward logic
- **Add a stdlib layer:** [stdlib/nsl/nn/](stdlib/nsl/nn/) — pure NSL, no Rust needed
- **Add a language feature:** Parser → AST → Semantic → Codegen pipeline across the 4 crates

## Known Limitations

- No REPL
- CUDA required for GPU features (ROCm/Metal/WebGPU KIR built, untested on real hardware)
- Windows requires Visual Studio Build Tools for linking
- Fusion fires on elementwise chains; matmul+epilogue fusion (fused bias+relu inside matmul) is analysis-only

## License

Apache 2.0
