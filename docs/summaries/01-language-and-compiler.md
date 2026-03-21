# NeuralScript: Language & Compiler Architecture

## Overview

NeuralScript (NSL) is a statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads. It compiles to native code via Cranelift with zero Python, C, or C++ dependencies at runtime.

**Key principles:**
- Python-familiar indentation-based syntax
- Compile-time tensor shape verification with named dimensions
- Native autodiff, quantization, and training as language features (not libraries)
- Hardware-aware compilation with GPU kernel generation
- Pure Rust compiler + NSL standard library

---

## Language Syntax

### Variables and Types
```
let x = 42              # mutable binding (type inferred)
const PI = 3.14159      # immutable binding
let name: str = "hello" # explicit type annotation
```

Primitive types: `int`, `float`, `bool`, `str`. Collections: `list[T]`, `dict[K, V]`, tuples.

### Functions
```
fn add(a: int, b: int) -> int:
    return a + b

fn greet(name: str = "world") -> str:
    return f"Hello, {name}!"
```

Functions support default arguments, keyword arguments, and type annotations on all parameters and return types.

### Control Flow
```
if condition:
    ...
elif other:
    ...
else:
    ...

for i in range(10):
    ...

while condition:
    ...
```

Also supports: `match` expressions (pattern matching), list comprehensions, tuple unpacking, early `return`/`break`/`continue`.

### Pipe Operator
```
let result = data |> normalize |> model.forward |> softmax
```
The `|>` operator chains operations left-to-right, passing the result of each as the first argument to the next.

---

## Tensor Type System

### Tensor Declaration
```
let x: Tensor<[3, 4], f32, cuda>        # concrete shape
let y: Tensor<[batch, seq_len], f16>     # symbolic dimensions
let z: Tensor<[batch="B", heads="H", seq="S", dim=64], fp8>  # named dimensions
```

Tensor types are parameterized by:
- **Shape**: concrete integers, symbolic names, or named dimensions
- **Dtype**: `f64`, `f32`, `f16`, `bf16`, `fp8`, `int4`, `int8`, `int32`
- **Device**: `cpu`, `cuda` (optional, defaults to cpu)

### Named Dimensions
Named dimensions eliminate positional confusion:
```
let q: Tensor<[batch="B", heads="H", seq="S", dim=64], f16>
let k: Tensor<[batch="B", heads="H", seq="S", dim=64], f16>
# Compiler verifies dimension compatibility by name, not position
```

### Compile-Time Shape Checking
The compiler statically verifies:
- Matrix multiplication dimension compatibility
- Broadcasting rules (NumPy-style)
- Reshape validity (total element count preserved)
- Slice bounds
- Named dimension consistency across operations

Shape errors are caught at compile time, not runtime.

### Shape Algebra (M49)
A constraint solver for symbolic dimensions:
```
# Compiler can prove: if batch divides total, reshape is valid
let x: Tensor<[total], f32>
let y = x.reshape([batch, total / batch])  # proven valid at compile time
```
Supports equality constraints, divisibility facts, and range bounds on symbolic dimensions.

---

## Model System

### Model Declaration
```
model Transformer:
    embedding: Embedding
    layers: [TransformerBlock; 12]    # fixed-size model array
    norm: LayerNorm
    head: Linear

    fn forward(self, x: Tensor<[B, S], int32>) -> Tensor<[B, S, vocab], f32>:
        let h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)
```

**`model` is a keyword**, not a class. Models have:
- Typed fields (parameters, buffers, sub-models)
- Methods with `self` access
- Fixed-size model arrays (`[Layer; N]`)
- Automatic parameter registration
- Native serialization (`.nslm` binary format, not pickle)

### Parameter Types
- **Param**: Trainable parameter (tracked by autodiff tape)
- **Buffer**: Non-trainable state (e.g., running mean in BatchNorm)

---

## Compiler Architecture

### Crate Structure (8 Rust crates, ~53K lines)

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| **nsl-errors** | Diagnostics and source spans | `Diagnostic`, `Span`, `SourceMap` |
| **nsl-lexer** | Tokenization with indentation tracking | `TokenKind` (250+ variants), `Lexer` |
| **nsl-ast** | Abstract syntax tree definitions | `Module`, `Expr`, `Stmt`, `FnDef`, `ModelDef` |
| **nsl-parser** | Recursive descent + Pratt precedence | `Parser` |
| **nsl-semantic** | Type checking, shape inference, validation | `TypeChecker`, `Type`, `Shape`, `DType` |
| **nsl-codegen** | Cranelift IR generation + GPU kernel compilation | `Compiler`, fusion passes, kernel backends |
| **nsl-runtime** | C ABI static library (tensor ops, autodiff, CUDA) | `NslTensor`, tape, CUDA contexts |
| **nsl-cli** | Command-line interface | `check`, `run`, `build`, `test` commands |

### Compilation Pipeline

```
NSL Source (.nsl files)
    │
    ▼
┌─────────────────────┐
│  nsl-lexer           │  Tokenize + indentation tracking (INDENT/DEDENT tokens)
│  Token stream        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  nsl-parser          │  Recursive descent + Pratt precedence climbing
│  AST (Module)        │  Handles: fn, model, train, kernel, quant, serve blocks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  nsl-semantic        │  Type inference, shape checking, ownership analysis,
│  Typed AST           │  effect tracking, NaN detection, determinism checking,
│                      │  performance budget validation, WCET analysis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  nsl-codegen         │  Cranelift IR generation
│  + optimization      │  GPU kernel compilation (PTX/AMDGPU/Metal/WGSL)
│  passes              │  Operator fusion, memory planning, cost modeling
│                      │  Weight-aware constant folding, standalone export
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cranelift backend   │  Native code generation (x86-64, ARM)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linker              │  Links nsl-runtime static library
│  Final executable    │  Resolves FFI symbols (CUDA, NCCL optional)
└─────────────────────┘
```

### Key Design Decisions

1. **AOT-only compilation**: No JIT, no interpreter. All code compiled ahead-of-time. This enables static analysis features (memory planning, WCET proofs, ZK circuits) that are impossible in dynamic runtimes.

2. **C ABI runtime**: The runtime is a Rust static library exposing C ABI functions. Generated Cranelift IR calls these functions directly. This means tensor operations, autodiff, and CUDA management are all in Rust.

3. **GPU kernels compiled at build time**: Kernel blocks and fused operations generate PTX (or other ISA) during compilation, not at runtime. This enables build-time autotuning and weight-aware optimization.

4. **Multi-file module system**: `import` statements resolve to `.nsl` files. The compiler builds a dependency graph and compiles modules in topological order.

5. **Two-pass function compilation**: First pass declares all function signatures (enabling mutual recursion), second pass compiles bodies.

---

## Standard Library (Written in NSL)

Located in `stdlib/nsl/`:

| Module | Contents |
|--------|----------|
| `nn/layers.nsl` | Linear, Embedding, MLP |
| `nn/activations.nsl` | relu, gelu, silu, sigmoid, tanh, softmax, log_softmax, elu |
| `nn/attention.nsl` | MultiHeadAttention, scaled_dot_product_attention |
| `nn/norms.nsl` | LayerNorm, RMSNorm, GroupNorm |
| `nn/losses.nsl` | mse_loss, l1_loss, cross_entropy, bce_loss |
| `nn/rope.nsl` | Rotary positional embedding |
| `nn/gqa.nsl` | Grouped query attention |
| `nn/transformer.nsl` | TransformerBlock |
| `nn/dropout.nsl` | Dropout (training-mode aware) |
| `optim/adam.nsl` | Adam optimizer |
| `optim/adamw.nsl` | AdamW optimizer |
| `optim/sgd.nsl` | SGD optimizer |
| `optim/lion.nsl` | Lion optimizer |
| `optim/muon.nsl` | Muon optimizer |
| `optim/soap.nsl` | SOAP optimizer |
| `optim/schedulers.nsl` | 7 LR schedulers (constant, step, exponential, linear, cosine, warmup_cosine, one_cycle) |
| `quant/ops.nsl` | INT4/INT8/FP8 quantization operations |
| `inference/generate.nsl` | Text generation loop |
| `inference/sampling.nsl` | top-k, top-p, temperature sampling |
| `data/loader.nsl` | DataLoader with batching and shuffling |
| `math.nsl` | clamp, lerp, sign, min_val, max_val |
| `io.nsl` | File I/O utilities |
| `tokenize/tokenizer.nsl` | BPE tokenizer wrapper |

---

## CLI Commands

```bash
nsl check file.nsl              # Parse + type-check
nsl check --dump-tokens file.nsl  # Show token stream
nsl check --dump-ast file.nsl     # Show AST as JSON
nsl check --dump-types file.nsl   # Show type map
nsl check --perf file.nsl         # Roofline analysis
nsl check --nan-analysis file.nsl # NaN/Inf risk detection
nsl check --deterministic file.nsl # Determinism checking
nsl check --wcet file.nsl         # WCET analysis
nsl check --weight-analysis file.nsl # Weight-aware analysis

nsl run file.nsl                # Compile and execute
nsl run --memory-profile file.nsl # With memory profiling
nsl run --trace file.nsl          # Chrome tracing output

nsl build --standalone file.nsl   # Zero-dependency binary
nsl test file.nsl                 # Run @test functions
```

---

## Runtime Tensor Representation

```rust
#[repr(C)]
pub struct NslTensor {
    pub data: *mut c_void,      // CPU: f64*, GPU: f32*
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub ndim: i64,
    pub len: i64,
    pub refcount: AtomicI64,
    pub device: u8,             // 0=CPU, 1+=CUDA device
    pub dtype: u16,             // 0=f64, 1=f32, 256+=custom
    pub owns_data: u8,
    pub data_owner: i64,        // for view tensors
}
```

Key: CPU uses f64, GPU uses f32. Device transfer (`.to(cuda)`) handles dtype conversion automatically.

---

## Specification Files

The formal language specification lives in `spec/01-*.nsl.md` through `spec/13-*.nsl.md`, covering syntax, tensor types, autodiff, models, training, quantization, tokenization, data pipeline, hardware abstraction, stdlib, interop, tooling, and a complete example.
