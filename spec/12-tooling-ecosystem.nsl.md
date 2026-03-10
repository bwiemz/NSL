# Section 12 — Tooling & Ecosystem

## Design Rationale

A language is only as good as its tooling. NSL ships with a complete developer experience
from day one: a unified CLI, an LSP server for IDE integration, an interactive REPL with
tensor visualization, a built-in profiler, and a package manager. The goal is that a developer
should never need to leave the NSL ecosystem for standard ML workflows.

## CLI Interface: `nsl`

```
nsl — The NeuralScript Language Toolchain

USAGE:
    nsl <COMMAND> [OPTIONS] [ARGS]

COMMANDS:
    run         Compile and execute an NSL program
    build       Compile an NSL program to a binary
    repl        Start the interactive REPL
    check       Type-check without compiling (fast feedback)
    fmt         Format NSL source files
    lint        Run the linter (style + correctness checks)
    test        Run tests in the project
    bench       Benchmark a model or program
    profile     Profile execution (CPU/GPU time, memory)
    export      Export a model (ONNX, CoreML, TFLite, QNF)
    pkg         Package manager (install, publish, search)
    doc         Generate documentation from doc comments
    init        Create a new NSL project
    lsp         Start the LSP server (for IDE integration)

GLOBAL OPTIONS:
    --device <DEVICE>     Target device: cpu, cuda, metal, rocm, npu
    --precision <DTYPE>   Default precision: fp32, fp16, bf16, fp8
    --verbose             Verbose output
    --quiet               Suppress non-error output
    --version             Print version
    --help                Print help
```

### Key CLI Commands

```bash
# Create a new project
$ nsl init my-project --template=transformer
# Creates:
#   my-project/
#   ├── nsl.toml          # project configuration
#   ├── src/
#   │   └── main.nsl      # entry point
#   ├── tests/
#   │   └── test_model.nsl
#   ├── data/
#   └── models/

# Run a program
$ nsl run src/main.nsl --device cuda
# Compiles to native code via LLVM, then executes

# Type-check only (fast — no code generation)
$ nsl check src/main.nsl
# Reports type errors, shape mismatches, device conflicts

# Format code
$ nsl fmt src/ --check    # check without modifying (CI-friendly)
$ nsl fmt src/            # format in place

# Lint
$ nsl lint src/
# Checks for: unused imports, shadowed variables, non-idiomatic patterns,
# potential performance issues (e.g., unnecessary copies, missing @fuse)

# Run tests
$ nsl test                           # run all tests
$ nsl test tests/test_model.nsl      # run specific test file
$ nsl test --filter="attention"      # run tests matching pattern

# Benchmark
$ nsl bench src/model.nsl --input-shape="[1, 2048]" --iters=100
# Output:
#   Throughput: 12,345 tokens/sec
#   Latency (p50): 2.3ms, (p99): 4.1ms
#   Memory peak: 1.2GB
#   FLOPs: 1.23 TFLOP

# Profile
$ nsl profile src/train.nsl --output=profile.html
# Generates interactive flame graph and memory timeline

# Export model
$ nsl export src/model.nsl --format=onnx --output=model.onnx --input-shape="[1, 2048]"
$ nsl export src/model.nsl --format=qnf --target=chimera --output=model.qnf

# Package management
$ nsl pkg init                       # initialize package in current directory
$ nsl pkg install flash-attention    # install a package
$ nsl pkg publish                    # publish to NSL registry
$ nsl pkg search "attention"         # search registry
$ nsl pkg update                     # update all dependencies
```

## Project Configuration: `nsl.toml`

```toml
[project]
name = "my-transformer"
version = "0.1.0"
authors = ["Developer <dev@example.com>"]
description = "A transformer language model in NeuralScript"
license = "MIT"
nsl-version = ">=0.1.0"

[build]
target = "native"           # native | wasm | library
optimization = "release"    # debug | release | size
device = "cuda"             # default target device
precision = "bf16"          # default precision

[dependencies]
flash-attention = "^2.0"
rotary-embeddings = "^1.0"

[dev-dependencies]
nsl-test = "^0.1"
nsl-bench = "^0.1"

[profile.release]
lto = true                  # link-time optimization
codegen-units = 1           # maximize optimization
strip = true                # strip debug symbols

[profile.debug]
debug-assertions = true
overflow-checks = true
opt-level = 0
```

## LSP Server

The NSL Language Server Protocol implementation provides:

```
Features:
├── Diagnostics
│   ├── Type errors (including shape mismatches)
│   ├── Device conflict detection
│   ├── Unused variable/import warnings
│   └── Quantization compatibility warnings
├── Completion
│   ├── Context-aware autocompletion
│   ├── Layer constructor signatures
│   ├── Tensor operation suggestions
│   └── Import path completion
├── Hover Information
│   ├── Tensor shape at each variable
│   ├── Inferred types for unannotated code
│   ├── Parameter count for model blocks
│   └── Documentation from doc comments
├── Navigation
│   ├── Go to definition
│   ├── Find all references
│   ├── Symbol outline (model layers, functions)
│   └── Workspace-wide symbol search
├── Refactoring
│   ├── Rename symbol (across files)
│   ├── Extract function
│   ├── Inline variable
│   └── Convert between model patterns
├── Code Actions
│   ├── Add type annotation
│   ├── Add missing import
│   ├── Fix shape mismatch (suggest reshape/transpose)
│   └── Apply @fuse suggestion
└── Tensor Shape Lens
    ├── Inline shape annotations on every tensor variable
    ├── Shape flow visualization (how shapes transform through model)
    └── Device tracking annotations
```

### VSCode Extension

```json
{
    "name": "nsl-vscode",
    "displayName": "NeuralScript",
    "description": "NeuralScript language support for VS Code",
    "categories": ["Programming Languages"],
    "contributes": {
        "languages": [{
            "id": "nsl",
            "aliases": ["NeuralScript", "nsl"],
            "extensions": [".nsl"],
            "configuration": "./language-configuration.json"
        }],
        "commands": [
            { "command": "nsl.run",     "title": "NSL: Run Current File" },
            { "command": "nsl.check",   "title": "NSL: Type Check" },
            { "command": "nsl.profile", "title": "NSL: Profile" },
            { "command": "nsl.bench",   "title": "NSL: Benchmark Model" },
            { "command": "nsl.export",  "title": "NSL: Export Model" },
            { "command": "nsl.repl",    "title": "NSL: Open REPL" }
        ]
    }
}
```

## Interactive REPL

The NSL REPL is designed for exploratory ML development — think Jupyter notebooks but
integrated into the language runtime.

```
$ nsl repl --device cuda

NeuralScript 0.1.0 | LLVM 18.0 | CUDA 12.4 | A100 80GB
Type :help for help, :quit to exit

nsl> let x = rand([3, 4], dtype=fp32, device=cuda)
x: Tensor<[3, 4], fp32, cuda>
┌──────────────────────────────────┐
│  0.2341  0.8723  0.1456  0.9312 │
│  0.5567  0.3412  0.7890  0.2134 │
│  0.6789  0.1234  0.4567  0.8901 │
└──────────────────────────────────┘

nsl> let y = x @ x.transpose(0, 1)
y: Tensor<[3, 3], fp32, cuda>
┌──────────────────────────────┐
│  1.7234  0.8912  1.1456  │
│  0.8912  1.2345  0.7890  │
│  1.1456  0.7890  1.5678  │
└──────────────────────────────┘

nsl> :shape y
Tensor<[3, 3], fp32, cuda> — 36 bytes, symmetric

nsl> let model = Linear(768, 3072).to(cuda)
model: LinearLayer — 2,362,368 parameters (9.0 MB in fp32)

nsl> :params model
┌─────────┬──────────────┬───────┬────────┐
│ Name    │ Shape        │ Dtype │ Device │
├─────────┼──────────────┼───────┼────────┤
│ weight  │ [3072, 768]  │ fp32  │ cuda   │
│ bias    │ [3072]       │ fp32  │ cuda   │
└─────────┴──────────────┴───────┴────────┘

nsl> :profile model.forward(rand([1, 128, 768], device=cuda))
Forward pass: 0.23ms
Memory: 1.5 MB peak
FLOPs: 0.60 GFLOP (2.6 TFLOP/s utilization)

nsl> :quit
```

### REPL Special Commands

| Command              | Description                                          |
|----------------------|------------------------------------------------------|
| `:help`              | Show help                                            |
| `:quit` / `:q`       | Exit REPL                                            |
| `:shape <expr>`      | Print detailed shape info                            |
| `:type <expr>`       | Print inferred type                                  |
| `:params <model>`    | List model parameters                                |
| `:profile <expr>`    | Profile an expression                                |
| `:memory`            | Show current memory usage                            |
| `:device`            | Show available devices                               |
| `:hist`              | Show history of tensor shapes                        |
| `:save <path>`       | Save REPL session to file                            |
| `:load <path>`       | Load and execute an NSL file                         |
| `:viz <tensor>`      | Visualize a tensor (heatmap for 2D, histogram for 1D)|
| `:graph <model>`     | Display computation graph                            |
| `:timeit <expr>`     | Benchmark an expression                              |

## Package Registry: `pkg.nsl.dev`

```
NSL Package Registry
├── Publishing
│   ├── nsl pkg publish — publishes to registry
│   ├── Semantic versioning enforced
│   ├── README.md and LICENSE required
│   └── Automatic API documentation generation
├── Discovery
│   ├── Web UI at pkg.nsl.dev
│   ├── CLI search: nsl pkg search <query>
│   ├── Categories: models, layers, optimizers, data, utils
│   └── Quality scores based on tests, docs, downloads
├── Security
│   ├── No arbitrary code execution at install time
│   ├── Packages are compiled, not interpreted
│   ├── Dependency auditing: nsl pkg audit
│   └── Signature verification on published packages
└── Example packages
    ├── flash-attention — optimized attention kernels
    ├── lora — low-rank adaptation implementation
    ├── rope — rotary positional embeddings
    ├── wandb-nsl — Weights & Biases integration
    ├── moe — mixture of experts implementations
    └── nsl-eval — evaluation benchmarks (MMLU, HellaSwag, etc.)
```

## Built-in Profiler

The profiler outputs interactive HTML reports with:

```
Profiler Report — model_training.nsl
═══════════════════════════════════════

Timeline View:
────────────────────────────────────
|  data loading  |  forward  |  backward  |  optimizer  |
|    0.5ms       |   2.3ms   |    3.1ms   |    0.4ms    |
────────────────────────────────────

Flame Graph:
────────────────────────────────────
| train_step                                              |
|   | forward                                     |      |
|   |   | embed | transformer_block[0..12] | head |      |
|   |   |       | attn  | ffn  | attn | ffn  |   |      |
|   | backward                                    | opt  |
────────────────────────────────────

Memory Timeline:
────────────────────────────────────
Peak: 4.2 GB
    ▁▂▃▅▇█████▇▅▃▂▁▁▂▃▅▇█████▇▅▃▂▁
    └── forward ──┘└── backward ──┘
────────────────────────────────────

Kernel Breakdown:
┌──────────────────────┬────────┬────────┬────────┐
│ Kernel               │ Time   │ Calls  │ % Total│
├──────────────────────┼────────┼────────┼────────┤
│ fused_attention      │ 1.2ms  │ 12     │ 28.5%  │
│ matmul_bf16          │ 0.8ms  │ 48     │ 19.0%  │
│ rms_norm             │ 0.3ms  │ 24     │  7.1%  │
│ cross_entropy_bwd    │ 0.2ms  │  1     │  4.8%  │
│ adamw_step           │ 0.4ms  │  1     │  9.5%  │
│ other                │ 1.3ms  │ 156    │ 31.1%  │
└──────────────────────┴────────┴────────┴────────┘
```

## Design Tensions & Tradeoffs

1. **Batteries-included vs Minimal core**: NSL ships with a rich CLI, LSP, REPL, profiler,
   and package manager. This increases the initial binary size (~200MB) but means developers
   never need to assemble a toolchain from disparate tools. The tradeoff is justified by the
   target audience (ML engineers who want to focus on models, not tooling).

2. **REPL vs Compiled language**: NSL is compiled (LLVM backend), but the REPL uses JIT
   compilation for interactive use. JIT code is ~20% slower than AOT-compiled code. For
   production training, users should use `nsl run` (full AOT compilation). The REPL is for
   exploration and debugging.

3. **Package registry trust**: Running arbitrary packages is a security risk. NSL mitigates
   this by: (a) packages are compiled code with no install-time scripts, (b) all packages
   are signed, (c) `nsl pkg audit` scans dependencies for known vulnerabilities.
