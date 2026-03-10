# NeuralScript Development Roadmap: M9-M20

**Date:** 2026-03-09
**Status:** Approved
**Architecture:** Rust compiler + Rust runtime + NSL standard library (no C/C++/Python)

## Current State (M1-M8 Complete)

NSL is a general-purpose compiled language with:
- 7-crate Rust compiler: nsl-errors, nsl-lexer, nsl-ast, nsl-parser, nsl-semantic, nsl-codegen, nsl-cli
- Cranelift 0.116 backend generating native executables
- C runtime (~770 LOC) for dynamic types (lists, dicts, strings, HOFs, file I/O, math)
- CLI with `check` and `build` commands
- Features: variables, functions, control flow, structs, enums, match, lists, dicts, tuples,
  lambdas, closures, string methods, HOFs, slicing, destructuring, nested functions, modules/imports

**What's missing:** The entire ML domain (tensors, models, autodiff, training, quantization,
tokenization, data pipelines, hardware abstraction, interop) and most developer tooling
(run, REPL, LSP, formatter, profiler, package manager).

---

## Architecture Principle

**No C, no C++, no Python in the final stack.**

| Layer | Language | Role |
|-------|----------|------|
| Compiler | Rust | Lexer, parser, semantic, Cranelift codegen |
| Runtime primitives | Rust | Tensor alloc, matmul, autodiff tape, BLAS, CUDA dispatch |
| Standard library | NSL | nsl.nn, nsl.optim, nsl.quant, nsl.data, nsl.tokenize, etc. |
| User code | NSL | Models, training scripts, everything in the spec |

The Rust runtime exposes low-level primitives via `#[no_mangle] extern "C"` functions that
Cranelift codegen calls directly. The NSL standard library builds higher-level abstractions
(layers, optimizers, losses) on top of these primitives, written in NSL itself.

The existing C runtime (`nsl_runtime.c`, ~770 LOC) is migrated to Rust in M9.

---

## Milestone Overview

### M9: Rust Runtime + Tensor Foundation
**ML:** Tensor runtime in Rust (f64 CPU), basic ops (zeros, ones, rand, add, sub, mul, matmul, reshape, transpose, print)
**Tooling:** `nsl run` command, C-to-Rust runtime migration
**Deliverable:** `let x = zeros([3,4]); print(x @ ones([4,2]))` compiles and runs

### M10: Tensor Type System
**ML:** Compile-time shape checking, dtype system (fp32/f64/bf16/fp16), named dimensions, symbolic dims, wildcards, broadcasting rules, shape algebra
**Tooling:** Shape error diagnostics with visual annotations
**Deliverable:** Compiler catches `[32,64] @ [128,64]` mismatch at compile time

### M11: Model System
**ML:** `model` keyword, Param/Buffer types, self references, forward dispatch, layer types (Linear, Embedding, LayerNorm, RMSNorm, Dropout) written in NSL
**Tooling:** `nsl fmt` formatter
**Deliverable:** Define a transformer block, instantiate it, run forward pass

### M12: Autodiff
**ML:** `grad` keyword/blocks, tape-based AD in Rust runtime, @no_grad, @checkpoint, gradient clipping, @backward custom gradients
**Tooling:** Basic REPL (JIT eval loop)
**Deliverable:** `let loss, grads = grad(params): ...` computes real gradients

### M13: Training DSL
**ML:** `train` block parsing/codegen, optimizers (SGD, Adam, AdamW) in NSL, schedulers (Cosine, WarmupCosine), loss functions in NSL
**Tooling:** `.nslm` checkpoint format (safe binary, no serialization vulnerabilities)
**Deliverable:** Train a model end-to-end from scratch

### M14: Data Pipeline
**ML:** `dataset` keyword, DataLoader with Rust multi-threaded prefetch, MemoryMapped/JSONL/CSV sources, sequence packing
**Tooling:** LSP server (diagnostics, hover types, go-to-def)
**Deliverable:** Load data, train, save -- with IDE support

### M15: Tokenization + Standard Library
**ML:** `tokenizer` keyword, BPE algorithm in Rust runtime, encode/decode, batch encoding; nsl.nn module in NSL (activations, norms, losses)
**Tooling:** `nsl test` command
**Deliverable:** Tokenize text -> feed to model -> train a language model

### M16: Quantization Foundations
**ML:** `quant static` block, `QuantizedTensor` type (packed INT4/INT8), weight-only RTN quantization, per-tensor/per-channel/per-group granularity, mixed-precision matmul, model monomorphization (compiler synthesizes quantized model type)
**Tooling:** `nsl bench` command
**Deliverable:** Quantize a trained model's weights to INT4/INT8, run inference with mixed-precision matmul

### M17: GPU + Kernels
**ML:** Device type system (CPU/CUDA), Rust CUDA runtime bindings, `kernel` keyword, @fuse, @autotune, GPU tensor ops
**Tooling:** `nsl profile` command
**Deliverable:** Train on GPU with fused kernels

### M18: Interop + Export
**ML:** py.call FFI (embed Python interpreter from Rust), from_torch/to_torch via DLPack, HuggingFace loading, safetensors, ONNX export
**Tooling:** `nsl export` command
**Deliverable:** Load HF model, quantize, export ONNX

### M19: Package Ecosystem
**ML:** nsl.dist basics (DDP), distributed training
**Tooling:** `nsl pkg` (registry client), `nsl doc` (doc generation)
**Deliverable:** Multi-GPU training, publish/install packages

### M20: v0.1 Release
**ML:** Full GPT-2 pipeline from spec 13, all in NSL
**Tooling:** Polish all tooling, error messages, examples, CI/CD
**Deliverable:** The spec 13 example runs end-to-end -- ship v0.1

### M21: Advanced Quantization
**ML:** FP8 dtype support, activation quantization (W8A8) with calibration data, mixed-precision layer assignment (`quant.mixed_precision` block), sensitivity analysis (measure per-layer accuracy impact)
**Tooling:** Quantization analysis reporting
**Deliverable:** Quantize a model with W8A8 activation quantization, auto-assign mixed precision based on sensitivity

### M22: Algorithmic Quantization
**ML:** QAT (quantization-aware training with fake quantize in forward pass), GPTQ (second-order weight quantization), AWQ (activation-aware weight quantization), SmoothQuant (activation difficulty migration), hardware-targeted quantization profiles
**Tooling:** Hardware profile configs
**Deliverable:** GPTQ-quantize a large model, deploy with hardware-specific profiles

---

## Rust Runtime vs NSL Standard Library Boundary

| Rust Runtime (primitives) | NSL Standard Library |
|---------------------------|---------------------|
| Tensor allocation, refcounting, memory layout | nsl.nn.Linear, nsl.nn.Attention, nsl.nn.MLP |
| Raw matmul, elementwise ops, BLAS dispatch | nsl.nn.cross_entropy, nsl.nn.softmax, nsl.nn.relu |
| Autodiff tape, backward engine | nsl.optim.Adam, nsl.optim.AdamW, nsl.optim.SGD |
| CUDA kernel launch, device management | nsl.optim.WarmupCosine, nsl.optim.OneCycle |
| BPE merge algorithm, vocab lookup | nsl.quant.quantize, nsl.quant.analyze_sensitivity |
| File I/O, mmap, threading primitives | nsl.data.DataLoader, nsl.data.SequencePacker |
| Quantized matmul kernels | nsl.export.to_onnx, nsl.bench.benchmark |

**Guiding principle:** If it touches raw memory, hardware, or needs to be fast at the byte level,
it's a Rust runtime primitive. If it's composing those primitives into ML abstractions, it's NSL.

---

## Dependency Graph

```
M9 (Runtime + Tensors)
 |-> M10 (Type System)
      |-> M11 (Models)
           |-> M12 (Autodiff)
                |-> M13 (Training)
                     |-> M14 (Data + LSP)
                     |-> M15 (Tokenization + Stdlib)
                     |-> M16 (Quantization Foundations)
                          |-> M17 (GPU)
                          |    |-> M18 (Interop + Export)
                          |         |-> M19 (Packages)
                          |              |-> M20 (v0.1 Release)
                          |-> M21 (Advanced Quantization)
                               |-> M22 (Algorithmic Quantization)
```

M14, M15, M16 can be developed in parallel after M13 is complete.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Cranelift can't express tensor ops efficiently | Tensor ops are Rust runtime calls, not Cranelift IR -- Cranelift just emits `call` instructions |
| C runtime migration breaks existing programs | Run all existing examples as regression tests before/after migration |
| NSL standard library needs language features not yet implemented | Each milestone adds the language features needed by the stdlib code it introduces |
| CUDA integration is complex | M17 is intentionally late; CPU-only is viable for all earlier milestones |
| Autodiff is hard to implement correctly | Start with a simple tape-based approach; complex optimizations come in later milestones |
