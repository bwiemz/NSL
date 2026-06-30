# nsl-runtime architecture

`nsl-runtime` is the static library linked into every compiled NSL program. It
is the **C ABI boundary**: Cranelift-generated code calls into it through
`extern "C"` functions that take raw pointers. That shapes everything about the
crate's public surface.

## FFI safety contract

Because almost every public function is an `extern "C"` raw-pointer entry point,
marking each one `unsafe` with its own `# Safety` block would add boilerplate to
hundreds of call sites that the compiler emits, not humans. Instead the crate
states one contract that **all** FFI functions uphold (see the crate-level docs
in `src/lib.rs`), and individual modules document deviations:

- **Ownership** — pointers are borrowed for the call only unless documented
  otherwise. Allocating functions return a pointer freed by the matching
  `*_free`.
- **Nullability** — inputs must be non-null and aligned unless documented
  nullable; violating this is UB.
- **Lifetimes** — returned pointers are valid until their free/reset call or
  until the owning arena drops.
- **Alignment & layout** — tensor buffers must match the layout the compiler
  emits. `nsl-codegen` and `nsl-runtime` are two halves of one ABI and must be
  built from the same revision.
- **Errors** — failures are reported via documented sentinel values, never by
  unwinding across the ABI. Panics are treated as aborts.

Boundary fuzz/property tests live in `src/fuzz.rs` (`cfg(test)`).

## Subsystem facades

Modules are declared at the crate root (keeping `nsl_runtime::foo` paths stable)
and re-surfaced through facade namespaces in `lib.rs`. Facade names avoid
colliding with the real `autodiff`/`data`/`serving`/`peft` modules and with the
`core` extern-prelude crate (`unikernel` uses bare `core::arch`).

| Facade | Responsibility | Representative modules |
|--------|----------------|------------------------|
| `builtins` | Language builtins / base runtime | `tensor`, `string`, `list`, `dict`, `math`, `memory`, `io`, `slab` |
| `gpu` | Device backend selection | `gpu_backend` (the `cpu`/`cuda` drivers stay `pub(crate)`) |
| `training` | Autodiff & training support | `autodiff`, `grad_context`, `backward_context`, `checkpoint`, `zero`, `vmap_runtime` |
| `quantization` | Reduced precision | `awq`, `gptq`, `fp8`, `quantize`, `packing`, `fase_bc` |
| `attention` | Attention kernels | `flash_attention`, `pca_rope_runtime`, `pca_tier_b_runtime` |
| `dataio` | Data loading & tokenization | `dataloader`, `data`, `data_source`, `tokenizer`, `sampling`, `grammar`, `calibration_data` |
| `inference` | Serving runtime | `serving`, `paged_kv`, `kv_compress`, `speculative`, `disaggregated`, `elastic` |
| `distributed` | Parallelism | `tensor_parallel`, `context_parallel`, `pipeline`, `moe` |
| `finetune` | PEFT adapters | `peft`, `fused_adapter` |
| `observability` | Profiling / tracing / health | `profiler`, `profiling`, `kernel_profiler`, `tensor_trace`, `trace_diff`, `health`, `inspect`, `deterministic_ops` |
| `ffi` | Always-on interop | `c_api`, `dlpack`, `weight_provider` |
| `interop` (`feature = "interop"`) | Optional framework bridges | `safetensors_io`, `huggingface`, `onnx`, `onnx_proto`, `weight_map`, `trace` |
| `experimental` | Research subsystems (**unstable**) | `cfie`, `cpdt`, `sparse`, `multimodal`, `unikernel`, `agent` |

## Feature flags

- `interop` — safetensors / Hugging Face / ONNX bridges. When disabled,
  `interop_stubs` provides the FFI symbols the compiler always declares, so the
  linker is satisfied even if the program never calls them.
- `onnx-rt-op` — ONNX Runtime custom-op registration surface.
- `cuda` — the CUDA driver backend.
- `test-hooks` — test-only introspection re-exports.

## Why modules stay at the crate root

As with `nsl-codegen`, the modules keep their root paths to avoid rewriting the
large body of `nsl_runtime::foo` references across the workspace and tests. The
facades supply the subsystem structure; physical relocation can follow once
consumers migrate.
