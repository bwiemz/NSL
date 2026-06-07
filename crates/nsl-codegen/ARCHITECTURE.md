# nsl-codegen architecture

`nsl-codegen` turns the type-checked NSL AST into native object code via
Cranelift, and runs the analysis/optimization passes in between. It is by far
the largest crate in the workspace, so this document maps its modules onto the
subsystems they belong to.

## Stable public API

Only a small, curated surface is intended for outside use. These are
re-exported at the crate root and are covered by the usual
backwards-compatibility expectations:

- `compile`, `compile_module`, `compile_entry`, `compile_test`,
  `compile_standalone`, and the `compile_*_returning_plan` family — the
  compilation entry points.
- `CompileOptions` — the configuration struct threaded through every pass.
- `CodegenError` — the crate error type.
- `create_weight_object`, `StandaloneConfig`, and the `WrgaInputs` /
  decorator-config structs consumed by the CLI.

Everything else is implementation detail. The modules are still reachable at
their crate-root paths for the CLI and the integration-test suite, but new code
should treat the subsystem facades below as the navigation map.

## Subsystem facades

Each module is declared once at the crate root (keeping `crate::foo` and
`nsl_codegen::foo` paths stable) and re-surfaced through a facade namespace in
`lib.rs`:

| Facade | Responsibility | Representative modules |
|--------|----------------|------------------------|
| `core` | The compilation pipeline itself | `compiler`, `stmt`, `expr`, `func`, `context`, `linker`, `c_header`, `c_wrapper`, `ownership`, `standalone` |
| `gpu` | Device backends & kernel lowering | `backend_ptx`, `backend_amdgpu`, `backend_metal`, `backend_wgsl`, `gpu_specs`, `gpu_target`, `kernel*`, `matmul_mma`, `ptxas_validation` |
| `training` | Autodiff & training-time codegen | `ad_rules`, `source_ad`, `wengert`, `wengert_lower`, `vmap`, `training_report` |
| `quantization` | Reduced-precision execution | `fp8`, `bitnet`, `weight_aware`, `pca_*` |
| `distributed` | Parallelism strategies | `tensor_parallel`, `context_parallel`, `pipeline`, `moe*`, `cpdt*` |
| `analysis` | Cost model, fusion, planning | `cost_model`, `autotune`, `fusion*`, `epilogue_fusion`, `reduction_fusion`, `memory_planner`, `wcet`, `profiling`, `inspect`, `flash_attention*`, `calibration` |
| `experimental` | Research subsystems (**unstable**) | `cep*`, `cfie*`, `csha*`, `wggo*`, `wrga*`, `fase*`, `zk`, `sparse`, `speculative`, `multimodal`, `unikernel*`, `experimental::fpga` (`hir`, `backend_verilog`, `kernel_lower_fpga`, `fpga_error`) |

## Experimental vs. supported

The `experimental` facade collects the research-grade subsystems described in
`docs/research/`. They compile and are tested, but their APIs, CLI flags, and
on-disk formats are explicitly **not stable** and may change or be removed
between releases. Treat code under `experimental::*` as opt-in.

## Pipeline overview

```
AST (nsl-ast) + TypeMap (nsl-semantic)
        │
        ▼
  compiler::Compiler           core
        │  collect → declare → compile (kernels, functions, main)
        ▼
  analysis passes              analysis / training / quantization / distributed
        │  fusion, AD, cost model, memory planning, calibration, …
        ▼
  kernel lowering              gpu  (+ experimental::fpga for synthesis)
        │
        ▼
  object emission + linking    core::linker
        │
        ▼
  native object / shared lib / unikernel image
```

## Why modules stay at the crate root

A physical move of every module into subdirectories would rewrite ~600
`nsl_codegen::foo` references across the CLI and the integration-test suite and
produce an unreviewable diff. Instead the modules keep their root paths and the
facades provide the subsystem structure on top. Demoting purely-internal
modules to `pub(crate)` and physically relocating files can follow
incrementally once consumers migrate to the facade paths.
