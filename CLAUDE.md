# NeuralScript (NSL) — Project Instructions

## Build & Test

- **Build**: `cargo build` from project root
- **Build (GPU)**: `cargo build --features cuda` for CUDA support
- **Build (release)**: `cargo build --release --features cuda` for production
- **Test**: `cargo test` runs all crate tests
- **Run**: `cargo run -- run examples/<file>.nsl` for execution
- **Run (source AD)**: `cargo run -- run --source-ad <file>.nsl` for compile-time autodiff
- **Clippy**: `cargo clippy -- -D warnings` for lint checks
- **Python**: use `py` (not `python`) on this Windows machine

## Code Style

- Follow existing Rust patterns in each crate
- Match naming conventions already in use (e.g., `compile_*` for codegen functions)
- Keep `nsl-parser`, `nsl-semantic`, `nsl-codegen`, `nsl-runtime`, `nsl-cli` crate boundaries clean

## GPU Runtime Rules

- `nsl_tensor_contiguous()` ALWAYS returns an owned tensor — must ALWAYS free the result
- `nsl_tensor_to_device()` ALWAYS returns an owned tensor — must ALWAYS free the result
- Never use `if c_ptr != tensor_ptr { free(c_ptr); }` — this leaks on the alias fast-path

## NotebookLM

Notebooks are organized by research domain. See memory file `notebooklm-research.md` for the registry. A hookify rule handles session-start reminders — no need to duplicate the protocol here.
