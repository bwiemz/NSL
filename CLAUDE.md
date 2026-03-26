# NeuralScript (NSL) — Project Instructions

## Build & Test
- **Build**: `cargo build` from project root
- **Test**: `cargo test` runs all crate tests
- **Run**: `cargo run -- check examples/<file>.nsl` for type-checking, `cargo run -- run examples/<file>.nsl` for execution
- **Clippy**: `cargo clippy -- -D warnings` for lint checks

## Code Style
- Follow existing Rust patterns in each crate
- Match naming conventions already in use (e.g., `compile_*` for codegen functions)
- Keep `nsl-parser`, `nsl-semantic`, `nsl-codegen`, `nsl-runtime`, `nsl-cli` crate boundaries clean

## NotebookLM Research Protocol

### When to Consult NotebookLM
Before any of these activities, check if a relevant research notebook exists and query it:

1. **Planning new milestones** — Query competitive landscape + relevant domain notebooks
2. **Designing features (M32-M62)** — Query the domain-specific notebook for that feature area
3. **Performance/optimization work** — Query performance theory notebook for grounded baselines
4. **Evaluating architectural decisions** — Query ML compiler theory notebook for prior art
5. **Writing design docs** — Cross-reference claims against uploaded papers

### How to Use
1. Run `list_notebooks` to see available research notebooks
2. `select_notebook` for the relevant domain
3. `ask_question` with targeted queries grounded in the current task
4. Cross-reference NotebookLM answers against NSL's actual codebase before acting
5. If NotebookLM returns findings that conflict with current implementation, flag to user

### Session Start Checklist
At the start of each session involving research, planning, or design:
- [ ] Check NotebookLM health (`get_health`)
- [ ] List available notebooks (`list_notebooks`)
- [ ] Identify which notebooks are relevant to the session's goals
- [ ] Query relevant notebooks before proposing designs or plans

### Notebook Categories
Notebooks are organized by research domain. See memory file `notebooklm-research.md` for the full registry with URLs, topics, and milestone mappings.

| Category | Purpose | Relevant Milestones |
|---|---|---|
| NSL General | Project spec, design docs, architecture | All |
| Competitive Landscape | Mojo, Julia, Triton, JAX, TVM, MLIR | M47, M32-M51 |
| ML Compiler Theory | Polyhedral, fusion, memory planning, IR design | M31, M36, M37, M40 |
| Type Systems & Safety | Linear types, effect systems, dependent types, shape algebra | M38, M49, M51 |
| Inference Optimization | FlashAttention, PagedAttention, speculative decoding, batching | M25, M27, M29, M33 |
| Frontier Features | WCET, ZK inference, neuromorphic, sparse, unikernels | M50, M53, M54, M55, M57 |
| Performance Theory | Roofline models, memory bandwidth, kernel profiling | M26, M37 |

### Rules
- **Never hallucinate paper citations** — only cite what NotebookLM returns from uploaded sources
- **Always cross-reference** — NotebookLM findings must be validated against NSL's actual architecture
- **Update notebooks** — when new papers/docs are found, suggest adding them to the appropriate notebook
- **Flag gaps** — if a query returns no useful results, flag that the notebook needs more sources
