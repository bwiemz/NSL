# Security Policy

## Supported Versions

Security fixes are applied to the development line and the latest minor
release. Older minors are not patched unless a fix is explicitly backported.

| Version            | Supported          |
|--------------------|--------------------|
| `main` (dev)       | Yes                |
| 0.9.x (latest)     | Yes                |
| <= 0.8.x           | No (unless backported) |

> NSL has not yet reached a 1.0 stability commitment. Until then, only the
> latest minor release receives security maintenance. See
> [`STATUS.md`](STATUS.md) for which subsystems are considered stable.

## Reporting a Vulnerability

If you discover a security vulnerability in NeuralScript, please report it
responsibly:

1. **Do NOT open a public issue** for security vulnerabilities.
2. **Use GitHub's private vulnerability reporting** (Security → "Report a
   vulnerability") or email the maintainer directly.
3. Include a description of the vulnerability, steps to reproduce, affected
   version/commit, and potential impact.
4. You will receive a response within 72 hours acknowledging receipt.

## Highest-Risk Areas

NSL is a compiler **and** a native runtime with a C ABI, `unsafe` code, file
parsers, and optional GPU/dynamic-library paths. The following areas carry the
most security weight and receive the most scrutiny in review:

- **Runtime C ABI boundary** (`crates/nsl-runtime/src/c_api/`) — raw pointers,
  ownership/lifetime contracts, alignment, and the "no unwinding across FFI"
  invariant. See `crates/nsl-runtime/ARCHITECTURE.md`.
- **Dynamic library loading** (`c_api::exports`) — `dlopen`/`dlsym` of
  model-emitted shared objects and the export dispatch table.
- **Model / weight loading** — `.nslm` checkpoints, `.safetensors`, ONNX
  protobuf parsing, and tokenizer files. Untrusted artifacts are an attack
  surface (deserialization, integer overflow, allocation bombs).
- **Path handling** — `load()`, `load_mmap()`, checkpoint and data-file I/O
  (path traversal, symlink following).
- **CUDA / accelerator launch boundary** — kernel launch parameters, stream
  semantics, and host↔device transfer sizing.
- **Generated code execution** — `nsl run`/`nsl build` compile and execute
  native code; treat compiling an untrusted `.nsl` file as running it.
- **Compiler denial-of-service** — pathological `.nsl` input causing unbounded
  work in the lexer, parser, or semantic passes.

## Out of Scope

- Vulnerabilities in upstream dependencies (report to those projects directly).
- Issues only reproducible with `--debug` builds.
- Experimental subsystems (`experimental::*`; see `STATUS.md`) that are not part
  of the stable contract — reports are welcome but are triaged at lower priority.
- Performance issues that are not security-relevant.
