# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | Yes       |
| 0.4.x   | Yes       |
| < 0.4   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in NeuralScript, please report it responsibly:

1. **Do NOT open a public issue** for security vulnerabilities
2. **Email** the maintainer directly or use GitHub's private vulnerability reporting
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. You will receive a response within 72 hours acknowledging receipt

## Scope

Security concerns relevant to NeuralScript include:
- Command injection via `.nsl` file compilation or execution
- Unsafe memory access in the runtime (buffer overflows, use-after-free)
- Deserialization vulnerabilities in `.nslm` checkpoint loading or `.safetensors` parsing
- Path traversal in `load()` or file I/O operations
- Denial of service via malicious `.nsl` input (infinite loops in parser/compiler)

## Out of Scope

- Vulnerabilities in upstream dependencies (report to those projects directly)
- Issues only reproducible with `--debug` builds
- Performance issues (not security-relevant)
