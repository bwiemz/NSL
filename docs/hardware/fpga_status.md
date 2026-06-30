# FPGA / Verilog Backend Status

NSL's codegen can lower a restricted subset of kernels to Verilog
(`backend_verilog`, `kernel_lower_fpga`). This is an **Experimental** research
backend (see [`STATUS.md`](../../STATUS.md)) — it is not part of the stable
build contract and carries no compatibility promise.

This doc states honestly what the FPGA path does and does not do, so "generates
Verilog" is not mistaken for "production HDL."

## What is validated

The nightly `fpga-nightly` workflow (`.github/workflows/fpga-nightly.yml`) runs
on `ubuntu-latest` with **Verilator** + **Yosys** installed, and exercises an
int8 MLP fixture (`mlp_int8_weights_v1`):

| Check                         | Tool       | What it proves |
|-------------------------------|------------|----------------|
| Fixture determinism           | NSL        | Regenerated weights are bit-identical to the committed fixture |
| Layer 2 full-diagnostic       | Verilator  | Simulation parity across 100 stimuli vs reference (up to 10 failures reported) |
| Layer 3 synthesizability      | Yosys      | RTL passes synthesis, including ABC technology mapping (nightly) and `-noabc` per-commit |

Tool provenance (Ubuntu / Verilator / Yosys versions) is recorded in the job log
on every run.

## Supported (today)

- Fixed-shape int8 matmul / MLP skeletons
- Simple affine loop nests over fixed-size tensors
- Deterministic, snapshot-stable generated RTL (fixture parity gate)

## Validated

- Verilator simulation parity (Layer 2)
- Yosys synthesis, with and without ABC mapping (Layer 3)
- Generated-fixture determinism

## Not yet supported

Do not assume any of the following work:

- Dynamic / data-dependent shapes
- Arbitrary control flow
- Floating-point lowering (path is int8-oriented)
- Multi-clock / CDC designs
- BRAM banking / memory-port optimization
- Place-and-route, timing closure, or device-specific resource reports
  (no vendor toolchain in CI — only lint/sim/synth)

## Roadmap to credibility

To graduate any FPGA claim from Experimental, the next evidence to add is:
device target declaration, a clock-constraint (SDC) file, post-synth resource
estimates, and timing reports from a vendor flow — none of which the current
Verilator/Yosys gate provides.
