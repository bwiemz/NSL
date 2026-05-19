"""Demonstrate the M62b Spec B per-call grad-context bridge.

Usage:
    cargo build --release -p nsl-cli
    # Linux/macOS:
    ./target/release/nsl build --shared-lib examples/m62_gradient_bridge.nsl \\
        -o /tmp/m62_gb.so
    python examples/m62_gradient_bridge.py /tmp/m62_gb.so
    # Windows (PowerShell):
    .\\target\\release\\nsl.exe build --shared-lib examples\\m62_gradient_bridge.nsl `
        -o $env:TEMP\\m62_gb.dll
    python examples\\m62_gradient_bridge.py $env:TEMP\\m62_gb.dll

API demo:
  1. Build the linear model (`y = w*x + b`, default w=1.0, b=0.0) into
     a shared library.
  2. Load with `nslpy.NslModel` + a minimal safetensors blob.
  3. `forward_grad(x=2.0)` returns `(y, ctx)` — `y` is the forward
     output as a Python list; `ctx` carries the recorded tape ops.
  4. `ctx.backward()` returns a `dict[str, list[float]]` of
     per-parameter gradients (keyed `param_<i>`; the runtime does not
     yet thread parameter names through the @export ABI).
  5. `ctx.destroy()` explicitly releases the context shell. Idempotent.

NOTE on numerical correctness (gradient values):
  The gradient values printed below are currently ZERO due to a
  pre-existing architectural mismatch in
  `crates/nsl-codegen/src/c_wrapper.rs`: the @export typed wrapper's
  `desc_to_nsl_tensor` builds fresh `NslTensor` shells whose
  `tape_id` fields do NOT match the model's weight tensors, so the
  chain rule never propagates back to the parameters. The forward
  direction is correct (`y == 2.0` with the default initializers).

  Once `c_wrapper.rs` is fixed, this harness should print
  (with default w=1.0, b=0.0 and a scalar-loss seed of 1.0):

      dL/dw  =  x  =  2.0
      dL/db  =  1  =  1.0

  The same xfail-flip mechanism that gates the corresponding pytest
  (`python/tests/test_m62_grad_context.py::test_forward_grad_then_backward_returns_correct_gradients`)
  will convert this example into a passing demo automatically when
  the upstream fix lands; no edits to this file required.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import nslpy


def _write_minimal_safetensors(path: Path) -> None:
    """Write a minimal (empty-header) safetensors file.

    The example model uses default initializers (`ones([1])` /
    `zeros([1])`), so no weights need to be threaded in from disk —
    but `NslModel(..., weights_path=...)` still requires a valid
    safetensors file to bootstrap the runtime's weight loader.
    An empty JSON object satisfies the parser.
    """
    header = b"{}"
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def main(lib_path: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        weights = Path(td) / "weights.safetensors"
        _write_minimal_safetensors(weights)

        model = nslpy.NslModel(lib_path, weights_path=str(weights))
        try:
            print(f"loaded model from {lib_path}")
            print("default initializers: w=1.0, b=0.0")

            # ── Forward pass with tape recording ──────────────────────
            y, ctx = model.forward_grad([2.0])
            y_val = list(y)
            print(f"forward: y = w*x + b = 1.0 * 2.0 + 0.0 = {y_val}")
            assert y_val == [2.0], (
                f"forward output incorrect: expected [2.0], got {y_val}"
            )

            # ── Backward pass through the recorded tape ───────────────
            try:
                grads = ctx.backward()
                print(f"backward returned {len(grads)} gradient(s)")
                for name, g in grads.items():
                    print(f"  {name} = {g}")
                if not any(any(v != 0.0 for v in g) for g in grads.values()):
                    print(
                        "  (all-zero gradients are expected on this branch — "
                        "see the c_wrapper.rs gating note in the module "
                        "docstring.)"
                    )
            finally:
                # ── Release the context shell ─────────────────────────
                ctx.destroy()
                print("ctx destroyed cleanly")

            print()
            print("API demo complete. The forward direction is correct; the")
            print("gradient-correctness gating is documented in the docstring.")
        finally:
            model.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: m62_gradient_bridge.py <path/to/lib.so|.dylib|.dll>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1])
