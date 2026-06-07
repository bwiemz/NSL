"""M62b Spec C gated E2E — ORT InferenceSession with NSL custom op.

Two tests live here:

* ``test_make_onnx_node_produces_valid_proto`` — unconditional sanity check
  for the ``nslpy.onnxrt.make_onnx_node`` helper. Only requires the ``onnx``
  package; runs on every CI lane that has the package installed.

* ``test_session_load_and_run_calls_nsl_add`` — full ORT round-trip. Builds a
  shared library from a one-line NSL source, loads it through ORT, and runs a
  1-node graph. Gated on ``NSL_TEST_ONNX_RT=1`` AND ``onnxruntime`` installed
  because:

    1. ``onnxruntime`` is a 100+ MB wheel we don't want to pull on every CI
       lane;
    2. the nsl-cli binary picked up via ``NSL_BIN`` must have been built with
       ``--features onnx-rt-op`` — without that feature the produced .so/.dll
       won't export ``RegisterCustomOps`` and the test will fail at load time.

The opt-in env var keeps the default ``pytest`` run cheap while still surfacing
the test in the dedicated ``test-onnx-rt`` CI job (see plan Task 7).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Gating flags
# ---------------------------------------------------------------------------

def _have_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _have_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_ONNX = _have_onnx()
_HAS_ORT = _have_ort()
_RUN_E2E = os.environ.get("NSL_TEST_ONNX_RT") == "1"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parents[2]


def _lib_ext() -> str:
    if os.name == "nt":
        return "dll"
    return "dylib" if platform.system() == "Darwin" else "so"


def _resolve_nsl_bin() -> Path:
    """NSL_BIN env override -> debug -> release."""
    env_bin = os.environ.get("NSL_BIN")
    if env_bin:
        return Path(env_bin)
    exe = "nsl.exe" if os.name == "nt" else "nsl"
    for profile in ("debug", "release"):
        candidate = WORKSPACE / "target" / profile / exe
        if candidate.is_file():
            return candidate
    # Fall back to debug path so the assertion below produces a clear message.
    return WORKSPACE / "target" / "debug" / exe


# ---------------------------------------------------------------------------
# Unconditional helper test — no ORT required, only the onnx package.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_ONNX, reason="requires the 'onnx' package")
def test_make_onnx_node_produces_valid_proto():
    """make_onnx_node should emit a NodeProto in the com.nsl domain."""
    from nslpy.onnxrt import make_onnx_node

    node = make_onnx_node("ping", inputs=["x"], outputs=["y"])
    assert node.op_type == "ping"
    assert node.domain == "com.nsl"
    assert list(node.input) == ["x"]
    assert list(node.output) == ["y"]


# ---------------------------------------------------------------------------
# Gated full E2E test — needs onnxruntime + a feature-enabled nsl-cli build.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_ORT or not _RUN_E2E,
    reason=(
        "set NSL_TEST_ONNX_RT=1 and install onnxruntime to run this test; "
        "nsl-cli must also be built with `--features onnx-rt-op` so the "
        "produced .so/.dll exports RegisterCustomOps"
    ),
)
def test_session_load_and_run_calls_nsl_add():
    """End-to-end: NSL @export double() invoked through ORT InferenceSession.

    Uses a single-input function (x -> x + x) because the v1 vtable hardcodes
    GetInputTypeCount == 1; multi-input exports require v2 variadic plumbing.
    """
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, TensorProto

    from nslpy.onnxrt import register_nsl_provider, make_onnx_node

    nsl_bin = _resolve_nsl_bin()
    assert nsl_bin.is_file(), (
        f"nsl binary not found at {nsl_bin}; set NSL_BIN or build via "
        f"`cargo build -p nsl-cli --features onnx-rt-op`"
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "double.nsl"
        src.write_text(
            "@export\n"
            "fn double(x: Tensor<[4], f32>)"
            " -> Tensor<[4], f32>:\n"
            "    return x + x\n"
        )
        lib = tmp / f"double.{_lib_ext()}"

        env = os.environ.copy()
        stdlib = WORKSPACE / "stdlib"
        if stdlib.is_dir():
            env["NSL_STDLIB_PATH"] = str(stdlib)

        # NOTE: `nsl build --shared-lib` does NOT accept a --features flag.
        # The onnx-rt-op feature must already be baked into the nsl-cli
        # binary at compile time; if it isn't, the produced .so/.dll will
        # be missing RegisterCustomOps and register_nsl_provider will fail.
        result = subprocess.run(
            [str(nsl_bin), "build", "--shared-lib", str(src), "-o", str(lib)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"nsl build failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert lib.is_file(), f"build reported success but {lib} missing"

        sess_opts = ort.SessionOptions()
        register_nsl_provider(sess_opts, str(lib))

        node = make_onnx_node("double", inputs=["x"], outputs=["y"])
        graph = helper.make_graph(
            nodes=[node],
            name="t",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [4]),
            ],
            outputs=[
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [4]),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("com.nsl", 1)],
            ir_version=8,
        )

        sess = ort.InferenceSession(model.SerializeToString(), sess_opts)
        out = sess.run(
            ["y"],
            {
                "x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            },
        )
        assert out[0].tolist() == [2.0, 4.0, 6.0, 8.0]


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    sys.exit(pytest.main([__file__, "-v"]))
