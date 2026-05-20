"""M62 @export model-method weight loading E2E test.

Proves that:
  1. An @export model method compiles to a shared library.
  2. nsl_model_create loads weights from a safetensors file.
  3. Calling predict(model, x) via ctypes returns W @ x correctly.
"""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

try:
    import numpy as np
    import safetensors.torch as st
    import torch
except ImportError:
    pytest.skip(
        "numpy + torch + safetensors required for this test",
        allow_module_level=True,
    )


FIXTURE = Path(__file__).parent / "fixtures" / "m62_predict_with_weights.nsl"
WORKSPACE_ROOT = Path(__file__).parents[2]
NSL_EXE = WORKSPACE_ROOT / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")


# ---------------------------------------------------------------------------
# NslTensorDesc ctypes mirror — must match `#[repr(C)] struct NslTensorDesc`
# in crates/nsl-runtime/src/c_api.rs byte-for-byte.
#
# C API dtype encoding (different from NSL internal!):
#   0 = f32,  1 = f64,  2 = f16,  3 = bf16,
#   4 = int32, 5 = int64, 6 = int8, 7 = uint8
# ---------------------------------------------------------------------------

class NslTensorDesc(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.c_void_p),
        ("shape",       ctypes.POINTER(ctypes.c_int64)),
        ("strides",     ctypes.POINTER(ctypes.c_int64)),
        ("ndim",        ctypes.c_int32),
        ("dtype",       ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_id",   ctypes.c_int32),
        ("tape_id",     ctypes.c_int64),
    ]


@pytest.fixture(scope="module")
def workdir():
    # Use ignore_cleanup_errors=True on Windows to tolerate the DLL file-lock
    # that ctypes holds until the process exits.  The temp directory cleanup
    # error is cosmetic; it does not affect test correctness.
    with tempfile.TemporaryDirectory(
        prefix="m62_wl_",
        ignore_cleanup_errors=True,
    ) as d:
        yield Path(d)


@pytest.fixture(scope="module")
def W_tensor():
    # Deterministic W so the matmul result is reproducible
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((4, 4)).astype(np.float32)


@pytest.fixture(scope="module")
def weights_file(workdir, W_tensor):
    path = workdir / "weights.safetensors"
    st.save_file({"W": torch.tensor(W_tensor)}, str(path))
    return path


@pytest.fixture(scope="module")
def shared_lib(workdir):
    if not NSL_EXE.exists():
        pytest.skip(f"nsl compiler binary not found at {NSL_EXE}")
    out = workdir / ("predict.dll" if os.name == "nt" else "predict.so")
    result = subprocess.run(
        [str(NSL_EXE), "build", str(FIXTURE), "--shared-lib", "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    assert out.exists(), f"shared lib not produced at {out}"
    return out


def _make_f32_desc(values, shape):
    """Build a CPU f32 NslTensorDesc from a flat list of Python floats."""
    n = len(values)
    data = (ctypes.c_float * n)(*values)
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    desc = NslTensorDesc(
        data=ctypes.cast(data, ctypes.c_void_p),
        shape=shape_arr,
        strides=None,
        ndim=len(shape),
        dtype=0,        # C API: 0 = f32
        device_type=0,  # 0 = CPU
        device_id=0,
    )
    # Return backing buffers so they stay alive for the duration of the ctypes call
    return desc, data, shape_arr


def test_predict_loads_weights_and_computes_W_at_x(shared_lib, weights_file, W_tensor):
    """predict(model, x) must return W @ x using weights from the safetensors file."""
    lib = ctypes.CDLL(str(shared_lib))

    # nsl_model_create takes an i64 (pointer to null-terminated path string) -> i64 handle
    lib.nsl_model_create.argtypes = [ctypes.c_int64]
    lib.nsl_model_create.restype = ctypes.c_int64
    # nsl_model_destroy takes an i64 handle -> i64 (returns 0 always)
    lib.nsl_model_destroy.argtypes = [ctypes.c_int64]
    lib.nsl_model_destroy.restype = ctypes.c_int64
    # nsl_get_last_error -> i64 (pointer to null-terminated C string)
    lib.nsl_get_last_error.argtypes = []
    lib.nsl_get_last_error.restype = ctypes.c_int64

    # predict(NslModel* as i64, const NslTensorDesc* x, NslTensorDesc* __ret) -> i32
    lib.predict.argtypes = [
        ctypes.c_int64,                       # NslModel* (opaque i64 handle)
        ctypes.POINTER(NslTensorDesc),        # x
        ctypes.POINTER(NslTensorDesc),        # __ret (output)
    ]
    lib.predict.restype = ctypes.c_int32

    # Build null-terminated path buffer and pass its address as i64
    path_bytes = str(weights_file).encode() + b"\0"
    path_buf = (ctypes.c_char * len(path_bytes))(*path_bytes)
    path_addr = ctypes.cast(path_buf, ctypes.c_void_p).value

    model = lib.nsl_model_create(path_addr)
    if model == 0:
        err_ptr = lib.nsl_get_last_error()
        if err_ptr != 0:
            err_msg = ctypes.cast(err_ptr, ctypes.c_char_p).value
            pytest.fail(f"nsl_model_create failed: {err_msg}")
        pytest.fail("nsl_model_create returned 0 (null model handle)")

    try:
        # x is a column vector [4, 1] (matmul requires rank >= 2)
        x_vals = [1.0, 2.0, 3.0, 4.0]
        x_desc, _x_buf, _x_shape = _make_f32_desc(x_vals, [4, 1])
        ret = NslTensorDesc()

        rc = lib.predict(
            ctypes.c_int64(model),
            ctypes.byref(x_desc),
            ctypes.byref(ret),
        )
        if rc != 0:
            err_ptr = lib.nsl_get_last_error()
            if err_ptr != 0:
                err_msg = ctypes.cast(err_ptr, ctypes.c_char_p).value
                pytest.fail(f"predict returned rc={rc}: {err_msg}")
            pytest.fail(f"predict returned rc={rc}")

        assert ret.data, "ret.data is null after successful predict call"

        # Result shape is [4, 1]; read 4 floats and compare against W @ x
        result_arr = np.ctypeslib.as_array(
            ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float)),
            shape=(4,),
        ).copy()

        x_col = np.array(x_vals, dtype=np.float32).reshape(4, 1)
        expected = (W_tensor @ x_col).reshape(4)

        np.testing.assert_allclose(result_arr, expected, atol=1e-5, rtol=1e-5)
    finally:
        lib.nsl_model_destroy(model)
