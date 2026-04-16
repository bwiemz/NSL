"""M62 @export E2E: compile to .so, load via ctypes, verify symbol and header."""

import ctypes
import os
import subprocess
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
NSL_BIN = WORKSPACE / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")
FIXTURE = WORKSPACE / "examples" / "m62_shared_lib.nsl"


@pytest.fixture(scope="module")
def shared_lib(tmp_path_factory):
    """Build the fixture as a .so/.dll and yield its path + generated header path."""
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found at {FIXTURE}")

    tmp = tmp_path_factory.mktemp("m62_export")
    if os.name == "nt":
        out = tmp / "libadd.dll"
    else:
        out = tmp / "libadd.so"

    result = subprocess.run(
        [str(NSL_BIN), "build", str(FIXTURE), "--shared-lib", "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    assert out.exists(), f"shared lib not produced at {out}"
    yield out


def test_shared_lib_has_add_symbol(shared_lib):
    """The @export fn add should be a reachable C symbol."""
    lib = ctypes.CDLL(str(shared_lib))
    add_fn = getattr(lib, "add", None)
    assert add_fn is not None, "'add' symbol not found in shared library"


def test_generated_header_exists_and_declares_add(shared_lib):
    """The .h file should be emitted alongside the .so and prototype `add`."""
    header_path = shared_lib.with_suffix(".h")
    assert header_path.exists(), f"header not emitted at {header_path}"
    content = header_path.read_text()
    assert "int add(NslModel" in content, f"missing `int add` prototype in header:\n{content}"
    assert "const NslTensorDesc*" in content, f"missing NslTensorDesc input type:\n{content}"
    assert "NslTensorDesc* __ret" in content, f"missing __ret out-param:\n{content}"


def test_no_export_functions_no_header(tmp_path_factory):
    """A shared-lib build without any @export should NOT produce a header."""
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")

    tmp = tmp_path_factory.mktemp("m62_no_export")
    src = tmp / "no_export.nsl"
    src.write_text("fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return a + b\n")

    if os.name == "nt":
        out = tmp / "libnoexport.dll"
    else:
        out = tmp / "libnoexport.so"

    result = subprocess.run(
        [str(NSL_BIN), "build", str(src), "--shared-lib", "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    header_path = out.with_suffix(".h")
    assert not header_path.exists(), f"header should not be emitted when no @export present, but found {header_path}"


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
    ]


def _make_f32_desc(values):
    """Build a CPU f32 NslTensorDesc from a list of Python floats.

    The fixture `add` function takes ``Tensor<[4], f32>`` inputs.  In the
    C API dtype encoding, f32 == 0.
    """
    n = len(values)
    data = (ctypes.c_float * n)(*values)
    shape = (ctypes.c_int64 * 1)(n)
    desc = NslTensorDesc(
        data=ctypes.cast(data, ctypes.c_void_p),
        shape=shape,
        strides=None,
        ndim=1,
        dtype=0,        # C API: 0 = f32
        device_type=0,  # 0 = CPU
        device_id=0,
    )
    return desc, data, shape


def test_add_actually_computes_sum(shared_lib):
    """Calling add(a, b) via ctypes should return a + b element-wise."""
    lib = ctypes.CDLL(str(shared_lib))
    lib.add.argtypes = [
        ctypes.c_void_p,               # NslModel* (opaque handle)
        ctypes.POINTER(NslTensorDesc), # a
        ctypes.POINTER(NslTensorDesc), # b
        ctypes.POINTER(NslTensorDesc), # __ret (output)
    ]
    lib.add.restype = ctypes.c_int32

    a_desc, a_buf, a_shape = _make_f32_desc([1.0, 2.0, 3.0, 4.0])
    b_desc, b_buf, b_shape = _make_f32_desc([10.0, 20.0, 30.0, 40.0])
    ret = NslTensorDesc()

    # A non-null dummy model pointer (value=1) bypasses the null guard.
    dummy_model = ctypes.c_void_p(1)
    rc = lib.add(
        dummy_model,
        ctypes.byref(a_desc),
        ctypes.byref(b_desc),
        ctypes.byref(ret),
    )
    assert rc == 0, f"add returned non-zero rc={rc}"
    assert ret.data, "ret.data must be non-null after successful call"

    result = ctypes.cast(ret.data, ctypes.POINTER(ctypes.c_float * 4)).contents
    assert list(result) == pytest.approx([11.0, 22.0, 33.0, 44.0]), (
        f"element-wise sum wrong: {list(result)}"
    )


def test_null_model_returns_error(shared_lib):
    """Passing NULL as the NslModel* must return -1 (null-guard in wrapper)."""
    lib = ctypes.CDLL(str(shared_lib))
    lib.add.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
        ctypes.POINTER(NslTensorDesc),
    ]
    lib.add.restype = ctypes.c_int32

    a_desc, _, _ = _make_f32_desc([1.0, 2.0, 3.0, 4.0])
    b_desc, _, _ = _make_f32_desc([10.0, 20.0, 30.0, 40.0])
    ret = NslTensorDesc()

    rc = lib.add(None, ctypes.byref(a_desc), ctypes.byref(b_desc), ctypes.byref(ret))
    assert rc == -1, f"expected rc=-1 for null model, got rc={rc}"
