"""Item 7 E2E: NslModel.forward returns OWNED torch tensors.

The roadmap criterion, verified end-to-end: torch tensor in via DLPack,
compiled @export runs, output comes back through the ownership-transfer path
(``nsl_model_call_alloc`` core) as a torch tensor whose memory torch's GC
owns — it stays valid after further forwards and after the model is closed.

Also covers the signature-driven ``call`` path (``nsl_model_call_into``),
scalar-f64 returns (the pre-item-7 wild read), export-signature
introspection, and the capsule-ABI regressions (``PyCapsule_GetPointer``
restype truncation; raw-int fallback on output conversion).
"""

import gc
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

WORKSPACE = Path(__file__).resolve().parents[2]
NSL_BIN = WORKSPACE / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")
STDLIB = WORKSPACE / "stdlib"


def _lib_ext() -> str:
    if os.name == "nt":
        return "dll"
    import platform
    return "dylib" if platform.system() == "Darwin" else "so"


def _check_nsl_binary():
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")


def _build_lib(tmp: Path):
    src = tmp / "own.nsl"
    src.write_text(
        "@export\n"
        "fn forward(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n"
        "    return x * 2.0\n"
        "\n"
        "@export\n"
        "fn gamma(x: Tensor<[4], f32>) -> f64:\n"
        "    return 7.5\n"
    )
    weights = tmp / "w.safetensors"
    weights.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    lib = tmp / f"own.{_lib_ext()}"
    env = os.environ.copy()
    env["NSL_STDLIB_PATH"] = str(STDLIB)
    result = subprocess.run(
        [str(NSL_BIN), "build", "--shared-lib", str(src), "-o", str(lib)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return lib, weights


@pytest.fixture(scope="module")
def model_lib():
    _check_nsl_binary()
    with tempfile.TemporaryDirectory(
        prefix="item7_own_", ignore_cleanup_errors=True
    ) as td:
        tmp = Path(td)
        yield _build_lib(tmp)


def test_forward_returns_owned_torch_tensor(model_lib):
    """The item-7 completion criterion, literally."""
    import nslpy

    lib_path, weights_path = model_lib
    model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    out = model.forward(x)

    assert isinstance(out, torch.Tensor), f"expected torch.Tensor, got {type(out)}"
    assert out.dtype == torch.float32
    assert out.tolist() == [2.0, 4.0, 6.0, 8.0]

    # OWNERSHIP part 1: a second forward must not disturb the first output
    # (the legacy desc path aliased runtime-internal memory).
    out2 = model.forward(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    assert out.tolist() == [2.0, 4.0, 6.0, 8.0]
    assert out2.tolist() == [20.0, 40.0, 60.0, 80.0]

    # OWNERSHIP part 2: the output survives model close + GC. The tensor's
    # memory belongs to torch (released by the DLPack deleter on GC), not to
    # any model-call scratch state.
    model.close()
    del model
    gc.collect()
    assert out.tolist() == [2.0, 4.0, 6.0, 8.0]
    assert out2.tolist() == [20.0, 40.0, 60.0, 80.0]


def test_forward_outputs_release_without_leaking(model_lib):
    """Dropping outputs must fire the deleter path without crashing, and
    repeated forwards must not accumulate (smoke-level; the byte-exact leak
    gate lives in the Rust integration test with RSS accounting)."""
    import nslpy

    lib_path, weights_path = model_lib
    model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    for _ in range(100):
        out = model.forward(x)
        assert out[0].item() == 2.0
        del out
    gc.collect()
    model.close()


def test_call_reads_values_via_capacity_contract(model_lib):
    import nslpy

    lib_path, weights_path = model_lib
    model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
    try:
        out = model.call("forward", [1.0, 2.0, 3.0, 4.0])
        assert out == [2.0, 4.0, 6.0, 8.0]
    finally:
        model.close()


def test_call_scalar_export_returns_the_value(model_lib):
    """gamma returns f64 7.5 — via nsl_model_call this wild-read the value
    as an address before item 7.

    This test is ALSO the symbol-interposition gate, and the export name
    `gamma` is load-bearing: libm exports `gamma(3)`, and libm is in every
    Python process's global scope. Before the artifact was linked with
    -Wl,-Bsymbolic-functions, the dispatch wrapper's internal PLT call to
    the typed `gamma` wrapper bound to LIBM's gamma — which computed lgamma
    into xmm0, wrote nothing into the scratch desc, and "succeeded" with
    rc=0 and value 0.0. Do not rename this export to something unique; a
    unique name cannot detect interposition."""
    import nslpy

    lib_path, weights_path = model_lib
    model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
    try:
        out = model.call("gamma", [1.0, 2.0, 3.0, 4.0])
        assert out == 7.5, f"scalar must arrive as a VALUE, got {out!r}"
    finally:
        model.close()


def test_export_signature_reports_shapes_and_arity(model_lib):
    import nslpy

    lib_path, weights_path = model_lib
    model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
    try:
        sig = model.export_signature("forward")
        assert sig["symbol_name"] == "forward"
        assert sig["return_type"]["Tensor"]["shape"] == ["4"]
        assert sig["return_type"]["Tensor"]["dtype"] == "F32"
        assert len(sig["params"]) == 1

        gsig = model.export_signature("gamma")
        assert gsig["return_type"]["Scalar"] == "F64"

        with pytest.raises(nslpy.NslError, match="no_such"):
            model.export_signature("no_such_export")
    finally:
        model.close()


def test_capsule_pointer_abi_is_full_width():
    """PyCapsule_GetPointer must have restype c_void_p — it defaulted to
    c_int, truncating every DLManagedTensor* above 4 GiB."""
    import ctypes

    import nslpy._bridge  # noqa: F401 — importing configures the ABI

    assert ctypes.pythonapi.PyCapsule_GetPointer.restype is ctypes.c_void_p


def test_convert_outputs_raises_instead_of_returning_raw_ints(monkeypatch):
    """A conversion failure must raise (releasing every pointer), never hand
    back a raw int masquerading as a tensor. `from_dlpack` is monkeypatched
    to fail so no garbage pointer is ever dereferenced — the subject here is
    the error-path release plumbing, not torch."""
    import ctypes

    import torch.utils.dlpack as tud
    from nslpy._bridge import convert_outputs

    def boom(_capsule):
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(tud, "from_dlpack", boom)

    freed = []

    class FakeLib:
        def nsl_dlpack_free(self, ptr):
            freed.append(ptr)

    buf = (ctypes.c_int64 * 2)(1234, 5678)
    with pytest.raises(RuntimeError, match="synthetic"):
        convert_outputs(buf, 2, FakeLib())
    # The failing pointer and the unconverted remainder must both be
    # released through nsl_dlpack_free.
    assert freed == [1234, 5678], f"released: {freed}"
