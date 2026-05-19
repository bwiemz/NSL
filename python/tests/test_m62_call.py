"""Named-dispatch E2E via the nslpy NslModel.call() Python facade.

Builds a multi-@export shared library, loads it via NslModel, and verifies
that NslModel.call(name, ...) dispatches correctly to each @export by name
and that an unknown name raises with a descriptive error.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
NSL_BIN = WORKSPACE / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")
STDLIB = WORKSPACE / "stdlib"


def _lib_ext() -> str:
    if os.name == "nt":
        return "dll"
    import platform
    return "dylib" if platform.system() == "Darwin" else "so"


def _build_multi_export_lib(tmp: Path):
    """Build a 2-@export shared library + minimal safetensors file."""
    src = tmp / "two.nsl"
    src.write_text(
        "@export\n"
        "fn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n"
        "    return x\n"
        "\n"
        "@export\n"
        "fn beta(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n"
        "    return x\n"
    )
    weights = tmp / "w.safetensors"
    weights.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    lib = tmp / f"two.{_lib_ext()}"

    env = os.environ.copy()
    env["NSL_STDLIB_PATH"] = str(STDLIB)
    result = subprocess.run(
        [str(NSL_BIN), "build", "--shared-lib", str(src), "-o", str(lib)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    assert lib.exists(), f"shared lib not produced at {lib}"
    return lib, weights


def _check_nsl_binary():
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")


def test_call_dispatches_each_export_by_name():
    _check_nsl_binary()
    with tempfile.TemporaryDirectory(
        prefix="m62_call_", ignore_cleanup_errors=True
    ) as td:
        tmp = Path(td)
        lib_path, weights_path = _build_multi_export_lib(tmp)

        import nslpy

        model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
        try:
            assert model.export_count == 2, (
                f"expected 2 exports in registry, got {model.export_count}"
            )

            x = [1.0, 2.0, 3.0, 4.0]
            out_a = model.call("alpha", x)
            out_b = model.call("beta", x)
            assert list(out_a) == [1.0, 2.0, 3.0, 4.0]
            assert list(out_b) == [1.0, 2.0, 3.0, 4.0]
        finally:
            model.close()


def test_call_unknown_name_raises():
    _check_nsl_binary()
    with tempfile.TemporaryDirectory(
        prefix="m62_call_unk_", ignore_cleanup_errors=True
    ) as td:
        tmp = Path(td)
        lib_path, weights_path = _build_multi_export_lib(tmp)

        import nslpy

        model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
        try:
            x = [1.0, 2.0, 3.0, 4.0]
            with pytest.raises(RuntimeError) as exc_info:
                model.call("does_not_exist", x)
            # Error message must name the missing export.
            assert "does_not_exist" in str(exc_info.value)
        finally:
            model.close()
