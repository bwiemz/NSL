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
