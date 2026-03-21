"""Tests for the core NslModel wrapper and library discovery."""

import pytest
from pathlib import Path
from nslpy._core import find_library, NslError


class TestLibraryDiscovery:
    def test_missing_library_raises(self):
        with pytest.raises(FileNotFoundError, match="Could not find NSL shared library"):
            find_library(search_paths=["/nonexistent/path"])

    def test_explicit_path_searched(self, tmp_path):
        # Create a fake library file
        import platform
        lib_name = {"Windows": "nsl_runtime.dll", "Darwin": "libnsl_runtime.dylib"}.get(
            platform.system(), "libnsl_runtime.so"
        )
        fake_lib = tmp_path / lib_name
        fake_lib.write_bytes(b"not a real library")

        path = find_library(search_paths=[str(tmp_path)])
        assert path == fake_lib

    def test_env_var_search(self, tmp_path, monkeypatch):
        import platform
        lib_name = {"Windows": "nsl_runtime.dll", "Darwin": "libnsl_runtime.dylib"}.get(
            platform.system(), "libnsl_runtime.so"
        )
        fake_lib = tmp_path / lib_name
        fake_lib.write_bytes(b"fake")
        monkeypatch.setenv("NSL_LIB_PATH", str(tmp_path))

        path = find_library()
        assert path == fake_lib


class TestNslError:
    def test_error_is_runtime_error(self):
        err = NslError("test error")
        assert isinstance(err, RuntimeError)
        assert str(err) == "test error"
