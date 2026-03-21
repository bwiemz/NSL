"""Core ctypes wrapper for the NSL shared library C API."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _lib_name() -> str:
    """Platform-specific shared library name."""
    system = platform.system()
    if system == "Windows":
        return "nsl_runtime.dll"
    elif system == "Darwin":
        return "libnsl_runtime.dylib"
    else:
        return "libnsl_runtime.so"


def find_library(search_paths: Optional[Sequence[str | Path]] = None) -> Path:
    """Locate the NSL shared library.

    Search order:
      1. Explicit *search_paths*
      2. NSL_LIB_PATH environment variable
      3. Next to this Python file (editable install)
      4. ``<nslpy_package>/../lib/``
      5. System library paths
    """
    name = _lib_name()

    candidates: list[Path] = []
    if search_paths:
        candidates.extend(Path(p) / name for p in search_paths)

    env_path = os.environ.get("NSL_LIB_PATH")
    if env_path:
        candidates.append(Path(env_path) / name)
        candidates.append(Path(env_path))  # maybe full path to lib

    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / name)
    candidates.append(pkg_dir.parent / "lib" / name)

    for c in candidates:
        if c.is_file():
            return c

    # Last resort: let ctypes search system paths
    try:
        ctypes.CDLL(name)
        return Path(name)
    except OSError:
        pass

    raise FileNotFoundError(
        f"Could not find NSL shared library '{name}'. "
        f"Set NSL_LIB_PATH or pass search_paths= to find_library()."
    )


# ---------------------------------------------------------------------------
# C API binding
# ---------------------------------------------------------------------------

class NslError(RuntimeError):
    """Error returned by the NSL runtime."""
    pass


def _load_lib(path: Optional[Path] = None) -> ctypes.CDLL:
    """Load and bind the NSL shared library."""
    if path is None:
        path = find_library()
    lib = ctypes.CDLL(str(path))

    # ── Model lifecycle ──────────────────────────────────────────────
    lib.nsl_model_create.argtypes = [ctypes.c_int64]
    lib.nsl_model_create.restype = ctypes.c_int64

    lib.nsl_model_destroy.argtypes = [ctypes.c_int64]
    lib.nsl_model_destroy.restype = ctypes.c_int64

    lib.nsl_model_forward_dlpack.argtypes = [
        ctypes.c_int64,  # model_ptr
        ctypes.c_int64,  # inputs_ptr (array of DLManagedTensor*)
        ctypes.c_int64,  # num_inputs
        ctypes.c_int64,  # outputs_ptr (output buffer)
        ctypes.c_int64,  # num_outputs_ptr
    ]
    lib.nsl_model_forward_dlpack.restype = ctypes.c_int64

    lib.nsl_model_get_version.argtypes = []
    lib.nsl_model_get_version.restype = ctypes.c_int64

    lib.nsl_model_num_weights.argtypes = [ctypes.c_int64]
    lib.nsl_model_num_weights.restype = ctypes.c_int64

    # ── DLPack ────────────────────────────────────────────────────────
    lib.nsl_dlpack_export.argtypes = [ctypes.c_int64]
    lib.nsl_dlpack_export.restype = ctypes.c_int64

    lib.nsl_dlpack_import.argtypes = [ctypes.c_int64]
    lib.nsl_dlpack_import.restype = ctypes.c_int64

    lib.nsl_dlpack_free.argtypes = [ctypes.c_int64]
    lib.nsl_dlpack_free.restype = None

    # ── Error handling ─────────────────────────────────────────────────
    lib.nsl_get_last_error.argtypes = []
    lib.nsl_get_last_error.restype = ctypes.c_int64

    lib.nsl_clear_error.argtypes = []
    lib.nsl_clear_error.restype = ctypes.c_int64

    # ── Backward pass (M62b) ──────────────────────────────────────────
    lib.nsl_model_backward.argtypes = [
        ctypes.c_int64,  # model_ptr
        ctypes.c_int64,  # grad_output_dlpack
        ctypes.c_int64,  # num_grad_outputs
        ctypes.c_int64,  # grad_inputs_buf (output)
        ctypes.c_int64,  # num_grad_inputs_ptr (output)
    ]
    lib.nsl_model_backward.restype = ctypes.c_int64

    return lib


# Lazy-loaded library singleton
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        try:
            _lib = _load_lib()
        except FileNotFoundError:
            # Allow import without library for documentation/type-checking
            raise
    return _lib


def _check_error(result: int, lib: ctypes.CDLL) -> int:
    """Check for NSL runtime error after a C API call."""
    if result < 0:
        err_ptr = lib.nsl_get_last_error()
        if err_ptr:
            msg = ctypes.cast(err_ptr, ctypes.c_char_p).value
            lib.nsl_clear_error()
            raise NslError(msg.decode("utf-8") if msg else f"NSL error (code {result})")
        raise NslError(f"NSL error (code {result})")
    return result


# ---------------------------------------------------------------------------
# NslModel — Python wrapper for compiled NSL models
# ---------------------------------------------------------------------------

class NslModel:
    """A compiled NSL model loaded from a .nslm or safetensors file.

    Args:
        path: Path to the model weights (safetensors format).
        lib_path: Optional explicit path to the NSL shared library.

    Example::

        model = NslModel("gpt2.safetensors")
        output = model(input_tokens)
    """

    def __init__(self, path: str | Path, lib_path: Optional[Path] = None):
        if lib_path is not None:
            self._lib = _load_lib(lib_path)
        else:
            self._lib = _get_lib()

        path_bytes = str(path).encode("utf-8")
        # Create a C string for the path
        path_buf = ctypes.create_string_buffer(path_bytes)
        path_ptr = ctypes.cast(path_buf, ctypes.c_void_p).value

        self._handle = self._lib.nsl_model_create(path_ptr)
        if self._handle == 0:
            raise NslError(f"Failed to load model from '{path}'")
        self._path = Path(path)
        self._destroyed = False

    @property
    def num_weights(self) -> int:
        """Number of weight tensors in the model."""
        return self._lib.nsl_model_num_weights(self._handle)

    @property
    def version(self) -> str:
        """NSL runtime version string."""
        ptr = self._lib.nsl_model_get_version()
        if ptr:
            return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
        return "unknown"

    def forward(self, *inputs):
        """Run the model forward pass.

        Args:
            *inputs: Input tensors. Accepts torch.Tensor (via DLPack),
                     numpy arrays, or raw ctypes pointers.

        Returns:
            Output tensor(s) as torch.Tensor if torch is available,
            otherwise as raw DLPack pointers.
        """
        from nslpy._bridge import prepare_inputs, convert_outputs
        dl_inputs, cleanup = prepare_inputs(inputs, self._lib)

        # Allocate output buffer (up to 8 outputs)
        max_outputs = 8
        out_buf = (ctypes.c_int64 * max_outputs)()
        num_out = ctypes.c_int64(0)

        result = self._lib.nsl_model_forward_dlpack(
            self._handle,
            ctypes.cast(dl_inputs, ctypes.c_int64) if dl_inputs else 0,
            len(inputs),
            ctypes.cast(out_buf, ctypes.c_int64),
            ctypes.addressof(num_out),
        )
        _check_error(result, self._lib)

        outputs = convert_outputs(out_buf, num_out.value, self._lib)
        cleanup()
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def backward(self, grad_output, inputs=None):
        """Run the model backward pass (gradient computation).

        Args:
            grad_output: Gradient of the loss w.r.t. the model output.
            inputs: Original inputs (needed for some AD modes).

        Returns:
            Tuple of gradient tensors for each input.
        """
        from nslpy._bridge import prepare_inputs, convert_outputs
        dl_grads, cleanup = prepare_inputs((grad_output,), self._lib)

        max_grad_inputs = 8
        grad_buf = (ctypes.c_int64 * max_grad_inputs)()
        num_grads = ctypes.c_int64(0)

        result = self._lib.nsl_model_backward(
            self._handle,
            ctypes.cast(dl_grads, ctypes.c_int64) if dl_grads else 0,
            1,
            ctypes.cast(grad_buf, ctypes.c_int64),
            ctypes.addressof(num_grads),
        )
        _check_error(result, self._lib)

        grads = convert_outputs(grad_buf, num_grads.value, self._lib)
        cleanup()
        return tuple(grads)

    def __call__(self, *inputs):
        """Shorthand for forward()."""
        return self.forward(*inputs)

    def __del__(self):
        if not self._destroyed and hasattr(self, "_handle") and self._handle:
            self._lib.nsl_model_destroy(self._handle)
            self._destroyed = True

    def __repr__(self) -> str:
        n = self.num_weights if not self._destroyed else "?"
        return f"NslModel(path='{self._path}', weights={n})"
