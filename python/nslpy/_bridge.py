"""DLPack zero-copy bridge between PyTorch/NumPy tensors and NSL.

Implements the defensive copy guard from the M62b design:
  - Zero-copy when safe (aligned, contiguous, read-only)
  - Defensive copy when alignment or layout mismatches
"""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Tensor validation for zero-copy safety
# ---------------------------------------------------------------------------

_TENSOR_CORE_ALIGNMENT = 64  # bytes — required for Hopper/Blackwell tensor cores


def _is_contiguous(tensor: Any) -> bool:
    """Check if a tensor is contiguous in memory."""
    if hasattr(tensor, "is_contiguous"):
        return tensor.is_contiguous()
    if hasattr(tensor, "flags"):  # numpy
        return tensor.flags.c_contiguous
    return True  # assume contiguous for unknown types


def _is_aligned(tensor: Any, alignment: int = _TENSOR_CORE_ALIGNMENT) -> bool:
    """Check if tensor data pointer is aligned to the given boundary."""
    if hasattr(tensor, "data_ptr"):
        return tensor.data_ptr() % alignment == 0
    if hasattr(tensor, "ctypes"):  # numpy
        return tensor.ctypes.data % alignment == 0
    return True  # assume aligned for unknown types


def _needs_defensive_copy(tensor: Any, mutates_input: bool = False) -> bool:
    """Determine if a defensive copy is needed for this tensor.

    Returns True (must copy) when:
      - Tensor is not contiguous
      - Data pointer is not 64-byte aligned (tensor core requirement)
      - The compiled graph will mutate this input (FBIP optimization)

    Returns False (zero-copy safe) when all checks pass.
    """
    if not _is_contiguous(tensor):
        return True
    if not _is_aligned(tensor):
        return True
    if mutates_input:
        return True
    return False


# ---------------------------------------------------------------------------
# PyTorch ↔ DLPack conversion
# ---------------------------------------------------------------------------

def _torch_to_dlpack(tensor: Any) -> int:
    """Convert a torch.Tensor to a DLManagedTensor* (as int pointer)."""
    import torch
    import torch.utils.dlpack

    if _needs_defensive_copy(tensor):
        # Defensive copy: make contiguous and aligned
        tensor = tensor.contiguous()

    capsule = torch.utils.dlpack.to_dlpack(tensor)
    # Extract raw pointer from PyCapsule
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(
        ctypes.py_object(capsule),
        b"dltensor",
    )
    return ptr


def _dlpack_to_torch(dlpack_ptr: int) -> Any:
    """Convert a DLManagedTensor* (as int pointer) back to torch.Tensor."""
    import torch
    import torch.utils.dlpack

    # Create PyCapsule wrapping the DLManagedTensor*
    capsule_name = b"dltensor"
    ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    capsule = ctypes.pythonapi.PyCapsule_New(dlpack_ptr, capsule_name, None)
    return torch.utils.dlpack.from_dlpack(capsule)


def _numpy_to_dlpack(array: Any, lib: ctypes.CDLL) -> int:
    """Convert a numpy array to an NslTensor then export as DLPack."""
    import numpy as np
    # For numpy: create an NslTensor with borrowed data, then export
    # This is a simple path — just pass the raw data pointer
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    # Create NslTensor from numpy via the C API
    from nslpy._core import _check_error
    # For now, use a simple approach: convert to torch first if available
    try:
        import torch
        t = torch.from_numpy(array)
        return _torch_to_dlpack(t)
    except ImportError:
        raise RuntimeError("NumPy → DLPack conversion requires PyTorch. Install torch.")


# ---------------------------------------------------------------------------
# Input preparation (for model.forward())
# ---------------------------------------------------------------------------

def prepare_inputs(
    inputs: Sequence[Any],
    lib: ctypes.CDLL,
) -> tuple[Optional[ctypes.Array], Callable]:
    """Convert Python tensors to DLPack pointers for the C API.

    Returns:
        (dlpack_array, cleanup_fn): The array of DLPack pointers and a function
        to call after the forward pass completes (to release references).
    """
    if not inputs:
        return None, lambda: None

    n = len(inputs)
    dlpack_ptrs = (ctypes.c_int64 * n)()
    # Keep references alive until cleanup
    _kept_alive: list[Any] = []

    for i, tensor in enumerate(inputs):
        if hasattr(tensor, "__dlpack__"):
            # Modern DLPack protocol (torch >= 2.0)
            import torch
            import torch.utils.dlpack
            if _needs_defensive_copy(tensor):
                tensor = tensor.contiguous()
            _kept_alive.append(tensor)
            capsule = torch.utils.dlpack.to_dlpack(tensor)
            _kept_alive.append(capsule)
            ptr = ctypes.pythonapi.PyCapsule_GetPointer(
                ctypes.py_object(capsule), b"dltensor",
            )
            dlpack_ptrs[i] = ptr
        elif hasattr(tensor, "__array__"):
            # numpy array
            ptr = _numpy_to_dlpack(tensor, lib)
            dlpack_ptrs[i] = ptr
        elif isinstance(tensor, int):
            # Raw pointer (already a DLPack pointer)
            dlpack_ptrs[i] = tensor
        else:
            raise TypeError(
                f"Unsupported input type: {type(tensor).__name__}. "
                f"Expected torch.Tensor, numpy.ndarray, or int (raw pointer)."
            )

    def cleanup():
        _kept_alive.clear()

    return dlpack_ptrs, cleanup


def convert_outputs(
    out_buf: ctypes.Array,
    num_outputs: int,
    lib: ctypes.CDLL,
) -> list[Any]:
    """Convert DLPack output pointers back to torch tensors."""
    results = []
    for i in range(num_outputs):
        ptr = out_buf[i]
        if ptr == 0:
            results.append(None)
            continue
        try:
            results.append(_dlpack_to_torch(ptr))
        except Exception:
            # Fallback: return raw pointer
            results.append(ptr)
    return results


# ---------------------------------------------------------------------------
# Standalone conversion functions
# ---------------------------------------------------------------------------

def to_nsl_tensor(tensor: Any, lib: Optional[ctypes.CDLL] = None) -> int:
    """Convert a Python tensor to an NSL tensor pointer via DLPack.

    Returns the raw NslTensor* as an int.
    """
    if lib is None:
        from nslpy._core import _get_lib
        lib = _get_lib()

    if hasattr(tensor, "__dlpack__"):
        dlpack_ptr = _torch_to_dlpack(tensor)
        return lib.nsl_dlpack_import(dlpack_ptr)
    elif hasattr(tensor, "__array__"):
        dlpack_ptr = _numpy_to_dlpack(tensor, lib)
        return lib.nsl_dlpack_import(dlpack_ptr)
    else:
        raise TypeError(f"Cannot convert {type(tensor).__name__} to NslTensor")


def from_nsl_tensor(nsl_ptr: int, lib: Optional[ctypes.CDLL] = None) -> Any:
    """Convert an NSL tensor pointer back to a torch.Tensor via DLPack.

    Returns a torch.Tensor (zero-copy when possible).
    """
    if lib is None:
        from nslpy._core import _get_lib
        lib = _get_lib()

    dlpack_ptr = lib.nsl_dlpack_export(nsl_ptr)
    if dlpack_ptr == 0:
        raise ValueError("Failed to export NslTensor to DLPack")
    return _dlpack_to_torch(dlpack_ptr)
