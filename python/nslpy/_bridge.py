"""DLPack zero-copy bridge between PyTorch/NumPy tensors and NSL.

Implements the defensive copy guard:
  - Zero-copy when safe (aligned, contiguous, read-only)
  - Defensive copy when alignment or layout mismatches

Also provides ``NslTensorDesc``-array marshaling helpers used by the
named-dispatch :meth:`NslModel.call` Python facade.
"""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional, Sequence


# ---------------------------------------------------------------------------
# NslTensorDesc — ctypes mirror of the runtime C struct
#
# Layout matches `#[repr(C)] struct NslTensorDesc` in
# `crates/nsl-runtime/src/c_api.rs` byte-for-byte.
#
# C API dtype encoding (NOT the internal NslTensor encoding!):
#   0 = f32,  1 = f64,  2 = f16,  3 = bf16,
#   4 = int32, 5 = int64, 6 = int8, 7 = uint8
# ---------------------------------------------------------------------------

class NslTensorDesc(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        # Autodiff tape id, copied verbatim across desc round-trips.
        # Python-supplied input tensors have no tape history → 0 (ctypes
        # auto-zeroes unset fields, but listing it explicitly here
        # keeps the C ABI mirror byte-exact at 48 bytes).
        ("tape_id", ctypes.c_int64),
    ]


_DTYPE_F32 = 0


def _f32_desc_from_floats(values: Sequence[float]):
    """Build a CPU f32 NslTensorDesc from a flat sequence of Python floats.

    Returns (desc, data_buf, shape_buf). The caller MUST hold references
    to data_buf and shape_buf for the duration of the FFI call.
    """
    n = len(values)
    data = (ctypes.c_float * n)(*values)
    shape = (ctypes.c_int64 * 1)(n)
    desc = NslTensorDesc(
        data=ctypes.cast(data, ctypes.c_void_p),
        shape=shape,
        strides=ctypes.cast(None, ctypes.POINTER(ctypes.c_int64)),
        ndim=1,
        dtype=_DTYPE_F32,
        device_type=0,
        device_id=0,
    )
    return desc, data, shape


def build_input_descs(inputs: Sequence[Any]):
    """Marshal a Python ``inputs`` sequence into a ``NslTensorDesc[n]`` array.

    v1 supports the common single-input @export shape: each input is a
    Python sequence (list/tuple) of Python floats, treated as a 1-D
    ``Tensor<[len], f32>`` on CPU. Inputs supporting ``__dlpack__`` /
    ``__array__`` are not yet supported by this helper; those paths go
    through :meth:`NslModel.forward` instead.

    Returns ``(descs_array, keepalive)`` where ``descs_array`` is a
    ``(NslTensorDesc * n)()`` ctypes array and ``keepalive`` is a list of
    backing buffers the caller must retain across the FFI call.
    """
    n = len(inputs)
    if n == 0:
        return (NslTensorDesc * 0)(), []
    arr = (NslTensorDesc * n)()
    keepalive: list[Any] = []
    for i, x in enumerate(inputs):
        if isinstance(x, (list, tuple)) and all(
            isinstance(v, (int, float)) for v in x
        ):
            desc, data, shape = _f32_desc_from_floats([float(v) for v in x])
            arr[i] = desc
            keepalive.append(data)
            keepalive.append(shape)
        else:
            raise TypeError(
                f"NslModel.call input #{i}: unsupported type "
                f"{type(x).__name__}. v1 accepts list/tuple of floats; "
                "DLPack/numpy inputs are not yet wired into the named-"
                "dispatch path."
            )
    return arr, keepalive


def allocate_output_descs(
    n_out: int,
    input_descs: Optional[Any] = None,
    n_in: int = 0,
    max_output_elems: int = 4096,
):
    """Allocate caller-owned ``NslTensorDesc[n_out]`` for dispatch outputs.

    The packed-array dispatch wrapper (see
    ``crates/nsl-codegen/src/c_wrapper.rs::emit_c_abi_dispatch_wrapper``)
    calls ``nsl_dispatch_apply_result`` which memcpy's the impl result
    into ``dst.data``. The caller therefore MUST preallocate
    ``dst.data`` large enough to hold the output. We size the buffer to
    ``max(input_elems, max_output_elems)`` * sizeof(f32); v1 assumes f32
    single-output exports whose result shape matches the largest input.

    Returns ``(descs_array, keepalive_buffers)``.
    """
    arr = (NslTensorDesc * n_out)()
    keepalive: list[Any] = []

    # Size estimate: max input element-count, falling back to
    # max_output_elems for nullary exports. f32 = 4 bytes/elem.
    n_elems = max_output_elems
    if input_descs is not None and n_in > 0:
        try:
            for i in range(n_in):
                ndim = int(input_descs[i].ndim)
                cnt = 1
                for j in range(ndim):
                    cnt *= int(input_descs[i].shape[j])
                if cnt > n_elems:
                    n_elems = cnt
        except Exception:
            pass

    for i in range(n_out):
        data_buf = (ctypes.c_float * n_elems)()
        # The wrapper overwrites shape/ndim/dtype with the impl result's
        # metadata, but we pre-populate shape so the desc is well-formed
        # if any consumer reads it before the apply step finishes.
        shape_buf = (ctypes.c_int64 * 1)(n_elems)
        arr[i] = NslTensorDesc(
            data=ctypes.cast(data_buf, ctypes.c_void_p),
            shape=shape_buf,
            strides=ctypes.cast(None, ctypes.POINTER(ctypes.c_int64)),
            ndim=1,
            dtype=_DTYPE_F32,
            device_type=0,
            device_id=0,
        )
        keepalive.append(data_buf)
        keepalive.append(shape_buf)
    return arr, keepalive


def read_f32_output_desc(desc: NslTensorDesc) -> list[float]:
    """Read a CPU f32 ``NslTensorDesc`` into a Python ``list[float]``.

    Computes the element count from ``ndim`` + ``shape[i]`` and copies
    ``ndim`` worth of f32 elements out of ``desc.data``. Raises
    :class:`ValueError` on a degenerate (null-data) result.
    """
    if not desc.data:
        raise ValueError("output desc has null data pointer")
    if desc.ndim <= 0:
        raise ValueError(f"output desc has non-positive ndim={desc.ndim}")
    n_elems = 1
    for i in range(desc.ndim):
        n_elems *= int(desc.shape[i])
    if n_elems <= 0:
        raise ValueError(f"output desc resolves to <=0 elements: {n_elems}")
    arr = ctypes.cast(desc.data, ctypes.POINTER(ctypes.c_float * n_elems))
    return list(arr.contents)

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

    # Create NslTensor from numpy via the C API.
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
