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
# Dtype encoding: the canonical NSL tag space (P4 item 16 unified the C API
# with `NslTensor.dtype` — there is no separate inverted C-API convention):
#   0 = f64,  1 = f32,  2 = f16,  3 = bf16,
#   4 = int8, 9 = int32
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


_DTYPE_F32 = 1


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


# Byte sizes for the compiler's ExportDtype names (as they appear in the
# export-signature JSON).
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "Bool": 1,
}

# Maximum output rank the sized allocator provisions shape/strides slots for.
# The runtime refuses (reporting the actual rank) if an output exceeds it.
_MAX_OUT_NDIM = 8


def allocate_output_descs_sized(n_out: int, capacity_bytes: int):
    """Allocate output descs for the ``nsl_model_call_into`` contract.

    Each desc gets a ``capacity_bytes`` data buffer, caller-owned
    shape/strides arrays of ``_MAX_OUT_NDIM`` slots, and ``ndim`` set to the
    slot capacity ON ENTRY (the call_into contract: the runtime validates
    the result rank against it, deep-copies dims in, then sets ``ndim`` to
    the result rank).

    Returns ``(descs_array, keepalive_buffers, capacities_array)`` where
    ``capacities_array`` is the ``uint64_t[n_out]`` passed as
    ``out_capacities``.
    """
    capacity_bytes = max(int(capacity_bytes), 8)
    arr = (NslTensorDesc * n_out)()
    caps = (ctypes.c_uint64 * n_out)()
    keepalive: list[Any] = []
    for i in range(n_out):
        data_buf = (ctypes.c_char * capacity_bytes)()
        shape_buf = (ctypes.c_int64 * _MAX_OUT_NDIM)()
        strides_buf = (ctypes.c_int64 * _MAX_OUT_NDIM)()
        arr[i] = NslTensorDesc(
            data=ctypes.cast(data_buf, ctypes.c_void_p),
            shape=shape_buf,
            strides=strides_buf,
            ndim=_MAX_OUT_NDIM,
            dtype=_DTYPE_F32,
            device_type=0,
            device_id=0,
        )
        caps[i] = capacity_bytes
        keepalive.extend((data_buf, shape_buf, strides_buf))
    return arr, keepalive, caps


def read_output_desc(desc: NslTensorDesc):
    """Read a call_into output desc into Python values.

    Handles the shapes the item-7 dispatch contract can produce: rank-0
    f64 scalars (returned as ``float``) and CPU f32/f64 tensors (returned
    as ``list[float]``).
    """
    if not desc.data:
        raise ValueError("output desc has null data pointer")
    if desc.ndim == 0:
        return ctypes.cast(desc.data, ctypes.POINTER(ctypes.c_double))[0]
    n_elems = 1
    for i in range(desc.ndim):
        n_elems *= int(desc.shape[i])
    if n_elems <= 0:
        raise ValueError(f"output desc resolves to <=0 elements: {n_elems}")
    if desc.dtype == 0:  # f64
        arr = ctypes.cast(desc.data, ctypes.POINTER(ctypes.c_double * n_elems))
        return list(arr.contents)
    if desc.dtype == _DTYPE_F32:
        arr = ctypes.cast(desc.data, ctypes.POINTER(ctypes.c_float * n_elems))
        return list(arr.contents)
    raise ValueError(
        f"read_output_desc: unsupported output dtype tag {desc.dtype} "
        "(v1 reads f64/f32 CPU results)"
    )


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

# CPython capsule FFI, configured ONCE at module scope. `PyCapsule_GetPointer`
# had no declared restype anywhere in this module, so ctypes applied its
# default of c_int: the returned DLManagedTensor* was silently TRUNCATED to
# 32 bits — every heap allocation above 4 GiB produced a corrupt pointer.
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_SetName.restype = ctypes.c_int


def _capsule_pointer(capsule: Any) -> int:
    """Extract the DLManagedTensor* from a 'dltensor' capsule (full width)."""
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
    return int(ptr or 0)


def _mark_capsule_consumed(capsule: Any) -> None:
    """Rename a 'dltensor' capsule to 'used_dltensor'.

    This is the DLPack exactly-once convention: a consumer that takes
    responsibility for the deleter renames the capsule so the producer's
    capsule destructor (which only fires on the name 'dltensor') becomes a
    no-op — otherwise the deleter can fire twice, once from the consumer and
    once from capsule GC.
    """
    ctypes.pythonapi.PyCapsule_SetName(capsule, b"used_dltensor")


def _torch_to_dlpack(tensor: Any) -> tuple[int, Any]:
    """Convert a torch.Tensor to ``(DLManagedTensor* as int, keepalive)``.

    The caller MUST hold the returned keepalive (the capsule) for as long as
    the pointer is dereferenced: the previous version returned the bare int
    and dropped the capsule, whose GC-time destructor calls the DLPack
    deleter — a use-after-free window on every standalone conversion.
    """
    import torch
    import torch.utils.dlpack

    if _needs_defensive_copy(tensor):
        # Defensive copy: make contiguous and aligned
        tensor = tensor.contiguous()

    capsule = torch.utils.dlpack.to_dlpack(tensor)
    ptr = _capsule_pointer(capsule)
    return ptr, capsule


def _dlpack_to_torch(dlpack_ptr: int, lib: Optional[ctypes.CDLL] = None) -> Any:
    """Convert an NSL-owned DLManagedTensor* into an OWNED torch.Tensor.

    torch consumes the capsule (renaming it 'used_dltensor') and takes
    responsibility for calling the deleter from its GC — for tensors produced
    by ``nsl_model_call_alloc`` that deleter releases the underlying NSL
    tensor, which is what makes the returned torch tensor owned.

    If conversion fails the DLManagedTensor is released via
    ``nsl_dlpack_free`` (exactly once) instead of leaking, and the error
    propagates.
    """
    import torch
    import torch.utils.dlpack

    capsule = ctypes.pythonapi.PyCapsule_New(dlpack_ptr, b"dltensor", None)
    try:
        return torch.utils.dlpack.from_dlpack(capsule)
    except Exception:
        if lib is not None and hasattr(lib, "nsl_dlpack_free"):
            lib.nsl_dlpack_free(dlpack_ptr)
        raise


def _numpy_to_dlpack(array: Any, lib: ctypes.CDLL) -> tuple[int, Any]:
    """Convert a numpy array to ``(DLManagedTensor*, keepalive)`` via torch."""
    import numpy as np
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    try:
        import torch
        t = torch.from_numpy(array)
        ptr, capsule = _torch_to_dlpack(t)
        # The torch tensor aliases the numpy buffer; keep all three alive.
        return ptr, (capsule, t, array)
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
            if _needs_defensive_copy(tensor):
                tensor = tensor.contiguous()
            _kept_alive.append(tensor)
            ptr, capsule = _torch_to_dlpack(tensor)
            _kept_alive.append(capsule)
            dlpack_ptrs[i] = ptr
        elif hasattr(tensor, "__array__"):
            # numpy array
            ptr, keepalive = _numpy_to_dlpack(tensor, lib)
            _kept_alive.append(keepalive)
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
    """Convert owned DLPack output pointers into owned torch tensors.

    Each pointer's ownership moves to exactly one place: torch's GC on
    success, or ``nsl_dlpack_free`` on failure. The previous version fell
    back to returning the raw int pointer on conversion failure — handing
    the caller a leak with no API to release it and a value that crashes
    anything treating it as a tensor.
    """
    results = []
    for i in range(num_outputs):
        ptr = out_buf[i]
        if ptr == 0:
            results.append(None)
            continue
        try:
            results.append(_dlpack_to_torch(ptr, lib))
        except Exception:
            # _dlpack_to_torch already released THIS pointer; release the
            # not-yet-converted remainder so a partial failure leaks nothing.
            for j in range(i + 1, num_outputs):
                if out_buf[j] and hasattr(lib, "nsl_dlpack_free"):
                    lib.nsl_dlpack_free(out_buf[j])
            raise
    return results


# ---------------------------------------------------------------------------
# Standalone conversion functions
# ---------------------------------------------------------------------------

# Keepalives for standalone `to_nsl_tensor` imports: the NSL wrapper BORROWS
# the producer's memory, so the producer-side capsule/tensor must outlive the
# NslTensor*. Keyed by the NslTensor pointer; released via
# `release_nsl_tensor`. (Before this existed the capsule was dropped on
# return — its GC destructor fired the producer's deleter while the borrowed
# NslTensor* was still live: a use-after-free, not a leak.)
_NSL_IMPORT_KEEPALIVE: dict[int, Any] = {}


def to_nsl_tensor(tensor: Any, lib: Optional[ctypes.CDLL] = None) -> int:
    """Convert a Python tensor to a BORROWING NSL tensor pointer via DLPack.

    Returns the raw NslTensor* as an int. The source tensor's memory is kept
    alive by an internal registry until :func:`release_nsl_tensor` is called
    with the returned pointer.
    """
    if lib is None:
        from nslpy._core import _get_lib
        lib = _get_lib()

    if hasattr(tensor, "__dlpack__"):
        dlpack_ptr, keepalive = _torch_to_dlpack(tensor)
    elif hasattr(tensor, "__array__"):
        dlpack_ptr, keepalive = _numpy_to_dlpack(tensor, lib)
    else:
        raise TypeError(f"Cannot convert {type(tensor).__name__} to NslTensor")
    nsl_ptr = lib.nsl_dlpack_import(dlpack_ptr)
    if nsl_ptr:
        _NSL_IMPORT_KEEPALIVE[int(nsl_ptr)] = (keepalive, tensor)
    return nsl_ptr


def release_nsl_tensor(nsl_ptr: int) -> None:
    """Drop the keepalive for a pointer returned by :func:`to_nsl_tensor`.

    After this the borrowed NslTensor* must not be dereferenced again.
    """
    _NSL_IMPORT_KEEPALIVE.pop(int(nsl_ptr), None)


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
