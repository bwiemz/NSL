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

    # ── Spec B: per-call grad context FFIs ──────────────────────────────
    # Bound centrally so the same ABI is enforced on both the standalone
    # runtime and on @export-emitted shared libraries (which statically
    # link the runtime). The new `nsl_model_backward` takes a
    # `*mut GradContext` as the first argument (NOT a model handle —
    # Spec B T8 removed the model-level grad path).
    _bind_grad_context_ffis(lib)

    _bind_named_dispatch_ffis(lib)

    return lib


def _bind_grad_context_ffis(lib: ctypes.CDLL) -> None:
    """Bind the per-call grad-context FFIs (Spec B §4.2/§4.3/§4.4).

    `nsl_model_forward_grad` records a forward pass into a fresh
    `GradContext` and writes the ctx pointer out via a caller-supplied
    `*mut *mut GradContext` slot.

    `nsl_model_backward` REPLACES the legacy `nsl_model_backward(model, ...)`
    signature — its first argument is now the `*mut GradContext` produced
    by `forward_grad`, NOT a model handle. The runtime validates a 4-byte
    magic header on the ctx pointer so a misuse (e.g. passing a model
    handle) is caught and returned as rc=-1 with a typed error, not UB.

    `nsl_grad_context_destroy` reclaims the ctx Box. Idempotent + safe
    on a freed or bogus pointer thanks to the same magic-header gate.
    """
    if hasattr(lib, "nsl_model_forward_grad"):
        lib.nsl_model_forward_grad.argtypes = [
            ctypes.c_int64,  # model_ptr
            ctypes.c_int64,  # inputs_ptr (NslTensorDesc*)
            ctypes.c_int64,  # num_inputs
            ctypes.c_int64,  # outputs_ptr (NslTensorDesc*)
            ctypes.c_int64,  # num_outputs
            ctypes.c_int64,  # grad_context_out (*mut *mut GradContext)
        ]
        lib.nsl_model_forward_grad.restype = ctypes.c_int64

    if hasattr(lib, "nsl_model_backward"):
        # NOTE: this REPLACES any prior binding. The legacy
        # `nsl_model_backward(model, ...)` ABI was removed in Spec B T8.
        lib.nsl_model_backward.argtypes = [
            ctypes.c_int64,  # ctx_ptr (*mut GradContext)
            ctypes.c_int64,  # grad_outputs_ptr (NslTensorDesc*) — v1 unused
            ctypes.c_int64,  # num_grad_outputs                   — v1 unused
            ctypes.c_int64,  # grad_inputs_ptr  (NslTensorDesc*)
            ctypes.c_int64,  # num_grad_inputs
        ]
        lib.nsl_model_backward.restype = ctypes.c_int64

    if hasattr(lib, "nsl_grad_context_destroy"):
        lib.nsl_grad_context_destroy.argtypes = [ctypes.c_int64]
        lib.nsl_grad_context_destroy.restype = None


def _bind_base_lifecycle_ffis(lib: ctypes.CDLL) -> None:
    """Bind the lifecycle + error-handling FFIs.

    These are present in both the standalone runtime and any
    ``@export``-emitted shared library (which statically links the
    runtime).
    """
    if hasattr(lib, "nsl_model_create"):
        lib.nsl_model_create.argtypes = [ctypes.c_int64]
        lib.nsl_model_create.restype = ctypes.c_int64
    if hasattr(lib, "nsl_model_destroy"):
        lib.nsl_model_destroy.argtypes = [ctypes.c_int64]
        lib.nsl_model_destroy.restype = ctypes.c_int64
    if hasattr(lib, "nsl_get_last_error"):
        lib.nsl_get_last_error.argtypes = []
        lib.nsl_get_last_error.restype = ctypes.c_int64
    if hasattr(lib, "nsl_clear_error"):
        lib.nsl_clear_error.argtypes = []
        lib.nsl_clear_error.restype = ctypes.c_int64
    if hasattr(lib, "nsl_model_num_weights"):
        lib.nsl_model_num_weights.argtypes = [ctypes.c_int64]
        lib.nsl_model_num_weights.restype = ctypes.c_int64
    if hasattr(lib, "nsl_model_get_version"):
        lib.nsl_model_get_version.argtypes = []
        lib.nsl_model_get_version.restype = ctypes.c_int64


def _fetch_last_error(lib: ctypes.CDLL) -> str:
    """Read the thread-local error string set by the NSL runtime."""
    if not hasattr(lib, "nsl_get_last_error"):
        return "<no nsl_get_last_error symbol>"
    err_ptr = lib.nsl_get_last_error()
    if not err_ptr:
        return "<no error message>"
    msg = ctypes.cast(err_ptr, ctypes.c_char_p).value
    if hasattr(lib, "nsl_clear_error"):
        lib.nsl_clear_error()
    return msg.decode("utf-8") if msg else "<empty error message>"


def _bind_named_dispatch_ffis(lib: ctypes.CDLL) -> None:
    """Bind the named-dispatch C API symbols on the supplied library handle.

    These FFIs may live in either the standalone runtime library OR in an
    emitted shared library (which statically links the runtime). The same
    binding helper is reused in both cases so the symbol-presence check is
    centralised.
    """
    for sym_name, argtypes, restype in (
        (
            "nsl_model_create_with_lib",
            [ctypes.c_int64, ctypes.c_int64],
            ctypes.c_int64,
        ),
        (
            "nsl_model_export_count",
            [ctypes.c_int64],
            ctypes.c_int64,
        ),
        (
            "nsl_model_call",
            [
                ctypes.c_int64,  # model_ptr
                ctypes.c_int64,  # name_ptr
                ctypes.c_int64,  # inputs_desc_ptr
                ctypes.c_int64,  # num_inputs
                ctypes.c_int64,  # outputs_desc_ptr
                ctypes.c_int64,  # num_outputs
            ],
            ctypes.c_int64,
        ),
        (
            "nsl_model_call_dlpack",
            [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
            ],
            ctypes.c_int64,
        ),
        (
            "nsl_model_lookup_function",
            [ctypes.c_int64, ctypes.c_int64],
            ctypes.c_int64,
        ),
    ):
        if not hasattr(lib, sym_name):
            # Optional symbols — older runtimes may pre-date the named-
            # dispatch FFIs. Leave them unbound; callers that need them
            # will fail at use time with a clearer error.
            continue
        fn = getattr(lib, sym_name)
        fn.argtypes = argtypes
        fn.restype = restype


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

    Two construction modes are supported:

    1. *Legacy weights-only mode* — the first argument is a path to a
       safetensors file and the standalone NSL runtime shared library
       (``libnsl_runtime.{so,dylib,dll}``) is used for dispatch::

           model = NslModel("gpt2.safetensors")
           output = model(input_tokens)  # forward()

    2. *Named-dispatch mode* — the first argument is a path to an
       ``@export``-emitted shared library and ``weights_path=`` provides
       the safetensors file. The model's export registry is populated
       eagerly by ``nsl_model_create_with_lib`` so that
       :meth:`call` can dispatch by name::

           model = NslModel("libmodel.so", weights_path="w.safetensors")
           y = model.call("forward", x)
    """

    def __init__(
        self,
        path: str | Path,
        lib_path: Optional[Path] = None,
        weights_path: Optional[str | Path] = None,
    ):
        # Initialise the always-present attributes first so __del__ stays
        # safe even if construction below fails halfway through.
        self._destroyed = True
        self._handle = 0
        self._lib: Optional[ctypes.CDLL] = None
        self._path = Path(path)
        self._owns_lib_handle = False

        if weights_path is not None:
            # Named-dispatch mode — load the emitted shared library
            # directly (it has the runtime statically linked in) and
            # eagerly populate the export registry.
            lib = ctypes.CDLL(str(path))
            _bind_named_dispatch_ffis(lib)
            # The base lifecycle/error FFIs we also need.
            _bind_base_lifecycle_ffis(lib)
            # Spec B per-call grad bridge FFIs — exported by the
            # @export-emitted shared library because the runtime is
            # statically linked.
            _bind_grad_context_ffis(lib)
            self._lib = lib
            self._owns_lib_handle = True

            if not hasattr(lib, "nsl_model_create_with_lib"):
                raise NslError(
                    f"Shared library '{path}' does not export "
                    "nsl_model_create_with_lib — rebuild against a runtime "
                    "with named-dispatch support."
                )
            weights_bytes = str(weights_path).encode("utf-8") + b"\x00"
            weights_buf = (ctypes.c_char * len(weights_bytes))(*weights_bytes)
            weights_ptr = ctypes.cast(weights_buf, ctypes.c_void_p).value or 0

            lib_bytes = str(path).encode("utf-8") + b"\x00"
            lib_buf = (ctypes.c_char * len(lib_bytes))(*lib_bytes)
            lib_ptr = ctypes.cast(lib_buf, ctypes.c_void_p).value or 0

            self._handle = lib.nsl_model_create_with_lib(weights_ptr, lib_ptr)
            if self._handle == 0:
                msg = _fetch_last_error(lib)
                raise NslError(
                    f"nsl_model_create_with_lib failed for weights='{weights_path}', "
                    f"lib='{path}': {msg}"
                )
            self._destroyed = False
            return

        # Legacy weights-only path — load the standalone runtime.
        if lib_path is not None:
            self._lib = _load_lib(lib_path)
        else:
            self._lib = _get_lib()

        path_bytes = str(path).encode("utf-8")
        path_buf = ctypes.create_string_buffer(path_bytes)
        path_ptr = ctypes.cast(path_buf, ctypes.c_void_p).value

        self._handle = self._lib.nsl_model_create(path_ptr)
        if self._handle == 0:
            raise NslError(f"Failed to load model from '{path}'")
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

    @property
    def export_count(self) -> int:
        """Number of ``@export`` functions registered with this model.

        Returns 0 when the model was created via the legacy weights-only
        constructor (no export registry was populated).
        """
        if not hasattr(self._lib, "nsl_model_export_count"):
            return 0
        return int(self._lib.nsl_model_export_count(self._handle))

    def call(self, name: str, *inputs):
        """Dispatch an ``@export``'d function by name.

        Args:
            name: The user-facing name of the ``@export`` (e.g. ``"forward"``,
                  ``"generate"``).
            *inputs: One or more input tensors. Each input may be a Python
                  sequence of floats (treated as a 1-D ``f32`` tensor) or any
                  object supporting ``__dlpack__`` / ``__array__``.

        Returns:
            The output as a Python ``list[float]`` (v1 — single-output exports
            of ``Tensor<[N], f32>`` returning their result through a
            caller-allocated ``NslTensorDesc`` slot).

        Raises:
            RuntimeError: if the export name is not in the registry, or if
                the dispatch returns a non-zero status code. The error
                message includes the runtime's thread-local error string,
                which names the missing export and lists available ones.
        """
        from nslpy._bridge import (
            build_input_descs,
            allocate_output_descs,
            read_f32_output_desc,
        )

        if self._handle == 0 or self._destroyed:
            raise NslError("NslModel.call: model is closed or invalid")
        if not hasattr(self._lib, "nsl_model_call"):
            raise NslError(
                "NslModel.call: nsl_model_call symbol not present in this "
                "library — rebuild against a runtime with named-dispatch support."
            )

        name_bytes = name.encode("utf-8") + b"\x00"
        name_buf = (ctypes.c_char * len(name_bytes))(*name_bytes)
        name_ptr = ctypes.cast(name_buf, ctypes.c_void_p).value or 0

        in_descs, in_keepalive = build_input_descs(inputs)
        n_in = len(inputs)
        inputs_ptr = ctypes.cast(in_descs, ctypes.c_void_p).value or 0 if n_in > 0 else 0

        n_out = 1
        out_descs, out_buffers = allocate_output_descs(n_out, in_descs, n_in)
        outputs_ptr = ctypes.cast(out_descs, ctypes.c_void_p).value or 0

        rc = self._lib.nsl_model_call(
            ctypes.c_int64(self._handle),
            ctypes.c_int64(name_ptr),
            ctypes.c_int64(inputs_ptr),
            ctypes.c_int64(n_in),
            ctypes.c_int64(outputs_ptr),
            ctypes.c_int64(n_out),
        )
        # Keep input buffers alive across the FFI call.
        _ = in_keepalive
        _ = out_buffers
        if rc != 0:
            msg = _fetch_last_error(self._lib)
            raise NslError(f"nsl_model_call('{name}') returned rc={rc}: {msg}")

        return read_f32_output_desc(out_descs[0])

    def close(self) -> None:
        """Explicitly destroy the underlying NSL model handle.

        Idempotent; safe to call multiple times. After calling
        :meth:`close`, further use of this object is undefined.
        """
        if not self._destroyed and self._handle and self._lib is not None:
            try:
                if hasattr(self._lib, "nsl_model_destroy"):
                    self._lib.nsl_model_destroy(self._handle)
            finally:
                self._destroyed = True
                self._handle = 0

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

    def forward_grad(self, *inputs):
        """Run a forward pass that records a gradient tape.

        Returns ``(output, ctx)`` where:
          - ``output`` is a Python ``list[float]`` (CPU f32, v1 — same
            convention as :meth:`call`).
          - ``ctx`` is a :class:`GradContext` wrapping the runtime's
            per-call ``*mut GradContext`` handle.  Pass ``ctx`` to
            :meth:`GradContext.backward` to compute parameter gradients.

        The model-level ``enable_grad`` toggle was removed in Spec B T8;
        every grad-tracked forward must go through this method.

        Raises:
            NslError: if the underlying ``nsl_model_forward_grad`` FFI
                returns non-zero (re-entry, missing weights, dispatch
                failure, etc.).
        """
        from nslpy._bridge import (
            build_input_descs,
            allocate_output_descs,
            read_f32_output_desc,
        )

        if self._handle == 0 or self._destroyed:
            raise NslError("NslModel.forward_grad: model is closed or invalid")
        if not hasattr(self._lib, "nsl_model_forward_grad"):
            raise NslError(
                "NslModel.forward_grad: nsl_model_forward_grad symbol not "
                "present in this library — rebuild against a runtime that "
                "exports the Spec B per-call grad FFIs."
            )

        in_descs, in_keepalive = build_input_descs(inputs)
        n_in = len(inputs)
        inputs_ptr = (
            ctypes.cast(in_descs, ctypes.c_void_p).value or 0 if n_in > 0 else 0
        )

        n_out = 1
        out_descs, out_buffers = allocate_output_descs(n_out, in_descs, n_in)
        outputs_ptr = ctypes.cast(out_descs, ctypes.c_void_p).value or 0

        ctx_slot = ctypes.c_int64(0)
        ctx_slot_addr = ctypes.addressof(ctx_slot)

        rc = self._lib.nsl_model_forward_grad(
            ctypes.c_int64(self._handle),
            ctypes.c_int64(inputs_ptr),
            ctypes.c_int64(n_in),
            ctypes.c_int64(outputs_ptr),
            ctypes.c_int64(n_out),
            ctypes.c_int64(ctx_slot_addr),
        )
        # Keep buffers alive across the FFI call.
        _ = in_keepalive
        _ = out_buffers
        if rc != 0:
            msg = _fetch_last_error(self._lib)
            raise NslError(f"nsl_model_forward_grad returned rc={rc}: {msg}")

        output = read_f32_output_desc(out_descs[0])
        ctx = GradContext(self._lib, ctx_slot.value)
        return output, ctx

    def __call__(self, *inputs):
        """Shorthand for forward()."""
        return self.forward(*inputs)

    def __del__(self):
        # Guard against partial __init__ (attributes may be missing) and
        # against the interpreter tearing down ctypes before we run.
        try:
            destroyed = getattr(self, "_destroyed", True)
            handle = getattr(self, "_handle", 0)
            lib = getattr(self, "_lib", None)
        except Exception:
            return
        if not destroyed and handle and lib is not None:
            try:
                if hasattr(lib, "nsl_model_destroy"):
                    lib.nsl_model_destroy(handle)
            except Exception:
                # Best-effort cleanup; suppress shutdown-time errors.
                pass
            self._destroyed = True
            self._handle = 0

    def __repr__(self) -> str:
        n = self.num_weights if not self._destroyed else "?"
        return f"NslModel(path='{self._path}', weights={n})"


# ---------------------------------------------------------------------------
# GradContext — Python wrapper for a per-call runtime grad-context handle
# ---------------------------------------------------------------------------


class GradContext:
    """A Python wrapper around the runtime's ``*mut GradContext`` handle.

    Produced by :meth:`NslModel.forward_grad`.  Consumed by exactly one
    successful :meth:`backward` call (double-backward raises) and
    optionally explicitly freed via :meth:`destroy`. ``__del__``
    best-effort destroys the ctx on garbage-collection.

    The runtime validates a 4-byte magic header on every FFI entry so
    using a stale or wrong-type handle returns ``-1`` with a typed
    error (or, in :meth:`destroy`'s case, is a silent no-op) rather
    than UB.
    """

    __slots__ = ("_lib", "_handle", "_consumed", "_destroyed")

    def __init__(self, lib: ctypes.CDLL, handle: int):
        self._lib = lib
        self._handle = int(handle)
        self._consumed = False
        self._destroyed = False

    @property
    def _handle_int(self) -> int:
        """Raw integer ctx handle (for tests + low-level callers)."""
        return self._handle

    def backward(self, grad_output=None):
        """Replay the recorded tape and return per-parameter gradients.

        Args:
            grad_output: Reserved for v2 explicit upstream-gradient
                seeding. v1 seeds with ``ones_like(loss)`` internally
                (scalar-loss convention).

        Returns:
            ``dict[str, list[float]]`` mapping parameter slot (currently
            indexed as ``"param_<i>"``; the runtime does not yet thread
            name metadata through) to its f32 CPU gradient as a
            Python list.

        Raises:
            NslError: if the context has already been consumed (double
                backward), has been destroyed, or if the FFI returns
                non-zero for any other reason.
        """
        if self._destroyed or self._handle == 0:
            raise NslError("GradContext.backward: context is destroyed")

        from nslpy._bridge import NslTensorDesc

        # Caller-allocated grad-input desc array. v1: one desc per
        # parameter slot; we don't know the param count up-front, so
        # allocate a generous fixed-size slab and trust the runtime to
        # fill only the parameters it actually has (see Spec B §4.3 —
        # only the first `num_grad_inputs` descs are written).
        max_grad_inputs = 64
        grad_descs = (NslTensorDesc * max_grad_inputs)()
        grad_descs_ptr = ctypes.cast(grad_descs, ctypes.c_void_p).value or 0

        # v1: grad_outputs are ignored by the runtime (scalar-loss seed
        # is computed internally); we keep the signature flexible for
        # the future v2 explicit-seed path but pass null+0 today.
        _ = grad_output

        rc = self._lib.nsl_model_backward(
            ctypes.c_int64(self._handle),
            ctypes.c_int64(0),
            ctypes.c_int64(0),
            ctypes.c_int64(grad_descs_ptr),
            ctypes.c_int64(max_grad_inputs),
        )
        if rc != 0:
            msg = _fetch_last_error(self._lib)
            if "already consumed" in msg:
                # Raise with a stable substring for tests to match on.
                raise RuntimeError(
                    f"GradContext.backward: context already consumed: {msg}"
                )
            raise NslError(
                f"nsl_model_backward returned rc={rc}: {msg}"
            )
        self._consumed = True

        # Read out grads. Walk descs until we hit one with null data
        # (sentinel for "runtime didn't fill this slot"); skip nulls
        # to keep the dict semantically aligned with the runtime's
        # actual param list.
        from nslpy._bridge import read_f32_output_desc

        grads: dict[str, list[float]] = {}
        for i in range(max_grad_inputs):
            desc = grad_descs[i]
            if not desc.data:
                continue
            try:
                grads[f"param_{i}"] = read_f32_output_desc(desc)
            except (ValueError, Exception):
                # Degenerate desc — skip rather than blowing up the
                # whole backward call.
                continue
        return grads

    def destroy(self) -> None:
        """Explicitly release the underlying ctx Box.

        Idempotent. Safe to call after a successful backward (which
        does NOT auto-destroy — the Spec B contract is that backward
        consumes-but-keeps the ctx until destroy clears it).
        """
        if self._destroyed or self._handle == 0:
            return
        try:
            if hasattr(self._lib, "nsl_grad_context_destroy"):
                self._lib.nsl_grad_context_destroy(
                    ctypes.c_int64(self._handle)
                )
        finally:
            self._destroyed = True
            self._handle = 0

    def __del__(self):
        try:
            destroyed = getattr(self, "_destroyed", True)
            handle = getattr(self, "_handle", 0)
            lib = getattr(self, "_lib", None)
        except Exception:
            return
        if not destroyed and handle and lib is not None:
            try:
                if hasattr(lib, "nsl_grad_context_destroy"):
                    lib.nsl_grad_context_destroy(ctypes.c_int64(handle))
            except Exception:
                pass

    def __repr__(self) -> str:
        state = "destroyed" if self._destroyed else (
            "consumed" if self._consumed else "live"
        )
        return f"GradContext(handle=0x{self._handle:x}, {state})"


# End of nslpy._core — NslModel and GradContext are defined above.
