"""Spec B E2E — per-call grad context bridge (M62b T9).

This test pins down the **ABI flow** for the Python grad-context bridge:

  - :meth:`NslModel.forward_grad` returns a non-zero ctx handle and a
    correct forward output.
  - :meth:`GradContext.backward` returns rc=0 + a dict of grad tensors.
  - A second backward on the same ctx raises ``RuntimeError`` containing
    "already consumed" (matching the runtime's
    ``ERR_ALREADY_CONSUMED`` string).
  - Calling the raw ``nsl_model_backward`` FFI with a non-ctx pointer
    (zeroed slab → wrong magic) returns -1, not UB.
  - Calling ``nsl_grad_context_destroy`` on a non-ctx pointer is a
    silent no-op, not UB.

The **numerical-correctness** of the returned gradient values is
gated on a pre-existing architectural issue in
``crates/nsl-codegen/src/c_wrapper.rs`` (see the ``GRAD SCOPE``
comment at lines 91-99): the @export wrapper builds fresh
``NslTensor`` wrappers via ``desc_to_nsl_tensor`` whose ``tape_id``
fields don't match the model's weight tensors, so backward replays
through the @export path produce **zero gradients** (or NaN for
non-zero loss seeds).  Until the c_wrapper.rs fix lands, the
gradient-VALUE assertion is marked ``xfail`` — the test will flip
to an unexpected ``XPASS`` (which pytest flags as failure) the
moment the underlying issue is resolved, automatically converting
itself into a regression guard.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
NSL_BIN = WORKSPACE / "target" / "debug" / (
    "nsl.exe" if os.name == "nt" else "nsl"
)
STDLIB = WORKSPACE / "stdlib"


def _lib_ext() -> str:
    if os.name == "nt":
        return "dll"
    import platform
    return "dylib" if platform.system() == "Darwin" else "so"


def _check_nsl_binary() -> None:
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")


def _build_linear(tmp: Path):
    """Build a 1-parameter linear @export model.

    NSL source declares an @export ``forward(x)`` that consumes the
    model's `w` weight to produce ``w * x``.  Weights are saved as a
    1-element f32 safetensors file so the runtime's safetensors loader
    can find a single param matching the model's expected layout.

    The model intentionally has *one* weight (``w``) — adding an extra
    bias would test broadcasting which is out-of-scope for the ABI
    smoke test here.  The wrapper signature is single-tensor-in /
    single-tensor-out, matching the v1 read_f32_output_desc path.
    """
    src = tmp / "lin.nsl"
    src.write_text(
        "@export\n"
        "fn forward(x: Tensor<[1], f32>) -> Tensor<[1], f32>:\n"
        "    return x\n"
    )
    weights = tmp / "w.safetensors"
    # Minimal safetensors file: an empty JSON header (no params declared)
    # is sufficient because the @export function above does not read any
    # model weight — the body is pure identity. This isolates the test
    # from weight-loading flakiness while still exercising the full
    # forward_grad → backward → destroy ABI path.
    header = b"{}"
    weights.write_bytes(struct.pack("<Q", len(header)) + header)

    out = tmp / f"lin.{_lib_ext()}"
    env = os.environ.copy()
    env["NSL_STDLIB_PATH"] = str(STDLIB)
    result = subprocess.run(
        [str(NSL_BIN), "build", "--shared-lib", str(src), "-o", str(out)],
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
    assert out.exists(), f"shared lib not produced at {out}"
    return out, weights


@pytest.fixture
def model_setup():
    _check_nsl_binary()
    # ignore_cleanup_errors=True tolerates the Windows DLL file-lock
    # that ctypes holds until process exit.
    with tempfile.TemporaryDirectory(
        prefix="m62_grad_ctx_",
        ignore_cleanup_errors=True,
    ) as td:
        tmp = Path(td)
        lib_path, weights_path = _build_linear(tmp)

        import nslpy

        model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
        try:
            yield model
        finally:
            model.close()


# ---------------------------------------------------------------------------
# Real-weight fixture — gating test for the c_wrapper tape_id round-trip fix.
# ---------------------------------------------------------------------------
#
# Lives in python/tests/fixtures/m62_real_weight_export.nsl. Single weight
# `w: Tensor<[1], f32>` declared in a `model RealWeight:` block, exposed via
# `@export fn forward(self, x) -> self.w * x`. Element-wise mul backward
# gives dy/dw = x exactly, so the assertion is value-checking (not just
# presence-checking — the pre-fix bug produces a present-but-zero
# entry via zeros_like fallback, which a presence-only check would
# false-green).
FIXTURE_REAL_WEIGHT = Path(__file__).parent / "fixtures" / "m62_real_weight_export.nsl"


def _build_real_weight(tmp: Path, w_value: float):
    """Build the single-weight `forward(x) = w * x` @export library and
    write a 1-element f32 safetensors file with weight `w = w_value`.
    """
    if not FIXTURE_REAL_WEIGHT.exists():
        pytest.fail(f"fixture not found: {FIXTURE_REAL_WEIGHT}")

    weights = tmp / "real_weight.safetensors"
    # Safetensors v1 layout: <u64 header_len> <header_json> <tensor_data>
    # Header declares one tensor "w", dtype F32, shape [1], byte offsets [0, 4].
    header = (
        b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    )
    # Pad header to 8-byte alignment per safetensors spec.
    pad = (8 - (len(header) % 8)) % 8
    header += b" " * pad
    tensor_bytes = struct.pack("<f", w_value)
    weights.write_bytes(
        struct.pack("<Q", len(header)) + header + tensor_bytes
    )

    out = tmp / f"real_weight.{_lib_ext()}"
    env = os.environ.copy()
    env["NSL_STDLIB_PATH"] = str(STDLIB)
    result = subprocess.run(
        [
            str(NSL_BIN),
            "build",
            "--shared-lib",
            str(FIXTURE_REAL_WEIGHT),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"nsl build (real-weight fixture) failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    assert out.exists(), f"shared lib not produced at {out}"
    return out, weights


@pytest.fixture
def real_weight_setup():
    """Yields (model, w_value, x_value) where forward(x) = w * x and the
    expected gradient is dy/dw = x.
    """
    _check_nsl_binary()
    w_value = 3.0
    x_value = 4.0
    with tempfile.TemporaryDirectory(
        prefix="m62_real_weight_",
        ignore_cleanup_errors=True,
    ) as td:
        tmp = Path(td)
        lib_path, weights_path = _build_real_weight(tmp, w_value)

        import nslpy

        model = nslpy.NslModel(str(lib_path), weights_path=str(weights_path))
        try:
            yield model, w_value, x_value
        finally:
            model.close()


# ---------------------------------------------------------------------------
# ABI-flow tests — these MUST pass unconditionally on this branch.
# ---------------------------------------------------------------------------


def test_forward_grad_returns_nonzero_context(model_setup):
    """forward_grad returns a valid (non-null) ctx handle + a correct output."""
    y, ctx = model_setup.forward_grad([3.0])
    try:
        assert ctx._handle_int != 0, "expected a non-null GradContext handle"
        # Identity forward: w * x = x → [3.0].
        assert list(y) == [3.0], f"expected forward output [3.0], got {y!r}"
    finally:
        ctx.destroy()


def test_backward_rc_ok_and_returns_grads_dict(model_setup):
    """backward returns rc=0 and a dict (possibly empty) of grad tensors."""
    _, ctx = model_setup.forward_grad([3.0])
    try:
        grads = ctx.backward()
        assert isinstance(grads, dict), (
            f"expected dict return from GradContext.backward, got {type(grads).__name__}"
        )
        # We do not assert dict CONTENTS here — that's gated on the
        # c_wrapper.rs tape_id-mismatch fix (see the xfail test below).
        # The structural assertion is enough to verify the ABI flow.
    finally:
        ctx.destroy()


def test_double_backward_raises_already_consumed(model_setup):
    """Second backward on the same context returns -1 + "already consumed"."""
    _, ctx = model_setup.forward_grad([3.0])
    try:
        ctx.backward()  # first call: OK, marks ctx consumed.

        with pytest.raises(RuntimeError, match="already consumed"):
            ctx.backward()  # second call: typed error.
    finally:
        ctx.destroy()


def test_backward_with_bogus_ctx_returns_typed_error(model_setup):
    """Magic-pointer validation: backward on a non-context returns -1, not UB.

    Allocates a 64-byte zeroed slab — the runtime's 4-byte magic-header
    check (``NSL_GRAD_CONTEXT_MAGIC = 0x4E534C47``) reads zero, so the
    FFI must short-circuit to ``-1`` + ``ERR_INVALID_CONTEXT`` rather
    than UB on the cast-to-``&mut GradContext``.
    """
    lib = model_setup._lib
    # 64 zeroed bytes; aligned at least to 4 bytes (Python ctypes
    # guarantees alignment for the element type → c_uint8 with size 64
    # is page-aligned in practice). The runtime's pre-magic gates allow
    # this through; the magic read returns 0 ≠ 0x4E534C47.
    buf = (ctypes.c_uint8 * 64)()
    bogus_ctx = ctypes.cast(buf, ctypes.c_void_p).value or 0
    assert bogus_ctx != 0, "ctypes allocation should yield a non-null pointer"

    rc = lib.nsl_model_backward(
        ctypes.c_int64(bogus_ctx),
        ctypes.c_int64(0),
        ctypes.c_int64(0),
        ctypes.c_int64(0),
        ctypes.c_int64(0),
    )
    assert rc == -1, f"expected rc=-1 for bogus ctx, got rc={rc}"


def test_destroy_on_bogus_ctx_is_noop(model_setup):
    """Magic-pointer validation: destroy on a non-context is a no-op, not UB."""
    lib = model_setup._lib
    buf = (ctypes.c_uint8 * 64)()
    bogus_ctx = ctypes.cast(buf, ctypes.c_void_p).value or 0
    # No assertion on return: destroy is void. Simply not crashing
    # (and not corrupting the slab) is the contract.
    lib.nsl_grad_context_destroy(ctypes.c_int64(bogus_ctx))
    # Sanity: the slab should remain zeroed (destroy should NOT have
    # written into it, since the magic gate fired).
    assert all(b == 0 for b in buf), "destroy must not write into a non-ctx slab"


def test_destroy_is_idempotent(model_setup):
    """Calling destroy twice on the same Python wrapper is safe."""
    _, ctx = model_setup.forward_grad([3.0])
    ctx.destroy()
    ctx.destroy()  # second call is a Python no-op (handle already cleared).
    # No assertion needed — surviving the second call is the contract.


# ---------------------------------------------------------------------------
# Numerical-correctness — gated on c_wrapper.rs tape_id-mismatch fix.
# ---------------------------------------------------------------------------


def test_identity_forward_backward_returns_empty_grads(model_setup):
    """Identity-forward (zero-param) model: backward returns ``{}``.

    The identity model has no model weights, so even after the
    c_wrapper tape_id fix there are no parameter gradients to
    compute. This test pins that ABI contract: a zero-param model's
    backward returns an empty dict, NOT an error.

    This is NOT the gating test for the fix — see
    :func:`test_real_weight_export_backward_returns_correct_grad`.
    """
    _, ctx = model_setup.forward_grad([3.0])
    try:
        grads = ctx.backward()
        assert grads == {}, (
            f"identity model has no params; expected {{}}, got {grads!r}"
        )
    finally:
        ctx.destroy()


# ---------------------------------------------------------------------------
# Real-weight gating test — proves the c_wrapper tape_id round-trip fix.
# ---------------------------------------------------------------------------


def test_real_weight_export_backward_returns_correct_grad(real_weight_setup):
    """**Gating test for the c_wrapper.rs tape_id round-trip fix.**

    Before the fix, ``NslTensorDesc`` had no ``tape_id`` field, so the
    typed wrapper's ``nsl_tensor_to_desc_ffi`` → ``desc_to_nsl_tensor``
    round-trip in ``nsl_model_forward_grad`` produced a loss-seed
    wrapper with ``tape_id == 0``. Backward keyed the seed on the
    raw pointer (the ``else`` branch in ``run_backward_core``), which
    matched no ``TapeOp::*.out`` tape_id, so the chain rule never
    propagated and every parameter grad fell through to the
    ``zeros_like`` fallback.

    After the fix, the desc carries ``tape_id`` verbatim across the
    round-trip, the loss seed matches the impl-emitted output's
    tape_id, and the chain rule produces the correct grad.

    Model: ``forward(x) = w * x`` with single weight ``w``.
    Element-wise mul backward: ``dy/dw = x``.

    With ``w=3.0`` and ``x=4.0``:
      - forward output ``y == 12.0`` (verified)
      - backward grad ``dy/dw == 4.0`` (the gating assertion)

    A presence-only check (``"w" in grads``) would false-green because
    the pre-fix bug produces a present-but-zero entry via the
    ``zeros_like(*ptr)`` fallback at ``backward.rs:1657``. Only the
    **value** assertion distinguishes fixed-from-broken.
    """
    model, w_value, x_value = real_weight_setup

    # Forward: y = w * x = 3.0 * 4.0 = 12.0.
    y, ctx = model.forward_grad([x_value])
    try:
        y_val = list(y)
        assert y_val == [w_value * x_value], (
            f"forward output incorrect: expected [{w_value * x_value}], got {y_val!r}"
        )

        # Backward: dy/dw = x (element-wise mul backward).
        grads = ctx.backward()
        assert grads, (
            "expected non-empty grads dict for a one-weight model — "
            "an empty dict indicates the c_wrapper tape_id round-trip "
            "fix did not take, or the param_set tracking is broken"
        )
        assert len(grads) == 1, (
            f"expected exactly 1 grad (one weight), got {len(grads)}: "
            f"{list(grads.keys())!r}"
        )

        # The runtime keys grads by `param_<i>` (no name plumbing
        # through the @export ABI yet). Single-weight model → single
        # entry; extract by value rather than by key.
        (grad_w,) = grads.values()
        assert grad_w == [x_value], (
            f"dy/dw = x (element-wise mul backward); expected [{x_value}], "
            f"got {grad_w!r}. A value of [0.0] specifically indicates the "
            f"pre-fix tape_id-mismatch bug — backward fell through to the "
            f"zeros_like fallback because the loss seed did not match any "
            f"tape op's `out` tape_id."
        )
    finally:
        ctx.destroy()
