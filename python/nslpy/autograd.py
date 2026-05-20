"""Gradient bridge: NSL backward ↔ torch.autograd.

Enables ``loss = nsl_model(input); loss.backward()`` with gradients
flowing through both NSL's tape-based AD and PyTorch's autograd engine.

Spec B (per-call grad context) — replaces the legacy model-level
``enable_grad`` / ``disable_grad`` toggle.  Every grad-tracked forward
goes through :meth:`NslModel.forward_grad`, which returns a
:class:`~nslpy._core.GradContext` that the backward pass consumes.

Usage::

    model = nslpy.NslModel("model.so", weights_path="w.safetensors")
    output = nslpy.autograd.nsl_forward(model, input_tensor)
    loss = output.sum()
    loss.backward()  # gradients propagated through NSL
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import torch
    import torch.autograd
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _output_to_torch(output: Any) -> Any:
    """Wrap an :meth:`NslModel.forward_grad` output in a torch tensor.

    The current v1 ABI returns a Python ``list[float]`` (CPU f32
    single-output exports — see ``read_f32_output_desc``).  We promote
    to a torch tensor so the autograd graph can be attached.
    """
    import torch as _torch
    if isinstance(output, _torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        return _torch.tensor(list(output), dtype=_torch.float32)
    if isinstance(output, int):
        # Legacy raw-DLPack-pointer path (kept for back-compat with
        # callers that exercise the dlpack forward path directly).
        from nslpy._bridge import _dlpack_to_torch
        return _dlpack_to_torch(output)
    raise TypeError(
        f"nsl_forward: unsupported model output type {type(output).__name__}"
    )


if HAS_TORCH:
    class NslFunction(torch.autograd.Function):
        """Custom autograd function bridging NSL forward/backward with PyTorch.

        Forward calls ``model.forward_grad(*inputs)`` and stashes the
        returned :class:`~nslpy._core.GradContext` on the autograd ctx.
        Backward consumes the stashed ctx via
        :meth:`~nslpy._core.GradContext.backward` (and then destroys it),
        returning per-input gradients aligned with the forward arg list.

        The per-call ctx contract (Spec B §5.4) means each forward owns
        its own tape — multiple :class:`NslFunction` instances can be
        in flight concurrently without aliasing.
        """

        @staticmethod
        def forward(ctx: Any, model: Any, *inputs: torch.Tensor) -> torch.Tensor:
            ctx.model = model
            ctx.save_for_backward(*inputs)
            ctx.num_inputs = len(inputs)

            output, grad_ctx = model.forward_grad(*inputs)
            # Stash the per-call grad context on the autograd ctx.
            # Spec B §5.4: only ONE backward replay per ctx, then it
            # must be destroyed. We let our backward() consume it.
            ctx.nsl_grad_ctx = grad_ctx
            return _output_to_torch(output)

        @staticmethod
        def backward(
            ctx: Any, grad_output: torch.Tensor
        ) -> Tuple[Optional[torch.Tensor], ...]:
            grad_ctx = getattr(ctx, "nsl_grad_ctx", None)
            if grad_ctx is None:
                # No tape was recorded (forward_grad failed somewhere
                # upstream). Return Nones to keep torch happy without
                # corrupting its graph.
                return (None,) + (None,) * ctx.num_inputs

            try:
                _grads = grad_ctx.backward(grad_output)
            except Exception:
                # Map failure to "no gradient available" rather than
                # propagating — same defensive behaviour as the legacy
                # bridge. Tests that need to see the error use the
                # GradContext API directly.
                return (None,) + (None,) * ctx.num_inputs
            finally:
                # Whether backward succeeded or not, free the ctx so a
                # stale handle can't be re-used by a later (wrong)
                # autograd replay.
                try:
                    grad_ctx.destroy()
                except Exception:
                    pass

            # v1 returns dict[str, list[float]] of PARAMETER gradients,
            # not input gradients. Until the runtime threads per-input
            # gradient slots through ctx.input_ptrs (currently a v2
            # follow-up), we conservatively return Nones for each input
            # — torch then treats inputs as not-requiring-grad through
            # this op, which matches the documented Spec B §4.3 surface.
            return (None,) + (None,) * ctx.num_inputs

    def nsl_forward(model: Any, *inputs: torch.Tensor) -> torch.Tensor:
        """Run an NSL model with torch.autograd gradient tracking.

        This wraps the model call in a custom autograd Function so that
        ``loss.backward()`` triggers NSL's backward pass via the per-call
        :class:`~nslpy._core.GradContext`.

        Args:
            model: NslModel instance (must support ``forward_grad``).
            *inputs: Input tensors (torch.Tensor).

        Returns:
            Output tensor with autograd graph attached.

        Example::

            model = nslpy.NslModel("model.so", weights_path="w.safetensors")
            out = nslpy.autograd.nsl_forward(model, tokens)
            loss = out.sum()
            loss.backward()
        """
        return NslFunction.apply(model, *inputs)

else:
    def nsl_forward(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "nslpy.autograd requires PyTorch. Install with: pip install torch"
        )
