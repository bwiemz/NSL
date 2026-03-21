"""Gradient bridge: NSL backward ↔ torch.autograd.

Enables ``loss = nsl_model(input); loss.backward()`` with gradients
flowing through both NSL's tape-based AD and PyTorch's autograd engine.

Usage::

    model = nslpy.NslModel("model.safetensors")
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


if HAS_TORCH:
    class NslFunction(torch.autograd.Function):
        """Custom autograd function that bridges NSL forward/backward with PyTorch.

        The forward pass runs the NSL model and saves context for backward.
        The backward pass calls NSL's gradient computation and returns
        gradients compatible with torch's autograd tape.
        """

        @staticmethod
        def forward(ctx: Any, model: Any, *inputs: torch.Tensor) -> torch.Tensor:
            """Run NSL model forward pass.

            Args:
                ctx: Autograd context for saving tensors.
                model: NslModel instance.
                *inputs: Input tensors.

            Returns:
                Output tensor from the NSL model.
            """
            ctx.model = model
            ctx.save_for_backward(*inputs)
            ctx.num_inputs = len(inputs)

            # Run NSL forward pass
            output = model.forward(*inputs)
            if not isinstance(output, torch.Tensor):
                # If the model returns a raw pointer, convert it
                from nslpy._bridge import _dlpack_to_torch
                if isinstance(output, int):
                    output = _dlpack_to_torch(output)
                elif isinstance(output, tuple):
                    output = output[0]
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
            """Compute gradients via NSL backward pass.

            Args:
                ctx: Autograd context with saved tensors.
                grad_output: Gradient of the loss w.r.t. the output.

            Returns:
                (None, grad_input_0, grad_input_1, ...) — None for the model arg.
            """
            inputs = ctx.saved_tensors
            model = ctx.model

            try:
                grad_inputs = model.backward(grad_output, inputs)
            except Exception:
                # If NSL backward is not implemented, return None gradients
                return (None,) + (None,) * ctx.num_inputs

            # Pad with None for inputs that don't have gradients
            result = [None]  # None for the model parameter
            for i in range(ctx.num_inputs):
                if i < len(grad_inputs) and grad_inputs[i] is not None:
                    result.append(grad_inputs[i])
                else:
                    result.append(None)

            return tuple(result)

    def nsl_forward(model: Any, *inputs: torch.Tensor) -> torch.Tensor:
        """Run an NSL model with torch.autograd gradient tracking.

        This wraps the model call in a custom autograd Function so that
        ``loss.backward()`` will call NSL's backward pass.

        Args:
            model: NslModel instance.
            *inputs: Input tensors (torch.Tensor).

        Returns:
            Output tensor with autograd graph attached.

        Example::

            model = nslpy.NslModel("weights.safetensors")
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
