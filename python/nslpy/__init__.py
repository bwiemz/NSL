"""nslpy — Python bindings for the NeuralScript (NSL) compiler and runtime.

Usage:
    import nslpy

    # Load a compiled NSL model
    model = nslpy.NslModel("path/to/model.nslm")

    # Forward pass with torch tensors (zero-copy via DLPack when safe)
    output = model(input_tensor)

    # With autograd gradient bridge
    output = nslpy.autograd.nsl_forward(model, input_tensor)
    output.backward()  # gradients flow through NSL and torch
"""

__version__ = "0.2.0"

from nslpy._core import NslModel, NslError, find_library
from nslpy._bridge import to_nsl_tensor, from_nsl_tensor

__all__ = ["NslModel", "NslError", "find_library", "to_nsl_tensor", "from_nsl_tensor"]
