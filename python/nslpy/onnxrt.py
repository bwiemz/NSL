"""ONNX Runtime integration — register NSL-compiled subgraphs as custom ops.

Usage::

    import onnxruntime as ort
    from nslpy.onnxrt import register_nsl_provider

    sess_opts = ort.SessionOptions()
    register_nsl_provider(sess_opts, "model.nslm")

    session = ort.InferenceSession("model.onnx", sess_opts)
    output = session.run(None, {"input": data})
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def register_nsl_provider(
    session_options,
    model_path: str | Path,
    *,
    device: str = "cpu",
) -> None:
    """Register an NSL-compiled model as an ONNX Runtime execution provider.

    Args:
        session_options: ``onnxruntime.SessionOptions`` instance.
        model_path: Path to the NSL compiled shared library (.so/.dll/.dylib).
        device: Target device ("cpu" or "cuda:N").

    Note:
        This is a stub for M62b — full ONNX Runtime custom op registration
        requires implementing the ORT C API for custom execution providers.
        Currently, this registers the shared library for custom op dispatch.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"NSL model library not found: {path}")

    session_options.register_custom_ops_library(str(path))
