"""ONNX Runtime integration — register NSL-compiled @export functions
as ORT custom ops in the com.nsl domain.

M62b Spec C. Requires the NSL shared library to be built with
``--features onnx-rt-op`` so the ``RegisterCustomOps`` entry point is
exported.

Usage::

    import onnxruntime as ort
    import numpy as np
    from nslpy.onnxrt import register_nsl_provider, make_onnx_node
    from onnx import helper, TensorProto

    sess_opts = ort.SessionOptions()
    register_nsl_provider(sess_opts, "/path/to/libmodel.so")

    node = make_onnx_node("forward", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        nodes=[node],
        name="nsl_forward_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("com.nsl", 1)],
    )

    sess = ort.InferenceSession(model.SerializeToString(), sess_opts)
    out = sess.run(["y"], {"x": np.array([3.0], dtype=np.float32)})
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def register_nsl_provider(session_options, lib_path: str | Path) -> None:
    """Register an NSL-compiled shared library as an ORT custom-ops source.

    The library must have been built with ``--features onnx-rt-op``
    (Spec C), which adds the ``RegisterCustomOps`` entry point ORT uses
    to discover the ``com.nsl`` domain and its operators.

    Args:
        session_options: ``onnxruntime.SessionOptions`` instance.
        lib_path: Path to the NSL compiled shared library.

    Raises:
        FileNotFoundError: If ``lib_path`` does not exist.
    """
    path = Path(lib_path)
    if not path.exists():
        raise FileNotFoundError(f"NSL model library not found: {path}")
    session_options.register_custom_ops_library(str(path))


def make_onnx_node(
    export_name: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    *,
    domain: str = "com.nsl",
    name: str | None = None,
):
    """Construct an ``onnx.NodeProto`` that invokes an NSL ``@export`` as a
    custom op.

    Args:
        export_name: NSL function name (the ``@export`` symbol).
        inputs: Input tensor names in the ONNX graph.
        outputs: Output tensor names in the ONNX graph.
        domain: Custom-op domain. Defaults to ``com.nsl``.
        name: Optional node name; defaults to ``export_name``.

    Returns:
        An ``onnx.NodeProto`` ready to insert into a graph.
    """
    from onnx import helper

    return helper.make_node(
        op_type=export_name,
        inputs=list(inputs),
        outputs=list(outputs),
        domain=domain,
        name=name or export_name,
    )
