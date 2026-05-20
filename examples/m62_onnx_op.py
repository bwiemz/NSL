"""M62b Spec C — NSL @export shown up as an ONNX Runtime custom op.

Prerequisites
-------------
1. Build nsl-cli with the ONNX-RT custom-op feature flag (one-time):

       cargo build --release -p nsl-cli --features onnx-rt-op

   `nsl build --shared-lib` itself does NOT accept a `--features` flag — the
   feature is a Cargo-time decision baked into the nsl-cli binary. When the
   feature is on, the .so/.dll produced by `nsl build --shared-lib` re-exports
   the runtime's `RegisterCustomOps` symbol, which is what ONNX Runtime's
   `SessionOptions.register_custom_ops_library` looks up at load time.

2. Compile this example to a shared library:

       ./target/release/nsl build --shared-lib examples/m62_onnx_op.nsl \\
           -o /tmp/m62_onnx.so           # Linux / macOS
       ./target/release/nsl build --shared-lib examples/m62_onnx_op.nsl \\
           -o C:/tmp/m62_onnx.dll        # Windows

3. Run this harness:

       python examples/m62_onnx_op.py /tmp/m62_onnx.so
"""

import sys

import numpy as np

import onnxruntime as ort
from onnx import helper, TensorProto

from nslpy.onnxrt import register_nsl_provider, make_onnx_node


def main(lib_path: str) -> None:
    # 1. Register the NSL .so/.dll as an ORT custom-ops library.
    sess_opts = ort.SessionOptions()
    register_nsl_provider(sess_opts, lib_path)

    # 2. Build a 1-node graph that calls the NSL "add" op.
    node = make_onnx_node("add", inputs=["a", "b"], outputs=["c"])
    graph = helper.make_graph(
        nodes=[node],
        name="nsl_add_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [4]),
        ],
        outputs=[
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [4]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("com.nsl", 1)],
    )

    # 3. Run the session through ORT.
    sess = ort.InferenceSession(model.SerializeToString(), sess_opts)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    out = sess.run(["c"], {"a": a, "b": b})

    print(f"NSL custom op result: {out[0].tolist()}")
    assert out[0].tolist() == [11.0, 22.0, 33.0, 44.0], (
        f"unexpected output: {out[0].tolist()}"
    )
    print("ORT-registered NSL custom op verified end-to-end.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "usage: python examples/m62_onnx_op.py <path-to-libm62_onnx.so>",
            file=sys.stderr,
        )
        sys.exit(2)
    main(sys.argv[1])
