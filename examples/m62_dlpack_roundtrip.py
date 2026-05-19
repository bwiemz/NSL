"""Demonstrate named dispatch + multi-export against a compiled NSL model.

Usage:
    cargo build --release -p nsl-cli
    # Linux/macOS:
    ./target/release/nsl build --shared-lib examples/m62_dlpack_roundtrip.nsl \\
        -o /tmp/m62_rt.so
    python examples/m62_dlpack_roundtrip.py /tmp/m62_rt.so
    # Windows (PowerShell):
    .\\target\\release\\nsl.exe build --shared-lib examples\\m62_dlpack_roundtrip.nsl `
        -o $env:TEMP\\m62_rt.dll
    python examples\\m62_dlpack_roundtrip.py $env:TEMP\\m62_rt.dll

Note: the artifact extension is platform-dependent (.so / .dylib / .dll). This
harness accepts whichever path you produced and doesn't assume an extension.
"""
import sys
import tempfile
from pathlib import Path


def main(lib_path: str) -> None:
    import nslpy

    # An empty safetensors header satisfies the runtime's weight loader
    # for this stateless example (no learnable parameters).
    with tempfile.TemporaryDirectory() as td:
        weights = Path(td) / "empty.safetensors"
        weights.write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")

        model = nslpy.NslModel(lib_path, weights_path=str(weights))
        try:
            x = [1.0, 2.0, 3.0, 4.0]
            out_identity = list(model.call("identity", x))
            out_double = list(model.call("double", x))

            print(f"identity({x}) = {out_identity}")
            print(f"double({x}) = {out_double}")

            assert out_identity == [1.0, 2.0, 3.0, 4.0], (
                f"identity output mismatch: {out_identity}"
            )
            assert out_double == [2.0, 4.0, 6.0, 8.0], (
                f"double output mismatch: {out_double}"
            )
            print("Named dispatch verified.")
        finally:
            model.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
