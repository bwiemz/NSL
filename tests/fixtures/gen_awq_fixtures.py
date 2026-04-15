"""Generate AWQ calibration fixture files for the end-to-end integration test.

Produces:
  tests/fixtures/awq_calib_data.safetensors  — [8, 4, 64] f32 tensor (key: "calibration")
  tests/fixtures/awq_calib_weights.safetensors — two weight tensors for TinyMLP

Run from the repo root:
    python tests/fixtures/gen_awq_fixtures.py
"""

import os
import numpy as np
from safetensors.numpy import save_file

np.random.seed(42)

# Calibration data: rank-3 [count=8, seq=4, hidden=64]
calib = np.random.randn(8, 4, 64).astype(np.float32)
# Ensure channels aren't all-equal (non-trivial variation across dim-2).
calib[:, :, 0] *= 10.0   # channel 0 dominates
calib[:, :, 32] *= 5.0   # channel 32 also elevated

# Weight tensors matching the TinyMLP projection weight shapes:
#   up_proj:   [128, 64]  → out=128, in_features=64
#   down_proj: [64, 128]  → out=64,  in_features=128
up_w   = np.random.randn(128, 64).astype(np.float32) * 0.1
down_w = np.random.randn(64, 128).astype(np.float32) * 0.1

out_dir = os.path.dirname(os.path.abspath(__file__))

save_file(
    {"calibration": calib},
    os.path.join(out_dir, "awq_calib_data.safetensors"),
)

save_file(
    {
        "TinyMLP.up_proj": up_w,
        "TinyMLP.down_proj": down_w,
    },
    os.path.join(out_dir, "awq_calib_weights.safetensors"),
)

print("Generated:")
print(f"  awq_calib_data.safetensors    shape={calib.shape}")
print(f"  awq_calib_weights.safetensors up_proj={up_w.shape} down_proj={down_w.shape}")
