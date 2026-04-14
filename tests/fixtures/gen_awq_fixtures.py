import numpy as np
from safetensors.numpy import save_file

np.random.seed(42)
# Calibration data FIRST — draws from the seeded RNG.
calib = np.random.randn(8, 4, 64).astype(np.float32)
save_file({"calibration": calib}, "tests/fixtures/awq_calib_data.safetensors")

# Weights SECOND — still from the same seeded RNG sequence.
up_w   = (np.random.randn(128, 64).astype(np.float32) * 0.1)
down_w = (np.random.randn(64, 128).astype(np.float32) * 0.1)
save_file(
    {"TinyMLP.up_proj": up_w, "TinyMLP.down_proj": down_w},
    "tests/fixtures/awq_calib_weights.safetensors",
)
print("wrote fixtures with seed 42")
