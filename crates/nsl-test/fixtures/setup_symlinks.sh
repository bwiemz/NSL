#!/usr/bin/env bash
# Set up fixture sidecar symlinks per M57 spec §6.1.
#
# `nsl fpga-compile foo.nsl` looks for `foo_weights.bin` by default
# (sidecar convention); this script creates the symlink so v1 test
# invocations work with the default-fixture-path lookup without an
# explicit --fixture flag.
#
# NOTE: On Windows, `ln -sf` requires Developer Mode or Administrator
# privileges. If this fails on your Windows environment, copy the file
# instead:
#   cp mlp_int8_weights_v1.bin v1_mlp_weights.bin
#   cp mlp_int8_weights_v1.toml v1_mlp_weights.toml
# This is a known limitation documented in the M57.1 gap list.

set -euo pipefail
cd "$(dirname "$0")"

ln -sf mlp_int8_weights_v1.bin  v1_mlp_weights.bin  || {
    echo "WARNING: ln -sf failed (Windows without Developer Mode?). Falling back to cp."
    cp mlp_int8_weights_v1.bin  v1_mlp_weights.bin
}
ln -sf mlp_int8_weights_v1.toml v1_mlp_weights.toml || {
    echo "WARNING: ln -sf failed. Falling back to cp."
    cp mlp_int8_weights_v1.toml v1_mlp_weights.toml
}

echo "Fixture sidecar symlinks (or copies) ready."
