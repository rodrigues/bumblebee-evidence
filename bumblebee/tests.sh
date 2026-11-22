#!/bin/sh
# Runs the Bumblebee MLX quantization tests on the tiny checkpoints in checkpoints/, in place of the bumblebee-testing repositories, which do not exist yet.
# Writes outputs/tests.txt.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$BUMBLEBEE_PATH"
local_test=test/bumblebee/conversion/mlx_quantization_local_test.exs
trap 'rm -f "$local_test"' EXIT
sed -e "s/MLXQuantizationTest do/MLXQuantizationLocalTest do/" \
  -e "s|{:hf, \"bumblebee-testing/\([^\"]*\)\"}|{:local, \"$here/checkpoints/\1\"}|" \
  test/bumblebee/conversion/mlx_quantization_test.exs > "$local_test"
MIX_ENV=test mix compile > /dev/null 2>&1
mix test --trace "$local_test" > "$here/outputs/tests.txt" 2>&1
