#!/bin/sh
# Runs the Bumblebee tests this change touches, with the MLX checkpoints read from checkpoints/ here and from the MLX quantized evidence,
# because the bumblebee-testing repositories the tests name do not exist yet. Writes outputs/tests.txt.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee MLX_EVIDENCE_PATH=/absolute/path/to/a/checkout/of/mlx-quantized ./tests.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$BUMBLEBEE_PATH"
local_test=test/bumblebee/conversion/mlx_quantization_local_test.exs
sed -e "s/MLXQuantizationTest do/MLXQuantizationLocalTest do/" \
  -e "s|{:hf, \"bumblebee-testing/\(tiny-random-Qwen3MoeForCausalLM[^\"]*\)\"}|{:local, \"$here/checkpoints/\1\"}|" \
  -e "s|{:hf, \"bumblebee-testing/\([^\"]*\)\"}|{:local, \"$MLX_EVIDENCE_PATH/bumblebee/checkpoints/\1\"}|" \
  test/bumblebee/conversion/mlx_quantization_test.exs > "$local_test"
mix test test/bumblebee/text/qwen3_test.exs test/bumblebee/layers_test.exs "$local_test" 2>&1 |
  grep -E '^\s+[0-9]+\) test|^Finished|^Result|\*\* \(' > "$here/outputs/tests.txt" || true
rm -f "$local_test"
cat "$here/outputs/tests.txt"
