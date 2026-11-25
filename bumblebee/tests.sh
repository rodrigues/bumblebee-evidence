#!/bin/sh
# Runs the Kimi Linear tests and the mixture of experts layer tests, with the checkpoint the tests name under bumblebee-testing read from checkpoints/ here,
# because that repository does not exist yet. Writes outputs/tests.txt.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$BUMBLEBEE_PATH"
local_test=test/bumblebee/text/kimi_linear_local_test.exs
sed -e "s/KimiLinearTest do/KimiLinearLocalTest do/" \
  -e "s|{:hf, \"bumblebee-testing/\([^\"]*\)\"}|{:local, \"$here/checkpoints/\1\"}|" \
  test/bumblebee/text/kimi_linear_test.exs > "$local_test"
mix test "$local_test" test/bumblebee/layers_test.exs 2>&1 |
  grep -E '^\s+[0-9]+\) test|^Finished|^Result|\*\* \(' > "$here/outputs/tests.txt" || true
rm -f "$local_test"
cat "$here/outputs/tests.txt"
