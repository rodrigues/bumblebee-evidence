#!/bin/sh
# Runs the Llama 4 tests and the mixture of experts layer tests, with the checkpoints the tests name under bumblebee-testing read from checkpoints/ here,
# because those repositories do not exist yet. Writes outputs/tests.txt.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$BUMBLEBEE_PATH"
local_test=test/bumblebee/text/llama4_local_test.exs
sed -e "s/Llama4Test do/Llama4LocalTest do/" \
  -e "s|{:hf, \"bumblebee-testing/\([^\"]*\)\"}|{:local, \"$here/checkpoints/\1\"}|" \
  -e "s|{\"bumblebee-testing/\(tiny[^\"]*\)\",|{\"$here/checkpoints/\1\",|" \
  -e "s|{:hf, unquote(repo)}|{:local, unquote(repo)}|" \
  test/bumblebee/text/llama4_test.exs > "$local_test"
mix test "$local_test" test/bumblebee/layers/mixture_of_experts_test.exs 2>&1 |
  grep -E '^\s+[0-9]+\) test|^Finished|^Result|\*\* \(' > "$here/outputs/tests.txt" || true
rm -f "$local_test"
cat "$here/outputs/tests.txt"
