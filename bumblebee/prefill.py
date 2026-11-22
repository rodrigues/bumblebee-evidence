# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Times one forward pass of Qwen/Qwen3.5-2B in mlx-lm, in bfloat16, over prompts of 128 to 8192 tokens, to compare the prompt time with Bumblebee on EMLX.
# Writes outputs/mlx-prefill.json.
# Run with: uv run prefill.py
import json, os, time
import mlx.core as mx
from mlx_lm import load

here = os.path.dirname(os.path.abspath(__file__))
model, _ = load("Qwen/Qwen3.5-2B")
results = {}

for n in [128, 512, 2048, 8192]:
    x = (mx.arange(n) % 1000 + 1000)[None]
    mx.eval(model(x)[0, -1, 0])
    start = time.perf_counter()
    mx.eval(model(x)[0, -1, 0])
    ms = round((time.perf_counter() - start) * 1000)
    print(f"mlx-lm {n} tokens: {ms} ms")
    results[n] = ms

with open(os.path.join(here, "outputs", "mlx-prefill.json"), "w") as f:
    json.dump(results, f, indent=2)
