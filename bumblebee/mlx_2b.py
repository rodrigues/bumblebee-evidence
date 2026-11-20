# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Decodes 30 tokens greedily with Qwen/Qwen3.5-2B in mlx-lm, in bfloat16, from the prompts in outputs/reference-2b.json,
# to compare the time per token with Bumblebee on EMLX. Writes outputs/mlx-2b.json.
# Run with: uv run mlx_2b.py
import json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

here = os.path.dirname(os.path.abspath(__file__))
reference = json.load(open(os.path.join(here, "outputs", "reference-2b.json")))
model, tokenizer = load("Qwen/Qwen3.5-2B")
results = {}

for name, entry in sorted(reference.items()):
    sample = lambda logits: mx.argmax(logits, axis=-1)
    list(generate_step(mx.array(entry["input_ids"]), model, max_tokens=30, sampler=sample))
    start = time.perf_counter()
    tokens = [int(t) for t, _ in generate_step(mx.array(entry["input_ids"]), model, max_tokens=30, sampler=sample)]
    seconds = time.perf_counter() - start
    results[name] = {"generated": tokens, "same_as_transformers_f32": tokens == entry["generated"], "ms_per_token": round(seconds * 1000 / len(tokens), 1)}

with open(os.path.join(here, "outputs", "mlx-2b.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: {kk: v[kk] for kk in ("same_as_transformers_f32", "ms_per_token")} for k, v in results.items()}, indent=2))
