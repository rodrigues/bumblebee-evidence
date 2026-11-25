# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Converts optimum-internal-testing/tiny-random-qwen3_moe to an MLX checkpoint in 4 bits with group size 32,
# and writes its mlx-lm logits to outputs/tiny-mlx-reference.json.
# Run with: MLX_ENABLE_TF32=0 uv run tiny_mlx.py
import json, os, shutil
import mlx.core as mx
from mlx_lm import convert, load

here = os.path.dirname(os.path.abspath(__file__))
name = "tiny-random-Qwen3MoeForCausalLM-mlx-4bit"
path = os.path.join(here, "checkpoints", name)
shutil.rmtree(path, ignore_errors=True)
convert("optimum-internal-testing/tiny-random-qwen3_moe", mlx_path=path, dtype="float32", quantize=True, q_bits=4, q_group_size=32)
model, _ = load(path)
reference = {}
for label, ids in [("logits", [[10, 20, 30, 40, 50, 60, 70, 80]]), ("logits of 2 tokens", [[10, 20]])]:
    logits = model(mx.array(ids)).astype(mx.float32)
    reference[label] = logits[0, :, 1:4].tolist()

with open(os.path.join(here, "outputs", "tiny-mlx-reference.json"), "w") as f:
    json.dump({name: reference}, f, indent=2)
print(json.dumps(reference))
