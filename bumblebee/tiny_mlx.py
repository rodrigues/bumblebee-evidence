# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Converts checkpoints/tiny-random-Qwen3_5ForConditionalGeneration to an unquantized MLX checkpoint,
# and writes its mlx-lm logits to outputs/tiny-mlx-reference.json.
# Run with: uv run tiny_mlx.py
import json, os, shutil
import mlx.core as mx
from mlx_lm import convert, load

here = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(here, "checkpoints", "tiny-random-Qwen3_5ForConditionalGeneration")
input_ids = mx.array([[10, 20, 30, 40, 50, 60, 70, 80]])
name = "tiny-random-Qwen3_5ForConditionalGeneration-mlx"

path = os.path.join(here, "checkpoints", name)
shutil.rmtree(path, ignore_errors=True)
convert(source, mlx_path=path, dtype="float32")
model, _ = load(path)
logits = model(input_ids).astype(mx.float32)
reference = {name: {"logits": logits[0, 1:4, 1:4].tolist()}}

with open(os.path.join(here, "outputs", "tiny-mlx-reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
