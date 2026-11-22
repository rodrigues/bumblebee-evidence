# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Converts tiny-random-Qwen3_5ForConditionalGeneration, from the checkpoints of https://github.com/rodrigues/bumblebee-qwen3-5-evidence, to an MLX checkpoint in 4 bits with group size 32,
# and writes its mlx-lm logits to outputs/tiny-qwen3_5-reference.json.
# Run with: QWEN3_5_EVIDENCE_PATH=path/to/bumblebee-qwen3-5-evidence uv run tiny_qwen3_5.py
import json, os, shutil
import mlx.core as mx
from mlx_lm import convert, load

here = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(os.environ["QWEN3_5_EVIDENCE_PATH"], "bumblebee", "checkpoints", "tiny-random-Qwen3_5ForConditionalGeneration")
input_ids = mx.array([[10, 20, 30, 40, 50, 60, 70, 80]])
name = "tiny-random-Qwen3_5ForConditionalGeneration-mlx-4bit"

path = os.path.join(here, "checkpoints", name)
shutil.rmtree(path, ignore_errors=True)
convert(source, mlx_path=path, dtype="float32", quantize=True, q_bits=4, q_group_size=32)
model, _ = load(path)
logits = model(input_ids).astype(mx.float32)
reference = {name: {"logits": logits[0, 1:4, 1:4].tolist()}}

with open(os.path.join(here, "outputs", "tiny-qwen3_5-reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
