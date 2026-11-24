# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Converts bumblebee-testing/tiny-random-Qwen3ForCausalLM to MLX quantized checkpoints in checkpoints/
# and writes their reference outputs to outputs/tiny-reference.json.
# Run with: uv run tiny.py
import json, os, shutil

# MLX computes float32 matrix products in TF32 by default on M5-class GPUs; 0 keeps full float32.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx_lm import convert, load

here = os.path.dirname(os.path.abspath(__file__))
source = "bumblebee-testing/tiny-random-Qwen3ForCausalLM"
input_ids = mx.array([[10, 20, 30, 40, 50, 60, 70, 80]])
reference = {}

for bits in [4, 8]:
    name = f"tiny-random-Qwen3ForCausalLM-mlx-{bits}bit"
    path = os.path.join(here, "checkpoints", name)
    shutil.rmtree(path, ignore_errors=True)
    # The tiny model's hidden size is 32, so groups of 64 would not fit
    convert(source, mlx_path=path, quantize=True, q_bits=bits, q_group_size=32)

    model, _ = load(path)
    logits = model(input_ids)
    embedding = model.model.embed_tokens
    table = mx.dequantize(embedding.weight, embedding.scales, embedding.biases, group_size=32, bits=bits)

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    quantized = sorted(k[: -len(".scales")] for k in mx.load(os.path.join(path, "model.safetensors")) if k.endswith(".scales"))

    reference[name] = {
        "quantization": config["quantization"],
        "quantized_weights": quantized,
        "logits": logits[0, 1:4, 1:4].tolist(),
        "embedding": table[10:13, 1:4].astype(mx.float32).tolist(),
    }

with open(os.path.join(here, "outputs", "tiny-reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
