# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Same as reference.py, with float32 activations instead of bfloat16, to compare without bfloat16 rounding.
# Writes outputs/reference-f32.json.
# Run with: uv run reference_f32.py
import json, os

# MLX computes float32 matrix products in TF32 by default on M5-class GPUs; 0 keeps full float32.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

repos = ["mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/Qwen3-4B-Instruct-2507-8bit"]
messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
reference = {}

for repo in repos:
    model, tokenizer = load(repo)
    model.set_dtype(mx.float32)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = [int(t) for t, _ in generate_step(mx.array(input_ids), model, max_tokens=30, sampler=lambda logits: mx.argmax(logits, axis=-1))]
    logits = model(mx.array([input_ids + tokens]))[0, len(input_ids) - 1 : -1].astype(mx.float32)
    top2 = mx.sort(logits, axis=-1)[:, -2:]
    reference[repo] = {
        "input_ids": input_ids,
        "generated": tokens,
        "teacher_forced_argmax": mx.argmax(logits, axis=-1).tolist(),
        "teacher_forced_margin": (top2[:, 1] - top2[:, 0]).tolist(),
    }
    del model

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference-f32.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2)
