# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Decodes 30 tokens greedily with lmstudio-community/Qwen3.8-27B-MLX-4bit in mlx-lm, with thinking on and off,
# and writes outputs/reference-27b.json with the tokens, the teacher-forced top choices, the speed and the peak memory.
# Run with: uv run reference_27b.py
import json, os, time

# MLX computes float32 matrix products in TF32 by default on M5-class GPUs; 0 keeps full float32.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

repo = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
model, tokenizer = load(repo)
reference = {"repo": repo, "memory_after_loading_gb": round(mx.get_active_memory() / 1e9, 2)}

for thinking in [True, False]:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    mx.reset_peak_memory()
    tokens = []
    start = time.perf_counter()
    for token, _ in generate_step(mx.array(input_ids), model, max_tokens=30, sampler=lambda logits: mx.argmax(logits, axis=-1)):
        tokens.append(int(token))
    seconds = time.perf_counter() - start
    peak = mx.get_peak_memory()
    # Teacher forcing: the logits for every position of the full sequence, to compare top choices
    logits = model(mx.array([input_ids + tokens]))[0, len(input_ids) - 1 : -1].astype(mx.float32)
    top2 = mx.sort(logits, axis=-1)[:, -2:]
    reference[f"thinking={str(thinking).lower()}"] = {
        "prompt": prompt,
        "input_ids": input_ids,
        "generated": tokens,
        "text": tokenizer.decode(tokens),
        "ms_per_token": round(seconds * 1000 / len(tokens), 1),
        "peak_memory_gb": round(peak / 1e9, 2),
        "teacher_forced_argmax": mx.argmax(logits, axis=-1).tolist(),
        "teacher_forced_margin": (top2[:, 1] - top2[:, 0]).tolist(),
    }

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference-27b.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("text", "ms_per_token", "peak_memory_gb")}) for k, v in reference.items()}, indent=2, ensure_ascii=False))
