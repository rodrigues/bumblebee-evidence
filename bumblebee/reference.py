# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Runs greedy decoding with mlx-lm on the MLX quantized checkpoints and writes outputs/reference.json.
# Run with: uv run reference.py
import json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

repos = ["mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/Qwen3-4B-Instruct-2507-8bit", "Qwen/Qwen3-4B-Instruct-2507"]
messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
max_tokens = 30

reference = {}
for repo in repos:
    model, tokenizer = load(repo)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = []
    start = time.perf_counter()
    for token, _ in generate_step(mx.array(input_ids), model, max_tokens=max_tokens, sampler=lambda logits: mx.argmax(logits, axis=-1)):
        tokens.append(int(token))
    seconds = time.perf_counter() - start
    # Teacher forcing: the logits for every position of the full sequence, to compare top choices
    logits = model(mx.array([input_ids + tokens]))[0, len(input_ids) - 1 : -1].astype(mx.float32)
    top2 = mx.sort(logits, axis=-1)[:, -2:]
    reference[repo] = {
        "prompt": prompt,
        "input_ids": input_ids,
        "generated": tokens,
        "text": tokenizer.decode(tokens),
        "ms_per_token": round(seconds * 1000 / len(tokens), 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "teacher_forced_argmax": mx.argmax(logits, axis=-1).tolist(),
        "teacher_forced_margin": (top2[:, 1] - top2[:, 0]).tolist(),
    }
    mx.reset_peak_memory()
    del model

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)
print(json.dumps(reference, indent=2, ensure_ascii=False))
