# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Decodes 100 tokens greedily with mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit in mlx-lm, for two prompts,
# and writes outputs/reference-30b.json with the tokens, the teacher-forced top choices and the speed.
# Run with: MLX_ENABLE_TF32=0 uv run reference_30b.py
import json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

repo = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
model, tokenizer = load(repo)
prompts = {
    "capital": "What is the capital of France? Answer in one sentence.",
    "story": "Write a short story about a lighthouse keeper.",
}
reference = {}

for name, content in prompts.items():
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        tokens = []
        first = None
        for token, _ in generate_step(mx.array(input_ids), model, max_tokens=100):
            tokens.append(int(token))
            if first is None:
                first = time.perf_counter()
        end = time.perf_counter()
        timings.append((first - start, (end - first) / (len(tokens) - 1)))
    prefill, decode = timings[-1]
    # Teacher forcing: the logits for every position of the full sequence, to compare top choices
    logits = model(mx.array([input_ids + tokens]))[0, len(input_ids) - 1 : -1].astype(mx.float32)
    top2 = mx.sort(logits, axis=-1)[:, -2:]
    reference[name] = {
        "prompt": prompt,
        "input_ids": input_ids,
        "generated": tokens,
        "text": tokenizer.decode(tokens),
        "prefill_ms": round(prefill * 1000, 1),
        "decode_ms_per_token": round(decode * 1000, 1),
        "teacher_forced_argmax": mx.argmax(logits, axis=-1).tolist(),
        "teacher_forced_margin": (top2[:, 1] - top2[:, 0]).tolist(),
    }

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference-30b.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)
print(json.dumps({k: {x: v[x] for x in ["text", "prefill_ms", "decode_ms_per_token"]} for k, v in reference.items()}, indent=2, ensure_ascii=False))
