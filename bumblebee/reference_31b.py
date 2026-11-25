# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Decodes 30 tokens greedily with the MLX 4-bit conversion of Gemma 4 31B in mlx-lm, with thinking off, from two chat prompts,
# and writes outputs/reference-31b.json with the tokens, the teacher-forced top choices, the speed and the memory.
# The conversion is the target model of Youssofal/Gemma4-MTPLX-Optimized-Speed, which has no chat template,
# so the template comes from google/gemma-4-31B-it; the templates of the smaller Gemma 4 models do not close the thinking channel when thinking is off.
# mlx-lm renders the template with thinking on unless enable_thinking is passed as False.
# Run with: MLX_ENABLE_TF32=0 GEMMA4_31B_PATH=path/to/target GEMMA4_TEMPLATE_PATH=path/to/chat_template.jinja uv run reference_31b.py
import json, os, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

model, tokenizer = load(os.environ["GEMMA4_31B_PATH"])
with open(os.environ["GEMMA4_TEMPLATE_PATH"]) as f:
    chat_template = f.read()
prompts = {
    "capital": "What is the capital of France? Answer in one sentence.",
    "story": "Write a short story about a lighthouse keeper.",
}
reference = {"memory_after_loading_gb": round(mx.get_active_memory() / 1e9, 2), "chat_template": chat_template}

for name, content in prompts.items():
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": content}], chat_template=chat_template, add_generation_prompt=True, tokenize=False, enable_thinking=False)
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
    reference[name] = {
        "prompt": prompt,
        "input_ids": input_ids,
        "generated": tokens,
        "text": tokenizer.decode(tokens),
        "ms_per_token": round(seconds * 1000 / len(tokens), 1),
        "peak_memory_gb": round(peak / 1e9, 2),
        "teacher_forced_argmax": mx.argmax(logits, axis=-1).tolist(),
        "teacher_forced_margin": (top2[:, 1] - top2[:, 0]).tolist(),
    }

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference-31b.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)
print(json.dumps({k: {kk: v[kk] for kk in ("text", "ms_per_token", "peak_memory_gb")} for k, v in reference.items() if k in prompts}, indent=2, ensure_ascii=False))
