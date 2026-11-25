# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors", "jinja2"]
# ///
# Decodes 30 tokens greedily with mistralai/Ministral-3-3B-Instruct-2512-BF16 in float32, for two prompts, and writes outputs/reference-3b.json.
# Run with: uv run reference_3b.py
import json, os, time, torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

repo = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = Mistral3ForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).eval()
# No end-of-sequence token, so decoding always gives 30 tokens; min_new_tokens would instead mask the token out and change the argmax
model.generation_config.eos_token_id = None
model.generation_config.pad_token_id = tokenizer.pad_token_id
prompts = {
    "capital": "What is the capital of France? Answer in one sentence.",
    "haiku": "Write a haiku about the sea.",
}
reference = {}

for name, content in prompts.items():
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=30, do_sample=False)
        logits = model(input_ids=generated[:, :-1]).logits[0, input_ids.shape[1] - 1 :].float()
    seconds = time.perf_counter() - start
    tokens = generated[0, input_ids.shape[1]:].tolist()
    top2 = logits.topk(2, dim=-1).values
    reference[name] = {
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "generated": tokens,
        "text": tokenizer.decode(tokens),
        "teacher_forced_margin": (top2[:, 0] - top2[:, 1]).tolist(),
        "seconds": round(seconds, 1),
    }

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference-3b.json")
with open(path, "w") as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)
print(json.dumps({k: {"text": v["text"], "min_margin": min(v["teacher_forced_margin"])} for k, v in reference.items()}, indent=2, ensure_ascii=False))
