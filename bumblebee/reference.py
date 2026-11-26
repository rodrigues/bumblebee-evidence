# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Records logits and 8 greedy tokens of optimum-internal-testing/tiny-random-qwen3_moe in transformers, in outputs/reference.json.
# Run with: uv run reference.py
import json, os, torch
from transformers import Qwen3MoeForCausalLM

repo = "optimum-internal-testing/tiny-random-qwen3_moe"
model = Qwen3MoeForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    short = model(input_ids=input_ids[:, :2]).logits
    generated = model.generate(input_ids[:, :8], attention_mask=attention_mask[:, :8], max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)

# The checkpoint normalizes the chosen experts' probabilities; this is the default of Qwen3MoeConfig, which does not
unnormalized = Qwen3MoeForCausalLM.from_pretrained(repo, dtype=torch.float32, norm_topk_prob=False).eval()
with torch.no_grad():
    unnormalized_logits = unnormalized(input_ids=input_ids, attention_mask=attention_mask).logits
    unnormalized_short = unnormalized(input_ids=input_ids[:, :2]).logits

reference = {
    "logits without normalizing the chosen experts' probabilities": unnormalized_logits[:, 1:4, 1:4].tolist(),
    "logits of 2 tokens without normalizing the chosen experts' probabilities": unnormalized_short[:, :, 1:4].tolist(),
    "logits": logits[:, 1:4, 1:4].tolist(),
    "logits of 2 tokens": short[:, :, 1:4].tolist(),
    "generated": generated[0, 8:].tolist(),
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference))
