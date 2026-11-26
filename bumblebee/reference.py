# /// script
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Records transformers outputs of the tiny Ministral 3 checkpoints, including positions past 16384, where the query scaling starts.
# Writes outputs/reference.json.
# Run with: uv run reference.py
import json, os, torch
from transformers import AutoModelForCausalLM, Mistral3ForConditionalGeneration

here = os.path.dirname(os.path.abspath(__file__))
input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
reference = {}

causal = AutoModelForCausalLM.from_pretrained("hf-tiny-v2/tiny-random-Ministral3ForCausalLM", dtype=torch.float32).eval()
with torch.no_grad():
    logits = causal(input_ids=input_ids, attention_mask=attention_mask).logits
    far = torch.arange(16380, 16390)[None]
    far_logits = causal(input_ids=input_ids, attention_mask=attention_mask, position_ids=far).logits
    generated = causal.generate(input_ids=input_ids[:, :8], max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)
# Random weights make attention nearly uniform, so scaling the queries barely changes it.
# Sharper queries make the query scaling past position 16384 change the logits.
with torch.no_grad():
    for layer in causal.model.layers:
        layer.self_attn.q_proj.weight.mul_(30.0)
    sharp_far_logits = causal(input_ids=input_ids, attention_mask=attention_mask, position_ids=far).logits
    causal.config.rope_parameters["llama_4_scaling_beta"] = 0.0
    unscaled_far_logits = causal(input_ids=input_ids, attention_mask=attention_mask, position_ids=far).logits
print("effect of the query scaling on the sharp logits:", (sharp_far_logits - unscaled_far_logits).abs().max().item())
# In float16, integer positions above 2048 are not exact, so this checks that the scale comes from the integer positions.
half = AutoModelForCausalLM.from_pretrained("hf-tiny-v2/tiny-random-Ministral3ForCausalLM", dtype=torch.float16).eval()
with torch.no_grad():
    for layer in half.model.layers:
        layer.self_attn.q_proj.weight.mul_(30.0)
    half_far_logits = half(input_ids=input_ids, attention_mask=attention_mask, position_ids=far).logits.float()
reference["tiny-random-Ministral3ForCausalLM"] = {
    "float16 sharp logits at positions 16381 to 16385": half_far_logits[:, 1:6, 1:4].tolist(),
    "sharp logits at positions 16380 to 16389": sharp_far_logits[:, 3:6, 1:4].tolist(),
    "logits": logits[:, 1:4, 1:4].tolist(),
    "logits at positions 16380 to 16389": far_logits[:, 3:6, 1:4].tolist(),
    "generated": generated[0, 8:].tolist(),
}

multimodal = Mistral3ForConditionalGeneration.from_pretrained("hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration", dtype=torch.float32).eval()
with torch.no_grad():
    logits = multimodal(input_ids=input_ids, attention_mask=attention_mask).logits
reference["tiny-random-Mistral3ForConditionalGeneration"] = {"logits": logits[:, 1:4, 1:4].tolist()}

with open(os.path.join(here, "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
