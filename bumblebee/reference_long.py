# /// script
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Records the logits of the tiny causal LM checkpoint for a 150-token input, which spans three of the 64-token chunks of the linear attention.
# The checkpoint's random decay forgets within a few tokens, so A_log is set to -4.0 to keep about a fifth of the state across a chunk.
# Writes outputs/reference-long.json.
# Run with: uv run reference_long.py
import json, os, torch
from transformers import AutoModelForCausalLM

here = os.path.dirname(os.path.abspath(__file__))
model = AutoModelForCausalLM.from_pretrained(os.path.join(here, "checkpoints", "tiny-random-Qwen3_5ForCausalLM"), dtype=torch.float32).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if name.endswith("linear_attn.A_log"):
            parameter.fill_(-4.0)
input_ids = (torch.arange(150) * 7 % 1000 + 1)[None]
with torch.no_grad():
    logits = model(input_ids=input_ids).logits
reference = {"input_ids": "(arange(150) * 7 % 1000 + 1)", "A_log": -4.0, "logits": logits[0, [63, 64, 149], 1:4].tolist()}
with open(os.path.join(here, "outputs", "reference-long.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
