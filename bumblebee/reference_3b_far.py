# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Teacher forces the prompts and generated tokens of outputs/reference-3b.json through mistralai/Ministral-3-3B-Instruct-2512-BF16 in float32, with positions starting at 20000, past the 16384 where the query scaling starts.
# Records the top token and its margin over the second at each generated position in outputs/reference-3b-far.json.
# Run with: uv run reference_3b_far.py
import json, os, torch
from transformers import Mistral3ForConditionalGeneration

here = os.path.dirname(os.path.abspath(__file__))
offset = 20000
repo = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
model = Mistral3ForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).eval()
reference = json.load(open(os.path.join(here, "outputs", "reference-3b.json")))
far = {"offset": offset}

for name, entry in reference.items():
    tokens = torch.tensor([entry["input_ids"] + entry["generated"]])
    positions = torch.arange(tokens.shape[1])[None] + offset
    start = len(entry["input_ids"]) - 1
    with torch.no_grad():
        logits = model(input_ids=tokens, position_ids=positions).logits[0, start : tokens.shape[1] - 1].float()
    top = logits.topk(2, dim=-1)
    far[name] = {
        "top": top.indices[:, 0].tolist(),
        "margin": (top.values[:, 0] - top.values[:, 1]).tolist(),
    }

with open(os.path.join(here, "outputs", "reference-3b-far.json"), "w") as f:
    json.dump(far, f, indent=2)
for name in reference:
    same = sum(a == b for a, b in zip(far[name]["top"], reference[name]["generated"]))
    print(f"{name}: top tokens at positions past {offset} equal the tokens generated from position 0 at {same} of 30")
