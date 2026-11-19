# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers==5.23.0", "tokenizers"]
# ///
# Tokenizes the prompts in e4b.exs with the reference tokenizer and writes outputs/tokenizer-reference.json.
# Run with: uv run tokenizer.py
import json, os
from transformers import AutoTokenizer

prompts = [
    "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n",
    "<|turn>user\nQual é a capital do Brasil? Responda numa frase.<turn|>\n<|turn>model\n",
]
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "tokenizer-reference.json")
with open(path, "w") as f:
    json.dump({"prompts": prompts, "input_ids": ids}, f, indent=2)
