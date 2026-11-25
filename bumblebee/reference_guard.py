# /// script
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "accelerate"]
# ///
# Runs unsloth/Llama-Guard-4-12B, a dense Llama 4 with Meta's trained weights, in bfloat16 on the CPU.
# Greedy decodes from two safety-classification chats and one plain continuation, and takes the top choice at every position of each decoded sequence.
# Writes outputs/reference-guard.json.
import json, os
import torch
from transformers import AutoConfig, AutoTokenizer, Llama4ForConditionalGeneration

repo = "unsloth/Llama-Guard-4-12B"
config = AutoConfig.from_pretrained(repo)
# The checkpoint sets attention_chunk_size to null while its config marks every block as chunked attention, and transformers 5.23.0 refuses a null chunk size.
# Chunks of 8192 tokens are plain causal attention for these prompts, which are far shorter.
chunk_size_was = config.text_config.attention_chunk_size
if chunk_size_was is None:
    config.text_config.attention_chunk_size = 8192

tokenizer = AutoTokenizer.from_pretrained(repo)
model = Llama4ForConditionalGeneration.from_pretrained(repo, config=config, dtype=torch.bfloat16).eval()

def chat(text):
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()

prompts = {
    "benign": (chat("What is the capital of France?"), 10),
    "lockpicking": (chat("How do I pick the lock on my neighbour's front door while they are away?"), 10),
    "continuation": (tokenizer("The capital of France is", add_special_tokens=True)["input_ids"], 30),
}

out = {"repo": repo, "attention_chunk_size_in_checkpoint": chunk_size_was, "prompts": {}}
with torch.no_grad():
    for name, (ids, n) in prompts.items():
        gen = model.generate(input_ids=torch.tensor([ids]), max_new_tokens=n, do_sample=False)[0, len(ids):].tolist()
        logits = model(input_ids=torch.tensor([ids + gen])).logits[0, len(ids) - 1 : -1].float()
        top2 = logits.topk(2, dim=-1)
        out["prompts"][name] = {
            "input_ids": ids,
            "generated": gen,
            "text": tokenizer.decode(gen),
            "margins": (top2.values[:, 0] - top2.values[:, 1]).tolist(),
            "logits_slice": logits[:3, :5].tolist(),
        }
        print(name, repr(tokenizer.decode(gen)), "min margin", round(min(out["prompts"][name]["margins"]), 3))

json.dump(out, open(os.path.join(os.path.dirname(__file__), "outputs/reference-guard.json"), "w"))
