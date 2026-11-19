# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Builds tiny random Gemma 4 checkpoints in checkpoints/ and writes their reference outputs to outputs/reference.json.
# Run with: uv run reference.py
import json, os, torch
from transformers import (
    Gemma4Config, Gemma4TextConfig, Gemma4AudioConfig, Gemma4VisionConfig,
    Gemma4TextModel, Gemma4ForCausalLM, Gemma4ForConditionalGeneration,
)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
reference = {}
text_config = dict(
    vocab_size=1024, hidden_size=32, intermediate_size=37, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
    layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    sliding_window=4, hidden_size_per_layer_input=8, vocab_size_per_layer_input=1024,
    num_kv_shared_layers=2, use_double_wide_mlp=True, final_logit_softcapping=30.0,
    max_position_embeddings=512, initializer_range=0.2,
)
vision_config = Gemma4VisionConfig(hidden_size=16, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
audio_config = Gemma4AudioConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=16)

input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

for cls in [Gemma4TextModel, Gemma4ForCausalLM, Gemma4ForConditionalGeneration]:
    name = cls.__name__
    if cls is Gemma4ForConditionalGeneration:
        config = Gemma4Config(text_config=text_config, vision_config=vision_config, audio_config=audio_config)
    else:
        config = Gemma4TextConfig(**text_config)

    torch.manual_seed(0)
    model = cls(config).eval()
    # Random init leaves norms at constants; perturb every parameter so they all matter.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    path = f"{out}/tiny-random-{name}"
    model.save_pretrained(path)

    # Published Gemma 4 checkpoints give the global attention shapes as top-level keys
    with open(f"{path}/config.json") as f:
        saved = json.load(f)
    text = saved.get("text_config", saved)
    text.pop("per_layer_config", None)
    text["global_head_dim"] = 16
    with open(f"{path}/config.json", "w") as f:
        json.dump(saved, f, indent=2)

    with torch.no_grad():
        o = model(input_ids=input_ids, attention_mask=attention_mask)
    entry = reference[f"tiny-random-{name}"] = {}
    if cls is Gemma4TextModel:
        entry["hidden_state"] = o.last_hidden_state[0, 1:4, 1:4].tolist()
    else:
        entry["logits"] = o.logits[0, 1:4, 1:4].tolist()
    if cls is Gemma4ForCausalLM:
        generated = model.generate(input_ids[:, :8], attention_mask=attention_mask[:, :8], max_new_tokens=8, min_new_tokens=8, do_sample=False)
        entry["generated"] = generated[0, 8:].tolist()

# A variant shaped like 31B: global layers use the key projection as values, with fewer key-value heads, and there are no per-layer inputs or shared layers.
config = Gemma4TextConfig(**{
    **text_config,
    "attention_k_eq_v": True, "num_global_key_value_heads": 1,
    "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "use_double_wide_mlp": False,
})
torch.manual_seed(0)
model = Gemma4ForCausalLM(config).eval()
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.1)
path = f"{out}/tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true"
model.save_pretrained(path)
with open(f"{path}/config.json") as f:
    saved = json.load(f)
saved.pop("per_layer_config", None)
saved["global_head_dim"] = 16
saved["num_global_key_value_heads"] = 1
with open(f"{path}/config.json", "w") as f:
    json.dump(saved, f, indent=2)
with torch.no_grad():
    o = model(input_ids=input_ids, attention_mask=attention_mask)
reference["tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true"] = {"logits": o.logits[0, 1:4, 1:4].tolist()}

os.makedirs(os.path.join(os.path.dirname(out), "outputs"), exist_ok=True)
with open(os.path.join(os.path.dirname(out), "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
