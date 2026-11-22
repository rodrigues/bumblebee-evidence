# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Builds tiny random Qwen3.5 checkpoints in checkpoints/ and writes their reference outputs to outputs/reference.json.
# Run with: uv run reference.py
import json, os, torch
from transformers import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
    Qwen3_5TextModel, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration,
)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
reference = {}
# Grouped like Qwen3.8-27B: fewer linear attention key heads than value heads, and a quarter of each head rotated.
text_config = dict(
    vocab_size=1024, hidden_size=32, intermediate_size=37, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    max_position_embeddings=512, initializer_range=0.2,
)
vision_config = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=37, num_heads=2, out_hidden_size=32, num_position_embeddings=16)

input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

for cls in [Qwen3_5TextModel, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration]:
    name = cls.__name__
    if cls is Qwen3_5ForConditionalGeneration:
        config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    else:
        config = Qwen3_5TextConfig(**text_config)

    torch.manual_seed(0)
    model = cls(config).eval()
    # Random init leaves norms at constants; perturb every parameter so they all matter.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    path = f"{out}/tiny-random-{name}"
    model.save_pretrained(path)

    with torch.no_grad():
        o = model(input_ids=input_ids, attention_mask=attention_mask)
    entry = reference[f"tiny-random-{name}"] = {}
    if cls is Qwen3_5TextModel:
        entry["hidden_state"] = o.last_hidden_state[0, 1:4, 1:4].tolist()
    else:
        entry["logits"] = o.logits[0, 1:4, 1:4].tolist()
    if cls is Qwen3_5ForCausalLM:
        with torch.no_grad():
            entry["hidden_state"] = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0, 1:4, 1:4].tolist()
        generated = model.generate(input_ids[:, :8], attention_mask=attention_mask[:, :8], max_new_tokens=8, min_new_tokens=8, do_sample=False)
        entry["generated"] = generated[0, 8:].tolist()

os.makedirs(os.path.join(os.path.dirname(out), "outputs"), exist_ok=True)
with open(os.path.join(os.path.dirname(out), "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference, indent=2))
