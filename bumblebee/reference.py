# /// script
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Builds two tiny random Llama 4 checkpoints in checkpoints/ and records their transformers outputs, with those of
# optimum-intel-internal-testing/tiny-random-llama4, a multimodal checkpoint whose expert weights are all zero.
# - tiny-random-Llama4ForCausalLM is shaped like Scout: experts in every block, one per token, and queries and keys normalized.
# - tiny-random-Llama4ForCausalLM-interleaved is shaped like Maverick: experts in every other block, without the normalization, here with two per token.
# Both attend in chunks of 4 tokens and step the query temperature every 2 positions, so both change the outputs of a short input.
# Writes outputs/reference.json.
# Run with: uv run reference.py
import json, os, torch
from huggingface_hub import hf_hub_download
from transformers import Llama4Config, Llama4ForConditionalGeneration, Llama4ForCausalLM, Llama4TextConfig

here = os.path.dirname(os.path.abspath(__file__))
input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 0, 0]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
reference = {}


def record(model):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        generated = model.generate(input_ids=input_ids[:, :6], max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)
    return {
        "hidden_state": outputs.hidden_states[-1][:, 1:4, 1:4].tolist(),
        "logits": outputs.logits[:, 1:4, 1:4].tolist(),
        "logits_sum": outputs.logits[:, :10].sum(dim=-1).tolist(),
        "generated": generated[0, 6:].tolist(),
    }


def build(name, **overrides):
    config = Llama4TextConfig(
        vocab_size=1024, hidden_size=32, intermediate_size=37, intermediate_size_mlp=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=4, attention_chunk_size=4,
        floor_scale=2, max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        no_rope_layer_interval=2, rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, **overrides,
    )
    torch.manual_seed(0)
    model = Llama4ForCausalLM(config).eval()
    # The initialization leaves experts at zero and norms at one, which would hide a misplaced weight
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            param.normal_(0.0, 0.3 if param.dim() > 1 else 0.1).add_(1.0 if "norm" in param_name else 0.0)
    model.save_pretrained(os.path.join(here, "checkpoints", name))
    return model, config


scout, config = build("tiny-random-Llama4ForCausalLM", num_experts_per_tok=1, interleave_moe_layer_step=1, use_qk_norm=True)
reference["tiny-random-Llama4ForCausalLM"] = record(scout) | {"layer_types": config.layer_types, "moe_layers": config.moe_layers}

# Each feature off in turn, to show that the outputs above depend on it
base = record(scout)["logits_sum"]
for attribute, value in [("attention_chunk_size", 8192), ("floor_scale", 8192), ("use_qk_norm", False)]:
    for layer in scout.model.layers:
        if attribute == "floor_scale":
            layer.self_attn.floor_scale = value
        elif attribute == "use_qk_norm" and hasattr(layer.self_attn, "qk_norm"):
            del layer.self_attn.qk_norm
    if attribute == "attention_chunk_size":
        scout.config.attention_chunk_size = value
    print(f"logits change with {attribute} = {value}:", (torch.tensor(record(scout)["logits_sum"]) - torch.tensor(base)).abs().max().item())
    scout, _ = build("tiny-random-Llama4ForCausalLM", num_experts_per_tok=1, interleave_moe_layer_step=1, use_qk_norm=True)

interleaved, config = build("tiny-random-Llama4ForCausalLM-interleaved", num_experts_per_tok=2, interleave_moe_layer_step=2, use_qk_norm=False)
reference["tiny-random-Llama4ForCausalLM-interleaved"] = record(interleaved) | {"layer_types": config.layer_types, "moe_layers": config.moe_layers}

# The checkpoint sets attn_temperature_tuning to 4, which transformers 5.23.0 refuses, as it expects a boolean
repo = "optimum-intel-internal-testing/tiny-random-llama4"
data = json.load(open(hf_hub_download(repo, "config.json")))
data["text_config"]["attn_temperature_tuning"] = bool(data["text_config"]["attn_temperature_tuning"])
multimodal = Llama4ForConditionalGeneration.from_pretrained(repo, config=Llama4Config(**data), dtype=torch.float32).eval()
reference["tiny-random-llama4"] = {k: v for k, v in record(multimodal).items() if k != "hidden_state"}

with open(os.path.join(here, "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps({k: v["generated"] for k, v in reference.items()}))
