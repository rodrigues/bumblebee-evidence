# /// script
# dependencies = ["torch==2.15.0", "transformers==5.23.0", "safetensors"]
# ///
# Builds checkpoints/tiny-random-KimiLinearForCausalLM, stored as Moonshot's checkpoints are, with their config keys and weight names,
# which transformers renames on load, and records its transformers outputs in outputs/reference.json.
# The checkpoint has Kimi Delta Attention and latent attention blocks, a dense first block, a shared expert and a selection bias,
# and the 22-token input spans two chunks of the chunked delta rule.
# Run with: uv run reference.py
import json, os, torch
from safetensors.torch import save_file
from transformers import KimiLinearConfig, KimiLinearForCausalLM

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "checkpoints", "tiny-random-KimiLinearForCausalLM")

original_config = {
    "architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear", "vocab_size": 1024, "hidden_size": 32,
    "intermediate_size": 24, "moe_intermediate_size": 12, "num_hidden_layers": 5, "num_attention_heads": 4,
    "num_key_value_heads": 4, "kv_lora_rank": 16, "q_lora_rank": None, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
    "v_head_dim": 8, "mla_use_nope": True, "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1,
    "num_expert_group": 1, "topk_group": 1, "moe_renormalize": True, "moe_router_activation_func": "sigmoid",
    "routed_scaling_factor": 2.446, "first_k_dense_replace": 1, "moe_layer_freq": 1, "use_grouped_topk": True,
    "linear_attn_config": {"full_attn_layers": [4], "kda_layers": [1, 2, 3, 5], "head_dim": 16, "num_heads": 2, "short_conv_kernel_size": 4},
    "hidden_act": "silu", "rms_norm_eps": 1e-5, "initializer_range": 0.02, "model_max_length": 1048576,
    "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 10000.0, "rope_scaling": None,
}

config = KimiLinearConfig(**original_config)
torch.manual_seed(0)
model = KimiLinearForCausalLM(config).eval()
# The initialization leaves experts and biases at zero and norms at one, which would hide a misplaced weight
with torch.no_grad():
    for name, param in model.named_parameters():
        if name.endswith("A_log"):
            param.uniform_(0.0, 1.0)
        elif "norm" in name:
            param.normal_(1.0, 0.1)
        else:
            param.normal_(0.0, 0.3 if param.dim() > 1 else 0.5)
    for name, buffer in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            buffer.normal_(0.0, 0.5)

# Stores the weights under the original names, undoing the renaming transformers does on load
state = {}
for name, tensor in model.state_dict().items():
    tensor = tensor.contiguous()
    name = name.replace(".self_attn.forget_gate.", ".self_attn.")
    layer = int(name.split(".")[2]) if name.startswith("model.layers.") else None
    sparse = layer is not None and config.mlp_layer_types[layer] == "sparse"
    if sparse:
        name = name.replace(".mlp.", ".block_sparse_moe.")
    if name.endswith("self_attn.conv1d.weight"):
        for kind, part in zip("qkv", tensor.chunk(3, dim=0)):
            state[name.replace("conv1d", f"{kind}_conv1d")] = part.contiguous()
    elif name.endswith("experts.gate_up_proj"):
        for expert, weight in enumerate(tensor):
            gate, up = weight.chunk(2, dim=0)
            state[name.replace("gate_up_proj", f"{expert}.w1.weight")] = gate.contiguous()
            state[name.replace("gate_up_proj", f"{expert}.w3.weight")] = up.contiguous()
    elif name.endswith("experts.down_proj"):
        for expert, weight in enumerate(tensor):
            state[name.replace("down_proj", f"{expert}.w2.weight")] = weight.contiguous()
    else:
        state[name] = tensor
os.makedirs(path, exist_ok=True)
save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(path, "config.json"), "w") as f:
    json.dump(original_config, f, indent=2)

input_ids = torch.tensor([[10 + 37 * i % 1000 for i in range(22)]])
attention_mask = torch.ones_like(input_ids)
attention_mask[0, -2:] = 0
reference = {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}

for name, checkpoint in [("built", model), ("reloaded", KimiLinearForCausalLM.from_pretrained(path, dtype=torch.float32).eval())]:
    with torch.no_grad():
        outputs = checkpoint(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        generated = checkpoint.generate(input_ids=input_ids[:, :18], max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)
    reference[name] = {
        "hidden_state": outputs.hidden_states[-1][:, 1:4, 1:4].tolist(),
        "logits": outputs.logits[:, 1:4, 1:4].tolist(),
        "logits_last": outputs.logits[:, 17:20, 1:4].tolist(),
        "generated": generated[0, 18:].tolist(),
    }
    print(name, reference[name]["generated"])
assert reference["built"] == reference["reloaded"], "the stored checkpoint does not reload to the same model"
print("layer types", config.layer_types, config.mlp_layer_types)

with open(os.path.join(here, "outputs", "reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
