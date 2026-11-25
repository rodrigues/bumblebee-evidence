# Bumblebee evidence

Each branch of this repository backs one change to Bumblebee, or to EMLX under it, with results checked against the reference implementation in Python.
Every branch keeps its results in `outputs/` folders, so nothing needs running to read them, and its README ends with the steps to reproduce them.

| Branch | Model or change | Bumblebee branch on rodrigues/bumblebee |
|---|---|---|
| [`gemma4`](https://github.com/rodrigues/bumblebee-evidence/tree/gemma4) | Gemma 4 text models, and Gemma 4 31B from its MLX 4-bit conversion | merged as elixir-nx/bumblebee#485; [`gemma4-mlx`](https://github.com/rodrigues/bumblebee/tree/gemma4-mlx) |
| [`qwen3-5`](https://github.com/rodrigues/bumblebee-evidence/tree/qwen3-5) | Qwen3.5, Qwen3.6 and the dense Qwen3.8 | [`qwen3-5`](https://github.com/rodrigues/bumblebee/tree/qwen3-5) |
| [`mlx-quantized`](https://github.com/rodrigues/bumblebee-evidence/tree/mlx-quantized) | MLX quantized checkpoints on EMLX, incl. Qwen3.8-27B in 4 bits | [`mlx-quantized`](https://github.com/rodrigues/bumblebee/tree/mlx-quantized), [`qwen3-8-mlx`](https://github.com/rodrigues/bumblebee/tree/qwen3-8-mlx) |
| [`qwen3-moe`](https://github.com/rodrigues/bumblebee-evidence/tree/qwen3-moe) | Mixture of experts and Qwen3-MoE, incl. its MLX 4-bit conversion | [`qwen3-moe`](https://github.com/rodrigues/bumblebee/tree/qwen3-moe), [`qwen3-moe-mlx`](https://github.com/rodrigues/bumblebee/tree/qwen3-moe-mlx) |
| [`llama4`](https://github.com/rodrigues/bumblebee-evidence/tree/llama4) | Llama 4 text models | [`llama4`](https://github.com/rodrigues/bumblebee/tree/llama4) |
| [`kimi-linear`](https://github.com/rodrigues/bumblebee-evidence/tree/kimi-linear) | Kimi Linear, and loading tiktoken tokenizers | [`kimi-linear`](https://github.com/rodrigues/bumblebee/tree/kimi-linear), [`tiktoken-tokenizers`](https://github.com/rodrigues/bumblebee/tree/tiktoken-tokenizers) |
| [`ministral3`](https://github.com/rodrigues/bumblebee-evidence/tree/ministral3) | Ministral 3 text models | [`ministral3`](https://github.com/rodrigues/bumblebee/tree/ministral3) |

The EMLX changes are on rodrigues/emlx, in branches `checkpoint-quantized`, `compiled-gather-qmm` and `compiled-while-body`.
