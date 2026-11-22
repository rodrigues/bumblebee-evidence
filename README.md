# Qwen3.5 and Qwen3.8 in Bumblebee: evidence

This repository backs the changes that let Elixir run the dense Qwen3.5, Qwen3.6 and Qwen3.8 models with Bumblebee:

- Bumblebee: the Qwen3.5 text model, on branch [`qwen3-5`](https://github.com/rodrigues/bumblebee/tree/qwen3-5) of rodrigues/bumblebee ([`bumblebee.patch`](bumblebee/bumblebee.patch)).
- Qwen3.8-27B in 4 bits also needs MLX quantized checkpoints on EMLX, from https://github.com/rodrigues/bumblebee-mlx-quantized-evidence; branch [`qwen3-8-mlx`](https://github.com/rodrigues/bumblebee/tree/qwen3-8-mlx) has both.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) builds three tiny random checkpoints in [`checkpoints/`](bumblebee/checkpoints) with transformers 5.23.0 and records their outputs in [`reference.json`](bumblebee/outputs/reference.json).
[`tiny_mlx.py`](bumblebee/tiny_mlx.py) converts the multimodal one with mlx-lm 0.32.2, unquantized and in 4 bits, and records mlx-lm's outputs in [`tiny-mlx-reference.json`](bumblebee/outputs/tiny-mlx-reference.json).

| Checkpoint | Covers |
|---|---|
| [tiny-random-Qwen3_5TextModel](bumblebee/checkpoints/tiny-random-Qwen3_5TextModel) | the `:base` architecture, from a checkpoint whose names have no prefix |
| [tiny-random-Qwen3_5ForCausalLM](bumblebee/checkpoints/tiny-random-Qwen3_5ForCausalLM) | the `:for_causal_language_modeling` architecture, 8 tokens of greedy generation through the cache, and left padding |
| [tiny-random-Qwen3_5ForConditionalGeneration](bumblebee/checkpoints/tiny-random-Qwen3_5ForConditionalGeneration) | a multimodal checkpoint, with the language model under `model.language_model` |
| [tiny-random-Qwen3_5ForConditionalGeneration-mlx](bumblebee/checkpoints/tiny-random-Qwen3_5ForConditionalGeneration-mlx) | the same weights in MLX's layout, with the norm shift added to the weights and the convolution kernel axes moved |
| [tiny-random-Qwen3_5ForConditionalGeneration-mlx-4bit](bumblebee/checkpoints/tiny-random-Qwen3_5ForConditionalGeneration-mlx-4bit) | the same model quantized to 4 bits, which needs both branches |

The model has four linear attention blocks and one full attention block, fewer linear attention key heads than value heads as in 27B, and rotary embedding on a quarter of each head.
The Bumblebee tests check these values on EXLA, within 1.0e-4 of transformers, and generation is identical token for token ([`tests.sh`](bumblebee/tests.sh), [`tests.txt`](bumblebee/outputs/tests.txt)).
`tests.sh` runs the test file on the checkpoints here, because the `bumblebee-testing` repositories it names do not exist yet.
The 4-bit checkpoint matches mlx-lm within 1.0e-3 on EMLX; MLX's float32 matrix products differ from the exact result by up to 7.4e-4, so a tighter tolerance would test MLX rather than Bumblebee.

## A real checkpoint: Qwen3.5-2B

[`reference_2b.py`](bumblebee/reference_2b.py) decodes 30 tokens greedily with transformers in float32, with thinking on and off, and [`bumblebee_2b.exs`](bumblebee/bumblebee_2b.exs) does the same in Bumblebee ([`results-2b.md`](bumblebee/outputs/results-2b.md)):

| Run | Thinking on | Thinking off | Time per token |
|---|---|---|---|
| EXLA, CPU, float32 | identical | identical | 57 ms |
| EMLX, GPU, float32 | identical | identical | 39 ms |
| EMLX, GPU, bfloat16 | identical | identical | 27 ms |

mlx-lm takes 17 ms per token for the same model in bfloat16 ([`mlx_2b.py`](bumblebee/mlx_2b.py), [`mlx-2b.json`](bumblebee/outputs/mlx-2b.json)).

The tokenizer gives the same ids as transformers for both prompts.
Qwen3.5-0.8B and 2B load with no missing parameters and no shape mismatches.

The reference disables the end-of-sequence token instead of setting `min_new_tokens`.
With `min_new_tokens`, transformers masks the end-of-sequence token out of the logits, so its greedy choice differs from the model's argmax wherever that token would win, which happens at the ninth token with thinking off.

## Qwen3.8-27B in 4 bits

[`reference_27b.py`](bumblebee/reference_27b.py) decodes 30 tokens greedily with `lmstudio-community/Qwen3.8-27B-MLX-4bit` in mlx-lm, and [`bumblebee_27b.exs`](bumblebee/bumblebee_27b.exs) does the same in Bumblebee on EMLX, in bfloat16 ([`results-27b.md`](bumblebee/outputs/results-27b.md)):

| | Bumblebee | mlx-lm |
|---|---|---|
| 30 greedy tokens, thinking on and off | identical | |
| Top choice at each of the 30 positions, teacher forced | identical | |
| Memory after generating | 17.05 GB | 15.13 GB after loading |
| Peak memory while generating | 17.68 GB | 15.52 GB |
| Time per token, thinking off and on | 99 and 136 ms | 63 and 82 ms |

The tokenizer gives the same ids as mlx-lm.
It loads in about 2 s with no missing parameters and no shape mismatches, and the only unused tensors are the 187 of the vision tower ([`params_27b.exs`](bumblebee/params_27b.exs), [`params-27b.txt`](bumblebee/outputs/params-27b.txt)).
Bumblebee uses about 2 GB more than mlx-lm because it dequantizes the embedding table, which MLX only materializes on first use, so memory read right after loading is lower than in use.

The 851 language model tensors of the bfloat16 checkpoint, `Qwen/Qwen3.8-27B`, have the same names as those of the MLX conversion, once the prefixes are mapped.

## Not checked

- Linux and Windows.
- `Qwen/Qwen3.8-27B` in bfloat16, which needs 54 GB; only its tensor names were compared.
- Qwen3.6 checkpoints, which use the same architecture.
- The mixture-of-experts variants, which Bumblebee refuses on load, and the vision tower.

## Reproduce

```sh
cd bumblebee
uv run reference.py
uv run tiny_mlx.py
BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
uv run reference_2b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_2b.exs
uv run mlx_2b.py
uv run reference_27b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_27b.exs
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir params_27b.exs
```

The 2B and 27B scripts read the checkpoints from the Hugging Face cache, so download them first with `hf download`.
`bumblebee_27b.exs` and `params_27b.exs` need the `qwen3-8-mlx` branch and about 20 GB of memory.
