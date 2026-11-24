# Qwen3.5 and Qwen3.8 in Bumblebee: evidence

This repository backs the changes that let Elixir run the dense Qwen3.5, Qwen3.6 and Qwen3.8 models with Bumblebee:

- Bumblebee: the Qwen3.5 text model, on branch [`qwen3-5`](https://github.com/rodrigues/bumblebee/tree/qwen3-5) of rodrigues/bumblebee ([`bumblebee.patch`](bumblebee/bumblebee.patch)).
- Qwen3.8-27B in 4 bits also needs MLX quantized checkpoints on EMLX, so its evidence is in https://github.com/rodrigues/bumblebee-mlx-quantized-evidence.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) builds three tiny random checkpoints in [`checkpoints/`](bumblebee/checkpoints) with transformers 5.23.0 and records their outputs in [`reference.json`](bumblebee/outputs/reference.json).
[`tiny_mlx.py`](bumblebee/tiny_mlx.py) converts the multimodal one to MLX's layout with mlx-lm 0.32.2, unquantized, and records mlx-lm's outputs in [`tiny-mlx-reference.json`](bumblebee/outputs/tiny-mlx-reference.json).

| Checkpoint | Covers |
|---|---|
| [tiny-random-Qwen3_5TextModel](bumblebee/checkpoints/tiny-random-Qwen3_5TextModel) | the `:base` architecture, from a checkpoint whose names have no prefix |
| [tiny-random-Qwen3_5ForCausalLM](bumblebee/checkpoints/tiny-random-Qwen3_5ForCausalLM) | the `:for_causal_language_modeling` architecture, 8 tokens of greedy generation through the cache, and left padding |
| [tiny-random-Qwen3_5ForConditionalGeneration](bumblebee/checkpoints/tiny-random-Qwen3_5ForConditionalGeneration) | a multimodal checkpoint, with the language model under `model.language_model` |
| [tiny-random-Qwen3_5ForConditionalGeneration-mlx](bumblebee/checkpoints/tiny-random-Qwen3_5ForConditionalGeneration-mlx) | the same weights in MLX's layout, with the norm shift added to the weights and the convolution kernel axes moved |

The model has four linear attention blocks and one full attention block, fewer linear attention key heads than value heads as in 27B, and rotary embedding on a quarter of each head.
[`bumblebee.exs`](bumblebee/bumblebee.exs) runs the same checkpoints through Bumblebee on EXLA and compares them in [`results.md`](bumblebee/outputs/results.md): every output is within 5.7e-6 of transformers, and generation is identical token for token.
The Bumblebee tests check these values on EXLA, within 1.0e-4 of transformers, and generation is identical token for token ([`tests.sh`](bumblebee/tests.sh), [`tests.txt`](bumblebee/outputs/tests.txt)).
`tests.sh` runs the test file on the checkpoints here, because the `bumblebee-testing` repositories it names do not exist yet.
[`reference_long.py`](bumblebee/reference_long.py) records the causal checkpoint's logits for two 150-token inputs, which span three chunks of the linear attention, in [`reference-long.json`](bumblebee/outputs/reference-long.json), and the tests check those too.
The second input repeats one token, so the keys within each chunk are identical.
That is the hardest case for the triangular system Bumblebee solves in each chunk: an earlier version inverted it by a series, which put these logits 3.2 from transformers on EXLA in float32.

## A real checkpoint: Qwen3.5-2B

[`reference_2b.py`](bumblebee/reference_2b.py) decodes 30 tokens greedily with transformers in float32, with thinking on and off, and [`bumblebee_2b.exs`](bumblebee/bumblebee_2b.exs) does the same in Bumblebee ([`results-2b.md`](bumblebee/outputs/results-2b.md)):

| Run | Thinking on | Thinking off | Time per token |
|---|---|---|---|
| EXLA, CPU, float32 | identical | identical | 57 ms |
| EMLX, GPU, float32 | identical | identical | 32 ms |
| EMLX, GPU, bfloat16 | identical | identical | 20 ms |

mlx-lm takes 17 ms per token for the same model in bfloat16 ([`mlx_2b.py`](bumblebee/mlx_2b.py), [`mlx-2b.json`](bumblebee/outputs/mlx-2b.json)).
The EMLX runs set MLX_ENABLE_TF32=0, because MLX computes float32 matrix products in TF32 by default on the M5.
With TF32 the results are the same, all identical ([`results-2b-tf32.md`](bumblebee/outputs/results-2b-tf32.md)).

The tokenizer gives the same ids as transformers for both prompts.
Qwen3.5-0.8B and 2B load with no missing parameters and no shape mismatches, and the only unused tensors are the vision tower's and, in 0.8B, the multi-token prediction layer's ([`params.exs`](bumblebee/params.exs), [`params.txt`](bumblebee/outputs/params.txt)).

The reference disables the end-of-sequence token instead of setting `min_new_tokens`.
With `min_new_tokens`, transformers masks the end-of-sequence token out of the logits, so its greedy choice differs from the model's argmax wherever that token would win, which happens at the ninth token with thinking off.

## Prompt time

Bumblebee computes the linear attention over a prompt in chunks of 64 tokens, as transformers does, and solves a triangular system in each chunk.
[`prefill.exs`](bumblebee/prefill.exs) times one forward pass of Qwen3.5-2B on EMLX in bfloat16 over prompts of 128 to 8192 tokens ([`prefill.json`](bumblebee/outputs/prefill.json)), and [`prefill.py`](bumblebee/prefill.py) times mlx-lm on the same prompts ([`mlx-prefill.json`](bumblebee/outputs/mlx-prefill.json)).
Each time is one pass after a warm-up pass:

| Prompt tokens | Bumblebee | mlx-lm |
|---|---|---|
| 128 | 43 ms | 31 ms |
| 512 | 128 ms | 78 ms |
| 2048 | 494 ms | 301 ms |
| 8192 | 2282 ms | 1298 ms |

A second run gave 43, 123, 497 and 2311 ms.
The earlier series inverse took 37, 101, 419 and 1993 ms, but lost precision on repeated tokens.

## Qwen3.8-27B

The 851 language model tensors of the bfloat16 checkpoint, `Qwen/Qwen3.8-27B`, have the same names as those of the MLX conversion, once the prefixes are mapped.
The MLX 4-bit conversion of 27B loads and decodes as mlx-lm does, as recorded in the MLX evidence.

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
uv run reference_long.py
BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
uv run reference_2b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_2b.exs
uv run mlx_2b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir prefill.exs
uv run prefill.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir params.exs
```

The 2B, prefill and params scripts read the checkpoints from the Hugging Face cache, so download Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B first with `hf download`.
