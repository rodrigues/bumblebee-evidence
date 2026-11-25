# Ministral 3 in Bumblebee: evidence

This repository backs the change that lets Elixir run Ministral 3 text models with Bumblebee, on branch `ministral3` ([`bumblebee.patch`](bumblebee/bumblebee.patch)).
Ministral 3's text model is the Mistral decoder with three changes: a head size set apart from the hidden size, yarn rotary scaling, and queries scaled by 1 + 0.1 · log(1 + floor(position / 16384)), which is 1 until position 16,384.
Its checkpoints are multimodal, `Mistral3ForConditionalGeneration`, and Bumblebee loads their language model.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) records the outputs of `hf-tiny-v2/tiny-random-Ministral3ForCausalLM` and `hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration` with transformers 5.23.0, and [`bumblebee.exs`](bumblebee/bumblebee.exs) compares Bumblebee on EXLA with them ([`results.md`](bumblebee/outputs/results.md)):

| Checkpoint | Output | Max abs difference |
|---|---|---|
| Ministral3ForCausalLM | logits | 3.0e-8 |
| Ministral3ForCausalLM | logits at positions 16,380 to 16,389 | 2.2e-8 |
| Ministral3ForCausalLM | the same, with the query projection scaled by 30 | 3.0e-8 |
| Ministral3ForCausalLM | 8 greedy tokens | identical |
| Mistral3ForConditionalGeneration | logits | 2.2e-8 |

Random weights make attention nearly uniform, so the query scale barely moves the logits.
Scaling the query projection by 30 sharpens attention, and the query scale then moves the logits by 2.3e-3, in transformers and in Bumblebee alike.

The Bumblebee tests check these, and also an original context of 8 positions over 20 tokens, where the query scale changes the logits by 1.2e-4 and the test tolerance is 1.0e-6 ([`tests.txt`](bumblebee/outputs/tests.txt)).

## A real checkpoint: Ministral-3-3B-Instruct

[`reference_3b.py`](bumblebee/reference_3b.py) decodes 30 tokens greedily with `mistralai/Ministral-3-3B-Instruct-2512-BF16` in transformers, in float32, from two chat prompts, and [`bumblebee_3b.exs`](bumblebee/bumblebee_3b.exs) does the same in Bumblebee ([`results-3b.md`](bumblebee/outputs/results-3b.md)):

| Run | capital | haiku | Time per token |
|---|---|---|---|
| EXLA, CPU, float32 | identical | identical | 202 to 208 ms |
| EMLX, GPU, float32 | identical | identical | 79 to 80 ms |
| EMLX, GPU, bfloat16 | identical | differs from token 17 | 39 ms |

Where bfloat16 differs, transformers' float32 margin between its top two tokens is 0.018.
The tokenizer gives the same ids as transformers for both prompts.
The checkpoint loads with no missing parameters and no shape mismatches, and the only unused tensors are the 129 of the vision tower and projector ([`params.exs`](bumblebee/params.exs), [`params.txt`](bumblebee/outputs/params.txt)).

## Not checked

- The repositories without the -BF16 suffix, whose weights are FP8 with one scale per tensor.
- The 8B and 14B sizes, and the Base and Reasoning variants, which use the same architecture.
- Positions past 16,384 on the real checkpoint.
- The vision tower.

## Reproduce

```sh
cd bumblebee
export MLX_ENABLE_TF32=0
uv run reference.py
BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
uv run reference_3b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_3b.exs
BUMBLEBEE_PATH=path/to/bumblebee elixir params.exs
```

The 3B scripts read the checkpoint from the Hugging Face cache, so download `mistralai/Ministral-3-3B-Instruct-2512-BF16` first with `hf download`.
