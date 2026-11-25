# Qwen3-MoE in Bumblebee: evidence

This repository backs the changes that let Elixir run Qwen3 mixture-of-experts models with Bumblebee:

- Bumblebee: a mixture of experts layer and the Qwen3-MoE model, on branch `qwen3-moe`, and the loading of MLX 4-bit conversions with packed experts, on branch `qwen3-moe-mlx`, which stacks on `mlx-quantized` ([`bumblebee.patch`](bumblebee/bumblebee.patch)).
- EMLX: products with experts taken from a 4-bit tensor compile to MLX's `gather_qmm`, on branch `compiled-gather-qmm`, and loop bodies compile with MLX's compiler on the GPU, on branch `compiled-while-body` on top of it; both stack on `checkpoint-quantized` ([`emlx.patch`](bumblebee/emlx.patch)).

The layer gathers only the chosen experts when the tokens times the experts per token are at most the number of experts, which is the decoding case, and whenever the expert weights are quantized; otherwise it computes every expert densely.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) records the outputs of `optimum-internal-testing/tiny-random-qwen3_moe` with transformers 5.23.0, and [`bumblebee.exs`](bumblebee/bumblebee.exs) compares Bumblebee on EXLA with them ([`results.md`](bumblebee/outputs/results.md)):

| Output | Max abs difference |
|---|---|
| logits of 10 tokens, every expert computed | 2.9e-6 |
| logits of 2 tokens, the chosen experts gathered | 4.8e-7 |
| 8 greedy tokens | identical |

[`tiny_mlx.py`](bumblebee/tiny_mlx.py) converts the same checkpoint to MLX 4 bits with mlx-lm 0.32.2 and records mlx-lm's logits in [`tiny-mlx-reference.json`](bumblebee/outputs/tiny-mlx-reference.json); the MLX quantization tests check Bumblebee on EMLX against them ([`tests.txt`](bumblebee/outputs/tests.txt), run by [`tests.sh`](bumblebee/tests.sh)).

## A real checkpoint: Qwen3-30B-A3B in 4 bits

[`reference_30b.py`](bumblebee/reference_30b.py) decodes 100 tokens greedily with `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` in mlx-lm from two chat prompts, and [`bumblebee_30b.exs`](bumblebee/bumblebee_30b.exs) does the same in Bumblebee on EMLX, in bfloat16 ([`results-30b.md`](bumblebee/outputs/results-30b.md)):

| Prompt | Same top choice as mlx-lm, teacher forced | Greedy tokens identical | Bumblebee decode | mlx-lm decode |
|---|---|---|---|---|
| capital | 98 of 100 | first 37 | 15.4 ms | 9.3 ms |
| story | 95 of 100 | first 40 | 15.6 ms | 9.3 ms |

Where the top choices differ, mlx-lm's margin between its top two tokens is at most 1.25, and 0.125 in four of the seven positions.
The tokenizer gives the same ids as mlx-lm for both prompts.

Before `compiled-gather-qmm`, each expert product went through `Nx.runtime_call` and decode took 4,264 ms per token; with it, 20.3 ms; with `compiled-while-body` too, 15.4 ms.
I have not profiled where the remaining 6 ms per token go.

## Not checked

- Qwen3-30B-A3B in bfloat16 against transformers: it needs about 60 GB and does not fit.
- The 235B model.

## Reproduce

```sh
cd bumblebee
export MLX_ENABLE_TF32=0
uv run reference.py
BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
uv run tiny_mlx.py
BUMBLEBEE_PATH=path/to/bumblebee MLX_EVIDENCE_PATH=path/to/bumblebee-mlx-quantized-evidence ./tests.sh
uv run reference_30b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_30b.exs
```

The 30B scripts read the checkpoint from the Hugging Face cache, so download `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` first with `hf download`.
