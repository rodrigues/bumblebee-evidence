# MLX quantized checkpoints in Bumblebee: evidence

This repository backs two changes that let Bumblebee run checkpoints quantized with MLX, such as the ones under [mlx-community](https://huggingface.co/mlx-community), on EMLX without dequantizing them:

- EMLX: transposing a quantized tensor, and the result type of quantized products, on branch [`checkpoint-quantized`](https://github.com/rodrigues/emlx/tree/checkpoint-quantized) of rodrigues/emlx ([`emlx.patch`](emlx/emlx.patch)).
- Bumblebee: loading the quantized weights, on branch [`mlx-quantized`](https://github.com/rodrigues/bumblebee/tree/mlx-quantized) of rodrigues/bumblebee ([`bumblebee.patch`](bumblebee/bumblebee.patch)).

Every result is in an `outputs/` folder, so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5, OTP 29.1.2 and mlx-lm 0.32.2.

## Bumblebee matches mlx-lm

[`reference.py`](bumblebee/reference.py) decodes 30 tokens greedily with mlx-lm from the 4-bit and 8-bit conversions of Qwen3-4B-Instruct-2507, and from the original bfloat16 checkpoint, and records each position's top choice when the whole sequence is fed at once.
[`bumblebee.exs`](bumblebee/bumblebee.exs) does the same in Bumblebee on EMLX, in bfloat16 ([`results.md`](bumblebee/outputs/results.md)):

| Checkpoint | 30 greedy tokens | Top choices, teacher forced | Memory after generating | Peak memory | mlx-lm peak memory | Time per token | mlx-lm time per token |
|---|---|---|---|---|---|---|---|
| mlx-community/Qwen3-4B-Instruct-2507-4bit | follow mlx-lm to token 24 | 29 of 30 | 3.09 GB | 3.26 GB | 2.37 GB | 17.3 ms | 11.0 ms |
| mlx-community/Qwen3-4B-Instruct-2507-8bit | follow mlx-lm to token 23 | 29 of 30 | 5.10 GB | 5.47 GB | 4.41 GB | 24.6 ms | 18.3 ms |
| Qwen/Qwen3-4B-Instruct-2507, bfloat16 | follow mlx-lm to token 26 | 29 of 30 | 8.87 GB | 8.92 GB | 8.13 GB | 37.8 ms | 42.0 ms |

The tokenizer gives the same ids as mlx-lm.
Bumblebee uses about 0.7 GB more than mlx-lm with the quantized checkpoints because it dequantizes the embedding table.

In bfloat16, each run follows mlx-lm up to one token where mlx-lm's own top two logits differ by 1.0 or less, and the unquantized checkpoint does the same.
With float32 activations on both sides, [`reference_f32.py`](bumblebee/reference_f32.py) and [`bumblebee_f32.exs`](bumblebee/bumblebee_f32.exs) give identical tokens and identical top choices for both quantized checkpoints ([`results-f32.md`](bumblebee/outputs/results-f32.md)).

## Tiny checkpoints for the tests

[`tiny.py`](bumblebee/tiny.py) converts `bumblebee-testing/tiny-random-Qwen3ForCausalLM` to MLX in 4 and 8 bits with group size 32, into [`checkpoints/`](bumblebee/checkpoints), and records mlx-lm's logits in [`tiny-reference.json`](bumblebee/outputs/tiny-reference.json).
Its down projection has 37 inputs, which is not a multiple of the group size, so mlx-lm leaves it unquantized, and the tests check that a checkpoint can mix both.

The Bumblebee tests check that every quantized kernel gives the same products as its dequantized weights, against an exact computation on `Nx.BinaryBackend`.
The quantized products are within 3.2e-7 of the exact result.
The reference is computed on the binary backend because EMLX's plain float32 matrix products, with the same kernels dequantized, are off by up to 7.4e-4 once the input has more than one row ([`precision.exs`](bumblebee/precision.exs), [`precision.md`](bumblebee/outputs/precision.md)).

## EMLX

The EMLX quantization tests pass ([`quantization-tests.txt`](emlx/outputs/quantization-tests.txt)), and so do the Bumblebee tests, run against the local checkpoints on a branch that also has Qwen3.5 ([`tests.txt`](bumblebee/outputs/tests.txt)).
In the full suite, ten tests in attention with sinks, async evaluation, linear algebra and the fused attention cache fail on the base commit as well as on the branch ([`failing-tests-base-and-branch.txt`](emlx/outputs/failing-tests-base-and-branch.txt)).
The test that compiles a chain of 20,000 operations on the CPU device runs the C++ compiler for over 50 minutes on this machine, so it was left out of that comparison.

## Not checked

- Quantization modes other than affine, which Bumblebee refuses on load.
- Linux, where EMLX does not run.
- Quantized weights outside dense layers staying quantized; embedding tables and any weight with a custom loader are dequantized on load.

## Reproduce

```sh
cd bumblebee
uv run reference.py
uv run tiny.py
uv run reference_f32.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee.exs
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_f32.exs
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir precision.exs
```

The Elixir scripts need macOS on Apple silicon for EMLX, and the bfloat16 checkpoint needs about 10 GB of memory.
