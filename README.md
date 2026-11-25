# Kimi Linear in Bumblebee: evidence

This repository backs the change that lets Elixir run Kimi Linear with Bumblebee, on branch `kimi-linear` ([`bumblebee.patch`](bumblebee/bumblebee.patch)), which stacks on the `llama4` branch for its mixture of experts layer.
Kimi Linear interleaves two kinds of attention, and Bumblebee implements both:

- Kimi Delta Attention, in three of every four blocks: the gated delta rule of Qwen3.5's linear attention, with a decay per key channel instead of per head, short convolutions over queries, keys and values, and an output norm gated by a sigmoid.
- Multi-head latent attention, in the rest, without position embedding: keys and values are projected from one compressed vector per token, and the cache keeps those compressed vectors.

Every block after the first has 256 experts, 8 per token, chosen by sigmoid scores plus a selection bias, weighted by the scores alone, renormalized and scaled by 2.446, beside a shared expert.
The mixture of experts layer gains two options for this, the selection bias and the output scale.
Bumblebee reads Moonshot's original weight names and those of the MLX conversion.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) builds a tiny random checkpoint in [`checkpoints/`](bumblebee/checkpoints) with transformers 5.23.0, stored under the original weight names, and records its outputs in [`reference.json`](bumblebee/outputs/reference.json).
It has four delta attention blocks and one latent attention block, a dense first block, and every weight, bias and selection bias set to random values.
The script checks that the stored checkpoint reloads in transformers to the same outputs, so the original names are right.

The Bumblebee test file on the branch checks the hidden state, the logits in both chunks of the chunked delta rule, 8 greedy tokens through the cache, and that left padding leaves the real tokens' logits unchanged; all pass on EXLA ([`tests.txt`](bumblebee/outputs/tests.txt), run by [`tests.sh`](bumblebee/tests.sh)).
The logits are within 8.8e-6 of transformers, and the generated tokens are identical.
The checkpoint loads with no missing parameters, no shape mismatches and no unused tensors ([`params.txt`](bumblebee/outputs/params.txt), from [`params.exs`](bumblebee/params.exs)).

The published tiny checkpoint, yujiepan/kimi-linear-tiny-random, does not load in transformers 5.23.0: it lists 5 layers but names layer 5 in neither layer list, so transformers rejects its layer types.

## A real checkpoint: Kimi-Linear-48B-A3B-Instruct in 4 bits

[`reference_48b.py`](bumblebee/reference_48b.py) decodes 100 tokens greedily with `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` in mlx-lm 0.32.2 from two chat prompts, and [`bumblebee_48b.exs`](bumblebee/bumblebee_48b.exs) does the same in Bumblebee on EMLX, in bfloat16 ([`results-48b.md`](bumblebee/outputs/results-48b.md)):

| Prompt | Same top choice as mlx-lm, teacher forced | Greedy tokens identical | Bumblebee decode | mlx-lm decode |
|---|---|---|---|---|
| capital | 100 of 100 | all 100 | 15.3 ms | 9.5 ms |
| story | 89 of 100 | first 4 | 15.4 ms | 9.5 ms |

Where the top choices differ, mlx-lm's margin between its top two tokens is at most 0.5, and the story diverges at a margin of 0.125; both continuations are coherent stories.
The checkpoint loads with nothing missing, mismatched or unused: [`params-48b.txt`](bumblebee/outputs/params-48b.txt) is empty because the load logged no differences.
The EMLX used has the `compiled-while-body` branch, which compiles the generation loop's body on the GPU.

Bumblebee cannot load this model's tokenizer, which ships as `tiktoken.model` with Python code and no `tokenizer.json`, so the Bumblebee script takes the token ids that mlx-lm's tokenizer gave, from the reference.

## Not checked

- The tokenizer, which Bumblebee does not load.
- The bfloat16 checkpoint against transformers: it is about 98 GB and does not fit.
- Expert groups, which neither the Instruct checkpoint nor its MLX conversion uses; Bumblebee refuses a config with more than one.
- Prompts long enough to fill many chunks of the delta rule on the real checkpoint.

## Reproduce

```sh
cd bumblebee
export MLX_ENABLE_TF32=0
uv run reference.py
BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
(cd path/to/bumblebee && MIX_ENV=test mix run "$OLDPWD/params.exs")
uv run reference_48b.py
BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_48b.exs
```

The 48B scripts read the checkpoint from the Hugging Face cache, so download `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` first with `hf download`.
