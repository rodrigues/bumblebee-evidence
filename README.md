# Llama 4 in Bumblebee: evidence

This repository backs the change that lets Elixir run Llama 4 text models with Bumblebee, on branch `llama4` ([`bumblebee.patch`](bumblebee/bumblebee.patch)), which stacks on the `qwen3-moe` branch for its mixture of experts layer.
Llama 4 differs from Llama 3 in four ways, and Bumblebee implements each:

- Three of every four blocks attend within chunks of 8,192 tokens and use rotary embedding over adjacent pairs; Bumblebee reorders the query and key weights on load, so its usual rotary embedding gives the same result.
- In Scout, those blocks normalize queries and keys to unit root mean square.
- The fourth block attends to the whole sequence, without position embedding, and scales its queries by 1 + 0.1 · log(1 + floor((position + 1) / 8192)).
- The feed-forward networks are mixtures of experts whose router weights each expert's input by a sigmoid, beside a shared expert; Maverick puts them in every other block.

Llama 4 routes in its own model: it weights each chosen expert's copy of the token by the sigmoid, and passes the shared mixture of experts layer weights of 1.
Multimodal checkpoints, `Llama4ForConditionalGeneration`, load as their language model.

Every result is in [`bumblebee/outputs/`](bumblebee/outputs), so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple M5 Pro, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) builds two tiny random checkpoints in [`checkpoints/`](bumblebee/checkpoints) with transformers 5.23.0, and records their outputs and those of `optimum-intel-internal-testing/tiny-random-llama4` in [`reference.json`](bumblebee/outputs/reference.json).
The Bumblebee test file on the branch checks the same values, and all of them pass on EXLA ([`tests.txt`](bumblebee/outputs/tests.txt), run by [`tests.sh`](bumblebee/tests.sh)):

| Checkpoint | Covers |
|---|---|
| tiny-random-Llama4ForCausalLM | Scout's shape: experts in every block, one per token, normalized queries and keys; hidden state, logits, and 8 greedy tokens |
| tiny-random-Llama4ForCausalLM-interleaved | Maverick's shape: experts in every other block, without the normalization, here with two experts per token; logits and 8 greedy tokens |
| tiny-random-llama4 | a multimodal checkpoint, whose config sets the temperature flag to the number 4; logits |

The two built checkpoints attend in chunks of 4 tokens and step the query temperature every 2 positions, so a 12-token input crosses both, and generation from 6 tokens crosses them through the cache.
Turning off each feature in transformers moves the logits by 144 (chunks), 5.3 (temperature) and 79 (normalization), so the tests depend on all three.
The published tiny checkpoint has all its expert weights at zero, so the built checkpoints set every weight to random values.
The shared layer's tests, on `qwen3-moe`, check both expert paths against the output spelled out by hand.
The checkpoints load with no missing parameters and no shape mismatches; the multimodal one leaves only its vision tower and projector unused ([`params.txt`](bumblebee/outputs/params.txt)).

## A real checkpoint: Llama Guard 4

Llama Guard 4 12B is the smallest Llama 4 with trained weights I found among the 60 most downloaded on the Hub: a dense model, without experts or chunked attention, with normalized queries and keys in every block.
[`reference_guard.py`](bumblebee/reference_guard.py) runs `unsloth/Llama-Guard-4-12B`, an open copy of Meta's release, in transformers 5.23.0 in bfloat16 on the CPU, and [`bumblebee_guard.exs`](bumblebee/bumblebee_guard.exs) runs it in Bumblebee on EMLX in bfloat16 on the GPU ([`results-guard.md`](bumblebee/outputs/results-guard.md)):

| Prompt | Output | Greedy tokens | Same top choice, teacher forced |
|---|---|---|---|
| benign chat | safe | identical | 3 of 3 |
| lockpicking chat | unsafe, S2 | identical | 6 of 6 |
| plain continuation | end of text | identical | 1 of 1 |

The logits differ by at most 0.5 over the first positions, the bfloat16 rounding of logits that size.
Only the vision tower and the projector are left unused.

Its config sets attention_chunk_size to null while marking every block as chunked attention.
transformers 5.23.0 refuses a null chunk size, so the reference sets it to 8192, which is plain causal attention for prompts this short; Bumblebee now loads null as attention to the whole sequence, and a test checks that it matches chunks as long as the input.

## Not checked

- A real checkpoint with experts or chunked attention: the smallest, Scout, is 61.1 GB in MLX 4-bit and does not fit a 64 GB Mac. `inference-optimization/Llama-4-Scout-1.7B-0.4B-Instruct` has Scout's shape, but 75 of its 100 text tensors, the embeddings, output head and experts among them, are all zeros.
- Llama 4's MLX conversions. I have not checked how they store the experts, and loading them needs the reordering of query and key weights applied to quantized weights too.
- Left padding: transformers counts the temperature and chunk positions from the start of the cache, and Bumblebee does the same, which ignores padding in both.
- The vision tower.

## Reproduce

```sh
cd bumblebee
uv run reference.py
BUMBLEBEE_PATH=path/to/bumblebee ./tests.sh
(cd path/to/bumblebee && MIX_ENV=test mix run "$OLDPWD/params.exs")
uv run reference_guard.py
MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_guard.exs
```
