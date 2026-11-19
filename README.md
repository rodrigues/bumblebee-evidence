# Gemma 4 in Bumblebee: evidence

This repository backs two changes that let Elixir run Gemma 4 with Bumblebee:

- Bumblebee: the Gemma 4 text model, on branch [`gemma4`](https://github.com/rodrigues/bumblebee/tree/gemma4) of rodrigues/bumblebee ([`bumblebee.patch`](bumblebee/bumblebee.patch)).
- safetensors: loading tensors over 2 GiB on macOS, [elixir-nx/safetensors#16](https://github.com/elixir-nx/safetensors/pull/16) ([`safetensors.patch`](safetensors/safetensors.patch)).

Every result is in an `outputs/` folder, so nothing needs running to read it; [Reproduce](#reproduce), at the end, reruns everything.
All runs were on macOS (Apple silicon, 64 GiB), with Elixir 1.20.5 and OTP 29.1.2.

## Bumblebee matches transformers

[`reference.py`](bumblebee/reference.py) builds four tiny random checkpoints in [`checkpoints/`](bumblebee/checkpoints) with transformers 5.23.0 and records their outputs in [`reference.json`](bumblebee/outputs/reference.json).
[`bumblebee.exs`](bumblebee/bumblebee.exs) runs the same checkpoints through Bumblebee and compares them in [`results.md`](bumblebee/outputs/results.md).

| Checkpoint | Covers |
|---|---|
| [tiny-random-Gemma4TextModel](bumblebee/checkpoints/tiny-random-Gemma4TextModel) | the `:base` architecture |
| [tiny-random-Gemma4ForCausalLM](bumblebee/checkpoints/tiny-random-Gemma4ForCausalLM) | the `:for_causal_language_modeling` architecture, and 8 tokens of greedy generation through the cache |
| [tiny-random-Gemma4ForConditionalGeneration](bumblebee/checkpoints/tiny-random-Gemma4ForConditionalGeneration) | a multimodal checkpoint, with the language model under `model.language_model` |
| [tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true](bumblebee/checkpoints/tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true) | global layers that use the keys as values, as in 31B and 12B |

The first three have sliding and global layers, a sliding window shorter than the input, two layers that reuse earlier keys and values, per-layer inputs and the double-wide feed-forward network.
Every output is within 3.7e-6 of transformers, and generation is identical token for token.
The Bumblebee test file on the branch checks the same values, and passes on EXLA.
`reference.py` is deterministic: a rerun in a fresh environment gives byte-identical checkpoints.

## A real checkpoint: E4B

[`e4b.exs`](bumblebee/e4b.exs) loads `google/gemma-4-E4B-it` on EMLX and records [`e4b.txt`](bumblebee/outputs/e4b.txt):

- It loads in about 4 s, with no missing parameters and no shape mismatches.
- The unused parameters are the audio and vision towers, and the key, value and key norm weights of layers 24 to 41, which reuse earlier keys and values; transformers does not use those weights either.
- The tokenizer is detected as `:gemma`, and its ids match transformers' ([`tokenizer.py`](bumblebee/tokenizer.py), [`tokenizer-reference.json`](bumblebee/outputs/tokenizer-reference.json)).
- Greedy generation answers "The capital of France is Paris." and "A capital do Brasil é Brasília."

## Model initialization

The decoder's blocks get the results of earlier blocks as an argument of the block function.
An earlier version kept them in the `block` map, and `Axon.trace_init` took 15 s at 9 blocks ([`trace_init-before-fix-profile.txt`](bumblebee/outputs/trace_init-before-fix-profile.txt)).
The current version takes 37 ms or less up to 42 blocks ([`trace_init.exs`](bumblebee/trace_init.exs), [`trace_init.txt`](bumblebee/outputs/trace_init.txt)).

## safetensors

On macOS, reading a tensor of 2 GiB or more failed with `{error, einval}`, because macOS refuses a `pread` larger than `INT_MAX`.
With EMLX, a tensor with more than 2^31 - 1 elements also failed, because it was built flat on the target backend.
The fix reads in chunks only when the single read fails, and builds each tensor on `Nx.BinaryBackend` before moving it to the default backend.
The chunked read briefly holds about twice the tensor's size: 6.5 GB at peak for a 3 GiB file ([`memory.sh`](safetensors/memory.sh), [`memory.txt`](safetensors/outputs/memory.txt)).
The safetensors test suite passes ([`tests.txt`](safetensors/outputs/tests.txt)).

## Not checked

- Linux and Windows.
- The real E2B, 12B and 31B checkpoints; the keys-as-values path of 31B and 12B is covered by the tiny checkpoint only.
- 12B's architecture, `Gemma4UnifiedForConditionalGeneration`, which Bumblebee does not map.
- The mixture-of-experts 26B-A4B, which Bumblebee refuses on load, and the vision and audio towers.

## Reproduce

```sh
cd bumblebee
uv run reference.py
BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
uv run tokenizer.py
BUMBLEBEE_PATH=path/to/bumblebee elixir e4b.exs > outputs/e4b.txt 2>&1
BUMBLEBEE_PATH=path/to/bumblebee elixir trace_init.exs > outputs/trace_init.txt

cd ../safetensors
./memory.sh
```

`e4b.exs` needs macOS for EMLX and about 30 GB of memory.
