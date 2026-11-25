# Gemma 4 31B, MLX 4-bit, in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Time per token | mlx-lm time per token | Tokenizer ids |
|---|---|---|---|---|---|
| capital | identical | 30 of 30 | 78.9 ms | 75.9 ms | identical |
| story | identical | 30 of 30 | 78.6 ms | 74.7 ms | identical |

| | Bumblebee | mlx-lm |
|---|---|---|
| Load time | 2.0 s | |
| Memory after loading | 18.06 GB | 17.27 GB |
| Memory after generating | 20.17 GB | |
| Peak memory while generating | 20.88 GB | 17.62 GB |

Parameter log: ignored, because of non-matching shape 0 %{}; missing 0 %{}; unused 0 %{}

A prompt of 1271 tokens, 30 tokens with the cache against the full sequence each step: identical
