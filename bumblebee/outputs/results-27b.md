# lmstudio-community/Qwen3.8-27B-MLX-4bit in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Time per token | mlx-lm time per token | Tokenizer ids |
|---|---|---|---|---|---|
| thinking=false | identical | 30 of 30 | 76.9 ms | 66.5 ms | identical |
| thinking=true | differs from token 25 | 29 of 30 (mlx-lm margins where they differ: [0.125]) | 75.5 ms | 95.0 ms | identical |

| | Bumblebee | mlx-lm |
|---|---|---|
| Memory after loading | 15.13 GB | 15.13 GB |
| Memory after generating | 17.05 GB | |
| Peak memory while generating | 17.68 GB | 15.52 GB |
| Load time | 2.6 s | |

- thinking=false
  - mlx-lm: "The capital of France is Paris.<|im_end|>\n<|endoftext|><|im_start|><|im_start|><|im_start|><|im_start|><|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|><|im_start|><think>\nThe user is asking a"
  - Bumblebee: "The capital of France is Paris.\n\nuser\n\n<think>\nThe user is asking a"
- thinking=true
  - mlx-lm: "The user asks: \"What is the capital of France? Answer in one sentence.\" This is a simple factual question. The capital of France is Paris"
  - Bumblebee: "The user asks: \"What is the capital of France? Answer in one sentence.\" This is a simple factual question. The answer is Paris. I"
