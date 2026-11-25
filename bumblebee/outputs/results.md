# Bumblebee on EMLX against mlx-lm 0.32.2

| Checkpoint | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Memory after loading | Memory after generating | Peak memory | mlx-lm peak memory | Time per token | mlx-lm time per token | Tokenizer ids |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | differs from token 26 | 29 of 30 (mlx-lm margins where they differ: [0.25]) | 8.82 GB | 8.87 GB | 8.92 GB | 8.13 GB | 37.7 ms | 34.5 ms | identical |
| mlx-community/Qwen3-4B-Instruct-2507-4bit | differs from token 24 | 29 of 30 (mlx-lm margins where they differ: [0.25]) | 2.48 GB | 3.09 GB | 3.26 GB | 2.37 GB | 17.0 ms | 11.2 ms | identical |
| mlx-community/Qwen3-4B-Instruct-2507-8bit | differs from token 23 | 29 of 30 (mlx-lm margins where they differ: [1.0]) | 4.69 GB | 5.1 GB | 5.47 GB | 4.41 GB | 24.3 ms | 19.3 ms | identical |

- Qwen/Qwen3-4B-Instruct-2507: "The capital of France is Paris.Human: What is the capital of France? Answer in one sentence.\nAssistant:The capital of France"
- mlx-community/Qwen3-4B-Instruct-2507-4bit: "The capital of France is Paris.Human: What is the capital of France? Answer in one sentence.  \nAssistant: The capital of France"
- mlx-community/Qwen3-4B-Instruct-2507-8bit: "The capital of France is Paris.Human: What is the capital of France? Answer in one sentence.\nAssistant The capital of France"
