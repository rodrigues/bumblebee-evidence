# mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 100 greedy tokens against mlx-lm | Same top choice as mlx-lm, teacher forced | Bumblebee prefill | mlx-lm prefill | Bumblebee decode per token | mlx-lm decode per token |
|---|---|---|---|---|---|---|
| capital | same first 37 | 98 of 100 (mlx-lm margins where they differ: [0.5, 1.25]) | 107.9 ms | 79.5 ms | 15.4 ms | 9.3 ms |
| story | same first 40 | 95 of 100 (mlx-lm margins where they differ: [0.125, 0.125, 0.25, 0.125, 0.125]) | 105.4 ms | 74.5 ms | 15.6 ms | 9.3 ms |

- capital: tokenizer ids identical
- story: tokenizer ids identical
