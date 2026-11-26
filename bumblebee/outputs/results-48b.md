# mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 100 greedy tokens against mlx-lm | Same top choice as mlx-lm, teacher forced | Bumblebee prefill | mlx-lm prefill | Bumblebee decode per token | mlx-lm decode per token |
|---|---|---|---|---|---|---|
| capital | identical | 100 of 100 | 98.1 ms | 75.9 ms | 13.8 ms | 9.5 ms |
| story | same first 4 | 89 of 100 (mlx-lm margins where they differ: [0.125, 0.125, 0.25, 0.25, 0.375, 0.0, 0.25, 0.375, 0.125, 0.25, 0.5]) | 86.4 ms | 66.5 ms | 13.6 ms | 9.5 ms |
