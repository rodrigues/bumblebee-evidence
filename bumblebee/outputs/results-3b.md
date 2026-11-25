# mistralai/Ministral-3-3B-Instruct-2512-BF16 in Bumblebee against transformers 5.23.0

| Run | Prompt | 30 greedy tokens against transformers f32 | Time per token |
|---|---|---|---|
| EXLA, CPU, f32 | capital | identical | 202.1 ms |
| EXLA, CPU, f32 | haiku | identical | 208.4 ms |
| EMLX, GPU, f32 | capital | identical | 79.3 ms |
| EMLX, GPU, f32 | haiku | identical | 79.9 ms |
| EMLX, GPU, bf16 | capital | identical | 38.7 ms |
| EMLX, GPU, bf16 | haiku | differs from token 17, where the transformers margin is 0.018 | 38.8 ms |

- capital: tokenizer ids identical
- haiku: tokenizer ids identical

Transformers output:

- capital: "The capital of France is **Paris**.</s></s></s></s></s></s></s></s></s></s>The capital of France is Paris. (repeated)\n\n"
- haiku: "Ocean’s breath,\nsilent waves kiss the shore—\ndreamers’ whispers rise.</s></s>*(or, if you prefer a different"
