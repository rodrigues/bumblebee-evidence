# optimum-internal-testing/tiny-random-qwen3_moe in Bumblebee on EXLA against transformers 5.23.0

| Output | Max abs difference |
|---|---|
| logits of 10 tokens, every expert computed | 2.86102294921875e-6 |
| logits of 2 tokens, the chosen experts gathered | 4.76837158203125e-7 |
| 8 greedy tokens | identical |
