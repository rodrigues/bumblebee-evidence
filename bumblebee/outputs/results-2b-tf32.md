# Qwen/Qwen3.5-2B in Bumblebee against transformers 5.23.0, with MLX_ENABLE_TF32=1

| Run | Prompt | 30 greedy tokens against transformers f32 | Time per token |
|---|---|---|---|
| EXLA, CPU, f32 | thinking=false | identical | 56.2 ms |
| EXLA, CPU, f32 | thinking=true | identical | 56.4 ms |
| EMLX, GPU, f32 | thinking=false | identical | 31.0 ms |
| EMLX, GPU, f32 | thinking=true | identical | 30.9 ms |
| EMLX, GPU, bf16 | thinking=false | identical | 19.2 ms |
| EMLX, GPU, bf16 | thinking=true | identical | 19.1 ms |

- thinking=false: tokenizer ids identical
- thinking=true: tokenizer ids identical

Transformers output:

- thinking=false: "The capital of France is Paris.<|im_end|>\n<|endoftext|><|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n"
- thinking=true: "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Question: \"What is the capital of France?\"\n    *"
