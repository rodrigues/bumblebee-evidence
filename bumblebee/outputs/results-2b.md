# Qwen/Qwen3.5-2B in Bumblebee against transformers 5.23.0

| Run | Prompt | 30 greedy tokens against transformers f32 | Time per token |
|---|---|---|---|
| EXLA, CPU, f32 | thinking=false | identical | 57.7 ms |
| EXLA, CPU, f32 | thinking=true | identical | 57.2 ms |
| EMLX, GPU, f32 | thinking=false | identical | 39.3 ms |
| EMLX, GPU, f32 | thinking=true | identical | 38.6 ms |
| EMLX, GPU, bf16 | thinking=false | identical | 27.1 ms |
| EMLX, GPU, bf16 | thinking=true | identical | 26.5 ms |

- thinking=false: tokenizer ids identical
- thinking=true: tokenizer ids identical

Transformers output:

- thinking=false: "The capital of France is Paris.<|im_end|>\n<|endoftext|><|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n"
- thinking=true: "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Question: \"What is the capital of France?\"\n    *"
