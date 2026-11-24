# Bumblebee against transformers 5.23.0

| Checkpoint | Output | Result | Detail |
|---|---|---|---|
| tiny-random-Qwen3_5ForCausalLM | generated | identical | [279, 537, 784, 485, 860, 775, 279, 973] |
| tiny-random-Qwen3_5ForCausalLM | hidden_state | within 1e-4 | max abs difference 3.2e-06 |
| tiny-random-Qwen3_5ForCausalLM | logits | within 1e-4 | max abs difference 2.7e-06 |
| tiny-random-Qwen3_5ForConditionalGeneration | logits | within 1e-4 | max abs difference 4.1e-06 |
| tiny-random-Qwen3_5TextModel | hidden_state | within 1e-4 | max abs difference 3.0e-06 |
| tiny-random-Qwen3_5ForCausalLM | logits, 150 tokens | within 1e-4 | max abs difference 1.4e-05 |
