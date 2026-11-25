# Bumblebee against transformers 5.23.0

| Checkpoint | Output | Result |
|---|---|---|
| hf-tiny-v2/tiny-random-Ministral3ForCausalLM | logits | max abs difference 3.0e-08 |
| hf-tiny-v2/tiny-random-Ministral3ForCausalLM | logits at positions 16380 to 16389 | max abs difference 2.2e-08 |
| hf-tiny-v2/tiny-random-Ministral3ForCausalLM | logits at positions 16380 to 16389, sharper queries | max abs difference 3.0e-08 |
| hf-tiny-v2/tiny-random-Ministral3ForCausalLM | 8 greedy tokens | identical |
| hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration | logits | max abs difference 2.2e-08 |
