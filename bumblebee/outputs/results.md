# Bumblebee against transformers 5.23.0

| Checkpoint | Output | Result | Detail |
|---|---|---|---|
| tiny-random-Gemma4ForCausalLM | generated | identical | [835, 952, 952, 952, 952, 952, 952, 952] |
| tiny-random-Gemma4ForCausalLM | logits | within 1e-4 | max abs difference 8.0e-7 |
| tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true | logits | within 1e-4 | max abs difference 3.7e-6 |
| tiny-random-Gemma4ForConditionalGeneration | logits | within 1e-4 | max abs difference 9.0e-7 |
| tiny-random-Gemma4TextModel | hidden_state | within 1e-4 | max abs difference 1.5e-6 |
