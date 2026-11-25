# Llama-Guard-4-12B against transformers

Bumblebee on EMLX, GPU, bfloat16, against transformers 5.23.0 on the CPU in bfloat16.
The checkpoint's attention_chunk_size is nil; Bumblebee loads it as nil.
Parameter log: ["the following PyTorch parameters were unused:"] %{"multi_modal_projector.linear_1" => 1, "vision_model.class_embedding" => 1, "vision_model.layernorm_post" => 2, "vision_model.layernorm_pre" => 2, "vision_model.model" => 144}

| Prompt | Bumblebee output | Greedy tokens | Same top choice, teacher forced | Largest logit difference, first 3 positions by 5 | Time per token, incl. prefill |
|---|---|---|---|---|---|
| benign | "\n\nsafe" | identical | 3 of 3 | 0.5 | 136.9 ms |
| continuation | "" | identical | 1 of 1 | 0.4375 | 126.2 ms |
| lockpicking | "\n\nunsafe\nS2" | identical | 6 of 6 | 0.25 | 108.0 ms |
