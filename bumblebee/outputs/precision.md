# EMLX float32 matrix products against the exact result

| Kernel | Input rows | Quantized kernel | Same kernel dequantized |
|---|---|---|---|
| decoder.blocks.0.ffn.gate | 1 | 8.2e-8 | 7.0e-9 |
| decoder.blocks.0.ffn.gate | 8 | 2.2e-8 | 1.51202e-4 |
| decoder.blocks.0.ffn.intermediate | 1 | 6.9e-8 | 1.5e-8 |
| decoder.blocks.0.ffn.intermediate | 8 | 3.0e-8 | 2.16395e-4 |
| decoder.blocks.0.self_attention.key | 1 | 9.1e-8 | 1.5e-8 |
| decoder.blocks.0.self_attention.key | 8 | 3.0e-8 | 2.53081e-4 |
| decoder.blocks.0.self_attention.output | 1 | 3.13e-7 | 3.7e-8 |
| decoder.blocks.0.self_attention.output | 8 | 1.19e-7 | 7.34031e-4 |
| decoder.blocks.0.self_attention.query | 1 | 1.04e-7 | 1.5e-8 |
| decoder.blocks.0.self_attention.query | 8 | 3.0e-8 | 3.0452e-4 |
| decoder.blocks.0.self_attention.value | 1 | 8.5e-8 | 1.5e-8 |
| decoder.blocks.0.self_attention.value | 8 | 3.0e-8 | 2.02864e-4 |
| decoder.blocks.1.ffn.gate | 1 | 8.2e-8 | 7.0e-9 |
| decoder.blocks.1.ffn.gate | 8 | 3.0e-8 | 1.34498e-4 |
| decoder.blocks.1.ffn.intermediate | 1 | 7.5e-8 | 7.0e-9 |
| decoder.blocks.1.ffn.intermediate | 8 | 2.2e-8 | 1.85937e-4 |
| decoder.blocks.1.self_attention.key | 1 | 6.9e-8 | 1.5e-8 |
| decoder.blocks.1.self_attention.key | 8 | 3.0e-8 | 2.21878e-4 |
| decoder.blocks.1.self_attention.output | 1 | 2.98e-7 | 4.5e-8 |
| decoder.blocks.1.self_attention.output | 8 | 1.19e-7 | 7.44641e-4 |
| decoder.blocks.1.self_attention.query | 1 | 7.5e-8 | 1.5e-8 |
| decoder.blocks.1.self_attention.query | 8 | 3.0e-8 | 2.30625e-4 |
| decoder.blocks.1.self_attention.value | 1 | 8.2e-8 | 1.5e-8 |
| decoder.blocks.1.self_attention.value | 8 | 3.0e-8 | 1.85445e-4 |
| language_modeling_head.output | 1 | 1.06e-7 | 3.0e-8 |
| language_modeling_head.output | 8 | 3.0e-8 | 4.03672e-4 |
