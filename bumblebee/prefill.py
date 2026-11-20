# /// script
# dependencies = ["mlx-lm==0.32.2"]
# ///
import time, mlx.core as mx
from mlx_lm import load
model, _ = load("Qwen/Qwen3.5-2B")
for n in [128, 512, 2048, 8192]:
    x = (mx.arange(n) % 1000 + 1000)[None]
    mx.eval(model(x)[0, -1, 0])
    t = time.perf_counter(); mx.eval(model(x)[0, -1, 0]); dt = time.perf_counter() - t
    print(f"mlx-lm {n} tokens: {dt*1000:.0f} ms")
