# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx-lm==0.32.2"]
# ///
# Converts checkpoints/tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true, which has 31B's keys-as-values attention, to MLX 4 bits with group size 32,
# and writes mlx-lm's logits in full float32 to outputs/tiny-mlx-reference.json.
# Its intermediate size, 37, is not a multiple of 32, so mlx-lm leaves the down projections unquantized.
# Run with: MLX_ENABLE_TF32=0 uv run tiny_mlx.py
import glob, json, os, shutil, tempfile
import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import convert, load

here = os.path.dirname(os.path.abspath(__file__))
name = "tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true-mlx-4bit"
path = os.path.join(here, "checkpoints", name)
shutil.rmtree(path, ignore_errors=True)
# The conversion needs a tokenizer, which the tiny checkpoint lacks, so it borrows Gemma 4's
source = tempfile.mkdtemp()
shutil.copytree(os.path.join(here, "checkpoints", "tiny-random-Gemma4ForCausalLM-attention_k_eq_v-true"), source, dirs_exist_ok=True)
tokenizer = snapshot_download("google/gemma-4-E4B-it", allow_patterns=["tokenizer.json", "tokenizer_config.json"])
for file in glob.glob(os.path.join(tokenizer, "tokenizer*.json")):
    shutil.copy(file, source)
convert(source, mlx_path=path, dtype="float32", quantize=True, q_bits=4, q_group_size=32)
model, _ = load(path)
# Bumblebee's tests load only the model, so the borrowed tokenizer is not kept
for file in glob.glob(os.path.join(path, "tokenizer*.json")):
    os.remove(file)
logits = model(mx.array([[10, 20, 30, 40, 50, 60, 70, 80]])).astype(mx.float32)
reference = {name: {"logits": logits[0, 1:4, 1:4].tolist()}}
with open(os.path.join(here, "outputs", "tiny-mlx-reference.json"), "w") as f:
    json.dump(reference, f, indent=2)
print(json.dumps(reference))
