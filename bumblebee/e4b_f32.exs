# Decodes 30 tokens greedily with google/gemma-4-E4B-it in f32 on EXLA, from the Bumblebee checkout at BUMBLEBEE_PATH, and writes the text to the file OUT.
# Run once from the gemma4 branch and once from gemma4-mlx, to show the MLX quantized loading leaves the f32 model unchanged:
# BUMBLEBEE_PATH=path/to/bumblebee OUT=outputs/e4b-f32-gemma4.json elixir e4b_f32.exs
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:safetensors, github: "elixir-nx/safetensors", branch: "main", override: true},
  {:exla, "1.0.0"},
  {:fine, "0.1.6", override: true},
  {:nx, "~> 1.0", override: true},
  {:jason, "~> 1.4"}
])

Nx.global_default_backend(EXLA.Backend)
repo = {:hf, "google/gemma-4-E4B-it"}
{:ok, model_info} = Bumblebee.load_model(repo, type: :f32)
{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation} = Bumblebee.load_generation_config(repo)
generation = Bumblebee.configure(generation, max_new_tokens: 30, strategy: %{type: :greedy_search})
serving = Bumblebee.Text.generation(model_info, tokenizer, generation, compile: [batch_size: 1, sequence_length: 64], defn_options: [compiler: EXLA])
prompt = "<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n"
%{results: [%{text: text}]} = Nx.Serving.run(serving, prompt)
File.write!(System.fetch_env!("OUT"), Jason.encode!(%{text: text}))
IO.puts(text)
