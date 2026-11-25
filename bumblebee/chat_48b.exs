# Answers a chat prompt with mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit entirely in Elixir, with Bumblebee's own tokenizer, converted from the repository's tiktoken vocabulary,
# on EMLX, greedily, and writes the prompt, the answer and the token ids to outputs/chat-48b.json.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir chat_48b.exs
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

backend = {EMLX.Backend, device: :gpu}
Nx.global_default_backend(backend)
repo = {:hf, "mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit"}

{:ok, model_info} = Bumblebee.load_model(repo, backend: backend, type: :bf16)
{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 60, strategy: %{type: :greedy_search})

serving =
  Bumblebee.Text.generation(model_info, tokenizer, generation_config,
    compile: [batch_size: 1, sequence_length: 64],
    defn_options: [compiler: EMLX]
  )

# The chat template of the repository, written out
prompt = "<|im_user|>user<|im_middle|>What is the capital of France? Answer in one sentence.<|im_end|><|im_assistant|>assistant<|im_middle|>"
%{results: [%{text: text}]} = Nx.Serving.run(serving, prompt)

ids = Bumblebee.configure(tokenizer, add_special_tokens: false) |> Bumblebee.apply_tokenizer(prompt) |> Map.fetch!("input_ids") |> Nx.to_flat_list()
File.write!(Path.join(__DIR__, "outputs/chat-48b.json"), Jason.encode!(%{prompt: prompt, input_ids: ids, answer: text}, pretty: true))
IO.puts(text)
