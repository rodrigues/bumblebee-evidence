# Loads google/gemma-4-E4B-it with Bumblebee on EMLX (macOS), logs the params diff, checks the tokenizer against outputs/tokenizer-reference.json and asks two questions.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir e4b.exs > outputs/e4b.txt 2>&1
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  # Loading tensors over 2 GiB on macOS, see the safetensors folder
  {:safetensors, github: "rodrigues/safetensors", branch: "large-tensors", override: true},
  {:emlx, github: "elixir-nx/emlx", ref: "d5c78ae5", sparse: "emlx"},
  {:fine, "0.1.6", override: true},
  {:nx, "~> 1.0", override: true},
  {:jason, "~> 1.4"}
])

# EMLX does not accept the :batch_keys option that Nx.Serving passes to the compiler
defmodule Compiler do
  @behaviour Nx.Defn.Compiler
  def __jit__(key, vars, fun, args, opts), do: EMLX.__jit__(key, vars, fun, args, Keyword.delete(opts, :batch_keys))
  def __compile__(key, vars, fun, opts), do: EMLX.__compile__(key, vars, fun, Keyword.delete(opts, :batch_keys))
  def __partitions_options__(opts), do: EMLX.__partitions_options__(opts)
  def __to_backend__(opts), do: EMLX.__to_backend__(opts)
end

Logger.configure(level: :debug)
:logger.update_handler_config(:default, :formatter, Logger.default_formatter(truncate: :infinity))
repo = {:hf, "google/gemma-4-E4B-it"}
backend = {EMLX.Backend, device: :gpu}
IO.puts("## Model")

{micro, {:ok, loaded}} =
  :timer.tc(fn -> Bumblebee.load_model(repo, type: :bf16, backend: backend, log_params_diff: true) end)

seconds = div(micro, 1_000_000)
IO.puts("loaded in #{seconds} s: #{inspect(loaded.spec.__struct__)} #{loaded.spec.architecture}")

{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
reference = __DIR__ |> Path.join("outputs/tokenizer-reference.json") |> File.read!() |> Jason.decode!()
IO.puts("\n## Tokenizer\ntype detected: #{inspect(tokenizer.type)}")
tokenizer = Bumblebee.configure(tokenizer, add_special_tokens: false)

for {prompt, expected} <- Enum.zip(reference["prompts"], reference["input_ids"]) do
  ids = Bumblebee.apply_tokenizer(tokenizer, prompt)["input_ids"] |> Nx.to_flat_list()
  verdict = if ids == expected, do: "identical to transformers", else: "DIFFERENT from transformers: #{inspect(expected)}"
  IO.puts("#{inspect(prompt)}\n  #{inspect(ids, limit: :infinity)}\n  #{verdict}")
end

{:ok, generation} = Bumblebee.load_generation_config(repo)
generation = Bumblebee.configure(generation, max_new_tokens: 60, strategy: %{type: :greedy_search})
serving = Bumblebee.Text.generation(loaded, tokenizer, generation, defn_options: [compiler: Compiler])
IO.puts("\n## Generation (greedy, 60 new tokens at most)")

for prompt <- reference["prompts"] do
  {micro, %{results: [%{text: text}]}} =
    :timer.tc(fn -> Nx.with_default_backend(backend, fn -> Nx.Serving.run(serving, prompt) end) end)

  milliseconds = div(micro, 1000)
  IO.puts("#{inspect(prompt)}\n→ #{inspect(text)} (#{milliseconds} ms)")
end
