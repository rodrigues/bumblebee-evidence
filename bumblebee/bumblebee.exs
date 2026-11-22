# Runs the tiny checkpoints in checkpoints/ through Bumblebee on EXLA and compares the outputs with outputs/reference.json and outputs/reference-long.json.
# Writes outputs/bumblebee.json and outputs/results.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
Mix.install(
  [
    {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
    {:exla, "1.0.0"},
    # EXLA 1.0.0 does not build with fine 0.2
    {:fine, "0.1.6", override: true},
    {:jason, "~> 1.4"}
  ],
  config: [nx: [default_backend: EXLA.Backend, default_defn_options: [compiler: EXLA]]]
)

dir = __DIR__
read = fn file -> dir |> Path.join("outputs/" <> file) |> File.read!() |> Jason.decode!() end
reference = read.("reference.json")
long_reference = read.("reference-long.json")

inputs = %{
  "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
  "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
}

load = fn name, opts -> Bumblebee.load_model({:local, Path.join([dir, "checkpoints", name])}, opts) end
slice = fn tensor -> tensor[[0, 1..3, 1..3]] |> Nx.to_list() end

run = fn name ->
  entry = reference[name]

  result =
    for key <- ["hidden_state", "logits"], Map.has_key?(entry, key), into: %{} do
      # The causal checkpoint records the hidden state of its base model
      opts = if key == "hidden_state", do: [architecture: :base], else: []
      {:ok, %{model: model, params: params}} = load.(name, opts)
      {key, slice.(Axon.predict(model, params, inputs)[String.to_atom(key)])}
    end

  if Map.has_key?(entry, "generated") do
    {:ok, %{model: model, params: params, spec: spec}} = load.(name, [])
    {:ok, generation_config} = Bumblebee.load_generation_config({:local, Path.join([dir, "checkpoints", name])})

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil,
        pad_token_id: 0
      )

    generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)

    %{token_ids: token_ids} =
      generate.(params, %{
        "input_ids" => inputs["input_ids"][[.., 0..7]],
        "attention_mask" => inputs["attention_mask"][[.., 0..7]],
        "seed" => Nx.tensor([0])
      })

    Map.put(result, "generated", Nx.to_flat_list(token_ids))
  else
    result
  end
end

results = Map.new(Map.keys(reference), &{&1, run.(&1)})

# As in reference_long.py, a slower decay makes the state carried between chunks matter
{:ok, %{model: model, params: params}} = load.("tiny-random-Qwen3_5ForCausalLM", [])

params =
  update_in(params.data, fn data ->
    Map.new(data, fn
      {"decoder.blocks." <> _ = layer, %{"a_log" => a_log} = layer_params} ->
        {layer, %{layer_params | "a_log" => Nx.fill(a_log, -4.0)}}

      other ->
        other
    end)
  end)

long_input_ids = Nx.iota({1, 150}) |> Nx.multiply(7) |> Nx.remainder(1000) |> Nx.add(1)
long_logits = Axon.predict(model, params, %{"input_ids" => long_input_ids}).logits
long_result = Nx.take(long_logits[[0, .., 1..3]], Nx.tensor([63, 64, 149])) |> Nx.to_list()

File.write!(
  Path.join(dir, "outputs/bumblebee.json"),
  Jason.encode!(Map.put(results, "tiny-random-Qwen3_5ForCausalLM 150 tokens", %{"logits" => long_result}), pretty: true)
)

max_difference = fn expected, actual ->
  Enum.zip(List.flatten(expected), List.flatten(actual))
  |> Enum.map(fn {a, b} -> abs(a - b) end)
  |> Enum.max()
end

format = fn value -> :erlang.float_to_binary(value, [{:scientific, 1}]) end

row = fn checkpoint, output, expected, actual ->
  if output == "generated" do
    result = if expected == actual, do: "identical", else: "different"
    "| #{checkpoint} | #{output} | #{result} | #{inspect(actual)} |"
  else
    difference = max_difference.(expected, actual)
    result = if difference <= 1.0e-4, do: "within 1e-4", else: "over 1e-4"
    "| #{checkpoint} | #{output} | #{result} | max abs difference #{format.(difference)} |"
  end
end

rows =
  for {name, entry} <- Enum.sort(reference), {output, expected} <- Enum.sort(entry) do
    row.(name, output, expected, results[name][output])
  end

long_row = row.("tiny-random-Qwen3_5ForCausalLM", "logits, 150 tokens", long_reference["logits"], long_result)

report =
  """
  # Bumblebee against transformers 5.23.0

  | Checkpoint | Output | Result | Detail |
  |---|---|---|---|
  #{Enum.join(rows ++ [long_row], "\n")}
  """

File.write!(Path.join(dir, "outputs/results.md"), report)
IO.puts(report)
