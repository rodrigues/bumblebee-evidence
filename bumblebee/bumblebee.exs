# Runs the tiny checkpoints in checkpoints/ through Bumblebee and compares the outputs with outputs/reference.json.
# Writes outputs/bumblebee.json and outputs/results.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:jason, "~> 1.4"}
])

dir = __DIR__
reference = dir |> Path.join("outputs/reference.json") |> File.read!() |> Jason.decode!()

inputs = %{
  "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
  "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
}

run = fn name ->
  repo = {:local, Path.join([dir, "checkpoints", name])}
  {:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)
  outputs = Axon.predict(model, params, inputs)
  slice = fn tensor -> tensor[[0, 1..3, 1..3]] |> Nx.to_list() end

  result =
    %{"architecture" => spec.architecture}
    |> then(&if(outputs[:hidden_state], do: Map.put(&1, "hidden_state", slice.(outputs.hidden_state)), else: &1))
    |> then(&if(outputs[:logits], do: Map.put(&1, "logits", slice.(outputs.logits)), else: &1))

  if Map.has_key?(reference[name], "generated") do
    {:ok, generation_config} = Bumblebee.load_generation_config(repo)

    generation_config =
      Bumblebee.configure(generation_config,
        max_new_tokens: 8,
        strategy: %{type: :greedy_search},
        eos_token_id: nil
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
File.write!(Path.join(dir, "outputs/bumblebee.json"), Jason.encode!(results, pretty: true))

max_difference = fn expected, actual ->
  Enum.zip(List.flatten(expected), List.flatten(actual))
  |> Enum.map(fn {a, b} -> abs(a - b) end)
  |> Enum.max()
end

rows =
  for {name, expected} <- Enum.sort(reference), {output, value} <- Enum.sort(expected) do
    actual = results[name][output]

    verdict =
      cond do
        output == "generated" and actual == value -> "identical"
        output == "generated" -> "DIFFERENT"
        max_difference.(value, actual) <= 1.0e-4 -> "within 1e-4"
        true -> "DIFFERENT"
      end

    detail =
      if output == "generated",
        do: inspect(actual),
        else: "max abs difference #{Float.round(max_difference.(value, actual), 7)}"

    "| #{name} | #{output} | #{verdict} | #{detail} |"
  end

table =
  Enum.join(
    ["| Checkpoint | Output | Result | Detail |", "|---|---|---|---|" | rows],
    "\n"
  )

File.write!(Path.join(dir, "outputs/results.md"), "# Bumblebee against transformers 5.23.0\n\n#{table}\n")
IO.puts(table)
