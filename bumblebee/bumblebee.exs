# Runs the tiny Ministral 3 checkpoints through Bumblebee on EXLA and compares the outputs with outputs/reference.json.
# Writes outputs/results.md.
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

reference = Path.join(__DIR__, "outputs/reference.json") |> File.read!() |> Jason.decode!()
input_ids = Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
max_diff = fn expected, actual -> Nx.tensor(expected) |> Nx.subtract(actual) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() end
format = fn value -> :erlang.float_to_binary(value, [{:scientific, 1}]) end

repo = "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"
{:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model({:hf, repo})
expected = reference["tiny-random-Ministral3ForCausalLM"]
logits = Axon.predict(model, params, %{"input_ids" => input_ids, "attention_mask" => attention_mask}).logits
far = Nx.iota({1, 10}) |> Nx.add(16380)
far_logits = Axon.predict(model, params, %{"input_ids" => input_ids, "attention_mask" => attention_mask, "position_ids" => far}).logits

# As in reference.py, sharper queries make the query scaling change the logits

sharp_params =
  update_in(params.data, fn data ->
    Map.new(data, fn
      {"decoder.blocks." <> _ = layer, %{"kernel" => kernel} = layer_params} = entry ->
        if String.ends_with?(layer, "self_attention.query"),
          do: {layer, %{layer_params | "kernel" => Nx.multiply(kernel, 30.0)}},
          else: entry

      entry ->
        entry
    end)
  end)

sharp_far_logits = Axon.predict(model, sharp_params, %{"input_ids" => input_ids, "attention_mask" => attention_mask, "position_ids" => far}).logits
unscaled_model = Bumblebee.build_model(Bumblebee.configure(spec, query_scaling_beta: nil))
unscaled_far_logits = Axon.predict(unscaled_model, sharp_params, %{"input_ids" => input_ids, "attention_mask" => attention_mask, "position_ids" => far}).logits
IO.puts("effect of the query scaling in Bumblebee: #{Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(sharp_far_logits, unscaled_far_logits))))}")

generation_config = Bumblebee.configure(Bumblebee.Text.GenerationConfig, max_new_tokens: 8, strategy: %{type: :greedy_search}, eos_token_id: nil, pad_token_id: 0)
generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config)
%{token_ids: token_ids} = generate.(params, %{"input_ids" => input_ids[[.., 0..7]], "seed" => Nx.tensor([0])})

rows = [
  "| #{repo} | logits | max abs difference #{format.(max_diff.(expected["logits"], logits[[.., 1..3, 1..3]]))} |",
  "| #{repo} | logits at positions 16380 to 16389 | max abs difference #{format.(max_diff.(expected["logits at positions 16380 to 16389"], far_logits[[.., 3..5, 1..3]]))} |",
  "| #{repo} | logits at positions 16380 to 16389, sharper queries | max abs difference #{format.(max_diff.(expected["sharp logits at positions 16380 to 16389"], sharp_far_logits[[.., 3..5, 1..3]]))} |",
  "| #{repo} | 8 greedy tokens | #{if Nx.to_flat_list(token_ids) == expected["generated"], do: "identical", else: "different: #{inspect(Nx.to_flat_list(token_ids))}"} |"
]

repo = "hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration"
{:ok, %{model: model, params: params}} = Bumblebee.load_model({:hf, repo})
logits = Axon.predict(model, params, %{"input_ids" => input_ids, "attention_mask" => attention_mask}).logits
rows = rows ++ ["| #{repo} | logits | max abs difference #{format.(max_diff.(reference["tiny-random-Mistral3ForConditionalGeneration"]["logits"], logits[[.., 1..3, 1..3]]))} |"]

report = """
# Bumblebee against transformers 5.23.0

| Checkpoint | Output | Result |
|---|---|---|
#{Enum.join(rows, "\n")}
"""

File.write!(Path.join(__DIR__, "outputs/results.md"), report)
IO.puts(report)
