# Compares Bumblebee on EXLA with the transformers outputs of optimum-internal-testing/tiny-random-qwen3_moe in outputs/reference.json,
# with 10 tokens, which computes every expert, and with 2, which gathers the chosen ones. Writes outputs/results.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir bumblebee.exs
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:exla, "1.0.0"},
  # EXLA 1.0.0 does not build with fine 0.2
  {:fine, "0.1.6", override: true},
  {:jason, "~> 1.4"}
])

Nx.global_default_backend(EXLA.Backend)
reference = __DIR__ |> Path.join("outputs/reference.json") |> File.read!() |> Jason.decode!()
repo = {:hf, "optimum-internal-testing/tiny-random-qwen3_moe"}
{:ok, %{model: model, params: params, spec: spec}} = Bumblebee.load_model(repo)
predict = &Axon.predict(model, params, &1, compiler: EXLA)
diff = fn a, b -> a |> Nx.subtract(Nx.tensor(b)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() end

input_ids = Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]])
attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
logits = predict.(%{"input_ids" => input_ids, "attention_mask" => attention_mask}).logits
short = predict.(%{"input_ids" => input_ids[[.., 0..1]]}).logits

generation_config =
  Bumblebee.configure(Bumblebee.Text.GenerationConfig,
    max_new_tokens: 8,
    strategy: %{type: :greedy_search},
    eos_token_id: nil,
    pad_token_id: 0
  )

generate = Bumblebee.Text.Generation.build_generate(model, spec, generation_config) |> Nx.Defn.jit(compiler: EXLA)
%{token_ids: token_ids} = generate.(params, %{"input_ids" => input_ids[[.., 0..7]], "seed" => Nx.tensor([0])})
generated = Nx.to_flat_list(token_ids)

report = """
# optimum-internal-testing/tiny-random-qwen3_moe in Bumblebee on EXLA against transformers 5.23.0

| Output | Max abs difference |
|---|---|
| logits of 10 tokens, every expert computed | #{diff.(logits[[.., 1..3, 1..3]], reference["logits"])} |
| logits of 2 tokens, the chosen experts gathered | #{diff.(short[[.., .., 1..3]], reference["logits of 2 tokens"])} |
| 8 greedy tokens | #{if generated == reference["generated"], do: "identical", else: "DIFFERENT: #{inspect(generated)}"} |
"""

File.write!(Path.join(__DIR__, "outputs/results.md"), report)
IO.puts(report)
