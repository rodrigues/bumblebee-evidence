# Decodes 30 tokens greedily with Qwen/Qwen3.5-2B in Bumblebee, from the prompts in outputs/reference-2b.json, and compares with transformers.
# Runs in float32 on EXLA for parity, and on EMLX for the time per token on Apple silicon.
# Writes outputs/bumblebee-2b.json and outputs/results-2b.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_2b.exs
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:exla, "1.0.0"},
  # EXLA 1.0.0 does not build with fine 0.2
  {:fine, "0.1.6", override: true},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

dir = __DIR__
reference = dir |> Path.join("outputs/reference-2b.json") |> File.read!() |> Jason.decode!()
[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*"))
repo = {:local, path}

{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)

generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 30,
    strategy: %{type: :greedy_search},
    pad_token_id: generation_config.eos_token_id,
    eos_token_id: nil
  )

run = fn backend, compiler, type ->
  {:ok, %{model: model, params: params, spec: spec}} =
    Bumblebee.load_model(repo, backend: backend, type: type)

  generate =
    model
    |> Bumblebee.Text.Generation.build_generate(spec, generation_config)
    |> Nx.Defn.jit(compiler: compiler)

  Map.new(reference, fn {name, %{"input_ids" => input_ids}} ->
    inputs = %{
      "input_ids" => Nx.tensor([input_ids]),
      "attention_mask" => Nx.broadcast(1, {1, length(input_ids)}),
      "seed" => Nx.tensor([0])
    }

    generate.(params, inputs).token_ids |> Nx.to_flat_list()
    {micro, tokens} = :timer.tc(fn -> generate.(params, inputs).token_ids |> Nx.to_flat_list() end)
    {name, %{"generated" => tokens, "ms_per_token" => Float.round(micro / 1000 / length(tokens), 1)}}
  end)
end

runs = [
  {"EXLA, CPU, f32", run.(EXLA.Backend, EXLA, :f32)},
  {"EMLX, GPU, f32", run.({EMLX.Backend, device: :gpu}, EMLX, :f32)},
  {"EMLX, GPU, bf16", run.({EMLX.Backend, device: :gpu}, EMLX, :bf16)}
]

tokenizer_ids =
  Map.new(reference, fn {name, %{"prompt" => prompt}} ->
    ids =
      Bumblebee.configure(tokenizer, add_special_tokens: false)
      |> Bumblebee.apply_tokenizer(prompt)
      |> Map.fetch!("input_ids")
      |> Nx.to_flat_list()

    {name, ids}
  end)

File.write!(
  Path.join(dir, "outputs/bumblebee-2b.json"),
  Jason.encode!(%{runs: Map.new(runs), tokenizer_input_ids: tokenizer_ids}, pretty: true)
)

rows =
  for {label, results} <- runs, {name, result} <- Enum.sort(results) do
    expected = reference[name]["generated"]

    verdict =
      if result["generated"] == expected do
        "identical"
      else
        index = Enum.zip(expected, result["generated"]) |> Enum.find_index(fn {a, b} -> a != b end)
        "differs from token #{index}, where the transformers margin is #{Float.round(Enum.at(reference[name]["teacher_forced_margin"], index), 3)}"
      end

    "| #{label} | #{name} | #{verdict} | #{result["ms_per_token"]} ms |"
  end

tokenizer_rows =
  for {name, ids} <- Enum.sort(tokenizer_ids) do
    "- #{name}: tokenizer ids #{if ids == reference[name]["input_ids"], do: "identical", else: "DIFFERENT"}"
  end

texts =
  for {name, %{"text" => text}} <- Enum.sort(reference), do: "- #{name}: #{inspect(text)}"

table =
  Enum.join(
    [
      "| Run | Prompt | 30 greedy tokens against transformers f32 | Time per token |",
      "|---|---|---|---|" | rows
    ],
    "\n"
  )

report =
  "# Qwen/Qwen3.5-2B in Bumblebee against transformers 5.23.0\n\n#{table}\n\n" <>
    Enum.join(tokenizer_rows, "\n") <> "\n\nTransformers output:\n\n" <> Enum.join(texts, "\n") <> "\n"

File.write!(Path.join(dir, "outputs/results-2b.md"), report)
IO.puts(report)
