# Compares Bumblebee with transformers on unsloth/Llama-Guard-4-12B, a dense Llama 4 with Meta's trained weights, reading outputs/reference-guard.json.
# Runs on EMLX in bfloat16 on the GPU, and writes outputs/results-guard.md.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_guard.exs
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true},
  {:jason, "~> 1.4"}
])

Application.ensure_all_started(:ex_unit)
backend = {EMLX.Backend, device: :gpu}
Nx.global_default_backend(backend)
reference = __DIR__ |> Path.join("outputs/reference-guard.json") |> File.read!() |> Jason.decode!()
repo = {:hf, reference["repo"]}

{{:ok, %{model: model, params: params, spec: spec}}, log} =
  ExUnit.CaptureLog.with_log([level: :debug], fn ->
    Bumblebee.load_model(repo, backend: backend, type: :bf16, log_params_diff: true)
  end)

{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
sections = Regex.scan(~r/the following [^\n]*:/, log) |> List.flatten()
names = Regex.scan(~r/^\s+\* (\S+)/m, log, capture: :all_but_first) |> List.flatten()
by_prefix = Enum.frequencies_by(names, &(&1 |> String.split(".") |> Enum.take(2) |> Enum.join(".")))
predict = Nx.Defn.jit(&Axon.predict(model, &1, &2), compiler: EMLX)

rows =
  for {name, ref} <- Enum.sort(reference["prompts"]) do
    ids = ref["input_ids"]
    n = length(ref["generated"])

    config =
      Bumblebee.configure(Bumblebee.Text.GenerationConfig,
        max_new_tokens: n,
        strategy: %{type: :greedy_search},
        pad_token_id: 0,
        eos_token_id: nil
      )

    generate = model |> Bumblebee.Text.Generation.build_generate(spec, config) |> Nx.Defn.jit(compiler: EMLX)
    inputs = %{"input_ids" => Nx.tensor([ids]), "seed" => Nx.tensor([0])}
    tokens = generate.(params, inputs).token_ids |> Nx.to_flat_list()
    {micro, _} = :timer.tc(fn -> generate.(params, inputs).token_ids |> Nx.to_flat_list() end)

    logits = predict.(params, %{"input_ids" => Nx.tensor([ids ++ ref["generated"]])}).logits[0][(length(ids) - 1)..-2//1]
    forced = logits |> Nx.argmax(axis: -1) |> Nx.to_flat_list()
    differ = for {{a, b}, m} <- Enum.zip(Enum.zip(forced, ref["generated"]), ref["margins"]), a != b, do: Float.round(m, 3)

    slice_diff =
      logits[[0..(min(n, 3) - 1), 0..4]] |> Nx.as_type(:f32) |> Nx.subtract(Nx.tensor(ref["logits_slice"])) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    verdict =
      case Enum.find_index(Enum.zip(tokens, ref["generated"]), fn {a, b} -> a != b end) do
        nil -> "identical"
        i -> "differs from token #{i}"
      end

    text = tokenizer |> Bumblebee.Tokenizer.decode(tokens) |> inspect()

    "| #{name} | #{text} | #{verdict} | #{n - length(differ)} of #{n}#{if differ != [], do: " (transformers margins where they differ: #{inspect(differ)})"} | #{Float.round(slice_diff, 4)} | #{Float.round(micro / 1000 / n, 1)} ms |"
  end

report = """
# Llama-Guard-4-12B against transformers

Bumblebee on EMLX, GPU, bfloat16, against transformers 5.23.0 on the CPU in bfloat16.
The checkpoint's attention_chunk_size is #{inspect(reference["attention_chunk_size_in_checkpoint"])}; Bumblebee loads it as #{inspect(spec.attention_chunk_size)}.
Parameter log: #{if sections == [], do: "nothing missing, unused or mismatched", else: "#{inspect(sections)} #{inspect(by_prefix)}"}

| Prompt | Bumblebee output | Greedy tokens | Same top choice, teacher forced | Largest logit difference, first 3 positions by 5 | Time per token, incl. prefill |
|---|---|---|---|---|---|
#{Enum.join(rows, "\n")}
"""

File.write!(Path.join(__DIR__, "outputs/results-guard.md"), report)
IO.puts(report)
