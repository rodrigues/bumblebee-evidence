# Loads the MLX quantized checkpoints with Bumblebee on EMLX, decodes greedily from the prompt in outputs/reference.json and compares with mlx-lm.
# Also loads the bf16 checkpoint of the same model, to compare memory and speed.
# Writes outputs/bumblebee.json and outputs/results.md.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true},
  {:jason, "~> 1.4"}
])

backend = {EMLX.Backend, device: :gpu}
Nx.global_default_backend(backend)
dir = __DIR__
reference = dir |> Path.join("outputs/reference.json") |> File.read!() |> Jason.decode!()
gb = fn bytes -> Float.round(bytes / 1.0e9, 2) end

run = fn repo, input_ids ->
  :erlang.garbage_collect()
  EMLX.clear_cache()
  before = EMLX.memory_info().active_memory
  {micro, {:ok, %{model: model, params: params, spec: spec}}} =
    :timer.tc(fn -> Bumblebee.load_model({:hf, repo}, backend: backend, type: :bf16) end)

  loaded = EMLX.memory_info().active_memory - before
  EMLX.reset_peak_memory()
  {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
  {:ok, generation_config} = Bumblebee.load_generation_config({:hf, repo})

  generation_config =
    Bumblebee.configure(generation_config,
      max_new_tokens: 30,
      strategy: %{type: :greedy_search},
      eos_token_id: nil
    )

  generate =
    model
    |> Bumblebee.Text.Generation.build_generate(spec, generation_config)
    |> Nx.Defn.jit(compiler: EMLX)

  inputs = %{
    "input_ids" => Nx.tensor([input_ids]),
    "attention_mask" => Nx.broadcast(1, {1, length(input_ids)}),
    "seed" => Nx.tensor([0])
  }

  generate.(params, inputs).token_ids |> Nx.to_flat_list()
  {gen_micro, tokens} = :timer.tc(fn -> generate.(params, inputs).token_ids |> Nx.to_flat_list() end)

  # Teacher forcing on the mlx-lm sequence, to compare top choices position by position
  teacher_forced =
    case reference[repo] do
      %{"generated" => generated} ->
        sequence = Nx.tensor([input_ids ++ generated])
        predict = Nx.Defn.jit(&Axon.predict(model, &1, &2), compiler: EMLX)
        logits = predict.(params, %{"input_ids" => sequence}).logits[0]
        logits = logits[(length(input_ids) - 1)..-2//1] |> Nx.as_type(:f32)
        Nx.argmax(logits, axis: -1) |> Nx.to_flat_list()

      nil ->
        nil
    end

  # MLX evaluates lazily, so the dequantized embedding table only takes memory once generation first uses it
  %{active_memory: active, peak_memory: peak} = EMLX.memory_info()

  %{
    "load_seconds" => Float.round(micro / 1.0e6, 1),
    "memory_after_loading_gb" => gb.(loaded),
    "memory_after_generating_gb" => gb.(active - before),
    "peak_memory_gb" => gb.(peak - before),
    "generated" => tokens,
    "text" => Bumblebee.Tokenizer.decode(tokenizer, tokens),
    "ms_per_token" => Float.round(gen_micro / 1000 / length(tokens), 1),
    "teacher_forced_argmax" => teacher_forced,
    "tokenizer_input_ids" =>
      Bumblebee.configure(tokenizer, add_special_tokens: false)
      |> Bumblebee.apply_tokenizer(reference |> Map.values() |> hd() |> Map.fetch!("prompt"))
      |> Map.fetch!("input_ids")
      |> Nx.to_flat_list()
  }
end

input_ids = reference |> Map.values() |> hd() |> Map.fetch!("input_ids")
repos = Enum.sort(Map.keys(reference))
# Each checkpoint runs in its own process; EMLX frees its memory a few milliseconds after the process exits, so the next one waits before measuring
results =
  Map.new(repos, fn repo ->
    Process.sleep(500)
    {repo, Task.async(fn -> run.(repo, input_ids) end) |> Task.await(:infinity)}
  end)
File.write!(Path.join(dir, "outputs/bumblebee.json"), Jason.encode!(results, pretty: true))

rows =
  for repo <- repos do
    result = results[repo]

    generated = result["generated"]

    verdict =
      case reference[repo] do
        nil -> "(bf16, no mlx-lm reference)"
        %{"generated" => ^generated} -> "identical to mlx-lm"
        %{"generated" => expected} ->
          index = Enum.zip(expected, generated) |> Enum.find_index(fn {a, b} -> a != b end)
          "differs from token #{index}"
      end

    tokenizer = if result["tokenizer_input_ids"] == input_ids, do: "identical", else: "DIFFERENT"

    teacher_forced =
      case reference[repo] do
        %{"teacher_forced_argmax" => expected, "teacher_forced_margin" => margins} ->
          differing =
            for {{e, g}, margin} <- Enum.zip(Enum.zip(expected, result["teacher_forced_argmax"]), margins),
                e != g,
                do: margin

          "#{length(expected) - length(differing)} of #{length(expected)}" <>
            if(differing == [], do: "", else: " (mlx-lm margins where they differ: #{inspect(differing)})")

        nil ->
          "-"
      end

    "| #{repo} | #{verdict} | #{teacher_forced} | #{result["memory_after_loading_gb"]} GB | #{result["memory_after_generating_gb"]} GB | #{result["peak_memory_gb"]} GB | #{reference[repo]["peak_memory_gb"]} GB | #{result["ms_per_token"]} ms | #{reference[repo]["ms_per_token"]} ms | #{tokenizer} |"
  end

table =
  Enum.join(
    ["| Checkpoint | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Memory after loading | Memory after generating | Peak memory | mlx-lm peak memory | Time per token | mlx-lm time per token | Tokenizer ids |", "|---|---|---|---|---|---|---|---|---|---|" | rows],
    "\n"
  )

texts = Enum.map_join(repos, "\n", fn repo -> "- #{repo}: #{inspect(results[repo]["text"])}" end)
File.write!(Path.join(dir, "outputs/results.md"), "# Bumblebee on EMLX against mlx-lm 0.32.2\n\n#{table}\n\n#{texts}\n")
IO.puts(table <> "\n\n" <> texts)
