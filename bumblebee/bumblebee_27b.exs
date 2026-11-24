# Loads lmstudio-community/Qwen3.8-27B-MLX-4bit with Bumblebee on EMLX, decodes 30 tokens greedily from the prompts in outputs/reference-27b.json,
# and compares the tokens and the teacher-forced top choices with mlx-lm.
# Writes outputs/bumblebee-27b.json and outputs/results-27b.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_27b.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs; 0 keeps full f32.
System.put_env("MLX_ENABLE_TF32", System.get_env("MLX_ENABLE_TF32", "0"))

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

backend = {EMLX.Backend, device: :gpu}
Nx.default_backend(backend)
dir = __DIR__
reference = dir |> Path.join("outputs/reference-27b.json") |> File.read!() |> Jason.decode!()
repo_name = reference["repo"]
[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--#{String.replace(repo_name, "/", "--")}/snapshots/*"))
repo = {:local, path}
gb = fn bytes -> Float.round(bytes / 1.0e9, 2) end
prompts = reference |> Map.keys() |> Enum.filter(&String.starts_with?(&1, "thinking=")) |> Enum.sort()

before = EMLX.memory_info().active_memory
{load_micro, {:ok, %{model: model, params: params, spec: spec}}} =
  :timer.tc(fn -> Bumblebee.load_model(repo, backend: backend, type: :bf16) end)

loaded = EMLX.memory_info().active_memory - before
EMLX.reset_peak_memory()

{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)

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

predict = Nx.Defn.jit(&Axon.predict(model, &1, &2), compiler: EMLX)

results =
  Map.new(prompts, fn name ->
    %{"input_ids" => input_ids, "generated" => expected, "prompt" => prompt} = reference[name]

    inputs = %{
      "input_ids" => Nx.tensor([input_ids]),
      "attention_mask" => Nx.broadcast(1, {1, length(input_ids)}),
      "seed" => Nx.tensor([0])
    }

    generate.(params, inputs).token_ids |> Nx.to_flat_list()
    {micro, tokens} = :timer.tc(fn -> generate.(params, inputs).token_ids |> Nx.to_flat_list() end)

    logits = predict.(params, %{"input_ids" => Nx.tensor([input_ids ++ expected])}).logits[0]
    logits = logits[(length(input_ids) - 1)..-2//1] |> Nx.as_type(:f32)

    tokenizer_ids =
      Bumblebee.configure(tokenizer, add_special_tokens: false)
      |> Bumblebee.apply_tokenizer(prompt)
      |> Map.fetch!("input_ids")
      |> Nx.to_flat_list()

    {name,
     %{
       "generated" => tokens,
       "text" => Bumblebee.Tokenizer.decode(tokenizer, tokens),
       "ms_per_token" => Float.round(micro / 1000 / length(tokens), 1),
       "teacher_forced_argmax" => Nx.argmax(logits, axis: -1) |> Nx.to_flat_list(),
       "tokenizer_input_ids" => tokenizer_ids
     }}
  end)

# MLX evaluates lazily, so the dequantized embedding table only takes memory once generation first uses it
%{active_memory: active, peak_memory: peak} = EMLX.memory_info()

summary = %{
  "load_seconds" => Float.round(load_micro / 1.0e6, 1),
  "memory_after_loading_gb" => gb.(loaded),
  "memory_after_generating_gb" => gb.(active - before),
  "peak_memory_gb" => gb.(peak - before),
  "results" => results
}

File.write!(Path.join(dir, "outputs/bumblebee-27b.json"), Jason.encode!(summary, pretty: true))

rows =
  for name <- prompts do
    result = results[name]
    expected = reference[name]

    tokens =
      if result["generated"] == expected["generated"] do
        "identical"
      else
        index = Enum.zip(expected["generated"], result["generated"]) |> Enum.find_index(fn {a, b} -> a != b end)
        "differs from token #{index}"
      end

    differing =
      for {{e, g}, margin} <-
            Enum.zip(Enum.zip(expected["teacher_forced_argmax"], result["teacher_forced_argmax"]), expected["teacher_forced_margin"]),
          e != g,
          do: Float.round(margin, 3)

    teacher_forced =
      "#{30 - length(differing)} of 30" <>
        if(differing == [], do: "", else: " (mlx-lm margins where they differ: #{inspect(differing)})")

    tokenizer = if result["tokenizer_input_ids"] == expected["input_ids"], do: "identical", else: "DIFFERENT"

    "| #{name} | #{tokens} | #{teacher_forced} | #{result["ms_per_token"]} ms | #{expected["ms_per_token"]} ms | #{tokenizer} |"
  end

texts =
  for name <- prompts do
    "- #{name}\n  - mlx-lm: #{inspect(reference[name]["text"])}\n  - Bumblebee: #{inspect(results[name]["text"])}"
  end

report = """
# #{repo_name} in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Time per token | mlx-lm time per token | Tokenizer ids |
|---|---|---|---|---|---|
#{Enum.join(rows, "\n")}

| | Bumblebee | mlx-lm |
|---|---|---|
| Memory after loading | #{summary["memory_after_loading_gb"]} GB | #{reference["memory_after_loading_gb"]} GB |
| Memory after generating | #{summary["memory_after_generating_gb"]} GB | |
| Peak memory while generating | #{summary["peak_memory_gb"]} GB | #{reference |> Map.take(prompts) |> Map.values() |> Enum.map(& &1["peak_memory_gb"]) |> Enum.max()} GB |
| Load time | #{summary["load_seconds"]} s | |

#{Enum.join(texts, "\n")}
"""

File.write!(Path.join(dir, "outputs/results-27b.md"), report)
IO.puts(report)
