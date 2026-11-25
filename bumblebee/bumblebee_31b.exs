# Loads the MLX 4-bit conversion of Gemma 4 31B with Bumblebee on EMLX, reads its parameter log, decodes 30 tokens greedily
# from the prompts in outputs/reference-31b.json, and compares the tokens and the teacher-forced top choices with mlx-lm.
# It also generates 30 tokens from a prompt of about 1,500 tokens, longer than the sliding window of 1,024,
# once with the cache and once by running the full sequence for each token.
# Writes outputs/bumblebee-31b.json and outputs/results-31b.md.
# Run with: MLX_ENABLE_TF32=0 GEMMA4_31B_PATH=path/to/target BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_31b.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
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
dir = __DIR__
reference = dir |> Path.join("outputs/reference-31b.json") |> File.read!() |> Jason.decode!()
repo = {:local, System.fetch_env!("GEMMA4_31B_PATH")}
gb = fn bytes -> Float.round(bytes / 1.0e9, 2) end
prompts = ["capital", "story"]

before = EMLX.memory_info().active_memory

{{load_micro, {:ok, %{model: model, params: params, spec: spec}}}, log} =
  ExUnit.CaptureLog.with_log([level: :debug], fn ->
    :timer.tc(fn -> Bumblebee.load_model(repo, backend: backend, type: :bf16, log_params_diff: true) end)
  end)

loaded = EMLX.memory_info().active_memory - before

params_log =
  for section <- ["missing", "unused", "ignored, because of non-matching shape"] do
    names =
      case Regex.run(~r/the following [^\n]*#{Regex.escape(section)}[^\n]*:\n\n((?:  \* [^\n]*\n)+)/, log) do
        [_, list] -> for "  * " <> name <- String.split(list, "\n", trim: true), do: name
        nil -> []
      end

    prefixes = names |> Enum.frequencies_by(&(&1 |> String.split(".") |> Enum.take(2) |> Enum.join(".")))
    {section, %{"count" => length(names), "by_prefix" => prefixes}}
  end
  |> Map.new()

{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)
tokenizer = Bumblebee.configure(tokenizer, add_special_tokens: false)

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
encode = fn text -> tokenizer |> Bumblebee.apply_tokenizer(text) |> Map.fetch!("input_ids") |> Nx.to_flat_list() end

generate_tokens = fn input_ids ->
  inputs = %{
    "input_ids" => Nx.tensor([input_ids]),
    "attention_mask" => Nx.broadcast(1, {1, length(input_ids)}),
    "seed" => Nx.tensor([0])
  }

  generate.(params, inputs).token_ids |> Nx.to_flat_list()
end

results =
  Map.new(prompts, fn name ->
    %{"input_ids" => input_ids, "generated" => expected, "prompt" => prompt} = reference[name]

    generate_tokens.(input_ids)
    {micro, tokens} = :timer.tc(fn -> generate_tokens.(input_ids) end)

    logits = predict.(params, %{"input_ids" => Nx.tensor([input_ids ++ expected])}).logits[0]
    logits = logits[(length(input_ids) - 1)..-2//1] |> Nx.as_type(:f32)

    {name,
     %{
       "generated" => tokens,
       "text" => Bumblebee.Tokenizer.decode(tokenizer, tokens),
       "ms_per_token" => Float.round(micro / 1000 / length(tokens), 1),
       "teacher_forced_argmax" => Nx.argmax(logits, axis: -1) |> Nx.to_flat_list(),
       "tokenizer_input_ids" => encode.(prompt)
     }}
  end)

%{active_memory: active, peak_memory: peak} = EMLX.memory_info()

# A prompt longer than the sliding window: the chat prompt around a long passage
passage = String.duplicate("The lighthouse keeper climbed the spiral stairs at dusk, counted the steps, trimmed the wick and wrote the weather in his log. ", 60)
long_prompt = String.replace(reference["capital"]["prompt"], "What is the capital of France? Answer in one sentence.", passage <> "Summarize this passage in one sentence.")
long_ids = encode.(long_prompt)
cached = generate_tokens.(long_ids)

full =
  Enum.reduce(1..30, [], fn _, generated ->
    logits = predict.(params, %{"input_ids" => Nx.tensor([long_ids ++ generated])}).logits
    generated ++ [logits[[0, -1]] |> Nx.argmax() |> Nx.to_number()]
  end)

summary = %{
  "load_seconds" => Float.round(load_micro / 1.0e6, 1),
  "memory_after_loading_gb" => gb.(loaded),
  "memory_after_generating_gb" => gb.(active - before),
  "peak_memory_gb" => gb.(peak - before),
  "params_log" => params_log,
  "results" => results,
  "long_prompt_tokens" => length(long_ids),
  "long_prompt_cached" => cached,
  "long_prompt_full_sequence" => full
}

File.write!(Path.join(dir, "outputs/bumblebee-31b.json"), Jason.encode!(summary, pretty: true))

rows =
  for name <- prompts do
    result = results[name]
    expected = reference[name]
    index = Enum.zip(expected["generated"], result["generated"]) |> Enum.find_index(fn {a, b} -> a != b end)
    tokens = if index, do: "differs from token #{index}", else: "identical"

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

long_index = Enum.zip(cached, full) |> Enum.find_index(fn {a, b} -> a != b end)

report = """
# Gemma 4 31B, MLX 4-bit, in Bumblebee on EMLX against mlx-lm 0.32.2

| Prompt | 30 greedy tokens | Same top choice as mlx-lm, teacher forced | Time per token | mlx-lm time per token | Tokenizer ids |
|---|---|---|---|---|---|
#{Enum.join(rows, "\n")}

| | Bumblebee | mlx-lm |
|---|---|---|
| Load time | #{summary["load_seconds"]} s | |
| Memory after loading | #{summary["memory_after_loading_gb"]} GB | #{reference["memory_after_loading_gb"]} GB |
| Memory after generating | #{summary["memory_after_generating_gb"]} GB | |
| Peak memory while generating | #{summary["peak_memory_gb"]} GB | #{prompts |> Enum.map(&reference[&1]["peak_memory_gb"]) |> Enum.max()} GB |

Parameter log: #{params_log |> Enum.map_join("; ", fn {section, %{"count" => count, "by_prefix" => by_prefix}} -> "#{section} #{count} #{inspect(by_prefix)}" end)}

A prompt of #{length(long_ids)} tokens, 30 tokens with the cache against the full sequence each step: #{if long_index, do: "differ from token #{long_index}", else: "identical"}
"""

File.write!(Path.join(dir, "outputs/results-31b.md"), report)
IO.puts(report)
