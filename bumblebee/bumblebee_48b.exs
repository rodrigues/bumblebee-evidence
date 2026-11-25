# Decodes 100 tokens greedily with mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit in Bumblebee on EMLX, from the prompts in outputs/reference-48b.json,
# and compares the tokens, the teacher-forced top choices and the speed with mlx-lm.
# Bumblebee does not load the tiktoken tokenizer of this repository, so the token ids come from mlx-lm's tokenizer, recorded in the reference.
# Writes outputs/bumblebee-48b.json, outputs/results-48b.md and the parameter loading log in outputs/params-48b.txt.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_48b.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

dir = __DIR__
reference = dir |> Path.join("outputs/reference-48b.json") |> File.read!() |> Jason.decode!()
backend = {EMLX.Backend, device: :gpu}
Nx.global_default_backend(backend)
[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--mlx-community--Kimi-Linear-48B-A3B-Instruct-4bit/snapshots/*"))
repo = {:local, path}

Application.ensure_all_started(:ex_unit)

load_log =
  ExUnit.CaptureLog.capture_log([level: :debug], fn ->
    send(self(), Bumblebee.load_model(repo, backend: backend, type: :bf16, log_params_diff: true))
  end)

{:ok, %{model: model, params: params, spec: spec}} = receive(do: (loaded -> loaded))

File.write!(
  Path.join(dir, "outputs/params-48b.txt"),
  load_log |> String.split("\n") |> Enum.filter(&(&1 =~ ~r/following|^\s+\* /)) |> Enum.join("\n")
)
{:ok, generation_config} = Bumblebee.load_generation_config(repo)

build = fn max_new_tokens ->
  config =
    Bumblebee.configure(generation_config,
      max_new_tokens: max_new_tokens,
      strategy: %{type: :greedy_search},
      pad_token_id: 0,
      eos_token_id: nil
    )

  model |> Bumblebee.Text.Generation.build_generate(spec, config) |> Nx.Defn.jit(compiler: EMLX)
end

[generate_one, generate] = [build.(1), build.(100)]

results =
  Map.new(reference, fn {name, %{"input_ids" => input_ids, "generated" => expected}} ->
    inputs = %{
      "input_ids" => Nx.tensor([input_ids]),
      "attention_mask" => Nx.broadcast(1, {1, length(input_ids)}),
      "seed" => Nx.tensor([0])
    }

    # The fastest of 3 runs, after a warm-up run
    measure = fn generate ->
      generate.(params, inputs).token_ids |> Nx.to_flat_list()
      runs = for _ <- 1..3, do: :timer.tc(fn -> generate.(params, inputs).token_ids |> Nx.to_flat_list() end)
      Enum.min_by(runs, &elem(&1, 0))
    end

    {prefill, _} = measure.(generate_one)
    {total, tokens} = measure.(generate)

    # Teacher forcing on the mlx-lm sequence, to compare the top choice at each position
    sequence = Nx.tensor([input_ids ++ Enum.drop(expected, -1)])
    logits = Axon.predict(model, params, %{"input_ids" => sequence}, compiler: EMLX).logits
    logits = Nx.slice_along_axis(logits, length(input_ids) - 1, length(expected), axis: 1)

    {name,
     %{
       "generated" => tokens,
       "teacher_forced_argmax" => Nx.argmax(logits, axis: -1) |> Nx.to_flat_list(),
       "prefill_ms" => Float.round(prefill / 1000, 1),
       "decode_ms_per_token" => Float.round((total - prefill) / 1000 / (length(tokens) - 1), 1)
     }}
  end)

File.write!(
  Path.join(dir, "outputs/bumblebee-48b.json"),
  Jason.encode!(%{results: results}, pretty: true)
)

rows =
  for {name, result} <- Enum.sort(results) do
    expected = reference[name]["generated"]
    index = Enum.zip(expected, result["generated"]) |> Enum.find_index(fn {a, b} -> a != b end)
    verdict = if index, do: "same first #{index}", else: "identical"

    differing =
      for {{e, g}, margin} <-
            Enum.zip(Enum.zip(reference[name]["teacher_forced_argmax"], result["teacher_forced_argmax"]), reference[name]["teacher_forced_margin"]),
          e != g,
          do: Float.round(margin, 3)

    teacher_forced =
      "#{length(expected) - length(differing)} of #{length(expected)}" <>
        if(differing == [], do: "", else: " (mlx-lm margins where they differ: #{inspect(differing)})")

    "| #{name} | #{verdict} | #{teacher_forced} | #{result["prefill_ms"]} ms | #{reference[name]["prefill_ms"]} ms | #{result["decode_ms_per_token"]} ms | #{reference[name]["decode_ms_per_token"]} ms |"
  end

report =
  """
  # mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit in Bumblebee on EMLX against mlx-lm 0.32.2

  | Prompt | 100 greedy tokens against mlx-lm | Same top choice as mlx-lm, teacher forced | Bumblebee prefill | mlx-lm prefill | Bumblebee decode per token | mlx-lm decode per token |
  |---|---|---|---|---|---|---|
  #{Enum.join(rows, "\n")}
  """

File.write!(Path.join(dir, "outputs/results-48b.md"), report)
IO.puts(report)
