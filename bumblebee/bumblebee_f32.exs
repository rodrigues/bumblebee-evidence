# Same as bumblebee.exs, with float32 activations instead of bfloat16, against outputs/reference-f32.json.
# Writes outputs/results-f32.md.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_f32.exs
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
reference = dir |> Path.join("outputs/reference-f32.json") |> File.read!() |> Jason.decode!()

rows =
  for {repo, %{"input_ids" => input_ids, "generated" => expected} = ref} <- Enum.sort(reference) do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, repo}, backend: backend, type: :f32)

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

    tokens = generate.(params, inputs).token_ids |> Nx.to_flat_list()

    predict = Nx.Defn.jit(&Axon.predict(model, &1, &2), compiler: EMLX)
    logits = predict.(params, %{"input_ids" => Nx.tensor([input_ids ++ expected])}).logits[0]
    argmax = logits[(length(input_ids) - 1)..-2//1] |> Nx.argmax(axis: -1) |> Nx.to_flat_list()

    differing =
      for {{e, g}, margin} <- Enum.zip(Enum.zip(ref["teacher_forced_argmax"], argmax), ref["teacher_forced_margin"]),
          e != g,
          do: Float.round(margin, 3)

    "| #{repo} | #{if tokens == expected, do: "identical", else: "DIFFERENT"} | #{30 - length(differing)} of 30 #{if differing != [], do: inspect(differing)} |"
  end

report = """
# Bumblebee on EMLX against mlx-lm 0.32.2, float32 activations

| Checkpoint | 30 greedy tokens | Same top choice as mlx-lm, teacher forced |
|---|---|---|
#{Enum.join(rows, "\n")}
"""

File.write!(Path.join(dir, "outputs/results-f32.md"), report)
IO.puts(report)
