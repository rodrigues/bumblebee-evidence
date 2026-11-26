# Runs the tiny Ministral 3 in float16 on EMLX at positions 16381 to 16385, with sharpened queries as in the tests, and compares with the transformers float16 logits in outputs/reference.json.
# Prints the largest difference per position; run it once on the branch and once on a checkout without the query scale fix.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir emlx_f16.exs
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

reference =
  __DIR__
  |> Path.join("outputs/reference.json")
  |> File.read!()
  |> Jason.decode!()
  |> get_in(["tiny-random-Ministral3ForCausalLM", "float16 sharp logits at positions 16381 to 16385"])
  |> Nx.tensor()

repo = {:hf, "hf-tiny-v2/tiny-random-Ministral3ForCausalLM"}
backend = {EMLX.Backend, device: :gpu}

{:ok, %{model: model, params: params}} = Bumblebee.load_model(repo, type: :f16, backend: backend)

params =
  update_in(params.data, fn data ->
    Map.new(data, fn
      {"decoder.blocks." <> _ = name, %{"kernel" => kernel} = layer} = entry ->
        if String.ends_with?(name, "self_attention.query"),
          do: {name, %{layer | "kernel" => Nx.multiply(kernel, 30.0)}},
          else: entry

      entry ->
        entry
    end)
  end)

inputs =
  Nx.with_default_backend(backend, fn ->
    %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
      "position_ids" => Nx.iota({1, 10}) |> Nx.add(16380)
    }
  end)

outputs = Axon.predict(model, params, inputs, compiler: EMLX)
output_backend = inspect(outputs.logits.data.__struct__)
IO.puts("logits computed on #{output_backend}, type #{inspect(Nx.type(outputs.logits))}")
logits = outputs.logits[[.., 1..5, 1..3]] |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)

diffs =
  logits
  |> Nx.subtract(reference)
  |> Nx.abs()
  |> Nx.reduce_max(axes: [0, 2])
  |> Nx.to_flat_list()
  |> Enum.map(&Float.round(&1, 5))

checkout = System.fetch_env!("BUMBLEBEE_PATH")
IO.puts("#{checkout}: max |EMLX float16 - transformers float16| at positions 16381..16385: #{inspect(diffs)}")
