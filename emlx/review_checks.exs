# Checks two findings of the review of the EMLX checkpoint-quantized branch, on the GPU:
# - the result type of a product with a quantized weight, eager and compiled, for five pairs of activation and scale types, against the type Nx expects from the two input types;
# - contracting a transposed square quantized weight on its last axis, against the dense product, and on its first axis.
# Run with: MLX_ENABLE_TF32=0 EMLX_PATH=path/to/emlx/emlx elixir review_checks.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") ||
  raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true}
])

gpu = &Nx.backend_transfer(&1, {EMLX.Backend, device: :gpu})
max_diff = fn a, b -> a |> Nx.subtract(b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() end

attempt = fn fun ->
  try do
    fun.()
  rescue
    error ->
      message = error |> Exception.message() |> String.split("\n") |> hd()
      "raises #{inspect(error.__struct__)}: #{message}"
  end
end

IO.puts("Result type of a product with a quantized weight")

for {activation_type, scales_type} <- [
      {{:f, 32}, {:f, 32}},
      {{:f, 16}, {:f, 32}},
      {{:f, 16}, {:bf, 16}},
      {{:bf, 16}, {:bf, 16}},
      {{:bf, 16}, {:f, 32}}
    ] do
  weight = Nx.iota({64, 128}, type: scales_type) |> Nx.sin() |> gpu.()
  quantized = EMLX.quantize(weight, [])
  input = Nx.iota({2, 128}, type: activation_type) |> Nx.divide(100) |> gpu.()
  expected = Nx.Type.merge(activation_type, Nx.type(quantized))

  eager = attempt.(fn -> input |> Nx.dot([1], quantized, [1]) |> Nx.type() |> inspect() end)

  compiled =
    attempt.(fn ->
      Nx.Defn.jit(&Nx.dot(&1, [1], &2, [1]), compiler: EMLX).(input, quantized)
      |> Nx.type()
      |> inspect()
    end)

  activation = inspect(activation_type)
  scales = inspect(scales_type)
  expected_text = inspect(expected)

  IO.puts(
    "  activation #{activation}, scales #{scales}: Nx expects #{expected_text}, eager #{eager}, compiled #{compiled}"
  )
end

IO.puts("Transposed square quantized weight, 64x64 sin")
weight = Nx.iota({64, 64}, type: :f32) |> Nx.sin() |> gpu.()
input = Nx.iota({2, 64}, type: :f32) |> Nx.divide(100) |> gpu.()
quantized = EMLX.quantize(weight, [])
dense = quantized |> EMLX.Quantization.dequantize() |> Nx.transpose()

transposed =
  try do
    Nx.transpose(quantized)
  rescue
    error ->
      message = error |> Exception.message() |> String.split("\n") |> hd()
      IO.puts("  transposing raises #{inspect(error.__struct__)}: #{message}")
      nil
  end

for {label, axis} <- [{"first", 0}, {"last", 1}], transposed != nil do
  want = Nx.dot(input, [1], dense, [axis])

  eager =
    attempt.(fn ->
      difference = max_diff.(want, Nx.dot(input, [1], transposed, [axis]))
      "max abs difference from the dense product #{difference}"
    end)

  compiled =
    attempt.(fn ->
      got = Nx.Defn.jit(&Nx.dot(&1, [1], &2, [axis]), compiler: EMLX).(input, transposed)
      difference = max_diff.(want, got)
      "max abs difference from the dense product #{difference}"
    end)

  IO.puts("  #{label} axis: eager #{eager}; compiled #{compiled}")
end
