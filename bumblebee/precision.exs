# Measures how far EMLX's float32 matrix products are from the exact result, for each quantized kernel of the tiny 4-bit checkpoint,
# with the kernel quantized and with the same kernel dequantized. The exact result is computed on Nx.BinaryBackend.
# Writes outputs/precision.md.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir precision.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true}
])

backend = {EMLX.Backend, device: :gpu}
path = Path.join(__DIR__, "checkpoints/tiny-random-Qwen3ForCausalLM-mlx-4bit")
{:ok, %{params: params}} = Bumblebee.load_model({:local, path}, backend: backend)

binary = &Nx.backend_copy(&1, Nx.BinaryBackend)
max_diff = fn a, b -> Nx.subtract(binary.(a), binary.(b)) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number() end

rows =
  for {layer, %{"kernel" => kernel}} <- Enum.sort(params.data), EMLX.Quantization.quantized?(kernel), rows <- [1, 8] do
    axis = if layer == "language_modeling_head.output", do: 1, else: 0
    size = elem(Nx.shape(kernel), axis)
    input = Nx.iota({1, rows, size}, backend: Nx.BinaryBackend) |> Nx.divide(size) |> Nx.sin()
    dense = EMLX.Quantization.dequantize(kernel)
    exact = Nx.dot(input, [2], binary.(dense), [axis])
    on_emlx = Nx.backend_copy(input, backend)
    dot = Nx.Defn.jit(fn x, w -> Nx.dot(x, [2], w, [axis]) end, compiler: EMLX)

    "| #{layer} | #{rows} | #{Float.round(max_diff.(dot.(on_emlx, kernel), exact), 9)} | #{Float.round(max_diff.(dot.(on_emlx, dense), exact), 9)} |"
  end

report = """
# EMLX float32 matrix products against the exact result

| Kernel | Input rows | Quantized kernel | Same kernel dequantized |
|---|---|---|---|
#{Enum.join(rows, "\n")}
"""

File.write!(Path.join(__DIR__, "outputs/precision.md"), report)
IO.puts(report)
