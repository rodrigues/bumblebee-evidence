# Times one forward pass of Qwen/Qwen3.5-2B in Bumblebee on EMLX, in bfloat16, over prompts of 128 to 8192 tokens, to compare the prompt time with mlx-lm.
# Writes outputs/prefill.json.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir prefill.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true},
  {:jason, "~> 1.4"}
])

dir = __DIR__
[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*"))
backend = {EMLX.Backend, device: :gpu}
{:ok, loaded} = Bumblebee.load_model({:local, path}, type: :bf16, backend: backend)
{_init, predict} = Axon.build(loaded.model)
forward = Nx.Defn.jit(predict, compiler: EMLX)

results =
  for n <- [128, 512, 2048, 8192], into: %{} do
    inputs = %{"input_ids" => Nx.iota({1, n}, backend: backend) |> Nx.remainder(1000) |> Nx.add(1000)}
    run = fn -> forward.(loaded.params, inputs).logits[[0, -1, 0]] |> Nx.to_number() end
    run.()
    {micro, _} = :timer.tc(run)
    ms = div(micro, 1000)
    IO.puts("bumblebee #{n} tokens: #{ms} ms")
    {n, ms}
  end

File.write!(Path.join(dir, "outputs/prefill.json"), Jason.encode!(results, pretty: true))
