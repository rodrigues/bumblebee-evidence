Mix.install([
  {:bumblebee, path: Path.expand("~/code/oss/bumblebee")},
  {:emlx, path: Path.expand("~/code/oss/emlx/emlx"), override: true}
])

[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*"))
backend = {EMLX.Backend, device: :gpu}
{:ok, loaded} = Bumblebee.load_model({:local, path}, type: :bf16, backend: backend)
{_init, predict} = Axon.build(loaded.model)
forward = Nx.Defn.jit(predict, compiler: EMLX)

for n <- [128, 512, 2048, 8192] do
  inputs = %{"input_ids" => Nx.iota({1, n}, backend: backend) |> Nx.remainder(1000) |> Nx.add(1000)}
  run = fn -> forward.(loaded.params, inputs).logits[[0, -1, 0]] |> Nx.to_number() end
  run.()
  {micro, _} = :timer.tc(run)
  IO.puts("bumblebee #{n} tokens: #{div(micro, 1000)} ms")
end
