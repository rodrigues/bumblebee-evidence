# Times one forward pass of tiny-random-Qwen3_5ForCausalLM on EXLA over prompts of 64, 1024 and 4096 tokens, which take 1, 16 and 64 chunks of the linear attention.
# Prints the compile time, the fastest of 5 runs, and the sum of the logits, to compare two checkouts.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir timing.exs
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:exla, "1.0.0"},
  # EXLA 1.0.0 does not build with fine 0.2
  {:fine, "0.1.6", override: true}
])

Nx.global_default_backend(EXLA.Backend)
dir = Path.join(__DIR__, "checkpoints/tiny-random-Qwen3_5ForCausalLM")
{:ok, %{model: model, params: params}} = Bumblebee.load_model({:local, dir})
checkout = System.fetch_env!("BUMBLEBEE_PATH")
IO.puts(checkout)

for length <- [64, 1024, 4096] do
  inputs = %{"input_ids" => Nx.iota({1, length}) |> Nx.remainder(90) |> Nx.add(5)}

  {compile_us, predict} =
    :timer.tc(fn -> Nx.Defn.compile(&Axon.predict(model, &1, &2), [params, inputs], compiler: EXLA) end)

  run_us =
    Enum.min(for _ <- 1..5, do: elem(:timer.tc(fn -> Nx.backend_transfer(predict.(params, inputs).logits) end), 0))

  checksum = predict.(params, inputs).logits |> Nx.sum() |> Nx.to_number() |> Float.round(4)
  compile_ms = div(compile_us, 1000)
  run_ms = Float.round(run_us / 1000, 1)
  IO.puts("  #{length} tokens: compile #{compile_ms} ms, run #{run_ms} ms (best of 5), logits sum #{checksum}")
end
