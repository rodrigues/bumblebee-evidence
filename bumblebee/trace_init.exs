# Times Axon.trace_init, which Bumblebee.load_model runs before reading weights, for Gemma 4 with growing numbers of blocks.
# Before the fix it took 15 s at 9 blocks (outputs/trace_init-before-fix-profile.txt); outputs/trace_init.txt has the times after it.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir trace_init.exs > outputs/trace_init.txt
Mix.install([{:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")}])

{:ok, base} = Bumblebee.load_spec({:local, Path.join(__DIR__, "checkpoints/tiny-random-Gemma4ForCausalLM")})

for blocks <- [6, 9, 12, 24, 42] do
  layer_types = for index <- 0..(blocks - 1), do: if(rem(index + 1, 3) == 0, do: :full_attention, else: :sliding_attention)
  spec = Bumblebee.configure(base, num_blocks: blocks, num_shared_key_value_blocks: div(blocks, 3), layer_types: layer_types)
  model = Bumblebee.build_model(spec)
  {micro, _} = :timer.tc(fn -> Axon.trace_init(model, Bumblebee.Text.Gemma4.input_template(spec)) end)
  milliseconds = div(micro, 1000)
  IO.puts("#{blocks} blocks: trace_init #{milliseconds} ms")
end
