# Loads Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B with the parameter log on, and summarizes it by tensor name prefix.
# Writes outputs/params.txt.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir params.exs
# MLX computes f32 matrix products in TF32 by default on M5-class GPUs, and reads MLX_ENABLE_TF32 from the shell, not from System.put_env.
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  # EMLX does not build with fine 0.2, which its "~> 0.1" requirement allows
  {:fine, "0.1.6", override: true}
])

Application.ensure_all_started(:ex_unit)

summarize = fn log ->
  for section <- ["missing", "unused", "ignored, because of non-matching shape"] do
    names =
      case Regex.run(~r/the following [^\n]*#{Regex.escape(section)}[^\n]*:\n\n((?:  \* [^\n]*\n)+)/, log) do
        [_, list] -> for "  * " <> name <- String.split(list, "\n", trim: true), do: name
        nil -> []
      end

    prefixes =
      names
      |> Enum.frequencies_by(&(&1 |> String.split(".") |> Enum.take(2) |> Enum.join(".")))
      |> Enum.map_join("\n", fn {prefix, count} -> "  - #{prefix}: #{count}" end)

    "#{section}: #{length(names)}\n#{prefixes}"
  end
  |> Enum.join("\n")
end

report =
  for repo <- ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"] do
    cache = "models--" <> String.replace(repo, "/", "--")
    [path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/#{cache}/snapshots/*"))

    log =
      ExUnit.CaptureLog.capture_log([level: :debug], fn ->
        {:ok, _} =
          Bumblebee.load_model({:local, path},
            backend: {EMLX.Backend, device: :gpu},
            type: :bf16,
            log_params_diff: true
          )
      end)

    "## #{repo}\n\n#{summarize.(log)}\n"
  end
  |> Enum.join("\n")

File.write!(Path.join(__DIR__, "outputs/params.txt"), report)
IO.puts(report)
