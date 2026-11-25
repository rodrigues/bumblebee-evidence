# Loads the three tiny checkpoints with the parameter log on, and writes the missing, unused and mismatched parameters to outputs/params.txt.
# Run from the Bumblebee checkout with: MIX_ENV=test mix run path/to/params.exs
Application.ensure_all_started(:ex_unit)
dir = Path.join(__DIR__, "checkpoints")

lines =
  for repo <- [
        {:local, "#{dir}/tiny-random-Llama4ForCausalLM"},
        {:local, "#{dir}/tiny-random-Llama4ForCausalLM-interleaved"},
        {:hf, "optimum-intel-internal-testing/tiny-random-llama4"}
      ] do
    log =
      ExUnit.CaptureLog.capture_log([level: :debug], fn ->
        {:ok, _} = Bumblebee.load_model(repo, log_params_diff: true)
      end)

    names = Regex.scan(~r/^\s+\* (\S+)/m, log, capture: :all_but_first) |> List.flatten()
    sections = Regex.scan(~r/the following [^\n]*:/, log) |> List.flatten()
    by_prefix = Enum.frequencies_by(names, &(&1 |> String.split(".") |> hd()))
    "- #{repo |> elem(1) |> Path.basename()}: #{if sections == [], do: "nothing missing, unused or mismatched", else: "#{inspect(sections)} #{inspect(by_prefix)}"}"
  end

report = "# Parameter log of the tiny checkpoints\n\n" <> Enum.join(lines, "\n") <> "\n"
File.write!(Path.join(__DIR__, "outputs/params.txt"), report)
IO.puts(report)
