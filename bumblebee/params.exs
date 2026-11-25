# Loads mistralai/Ministral-3-3B-Instruct-2512-BF16 with the parameter log, and writes the missing, unused and mismatched parameters to outputs/params.txt.
# Run with: BUMBLEBEE_PATH=path/to/bumblebee elixir params.exs
Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:exla, "1.0.0"},
  # EXLA 1.0.0 does not build with fine 0.2
  {:fine, "0.1.6", override: true}
])

Application.ensure_all_started(:ex_unit)
[path] = Path.wildcard(Path.expand("~/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/*"))

log =
  ExUnit.CaptureLog.capture_log([level: :debug], fn ->
    {:ok, _} = Bumblebee.load_model({:local, path}, log_params_diff: true, backend: EXLA.Backend, type: :bf16)
  end)

names = Regex.scan(~r/^\s+\* (\S+)/m, log, capture: :all_but_first) |> List.flatten()
{vision, other} = Enum.split_with(names, &String.starts_with?(&1, ["vision_tower.", "multi_modal_projector."]))

report = """
# Parameter log of mistralai/Ministral-3-3B-Instruct-2512-BF16

- Log sections: #{Regex.scan(~r/the following .*:/, log) |> List.flatten() |> Enum.uniq() |> inspect()}
- Vision tower and projector tensors listed: #{length(vision)}
- Other tensors listed: #{length(other)} #{inspect(other)}
"""

File.write!(Path.join(__DIR__, "outputs/params.txt"), report)
IO.puts(report)
