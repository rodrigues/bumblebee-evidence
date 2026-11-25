# Loads the tiny checkpoint with log_params_diff and writes what Bumblebee reports to outputs/params.txt.
# Run from a Bumblebee checkout: (cd path/to/bumblebee && MIX_ENV=test mix run "$OLDPWD/params.exs")
Application.ensure_all_started(:ex_unit)
dir = __DIR__

checkpoints = [
  {"tiny-random-KimiLinearForCausalLM", {:local, Path.join(dir, "checkpoints/tiny-random-KimiLinearForCausalLM")}}
]

report =
  for {name, repo} <- checkpoints do
    log =
      ExUnit.CaptureLog.capture_log([level: :debug], fn ->
        {:ok, _} = Bumblebee.load_model(repo, log_params_diff: true)
      end)

    lines = log |> String.split("\n") |> Enum.filter(&(&1 =~ ~r/following|^\s+\* /))
    "## #{name}\n\n" <> if(lines == [], do: "Every parameter loads, and no tensor is unused.\n", else: Enum.join(lines, "\n") <> "\n")
  end

File.write!(Path.join(dir, "outputs/params.txt"), Enum.join(report, "\n"))
IO.puts(Enum.join(report, "\n"))
