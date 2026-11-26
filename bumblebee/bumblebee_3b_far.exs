# Teacher forces the same tokens as reference_3b_far.py through mistralai/Ministral-3-3B-Instruct-2512-BF16 in Bumblebee, in bfloat16 on EMLX, with positions starting at the same offset, and compares the top tokens with transformers.
# Writes outputs/results-3b-far.md.
# Run with: MLX_ENABLE_TF32=0 BUMBLEBEE_PATH=path/to/bumblebee EMLX_PATH=path/to/emlx/emlx elixir bumblebee_3b_far.exs
System.get_env("MLX_ENABLE_TF32") || raise "set MLX_ENABLE_TF32 in the shell, 0 for full float32 matrix products"

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:emlx, path: System.fetch_env!("EMLX_PATH"), override: true},
  {:jason, "~> 1.4"}
])

dir = __DIR__
read = fn name -> dir |> Path.join("outputs/#{name}") |> File.read!() |> Jason.decode!() end
reference = read.("reference-3b.json")
far = read.("reference-3b-far.json")
offset = far["offset"]

[path] =
  Path.wildcard(
    Path.expand("~/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/*")
  )

backend = {EMLX.Backend, device: :gpu}
{:ok, %{model: model, params: params}} = Bumblebee.load_model({:local, path}, type: :bf16, backend: backend)
predict = Nx.Defn.jit(&Axon.predict(model, &1, &2), compiler: EMLX)

rows =
  for {name, entry} <- Enum.sort(reference) do
    tokens = entry["input_ids"] ++ entry["generated"]
    length = length(tokens)
    start = length(entry["input_ids"]) - 1

    inputs =
      Nx.with_default_backend(backend, fn ->
        %{
          "input_ids" => Nx.tensor([tokens]),
          "position_ids" => Nx.iota({1, length}) |> Nx.add(offset)
        }
      end)

    top =
      predict.(params, inputs).logits[[0, start..(length - 2)]]
      |> Nx.argmax(axis: -1)
      |> Nx.to_flat_list()

    expected = far[name]["top"]
    margins = far[name]["margin"]
    differing = for {{a, b}, m} <- Enum.zip(Enum.zip(top, expected), margins), a != b, do: m
    same = 30 - length(differing)
    largest = if differing == [], do: "none differ", else: Float.round(Enum.max(differing), 3)
    "| #{name} | #{same} of 30 | #{largest} |"
  end

offset_text = Integer.to_string(offset)

table =
  [
    "Top token at each of the 30 generated positions, teacher forced from position #{offset_text}: Bumblebee bfloat16 on EMLX against transformers float32.",
    "",
    "| Prompt | Same top token | transformers' largest margin where they differ |",
    "|---|---|---|"
    | rows
  ]
  |> Enum.join("\n")

File.write!(Path.join(dir, "outputs/results-3b-far.md"), table <> "\n")
IO.puts(table)
