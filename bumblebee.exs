# Renders every conversation in cases.json with every checkpoint in repos.txt, using Bumblebee.
#
# BUMBLEBEE_PATH points at a checkout of the chat-templates branch.
# With INPUT=plain, messages and tools are plain Elixir maps, as a user writes them,
# instead of keeping the key order of the JSON.

Mix.install([
  {:bumblebee, path: System.fetch_env!("BUMBLEBEE_PATH")},
  {:otera, "0.2.0"}
])

defmodule Plain do
  def convert(%Otera.Dict{pairs: pairs}), do: Map.new(pairs, fn {key, value} -> {key, convert(value)} end)
  def convert(list) when is_list(list), do: Enum.map(list, &convert/1)
  def convert(value), do: value
end

plain? = System.get_env("INPUT") == "plain"
root = __DIR__
cases = root |> Path.join("cases.json") |> File.read!() |> Otera.JSON.decode!()
out = Path.join([root, "outputs", if(plain?, do: "bumblebee-plain", else: "bumblebee")])
File.mkdir_p!(out)

for repo <- root |> Path.join("repos.txt") |> File.read!() |> String.split() do
  # Gemma 4 is not on main yet, so its tokenizer type cannot be inferred
  opts = if repo =~ "gemma-4", do: [type: :gemma], else: []
  {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo}, opts)

  for case <- cases do
    name = String.replace(repo, "/", "--")
    case_name = case["name"]
    target = Path.join(out, "#{name}.#{case_name}.txt")

    options = Enum.map(case["options"].pairs, fn {key, value} -> {String.to_atom(key), value} end)
    variables = Map.new((case["variables"] || Otera.Dict.new([])).pairs)
    variables = if plain?, do: Plain.convert(variables), else: variables
    messages = if plain?, do: Plain.convert(case["messages"]), else: case["messages"]
    tools = if plain?, do: Plain.convert(case["tools"]), else: case["tools"]
    tools = if tools, do: [tools: tools], else: []

    result =
      try do
        Bumblebee.apply_chat_template(tokenizer, messages, options ++ tools ++ [variables: variables])
      rescue
        error ->
          module = inspect(error.__struct__)
          message = Exception.message(error)
          "ERROR #{module}: #{message}"
      end

    File.write!(target, result)
  end
end
