# Chat template parity

This checks that `Bumblebee.apply_chat_template/3` renders the same prompts as `apply_chat_template` in transformers 5.23.

`repos.txt` lists 13 chat checkpoints of architectures Bumblebee supports, one of them with named templates.
`cases.json` holds 32 conversations, chosen to reach the branches of those templates: tool calls in every shape the templates check for, content as a string, a list of parts or null, reasoning under each of its names, template variables such as `enable_thinking`, `xml_tools` and `preserve_thinking`, and inputs that templates reject.

`coverage/report.md` shows which template branches the conversations reach, 523 of 590 or 89%, and lists the ones they do not.
It compiles each template to Python with Jinja and traces it with coverage.py.
Conditional expressions inside `{{ ... }}` are not counted, because they compile to a single line.

`results-bumblebee.md` compares Bumblebee with transformers when messages and tools keep the key order of the JSON, decoded with `Otera.JSON.decode!/1`.
`results-bumblebee-plain.md` does the same with plain Elixir maps, as a user writes them.
A render counts as matching when the text is identical, or when both raise the same template error.
`outputs/` holds every render of the last run.

The SmolLM3 template writes today's date into the prompt, so run both renders on the same day.

To run it again:

```sh
uv run reference.py
uv run coverage_report.py
BUMBLEBEE_PATH=/path/to/bumblebee elixir bumblebee.exs
INPUT=plain BUMBLEBEE_PATH=/path/to/bumblebee elixir bumblebee.exs
python3 compare.py bumblebee
python3 compare.py bumblebee-plain
```
