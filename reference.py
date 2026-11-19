# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.23.0", "jinja2==3.1.6"]
# ///
"""Renders every conversation in cases.json with every checkpoint in repos.txt, using transformers."""

import json
from pathlib import Path

from transformers import AutoTokenizer

root = Path(__file__).parent
cases = json.loads((root / "cases.json").read_text())
out = root / "outputs" / "transformers"
out.mkdir(parents=True, exist_ok=True)

for repo in (root / "repos.txt").read_text().split():
    tokenizer = AutoTokenizer.from_pretrained(repo)

    for case in cases:
        target = out / f"{repo.replace('/', '--')}.{case['name']}.txt"

        try:
            prompt = tokenizer.apply_chat_template(
                case["messages"],
                tools=case.get("tools"),
                tokenize=False,
                **case["options"],
                **case.get("variables", {}),
            )
            target.write_text(prompt)
        except Exception as error:
            target.write_text(f"ERROR {type(error).__name__}: {error}")
