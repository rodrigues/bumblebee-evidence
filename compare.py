"""Compares the transformers and Bumblebee outputs and writes results.md."""

import json
import re
import sys
from pathlib import Path

root = Path(__file__).parent
variant = sys.argv[1] if len(sys.argv) > 1 else "bumblebee"
cases = [case["name"] for case in json.loads((root / "cases.json").read_text())]
repos = (root / "repos.txt").read_text().split()


def error_message(text):
    match = re.match(r"ERROR [\w.]+: (.*?)(?: \(line \d+\))?$", text, re.S)
    return match and match.group(1)


counts = {"identical": 0, "same error": 0, "DIFFERENT": 0}
rows = []

for repo in repos:
    cells = []
    for case in cases:
        name = f"{repo.replace('/', '--')}.{case}.txt"
        expected = (root / "outputs" / "transformers" / name).read_text()
        actual = (root / "outputs" / variant / name).read_text()

        if expected == actual:
            result = "identical"
        elif error_message(expected) and error_message(expected) == error_message(actual):
            result = "same error"
        else:
            result = "DIFFERENT"

        counts[result] += 1
        cells.append(result)
    rows.append(f"| {repo} | " + " | ".join(cells) + " |")

summary = ", ".join(f"{count} {result}" for result, count in counts.items())

report = "\n".join(
    [
        f"# Chat template parity: {variant}",
        "",
        f"{summary}, of {sum(counts.values())} renders.",
        "",
        "| Checkpoint | " + " | ".join(cases) + " |",
        "|---|" + "---|" * len(cases),
        *rows,
        "",
    ]
)

(root / f"results-{variant}.md").write_text(report)
print(summary)
