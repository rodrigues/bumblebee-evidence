# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers==5.23.0", "jinja2==3.1.6", "coverage==7.10.7"]
# ///
"""Measures which branches of each chat template the conversations in cases.json reach.

Each template is compiled by Jinja into Python, written to a file and traced with coverage.py.
A branch is an {% if %}, {% elif %}, {% for %} or macro path, mapped back to its template line.
Conditional expressions inside {{ ... }}, such as `a if b else c`, compile to a single line and are not counted.
"""

import bisect
import json
import shutil
import sys
from pathlib import Path

import coverage
from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template

root = Path(__file__).parent
cases_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "cases.json"
cases = json.loads(cases_path.read_text())
compiled_dir = root / "coverage" / "compiled"
shutil.rmtree(compiled_dir, ignore_errors=True)
compiled_dir.mkdir(parents=True)

templates = []

for repo in (root / "repos.txt").read_text().split():
    tokenizer = AutoTokenizer.from_pretrained(repo)
    chat_template = tokenizer.chat_template
    named = chat_template if isinstance(chat_template, dict) else {"default": chat_template}
    special = {k: v for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)}

    for name, source in named.items():
        environment = _compile_jinja_template(source).environment
        python_source = environment.compile(source, raw=True)
        path = compiled_dir / f"{repo.replace('/', '--')}--{name}.py"
        path.write_text(python_source)
        code = compile(python_source, str(path), "exec")
        template = environment.template_class.from_code(environment, code, environment.make_globals(None))
        label = repo if len(named) == 1 else f"{repo} ({name})"
        templates.append({"label": label, "path": path, "template": template, "source": source, "special": special})

tracer = coverage.Coverage(branch=True, data_file=None, include=[str(compiled_dir / "*.py")])
tracer.start()

for entry in templates:
    for case in cases:
        options = case["options"]
        context = {
            "messages": case["messages"],
            "tools": case.get("tools"),
            "documents": case.get("documents"),
            "add_generation_prompt": options.get("add_generation_prompt", False),
            **entry["special"],
            **case.get("variables", {}),
        }

        try:
            entry["template"].render(**context)
        except Exception:
            pass

tracer.stop()
report_path = root / "coverage" / "coverage.json"
tracer.json_report(outfile=str(report_path))
report = json.loads(report_path.read_text())["files"]

lines = ["# Chat template branch coverage", "", f"Cases: {cases_path.name}, {len(cases)} conversations.", ""]
lines += ["| Template | Branches | Reached | Coverage |", "|---|---|---|---|"]
details = []
total = reached = 0

for entry in templates:
    summary = report[str(entry["path"])]["summary"]
    branches = summary["num_branches"]
    covered = summary["covered_branches"]
    total += branches
    reached += covered
    percent = 100 * covered / branches if branches else 100
    lines.append(f"| {entry['label']} | {branches} | {covered} | {percent:.0f}% |")

    mapping = sorted(entry["template"].debug_info, key=lambda pair: pair[1])
    code_lines = [code_line for _, code_line in mapping]
    source_lines = entry["source"].splitlines()

    def template_line(code_line):
        index = bisect.bisect_right(code_lines, code_line) - 1
        return mapping[index][0] if index >= 0 else None

    missing = {}
    for from_line, _to_line in report[str(entry["path"])]["missing_branches"]:
        line = template_line(from_line)
        if line:
            missing[line] = missing.get(line, 0) + 1

    if missing:
        details += ["", f"## {entry['label']}", ""]
        for line in sorted(missing):
            text = source_lines[line - 1].strip() if line <= len(source_lines) else ""
            details.append(f"- line {line}: `{text[:110]}`")

lines.append(f"| All | {total} | {reached} | {100 * reached / total:.0f}% |")
lines += ["", "Untaken branches, by the template line of the tag that branches:"] + details + [""]

(root / "coverage" / "report.md").write_text("\n".join(lines))
print(f"{reached}/{total} branches, {100 * reached / total:.0f}%")
