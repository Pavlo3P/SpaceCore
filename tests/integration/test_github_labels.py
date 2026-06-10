import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _template_labels() -> set[str]:
    labels: set[str] = set()
    for path in (ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"):
        for line in path.read_text().splitlines():
            if line.startswith("labels:"):
                labels.update(ast.literal_eval(line.split(":", 1)[1].strip()))
    return labels


def _setup_script_labels() -> set[str]:
    script = (ROOT / "scripts" / "setup_labels.sh").read_text()
    return set(re.findall(r'gh label create "([^"]+)"', script))


def test_issue_template_labels_are_created_by_setup_script():
    assert _template_labels() <= _setup_script_labels()
