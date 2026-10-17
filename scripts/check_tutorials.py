"""Execute tutorial Python blocks and assertions against the installed SpaceCore."""

from pathlib import Path
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK = re.compile(
    r"^\.\. (?:code|code-block):: python\n\n((?:[ \t]+[^\n]*\n|\n)+)",
    re.MULTILINE,
)


def main():
    """Run each page in an independent namespace; propagate any failure."""
    plt.show = lambda: plt.close("all")
    names = sys.argv[1:]  # optional page names, e.g. 03_functionals.rst; default: every page
    pages = [ROOT / "docs/source/tutorials" / name for name in names] or sorted((ROOT / "docs/source/tutorials").glob("0*.rst"))
    for page in pages:
        namespace = {"__name__": "__main__"}
        source = page.read_text()
        blocks = list(PYTHON_BLOCK.finditer(source))
        assert blocks, f"No Python examples found in {page}"
        for block in blocks:
            line = source.count("\n", 0, block.start(1)) + 1
            code = "\n" * (line - 1) + textwrap.dedent(block[1])
            exec(compile(code, str(page), "exec"), namespace)
        plt.close("all")
        print(f"PASS {page.name}: {len(blocks)} blocks", flush=True)


if __name__ == "__main__":
    main()
