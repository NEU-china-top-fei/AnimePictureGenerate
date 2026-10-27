"""Validate portable source syntax, configuration and Markdown asset links."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from repository_files import repository_files

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    files = repository_files(root)
    selected = {p.resolve() for p in files}
    errors = []
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        elif path.suffix == ".md":
            for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                ref = urlsplit(link.strip("<>"))
                if ref.scheme or not ref.path:
                    continue
                target = (path.parent / unquote(ref.path)).resolve()
                if target not in selected:
                    errors.append(f"{path.relative_to(root)}: missing archive asset {link}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Checked {len(files)} repository files: syntax, JSON and Markdown links OK")


if __name__ == "__main__":
    check()
