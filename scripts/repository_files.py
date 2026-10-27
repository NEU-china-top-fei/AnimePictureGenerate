"""Explicit file selection for the source repository export."""
from pathlib import Path

TOP_LEVEL = ("README.md", "NOTICE.md", "Makefile", ".gitignore", ".gitattributes", ".editorconfig")
DIRECTORIES = {
    "src": {".py"},
    "scripts": {".py"},
    "tests": {".py"},
    "configs": {".json"},
    "requirements": {".txt"},
    ".github/workflows": {".yml", ".yaml"},
}
LOCAL_PARTS = {"__pycache__", ".cache", ".local", ".kaggle", ".aws", ".agents", ".codex"}
SECRET_NAMES = {"access_token", "kaggle.json", ".env"}


def allowed(path):
    return (not path.is_symlink() and path.is_file()
            and not any(p in LOCAL_PARTS or p.startswith(".venv") for p in path.parts)
            and path.name not in SECRET_NAMES
            and not path.name.startswith((".env.", "~$", ".~lock.")))


def repository_files(root):
    """Select repository content without requiring local run outputs."""
    root = Path(root)
    selected = {root / p for p in TOP_LEVEL if allowed(root / p)}
    for directory, suffixes in DIRECTORIES.items():
        selected.update(p for p in (root / directory).rglob("*")
                        if p.suffix in suffixes and allowed(p))
    selected.update(p for p in (root / "results/analysis").glob("*")
                    if p.suffix in {".json", ".csv"} and allowed(p))
    readme = root / "results/README.md"
    if allowed(readme):
        selected.add(readme)
    return sorted(selected)
