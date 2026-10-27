"""Export only source repository files, excluding local coursework archives."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from repository_files import repository_files

ROOT = Path(__file__).resolve().parents[1]


def export(root=ROOT):
    root = Path(root)
    target = root / "output/anime_lora_github.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = repository_files(root)
    hashes = {}
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr("anime_lora/" + name, content)
        archive.writestr("anime_lora/SHA256SUMS.json", json.dumps(hashes, indent=2) + "\n")
    print(f"{target}: {len(files)} files, {target.stat().st_size / 1e6:.2f} MB")
    return target


if __name__ == "__main__":
    export()
