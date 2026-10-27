"""Ensure the code export excludes course reports and platform tooling."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from export_repository import export
from repository_files import repository_files
import experiment


class RepositoryTests(unittest.TestCase):
    def test_export_excludes_reports_platform_files_and_local_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            keep = {"README.md", "src/experiment.py", "requirements/train.txt",
                    "results/analysis/metrics.json"}
            omit = {".local/archive/scripts/build_report.py", "reports/report.md",
                    "reports/report.pdf", "kaggle/run_experiment.py", ".env",
                    "src/__pycache__/module.py", "output/old.zip", "tmp/page.png",
                    "results/anime_lora/config.json", "results/anime_lora/generated/E0/image.png"}
            for name in keep | omit:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("example\n")
            self.assertEqual({str(p.relative_to(root)) for p in repository_files(root)}, keep)
            with contextlib.redirect_stdout(io.StringIO()):
                target = export(root)
            with ZipFile(target) as archive:
                hashes = json.loads(archive.read("anime_lora/SHA256SUMS.json"))
                self.assertEqual(set(hashes), keep)
                for name, digest in hashes.items():
                    self.assertEqual(hashlib.sha256(archive.read("anime_lora/" + name)).hexdigest(), digest)

    def test_local_run_initializes_snapshot_and_preserves_existing_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "experiment.json"
            config.write_text(json.dumps({"budget": {"session_hours_cap": 7}}))
            run, cache = root / "run", root / "cache"
            with patch.object(experiment, "ROOT", run), patch.object(experiment, "TMP", cache), patch.object(experiment, "CONFIG", config):
                experiment.initialize_run()
                original = json.loads((run / "budget.json").read_text())
                config.write_text(json.dumps({"budget": {"session_hours_cap": 20}}))
                experiment.initialize_run()
                self.assertEqual(experiment.config()["budget"]["session_hours_cap"], 7)
                self.assertEqual(json.loads((run / "budget.json").read_text())["deadline"], original["deadline"])
                self.assertTrue(cache.is_dir())


if __name__ == "__main__":
    unittest.main()
