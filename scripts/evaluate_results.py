"""Post-run CPU CLIP evaluation with paired, filter-aware summaries."""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache" / "evaluation_hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/anime_lora")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis")
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import CLIPModel, CLIPProcessor
    from safetensors.torch import load_file

    source = args.source
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    repo = "openai/clip-vit-base-patch32"
    revision_file = output / "clip_model.json"
    if revision_file.exists():
        revision = json.loads(revision_file.read_text())["revision"]
    else:
        revision = HfApi().model_info(repo).sha
        revision_file.write_text(json.dumps({"repo_id": repo, "revision": revision,
                                             "metric": "raw cosine similarity, not scaled CLIPScore",
                                             "device": "cpu", "torch": torch.__version__}, indent=2))
    print(f"Loading frozen CLIP evaluator, revision {revision}", flush=True)
    model = CLIPModel.from_pretrained(repo, revision=revision, use_safetensors=False).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(repo, revision=revision, use_fast=False)
    records = []
    for exp in ("E0", "E1", "E2"):
        rows = [json.loads(line) for line in (source / "generated" / exp / "manifest.jsonl").read_text().splitlines()]
        if len(rows) != 96 or len({(r["prompt_id"], r["seed"]) for r in rows}) != 96:
            raise ValueError(f"Incomplete or duplicate generated set: {exp}")
        for start in range(0, len(rows), 12):
            chunk = rows[start:start + 12]
            images = [Image.open(source / "generated" / exp / r["file"]).convert("RGB") for r in chunk]
            if any(image.size != (512, 512) for image in images):
                raise ValueError("Unexpected generated resolution")
            inputs = processor(text=[r["prompt"] for r in chunk], images=images,
                               padding=True, truncation=True, return_tensors="pt")
            with torch.inference_mode():
                pred = model(**inputs)
                scores = (pred.image_embeds * pred.text_embeds).sum(dim=-1).tolist()
            for row, im, score in zip(chunk, images, scores):
                arr = np.asarray(im)
                record = {"experiment": exp, **row, "clip_cosine": score,
                          "all_black": bool(np.max(arr) == 0),
                          "image_sha256": hashlib.sha256(im.tobytes()).hexdigest()}
                records.append(record)
                im.close()
            print(f"Evaluated {exp}: {min(start + 12, len(rows))}/96", flush=True)
    with (output / "clip_per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lookup = {exp: {(r["prompt_id"], r["seed"]): r for r in records if r["experiment"] == exp}
              for exp in ("E0", "E1", "E2")}
    keys = sorted(lookup["E0"])
    shared = [key for key in keys if all(not lookup[e][key]["safety_flag"] and not lookup[e][key]["all_black"]
                                        for e in lookup)]
    prompts = sorted({key[0] for key in shared})
    table = {}
    for exp in lookup:
        rows = list(lookup[exp].values())
        valid = [r for r in rows if not r["safety_flag"] and not r["all_black"]]
        by_prompt = [float(np.mean([lookup[exp][key]["clip_cosine"] for key in shared if key[0] == p])) for p in prompts]
        table[exp] = {"images": len(rows), "safety_flag_count": sum(r["safety_flag"] for r in rows),
                      "black_image_count": sum(r["all_black"] for r in rows),
                      "unfiltered_images": len(valid),
                      "clip_mean_own_unfiltered": float(np.mean([r["clip_cosine"] for r in valid])),
                      "clip_mean_common_unfiltered": float(np.mean([lookup[exp][k]["clip_cosine"] for k in shared])),
                      "clip_common_prompt_balanced": float(np.mean(by_prompt)),
                      "generation_seconds_mean": float(np.mean([r["seconds"] for r in rows])),
                      "generation_seconds_median": float(np.median([r["seconds"] for r in rows]))}
    rng = np.random.default_rng(20261013)
    differences = {}
    for a, b in (("E1", "E0"), ("E2", "E0"), ("E1", "E2")):
        values = np.array([np.mean([lookup[a][k]["clip_cosine"] - lookup[b][k]["clip_cosine"]
                                   for k in shared if k[0] == p]) for p in prompts])
        bootstrap = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        differences[f"{a}-{b}"] = {"mean_difference": float(values.mean()),
                                     "ci95_prompt_bootstrap": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
                                     "prompt_count": len(values)}
    training = {}
    for exp in ("E1", "E2"):
        log = [json.loads(line) for line in (source / exp / "training.jsonl").read_text().splitlines()]
        steps = [r for r in log if "loss" in r]
        vals = [r for r in log if "validation_loss" in r]
        if [r["step"] for r in steps] != list(range(1, 2001)):
            raise ValueError("Training step sequence is not continuous")
        weights = source / exp / "final/pytorch_lora_weights.safetensors"
        tensors = load_file(weights)
        if not all(torch.isfinite(t).all().item() for t in tensors.values()):
            raise ValueError("Non-finite adapter weights")
        training[exp] = {"optimizer_updates": len(steps), "last_validation_loss": vals[-1]["validation_loss"],
                         "best_validation_loss": min(r["validation_loss"] for r in vals),
                         "update_seconds_total": sum(r["seconds"] for r in steps),
                         "update_seconds_median": float(np.median([r["seconds"] for r in steps])),
                         "first100_mean_loss": float(np.mean([r["loss"] for r in steps[:100]])),
                         "last100_mean_loss": float(np.mean([r["loss"] for r in steps[-100:]])),
                         "peak_allocated_gpu_gb": max(r["peak_memory_gb"] for r in steps),
                         "adapter_bytes": weights.stat().st_size,
                         "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                         "adapter_tensor_count": len(tensors),
                         "adapter_parameters": sum(t.numel() for t in tensors.values())}
    summary = {"clip_model": json.loads(revision_file.read_text()), "models": table,
               "common_unfiltered_images_per_model": len(shared), "common_prompt_count": len(prompts),
               "paired_differences": differences, "training": training,
               "shared_keys": shared,
               "limitations": ["one training seed", "CLIP similarity is not style or anatomy quality",
                               "safety-filtered images excluded from semantic comparison; selection bias possible",
                               "bootstrap resamples prompts, not independent training runs"]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"models": table, "paired_differences": differences,
                      "common_unfiltered_images_per_model": len(shared)}), flush=True)


if __name__ == "__main__":
    main()
