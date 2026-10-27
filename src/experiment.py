"""SD 1.5 LoRA experiments on two CUDA GPUs, with bounded runtime.

LoRA APIs follow Hugging Face Diffusers' Apache-2.0 text-to-image example:
https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py
This implementation adds cached inputs, paired experiments, time limits and exports.
"""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from control import atomic_json, budget_left, choose_updates, init_budget

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/experiment.json"
ROOT = Path(os.environ.get("EXPERIMENT_ROOT", str(PROJECT / "outputs/experiment"))).resolve()
TMP = Path(os.environ.get("EXPERIMENT_CACHE", str(PROJECT / ".cache/experiment"))).resolve()
STYLE = "anime illustration, cel shading, "
PROMPTS = [
    "a silver-haired traveler in a blue cloak at a night railway station, half-body portrait",
    "a red-haired musician playing a violin on a garden terrace, full body",
    "a black-haired librarian wearing round glasses, holding a green book in a library",
    "a brown-haired chef in a white apron cooking in a sunny kitchen",
    "a young adult explorer with a yellow raincoat walking through a forest",
    "an elderly captain with a white beard on the deck of a wooden ship",
    "a blue-haired woman in a red jacket holding a white umbrella in the rain",
    "a dark-haired man wearing a green scarf and a brown coat, snowy village background",
    "a young adult female knight in silver armor with a purple cape, castle courtyard",
    "an adult male painter in a blue shirt holding a paintbrush in a studio",
    "a black-haired woman reading a book beside a window, warm sunset light",
    "a young adult man with orange hair riding a bicycle beside a river",
    "a woman with short white hair planting flowers in a garden",
    "a man with dark curly hair holding a lantern on a mountain path at night",
    "a female traveler with a backpack and a wide-brimmed hat, desert background",
    "an adult male violinist wearing a dark suit, empty concert hall",
    "a young adult woman in a yellow coat looking at a lighthouse, ocean background",
    "a red-haired adult man holding a basket of apples in an orchard",
    "an adult female mechanic in blue overalls beside a small robot",
    "an elderly woman in a purple shawl drinking tea on a wooden porch",
    "a black-haired adult man practicing martial arts in a bamboo grove, full body",
    "a brown-haired woman wearing a red scarf walking across a stone bridge",
    "an adult male astronomer with glasses beside a telescope under a starry sky",
    "a silver-haired adult woman in a green dress standing in a flower shop",
]
SEEDS = [101, 202, 303, 404]


def config():
    snapshot = ROOT / "config.json"
    return json.loads((snapshot if snapshot.exists() else CONFIG).read_text())


def initialize_run():
    """Create local run inputs without relying on a platform bootstrap script."""
    ROOT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cfg = config()
    if not (ROOT / "config.json").exists():
        atomic_json(ROOT / "config.json", cfg)
    init_budget(ROOT / "budget.json", cfg["budget"]["session_hours_cap"] * 3600)


def emit(message):
    print(message, flush=True)


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image, ImageOps, ImageDraw
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = config()
    TMP.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected two visible CUDA GPUs, found {torch.cuda.device_count()}")
    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpus": [{"name": torch.cuda.get_device_name(i),
                          "bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(2)]}
    atomic_json(ROOT / "hardware.json", hardware)
    emit(f"STARTUP GPU_CHECK {hardware}")
    dataset = snapshot_download(cfg["dataset"]["repo_id"], repo_type="dataset",
                               revision=cfg["dataset"]["revision"], cache_dir=str(TMP / "hub"),
                               allow_patterns=["data/*.parquet", "README.md", "dataset_infos.json"], max_workers=2)
    emit("STARTUP DATASET_DOWNLOADED")
    model = snapshot_download(cfg["model"]["repo_id"], revision=cfg["model"]["revision"],
                             cache_dir=str(TMP / "hub"), max_workers=2,
                             allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                             "feature_extractor/*", "*/config.json", "*/*.fp16.safetensors"])
    (TMP / "model_path.txt").write_text(model)
    emit("STARTUP MODEL_DOWNLOADED")
    rows = []
    rejected = []
    exact_seen = set()
    img_dir = TMP / "images"
    img_dir.mkdir(exist_ok=True)
    for part in sorted(Path(dataset).glob("data/*.parquet")):
        table = pq.read_table(part)
        for item in table.to_pylist():
            original_index = len(rows) + len(rejected)
            try:
                raw = item["image"]["bytes"]
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                text = str(item["text"]).strip()
                if not text or min(image.size) < 128:
                    raise ValueError("empty caption or short side under 128 pixels")
                digest = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
                if digest in exact_seen:
                    raise ValueError("exact duplicate")
                exact_seen.add(digest)
                # Similar-image groups reduce leakage between near-duplicate frames.
                small = np.asarray(image.resize((9, 8)).convert("L"), dtype=np.int16)
                bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
                dhash = sum(int(b) << k for k, b in enumerate(bits))
                thumb = np.asarray(image.resize((16, 16)).convert("L"), dtype=np.float32) / 255
                index = len(rows)
                ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS).save(img_dir / f"{index:05d}.png")
                rows.append({"index": index, "source_index": original_index, "sha256": digest,
                             "caption": text, "width": image.width, "height": image.height,
                             "dhash": dhash, "thumb": thumb})
            except Exception as exc:
                rejected.append({"source_index": original_index, "reason": str(exc)})
        del table
        gc.collect()
    if len(rows) < 200:
        raise RuntimeError(f"Only {len(rows)} usable images; refusing long training")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if (a["dhash"] ^ b["dhash"]).bit_count() <= 4 and float(np.mean((a["thumb"] - b["thumb"]) ** 2)) < 0.02:
                parents[find(i)] = find(j)
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    group_list = list(groups.values())
    random.Random(42).shuffle(group_list)
    splits = {"train": [], "val": [], "test": []}
    for members in group_list:
        split = "train" if len(splits["train"]) < len(rows) * 0.8 else "val" if len(splits["val"]) < len(rows) * 0.1 else "test"
        splits[split].extend(members)
        for i in members:
            rows[i]["split"] = split
            rows[i]["group"] = find(i)
    if min(map(len, splits.values())) < 10:
        raise RuntimeError("Insufficient disjoint validation/test samples after grouping")
    manifest = [{k: v for k, v in r.items() if k != "thumb"} for r in rows]
    atomic_json(ROOT / "data_manifest.json", {"images": manifest, "splits": splits, "rejected": rejected})
    grid = Image.new("RGB", (4 * 256, 4 * 280), "white")
    draw = ImageDraw.Draw(grid)
    for k, i in enumerate(splits["train"][:16]):
        grid.paste(Image.open(img_dir / f"{i:05d}.png").resize((256, 256)), ((k % 4) * 256, (k // 4) * 280))
        draw.text(((k % 4) * 256 + 4, (k // 4) * 280 + 259), f"train image {i}", fill="black")
    grid.save(ROOT / "training_samples.jpg")
    emit(f"STARTUP DATA_READY { {k: len(v) for k, v in splits.items()} }")

    # Frozen modules are used only to cache inputs, then released before U-Net training.
    torch.set_num_threads(2)
    vae = AutoencoderKL.from_pretrained(model, subfolder="vae", variant="fp16", torch_dtype=torch.float32,
                                      local_files_only=True).to("cuda:0").eval()
    means, logvars = [], []
    with torch.inference_mode():
        for i in range(len(rows)):
            arr = np.asarray(Image.open(img_dir / f"{i:05d}.png"), dtype=np.float32) / 127.5 - 1
            pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda:0")
            posterior = vae.encode(pixels).latent_dist
            means.append(posterior.mean.cpu().half())
            logvars.append(posterior.logvar.cpu().half())
    scaling = float(vae.config.scaling_factor)
    del vae
    torch.cuda.empty_cache()
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", variant="fp16",
                                                torch_dtype=torch.float16, local_files_only=True).to("cuda:0").eval()
    embeddings = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            captions = [STYLE + r["caption"] for r in rows[start:start + 16]]
            truncated += sum(len(tokenizer(t).input_ids) > tokenizer.model_max_length for t in captions)
            tokens = tokenizer(captions, max_length=tokenizer.model_max_length, truncation=True,
                               padding="max_length", return_tensors="pt").input_ids.to("cuda:0")
            embeddings.append(text_encoder(tokens)[0].cpu())
        tokens = tokenizer([STYLE.rstrip(", ")], max_length=tokenizer.model_max_length, truncation=True,
                           padding="max_length", return_tensors="pt").input_ids.to("cuda:0")
        simple = text_encoder(tokens)[0].cpu()
    torch.save({"mean": torch.cat(means), "logvar": torch.cat(logvars),
                "detailed": torch.cat(embeddings), "simple": simple,
                "scaling": scaling, "train": torch.tensor(splits["train"]),
                "val": torch.tensor(splits["val"])}, TMP / "cache.pt")
    atomic_json(ROOT / "data_summary.json", {"total_usable": len(rows), "split_counts": {k: len(v) for k,v in splits.items()},
                                           "rejected": len(rejected), "truncated_captions": truncated,
                                           "cache_policy": "posterior mean/logvar; resample latent each training microstep"})
    atomic_json(ROOT / "prompts.json", {"prefix": STYLE, "test": PROMPTS, "seeds": SEEDS})
    emit("STARTUP INPUT_CACHE_READY")


def train(experiment):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    cfg = config()
    out = ROOT / experiment
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES binds each worker.
    model = (TMP / "model_path.txt").read_text()
    data = torch.load(TMP / "cache.pt", map_location="cpu", weights_only=True, mmap=True)
    unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", variant="fp16",
                                              torch_dtype=torch.float16, local_files_only=True).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                               target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for name, param in unet.named_parameters():
        if param.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter {name}")
            param.data = param.data.float()
    unet.enable_gradient_checkpointing()
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    probe = next(p for n, p in unet.named_parameters() if "lora_B" in n and p.requires_grad)
    probe_before = probe.detach().clone()
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler.from_pretrained(model, subfolder="scheduler", local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("This implementation requires the pinned epsilon-prediction SD 1.5 scheduler")
    sample_rng = torch.Generator(device="cpu").manual_seed(42)
    noise_rng = torch.Generator(device=device).manual_seed(42)
    step = 0
    best_val = float("inf")
    state_path = out / "resume.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(unet, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        sample_rng.set_state(state["sample_rng"])
        noise_rng.set_state(state["noise_rng"])
        step, best_val = state["step"], state["best_val"]
        probe_before = probe.detach().clone()
    training_budget = json.loads((ROOT / "training_budget.json").read_text())
    deadline = training_budget["deadline"]
    tr = data["train"]
    aggregate = cfg["training"]["gradient_accumulation_steps"]
    timings = []
    first_marker = False
    skip_count = 0
    atomic_json(out / "model_info.json", {"trainable_parameters": sum(p.numel() for p in params),
                                         "total_parameters": sum(p.numel() for p in unet.parameters()),
                                         "caption_mode": "detailed" if experiment == "E1" else "style_only"})

    def adapter_state():
        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(unet).items()}

    def save_checkpoint(label):
        folder = out / label
        folder.mkdir(exist_ok=True)
        state = adapter_state()
        StableDiffusionPipeline.save_lora_weights(str(folder),
            unet_lora_layers=convert_state_dict_to_diffusers(state), safe_serialization=True)
        atomic_json(folder / "metadata.json", {"step": step, "best_validation_loss": best_val if best_val < float("inf") else None,
                                              "experiment": experiment})
        tmp = out / "resume.tmp.pt"
        torch.save({"step": step, "best_val": best_val, "adapter": state,
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "sample_rng": sample_rng.get_state(), "noise_rng": noise_rng.get_state()}, tmp)
        tmp.replace(state_path)

    def validate():
        unet.eval()
        rng = torch.Generator(device=device).manual_seed(8675309)
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for idx in data["val"][:24]:
                i = int(idx)
                mean = data["mean"][i:i+1].to(device).float()
                std = (0.5 * data["logvar"][i:i+1].to(device).float()).exp()
                latent = ((mean + std * torch.randn(mean.shape, generator=rng, device=device)) * data["scaling"]).half()
                noise = torch.randn(latent.shape, generator=rng, device=device, dtype=torch.float16)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=rng, device=device)
                # Same detailed validation condition for E1 and E2.
                pred = unet(scheduler.add_noise(latent, noise, t), t, data["detailed"][i:i+1].to(device)).sample
                losses.append(float(F.mse_loss(pred.float(), noise.float())))
        unet.train()
        return sum(losses) / len(losses)

    target = cfg["training"]["max_optimizer_updates"]
    with (out / "training.jsonl").open("a", buffering=1) as log:
        try:
            while step < target and time.time() < deadline - 120:
                if (ROOT / "training_target.json").exists():
                    target = json.loads((ROOT / "training_target.json").read_text())["updates"]
                    if step >= target:
                        break
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(aggregate):
                    idx = int(tr[torch.randint(len(tr), (1,), generator=sample_rng)])
                    mean = data["mean"][idx:idx+1].to(device).float()
                    std = (0.5 * data["logvar"][idx:idx+1].to(device).float()).exp()
                    latent = ((mean + std * torch.randn(mean.shape, generator=noise_rng, device=device)) * data["scaling"]).half()
                    hidden = data["detailed"][idx:idx+1] if experiment == "E1" else data["simple"]
                    hidden = hidden.to(device)
                    noise = torch.randn(latent.shape, generator=noise_rng, device=device, dtype=torch.float16)
                    t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=noise_rng, device=device)
                    with torch.autocast("cuda", dtype=torch.float16):
                        prediction = unet(scheduler.add_noise(latent, noise, t), t, hidden).sample
                        loss = F.mse_loss(prediction.float(), noise.float())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite denoising loss")
                    loss_sum += float(loss.detach()) / aggregate
                    scaler.scale(loss / aggregate).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                if scaler.get_scale() < scale_before:
                    skip_count += 1
                    if skip_count > 20:
                        raise RuntimeError("Repeated fp16 overflow")
                    continue
                step += 1
                dt = time.perf_counter() - started
                if step > 5:
                    timings.append(dt)
                row = {"step": step, "loss": loss_sum, "seconds": dt,
                       "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                log.write(json.dumps(row) + "\n")
                if not first_marker:
                    if torch.equal(probe.detach(), probe_before):
                        raise RuntimeError("First optimizer update did not change LoRA weights")
                    atomic_json(out / "training_started.json", {"step": step, "loss": loss_sum, "time": time.time(),
                                                               "lora_weights_changed": True})
                    emit(f"TRAINING_STARTED {experiment} optimizer_step={step} loss={loss_sum:.6f} lora_weights_changed=true")
                    first_marker = True
                if step % 25 == 0:
                    emit(f"{experiment} step={step} loss={loss_sum:.5f} seconds={dt:.2f}")
                if step == 50:
                    atomic_json(out / "pilot.json", {"seconds_per_update": sum(timings) / len(timings), "step": step})
                    while not (ROOT / "training_target.json").exists():
                        if time.time() >= deadline - 180:
                            raise RuntimeError("Pilot coordination exceeded training deadline")
                        time.sleep(2)
                if step % 250 == 0:
                    value = validate()
                    log.write(json.dumps({"step": step, "validation_loss": value}) + "\n")
                    if value < best_val:
                        best_val = value
                        save_checkpoint("best_validation")
                    save_checkpoint(f"checkpoint-{step}")
                    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
                    for obsolete in checkpoints[:-2]:
                        shutil.rmtree(obsolete)
            save_checkpoint("final")
            atomic_json(out / "completed.json", {"step": step, "target": target, "target_reached": step >= target,
                                                 "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9})
            emit(f"TRAINING_FINISHED {experiment} steps={step}")
        except Exception:
            save_checkpoint("interrupted")
            raise


def generate(experiments):
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    cfg = config()
    torch.set_num_threads(2)
    model = (TMP / "model_path.txt").read_text()
    pipe = StableDiffusionPipeline.from_pretrained(model, variant="fp16", torch_dtype=torch.float16,
                                                  local_files_only=True).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_vae_slicing()
    budget = json.loads((ROOT / "budget.json").read_text())
    for exp in experiments.split(","):
        if exp != "E0":
            pipe.load_lora_weights(str(ROOT / exp / "final"))
        out = ROOT / "generated" / exp
        out.mkdir(parents=True, exist_ok=True)
        with (out / "manifest.jsonl").open("a", buffering=1) as manifest:
            for p, text in enumerate(PROMPTS):
                for seed in SEEDS:
                    if budget_left(budget, reserve=900) <= 0:
                        emit(f"GENERATION_BUDGET_STOP {exp}")
                        return
                    name = f"prompt_{p:02d}_seed_{seed}.png"
                    if (out / name).exists():
                        continue
                    # CPU-generated common latent noise gives all models the same input.
                    latent = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(seed), dtype=torch.float32).half().cuda()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = pipe(STYLE + text, latents=latent, num_inference_steps=25,
                                  guidance_scale=7.0, height=512, width=512)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    result.images[0].save(out / name)
                    manifest.write(json.dumps({"prompt_id": p, "prompt": STYLE + text, "seed": seed,
                                               "file": name, "seconds": elapsed,
                                               "safety_flag": bool(result.nsfw_content_detected[0]) if result.nsfw_content_detected else False}) + "\n")
        if exp != "E0":
            pipe.unload_lora_weights()
        emit(f"GENERATION_FINISHED {exp}")


def summarize():
    from PIL import Image, ImageDraw
    summaries = {}
    for exp in ("E1", "E2"):
        file = ROOT / exp / "completed.json"
        if file.exists():
            summaries[exp] = json.loads(file.read_text())
    counts = {exp: len(list((ROOT / "generated" / exp).glob("*.png"))) for exp in ("E0", "E1", "E2")}
    atomic_json(ROOT / "summary.json", {"training": summaries, "generated_counts": counts,
                                       "automatic_quality_scores": "not computed; human evaluation pending"})
    # Fixed first seed, first eight prompts, never cherry-picked samples.
    grid = Image.new("RGB", (3 * 256, 8 * 280), "white")
    draw = ImageDraw.Draw(grid)
    for row in range(8):
        for col, exp in enumerate(("E0", "E1", "E2")):
            path = ROOT / "generated" / exp / f"prompt_{row:02d}_seed_101.png"
            if path.exists():
                grid.paste(Image.open(path).resize((256, 256)), (col * 256, row * 280))
            draw.text((col * 256 + 4, row * 280 + 259), f"{exp} prompt {row} seed 101", fill="black")
    grid.save(ROOT / "comparison_grid.jpg")
    with (ROOT / "human_evaluation.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prompt", "style_1_to_5", "text_match_1_to_5", "anatomy_1_to_5", "overall_1_to_5"])
        candidates = []
        for exp in ("E0", "E1", "E2"):
            for path in (ROOT / "generated" / exp).glob("*.png"):
                candidates.append((exp, path))
        random.Random(517).shuffle(candidates)
        mapping = []
        blind = ROOT / "blind_evaluation"
        blind.mkdir(exist_ok=True)
        for i, (exp, path) in enumerate(candidates):
            name = f"image_{i:04d}.png"
            shutil.copy2(path, blind / name)
            p = int(path.name.split('_')[1])
            writer.writerow([name, STYLE + PROMPTS[p], "", "", "", ""])
            mapping.append({"image_id": name, "experiment": exp, "original": path.name})
        atomic_json(ROOT / "evaluation_key.json", mapping)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for exp in ("E1", "E2"):
        file = ROOT / exp / "training.jsonl"
        if file.exists():
            records = [json.loads(line) for line in file.read_text().splitlines() if line]
            records = [r for r in records if "loss" in r]
            if records:
                # Non-overlapping 25-update means for readable training curves.
                xs, ys = [], []
                for k in range(0, len(records), 25):
                    chunk = records[k:k+25]
                    xs.append(chunk[-1]["step"])
                    ys.append(sum(r["loss"] for r in chunk)/len(chunk))
                ax.plot(xs, ys, label=exp)
    ax.set(xlabel="Optimizer updates", ylabel="Denoising MSE", title="Training losses (25-update means)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "training_curves.png", dpi=160)
    plt.close(fig)
    emit("ARTIFACTS_SAVED")


def launch_worker(mode, arg=None, gpu=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, str(Path(__file__).resolve()), mode]
    if arg is not None:
        command.append(arg)
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    def relay():
        with (ROOT / f"{mode}_{arg or 'all'}.log").open("a", buffering=1) as log:
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    return proc, thread


def stop_workers(workers):
    import signal
    for proc, _ in workers:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
    for proc, thread in workers:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        thread.join(timeout=2)


def orchestrate():
    cfg = config()
    budget = json.loads((ROOT / "budget.json").read_text())
    workers = []
    try:
        prep, thread = launch_worker("prepare")
        workers = [(prep, thread)]
        prep.wait(timeout=min(3600, budget_left(budget, reserve=5400)))
        thread.join(timeout=5)
        if prep.returncode:
            raise RuntimeError("Preparation failed; see prepare_all.log")
        deadline = min(time.time() + 3.5 * 3600, budget["deadline"] - 5400)
        atomic_json(ROOT / "training_budget.json", {"deadline": deadline})
        workers = [launch_worker("train", "E1", 0), launch_worker("train", "E2", 1)]
        target_written = False
        while any(p.poll() is None for p, _ in workers):
            for p, _ in workers:
                if p.poll() is not None and p.returncode != 0:
                    raise RuntimeError("A training process failed; its paired run will be stopped")
            if not target_written and all((ROOT / exp / "pilot.json").exists() for exp in ("E1", "E2")):
                s = max(json.loads((ROOT / exp / "pilot.json").read_text())["seconds_per_update"] for exp in ("E1", "E2"))
                remaining = min(3 * 3600, deadline - time.time() - 180)
                updates = choose_updates(s, remaining)
                if updates < 500:
                    raise RuntimeError("Measured throughput cannot fit 500 paired updates in budget")
                atomic_json(ROOT / "training_target.json", {"updates": updates, "slowest_seconds_per_update": s})
                emit(f"TRAINING_TARGET updates={updates} slowest_seconds={s:.3f}")
                target_written = True
            if time.time() > deadline + 60:
                raise TimeoutError("Training subprocess exceeded its saving deadline")
            time.sleep(2)
        for p, thread in workers:
            thread.join(timeout=5)
            if p.returncode:
                raise RuntimeError("A training process failed")
        steps = [json.loads((ROOT / exp / "completed.json").read_text())["step"] for exp in ("E1", "E2")]
        if len(set(steps)) != 1:
            raise RuntimeError("Paired runs completed different update counts; refusing misleading main comparison")
        workers = [launch_worker("generate", "E0,E2", 0), launch_worker("generate", "E1", 1)]
        for p, _ in workers:
            p.wait(timeout=max(1, budget_left(budget, reserve=600)))
        for p, thread in workers:
            thread.join(timeout=5)
            if p.returncode:
                raise RuntimeError("Generation failed")
        summarize()
        atomic_json(ROOT / "job_status.json", {"state": "completed", "finished_at": time.time(),
                                               "elapsed_hours": (time.time()-budget["started_at"])/3600,
                                               "human_scoring_pending": True})
    except Exception as exc:
        atomic_json(ROOT / "job_status.json", {"state": "failed", "error": str(exc), "time": time.time()})
        emit(f"JOB_FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        stop_workers(workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["orchestrate", "prepare", "train", "generate"])
    parser.add_argument("experiment", nargs="?")
    args = parser.parse_args()
    if args.mode == "train" and args.experiment not in ("E1", "E2"):
        parser.error("train requires E1 or E2")
    if args.mode == "generate" and (not args.experiment or
            any(value not in ("E0", "E1", "E2") for value in args.experiment.split(","))):
        parser.error("generate requires a comma-separated selection of E0,E1,E2")
    try:
        if args.mode in ("orchestrate", "prepare"):
            initialize_run()
        if args.mode == "orchestrate":
            orchestrate()
        elif args.mode == "prepare":
            prepare()
        elif args.mode == "train":
            train(args.experiment)
        else:
            generate(args.experiment)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
