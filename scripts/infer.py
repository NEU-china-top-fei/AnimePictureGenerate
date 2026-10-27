"""Generate one image with a completed adapter; retains the base safety checker."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="Folder containing pytorch_lora_weights.safetensors; omit for E0")
    parser.add_argument("--prompt", required=True, help="English content description without the anime prefix")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.png"))
    args = parser.parse_args()
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/experiment.json").read_text())
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this FP16 reproduction command.")
    pipe = StableDiffusionPipeline.from_pretrained(cfg["model"]["repo_id"],
        revision=cfg["model"]["revision"], variant="fp16", torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    if args.adapter:
        pipe.load_lora_weights(str(args.adapter), adapter_name="coursework")
        pipe.set_adapters("coursework", adapter_weights=args.scale)
    latent = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(args.seed)).half().cuda()
    prompt = "anime illustration, cel shading, " + args.prompt
    result = pipe(prompt, latents=latent, num_inference_steps=25, guidance_scale=7.0,
                  height=512, width=512)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"prompt": prompt, "seed": args.seed,
        "scale": args.scale if args.adapter else None, "safety_flag": bool(result.nsfw_content_detected[0])}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
