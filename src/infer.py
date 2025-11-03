"""
Simple inference script for generating images with a fine-tuned Stable Diffusion model.

Usage:
    python -m astro.infer --config configs/infer.yaml --prompt "a colorful spiral galaxy" --out outputs/samples
"""

import argparse
import os
import yaml
import torch
from diffusers import StableDiffusionPipeline
import logging as log

def load_cfg(path: str):
    return yaml.safe_load(open(path, "r"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to inference YAML config")
    parser.add_argument("--prompt", required=True, help="Prompt to generate image")
    parser.add_argument("--out", default="outputs/samples", help="Directory to save generated image")
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_dir"],
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        cache_dir=cfg["cache_dir"],
    ).to("cuda")

    img = pipe(
        args.prompt,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        height=cfg["height"],
        width=cfg["width"],
    ).images[0]
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "gen.png")
    img.save(fp)
    log.info("Saved image to", fp)


if __name__ == "__main__":
    main()
