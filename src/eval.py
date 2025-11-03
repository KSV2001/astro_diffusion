"""
Evaluation utilities: generate grids of images for prompts, compute CLIP scores and FID.

This module provides:
  - load_pipe: load a Stable Diffusion pipeline with or without a LoRA adapter.
  - grid_for_prompts: generate a grid of images for a list of prompts.
  - eval_clip: compute mean CLIP score for a list of prompts using a pipeline.
  - fid_from_lists: compute FID and IS between generated and real images using torch-fidelity.

Usage example (within a script or notebook):
    from astro.eval import load_pipe, grid_for_prompts, eval_clip, fid_from_lists
"""

import os
import numpy as np
from typing import List, Optional, Tuple

import torch
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import open_clip
import torch_fidelity


def load_pipe(cfg, lora_dir: Optional[str] = None):
    """
    Load a Stable Diffusion pipeline. If lora_dir is provided, wrap the UNet with the same LoRA
    configuration used for training and load the adapter weights.
    """
    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id, torch_dtype=dtype
    ).to(Accelerator().device)
    if cfg.use_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    if lora_dir:
        lconf = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        pipe.unet = get_peft_model(pipe.unet, lconf)
        pipe.unet.load_adapter(lora_dir, adapter_name="default")
        pipe.unet.set_adapter("default")
        for p in pipe.unet.parameters():
            p.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.inference_mode()
def grid_for_prompts(
    cfg, prompts: List[str], lora_dir: Optional[str] = None, seed: int = 0, steps: int = 30, guidance: float = 7.5
) -> torch.Tensor:
    """
    Generate a horizontal grid of images for a list of prompts. If lora_dir is provided, use the LoRA adapter.
    """
    device = Accelerator().device
    g = torch.Generator(device=device).manual_seed(seed)
    pipe = load_pipe(cfg, lora_dir=lora_dir)
    images = []
    for p in prompts:
        img = pipe(p, num_inference_steps=steps, guidance_scale=guidance, generator=g).images[0]
        images.append(transforms.ToTensor()(img))
    grid = make_grid(torch.stack(images, dim=0), nrow=len(prompts), padding=8, normalize=True, value_range=(0, 1))
    return grid


def eval_clip(prompts: List[str], pipe) -> Tuple[float, float]:
    """
    Compute mean and std of CLIP scores for a list of prompts using a given pipeline.
    """
    device = Accelerator().device
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer_oc = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    def clipscore(img_pil, text):
        with torch.inference_mode():
            image = preprocess(img_pil).unsqueeze(0).to(device)
            text_tokens = tokenizer_oc([text]).to(device)
            img_feat = model.encode_image(image)
            txt_feat = model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).item()

    scores = []
    for p in prompts:
        img = pipe(p, num_inference_steps=30, guidance_scale=7.5).images[0]
        scores.append(clipscore(img, p))
    scores = np.array(scores)
    return scores.mean().item(), scores.std().item()


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str]):
        self.paths = list(paths)
        self.to_uint8 = transforms.PILToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = Image.open(self.paths[index]).convert("RGB")
        x = self.to_uint8(x)  # uint8 CHW
        return x


def fid_from_lists(gen_paths: List[str], real_paths: List[str], batch_size: int = 32, device: str = "cuda"):
    """
    Compute FID and Inception scores between generated and real image sets using torch-fidelity.

    Returns a dictionary with keys: 'frechet_inception_distance' and 'inception_score_mean', etc.
    """
    dl_gen = ImageListDataset(gen_paths)
    dl_real = ImageListDataset(real_paths)
    metrics = torch_fidelity.calculate_metrics(
        input1=dl_gen,
        input2=dl_real,
        batch_size=batch_size,
        cuda=(device == "cuda"),
        fid=True,
        isc=True,
        kid=False,
        verbose=False,
    )
    return metrics
