# astro_diffusion/src/train/train_models.py
from __future__ import annotations
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

def load_components(repo_or_base: str, cache_dir: str | None = None):
    tok  = CLIPTokenizer.from_pretrained(repo_or_base, subfolder="tokenizer", cache_dir=cache_dir)
    txt  = CLIPTextModel.from_pretrained(repo_or_base, subfolder="text_encoder", cache_dir=cache_dir)
    vae  = AutoencoderKL.from_pretrained(repo_or_base, subfolder="vae", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_pretrained(repo_or_base, subfolder="unet", cache_dir=cache_dir)
    sch  = DDPMScheduler.from_pretrained(repo_or_base, subfolder="scheduler", cache_dir=cache_dir)
    return tok, txt, vae, unet, sch

def freeze_inference_modules(text_encoder, vae):
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

def maybe_lora_wrap_unet(unet, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return unet
    targets = ["to_q","to_k","to_v","to_out.0"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets)
    return get_peft_model(unet, cfg)

def maybe_enable_xformers(unet, enable: bool):
    if not enable: return
    try: unet.enable_xformers_memory_efficient_attention()
    except Exception: pass

def place_frozen_modules(text_encoder, vae, device, mp_dtype: str):
    dtype = torch.bfloat16 if mp_dtype=="bf16" else torch.float16 if mp_dtype=="fp16" else torch.float32
    vae.to(device, dtype=torch.float32)   # keep fp32
    text_encoder.to(device, dtype=dtype)
