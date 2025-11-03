# astro_diffusion/src/train/train_optim.py
from __future__ import annotations
import torch

def build_optimizer(params, lr: float, betas: tuple[float,float], weight_decay: float, use_8bit_adam: bool):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

def build_warmup_scheduler(optim, warmup_steps: int):
    def lr_lambda(step: int):
        if warmup_steps <= 0: return 1.0
        return min(1.0, step / float(max(1, warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
