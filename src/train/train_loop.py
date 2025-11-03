# astro_diffusion/src/train/train_loop.py
from __future__ import annotations
import os, torch
from accelerate import Accelerator
from train_models import (
    load_components, freeze_inference_modules, maybe_lora_wrap_unet,
    maybe_enable_xformers, place_frozen_modules
)
from train_data import build_dataloader
from train_optim import build_optimizer, build_warmup_scheduler

def run(cfg):
    accelerator = Accelerator(mixed_precision=cfg.model.dtype, gradient_accumulation_steps=cfg.train.grad_accum)

    tok, txt, vae, unet, sch = load_components(cfg.model.base, cfg.model.cache_dir)
    freeze_inference_modules(txt, vae)
    unet = maybe_lora_wrap_unet(unet, cfg.train.use_lora, cfg.train.lora_r, cfg.train.lora_alpha, cfg.train.lora_dropout)
    maybe_enable_xformers(unet, cfg.train.use_xformers)

    train_dl = build_dataloader(cfg.data.train_manifest, cfg.data.image_size, cfg.train.batch_size,
                                cfg.data.num_workers, tok)

    params = [p for p in unet.parameters() if p.requires_grad]
    optim = build_optimizer(params, cfg.train.lr, cfg.train.betas, cfg.train.weight_decay, cfg.train.use_8bit_adam)
    sched = build_warmup_scheduler(optim, cfg.train.lr_warmup_steps)

    unet, optim, sched, train_dl = accelerator.prepare(unet, optim, sched, train_dl)
    place_frozen_modules(txt, vae, accelerator.device, cfg.model.dtype)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    unet.train()
    global_step = 0

    while global_step < cfg.train.max_steps:
        for batch in train_dl:
            if global_step >= cfg.train.max_steps: break
            with accelerator.accumulate(unet):
                px = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                ids = batch["input_ids"].to(accelerator.device, non_blocking=True)

                with torch.no_grad():
                    lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(lat)
                    t = torch.randint(0, sch.config.num_train_timesteps, (lat.size(0),), device=lat.device, dtype=torch.long)
                    noisy = sch.add_noise(lat, noise, t)
                    h = txt(ids)[0]

                pred = unet(noisy, t, h).sample
                loss = torch.nn.functional.mse_loss(pred, noise, reduction="mean")

                accelerator.backward(loss)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                with torch.no_grad():
                    mean_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
                if accelerator.is_local_main_process and (global_step % cfg.train.log_every == 0 or global_step == 1):
                    lr = sched.get_last_lr()[0]
                    print(f"[step {global_step}/{cfg.train.max_steps}] loss={mean_loss:.4f} lr={lr:.2e}", flush=True)

                sched.step()
                if accelerator.is_main_process and (global_step % cfg.train.checkpoint_every == 0):
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(cfg.train.output_dir, f"unet_lora_step_{global_step}"))

    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(cfg.train.output_dir, "unet_lora_final"))
