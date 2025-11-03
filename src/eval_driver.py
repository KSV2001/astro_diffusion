import os, json, pandas as pd
from torchvision.utils import save_image
from astro_like_cfg import CFG  # or inline your config object
from astro_diffusion.src.eval import grid_for_prompts
import logging as log


prompts = [
    "a high-resolution spiral galaxy with blue star-forming arms and a bright yellow core",
    "a crimson emission nebula with dark dust lanes and scattered newborn stars",
    "a ringed gas giant with visible storm bands and subtle shadow on rings",
    "an accretion disk around a black hole with relativistic jets, high contrast",
]

before = grid_for_prompts(CFG, prompts, lora_dir=None, seed=1234)
after  = grid_for_prompts(CFG, prompts, lora_dir=os.path.join(CFG.out_dir, "unet_lora_final"), seed=1234)
os.makedirs("eval_grids", exist_ok=True)
save_image(before, "eval_grids/before.png")
save_image(after,  "eval_grids/after.png")
log.info("Saved eval_grids images")
