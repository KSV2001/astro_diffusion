# astro_diffusion/src/train/train.py

'''
Launching :

Single GPU : 
python -m accelerate.commands.launch --num_processes 1 astro_diffusion/src/train/train.py --cfg sft_sd15.yaml

Multi-GPU:
accelerate launch --num_processes 4 astro_diffusion/src/train/train.py --cfg sft_sd15.yaml


'''
from __future__ import annotations
import argparse
from accelerate import Accelerator
from configs.train_config import load_cfg
from src.train.train_env import set_dist_env, set_seeds
from src.train.train_loop import run as train_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="sft_sd15.yaml")
    args = ap.parse_args()

    set_dist_env()
    cfg = load_cfg(args.cfg)

    acc = Accelerator()
    set_seeds(cfg.seed, acc.process_index)
    train_run(cfg)

if __name__ == "__main__":
    main()

