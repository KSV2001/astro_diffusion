# astro_diffusion/configs/train_config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class ModelCfg:
    base: str
    dtype: str = "bf16"
    cache_dir: str | None = None

@dataclass
class DataCfg:
    train_manifest: str
    val_manifest: str
    image_size: int = 512
    num_workers: int = 4

@dataclass
class TrainCfg:
    output_dir: str = "outputs/"
    batch_size: int = 4
    grad_accum: int = 1
    max_steps: int = 3000
    lr: float = 1e-4
    lr_warmup_steps: int = 200
    checkpoint_every: int = 500
    enable_ema: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = False
    use_8bit_adam: bool = False
    log_every: int = 50
    use_xformers: bool = True
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)

@dataclass
class Cfg:
    seed: int
    model: ModelCfg
    data: DataCfg
    train: TrainCfg

def load_cfg(path: str | Path) -> Cfg:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return Cfg(
        seed=y["seed"],
        model=ModelCfg(**y["model"]),
        data=DataCfg(**y["data"]),
        train=TrainCfg(**y["train"]),
    )
