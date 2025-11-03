# astro_diffusion/src/train/train_data.py
from __future__ import annotations
from typing import Dict, List
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from dataset import JsonlImageCaptionDataset
from transforms import train_transforms

def make_collate_fn(tokenizer: CLIPTokenizer):
    max_len = tokenizer.model_max_length
    def _collate(batch: List[Dict]):
        px = [b["pixel_values"] for b in batch]
        txt = [b["text"] for b in batch]
        enc = tokenizer(txt, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        return {"pixel_values": __import__("torch").stack(px, 0), "input_ids": enc.input_ids}
    return _collate

def build_dataloader(train_manifest: str, image_size: int, batch_size: int,
                     num_workers: int, tokenizer: CLIPTokenizer):
    ds = JsonlImageCaptionDataset(train_manifest, transform=train_transforms(image_size))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
        collate_fn=make_collate_fn(tokenizer),
    )
