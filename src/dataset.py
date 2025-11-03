from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
from PIL import Image
import json


class JsonlImageCaptionDataset(Dataset):
    """
    A simple dataset that loads image-caption pairs from a JSONL manifest.

    Each line in the manifest must be a JSON object with keys:
      - "path": path to an image file
      - "caption": text caption

    Args:
        manifest (str or Path): Path to the JSONL file.
        transform (callable, optional): Optional transform to apply to each image.
    """

    def __init__(self, manifest: str | Path, transform=None):
        self.manifest = Path(manifest)
        self.items = [json.loads(l) for l in self.manifest.read_text().splitlines()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict:
        record = self.items[index]
        img = Image.open(record["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "text": record["caption"]}
