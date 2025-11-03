#!/usr/bin/env python3
"""
Create a JSONL manifest from a directory of images and a captions JSON file.

The captions JSON should be a dictionary mapping filename (with or without extension) to caption.
The output JSONL will contain lines with "path" and "caption" fields.

Example: (Change the paths as per your folder structure)
    python scripts/make_manifest_from_folder.py --images-dir data/astro/train_clean \
        --captions-json captions.json --out-jsonl data/train.jsonl
"""

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--captions-json", required=True, help="JSON file mapping image names to captions")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    caps = json.loads(pathlib.Path(args.captions_json).read_text())
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for img in sorted(pathlib.Path(args.images_dir).glob("*.jpg")):
            name = img.name
            caption = caps.get(name) or caps.get(img.stem) or ""
            f.write(json.dumps({"path": str(img), "caption": caption}, ensure_ascii=False) + "\n")
    print("Wrote manifest to", args.out_jsonl)


if __name__ == "__main__":
    main()
