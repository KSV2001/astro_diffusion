"""
Deduplicate images within a dataset using OpenCLIP embeddings and FAISS.

Given a directory of images with nested subfolders (e.g. stars/, nebulae/, etc.),
this script computes CLIP embeddings, finds near-duplicates via cosine similarity,
and copies only unique images to a sibling directory with a `_clean` suffix.

Example:
    python scripts/preprocess_images.py --data-root data/astro/train --suffix _clean --batch-size 64
"""

import os
import glob
import argparse
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import open_clip
import logging as log

def dedup_subdir(in_dir: str, out_dir: str, batch_size: int, device: str = "cuda"):
    '''
    Load all the .jpg's in in_dir folder, get the image-embeddings from ViT model, and remove the duplicates with high similarity.
    Save the outputs in out_dir
    Process the encoding of images in batches of batch_size
    '''
    # load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()

    # optional resize/crop for large images
    pre_resize = T.Compose([T.Resize(256, Image.BICUBIC), T.CenterCrop(224)])

    # gather image paths
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths += glob.glob(os.path.join(in_dir, "**", ext), recursive=True)
    if not paths:
        return

    embeddings, keep_paths = [], []

    # batch encode images
    for i in tqdm(range(0, len(paths), batch_size), desc=f"Embed {os.path.basename(in_dir)}"):
        batch = paths[i : i + batch_size]
        imgs = []
        for p in batch:
            try:
                im = Image.open(p).convert("RGB")
                im = pre_resize(im)
                imgs.append(preprocess(im))
                keep_paths.append(p)
            except Exception:
                # skip problematic files
                pass
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            feats = model.encode_image(x)
        feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy().astype("float32")
        embeddings.append(feats)

    if not embeddings:
        return

    X = np.vstack(embeddings)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    D, I = index.search(X, 2)  # find nearest neighbor for each
    dup_mask = (D[:, 1] > 0.995)  # cosine similarity threshold

    # copy non-duplicate images to output directory
    for i, p in enumerate(keep_paths):
        if dup_mask[i]:
            continue
        rel = os.path.relpath(p, in_dir)
        out_p = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        try:
            Image.open(p).save(out_p, quality=95)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Top-level folder containing class subfolders")
    parser.add_argument("--suffix", default="_clean", help="Suffix for deduplicated folders")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CLIP encoding")
    parser.add_argument("--device", default="cuda", help="Device for CLIP model")
    args = parser.parse_args()

    for name in os.listdir(args.data_root):
        in_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(in_dir) or name.endswith(args.suffix):
            continue
        out_dir = in_dir + args.suffix
        os.makedirs(out_dir, exist_ok=True)
        dedup_subdir(in_dir, out_dir, args.batch_size, args.device)
    log.info("Deduplication complete.")


if __name__ == "__main__":
    main()
