"""
Convert generated images into short videos using Stable Video Diffusion (SVD).

This is an optional addition and nice-to-have. I have not finetuned this SVD model. Just used the existing default.

Given a directory of PNG images (e.g. eval outputs), this script loads StableVideoDiffusion,
applies motion buckets and noise augmentation, and writes MP4 videos to an output directory.

Example:
    python scripts/generate_videos.py --image-dir eval_out_unet_lora --out-dir video_out_unet_lora \
        --motion-bucket-id 150 --noise-aug 0.03 --frames 24 --fps 2
"""

import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video


def load_svd(device="cuda", dtype=torch.float16):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", torch_dtype=dtype, variant="fp16"
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Resize image to be SVD-friendly: shortest side ≈ 576 and divisible by 8.
    """
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) != 576:
        if w < h:
            nw, nh = 576, int(h * 576 / w)
        else:
            nw, nh = int(w * 576 / h), 576
        img = img.resize((nw - nw % 8, nh - nh % 8), Image.BICUBIC)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory of generated PNG images")
    parser.add_argument("--out-dir", required=True, help="Directory to save videos")
    parser.add_argument("--motion-bucket-id", type=int, default=150, help="Motion bucket ID (higher → more motion)")
    parser.add_argument("--noise-aug", type=float, default=0.03, help="Noise augmentation strength")
    parser.add_argument("--frames", type=int, default=24, help="Number of frames per video")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second of output video")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    svd = load_svd(device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    images = [p for p in sorted(os.listdir(args.image_dir)) if p.lower().endswith(".png")]
    for fn in tqdm(images, desc="Converting to video"):
        path = os.path.join(args.image_dir, fn)
        img = Image.open(path).convert("RGB")
        prepped = preprocess_image(img)
        with torch.inference_mode():
            out = svd(
                prepped,
                decode_chunk_size=1,
                motion_bucket_id=args.motion_bucket_id,
                noise_aug_strength=args.noise_aug,
                num_frames=args.frames,
            )
        frames = out.frames[0]
        name = os.path.splitext(fn)[0]
        export_to_video(frames, os.path.join(args.out_dir, f"{name}.mp4"), fps=args.fps)
    print("Video conversion complete.")


if __name__ == "__main__":
    main()
