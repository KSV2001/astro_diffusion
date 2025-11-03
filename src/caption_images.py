"""
Caption images using Qwen2-VL-7B-Instruct in NF 4-bit mode.

Given a directory containing `_clean` images, this script resizes each image to 512×512,
generates a descriptive few-sentence caption describing the object and visual details, and writes
a JSONL manifest with image paths and captions.

Example: (change the paths as per your repo structure)
    python scripts/caption_images.py --image-dir data/train_512 \
        --out-jsonl data/astro_captions.jsonl --model-root models
"""

import os
import json
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import logging as log


def load_model(model_root: str):
    repo_id = "Qwen/Qwen2-VL-7B-Instruct"
    cache = os.path.join(model_root, "hf_cache")
    local_repo = os.path.join(model_root, "qwen2vl-7b-instruct")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(local_repo, exist_ok=True)

    # redirect HF caches
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["TRANSFORMERS_CACHE"] = cache
    os.environ["HF_DATASETS_CACHE"] = cache
    os.environ["XDG_CACHE_HOME"] = cache

    # download snapshot locally (symlinks)
    snapshot_download(
        repo_id=repo_id, cache_dir=cache, local_dir=local_repo, local_dir_use_symlinks=True, revision="main"
    )

    # 4-bit quantization config
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    max_mem = {i: "10GiB" for i in range(torch.cuda.device_count())}
    max_mem["cpu"] = "32GiB"

    # load processor and model from local snapshot
    processor = AutoProcessor.from_pretrained(local_repo, cache_dir=cache, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        local_repo,
        cache_dir=cache,
        trust_remote_code=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
    ).eval()
    return processor, model


def resize_rgb(path: str, size: int = 512) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return im


@torch.inference_mode()
def caption_batch(model, processor, images, prompt: str):
    # build conversations for each image
    conversations = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        for img in images
    ]
    chats = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations
    ]
    inputs = processor(text=chats, images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # ensure pad/eos tokens
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    gen = model.generate(**inputs, max_new_tokens=96, do_sample=False, temperature=0.0)
    # compute output tokens per example
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    captions = []
    for r, L in enumerate(prompt_lens):
        new_tokens = gen[r, L:]
        captions.append(processor.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0].strip())
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory with *_clean images to caption")
    parser.add_argument("--out-jsonl", default ="data/astro_captions.jsonl", required=True, help="Path to save JSONL with captions")
    parser.add_argument("--model-root", default="models", help="Directory to store local HF models")
    parser.add_argument("--batch-size", type=int, default=50, help="Captioning batch size")
    args = parser.parse_args()

    proc, model = load_model(args.model_root)
    prompt_text = (
        "Caption this astronomical image in one precise sentence. "
        "Name the object class and describe specific visual details (colors, structures)."
    )
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image_paths = [*glob.glob(os.path.join(args.image_dir, "**", "*.jpg"), recursive=True)]
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Caption images"):
            batch_paths = image_paths[i : i + args.batch_size]
            imgs = [resize_rgb(p) for p in batch_paths]
            caps = caption_batch(model, proc, imgs, prompt_text)
            for p, c in zip(batch_paths, caps):
                f.write(json.dumps({"path": p, "caption": c}, ensure_ascii=False) + "\n")
    log.info("Saved captions to", args.out_jsonl)


if __name__ == "__main__":
    main()
