# src/ui_gradio.py
import os, yaml, torch, gradio as gr
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image
from peft import LoraConfig, get_peft_model


def resolve_cache_dir(cfg: dict) -> str:
    ## In the strict order of priority
    return (
        os.getenv("HF_HOME")
        or os.getenv("TRANSFORMERS_CACHE")
        or cfg.get("cache_dir")
        or "./hf_cache"
    )


def supports_bf16(device: str = "cuda") -> bool:
    if device != "cuda" or not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()
    return False


def build_base_pipe(base_id: str, cache_dir: str, cfg: dict, device: str = "cuda"):
    use_bf16 = str(cfg.get("mixed_precision", "")).lower() == "bf16" and supports_bf16(device)
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        base_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def ensure_lora_local(lora_hf_id, lora_subdir=".", token=None):
    try:
        local_dir = snapshot_download(repo_id=lora_hf_id, token=token)
        return os.path.join(local_dir, lora_subdir) if lora_subdir else local_dir
    except Exception as e:
        print(f"LoRA download failed: {e}")
        return None


def run_both(pipe, prompt, steps, scale, h, w, seed, lora_path, cfg, eta_val):
    # 1) base run
    gen = None
    if seed not in (None, ""):
        try:
            s = int(seed)
        except ValueError:
            s = int(cfg.get("seed", 42))
        gen = torch.Generator(device=pipe.device).manual_seed(s)

    with torch.autocast("cuda"):
        base_img = pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(scale),
            height=int(h),
            width=int(w),
            generator=gen,
            eta=float(eta_val),
        ).images[0]

    # 2) LoRA run (PEFT style)
    if not lora_path or not os.path.exists(lora_path):
        blank = Image.new("RGB", base_img.size, (30, 30, 30))
        return base_img, blank, f"LoRA not found at {lora_path or 'N/A'}"

    try:
        lconf = LoraConfig(
            r=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 8),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            bias="none",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        pipe.unet = get_peft_model(pipe.unet, lconf)
        pipe.unet.load_adapter(lora_path, adapter_name="astro")
        pipe.unet.set_adapter("astro")
        for p in pipe.unet.parameters():
            p.requires_grad_(False)

        # same seed for LoRA pass
        gen2 = None
        if gen is not None:
            gen2 = torch.Generator(device=pipe.device).manual_seed(gen.initial_seed())

        with torch.autocast("cuda"):
            lora_img = pipe(
                prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(scale),
                height=int(h),
                width=int(w),
                generator=gen2,
                eta=float(eta_val),
            ).images[0]

        return base_img, lora_img, "LoRA applied successfully."
    except Exception as e:
        err_img = Image.new("RGB", base_img.size, (60, 0, 0))
        return base_img, err_img, f"Error loading/applying LoRA: {e}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora-hf-id", required=True)
    ap.add_argument("--lora-subdir", default="unet_lora_final")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = resolve_cache_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = build_base_pipe(args.base_id, cache_dir, cfg, device=device)

    lora_path = ensure_lora_local(
        args.lora_hf_id,
        args.lora_subdir,
        os.getenv("HF_TOKEN"),
    )

    default_seed = str(cfg.get("seed", 42))

    with gr.Blocks(title="Astro-Diffusion: Base vs LoRA") as demo:
        gr.Markdown("## Astro-Diffusion: Base vs LoRA Comparison")
        gr.Markdown("**Video Generation coming up..**")

        prompt = gr.Textbox(
            value="a vibrant spiral galaxy with glowing nebulae and dust lanes",
            label="Prompt",
        )

        with gr.Row():
            steps = gr.Slider(10, 60, value=cfg.get("num_inference_steps", 30), step=1, label="Steps")
            scale = gr.Slider(1.0, 12.0, value=cfg.get("guidance_scale", 7.5), step=0.5, label="Guidance")
            height = gr.Number(value=cfg.get("height", 512), label="Height")
            width = gr.Number(value=cfg.get("width", 512), label="Width")
            seed = gr.Textbox(value=default_seed, label="Seed")
            eta = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Eta")

        btn = gr.Button("Generate")
        out_base = gr.Image(label="Base Model Output")
        out_sft = gr.Image(label="LoRA Model Output")
        status = gr.Textbox(label="Status", interactive=False)

        def _infer(p, st, sc, h, w, sd, et):
            return run_both(pipe, p, st, sc, h, w, sd, lora_path, cfg, et)

        btn.click(
            _infer,
            [prompt, steps, scale, height, width, seed, eta],
            [out_base, out_sft, status],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
