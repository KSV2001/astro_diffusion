# src/ui_gradio.py
import os, yaml, torch, gradio as gr
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image


def build_base_pipe(base_id, cache_dir, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    pipe.to(device)
    return pipe


def ensure_lora_local(lora_hf_id, lora_subdir=".", token=None):
    try:
        local_dir = snapshot_download(
            repo_id=lora_hf_id,
            token=token,
        )
        return os.path.join(local_dir, lora_subdir) if lora_subdir else local_dir
    except Exception as e:
        print(f"LoRA download failed: {e}")
        return None


def run_both(pipe, prompt, steps, scale, h, w, lora_path):
    # base image
    with torch.autocast("cuda"):
        base_img = pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(scale),
            height=int(h),
            width=int(w),
        ).images[0]

    # LoRA missing
    if not lora_path or not os.path.exists(lora_path):
        msg = f"LoRA weights not found at {lora_path or 'N/A'}."
        blank = Image.new("RGB", base_img.size, (30, 30, 30))
        return base_img, blank, msg

    # LoRA applied
    try:
        pipe.load_lora_weights(lora_path)
        with torch.autocast("cuda"):
            sft_img = pipe(
                prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(scale),
                height=int(h),
                width=int(w),
            ).images[0]
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
        msg = "LoRA applied successfully."
    except Exception as e:
        sft_img = Image.new("RGB", base_img.size, (60, 0, 0))
        msg = f"Error loading LoRA: {e}"

    return base_img, sft_img, msg


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora-hf-id", required=True)
    ap.add_argument("--lora-subdir", default="unet_lora_final")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pipe = build_base_pipe(args.base_id, cfg["cache_dir"])

    lora_path = ensure_lora_local(
        args.lora_hf_id,
        args.lora_subdir,
        os.getenv("HF_TOKEN"),
    )

    with gr.Blocks(title="Astro-Diffusion: Base vs LoRA") as demo:
        gr.Markdown("## Astro-Diffusion: Base vs LoRA Comparison")

        prompt = gr.Textbox(
            value="a vibrant spiral galaxy with glowing nebulae and dust lanes",
            label="Prompt",
        )

        with gr.Row():
            steps = gr.Slider(10, 60, value=cfg["num_inference_steps"], step=1, label="Steps")
            scale = gr.Slider(1.0, 12.0, value=cfg["guidance_scale"], step=0.5, label="Guidance")
            height = gr.Number(value=cfg["height"], label="Height")
            width = gr.Number(value=cfg["width"], label="Width")

        btn = gr.Button("Generate")
        out_base = gr.Image(label="Base Model Output")
        out_sft = gr.Image(label="LoRA Model Output")
        status = gr.Textbox(label="Status", interactive=False)

        def _infer(p, st, sc, h, w):
            return run_both(pipe, p, st, sc, h, w, lora_path)

        btn.click(_infer, [prompt, steps, scale, height, width], [out_base, out_sft, status])

    demo.launch(server_name="0.0.0.0", server_port=7860, launch = True)


if __name__ == "__main__":
    main()
