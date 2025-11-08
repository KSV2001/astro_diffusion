# src/ui_gradio.py
import os
import yaml
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from peft import LoraConfig, get_peft_model
from src.ratelimits import RateLimiter
import time


# ---------------------------------------------------------
# utils
# ---------------------------------------------------------
def resolve_cache_dir(cfg: dict) -> str:
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


def ensure_lora_local(
    lora_hf_id: str,
    lora_subdir: str = None,
    token: str = None,
    cache_dir: str = None,
) -> str:
    repo_root = snapshot_download(
        repo_id=lora_hf_id,
        token=token,
        cache_dir=cache_dir,
        local_dir_use_symlinks=True,
    )

    if lora_subdir:
        lora_dir = os.path.join(repo_root, lora_subdir)
    else:
        lora_dir = repo_root

    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"[LoRA] {lora_dir} does not contain adapter_config.json. "
            f"Fix your HF repo layout (expected {lora_hf_id}/{lora_subdir}/adapter_config.json)."
        )

    return lora_dir


def get_autocast_ctx(pipe):
    dev = getattr(pipe, "device", None)
    if dev is None:
        return torch.autocast("cuda")
    dev_type = getattr(dev, "type", "cuda")
    if dev_type == "cpu":
        return torch.cpu.amp.autocast()
    return torch.autocast(dev_type)


# ---------------------------------------------------------
# inference
# ---------------------------------------------------------
def run_both_two_pipes(
    base_pipe,
    lora_pipe,
    prompt,
    steps,
    scale,
    h,
    w,
    seed,
    has_lora,
    cfg,
    eta_val,
):
    gen = None
    if seed not in (None, ""):
        try:
            s = int(seed)
        except ValueError:
            s = int(cfg.get("seed", 42))
        gen = torch.Generator(device=base_pipe.device).manual_seed(s)

    # base
    with get_autocast_ctx(base_pipe):
        base_out = base_pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(scale),
            height=int(h),
            width=int(w),
            generator=gen,
            eta=float(eta_val),
        )
    base_img = base_out.images[0]

    base_flagged = False
    if hasattr(base_out, "nsfw_content_detected") and base_out.nsfw_content_detected:
        if base_out.nsfw_content_detected[0]:
            base_flagged = True
            base_img = Image.new("RGB", (int(w), int(h)), (40, 40, 40))

    # if no LoRA, just return
    if not has_lora:
        status = "LoRA not found or failed to load at startup."
        if base_flagged:
            status = "Base output flagged by safety checker."
        return base_img, Image.new("RGB", base_img.size, (30, 30, 30)), status

    # lora
    gen2 = None
    if gen is not None:
        gen2 = torch.Generator(device=lora_pipe.device).manual_seed(int(gen.initial_seed()))

    with get_autocast_ctx(lora_pipe):
        lora_out = lora_pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(scale),
            height=int(h),
            width=int(w),
            generator=gen2,
            eta=float(eta_val),
        )
    lora_img = lora_out.images[0]

    lora_flagged = False
    if hasattr(lora_out, "nsfw_content_detected") and lora_out.nsfw_content_detected:
        if lora_out.nsfw_content_detected[0]:
            lora_flagged = True
            lora_img = Image.new("RGB", (int(w), int(h)), (60, 60, 60))

    # build status msg
    if base_flagged and lora_flagged:
        status = "Both base and LoRA outputs were flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    elif base_flagged:
        status = "Base output was flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    elif lora_flagged:
        status = "LoRA output was flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    else:
        status = "LoRA applied successfully."

    return base_img, lora_img, status

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora-hf-id", required=True)          # e.g. Srikasi/astro-diffusion
    ap.add_argument("--lora-subdir", default="unet_lora_final")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cache_dir = resolve_cache_dir(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) pure base pipe
    base_pipe = build_base_pipe(args.base_id, cache_dir, cfg, device=device)
    # force DDIM so eta is honored
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)

    # 2) second pipe which will get LoRA
    lora_pipe = build_base_pipe(args.base_id, cache_dir, cfg, device=device)
    # force DDIM so eta is honored
    lora_pipe.scheduler = DDIMScheduler.from_config(lora_pipe.scheduler.config)

    # 3) download LoRA
    try:
        lora_path = ensure_lora_local(
            args.lora_hf_id,
            args.lora_subdir,
            os.getenv("HF_TOKEN"),
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"[LoRA] download/resolve failed: {e}")
        lora_path = None

    has_lora = False
    if lora_path:
        try:
            lconf = LoraConfig(
                r=cfg.get("lora_rank", 16),
                lora_alpha=cfg.get("lora_alpha", 16),
                lora_dropout=cfg.get("lora_dropout", 0.0),
                bias="none",
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
            lora_pipe.unet = get_peft_model(lora_pipe.unet, lconf)
            lora_pipe.unet.load_adapter(lora_path, adapter_name="astro")

            # freeze
            for p in lora_pipe.unet.parameters():
                p.requires_grad_(False)

            if hasattr(lora_pipe.unet, "set_adapter"):
                lora_pipe.unet.set_adapter("astro")

            has_lora = True
            print(f"[LoRA] loaded from {lora_path} into lora_pipe")
        except Exception as e:
            print(f"[LoRA] attach failed: {e}")
            has_lora = False

    default_seed = str(cfg.get("seed", 1234))

    limiter = RateLimiter()
    
    # 4) Gradio UI
    with gr.Blocks(title="Astro-Diffusion: Base vs LoRA") as demo:
        session_state = gr.State({"count": 0, "started_at": time.time()})
        gr.Markdown("## Astro-Diffusion: Base vs LoRA Comparison")
        gr.Markdown("**Video Generation coming up..**")

        prompt = gr.Textbox(
            value="a vibrant spiral galaxy with glowing nebulae and dust lanes",
            label="Prompt",
        )

        with gr.Row():
            steps = gr.Slider(10, 60, value=cfg.get("num_inference_steps", 30), step=1, label="Steps")
            scale = gr.Slider(1.0, 12.0, value=cfg.get("guidance_scale", 7.5), step=0.5, label="Guidance")
            height = gr.Number(
                value=min(int(cfg.get("height", 512)), 512),
                label="Height",
                minimum=32,
                maximum=512,
            )
            width = gr.Number(
                value=min(int(cfg.get("width", 512)), 512),
                label="Width",
                minimum=32,
                maximum=512,
            )
            seed = gr.Textbox(value=default_seed, label="Seed")
            eta = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Eta")

        btn = gr.Button("Generate")
        out_base = gr.Image(label="Base Model Output")
        out_sft = gr.Image(label="LoRA Model Output")
        status = gr.Textbox(label="Status", interactive=False)

        #---------------------------------------------------------------------------
        # Inner function for inference helping
        def _infer(p, st, sc, h, w, sd, et, sess_state, request: gr.Request):
            # try real client IP behind RunPod/reverse proxy
            if request is not None:
                # header names can vary in casing
                hdrs = {k.lower(): v for k, v in request.headers.items()} if request.headers else {}
                xff = hdrs.get("x-forwarded-for")
                if xff:
                    ip = xff.split(",")[0].strip()
                elif request.client:
                    ip = request.client.host
                else:
                    ip = "unknown"
            else:
                ip = "unknown"
            print(f"[INFER] ip={ip} sess_state={sess_state}")

            # pre-check
            allowed, reason = limiter.pre_check(ip, sess_state)
            if not allowed:
                print(f"[RL] blocked ip={ip} reason={reason}")
                blank = Image.new("RGB", (int(w), int(h)), (30, 30, 30))
                # return sess_state unchanged
                return blank, blank, f"Rate limited: {reason}", sess_state

            t0 = time.time()
            # your original generation
            base_img, lora_img, msg = run_both_two_pipes(
                base_pipe,
                lora_pipe,
                p,
                st,
                sc,
                h,
                w,
                sd,
                has_lora,
                cfg,
                et,
            )
            dt = time.time() - t0

            # post-consume for time + cost
            limiter.post_consume(ip, dt)
            print(f"[RL] ip={ip} duration={dt:.3f}s")

            return base_img, lora_img, msg, sess_state
        #------------------------------------------------------------------------

        btn.click(
            _infer,
            [prompt, steps, scale, height, width, seed, eta, session_state],
            [out_base, out_sft, status, session_state],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
