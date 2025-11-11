# src/api.py
import os, io, base64, time, yaml, torch, logging, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model

from src.ratelimits import RateLimiter

logging.basicConfig(
    level=os.getenv("AD_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("astro-diffusion-api")

app = FastAPI(title="astro-diffusion-api")
limiter = RateLimiter()

# -------- simple metrics (in-process) --------
REQUEST_COUNT = 0
INFER_COUNT = 0
LAST_REQUEST_AT = 0.0

# -------- sessions --------
SESSIONS: dict[str, dict] = {}
SESSION_TTL = 30 * 60  # 30 min
MAX_SESSIONS = 2000  # cap to avoid unbounded growth


def get_real_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def resolve_session_id(
    request: Request,
    body_session_id: str | None,
) -> str:
    # do NOT fall back to IP here
    sid = (
        request.headers.get("x-session-id")
        or request.headers.get("x-gradio-session")
        or body_session_id
    )
    if not sid:
        sid = f"sid-{uuid.uuid4().hex}"
    return sid


def get_session_state(sid: str):
    now = time.time()
    sess = SESSIONS.get(sid)
    # create or refresh if expired
    if not sess or (now - sess.get("started_at", now)) > SESSION_TTL:
        # enforce cap
        if len(SESSIONS) >= MAX_SESSIONS:
            oldest_sid = min(
                SESSIONS.items(),
                key=lambda kv: kv[1].get("started_at", 0)
            )[0]
            SESSIONS.pop(oldest_sid, None)
            log.info(f"[session] evicted sid={oldest_sid} (cap={MAX_SESSIONS})")

        sess = {
            "count": 0,
            "started_at": now,
            "last_seen_at": now,
        }
        SESSIONS[sid] = sess
        log.info(f"[session] created sid={sid}")
    else:
        sess["last_seen_at"] = now
    return sess


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


def ensure_lora_local(lora_hf_id: str, lora_subdir: str, token: str, cache_dir: str):
    repo_root = snapshot_download(
        repo_id=lora_hf_id,
        token=token,
        cache_dir=cache_dir,
        local_dir_use_symlinks=True,
    )
    return os.path.join(repo_root, lora_subdir) if lora_subdir else repo_root


def get_autocast_ctx(pipe):
    dev = getattr(pipe, "device", None)
    if dev is None:
        return torch.autocast("cuda")
    dev_type = getattr(dev, "type", "cuda")
    if dev_type == "cpu":
        return torch.cpu.amp.autocast()
    return torch.autocast(dev_type)


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

    if not has_lora:
        status = "LoRA not found or failed to load."
        if base_flagged:
            status = "Base output flagged."
        return base_img, Image.new("RGB", base_img.size, (30, 30, 30)), status

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

    if base_flagged and lora_flagged:
        status = "Both base and LoRA outputs were flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    elif base_flagged:
        status = "Base output was flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    elif lora_flagged:
        status = "LoRA output was flagged as NSFW by safety checker. Please try with different values of Seed/Guidance/Steps/Eta."
    else:
        status = "LoRA applied successfully."
    return base_img, lora_img, status


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


CFG_PATH = os.getenv("AD_CONFIG_PATH", "configs/infer.yaml")
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = resolve_cache_dir(cfg)

BASE_ID = os.getenv("AD_BASE_ID", "runwayml/stable-diffusion-v1-5")
LORA_ID = os.getenv("AD_LORA_ID", "Srikasi/astro-diffusion")
LORA_SUBDIR = os.getenv("AD_LORA_SUBDIR", "unet_lora_final")

base_pipe = build_base_pipe(BASE_ID, cache_dir, cfg, device=device)
base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)

lora_pipe = build_base_pipe(BASE_ID, cache_dir, cfg, device=device)
lora_pipe.scheduler = DDIMScheduler.from_config(lora_pipe.scheduler.config)

has_lora = False
try:
    lora_path = ensure_lora_local(LORA_ID, LORA_SUBDIR, os.getenv("HF_TOKEN"), cache_dir=cache_dir)
    lconf = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=0.0,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    lora_pipe.unet = get_peft_model(lora_pipe.unet, lconf)
    lora_pipe.unet.load_adapter(lora_path, adapter_name="astro")
    for p in lora_pipe.unet.parameters():
        p.requires_grad_(False)
    if hasattr(lora_pipe.unet, "set_adapter"):
        lora_pipe.unet.set_adapter("astro")
    has_lora = True
except Exception as e:
    log.error(f"[LoRA] attach failed: {e}")
    has_lora = False


class InferRequest(BaseModel):
    prompt: str
    steps: int
    scale: float
    height: int
    width: int
    seed: str
    eta: float = 0.0
    session_id: str | None = None


@app.post("/infer")
async def infer(req: InferRequest, request: Request):
    global INFER_COUNT, LAST_REQUEST_AT

    ip = get_real_ip(request)
    sid = resolve_session_id(request, req.session_id)
    session_state = get_session_state(sid)

    log.info(
        f"[infer] incoming ip={ip} sid={sid} sess_count={session_state.get('count', 0)} "
        f"prompt_len={len(req.prompt)} h={req.height} w={req.width} steps={req.steps}"
    )

    ok, reason = limiter.pre_check(ip, session_state)
    if not ok:
        log.warning(f"[infer] rate-limited ip={ip} sid={sid} reason={reason}")
        return JSONResponse({"error": reason, "session_id": sid}, status_code=429)

    t0 = time.time()
    base_img, lora_img, status = run_both_two_pipes(
        base_pipe,
        lora_pipe,
        req.prompt,
        req.steps,
        req.scale,
        req.height,
        req.width,
        req.seed,
        has_lora,
        cfg,
        req.eta,
    )
    dt = time.time() - t0

    # IP-based accounting
    limiter.post_consume(ip, dt)

    # session-based accounting
    session_state["count"] = session_state.get("count", 0) + 1

    INFER_COUNT += 1
    LAST_REQUEST_AT = time.time()

    log.info(
        f"[infer] done ip={ip} sid={sid} duration={dt:.3f}s status='{status}' "
        f"sess_count={session_state.get('count')} infer_count={INFER_COUNT}"
    )

    return {
        "base_image": pil_to_b64(base_img),
        "lora_image": pil_to_b64(lora_img),
        "status": status,
        "duration": dt,
        "session_id": sid,
    }


@app.middleware("http")
async def count_all_requests(request: Request, call_next):
    global REQUEST_COUNT, LAST_REQUEST_AT
    REQUEST_COUNT += 1
    LAST_REQUEST_AT = time.time()
    response = await call_next(request)
    return response


@app.get("/metrics-lite")
async def metrics_lite():
    return {
        "requests_total": REQUEST_COUNT,
        "infer_total": INFER_COUNT,
        "sessions_total": len(SESSIONS),
        "last_request_at": LAST_REQUEST_AT,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
