# PyTorch 2.8.0 built for CUDA 12.1 + cuDNN 9
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
RUN pip install --upgrade torch==2.8.0 torchvision==0.23.0

# Basic environment hygiene
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_CACHE=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

WORKDIR /app

# Minimal system packages required by Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Gradio UI port
EXPOSE 7860

# Default startup command
CMD ["bash","-lc","python src/ui_gradio.py \
  --config configs/infer.yaml \
  --base-id runwayml/stable-diffusion-v1-5 \
  --lora-hf-id Srikasi/astro-diffusion \
  --lora-subdir unet_lora_final"]

