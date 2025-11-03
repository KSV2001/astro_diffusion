.PHONY: setup train ui eval docker-build docker-run

setup:
\tpip install --upgrade pip && pip install -r requirements.txt

train:
\taccelerate launch src/training.py --config configs/train_sd15.yaml

ui:
\tpython src/ui_gradio.py --config configs/infer.yaml

eval:
\tpython -c "print('Add your eval runner or call astro eval functions from a script')"

docker-build:
\tdocker build -t astro-diffusion:latest .

# mount local models/cache/data for speed; pass CUDA into container
docker-run:
\tdocker run --gpus all --rm -it \\
\t  -p 7860:7860 \\
\t  -v $$(pwd)/hf_cache:/app/hf_cache \\
\t  -v $$(pwd)/outputs:/app/outputs \\
\t  -v $$(pwd)/data:/app/data \\
\t  --name astro-ui astro-diffusion:latest
