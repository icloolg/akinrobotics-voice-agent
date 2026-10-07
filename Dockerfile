# AKINROBOTICS voice agent server (STT + RAG agent + TTS). The LLM runs in a
# separate Ollama container, see docker-compose.yml.
#
# The same image runs on CPU and on an NVIDIA GPU: Whisper's device is chosen
# at start with STT_DEVICE (cpu | cuda).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of PyTorch, installed first. Only the embedding model and the VAD
# use PyTorch and both run on the CPU; the default wheel would add ~2.5 GB of
# CUDA libraries that are never used.
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# GPU Whisper (CTranslate2) needs cuBLAS and cuDNN; they come from the
# nvidia-* pip packages and only have to be on the library path.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY config.yaml .
COPY app app
COPY scripts scripts
COPY knowledge knowledge

# Bake the models named in config.yaml into the image (Piper voices, Whisper,
# embedding model), so the container starts fast and without internet access.
RUN python -m scripts.download_models --all
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 \
    CMD python -c "import urllib.request, json, sys; sys.exit(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['status'] != 'ok')"

# Index the knowledge folder at every start (a few seconds), then serve.
# Adding a document = put it in knowledge/ and restart the container.
CMD ["sh", "-c", "python -m scripts.ingest && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
