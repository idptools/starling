# ---------- stage 1: build + download weights ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        build-essential gcc g++ gfortran git curl \
    && rm -rf /var/lib/apt/lists/*

# Create a venv so we get a working pip (ensurepip is disabled on system python)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /opt/starling

# Copy project files
# Build with: docker build -f Dockerfile -t starling .. (from the docker/ directory)
COPY pyproject.toml setup.cfg MANIFEST.in hubconf.py README.md ./
COPY starling/ ./starling/

# Install CUDA-enabled PyTorch (automatically falls back to CPU at runtime
# when no GPU is available) then install starling
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir .

# Download model weights at build time so they are baked into the image.
# ModelManager uses torch.hub to cache under ~/.cache/torch/hub/checkpoints/
RUN python -c "from starling.inference.model_loading import ModelManager; ModelManager().load_models(None, None, 'cpu')"

# Download search artifacts (FAISS index, seqstore, manifest) from Zenodo
RUN python -c "from starling.configs import ensure_search_artifacts; ensure_search_artifacts(download=True)"

# ---------- stage 2: slim runtime image ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy cached model weights
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy search artifacts
COPY --from=builder /root/.starling_search /root/.starling_search

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /work

ENTRYPOINT ["starling"]
