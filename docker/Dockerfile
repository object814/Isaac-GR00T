FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Prevent python bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONNOUSERSITE=1

# Arguments
ARG USER_NAME=developer
ARG USER_UID=1000
ARG WANDB_API_KEY=""

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg \
    libegl1 \
    sudo \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python setup
# ---------------------------------------------------------
RUN pip install --upgrade pip setuptools
RUN pip install uv

# Fix constraints inside NVIDIA pytorch container (keep your fixes)
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt
RUN sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt

# ---------------------------------------------------------
# Install ONLY safe lightweight deps (no ML stack)
# ---------------------------------------------------------
RUN pip install \
    imageio \
    h5py \
    boto3 \
    termcolor

# ---------------------------------------------------------
# Install PyTorch3D (from GR00T Dockerfile)
# ---------------------------------------------------------
RUN mkdir -p /opt && \
    cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd /opt/pytorch3d && pip install .

# ---------------------------------------------------------
# Workspace
# ---------------------------------------------------------
RUN mkdir -p /Isaac-GR00T
WORKDIR /Isaac-GR00T

# ---------------------------------------------------------
# NVIDIA EGL / Vulkan configuration
# ---------------------------------------------------------
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF

# ---------------------------------------------------------
# Create non-root user
# ---------------------------------------------------------
RUN addgroup --gid ${USER_UID} ${USER_NAME} && \
    adduser --disabled-password --gecos "" \
        --uid ${USER_UID} \
        --gid ${USER_UID} \
        ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME}

RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${USER_NAME}:${USER_NAME} /Isaac-GR00T

# ---------------------------------------------------------
# Environment variables (kept exactly as yours)
# ---------------------------------------------------------
ENV PYTHONPATH="${PYTHONPATH}:/Isaac-GR00T"
ENV HYDRA_FULL_ERROR=1
ENV TORCH_LOGS="-all"
ENV TORCHDYNAMO_VERBOSE=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

# HuggingFace / Torch cache locations
ENV HF_HOME=/Isaac-GR00T/.cache/hf
ENV HUGGINGFACE_HUB_CACHE=/Isaac-GR00T/.cache/hf
ENV TRANSFORMERS_CACHE=/Isaac-GR00T/.cache/hf
ENV HF_DATASETS_CACHE=/Isaac-GR00T/.cache/hf
ENV TORCH_HOME=/Isaac-GR00T/.cache/torch
ENV TMPDIR=/Isaac-GR00T/.cache/tmp
ENV UV_CACHE_DIR=/Isaac-GR00T/.cache/uv-cache

RUN mkdir -p /Isaac-GR00T/.cache/tmp
RUN mkdir -p /Isaac-GR00T/.cache/hf \
    /Isaac-GR00T/.cache/torch \
    /Isaac-GR00T/.cache/uv-cache && \
    chown -R ${USER_NAME}:${USER_NAME} /Isaac-GR00T/.cache

# Switch to user
USER ${USER_NAME}

# Default shell
CMD ["bash"]