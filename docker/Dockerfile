FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Prevent python bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Arguments passed from docker-compose /.env
ARG USER_NAME=developer
ARG USER_UID=1000
ARG WANDB_API_KEY=""

# ---------------------------------------------------------
# System dependencies (from GR00T Dockerfile)
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

# Fix constraints inside NVIDIA pytorch container
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt
RUN sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt

# Install python dependencies (from GR00T Dockerfile)
RUN pip install \
    imageio \
    h5py \
    boto3 \
    transformers[torch] \
    deepspeed \
    timm \
    peft \
    diffusers \
    wandb \
    tianshou \
    dm_tree \
    openai \
    albumentations==1.4.18 \
    decord==0.6.0 \
    torchcodec==0.4.0

# ---------------------------------------------------------
# Install PyTorch3D (from GR00T Dockerfile)
# ---------------------------------------------------------
RUN mkdir -p /opt
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd /opt/pytorch3d && pip install .

# ---------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------
RUN mkdir -p /Isaac-GR00T
WORKDIR /Isaac-GR00T

# ---------------------------------------------------------
# NVIDIA EGL / Vulkan configuration (from GR00T Dockerfile)
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

# Allow passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Fix permissions for workspace
RUN chown -R ${USER_NAME}:${USER_NAME} /Isaac-GR00T

# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------
ENV PYTHONPATH="${PYTHONPATH}:/Isaac-GR00T"
ENV HYDRA_FULL_ERROR=1
ENV TORCH_LOGS="-all"
ENV TORCHDYNAMO_VERBOSE=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Switch to user
USER ${USER_NAME}

# Default shell
CMD ["bash"]