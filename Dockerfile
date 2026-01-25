# FROM mayavb/unet:local_user_V3
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG USERNAME=mayavb
ARG UID=30772
ARG GID=2102

ENV PATH="/home/$USERNAME/.local/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Ensure we are root to install system dependencies
USER root

# Update system and install required packages
RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    x11-apps \
    libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Create user and groups (safely)
RUN groupadd --gid 2102 dsi || true && \
    groupadd --gid 4960 docker || true && \
    groupadd --gid 6648 ug_dsi || true && \
    groupadd --gid 6653 ug_gannot || true && \
    groupadd --gid 6656 ug_dgx || true && \
    groupadd --gid 41052 ug_hpc || true && \
    groupadd --gid 668400055 ug_audience || true && \
    id -u mayavb >/dev/null 2>&1 || useradd --uid 30772 --gid 2102 --groups dsi,docker,ug_dsi,ug_gannot,ug_dgx,ug_hpc,ug_audience \
    --create-home --shell /bin/bash mayavb && \
    echo "mayavb ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/mayavb && \
    chmod 0440 /etc/sudoers.d/mayavb

# Set environment variables for display and NVIDIA GPU
ENV DISPLAY=:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Ensure /dev/shm exists for shared memory usage
RUN mkdir -p /dev/shm && chmod 1777 /dev/shm

# Change ownership of home directory
RUN chown -R $USERNAME:$GID /home/$USERNAME

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --pre \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu128

# Python deps for SRP-DNN (root)
COPY requirements-srpdnn.txt /tmp/requirements-srpdnn.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements-srpdnn.txt

# Build deps for gpuRIR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    git \
    build-essential \
    cmake \
  && rm -rf /var/lib/apt/lists/*

# Install gpuRIR from source
RUN git clone --depth 1 https://github.com/DavidDiazGuerra/gpuRIR.git /tmp/gpuRIR && \
    python3 -m pip install --no-cache-dir /tmp/gpuRIR && \
    rm -rf /tmp/gpuRIR

# Switch to non-root user
USER $USERNAME

# Default command to start MATLAB inside /src
# CMD ["matlab", "-nojvm", "-nodisplay", "-nosplash", "-r", "cd /src;"]
