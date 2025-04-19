FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-distutils \
    curl \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python and pip aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to path
ENV PATH="/root/.local/bin:${PATH}"

# Create a working directory
WORKDIR /app

# Create a virtual environment using uv
RUN . /root/.local/bin/env && \
    uv venv /app/myenv --python 3.12 --seed

# Install Python dependencies using uv
RUN . /root/.local/bin/env && \
    . /app/myenv/bin/activate && \
    uv pip install vllm runpod

# Add venv to PATH
ENV PATH="/app/myenv/bin:$PATH"

# Copy application files
COPY rp_handler.py /app/

# Set default command
CMD ["python", "-u", "rp_handler.py"]
