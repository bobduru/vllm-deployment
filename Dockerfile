FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3.10-dev \
    curl \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to path
ENV PATH="/root/.local/bin:${PATH}"

# Create working directory
WORKDIR /app

# Create a virtual environment using uv
RUN uv venv /app/myenv --python 3.10 --seed

# Install Python dependencies inside the venv using system uv
RUN uv pip install --python /app/myenv/bin/python vllm runpod

# Add venv to PATH for runtime use
ENV PATH="/app/myenv/bin:$PATH"

# Copy your application
COPY rp_handler.py /app/

# Default command
CMD ["python", "-u", "rp_handler.py"]
