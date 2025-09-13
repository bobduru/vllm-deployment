FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/hf_cache

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
ENV PATH="/root/.local/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/


#     # Create a virtual environment using uv
RUN uv venv /app/myenv --python 3.10 --seed

# # Install Python dependencies inside the venv using system uv
# RUN uv pip install --python /app/myenv/bin/python -r /app/requirements.txt

RUN uv pip install --python /app/myenv/bin/python vllm==0.10.1.1 runpod==1.7.13 pandas==2.3.2 dotenv
# RUN uv pip install --python /app/myenv/bin/python vllm==0.10.1.1 runpod==1.7.13 pandas==2.3.2 dotenv==0.9.9



# Add venv to PATH
ENV PATH="/app/myenv/bin:$PATH"

# Copy application files
COPY rp_handler.py /app/
COPY keywords.csv /app/

CMD ["python", "-u", "rp_handler.py"]
