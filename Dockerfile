FROM python:3.12-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set environment variables for uv
ENV PATH="/root/.local/bin:${PATH}"

# Create a working directory
WORKDIR /app

# Create a Python virtual environment using uv
RUN . /root/.local/bin/env && \
    uv venv /app/myenv --python 3.12 --seed

# Install required packages using uv
RUN . /root/.local/bin/env && \
    . /app/myenv/bin/activate && \
    uv pip install vllm runpod

# Add venv to PATH
ENV PATH="/app/myenv/bin:$PATH"

# Copy your handler
COPY rp_handler.py /app/

# Set default command
CMD ["python", "-u", "rp_handler.py"]
