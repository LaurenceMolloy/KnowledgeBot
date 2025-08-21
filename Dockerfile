FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed by some tokenizers like tiktoken)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY KnowledgeBot.py  .
COPY pytest.ini       .
COPY LLMService       ./LLMService
COPY VectorDatabase   ./VectorDatabase
COPY tests            ./tests

# Allows script to be container aware (skips reading the .env file)
ENV IN_DOCKER=1

VOLUME ["/app/data"]

CMD ["python", "./KnowledgeBot.py"]
#CMD ["pytest", "-m", "slack_preflight"]
