FROM python:3.11-slim

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

# Allows script to be container aware (skips reading the .env file)
ENV IN_DOCKER=1

# Security optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1000 knowledgebot
USER knowledgebot

# make sure the user-local bin dir is visible
ENV PATH=/home/knowledgebot/.local/bin:$PATH

# Create application directory and et secure permissions
WORKDIR /app
RUN chmod 700 /app && chown knowledgebot:knowledgebot /app

VOLUME ["/app/data"]

COPY --chown=knowledgebot:knowledgebot requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=knowledgebot:knowledgebot test_resolver.py  .
COPY --chown=knowledgebot:knowledgebot KnowledgeBot.py  .
COPY --chown=knowledgebot:knowledgebot pytest.ini       .
COPY --chown=knowledgebot:knowledgebot Config           ./Config
COPY --chown=knowledgebot:knowledgebot LLMService       ./LLMService
COPY --chown=knowledgebot:knowledgebot VectorDatabase   ./VectorDatabase
COPY --chown=knowledgebot:knowledgebot tests            ./tests


CMD ["python", "./KnowledgeBot.py"]
#CMD ["pytest", "-m", "ollama_preflight"]
