FROM python:3.11-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY scripts ./scripts
COPY experiments ./experiments
COPY Makefile ./Makefile

RUN pip install --upgrade pip
RUN pip install -e ".[dev]"

CMD ["bash"]
