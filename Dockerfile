FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml requirements.txt README_langchain_agent.md langchain_autosolver_agent.py ./
COPY autosolver_agent ./autosolver_agent
COPY solvers ./solvers
COPY examples ./examples
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN useradd --create-home --uid 1000 autosolver \
    && mkdir -p /app/runs \
    && chown -R autosolver:autosolver /app

USER autosolver

ENTRYPOINT ["python", "/app/langchain_autosolver_agent.py"]
CMD ["--help"]
