FROM python:3.10-slim AS runtime

ARG VERSION=1.0.0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="AutoSolver Agent" \
      org.opencontainers.image.description="LangGraph/LangChain delivery assignment solver generation agent" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="local" \
      org.opencontainers.image.licenses="UNLICENSED"

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
    && python -m pip install . \
    && autosolver-agent --version

RUN useradd --create-home --uid 1000 autosolver \
    && mkdir -p /app/runs \
    && chown -R autosolver:autosolver /app

USER autosolver

ENTRYPOINT ["autosolver-agent"]
CMD ["--help"]
