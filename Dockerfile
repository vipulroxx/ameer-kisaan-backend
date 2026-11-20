# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /venv && /venv/bin/pip install -r requirements.txt \
 # pip is not needed at runtime; dropping it removes its vendored libraries from the image
 && /venv/bin/pip uninstall -y pip

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/venv/bin:$PATH" PYTHONPATH=/app \
    INDEX_CACHE_DIR=/tmp/index
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
 && python -m pip uninstall -y pip \
 && groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /venv /venv
COPY --chown=app:app app ./app
COPY --chown=app:app knowledge ./knowledge
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app models ./models
USER app
# Writable scratch path for a read-only root filesystem (ECS copies these permissions).
VOLUME ["/tmp"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \
     "--no-server-header"]
