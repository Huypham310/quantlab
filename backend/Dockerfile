# One image for the API, the Celery worker and the stream consumer: compose picks the command.
#
#   docker build -t quantlab-backend backend
#   docker build -t quantlab-backend:ml --build-arg EXTRAS=ml backend    # + LightGBM / torch for train jobs
#
# Build context is ./backend (see .dockerignore).

# ---- build stage: wheels get built here so the final image has no compiler -----------------
FROM python:3.12-slim AS build

ARG EXTRAS=""
# torch from the default index is the CUDA build (~2 GB), the CPU wheel is enough for the LSTM
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src

# copy the metadata first so the dependency layer is cached until pyproject.toml changes
COPY pyproject.toml ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# plain (non-editable) install: the package is copied into site-packages
RUN if [ -n "$EXTRAS" ]; then \
      pip install ".[${EXTRAS}]" --extra-index-url "$TORCH_INDEX"; \
    else \
      pip install .; \
    fi

# ---- runtime stage ----------------------------------------------------------------------------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/numba

# libgomp1: OpenMP runtime that LightGBM / torch want when built with EXTRAS=ml
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# không chạy bằng root trong container / never run as root inside the container
RUN useradd --create-home --uid 10001 quantlab
COPY --from=build /opt/venv /opt/venv

USER quantlab
WORKDIR /app
# reports/ is where `python -m quantlab.model.run` writes its json files
RUN mkdir -p /app/reports

EXPOSE 8000
# default = API; compose overrides `command` for the worker and the consumer
CMD ["uvicorn", "quantlab.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
