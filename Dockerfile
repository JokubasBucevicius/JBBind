# JBBind — per-residue protein binding-site prediction.
#
# Two build targets from one file:
#   CPU  (default)  docker build -t jbbind:cpu .
#   CUDA            docker build -t jbbind:cuda \
#                     --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#                     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .
# The application code is identical and device-agnostic; only the torch wheel differs.
#
# Layer order is deliberate: the 2.6 GB ESM-2 checkpoint sits BELOW the application code,
# so editing jbbind/ rebuilds and re-pushes megabytes, not gigabytes.

# ---------------------------------------------------------------- voronota builder
FROM debian:bookworm-slim AS voronota

ARG VORONOTA_VERSION=1.29.4781
ARG VORONOTA_SHA256=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Built from source rather than vendoring a 16 MB opaque binary into git. C++14, all
# dependencies vendored in expansion_js/src/dependencies, no external libs.
# Note: expansion_js/src/duktaper/stocked_data_resources.h is a ~31 MB header of embedded
# VoroMQA data, so that single translation unit takes several minutes and a few GB of RAM.
RUN curl -fsSL -o voronota.tar.gz \
      "https://github.com/kliment-olechnovic/voronota/releases/download/v${VORONOTA_VERSION}/voronota_${VORONOTA_VERSION}.tar.gz" \
    && if [ -n "$VORONOTA_SHA256" ]; then echo "${VORONOTA_SHA256}  voronota.tar.gz" | sha256sum -c -; fi \
    && tar xf voronota.tar.gz && mv voronota_${VORONOTA_VERSION} src

RUN cd src/expansion_js && cmake . && make -j"$(nproc)" \
    && test -x ./voronota-js

# ---------------------------------------------------------------- runtime
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE} AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG ESM_LOCAL=""

# bash/coreutils/awk/sed/grep are required: the voronota wrapper scripts are bash and
# call mktemp, awk and sed. tini reaps the subprocesses voronota spawns. curl fetches the
# ESM weights below.
#
# The CUDA base images ship no Python, so install one when the base does not provide it
# (python:3.13-slim already has it and this branch is a no-op there).
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash coreutils gawk sed grep ca-certificates curl tini \
    && if ! command -v python >/dev/null 2>&1; then \
         apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
         && ln -sf /usr/bin/python3 /usr/local/bin/python \
         && ln -sf /usr/bin/pip3 /usr/local/bin/pip; \
       fi \
    && rm -rf /var/lib/apt/lists/* \
    && python --version && pip --version

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    JBBIND_MODELS=/app/models \
    JBBIND_CACHE=/data/cache \
    # Unbounded intra-op threads is a real pathology for the small GNN forwards this
    # service runs; it oversubscribes every core for no gain.
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

COPY requirements.txt .
# --break-system-packages is a no-op on python:3.13-slim; it is needed only on the
# Debian/Ubuntu-based CUDA images, whose system Python is PEP 668 "externally managed".
RUN pip install --no-cache-dir --break-system-packages \
        --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        -r requirements.txt \
 || pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
        -r requirements.txt

COPY --from=voronota /build/src/expansion_js/voronota-js /usr/local/bin/voronota-js
RUN chmod +x /usr/local/bin/voronota-js && voronota-js --help >/dev/null 2>&1 || true

# ESM-2 650M in its own layer, below the app code. ~2.6 GB.
# Offline builds: pass --build-arg ESM_LOCAL=weights (a directory in the build context
# holding the two .pt files) to copy instead of download.
COPY ${ESM_LOCAL:-requirements.txt} /tmp/esm-local/
RUN mkdir -p "${TORCH_HOME}/hub/checkpoints" \
    && if [ -f /tmp/esm-local/esm2_t33_650M_UR50D.pt ]; then \
         cp /tmp/esm-local/esm2_t33_650M_UR50D*.pt "${TORCH_HOME}/hub/checkpoints/"; \
       else \
         curl -fsSL -o "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D.pt" \
           https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt && \
         curl -fsSL -o "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt" \
           https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt; \
       fi \
    && rm -rf /tmp/esm-local \
    && python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D()" >/dev/null

# Changes most often -> last.
COPY tools/ /app/tools/
COPY models/ /app/models/
COPY jbbind/ /app/jbbind/
COPY tests/ /app/tests/
COPY pytest.ini /app/
RUN chmod +x /app/tools/describe-receptor-chain

# A real `jbbind` executable, so `docker run <img> jbbind ...` works alongside the
# default uvicorn CMD.
RUN printf '#!/bin/sh\ncd /app && exec python -m jbbind.cli "$@"\n' > /usr/local/bin/jbbind \
    && chmod +x /usr/local/bin/jbbind

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin jbbind \
    && mkdir -p /data/cache && chown -R 10001:10001 /data /app
USER 10001
VOLUME ["/data"]
EXPOSE 8000

# ESM-2 takes 30-60 s to load, so /readyz must not be polled before then.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=8).status==200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
# One worker: each holds its own 2.6 GB copy of ESM-2. Scale with the job queue, not
# with processes.
CMD ["uvicorn", "jbbind.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
