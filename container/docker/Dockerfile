# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      libgomp1 \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/phase1-venv
ENV PATH=/opt/phase1-venv/bin:$PATH \
    PYTHONPATH=/opt/high_fidelity_schema_study \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /opt/high_fidelity_schema_study
COPY requirements.txt requirements-phase1-paper-to-schema.txt requirements-phase1-inference.txt ./high_fidelity_schema_study/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
      -r high_fidelity_schema_study/requirements-phase1-paper-to-schema.txt \
      -r high_fidelity_schema_study/requirements-phase1-inference.txt

# Keep the Phase 1 base and dependency layers above byte-identical to the
# existing runtime so they can reuse its cache without inheriting source files.
COPY requirements_four_category_offline.txt /opt/requirements/requirements_four_category_offline.txt
COPY requirements_four_category_structured.txt /opt/requirements/requirements_four_category_structured.txt
RUN pip install --no-cache-dir -r /opt/requirements/requirements_four_category_offline.txt \
    && pip install --no-cache-dir -r /opt/requirements/requirements_four_category_structured.txt \
    && pip check \
    && pip freeze --all > /opt/requirements/pip-freeze.txt \
    && python -c 'import torch, transformers, jsonschema, pdfplumber, pypdfium2, openpyxl, numpy, h5py, scipy, pyarrow, xgrammar; print("runtime imports OK; PDFium", pypdfium2.PDFIUM_INFO)'

# Build context is the public repository and contains only its allowlisted
# package tree; historical research files, datasets, weights, and credentials
# are not copied into the runtime image.
COPY high_fidelity_schema_study /workspace/high_fidelity_schema_study
COPY scheduler_v2 /workspace/scheduler_v2
COPY scripts /workspace/scripts
COPY container/docker/entrypoint.sh /usr/local/bin/schema-study
RUN chmod 0755 /usr/local/bin/schema-study

ARG IMAGE_VERSION=v13-source-unqualified
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="Schema Study Mercury CUDA runtime" \
      org.opencontainers.image.description="Offline-capable runtime for the public schema-study package; contains no model weights or credentials" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/williamQ96/schema_study" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

ENV PYTHONPATH=/workspace \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace/high_fidelity_schema_study
ENTRYPOINT ["/usr/local/bin/schema-study"]
CMD ["--help"]
