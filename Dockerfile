# syntax=docker/dockerfile:1

# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim AS builder

LABEL description="perturber: semantics-preserving text perturbations"

ENV PYTHONPATH="/app" \
    NLTK_DATA="/app/nltk_data" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY . .

# Install the project and its dependencies from pyproject.toml, which is the single source of
# truth for them. The `llm` extra pulls in the `openai` client the model-backed perturbations use.
# Installing the distribution registers it, so `perturber.api:app` imports as an installed package
# and importlib.metadata.version("perturber") reports the real version.
RUN pip install --no-cache-dir '.[llm]'

# Pre-fetch the corpora the native `synonym` perturbation needs (WordNet + OMW back the
# POS-constrained synonym substitution, and the perceptron tagger disambiguates word sense), so
# `synonym` needs no network at runtime. Modern nltk resolves `pos_tag` to the
# `averaged_perceptron_tagger_eng` package (the unsuffixed name is the legacy one), which is what
# `word_level._pos_tagger` asks for; both are fetched so either nltk version works.
#
# NLTK leaves wordnet/omw as .zip archives that its loader can normally read lazily, but the
# zip-backed WordNet reader is unreliable under a read-only/non-root runtime, so unzip them here and
# check the lookups actually resolve. The verification is deliberately NOT guarded with `|| true`:
# when the prefetch silently fails, `synonym` degrades to returning its input unperturbed, which is
# far worse than a failed build.
RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; [nltk.download(p, download_dir='${NLTK_DATA}') for p in ('wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng')]" && \
    for z in ${NLTK_DATA}/corpora/*.zip; do [ -e "$z" ] && unzip -q -o "$z" -d ${NLTK_DATA}/corpora && rm -f "$z"; done; \
    python -c "import nltk; from nltk.corpus import wordnet; assert wordnet.synsets('quick'), 'wordnet returned no synsets'; assert nltk.pos_tag(['quick','fox']), 'pos_tag returned nothing'; print('nltk corpora verified')" && \
    chmod -R g=u ${NLTK_DATA}

# Runtime stage - final slim image.
FROM python:3.12-slim

LABEL description="perturber: semantics-preserving text perturbations"

ENV PYTHONPATH="/app" \
    NLTK_DATA="/app/nltk_data" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed packages, entry points, and project tree (excludes build tools).
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Make the app tree and prefetched data group-writable so the image runs cleanly under an
# arbitrary non-root UID in group 0 (`g=u` mirrors the owner's permissions onto the group).
RUN chmod -R g=u ${NLTK_DATA} && chmod g=u /app

EXPOSE 8080

# Model-backed perturbations read their credentials per request (Authorization: Bearer) or from
# OPENAI_API_KEY / OPENAI_BASE_URL in the environment; no secret is baked into the image.
CMD ["bash", "scripts/run_prod.sh"]
