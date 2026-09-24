# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application.

Every registered perturbation gets its own explicit, typed endpoint, generated from the
registry. Endpoints are grouped into a router by provenance category (``native``) and each is
tagged by family (``word_level``, ``padding``, ``context``, ``control``) so the OpenAPI/Swagger
docs group by family as well.

Note: this module intentionally does not use ``from __future__ import annotations`` because
route handlers are given dynamically created Pydantic request models as real annotations,
which FastAPI must resolve at registration time.
"""

import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, create_model

from .core import DEFAULT_SEED, perturb
from .registry import _PY_TYPES, REGISTRY, Perturbation

logger = logging.getLogger("perturber.api")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class PerturbResponse(BaseModel):
    texts: List[str]
    category: str
    name: str
    family: str
    params: dict
    seed: int
    # Wall-clock time to perturb each input string, in milliseconds, parallel to ``texts``.
    elapsed_ms: List[float]


def _param_fields(perturbation: Perturbation) -> dict:
    """Pydantic field definitions for a perturbation's own parameters."""
    fields = {}
    for spec in perturbation.params:
        py_type = _PY_TYPES.get(spec.type, str)
        fields[spec.name] = (py_type, spec.default)
    return fields


def _request_model(perturbation: Perturbation):
    # Qualify the model name with the category: the same perturbation name can exist in
    # more than one category, and a duplicate model __name__ would collide in the generated
    # OpenAPI schema.
    prefix = perturbation.category.title().replace("_", "")
    return create_model(
        f"{prefix}{perturbation.name.title().replace('_', '')}Request",
        texts=(List[str], ...),
        seed=(int, DEFAULT_SEED),
        **_param_fields(perturbation),
    )


def _split_payload(perturbation: Perturbation, payload: dict):
    """Separate the transport fields (seed) from the perturbation parameters."""
    seed = payload.pop("seed", DEFAULT_SEED)
    params = {spec.name: payload.get(spec.name) for spec in perturbation.params}
    return seed, params


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract the token from an ``Authorization: Bearer <token>`` header, if present."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    # Tolerate a bare token with no scheme.
    return authorization.strip() or None


def _make_handler(perturbation: Perturbation, Model):
    # The Authorization: Bearer header (or the X-LLM-API-Key alias) carries a per-request secret
    # for model-backed perturbations, and X-LLM-Base-URL optionally overrides the endpoint. Both
    # are threaded as transport-only context and never enter params or the response.
    #
    # This handler is deliberately a plain ``def`` (not ``async def``). ``perturb`` runs
    # synchronously, and the LLM-backed perturbations make blocking network calls that can take
    # seconds. FastAPI runs a sync handler in a worker thread, so that blocking work stays off the
    # event loop; an ``async def`` here would freeze the loop for the duration of the call,
    # starving the ``/health`` liveness probe and getting the pod restarted mid-request.
    def handler(
        body: Model,  # noqa: N803 - Model is a dynamic type
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_llm_api_key: Optional[str] = Header(default=None, alias="X-LLM-API-Key"),
        x_llm_base_url: Optional[str] = Header(default=None, alias="X-LLM-Base-URL"),
    ):
        payload = body.model_dump()
        texts = payload.pop("texts")
        seed, params = _split_payload(perturbation, payload)
        api_key = _bearer_token(authorization) or x_llm_api_key
        context = None
        if api_key or x_llm_base_url:
            context = {"api_key": api_key, "base_url": x_llm_base_url}
        results = []
        elapsed_ms = []
        try:
            for text in texts:
                start = time.perf_counter()
                results.append(
                    perturb(
                        text,
                        perturbation.name,
                        params=params,
                        seed=seed,
                        category=perturbation.category,
                        context=context,
                    )
                )
                elapsed_ms.append(round((time.perf_counter() - start) * 1000, 3))
        except ValueError as exc:
            # Missing key, unknown param, or an upstream LLM failure, i.e. a client-facing error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - any other error is an internal fault.
            # Log the detail server-side; return a generic 500 rather than leaking a traceback.
            logger.exception("Perturbation '%s/%s' failed", perturbation.category, perturbation.name)
            raise HTTPException(status_code=500, detail="Internal error applying the perturbation.") from exc
        return PerturbResponse(
            texts=results,
            category=perturbation.category,
            name=perturbation.name,
            family=perturbation.family,
            params=params,
            seed=seed,
            elapsed_ms=elapsed_ms,
        )

    return handler


def _register_perturbation_routes(router: APIRouter, perturbation: Perturbation) -> None:
    model = _request_model(perturbation)
    router.add_api_route(
        f"/{perturbation.name}",
        _make_handler(perturbation, model),
        methods=["POST"],
        response_model=PerturbResponse,
        tags=[perturbation.family],
        summary=perturbation.description,
        name=f"perturb_{perturbation.category}_{perturbation.name}",
    )


def _grouped_catalog(category: str | None = None) -> dict:
    grouped = REGISTRY.by_category(category)
    return {cat: [p.describe() for p in perts] for cat, perts in grouped.items()}


def create_app() -> FastAPI:
    app = FastAPI(
        title="perturber",
        description=(
            "Applies semantics-preserving perturbations to text. Each perturbation has its "
            "own explicit, typed endpoint, grouped by provenance category (native) and tagged "
            "by family. Apply one perturbation per call; compose by chaining calls."
        ),
        version=_version(),
    )

    # One router per category that currently has registered perturbations.
    for category in REGISTRY.categories():
        router = APIRouter(prefix=f"/perturb/{category}")
        for perturbation in REGISTRY.by_category(category)[category]:
            _register_perturbation_routes(router, perturbation)
        app.include_router(router)

    @app.get("/perturbations", tags=["catalog"])
    async def list_all_perturbations() -> dict:
        return _grouped_catalog()

    @app.get("/perturbations/{category}", tags=["catalog"])
    async def list_category_perturbations(category: str) -> dict:
        catalog = _grouped_catalog(category)
        if not catalog:
            raise HTTPException(status_code=404, detail=f"Unknown or empty category '{category}'")
        return catalog

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/version", tags=["health"])
    async def version() -> dict:
        """Return the deployed service version, so a client (e.g. the playground) can show it."""
        return {"version": app.version}

    # Interactive playground. The page is data-driven off GET /perturbations, so it stays correct
    # as perturbations are added or removed. Served only when the static asset is present, so a
    # trimmed build without it still exposes the full API.
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def landing() -> FileResponse:
            return FileResponse(index_path)

    return app


def _version() -> str:
    """Resolve the service version.

    Prefer the ``VERSION`` file shipped next to the package: it is the single source of truth the
    build and deploy script bump, so it always matches what was actually deployed (installed
    distribution metadata can be stale in an editable dev install). Fall back to that metadata, and
    only then to a placeholder.
    """
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION")
    try:
        with open(version_file, encoding="utf-8") as handle:
            file_version = handle.read().strip()
        if file_version:
            return file_version
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return version("perturber")
    except Exception:  # pragma: no cover - no VERSION file and not installed as a distribution
        return "0.0.0"


app = create_app()
