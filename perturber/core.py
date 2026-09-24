# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

"""The single perturbation primitive.

``perturb`` applies exactly one named perturbation to one string. Composition of several
perturbations is performed by the caller chaining calls (feeding each result into the next
call); the service has no stack grammar.

Determinism: each call reseeds the random sources to the request ``seed`` immediately
before applying the perturbation, so the result is a pure function of
``(text, name, params, seed)``. Because some perturbations draw from the global ``random``
and ``numpy`` generators, the seed-and-apply section is guarded by a lock to keep results
deterministic even when the API serves requests concurrently.
"""

from __future__ import annotations

import inspect
import random
import threading
from typing import Optional

from .registry import REGISTRY, Perturbation

try:  # numpy is optional; only some backing engines use it.
    import numpy as _np
except Exception:  # pragma: no cover - exercised only when numpy is absent
    _np = None

# Serialises the seed-and-apply section so that perturbations relying on the global RNGs
# remain deterministic under concurrent requests.
_SEED_LOCK = threading.Lock()

# numpy's legacy seed accepts values in [0, 2**32).
_NUMPY_SEED_MODULUS = 2**32

DEFAULT_SEED = 0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed % _NUMPY_SEED_MODULUS)


def perturb(
    text: str,
    name: str,
    params: Optional[dict] = None,
    seed: int = DEFAULT_SEED,
    category: Optional[str] = None,
    *,
    context: Optional[dict] = None,
) -> str:
    """Apply the single perturbation ``name`` to ``text`` and return the result.

    Parameters
    ----------
    text:
        The input string.
    name:
        The registered perturbation name (for example ``"typos"`` or ``"baseline"``).
    params:
        Optional perturbation parameters; missing values fall back to the perturbation's
        declared defaults. Unknown parameter names raise ``ValueError``.
    seed:
        Seed for any random choice the perturbation makes. Defaults to ``0`` so the service
        is deterministic by default. Randomness never derives from wall-clock or process
        state.
    category:
        Optional provenance category (``"native"``). A name is unique only within its
        category, so ``category`` disambiguates when the same name is registered in more than
        one. When omitted, a name that is unique across the registered categories still
        resolves; an ambiguous one raises ``KeyError``.
    context:
        Optional transport-only inputs for perturbations that need them (for example
        ``{"api_key": ...}`` for model-backed perturbations). Passed through to the
        perturbation's ``apply`` when it accepts a ``context`` keyword; it is never part of the
        deterministic ``(text, name, params, seed)`` identity and is never returned.

    Raises
    ------
    KeyError
        If ``name`` is not a registered perturbation, or is ambiguous without ``category``.
    ValueError
        If ``params`` contains an unknown parameter name.
    """
    perturbation: Perturbation = REGISTRY.get(name, category)
    resolved = perturbation.resolve_params(params)
    rng = random.Random(seed)
    # Pass ``context`` only to perturbations whose apply accepts it, keeping the plain
    # ``(text, params, rng)`` contract intact for every existing perturbation.
    pass_context = "context" in inspect.signature(perturbation.apply).parameters
    with _SEED_LOCK:
        _seed_all(seed)
        if pass_context:
            return perturbation.apply(text, resolved, rng, context=context)
        return perturbation.apply(text, resolved, rng)


def list_perturbations(category: Optional[str] = None) -> list[dict]:
    """Return metadata for the registered perturbations, optionally filtered by category."""
    return [
        p.describe()
        for p in REGISTRY
        if category is None or p.category == category
    ]
