"""perturber: semantics-preserving text perturbations as a library, CLI, and HTTP service.

The public entry point is :func:`perturb`, which applies exactly one named perturbation to
one string. Compose several perturbations by chaining calls. See :func:`list_perturbations`
for the available perturbations.

All perturbations are permissively licensed, first-party implementations. Most are
deterministic string transforms with no model dependency; the model-backed ones call a generic
OpenAI-compatible ``/chat/completions`` endpoint configured by the caller.
"""

from __future__ import annotations

from .core import DEFAULT_SEED, list_perturbations, perturb
from .registry import REGISTRY
from .perturbations import register_native

# Populate the process-wide registry on import with the native perturbations.
register_native(REGISTRY)

__all__ = ["perturb", "list_perturbations", "DEFAULT_SEED", "REGISTRY"]
