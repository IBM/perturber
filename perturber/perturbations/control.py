"""Control perturbations (native tier).

The control family holds the identity transform, which is useful as an explicit baseline
in robustness studies.
"""

from __future__ import annotations

import random

from ..registry import CATEGORY_NATIVE, FAMILY_CONTROL, Perturbation, Registry


def _baseline(text: str, params: dict, rng: random.Random) -> str:
    """Return the text unchanged."""
    return text


BASELINE = Perturbation(
    name="baseline",
    category=CATEGORY_NATIVE,
    family=FAMILY_CONTROL,
    description="Identity transform; returns the text unchanged. The explicit control.",
    apply=_baseline,
    params=(),
    seed_sensitive=False,
)


def register(registry: Registry) -> None:
    registry.register(BASELINE)
