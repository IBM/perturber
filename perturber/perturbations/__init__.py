"""Native tier perturbations (commercial-safe, permissively licensed).

This package holds clean-room perturbation implementations plus the ``baseline`` control.
The word-level and padding families are clean-room, standard-library-only implementations that
add no dependencies.

Every perturbation here is a permissively licensed, first-party implementation.
"""

from __future__ import annotations

from ..registry import Registry
from . import (
    adversarial,
    character,
    control,
    formatting,
    llm,
    normalization,
    padding,
    schema,
    word_level,
)


def register_native(registry: Registry) -> None:
    """Register every native-tier perturbation into ``registry``."""
    control.register(registry)
    word_level.register(registry)
    padding.register(registry)
    character.register(registry)
    formatting.register(registry)
    normalization.register(registry)
    schema.register(registry)
    adversarial.register(registry)
    llm.register(registry)
