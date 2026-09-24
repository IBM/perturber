# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the native perturbation modules.

Centralizes the small pieces that several perturbation modules would otherwise duplicate: the
``rate`` parameter spec, random-subset selection for seed-sensitive transforms, and two
capitalization-matching helpers.
"""

from __future__ import annotations

import random

from ..registry import ParamSpec

_DEFAULT_RATE_DESCRIPTION = "Fraction of eligible positions to transform (0 to 1)."


def rate_param(default: float, description: str = _DEFAULT_RATE_DESCRIPTION) -> ParamSpec:
    """Return the ``rate`` ParamSpec used by seed-sensitive subset transforms."""
    return ParamSpec("rate", "float", default, description)


def subset(indices: list[int], rate: float, rng: random.Random) -> set[int]:
    """Return a random subset of ``indices`` of size about ``rate`` of the total.

    ``rate`` <= 0 selects nothing and ``rate`` >= 1 selects everything; in between, at least one
    index is chosen. Selection is drawn from ``rng`` so it is reproducible for a given seed.
    """
    if rate <= 0 or not indices:
        return set()
    if rate >= 1:
        return set(indices)
    count = max(1, round(len(indices) * rate))
    return set(rng.sample(indices, min(count, len(indices))))


def match_char_case(source: str, replacement: str) -> str:
    """Uppercase a single replacement character when ``source`` is uppercase; else leave it.

    Used for character-level substitutions where the replacement should follow the source's case
    without otherwise altering it.
    """
    return replacement.upper() if source.isupper() else replacement


def match_leading_case(source: str, replacement: str) -> str:
    """Capitalize the first letter of ``replacement`` when ``source`` starts with a capital.

    Used for whole-word substitutions (for example expanding an abbreviation) so the replacement
    keeps the source word's leading capitalization.
    """
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
