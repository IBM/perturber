"""Padding perturbations (native tier).

Clean-room, standard-library-only implementations of the padding family. Each perturbation
prepends and appends ``n`` copies of a fixed character to the text, wrapping it as
``<pad> <text> <pad>``: that is, the padding, then a single literal space, then the text, then
a single literal space, then the padding. These transforms are deterministic and ignore the seed.
"""

from __future__ import annotations

import random

from ..registry import (
    CATEGORY_NATIVE,
    FAMILY_PADDING,
    ParamSpec,
    Perturbation,
    Registry,
)

_QUOTE = '"'
_SPACE = " "
_NEW_LINE = "\n"


def _pad(text: str, char: str, n: int) -> str:
    """Wrap ``text`` with ``n`` copies of ``char`` on each side: ``<pad> <text> <pad>``."""
    pad = char * n
    return f"{pad} {text} {pad}"


def _make_padding_apply(char: str):
    def apply(text: str, params: dict, rng: random.Random) -> str:
        return _pad(text, char, params["n"])

    return apply


def _padding_perturbation(name: str, char: str, char_label: str) -> Perturbation:
    return Perturbation(
        name=name,
        category=CATEGORY_NATIVE,
        family=FAMILY_PADDING,
        description=f"Prepend and append N {char_label} characters.",
        apply=_make_padding_apply(char),
        params=(
            ParamSpec("n", "int", 5, f"Number of {char_label} characters to pad with on each side."),
        ),
        seed_sensitive=False,
    )


_QUOTES = _padding_perturbation("quotes", _QUOTE, "quote")
_SPACES = _padding_perturbation("spaces", _SPACE, "space")
_NEW_LINES = _padding_perturbation("new_lines", _NEW_LINE, "newline")

_PERTURBATIONS = (_QUOTES, _SPACES, _NEW_LINES)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
