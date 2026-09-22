"""Character / orthographic perturbations (native tier).

Clean-room, standard-library-only transforms that operate on characters: homoglyph and
full-width substitution, leetspeak, case changes, diacritics, and zero-width insertion. These
are common robustness probes: several deliberately produce output that reads the same to a human
but is not byte-identical ASCII (homoglyphs, zero-width, full-width), which is the point.

Seed-sensitive ones take a ``rate`` (fraction of eligible positions to transform) and draw from
the per-call ``rng``, mirroring the other native perturbations. Whole-text transforms
(``upper_case``, ``title_case``, ``diacritics_strip``, ``full_width``) are deterministic.
"""

from __future__ import annotations

import random
import unicodedata

from ..registry import CATEGORY_NATIVE, FAMILY_CHARACTER, Perturbation, Registry
from ._common import match_char_case, rate_param, subset

# --- clean-room data (authored here) ---

# Latin letter mapped to a visually similar Unicode confusable (Cyrillic or Greek). Lowercase; case preserved on use.
_HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р",
    "s": "ѕ", "x": "х", "y": "у", "h": "һ", "k": "κ", "n": "ո", "b": "Ƅ",
}

# Leetspeak character map.
_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "l": "1", "b": "8"}

# A combining acute accent applied to vowels for diacritics_add.
_COMBINING_ACUTE = "́"
_VOWELS = "aeiouAEIOU"

# Zero-width space.
_ZERO_WIDTH = "​"

# ASCII printable range mapped to the full-width offset (U+FF01..U+FF5E maps ! .. ~).
_FULLWIDTH_OFFSET = 0xFF01 - 0x21


def _homoglyph(text: str, params: dict, rng: random.Random) -> str:
    """Replace a random subset of letters with visually similar Unicode confusables."""
    positions = [i for i, c in enumerate(text) if c.lower() in _HOMOGLYPHS]
    chosen = subset(positions, params["rate"], rng)
    return "".join(
        (match_char_case(c, _HOMOGLYPHS[c.lower()]) if i in chosen else c)
        for i, c in enumerate(text)
    )


def _leetspeak(text: str, params: dict, rng: random.Random) -> str:
    """Replace a random subset of leet-mappable letters with digits/symbols."""
    positions = [i for i, c in enumerate(text) if c.lower() in _LEET]
    chosen = subset(positions, params["rate"], rng)
    return "".join(
        (_LEET[c.lower()] if i in chosen else c) for i, c in enumerate(text)
    )


def _random_case(text: str, params: dict, rng: random.Random) -> str:
    """Flip the case of a random subset of letters."""
    positions = [i for i, c in enumerate(text) if c.isalpha()]
    chosen = subset(positions, params["rate"], rng)
    return "".join(
        ((c.lower() if c.isupper() else c.upper()) if i in chosen else c)
        for i, c in enumerate(text)
    )


def _upper_case(text: str, params: dict, rng: random.Random) -> str:
    """Uppercase the whole text."""
    return text.upper()


def _title_case(text: str, params: dict, rng: random.Random) -> str:
    """Title-case the whole text."""
    return text.title()


def _diacritics_add(text: str, params: dict, rng: random.Random) -> str:
    """Add a combining acute accent to a random subset of vowels."""
    positions = [i for i, c in enumerate(text) if c in _VOWELS]
    chosen = subset(positions, params["rate"], rng)
    out = []
    for i, c in enumerate(text):
        out.append(c + _COMBINING_ACUTE if i in chosen else c)
    return "".join(out)


def _diacritics_strip(text: str, params: dict, rng: random.Random) -> str:
    """Remove all diacritics: decompose, drop combining marks, then recompose."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


def _zero_width(text: str, params: dict, rng: random.Random) -> str:
    """Insert a zero-width space after a random subset of interior character positions."""
    positions = list(range(len(text) - 1))
    chosen = subset(positions, params["rate"], rng)
    out = []
    for i, c in enumerate(text):
        out.append(c)
        if i in chosen:
            out.append(_ZERO_WIDTH)
    return "".join(out)


def _full_width(text: str, params: dict, rng: random.Random) -> str:
    """Map printable ASCII to its full-width Unicode forms; space becomes the ideographic space."""
    out = []
    for c in text:
        code = ord(c)
        if 0x21 <= code <= 0x7E:
            out.append(chr(code + _FULLWIDTH_OFFSET))
        elif c == " ":
            out.append("　")  # ideographic space
        else:
            out.append(c)
    return "".join(out)


def _disemvowel(text: str, params: dict, rng: random.Random) -> str:
    """Drop a random subset of vowels (SMS style: hello becomes hll)."""
    positions = [i for i, c in enumerate(text) if c in _VOWELS]
    chosen = subset(positions, params["rate"], rng)
    return "".join(c for i, c in enumerate(text) if i not in chosen)


# Right-to-left override / pop-directional-formatting: wraps the text so it displays reversed
# in many renderers; a classic bidi spoofing / robustness probe.
_RLO = "‮"
_PDF = "‬"


def _bidi_override(text: str, params: dict, rng: random.Random) -> str:
    """Wrap the text in a Unicode right-to-left override. Whole-text, deterministic."""
    return f"{_RLO}{text}{_PDF}"


# Ordered from the most everyday transforms (case), through playful letter substitutions, to the
# Unicode-confusable and invisible-character ones that diverge most from plain ASCII.
_PERTURBATIONS = (
    Perturbation("random_case", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Flip the case of a random subset of letters.",
                 _random_case, params=(rate_param(0.5),), seed_sensitive=True),
    Perturbation("upper_case", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Uppercase the whole text.", _upper_case, params=(), seed_sensitive=False),
    Perturbation("title_case", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Title-case the whole text.", _title_case, params=(), seed_sensitive=False),
    Perturbation("leetspeak", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Replace letters with leetspeak digits/symbols (e to 3, o to 0, a to @, and so on).",
                 _leetspeak, params=(rate_param(0.5),), seed_sensitive=True),
    Perturbation("disemvowel", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Drop a random subset of vowels (SMS style: hello becomes hll).",
                 _disemvowel, params=(rate_param(0.5),), seed_sensitive=True),
    Perturbation("diacritics_add", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Add a combining accent to a random subset of vowels.",
                 _diacritics_add, params=(rate_param(0.5),), seed_sensitive=True),
    Perturbation("diacritics_strip", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Strip all diacritics/accents from the text.",
                 _diacritics_strip, params=(), seed_sensitive=False),
    Perturbation("homoglyph", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Replace letters with visually similar Unicode confusables.",
                 _homoglyph, params=(rate_param(0.3),), seed_sensitive=True),
    Perturbation("full_width", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Map ASCII characters to their full-width Unicode forms.",
                 _full_width, params=(), seed_sensitive=False),
    Perturbation("zero_width", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Insert zero-width spaces between a random subset of characters.",
                 _zero_width, params=(rate_param(0.3),), seed_sensitive=True),
    Perturbation("bidi_override", CATEGORY_NATIVE, FAMILY_CHARACTER,
                 "Wrap the text in a Unicode right-to-left override.",
                 _bidi_override, params=(), seed_sensitive=False),
)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
