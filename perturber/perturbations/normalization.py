"""Text-normalization perturbations (native tier).

Standard-library-only transforms that renormalise surface conventions while preserving meaning:
expanding abbreviations, reformatting dates/units, swapping quote styles, and converting between
emoji and text emoticons. These are deterministic (whole-text) except where a ``rate`` subset is
the natural unit.
"""

from __future__ import annotations

import random
import re

from ..registry import CATEGORY_NATIVE, FAMILY_NORMALIZATION, Perturbation, Registry
from ._common import match_leading_case

# --- clean-room data ---

# Abbreviation mapped to expansion (word boundary, case insensitive; expansion direction).
_ABBREVIATIONS = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "and so on", "vs.": "versus",
    "approx.": "approximately", "dept.": "department", "govt.": "government",
    "mr.": "mister", "dr.": "doctor", "st.": "street", "min.": "minutes",
    "hr.": "hours", "yr.": "years", "info": "information", "asap": "as soon as possible",
    "fyi": "for your information", "aka": "also known as",
}

# Unit abbreviation mapped to the full word (used after a number).
_UNITS = {
    "km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
    "kg": "kilograms", "g": "grams", "mg": "milligrams", "lb": "pounds", "oz": "ounces",
    "hr": "hours", "min": "minutes", "sec": "seconds", "ml": "milliliters", "l": "liters",
}

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]

# Emoji and emoticon pairs, and the reverse map (first or canonical emoticon per emoji), built once.
_EMOTICON_TO_EMOJI = {
    ":)": "🙂", ":-)": "🙂", ":(": "🙁", ":-(": "🙁", ":D": "😀", ":-D": "😀",
    ";)": "😉", ";-)": "😉", ":P": "😛", ":-P": "😛", ":'(": "😢", "<3": "❤️", ":o": "😮",
}
_EMOJI_TO_EMOTICON: dict = {}
for _emoticon, _emoji in _EMOTICON_TO_EMOJI.items():
    _EMOJI_TO_EMOTICON.setdefault(_emoji, _emoticon)

# Matches a number followed by a known unit abbreviation; compiled once from _UNITS.
_UNIT_PATTERN = re.compile(r"\b(\d+)\s*(" + "|".join(_UNITS) + r")\b")

# Typographic characters folded back to ASCII by smart_to_ascii (the inverse of quote_style),
# modelling what happens when text is copied out of a word processor into a plain-text field.
_SMART_TO_ASCII = {
    "“": '"', "”": '"',      # curly double quotes
    "‘": "'", "’": "'",      # curly single quotes / apostrophe
    "–": "-", "—": "--",     # en dash, em dash
    "…": "...",                    # ellipsis
    " ": " ",                      # non-breaking space
}

# URL and email patterns masked by url_email_mask; compiled once.
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _abbreviations(text: str, params: dict, rng: random.Random) -> str:
    """Expand common abbreviations (for example, 'e.g.' becomes 'for example'). Deterministic."""
    out = text
    for abbr in sorted(_ABBREVIATIONS, key=len, reverse=True):
        # Escape and match; abbreviations may end in '.', so match literally with boundaries.
        pat = re.compile(rf"(?<!\w){re.escape(abbr)}(?!\w)", re.IGNORECASE)
        out = pat.sub(lambda m: match_leading_case(m.group(0), _ABBREVIATIONS[abbr]), out)
    return out


def _date_unit(text: str, params: dict, rng: random.Random) -> str:
    """Reformat ISO dates (2026-01-01 becomes January 1, 2026) and spell out units after numbers."""
    def date_repl(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{_MONTHS[mo - 1]} {d}, {y}"
        return m.group(0)

    out = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", date_repl, text)

    # "5km" or "5 km" becomes "5 kilometers" (require a word boundary after the unit).
    return _UNIT_PATTERN.sub(lambda m: f"{m.group(1)} {_UNITS[m.group(2)]}", out)


def _quote_style(text: str, params: dict, rng: random.Random) -> str:
    """Swap straight quotes for typographic ('curly') quotes. Deterministic."""
    # Double quotes: alternate open/close.
    out = []
    open_d = True
    for c in text:
        if c == '"':
            out.append("“" if open_d else "”")
            open_d = not open_d
        else:
            out.append(c)
    text = "".join(out)
    # Apostrophes become the right single quote.
    return text.replace("'", "’")


def _emoji(text: str, params: dict, rng: random.Random) -> str:
    """Convert text emoticons to emoji (':)' becomes '🙂'). Deterministic; longest match first."""
    out = text
    for emo in sorted(_EMOTICON_TO_EMOJI, key=len, reverse=True):
        out = out.replace(emo, _EMOTICON_TO_EMOJI[emo])
    return out


def _emoticon(text: str, params: dict, rng: random.Random) -> str:
    """Convert emoji back to text emoticons ('🙂' becomes ':)'). Deterministic."""
    out = text
    for emoji, emoticon in _EMOJI_TO_EMOTICON.items():
        out = out.replace(emoji, emoticon)
    return out


def _smart_to_ascii(text: str, params: dict, rng: random.Random) -> str:
    """Fold typographic punctuation to plain ASCII. Deterministic; the inverse of quote_style."""
    out = text
    for smart, ascii_form in _SMART_TO_ASCII.items():
        out = out.replace(smart, ascii_form)
    return out


def _url_email_mask(text: str, params: dict, rng: random.Random) -> str:
    """Replace URLs and email addresses with stable placeholders. Deterministic."""
    out = _URL_PATTERN.sub("<URL>", text)
    return _EMAIL_PATTERN.sub("<EMAIL>", out)


# Ordered as quote/punctuation normalization, then abbreviation/number expansion, then the emoji
# pair; url_email_mask is last because it is out of scope (it removes content).
_PERTURBATIONS = (
    Perturbation("quote_style", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Swap straight quotes for typographic (curly) quotes.",
                 _quote_style, params=(), seed_sensitive=False),
    Perturbation("smart_to_ascii", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Fold typographic quotes, dashes, and ellipses to plain ASCII.",
                 _smart_to_ascii, params=(), seed_sensitive=False),
    Perturbation("abbreviations", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Expand common abbreviations (for example, 'e.g.' becomes 'for example').",
                 _abbreviations, params=(), seed_sensitive=False),
    Perturbation("date_unit", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Reformat ISO dates and spell out units after numbers.",
                 _date_unit, params=(), seed_sensitive=False),
    Perturbation("emoji", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Convert text emoticons to emoji (for example, ':)' becomes '🙂').",
                 _emoji, params=(), seed_sensitive=False),
    Perturbation("emoticon", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Convert emoji to text emoticons (for example, '🙂' becomes ':)').",
                 _emoticon, params=(), seed_sensitive=False),
    # Removes content (URLs/emails become placeholders) rather than varying its surface form, so it
    # is out of scope for the robustness benchmark: in_scope=False.
    Perturbation("url_email_mask", CATEGORY_NATIVE, FAMILY_NORMALIZATION,
                 "Replace URLs and emails with <URL> and <EMAIL> placeholders.",
                 _url_email_mask, params=(), seed_sensitive=False, in_scope=False),
)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
