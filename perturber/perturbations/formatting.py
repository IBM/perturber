"""Formatting / whitespace perturbations (native tier).

Standard-library-only transforms that change the text's layout rather than its words:
tab-for-space substitution, extra whitespace at word boundaries, and Markdown code-fence
wrapping. ``extra_whitespace`` is seed-sensitive (a random subset of boundaries via ``rate``);
the others are whole-text and deterministic.
"""

from __future__ import annotations

import random
import re

from ..registry import CATEGORY_NATIVE, FAMILY_FORMATTING, ParamSpec, Perturbation, Registry
from ._common import rate_param, subset

# Markdown markup removed by strip_markdown, as (compiled pattern, replacement) pairs applied in
# order. Each keeps the inner text and drops only the surrounding syntax.
_MARKDOWN_RULES = [
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE), ""),         # ATX headings
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),                         # bold
    (re.compile(r"__(.+?)__"), r"\1"),                            # bold (underscores)
    (re.compile(r"\*(.+?)\*"), r"\1"),                             # italic
    (re.compile(r"_(.+?)_"), r"\1"),                              # italic (underscores)
    (re.compile(r"`(.+?)`"), r"\1"),                              # inline code
    (re.compile(r"\[(.+?)\]\((.+?)\)"), r"\1"),                    # links: keep the label
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),              # bullet markers
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),              # numbered-list markers
]

# Line-start numbered-list markers turned into dash bullets by bullet_reformat.
_NUMBERED_MARKER = re.compile(r"^(\s*)\d+\.\s+", re.MULTILINE)


def _tabs_for_spaces(text: str, params: dict, rng: random.Random) -> str:
    """Replace each run of spaces with a single tab."""
    return re.sub(r" +", "\t", text)


def _line_wrap(text: str, params: dict, rng: random.Random) -> str:
    """Hard-wrap the text at ``width`` columns with newlines, breaking on spaces.

    Models text pasted from an email or terminal where lines were wrapped at a fixed width. Long
    single words are left intact rather than split mid-word.
    """
    width = max(1, int(params["width"]))
    out_lines = []
    for line in text.split("\n"):
        current = ""
        for word in line.split(" "):
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                out_lines.append(current)
                current = word
        out_lines.append(current)
    return "\n".join(out_lines)


def _strip_markdown(text: str, params: dict, rng: random.Random) -> str:
    """Remove common Markdown markup, leaving the plain text. Deterministic."""
    out = text
    for pattern, replacement in _MARKDOWN_RULES:
        out = pattern.sub(replacement, out)
    return out


def _bullet_reformat(text: str, params: dict, rng: random.Random) -> str:
    """Convert numbered-list markers at line starts to dash bullets. Deterministic."""
    return _NUMBERED_MARKER.sub(r"\1- ", text)


def _extra_whitespace(text: str, params: dict, rng: random.Random) -> str:
    """Double the space at a random subset of single-space word boundaries."""
    positions = [i for i, c in enumerate(text) if c == " "]
    chosen = subset(positions, params["rate"], rng)
    return "".join((c + " " if i in chosen else c) for i, c in enumerate(text))


def _markdown_wrap(text: str, params: dict, rng: random.Random) -> str:
    """Wrap the text in a Markdown code fence."""
    return f"```\n{text}\n```"


# Ordered as whitespace/layout transforms first, then Markdown-markup transforms.
_PERTURBATIONS = (
    Perturbation("extra_whitespace", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Add extra spaces at a random subset of word boundaries.", _extra_whitespace,
                 params=(rate_param(0.3),), seed_sensitive=True),
    Perturbation("tabs_for_spaces", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Replace runs of spaces with tabs.", _tabs_for_spaces,
                 params=(), seed_sensitive=False),
    Perturbation("line_wrap", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Hard-wrap the text at a fixed column width (email/terminal paste).", _line_wrap,
                 params=(ParamSpec("width", "int", 40, "Column width to wrap at."),),
                 seed_sensitive=False),
    Perturbation("markdown_wrap", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Wrap the text in a Markdown code fence.", _markdown_wrap,
                 params=(), seed_sensitive=False),
    Perturbation("strip_markdown", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Remove Markdown markup, leaving the plain text.", _strip_markdown,
                 params=(), seed_sensitive=False),
    Perturbation("bullet_reformat", CATEGORY_NATIVE, FAMILY_FORMATTING,
                 "Convert numbered list markers to dash bullets.", _bullet_reformat,
                 params=(), seed_sensitive=False),
)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
