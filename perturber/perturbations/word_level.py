# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

"""Word-level perturbations (native tier).

Clean-room, standard-library-only implementations of the word-level perturbation
family, written from documented method descriptions. They use their own algorithms and
add no dependencies.

Determinism follows the service contract: each perturbation is a pure function of
``(text, params, seed)``. Seed-sensitive perturbations draw from the per-call ``rng``
supplied by :func:`perturber.core.perturb`; the others ignore it. ``word_merge`` and
``word_split`` deliberately use a fixed internal seed so their output does not depend
on the request seed, matching the documented behaviour of the reference tier.
"""

from __future__ import annotations

import random
import re
import string

from ..registry import CATEGORY_NATIVE, FAMILY_WORD_LEVEL, ParamSpec, Perturbation, Registry
from ._common import match_leading_case, rate_param, subset

# A word is eligible for a character-level typo when it is alphabetic and long enough
# that a single edit is unlikely to destroy its recognisability. This standard-library
# heuristic stands in for the reference tier's wordfreq-based "natural word" detection.
_MIN_TYPO_WORD_LEN = 4

# A small, curated stop-word set (clean-room; not sourced from any external corpus).
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by",
        "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "from", "into", "than",
        "then", "so", "such", "there", "here", "about", "over", "under", "again",
    }
)

# Words that are never dropped even if they would otherwise be treated as stop words:
# question words, negations, and logical connectives carry meaning that the
# perturbation must preserve (the family is semantics-preserving).
_KEEP_WORDS = frozenset(
    {
        "not", "no", "nor", "never", "none", "cannot",
        "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
        "if", "unless", "because", "and", "or",
    }
)


# --- realistic_typos data (clean-room; authored here, not from any corpus) ---

# QWERTY neighbours for each key, used for fat-finger substitution and insertion. Lowercase;
# case is preserved by the mechanisms when they apply an edit.
_KEYBOARD_ADJACENCY = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh",
    "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

# Frequent real-world misspellings (lowercase base word mapped to a misspelling).
_COMMON_MISSPELLINGS = {
    "the": "teh", "and": "adn", "you": "yuo", "that": "taht", "this": "tihs",
    "with": "wtih", "have": "ahve", "there": "thier", "their": "thier",
    "receive": "recieve", "believe": "beleive", "achieve": "acheive",
    "separate": "seperate", "definitely": "definately", "because": "becuase",
    "tomorrow": "tommorow", "necessary": "neccessary", "occurred": "occured",
    "beginning": "begining", "grammar": "grammer", "government": "goverment",
    "environment": "enviroment", "restaurant": "restarant", "surprise": "suprise",
    "weird": "wierd", "friend": "freind", "which": "wich", "would": "wuold",
    "should": "shuold", "people": "pepole", "really": "realy", "until": "untill",
}

# A few clean-room phonetic-style rewrites. Each is (compiled regex, replacement); applied
# only when the pattern is present in the word.
_PHONETIC_RULES = [
    (re.compile(r"ph"), "f"),
    (re.compile(r"tion"), "shun"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"que\b"), "k"),
    (re.compile(r"([bcdfghjklmnpqrstvwxyz])\1"), r"\1"),  # collapse a doubled consonant
]


def _match_case_lower(src: str, ch: str) -> str:
    """Case a replacement letter to match ``src`` (uppercase if ``src`` is; else lowercase).

    Distinct from ``_common.match_char_case``: this variant *forces* lowercase when the source is
    lowercase, which suits the typo mechanisms that draw replacement letters from a lowercase map.
    """
    return ch.upper() if src.isupper() else ch.lower()


def _kbd_sub(word: str, rng: random.Random) -> str:
    """Replace one letter with a physically adjacent QWERTY key, preserving case."""
    positions = [i for i, c in enumerate(word) if c.lower() in _KEYBOARD_ADJACENCY]
    if not positions:
        return word
    i = rng.choice(positions)
    neighbours = _KEYBOARD_ADJACENCY[word[i].lower()]
    repl = _match_case_lower(word[i], rng.choice(neighbours))
    return word[:i] + repl + word[i + 1:]


def _kbd_insert(word: str, rng: random.Random) -> str:
    """Insert a key adjacent to the one at a random position (fat-finger double-hit)."""
    positions = [i for i, c in enumerate(word) if c.lower() in _KEYBOARD_ADJACENCY]
    if not positions:
        return word
    i = rng.choice(positions)
    extra = _match_case_lower(word[i], rng.choice(_KEYBOARD_ADJACENCY[word[i].lower()]))
    return word[:i] + extra + word[i:]


def _transpose(word: str, rng: random.Random) -> str:
    """Swap two adjacent letters."""
    if len(word) < 2:
        return word
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _double(word: str, rng: random.Random) -> str:
    """Duplicate one letter (a key held a beat too long)."""
    i = rng.randrange(len(word))
    return word[:i] + word[i] + word[i:]


def _drop(word: str, rng: random.Random) -> str:
    """Drop one letter."""
    if len(word) < 2:
        return word
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1:]


def _capslip(word: str, rng: random.Random) -> str:
    """Flip the case of one letter (missed or stray shift)."""
    positions = [i for i, c in enumerate(word) if c.isalpha()]
    if not positions:
        return word
    i = rng.choice(positions)
    ch = word[i]
    flipped = ch.lower() if ch.isupper() else ch.upper()
    return word[:i] + flipped + word[i + 1:]


def _vowel_confusion(word: str, rng: random.Random) -> str:
    """Swap one vowel for a plausible wrong vowel (a common unstressed-syllable slip).

    Models the "seperate"/"definately" class of error generally, rather than via the fixed
    misspelling table: it picks a vowel that is not the word's first or last letter (so the word
    stays recognisable) and replaces it with a different vowel, preserving case.
    """
    positions = [i for i, c in enumerate(word) if c.lower() in "aeiou" and 0 < i < len(word) - 1]
    if not positions:
        return word
    i = rng.choice(positions)
    others = [v for v in "aeiou" if v != word[i].lower()]
    return word[:i] + _match_case_lower(word[i], rng.choice(others)) + word[i + 1:]


def _consonant_double(word: str, rng: random.Random) -> str:
    """Wrongly double an interior consonant (the "runing" -> "runnning" / "adress" class).

    The inverse direction (collapsing a real double) is already covered by the phonetic rule, so
    this only *adds* a duplicate to a single, currently-undoubled interior consonant.
    """
    positions = [
        i for i, c in enumerate(word)
        if c.lower() in "bcdfghjklmnpqrstvwxyz"
        and 0 < i < len(word)
        and word[i].lower() != word[i - 1].lower()
    ]
    if not positions:
        return word
    i = rng.choice(positions)
    return word[:i] + word[i] + word[i:]


def _misspell(word: str, rng: random.Random) -> str:
    """Replace the word with a known misspelling, preserving a leading capital."""
    return match_leading_case(word, _COMMON_MISSPELLINGS[word.lower()])


def _phonetic(word: str, rng: random.Random) -> str:
    """Apply one matching phonetic rule; return the word unchanged if none match."""
    applicable = [(pat, repl) for pat, repl in _PHONETIC_RULES if pat.search(word.lower())]
    if not applicable:
        return word
    pat, repl = rng.choice(applicable)
    # Apply on the lowercase form to keep the rule simple, then restore a leading capital.
    return match_leading_case(word, pat.sub(repl, word.lower(), count=1))


# Character-level mechanisms usable on any eligible word.
_CHAR_MECHANISMS = (
    _kbd_sub, _kbd_insert, _transpose, _double, _drop, _capslip,
    _vowel_confusion, _consonant_double,
)


def _realistic_typo(word: str, rng: random.Random) -> str:
    """Apply one realistic typo to ``word``.

    Prefers a word-level mechanism when it applies (a known misspelling, or a phonetic
    rule matches); otherwise picks a character-level mechanism. This keeps every event
    productive and the mix realistic.
    """
    word_level = []
    if word.lower() in _COMMON_MISSPELLINGS:
        word_level.append(_misspell)
    if any(pat.search(word.lower()) for pat, _ in _PHONETIC_RULES):
        word_level.append(_phonetic)
    # Offer word-level mechanisms alongside the character-level ones so the choice is varied
    # rather than always preferring a misspelling when one exists.
    mechanism = rng.choice(word_level + list(_CHAR_MECHANISMS))
    return mechanism(word, rng)


def _iter_word_indices(tokens: list[str]) -> list[int]:
    """Indices of ``tokens`` that are eligible for a character-level typo."""
    return [
        i
        for i, tok in enumerate(tokens)
        if tok.isalpha() and len(tok) >= _MIN_TYPO_WORD_LEN
    ]


def _iter_realistic_indices(tokens: list[str]) -> list[int]:
    """Indices eligible for a realistic typo (alphabetic, length >= 3)."""
    return [i for i, tok in enumerate(tokens) if tok.isalpha() and len(tok) >= 3]


def _one_typo(word: str, rng: random.Random) -> str:
    """Apply a single character-level edit to ``word`` using ``rng``."""
    edits = ("swap", "delete", "duplicate", "replace")
    edit = rng.choice(edits)
    i = rng.randrange(len(word))
    if edit == "swap" and len(word) >= 2:
        j = min(i + 1, len(word) - 1)
        if j == i:
            j = i - 1
        chars = list(word)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    if edit == "delete" and len(word) >= 2:
        return word[:i] + word[i + 1:]
    if edit == "duplicate":
        return word[:i] + word[i] + word[i:]
    # "replace" (and the fallback for short words): swap the character for a nearby
    # letter of the same case.
    ch = word[i]
    if ch.isalpha():
        base = ord("a") if ch.islower() else ord("A")
        offset = (ord(ch.lower()) - ord("a") + rng.choice((-1, 1))) % 26
        repl = chr(base + offset)
        return word[:i] + repl + word[i + 1:]
    return word


def _typos(text: str, params: dict, rng: random.Random) -> str:
    """Introduce up to ``count`` single-character typos into eligible words."""
    count = params["count"]
    if count <= 0 or not text:
        return text
    tokens = text.split(" ")
    eligible = _iter_word_indices(tokens)
    if not eligible:
        return text
    for _ in range(count):
        idx = rng.choice(eligible)
        tokens[idx] = _one_typo(tokens[idx], rng)
    return " ".join(tokens)


def _realistic_typos(text: str, params: dict, rng: random.Random) -> str:
    """Introduce ``count`` realistic typos, modelling how people actually mistype.

    Each event picks a random eligible word and applies one mechanism: keyboard-adjacent
    substitution or insertion, transposition, a doubled or dropped key, a capitalisation
    slip, a vowel confusion, a wrongly doubled consonant, a common misspelling, or a
    phonetic slip. An event may instead be a space slip (a missed space that merges two
    adjacent words), the single-keystroke analogue of the systematic ``word_merge``.
    Word-level mechanisms fire only when they apply, so the output stays recognisable.
    """
    count = params["count"]
    if count <= 0 or not text:
        return text
    tokens = text.split(" ")
    if not _iter_realistic_indices(tokens):
        return text
    for _ in range(count):
        # A minority of events are a missed-space merge (needs two adjacent tokens); the rest are
        # per-word mechanisms. Re-derive eligibility each step since a merge changes the tokens.
        merge_spots = [i for i in range(len(tokens) - 1) if tokens[i] and tokens[i + 1]]
        if merge_spots and rng.random() < 0.15:
            i = rng.choice(merge_spots)
            tokens[i : i + 2] = [tokens[i] + tokens[i + 1]]
            continue
        eligible = _iter_realistic_indices(tokens)
        if not eligible:
            break
        idx = rng.choice(eligible)
        tokens[idx] = _realistic_typo(tokens[idx], rng)
    return " ".join(tokens)


def _drop_stop_words(text: str, params: dict, rng: random.Random) -> str:
    """Remove common stop words, keeping question, negation, and logic words."""
    tokens = text.split(" ")
    kept = []
    for tok in tokens:
        core = tok.lower().strip(string.punctuation)
        if core in _STOP_WORDS and core not in _KEEP_WORDS:
            continue
        kept.append(tok)
    return " ".join(kept)


def _punctuation_spaces(text: str, params: dict, rng: random.Random) -> str:
    """Insert a space on each side of punctuation characters."""
    out = []
    for ch in text:
        if ch in string.punctuation:
            out.append(" " + ch + " ")
        else:
            out.append(ch)
    # Collapse the runs of spaces this introduces (and trim the ends) so the result
    # stays readable.
    return re.sub(r" +", " ", "".join(out)).strip()


def _sequence_spaces(text: str, params: dict, rng: random.Random) -> str:
    """Replace the separator between tokens with a random-length run of spaces."""
    tokens = text.split(" ")
    if len(tokens) < 2:
        return text
    out = tokens[0]
    for tok in tokens[1:]:
        out += " " * rng.randint(1, 5) + tok
    return out


def _word_merge(text: str, params: dict, rng: random.Random) -> str:
    """Merge one randomly chosen adjacent word pair by removing the space between them.

    Seed-sensitive: the seed selects which pair is merged.
    """
    tokens = text.split(" ")
    if len(tokens) < 2:
        return text
    i = rng.randrange(len(tokens) - 1)
    merged = tokens[:i] + [tokens[i] + tokens[i + 1]] + tokens[i + 2:]
    return " ".join(merged)


def _word_split(text: str, params: dict, rng: random.Random) -> str:
    """Split one randomly chosen word by inserting a space at an interior position.

    Seed-sensitive: the seed selects which word is split and where.
    """
    tokens = text.split(" ")
    splittable = [i for i, tok in enumerate(tokens) if len(tok) >= 2]
    if not splittable:
        return text
    i = rng.choice(splittable)
    word = tokens[i]
    pos = rng.randrange(1, len(word))
    tokens[i] = word[:pos] + " " + word[pos:]
    return " ".join(tokens)


_TYPOS = Perturbation(
    name="typos",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Introduce a fixed number of character-level typos into natural-language words.",
    apply=_typos,
    params=(ParamSpec("count", "int", 5, "Number of character-level typos to introduce."),),
    seed_sensitive=True,
)

_REALISTIC_TYPOS = Perturbation(
    name="realistic_typos",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description=(
        "Introduce realistic typos: keyboard-adjacent slips, common misspellings, "
        "transpositions, doubled/dropped keys, vowel confusions, doubled consonants, "
        "missed-space merges, phonetic slips, and capitalisation errors."
    ),
    apply=_realistic_typos,
    params=(ParamSpec("count", "int", 3, "Number of realistic typos to introduce."),),
    seed_sensitive=True,
)

_DROP_STOP_WORDS = Perturbation(
    name="drop_stop_words",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Remove common stop words (keeping question, negation, and logic words).",
    apply=_drop_stop_words,
    params=(),
    seed_sensitive=False,
)

_PUNCTUATION_SPACES = Perturbation(
    name="punctuation_spaces",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Insert spaces around punctuation.",
    apply=_punctuation_spaces,
    params=(),
    seed_sensitive=False,
)

_SEQUENCE_SPACES = Perturbation(
    name="sequence_spaces",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Insert random-length space sequences between tokens.",
    apply=_sequence_spaces,
    params=(),
    seed_sensitive=True,
)

_WORD_MERGE = Perturbation(
    name="word_merge",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Merge adjacent words by removing a space.",
    apply=_word_merge,
    params=(),
    seed_sensitive=True,
)

_WORD_SPLIT = Perturbation(
    name="word_split",
    category=CATEGORY_NATIVE,
    family=FAMILY_WORD_LEVEL,
    description="Split a word by inserting a space.",
    apply=_word_split,
    params=(),
    seed_sensitive=True,
)

# --- Additional deterministic word-level transforms (clean-room data) ---

# Expansion mapped to contraction (contractions() contracts by default).
_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't", "is not": "isn't",
    "are not": "aren't", "was not": "wasn't", "were not": "weren't", "have not": "haven't",
    "has not": "hasn't", "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
    "can not": "can't", "cannot": "can't", "could not": "couldn't", "should not": "shouldn't",
    "it is": "it's", "that is": "that's", "there is": "there's", "you are": "you're",
    "they are": "they're", "we are": "we're", "i am": "I'm", "let us": "let's",
}

# American mapped to British spelling (curated).
_UK_SPELLINGS = {
    "color": "colour", "colors": "colours", "flavor": "flavour", "honor": "honour",
    "favorite": "favourite", "behavior": "behaviour", "neighbor": "neighbour",
    "center": "centre", "theater": "theatre", "meter": "metre", "liter": "litre",
    "organize": "organise", "recognize": "recognise", "realize": "realise",
    "analyze": "analyse", "apologize": "apologise", "catalog": "catalogue",
    "traveled": "travelled", "traveling": "travelling", "canceled": "cancelled",
    "defense": "defence", "offense": "offence", "license": "licence", "gray": "grey",
}



def _contractions(text: str, params: dict, rng: random.Random) -> str:
    """Contract common expanded phrases (do not becomes don't). Deterministic."""
    out = text
    # Longest phrases first so "can not" wins over shorter overlaps.
    for phrase in sorted(_CONTRACTIONS, key=len, reverse=True):
        out = re.sub(
            rf"\b{re.escape(phrase)}\b",
            lambda m: match_leading_case(m.group(0), _CONTRACTIONS[phrase]),
            out,
            flags=re.IGNORECASE,
        )
    return out


def _apostrophe_drop(text: str, params: dict, rng: random.Random) -> str:
    """Drop the apostrophe from contractions and possessives (don't -> dont, it's -> its).

    One of the most frequent real-world real-word errors, and distinct from the other word-level
    perturbations: ``contractions`` goes the other way (do not -> don't), and ``confusable_words``
    swaps whole words. Deterministic: it removes a straight or curly apostrophe that sits between two
    letters (so possessive/contraction forms), leaving leading/trailing quotes untouched.
    """
    return re.sub(r"(?<=[A-Za-z])['’](?=[A-Za-z])", "", text)


def _number_format(text: str, params: dict, rng: random.Random) -> str:
    """Add thousands separators to bare integers of 4+ digits (1000 becomes 1,000)."""
    return re.sub(r"\b\d{4,}\b", lambda m: format(int(m.group(0)), ","), text)


def _spelling_uk(text: str, params: dict, rng: random.Random) -> str:
    """Convert American spellings to British ones. Deterministic."""
    def repl(m):
        return match_leading_case(m.group(0), _UK_SPELLINGS[m.group(0).lower()])

    pattern = r"\b(" + "|".join(re.escape(w) for w in _UK_SPELLINGS) + r")\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _wordnet():
    """Return the WordNet corpus, auto-downloading the data on first use if needed.

    ``nltk`` is a core dependency, so the package is expected to be importable. The WordNet corpus
    (``wordnet`` + ``omw-1.4``) may not be downloaded yet; if loading fails for that reason, this
    fetches it once and retries. It raises ``ValueError`` (never silently returns ``None``) when
    nltk is missing or the corpus cannot be obtained, so a synonym run fails loudly rather than
    passing the text through unchanged, which would look like a false negative in a benchmark.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError as exc:
        raise ValueError(
            "The 'synonym' perturbation requires nltk (a core dependency). Install perturber's "
            "core dependencies to use it."
        ) from exc

    try:
        wordnet.ensure_loaded()
        return wordnet
    except LookupError:
        # Corpus not downloaded yet: fetch it once, then retry.
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    try:
        wordnet.ensure_loaded()
        return wordnet
    except Exception as exc:  # noqa: BLE001 - report why the corpus is unusable
        raise ValueError(
            "The 'synonym' perturbation could not load the WordNet corpus. Ensure network access "
            "for the one-time download, or pre-fetch it with "
            "nltk.download('wordnet'); nltk.download('omw-1.4')."
        ) from exc


def _pos_tagger():
    """Return ``nltk.pos_tag`` with its perceptron-tagger data present (auto-downloaded once).

    Used to constrain synonym substitution to the word's part of speech; without it, WordNet
    synonymy pulls wrong-POS senses (for example, treating the noun 'capital' as a verb). Fails
    loud like :func:`_wordnet` rather than silently degrading.
    """
    import nltk
    try:
        nltk.pos_tag(["probe"])
        return nltk.pos_tag
    except LookupError:
        # Tagger data not present yet: fetch both the current and legacy resource names, then retry.
        for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                nltk.download(pkg)
            except Exception:  # noqa: BLE001 - try the next name
                pass
    try:
        nltk.pos_tag(["probe"])
        return nltk.pos_tag
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "The 'synonym' perturbation could not load the nltk POS tagger. Ensure network access "
            "for the one-time download, or pre-fetch it with "
            "nltk.download('averaged_perceptron_tagger_eng')."
        ) from exc


# Penn Treebank tag prefix -> WordNet POS. Only content-word classes are substitutable; anything
# else (prepositions, determiners, ...) maps to None and is left unchanged.
def _penn_to_wordnet(tag: str, wn):
    return {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ, "R": wn.ADV}.get(tag[:1])


def _synonym(text: str, params: dict, rng: random.Random) -> str:
    """Replace a random subset of content words with a context-appropriate WordNet synonym.

    To keep the substitution meaning-preserving, each candidate word is disambiguated in context:
    the sentence is POS-tagged, the word's part of speech constrains the WordNet senses considered,
    and Lesk picks the sense whose gloss best overlaps the surrounding words. Synonyms are drawn
    only from that one chosen sense. A word whose POS is not a content class, or whose chosen sense
    has no alternative lemma, is left unchanged rather than swapped for a wrong-sense word (which is
    what an unconstrained all-senses lookup produced: 'capital' -> 'majuscule', 'stop' ->
    'barricade'). All steps (POS tagging, Lesk) are deterministic, so output stays a pure function
    of (text, params, seed).

    Requires WordNet and the nltk POS tagger; the loaders auto-download on first use and raise if
    they cannot be obtained (this perturbation never silently returns the text unchanged).
    """
    from nltk.wsd import lesk

    wn = _wordnet()
    pos_tag = _pos_tagger()
    tokens = text.split(" ")
    # POS-tag against the raw whitespace tokens so tag indices line up with `tokens`.
    tags = [t for _, t in pos_tag(tokens)]
    eligible = [i for i, t in enumerate(tokens) if t.isalpha() and len(t) >= 4]
    for i in subset(eligible, params["rate"], rng):
        word = tokens[i]
        wn_pos = _penn_to_wordnet(tags[i], wn)
        if wn_pos is None:
            continue  # not a content word in this context; do not substitute
        sense = lesk(tokens, word.lower(), pos=wn_pos)
        if sense is None:
            continue
        syns = sorted({
            lemma.name().replace("_", " ")
            for lemma in sense.lemmas()
            if lemma.name().replace("_", " ").lower() != word.lower()
        })
        if syns:
            tokens[i] = match_leading_case(word, rng.choice(syns))
    return " ".join(tokens)


_CONTRACTIONS_P = Perturbation(
    name="contractions", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Contract common expanded phrases (for example, 'do not' becomes \"don't\").",
    apply=_contractions, params=(), seed_sensitive=False,
)
_APOSTROPHE_DROP_P = Perturbation(
    name="apostrophe_drop", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Drop apostrophes from contractions and possessives (for example, \"don't\" becomes 'dont').",
    apply=_apostrophe_drop, params=(), seed_sensitive=False,
)
_NUMBER_FORMAT_P = Perturbation(
    name="number_format", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Add thousands separators to large integers (for example, '1000' becomes '1,000').",
    apply=_number_format, params=(), seed_sensitive=False,
)
_SPELLING_UK_P = Perturbation(
    name="spelling_uk", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Convert American spellings to British (for example, 'color' becomes 'colour').",
    apply=_spelling_uk, params=(), seed_sensitive=False,
)
_SYNONYM_P = Perturbation(
    name="synonym", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Replace content words with WordNet synonyms.",
    apply=_synonym,
    params=(rate_param(0.3, "Fraction of content words to replace (0 to 1)."),),
    seed_sensitive=True,
)


# --- Realistic-noise word-level transforms (clean-room data) ---

# Confusion set: valid-but-wrong words that are commonly swapped for one another (homophones and
# near-homophones). Substituting one produces a real-word error, one that stays a valid word, unlike
# a nonword typo. Lowercase base word mapped to the confusable target; leading-letter case is
# preserved on use.
_CONFUSABLE_WORDS = {
    "their": "there", "there": "their", "your": "you're", "its": "it's",
    "then": "than", "than": "then", "to": "too", "too": "to", "of": "off",
    "were": "we're", "loose": "lose", "affect": "effect", "definitely": "defiantly",
}


def _repeated_chars(text: str, params: dict, rng: random.Random) -> str:
    """Stretch a random subset of words for emphasis ("so" becomes "sooo").

    Duplicates the final letter of a chosen word two or three times, and lengthens a trailing
    run of "!" or "?" the same way, mimicking casual emphatic typing.
    """
    tokens = text.split(" ")
    eligible = [i for i, t in enumerate(tokens) if t and (t[-1].isalpha() or t[-1] in "!?")]
    chosen = subset(eligible, params["rate"], rng)
    for i in chosen:
        token = tokens[i]
        extra = rng.randint(2, 3)
        tokens[i] = token + token[-1] * extra
    return " ".join(tokens)


def _confusable_words(text: str, params: dict, rng: random.Random) -> str:
    """Replace a random subset of confusable words with a wrong-but-valid alternative."""
    tokens = text.split(" ")
    eligible = [i for i, t in enumerate(tokens) if t.strip(".,!?;:").lower() in _CONFUSABLE_WORDS]
    chosen = subset(eligible, params["rate"], rng)
    for i in chosen:
        token = tokens[i]
        # Preserve any trailing punctuation attached to the token.
        core = token.rstrip(".,!?;:")
        trailing = token[len(core):]
        tokens[i] = match_leading_case(core, _CONFUSABLE_WORDS[core.lower()]) + trailing
    return " ".join(tokens)


# Sentence-terminating punctuation dropped by punctuation_drop, and the boundary it splits on.
_SENTENCE_BOUNDARY = re.compile(r"([.!?]+)(\s+)")


def _punctuation_drop(text: str, params: dict, rng: random.Random) -> str:
    """Casual style: drop terminal sentence punctuation and lowercase each sentence start.

    Whole-text and deterministic: splits on sentence-ending punctuation, lowercases the first
    letter of each resulting sentence, and removes the terminal punctuation.
    """
    # Lowercase the first alphabetic character of the whole text.
    def lower_first(s: str) -> str:
        for idx, ch in enumerate(s):
            if ch.isalpha():
                return s[:idx] + ch.lower() + s[idx + 1:]
        return s

    # Split into sentences on the terminal punctuation, keeping the separating whitespace.
    parts = _SENTENCE_BOUNDARY.split(text)
    # parts alternates: sentence, punctuation, whitespace, sentence, punctuation, whitespace, ...
    out = []
    i = 0
    while i < len(parts):
        sentence = lower_first(parts[i])
        out.append(sentence)
        # Skip the captured punctuation (parts[i+1]); keep the whitespace (parts[i+2]).
        if i + 2 < len(parts):
            out.append(parts[i + 2])
        i += 3
    result = "".join(out)
    # Also strip any trailing terminal punctuation left on the final sentence.
    return result.rstrip(".!?")


_REPEATED_CHARS_P = Perturbation(
    name="repeated_chars", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Stretch words for emphasis (for example, 'so' becomes 'sooo').",
    apply=_repeated_chars, params=(rate_param(0.3),), seed_sensitive=True,
)
_CONFUSABLE_WORDS_P = Perturbation(
    name="confusable_words", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Swap confusable words for a wrong-but-valid alternative (their becomes there, to becomes too).",
    apply=_confusable_words, params=(rate_param(0.5),), seed_sensitive=True,
)
_PUNCTUATION_DROP_P = Perturbation(
    name="punctuation_drop", category=CATEGORY_NATIVE, family=FAMILY_WORD_LEVEL,
    description="Drop terminal punctuation and lowercase each sentence start (casual style).",
    apply=_punctuation_drop, params=(), seed_sensitive=False,
)

# Ordered for readability: mistyping/noise first (realistic_typos leads, it is the UI default),
# then lexical rewrites, then word-boundary edits, then spacing and punctuation.
_PERTURBATIONS = (
    _REALISTIC_TYPOS,
    _TYPOS,
    _CONFUSABLE_WORDS_P,
    _REPEATED_CHARS_P,
    _SYNONYM_P,
    _CONTRACTIONS_P,
    _APOSTROPHE_DROP_P,
    _SPELLING_UK_P,
    _NUMBER_FORMAT_P,
    _DROP_STOP_WORDS,
    _WORD_MERGE,
    _WORD_SPLIT,
    _PUNCTUATION_SPACES,
    _SEQUENCE_SPACES,
    _PUNCTUATION_DROP_P,
)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
