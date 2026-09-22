"""Registry of perturbations.

Every perturbation is registered under a unique name together with metadata describing
its provenance category, its family, its parameters, and a callable that applies it to a
single string. The registry is the single place the library, CLI, and HTTP API consult to
resolve a perturbation by name; adding a perturbation therefore requires no change to the
core, the API, or the CLI.

Terminology:

- ``category`` is the user-facing label for the provenance tier. This distribution ships only
  the ``native`` tier (clean-room, permissively licensed implementations plus the ``baseline``
  control).
- ``family`` groups perturbations by the kind of transformation they perform and is used
  only for documentation and OpenAPI grouping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Provenance categories (user-facing tier labels). This distribution ships only ``native``.
CATEGORY_NATIVE = "native"
CATEGORIES = (CATEGORY_NATIVE,)

# Families, used for documentation and OpenAPI grouping.
FAMILY_WORD_LEVEL = "word_level"
FAMILY_PADDING = "padding"
FAMILY_CONTEXT = "context"
FAMILY_CONTROL = "control"
FAMILY_PARAPHRASING = "paraphrasing"
FAMILY_CHARACTER = "character"
FAMILY_FORMATTING = "formatting"
FAMILY_NORMALIZATION = "normalization"
FAMILY_ADVERSARIAL = "adversarial"
FAMILY_SCHEMA = "schema"
FAMILY_REASONING = "reasoning"
FAMILIES = (
    FAMILY_WORD_LEVEL,
    FAMILY_PADDING,
    FAMILY_CONTEXT,
    FAMILY_CONTROL,
    FAMILY_PARAPHRASING,
    FAMILY_CHARACTER,
    FAMILY_FORMATTING,
    FAMILY_NORMALIZATION,
    FAMILY_ADVERSARIAL,
    FAMILY_SCHEMA,
    FAMILY_REASONING,
)

# A perturbation applies one transformation to one string. ``params`` are the resolved
# parameters (defaults merged with any caller-supplied values); ``rng`` is a per-call
# random source seeded deterministically from the request seed. Perturbations backed by
# a global RNG may ignore ``rng`` because the core also seeds the global generators before
# calling; see ``core.perturb``.
#
# ``apply`` may accept an optional trailing keyword ``context`` (a dict) carrying
# transport-only, non-deterministic inputs that must not become part of the perturbation's
# identity; for example a per-request ``api_key`` for model-backed perturbations. Most
# perturbations ignore it; the core passes it only when given. It is never persisted or echoed.
ApplyFn = Callable[..., str]

_PY_TYPES = {"int": int, "float": float, "str": str, "bool": bool}


@dataclass(frozen=True)
class ParamSpec:
    """Specification of a single perturbation parameter."""

    name: str
    type: str  # one of _PY_TYPES
    default: object
    description: str = ""
    # Optional allowed values. When set, this is a closed set the caller should pick from;
    # UIs render it as a dropdown. Stored as a tuple so the frozen dataclass stays hashable.
    choices: tuple = ()

    def coerce(self, value: object) -> object:
        """Coerce a raw value (for example a CLI string) to the declared type."""
        if value is None:
            return self.default
        if self.type == "bool" and isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        py_type = _PY_TYPES.get(self.type)
        if py_type is None or isinstance(value, py_type):
            return value
        return py_type(value)

    def describe(self) -> dict:
        described = {"type": self.type, "default": self.default, "description": self.description}
        if self.choices:
            described["choices"] = list(self.choices)
        return described


@dataclass(frozen=True)
class Perturbation:
    """A named, single-string perturbation and its metadata."""

    name: str
    category: str
    family: str
    description: str
    apply: ApplyFn
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)
    seed_sensitive: bool = False
    # Optional: the exact system prompt a model-backed perturbation sends, exposed so UIs can
    # show it. Empty for deterministic perturbations.
    prompt: str = ""
    # Whether this is an LLM robustness perturbation in the sense this project uses (a natural,
    # semantics-preserving surface transformation). A few registered perturbations fall outside
    # that definition (prompt-injection probes, content-removing or heavy-rewrite transforms);
    # they are still callable but are marked out of scope so UIs and benchmark harnesses can
    # exclude them. This flag (exposed in ``describe()``) is the single source of truth for scope.
    in_scope: bool = True

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.params}

    def resolve_params(self, given: Optional[dict]) -> dict:
        """Merge caller-supplied parameters over the defaults, coercing to declared types.

        Unknown parameter names are rejected so that typos in a request surface as an error
        rather than being silently ignored.
        """
        specs = {p.name: p for p in self.params}
        if given:
            unknown = set(given) - set(specs)
            if unknown:
                raise ValueError(
                    f"Unknown parameter(s) for '{self.name}': {sorted(unknown)}; "
                    f"valid parameters: {sorted(specs)}"
                )
        resolved: dict = {}
        for name, spec in specs.items():
            value = given.get(name) if given else None
            resolved[name] = spec.coerce(value)
        return resolved

    def describe(self) -> dict:
        described = {
            "name": self.name,
            "category": self.category,
            "family": self.family,
            "description": self.description,
            "params": {p.name: p.describe() for p in self.params},
            "seed_sensitive": self.seed_sensitive,
            "in_scope": self.in_scope,
        }
        if self.prompt:
            described["prompt"] = self.prompt
        return described


class Registry:
    """An ordered collection of perturbations keyed by ``(category, name)``.

    A perturbation name is unique only within its category, so the registry keys on the
    ``(category, name)`` pair. Lookups by bare name still succeed when the name is
    unambiguous across the registered categories; when it is not, ``get`` requires the
    caller to specify the category. (This distribution ships only the ``native`` category.)
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Perturbation] = {}

    def register(self, perturbation: Perturbation, *, replace: bool = False) -> Perturbation:
        if perturbation.category not in CATEGORIES:
            raise ValueError(
                f"Perturbation '{perturbation.name}' has unknown category "
                f"'{perturbation.category}'; expected one of {CATEGORIES}"
            )
        if perturbation.family not in FAMILIES:
            raise ValueError(
                f"Perturbation '{perturbation.name}' has unknown family "
                f"'{perturbation.family}'; expected one of {FAMILIES}"
            )
        key = (perturbation.category, perturbation.name)
        if key in self._items and not replace:
            raise ValueError(
                f"Perturbation '{perturbation.name}' is already registered in category "
                f"'{perturbation.category}'"
            )
        self._items[key] = perturbation
        return perturbation

    def register_all(self, perturbations, *, replace: bool = False) -> None:
        """Register every perturbation in ``perturbations`` (a convenience over ``register``)."""
        for perturbation in perturbations:
            self.register(perturbation, replace=replace)

    def get(self, name: str, category: Optional[str] = None) -> Perturbation:
        """Resolve a perturbation by ``name``, optionally scoped to ``category``.

        With a ``category`` the lookup is exact. Without one, a bare name resolves when
        it is registered in exactly one category; if it exists in several, a
        ``KeyError`` asks the caller to disambiguate.
        """
        if category is not None:
            try:
                return self._items[(category, name)]
            except KeyError:
                raise KeyError(
                    f"Unknown perturbation '{name}' in category '{category}'. "
                    f"Registered: {sorted(self.names())}"
                ) from None
        matches = [p for (cat, n), p in self._items.items() if n == name]
        if not matches:
            raise KeyError(
                f"Unknown perturbation '{name}'. Registered: {sorted(self.names())}"
            )
        if len(matches) > 1:
            cats = sorted(p.category for p in matches)
            raise KeyError(
                f"Perturbation '{name}' is ambiguous across categories {cats}; "
                f"specify a category"
            )
        return matches[0]

    def __contains__(self, key: object) -> bool:
        # Accept either a bare name or a (category, name) pair.
        if isinstance(key, tuple):
            return key in self._items
        return any(n == key for (_cat, n) in self._items)

    def __iter__(self):
        return iter(self._items.values())

    def names(self) -> list[str]:
        # Distinct names across all categories, preserving first-seen order.
        seen: dict[str, None] = {}
        for _cat, name in self._items:
            seen.setdefault(name, None)
        return list(seen)

    def categories(self) -> list[str]:
        """Categories that currently have at least one registered perturbation."""
        return [c for c in CATEGORIES if any(p.category == c for p in self._items.values())]

    def by_category(self, category: Optional[str] = None) -> dict[str, list[Perturbation]]:
        grouped: dict[str, list[Perturbation]] = {}
        for p in self._items.values():
            if category is not None and p.category != category:
                continue
            grouped.setdefault(p.category, []).append(p)
        return grouped


# The process-wide registry. Populated by ``perturber``'s package import with the native
# perturbations (see ``perturber.perturbations.register_native``).
REGISTRY = Registry()
