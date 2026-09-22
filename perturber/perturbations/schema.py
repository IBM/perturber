"""JSON Schema perturbations (native tier).

Perturbations whose input is a JSON Schema, the exact value a caller puts in a chat-completions
request's ``response_format.json_schema.schema`` field, and whose output is a perturbed schema that
is a valid drop-in for the same field. They exist for constrained-decoding robustness studies: a
decoding engine (for example vLLM's OpenAI-compatible server, or Ollama) compiles the schema to a
grammar and forces the model's output to conform, so validity is guaranteed and the question is
whether the model's *content* changes when the schema is varied in ways that keep the accepted
document set the same (or nearly so).

The highest-signal transforms reorder parts of the schema: constrained decoders emit object keys in
``properties`` order, so permuting them changes the generation order (and therefore what each field
is conditioned on) with no change to the set of valid documents. Others rewrite the schema into an
equivalent but differently-compiled grammar (expanding type shorthand, inlining ``$ref``\\ s, adding
tautological constraints).

Contract: the input string must parse (via ``json``) to a JSON object; otherwise the perturbation
returns it unchanged (a graceful no-op, never an error). Output is ``json.dumps`` of the perturbed
schema with key order preserved. Reorderings draw from the per-call ``rng`` and are seed-sensitive;
the structural rewrites are deterministic.

Scope: ``perturber`` only transforms the schema string. Extracting it from and re-inserting it into
a request envelope (``response_format`` for vLLM/OpenAI, ``format`` for Ollama) is the caller's job,
so the same perturbations serve every backend.
"""

from __future__ import annotations

import json
import random
from typing import Callable

from ..registry import CATEGORY_NATIVE, FAMILY_SCHEMA, Perturbation, Registry

# JSON Schema keywords whose values are themselves subschemas (recurse into these).
_SUBSCHEMA_KEYS = ("items", "additionalItems", "contains", "additionalProperties", "not",
                   "if", "then", "else", "propertyNames")
# Keywords whose value is a mapping of name to subschema.
_SUBSCHEMA_MAP_KEYS = ("properties", "patternProperties", "$defs", "definitions")
# Keywords whose value is a list of subschemas.
_SUBSCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")


def _load(text: str):
    """Parse ``text`` as JSON, returning the object or ``None`` if it is not a JSON object."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _detect_indent(text: str):
    """Infer the input's pretty-print indent so the output keeps the same formatting.

    Returns the indent argument for ``json.dumps``: an int (number of spaces) or a string (e.g. a
    tab) when the input was indented, or ``None`` when it was compact (single line, or no leading
    whitespace on its second line). This keeps an indented schema indented and a minified one
    minified, rather than normalizing every schema to one style.
    """
    for line in text.split("\n")[1:]:
        stripped = line.lstrip(" \t")
        if stripped:  # first non-blank line after the first
            lead = line[: len(line) - len(stripped)]
            if not lead:
                return None  # second line starts at column 0 => not pretty-printed
            if "\t" in lead:
                return "\t"
            return len(lead)
    return None  # single line => compact


def _dump(schema, indent=None) -> str:
    """Serialize a schema back to a string, preserving key order (never sort keys).

    ``indent`` mirrors the input's formatting (see :func:`_detect_indent`). Compact output uses the
    tight ``(",", ":")`` separators so a minified input round-trips minified.
    """
    if indent is None:
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(schema, ensure_ascii=False, indent=indent)


def _walk(node, fn: Callable[[dict, random.Random], dict], rng: random.Random):
    """Recursively apply ``fn`` to every subschema (object) node, bottom-up.

    Children are transformed first so a node-level ``fn`` sees already-processed subschemas. Lists
    and mappings that hold subschemas are traversed; other values are left untouched.
    """
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in _SUBSCHEMA_MAP_KEYS and isinstance(value, dict):
                node[key] = {k: _walk(v, fn, rng) for k, v in value.items()}
            elif key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
                node[key] = [_walk(v, fn, rng) for v in value]
            elif key in _SUBSCHEMA_KEYS:
                node[key] = _walk(value, fn, rng)
        return fn(node, rng)
    if isinstance(node, list):
        return [_walk(v, fn, rng) for v in node]
    return node


def _apply(text: str, node_fn: Callable[[dict, random.Random], dict], rng: random.Random) -> str:
    """Shared driver: parse, walk applying ``node_fn`` to each object, re-serialize.

    Output formatting mirrors the input: an indented schema stays indented, a minified one stays
    minified (see :func:`_detect_indent`).
    """
    schema = _load(text)
    if schema is None:
        return text
    return _dump(_walk(schema, node_fn, rng), _detect_indent(text))


# --- reorderings (seed-sensitive; the accepted document set is unchanged) ---

def _reorder_properties(text: str, params: dict, rng: random.Random) -> str:
    """Permute the key order of every ``properties`` object (and its ``required`` array).

    Constrained decoders emit keys in ``properties`` order, so this changes the generation order
    while accepting exactly the same documents.
    """
    def fn(node: dict, rng: random.Random) -> dict:
        props = node.get("properties")
        if isinstance(props, dict) and len(props) > 1:
            keys = rng.sample(list(props), len(props))
            node["properties"] = {k: props[k] for k in keys}
        required = node.get("required")
        if isinstance(required, list) and len(required) > 1:
            node["required"] = rng.sample(required, len(required))
        return node

    return _apply(text, fn, rng)


def _reorder_enum(text: str, params: dict, rng: random.Random) -> str:
    """Permute the order of every ``enum`` value list."""
    def fn(node: dict, rng: random.Random) -> dict:
        values = node.get("enum")
        if isinstance(values, list) and len(values) > 1:
            node["enum"] = rng.sample(values, len(values))
        return node

    return _apply(text, fn, rng)


def _reorder_union(text: str, params: dict, rng: random.Random) -> str:
    """Permute the branch order of every ``anyOf`` / ``oneOf`` / ``allOf``."""
    def fn(node: dict, rng: random.Random) -> dict:
        for key in ("anyOf", "oneOf", "allOf"):
            branches = node.get(key)
            if isinstance(branches, list) and len(branches) > 1:
                node[key] = rng.sample(branches, len(branches))
        return node

    return _apply(text, fn, rng)


# --- structural rewrites (deterministic; equivalent accepted set, different compiled grammar) ---

def _expand_type_shorthand(text: str, params: dict, rng: random.Random) -> str:
    """Rewrite ``"type": [..]`` list shorthand as an ``anyOf`` of single-type subschemas.

    ``{"type": ["string", "null"]}`` becomes ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``,
    the same accepted set expressed through a different grammar construct.
    """
    def fn(node: dict, rng: random.Random) -> dict:
        type_value = node.get("type")
        # Only expand a plain list of types on a node that is not already a union, and only when
        # no other constraints are attached to the shared node (keep the rewrite meaning-preserving).
        if (isinstance(type_value, list) and len(type_value) > 1
                and not any(k in node for k in ("anyOf", "oneOf", "allOf"))):
            del node["type"]
            node["anyOf"] = [{"type": t} for t in type_value]
        return node

    return _apply(text, fn, rng)


def _inline_defs(text: str, params: dict, rng: random.Random) -> str:
    """Inline every local ``$ref`` with a copy of its target, then drop the unused definitions.

    Only local refs into ``$defs`` / ``definitions`` are inlined. Refs that cannot be resolved (for
    example remote URLs, or a recursive ref that would not terminate) are left untouched.
    """
    schema = _load(text)
    if schema is None:
        return text

    defs = {}
    for container in ("$defs", "definitions"):
        if isinstance(schema.get(container), dict):
            for name, sub in schema[container].items():
                defs[f"#/{container}/{name}"] = sub

    if not defs:
        return text

    def resolve(node, seen: frozenset):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref in defs and ref not in seen:
                # Inline a deep copy of the target, resolving nested refs (guarding against cycles).
                target = json.loads(json.dumps(defs[ref]))
                return resolve(target, seen | {ref})
            return {k: resolve(v, seen) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    inlined = {k: resolve(v, frozenset()) for k, v in schema.items()
               if k not in ("$defs", "definitions")}
    # If any unresolved ref into the defs remains (e.g. a cycle), keep the defs to stay valid.
    remaining = json.dumps(inlined)
    if any(ref in remaining for ref in defs):
        return text
    return _dump(inlined, _detect_indent(text))


def _add_redundant_constraints(text: str, params: dict, rng: random.Random) -> str:
    """Add tautological keywords that do not narrow the accepted set.

    Adds ``"minItems": 0`` to array schemas and ``"minProperties": 0`` to object schemas when
    absent. These are always satisfied, so the set of valid documents is unchanged; only the
    grammar text differs.
    """
    def fn(node: dict, rng: random.Random) -> dict:
        node_type = node.get("type")
        if node_type == "array" and "minItems" not in node:
            node["minItems"] = 0
        if node_type == "object" and "minProperties" not in node:
            node["minProperties"] = 0
        return node

    return _apply(text, fn, rng)


# --- near-equivalent (narrows the accepted set): out of scope for a strict study ---

def _additional_properties_false(text: str, params: dict, rng: random.Random) -> str:
    """Set ``additionalProperties: false`` on object schemas that omit it.

    This closes open objects, which narrows the accepted document set, so it is *not* strictly
    equivalent and is marked out of scope for the strict robustness set.
    """
    def fn(node: dict, rng: random.Random) -> dict:
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        return node

    return _apply(text, fn, rng)


_PERTURBATIONS = (
    Perturbation("reorder_properties", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Permute the key order of every 'properties' object (and 'required').",
                 _reorder_properties, params=(), seed_sensitive=True),
    Perturbation("reorder_enum", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Permute the order of every 'enum' value list.",
                 _reorder_enum, params=(), seed_sensitive=True),
    Perturbation("reorder_union", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Permute the branch order of every anyOf/oneOf/allOf.",
                 _reorder_union, params=(), seed_sensitive=True),
    Perturbation("expand_type_shorthand", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Rewrite a 'type' list as an equivalent anyOf of single-type subschemas.",
                 _expand_type_shorthand, params=(), seed_sensitive=False),
    Perturbation("inline_defs", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Inline local $ref definitions, dropping the unused $defs.",
                 _inline_defs, params=(), seed_sensitive=False),
    Perturbation("add_redundant_constraints", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Add tautological keywords (minItems/minProperties 0) that do not narrow the set.",
                 _add_redundant_constraints, params=(), seed_sensitive=False),
    # Closes open objects, which narrows the accepted document set, so it is not strictly
    # equivalent and is out of scope for the strict robustness set: in_scope=False.
    Perturbation("additional_properties_false", CATEGORY_NATIVE, FAMILY_SCHEMA,
                 "Set additionalProperties:false on object schemas that omit it.",
                 _additional_properties_false, params=(), seed_sensitive=False, in_scope=False),
)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
