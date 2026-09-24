# Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface.

Applies one perturbation per invocation:

    perturber <name> [--<param> <value> ...] [--text TEXT] [--seed N]

The input string is taken from ``--text`` or, if that is omitted, from standard input (a
single trailing newline is stripped). The result is written to standard output verbatim,
with no added newline, so that whitespace-manipulating perturbations round-trip exactly and
invocations can be piped to compose:

    echo "What is the capital of France?" | perturber typos --seed 0 | perturber drop_stop_words --seed 0

``perturber list`` prints the available perturbations. ``perturber-serve`` launches the API.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .core import DEFAULT_SEED, list_perturbations, perturb
from .registry import _PY_TYPES, REGISTRY, Perturbation

# argparse coerces via a callable; ``bool`` is omitted because argparse ``type=bool`` treats any
# non-empty string as True. Boolean params are parsed from their string value elsewhere.
_ARGPARSE_TYPES = {name: py for name, py in _PY_TYPES.items() if py is not bool}


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    data = sys.stdin.read()
    if data.endswith("\n"):
        data = data[:-1]
    return data


def _add_perturbation_parser(subparsers, name: str, variants: list[Perturbation]) -> None:
    """Add one subcommand for ``name``, covering all its per-category variants.

    A name may exist in more than one category. A single ``--category`` option selects the
    variant; it defaults to the sole category when the name is unique and is required when it
    is not.
    """
    categories = sorted({p.category for p in variants})
    representative = variants[0]
    cat_hint = "/".join(f"{p.category}" for p in variants)
    parser = subparsers.add_parser(
        name,
        help=representative.description,
        description=f"[{cat_hint}] {representative.description}",
    )
    parser.add_argument("--text", default=None, help="Input text (default: read from stdin).")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--category",
        choices=categories,
        default=(categories[0] if len(categories) == 1 else None),
        required=len(categories) > 1,
        help="Provenance category" + ("" if len(categories) == 1 else " (required; name exists in several)"),
    )
    # Declare the union of parameters across variants (they share names/types in practice).
    seen_params: set[str] = set()
    for perturbation in variants:
        for spec in perturbation.params:
            if spec.name in seen_params:
                continue
            seen_params.add(spec.name)
            parser.add_argument(
                f"--{spec.name}",
                type=_ARGPARSE_TYPES.get(spec.type, str),
                default=None,
                help=f"{spec.description} (default: {spec.default})",
            )
    parser.set_defaults(_name=name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perturber",
        description="Apply a semantics-preserving perturbation to text (one per invocation).",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<perturbation>|list")

    list_parser = subparsers.add_parser("list", help="List available perturbations.")
    list_parser.add_argument(
        "--category", default=None, help="Restrict the listing to one category."
    )

    # Group registered perturbations by name so a name shared across categories gets one
    # subcommand with a --category selector rather than a duplicate parser.
    by_name: dict[str, list[Perturbation]] = {}
    for perturbation in REGISTRY:
        by_name.setdefault(perturbation.name, []).append(perturbation)
    for name, variants in by_name.items():
        _add_perturbation_parser(subparsers, name, variants)

    return parser


def _print_list(category: Optional[str]) -> None:
    for entry in list_perturbations(category):
        params = ", ".join(
            f"{name}={spec['default']}" for name, spec in entry["params"].items()
        )
        params = f" ({params})" if params else ""
        print(f"{entry['category']:12} {entry['family']:12} {entry['name']}{params}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "list":
        _print_list(args.category)
        return 0

    perturbation: Perturbation = REGISTRY.get(args._name, args.category)
    params = {spec.name: getattr(args, spec.name) for spec in perturbation.params}
    text = _read_text(args)
    # Model-backed perturbations read the API key and base URL from the environment
    # (OPENAI_API_KEY / OPENAI_BASE_URL); llm.py resolves those directly, so no transport
    # context is needed from the CLI.
    result = perturb(
        text,
        perturbation.name,
        params=params,
        seed=args.seed,
        category=perturbation.category,
    )
    sys.stdout.write(result)
    return 0


def serve_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``perturber-serve``: launch the API with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="perturber-serve", description="Launch the perturber API.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development).")
    args = parser.parse_args(argv)

    uvicorn.run("perturber.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
