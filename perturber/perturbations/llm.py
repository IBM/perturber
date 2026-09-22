"""Model-backed perturbations (native tier), served via a generic OpenAI-compatible endpoint.

These are perturbations that require a language model rather than a deterministic string
transform. They include the paraphrasing family the robustness literature defines
(``lexical_paraphrasing``, ``syntactic_paraphrasing``, ``rule_free_paraphrasing``) and the
``emotion_prompt`` context augmentation, plus first-party additions: paraphrase-style rewrites
(``register_shift`` and ``translate_roundtrip`` are param-driven, taking a ``register`` /
``language`` choice substituted into the prompt template; ``simplification``, ``verbose_expansion``,
``typo_paraphrase``, ``persona_paraphrase``) and a ``reasoning`` family of robustness probes
(``reasoning_restructure``, ``negation_paraphrase``, ``distractor_injection``, ``sycophancy_bait``;
plus ``elaborate_then_ask`` in the context family). The content-adding probes are ``in_scope=False``.
Each calls a chat model over any OpenAI-compatible ``/chat/completions`` API using the ``openai``
client, with the key supplied as the standard ``Authorization: Bearer`` credential.

Determinism: model output is not bit-reproducible, so these perturbations are registered as *not*
seed-sensitive. The ``temperature`` parameter defaults to 0.7; set it to 0 for the most stable
output. Results are not guaranteed identical across runs.

Model: the ``model`` parameter (default ``gpt-4o-mini``) selects the model per request. Its choices
are a static, configurable list (see ``PERTURBER_LLM_MODELS``); any model id the endpoint serves may
be supplied.

Credentials: the API key is a per-request secret, never baked into the service. It arrives via the
request ``context`` (the API reads the ``Authorization`` / ``X-LLM-API-Key`` header; the CLI/library
fall back to the ``OPENAI_API_KEY`` environment variable). It is never persisted or echoed.

Endpoint configuration is per request (bring-your-own): the caller may supply a base URL via the
request ``context`` (the ``X-LLM-Base-URL`` header). When absent it falls back to the environment:

- ``OPENAI_BASE_URL`` (default: ``https://api.openai.com/v1``)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..registry import (
    CATEGORY_NATIVE,
    FAMILY_CONTEXT,
    FAMILY_PARAPHRASING,
    FAMILY_REASONING,
    ParamSpec,
    Perturbation,
    Registry,
)

logger = logging.getLogger("perturber.llm")

# Timeout (seconds) for any LLM HTTP call, so a slow or hung endpoint cannot block a request
# (and its server worker) indefinitely.
_LLM_TIMEOUT_SECONDS = 30

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"

# Model identifiers offered as choices for the ``model`` parameter. This is a static default list;
# an operator can override it with the ``PERTURBER_LLM_MODELS`` environment variable (a
# comma-separated list, first entry becoming the default). Any model id the configured endpoint
# serves may be supplied even if it is not listed here.
def _model_choices() -> tuple[str, ...]:
    raw = os.environ.get("PERTURBER_LLM_MODELS", "").strip()
    if raw:
        choices = tuple(m.strip() for m in raw.split(",") if m.strip())
        if choices:
            return choices
    return (
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    )


_MODEL_CHOICES = _model_choices()
_DEFAULT_MODEL = _MODEL_CHOICES[0]

# Shared guard for the *reword* perturbations: a capable model, given a solvable question stem,
# will otherwise answer/solve it instead of rewording it, changing the task and inflating downstream
# accuracy (see issue #1). Appended to every rewrite prompt via _PRESERVE_AS_QUESTION_NAMES below.
# NOT added to the prepend/add perturbations (emotion_prompt, sycophancy_bait, elaborate_then_ask,
# distractor_injection), which reproduce the original text verbatim and so cannot leak an answer, nor
# to reasoning_restructure, whose prompt already carries an equivalent "preserve the answer" clause.
_PRESERVE_CLAUSE = (
    " CRITICAL: the text is a question or task to be preserved, NOT answered. Do NOT answer, solve, "
    "compute, complete, or evaluate it; if it is a question it must remain the same question."
)
_PRESERVE_AS_QUESTION_NAMES = frozenset({
    "lexical_paraphrasing", "syntactic_paraphrasing", "rule_free_paraphrasing", "simplification",
    "verbose_expansion", "code_switch", "register_shift", "translate_roundtrip", "typo_paraphrase",
    "persona_paraphrase", "negation_paraphrase",
})

# System instructions per perturbation. Each asks the model to preserve meaning and return only
# the transformed text (no preamble), so the output can be used verbatim.
_INSTRUCTIONS = {
    "lexical_paraphrasing": (
        "Paraphrase the user's text by substituting different words and synonyms while keeping "
        "the sentence structure and meaning the same. Return only the paraphrased text, with no "
        "preamble, quotes, or explanation."
    ),
    "syntactic_paraphrasing": (
        "Paraphrase the user's text by changing its grammatical structure (for example, reorder "
        "clauses or switch between active and passive voice) while keeping the wording and "
        "meaning as close as possible. Return only the paraphrased text, with no preamble, "
        "quotes, or explanation."
    ),
    "rule_free_paraphrasing": (
        "Paraphrase the user's text freely while preserving its meaning. Return only the "
        "paraphrased text, with no preamble, quotes, or explanation."
    ),
    "emotion_prompt": (
        "Prepend a short emotional-urgency phrase (for example expressing that this is very "
        "important to the user) to the user's text, then reproduce the original text verbatim "
        "after it. Return only the resulting text, with no preamble, quotes, or explanation."
    ),
    "simplification": (
        "Rewrite the user's text in simpler, plainer language that is easier to read, while "
        "preserving its meaning. Return only the rewritten text, with no preamble, quotes, or "
        "explanation."
    ),
    "verbose_expansion": (
        "Rewrite the user's text in a wordier, more roundabout way, adding filler and elaboration "
        "without adding new information, while preserving its meaning. Return only the rewritten "
        "text, with no preamble, quotes, or explanation."
    ),
    "code_switch": (
        "Rewrite the user's text so that some words and short phrases are naturally switched into "
        "Spanish while the rest stays in English (code-switching), preserving the overall meaning. "
        "Return only the rewritten text, with no preamble, quotes, or explanation."
    ),
    # --- Templated prompts: the {register}/{language} placeholder is filled from a param at apply
    # time (see _make_apply). The stored string is a template; the playground shows it verbatim.
    "register_shift": (
        "Rewrite the user's text in a {target_register} register while preserving its exact "
        "meaning. Return only the rewritten text, with no preamble, quotes, or explanation."
    ),
    "translate_roundtrip": (
        "Translate the user's text to {language}, then translate it back to English. Return only "
        "the final English text (the natural paraphrase that results), with no preamble, quotes, "
        "or explanation. A language more distant from English yields a stronger paraphrase."
    ),
    "typo_paraphrase": (
        "Rewrite the user's text as a hurried person typing quickly would produce it, introducing "
        "natural, realistic typos and misspellings while keeping the words and meaning "
        "recognisable. Return only the rewritten text, with no preamble, quotes, or explanation."
    ),
    "persona_paraphrase": (
        "Rewrite the user's text as if written by a specific person with their own voice and style "
        "(pick a plausible persona), while preserving the meaning. Return only the rewritten text, "
        "with no preamble, quotes, or explanation."
    ),
    # --- reasoning family: probe a distinct robustness axis; meaning/answer preserved but framing
    # or context changes. The content-adding ones are marked in_scope=False below.
    "reasoning_restructure": (
        "Restate the user's text so the same question or task is posed with its parts reordered or "
        "its framing inverted (for example, state the conditions before the question, or ask for "
        "the same result from the opposite direction). Preserve the exact task and answer. Return "
        "only the restated text, with no preamble, quotes, or explanation."
    ),
    "negation_paraphrase": (
        "Restate the user's text using DOUBLE negations that CANCEL OUT, so the restatement is "
        "logically equivalent to the original and asserts exactly the same thing. Each negation you "
        "add must be paired with a second negation that reverses it (for example, 'all are valid' "
        "becomes 'none are invalid'; 'the door is open' becomes 'the door is not closed'; 'she "
        "agreed' becomes 'she did not disagree'). CRITICAL: do NOT invert, flip, or contradict the "
        "meaning, and do NOT negate a question (never turn 'what is X' into 'what is not X'); the "
        "truth conditions must be identical to the original. If you cannot negate a part without "
        "changing its meaning, leave that part unchanged. Return only the restated text, with no "
        "preamble, quotes, or explanation."
    ),
    "distractor_injection": (
        "Add one plausible but irrelevant sentence to the user's text that does not change the task "
        "or its answer, keeping the original content intact. Return only the resulting text, with "
        "no preamble, quotes, or explanation."
    ),
    "sycophancy_bait": (
        "Prepend a short phrase stating the user's own (possibly mistaken) belief or preferred "
        "answer, then reproduce the original text verbatim after it. Do not change the original "
        "text. Return only the resulting text, with no preamble, quotes, or explanation."
    ),
    "elaborate_then_ask": (
        "Add a sentence or two of accurate, relevant background before the user's text, then "
        "reproduce the original text verbatim after it. Do not change the original text or its "
        "answer. Return only the resulting text, with no preamble, quotes, or explanation."
    ),
}

# Inject the preserve-as-question guard into the reword perturbations (issue #1). Inserted before the
# trailing "Return only ..." sentence so the guard reads naturally and the return instruction stays
# last. Every targeted prompt ends with that sentence, so the split is stable.
for _name in _PRESERVE_AS_QUESTION_NAMES:
    _base = _INSTRUCTIONS[_name]
    _marker = " Return only "
    _head, _sep, _tail = _base.partition(_marker)
    _INSTRUCTIONS[_name] = _head + _PRESERVE_CLAUSE + _sep + _tail

# Cache OpenAI clients by (base_url, api_key) so repeated calls reuse the connection.
_client_cache: dict = {}


def _default_base_url() -> str:
    """Return the OpenAI-compatible base URL configured in the environment (or the default)."""
    return os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)


def _resolve_key(context: Optional[dict]) -> str:
    """Return the API key from the request context or the environment."""
    key = (context or {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "An API key is required for this perturbation. Provide it via the request "
            "'Authorization: Bearer <key>' header (or the OPENAI_API_KEY environment variable)."
        )
    return key


def _openai_base_url(endpoint: Optional[str]) -> str:
    """Return the OpenAI-compatible base URL.

    ``endpoint`` is a caller-supplied base URL (bring-your-own, via the request context). It is
    used verbatim (trailing slash trimmed). When absent, fall back to the environment-configured
    base URL (``OPENAI_BASE_URL``, defaulting to the OpenAI public API).
    """
    if endpoint:
        return endpoint.rstrip("/")
    return _default_base_url()


def _client(base_url: str, key: str):
    """Build (and cache) an OpenAI-compatible client for ``base_url``."""
    cache_key = (base_url, key)
    client = _client_cache.get(cache_key)
    if client is None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ValueError(
                "The 'openai' package is required for model-backed perturbations; install "
                "perturber[llm]."
            ) from exc
        client = openai.OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
        _client_cache[cache_key] = client
    return client


def _chat(system: str, text: str, key: str, model: str, temperature: float,
          endpoint: Optional[str] = None) -> str:
    """Run one chat completion against the configured endpoint and return the assistant content."""
    client = _client(_openai_base_url(endpoint), key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            timeout=_LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - any client/network error becomes a clean message.
        # Log the underlying detail server-side; do not leak it (URLs, auth hints) to the caller.
        logger.warning("LLM chat completion failed: %s", exc)
        raise ValueError("The LLM request failed. Check the model, endpoint, and API key.") from exc
    # A well-formed response can still carry no choices (e.g. content filtering); treat as no change.
    if not completion.choices:
        return text
    content = completion.choices[0].message.content
    return content if content is not None else text


def _make_apply(name: str):
    template = _INSTRUCTIONS[name]

    def apply(text: str, params: dict, rng, *, context: Optional[dict] = None) -> str:
        if not text:
            return text
        key = _resolve_key(context)
        model = params.get("model") or _DEFAULT_MODEL
        temperature = params.get("temperature")
        if temperature is None:
            temperature = 0.0
        endpoint = (context or {}).get("base_url")
        # For templated perturbations, fill the {register}/{language} placeholder from its extra
        # param (falling back to that param's declared default). Non-templated prompts pass through.
        system = template
        for spec in _EXTRA_PARAMS.get(name, ()):
            value = params.get(spec.name)
            if value is None:
                value = spec.default
            system = system.replace("{" + spec.name + "}", str(value))
        return _chat(system, text, key, model, temperature, endpoint)

    return apply


_DESCRIPTIONS = {
    "lexical_paraphrasing": "Reword with different vocabulary, preserving meaning (LLM).",
    "syntactic_paraphrasing": "Restructure grammar/syntax, preserving meaning (LLM).",
    "rule_free_paraphrasing": "Free-form paraphrase, preserving meaning (LLM).",
    "emotion_prompt": "Prepend an emotional-urgency phrase to the text (LLM).",
    "register_shift": "Rewrite in a chosen register (formal/casual/academic/legal/child), preserving meaning (LLM).",
    "simplification": "Rewrite in simpler, plainer language, preserving meaning (LLM).",
    "verbose_expansion": "Rewrite in a wordier, more roundabout way, preserving meaning (LLM).",
    "translate_roundtrip": "Paraphrase by round-tripping through a chosen pivot language; more distant languages rewrite more (LLM).",
    "typo_paraphrase": "Rewrite with natural, LLM-generated human typos, keeping words recognisable (LLM).",
    "persona_paraphrase": "Rewrite in the voice of a plausible persona, preserving meaning (LLM).",
    "code_switch": "Rewrite mixing English and Spanish (code-switching), preserving meaning (LLM).",
    "reasoning_restructure": "Reorder or invert how the task is framed, preserving the task and answer (LLM).",
    "negation_paraphrase": "Restate using meaning-preserving negations/double-negatives (LLM).",
    "distractor_injection": "Add one plausible but irrelevant sentence, leaving the answer unchanged (LLM).",
    "sycophancy_bait": "Prepend a stated (possibly wrong) user belief before the text (LLM).",
    "elaborate_then_ask": "Prepend relevant background before the text, leaving the task unchanged (LLM).",
}

_FAMILIES = {
    "lexical_paraphrasing": FAMILY_PARAPHRASING,
    "syntactic_paraphrasing": FAMILY_PARAPHRASING,
    "rule_free_paraphrasing": FAMILY_PARAPHRASING,
    "emotion_prompt": FAMILY_CONTEXT,
    "register_shift": FAMILY_PARAPHRASING,
    "simplification": FAMILY_PARAPHRASING,
    "verbose_expansion": FAMILY_PARAPHRASING,
    "translate_roundtrip": FAMILY_PARAPHRASING,
    "typo_paraphrase": FAMILY_PARAPHRASING,
    "persona_paraphrase": FAMILY_PARAPHRASING,
    "code_switch": FAMILY_PARAPHRASING,
    "reasoning_restructure": FAMILY_REASONING,
    "negation_paraphrase": FAMILY_REASONING,
    "distractor_injection": FAMILY_REASONING,
    "sycophancy_bait": FAMILY_REASONING,
    "elaborate_then_ask": FAMILY_CONTEXT,
}


_MODEL_PARAM = ParamSpec(
    "model",
    "str",
    _DEFAULT_MODEL,
    "Model id to use (any id the configured OpenAI-compatible endpoint serves).",
    choices=_MODEL_CHOICES,
)
_TEMPERATURE_PARAM = ParamSpec(
    "temperature",
    "float",
    0.7,
    "Sampling temperature (0 to 2). 0 is most deterministic; higher is more varied.",
)

# Extra, per-perturbation params beyond model/temperature. The param's value is substituted into the
# perturbation's system-prompt template (a {register}/{language} placeholder; see _make_apply) so one
# perturbation covers a family of related rewrites.
_EXTRA_PARAMS = {
    "register_shift": (
        ParamSpec(
            "target_register", "str", "formal",
            "Target register to rewrite into.",
            choices=("formal", "casual", "academic", "legal", "child"),
        ),
    ),
    "translate_roundtrip": (
        ParamSpec(
            "language", "str", "French",
            "Pivot language for the round-trip; more distant languages rewrite more.",
            choices=("French", "German", "Japanese", "Chinese", "Arabic"),
        ),
    ),
}


# Model-based rewrites that are a heavy language-level transformation rather than natural
# in-language surface variation fall outside the robustness-perturbation definition: in_scope=False.
# code_switch is a heavy rewrite; the other three ADD content (an irrelevant sentence, a stated user
# belief, background) so they change the prompt rather than preserve its surface, and are probes
# rather than semantics-preserving surface transforms.
_OUT_OF_SCOPE = frozenset(
    {"code_switch", "distractor_injection", "sycophancy_bait", "elaborate_then_ask"}
)


def _perturbation(name: str) -> Perturbation:
    return Perturbation(
        name=name,
        category=CATEGORY_NATIVE,
        family=_FAMILIES[name],
        description=_DESCRIPTIONS[name],
        apply=_make_apply(name),
        params=(_MODEL_PARAM, _TEMPERATURE_PARAM) + _EXTRA_PARAMS.get(name, ()),
        # Model output is not reproducible; temperature is best-effort determinism only.
        seed_sensitive=False,
        prompt=_INSTRUCTIONS[name],
        in_scope=name not in _OUT_OF_SCOPE,
    )


# Registration/display order: the paper's three paraphrase styles first, then emotion_prompt (the
# paper's fourth model-based type), then this project's paraphrase rewrites, the reasoning-axis
# probes, and finally the out-of-scope rewrites. Any name not listed falls back to dict order.
_ORDER = (
    "lexical_paraphrasing",
    "syntactic_paraphrasing",
    "rule_free_paraphrasing",
    "emotion_prompt",
    "simplification",
    "register_shift",
    "verbose_expansion",
    "translate_roundtrip",
    "typo_paraphrase",
    "persona_paraphrase",
    "reasoning_restructure",
    "negation_paraphrase",
    "distractor_injection",
    "sycophancy_bait",
    "elaborate_then_ask",
    "code_switch",
)
_ordered_names = list(_ORDER) + [n for n in _INSTRUCTIONS if n not in _ORDER]
_PERTURBATIONS = tuple(_perturbation(n) for n in _ordered_names)


def register(registry: Registry) -> None:
    registry.register_all(_PERTURBATIONS)
