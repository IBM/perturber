# perturber

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A small, stateless service and library that applies semantics-preserving perturbations to
text. The caller sends a string, a perturbation name, its parameters, and a seed; the service
returns the perturbed string. Everything downstream, which model consumes the string, how it
is evaluated, is the caller's responsibility. `perturber` only transforms text.

The service has exactly one primitive: **apply one named perturbation to one string, under a
seed.** There is no stack grammar; to apply several perturbations in sequence, chain the calls.

All perturbations are permissively licensed, first-party implementations. Most are deterministic
string transforms with no model dependency; the model-backed ones call any OpenAI-compatible
`/chat/completions` endpoint you configure per request.

## Citation

If you use `perturber` in your work, please cite it:

```bibtex
@software{tran2026perturber,
  title   = {perturber: A Stateless Service for Semantics-Preserving Text Perturbations},
  author  = {Tran, Khoi-Nguyen},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/IBM/perturber},
  license = {Apache-2.0}
}
```

## Example

```python
from perturber import perturb

perturb("hello world", "title_case", seed=0)
# -> 'Hello World'

perturb("hello world", "word_merge", seed=0)
# -> 'helloworld'
```

Composition is the caller chaining calls, so it is order-dependent:

```python
text = perturb("hello world", "title_case", seed=0)
text = perturb(text, "word_merge", seed=0)        # -> 'HelloWorld'

text = perturb("hello world", "word_merge", seed=0)
text = perturb(text, "title_case", seed=0)        # -> 'Helloworld'
```

For a fixed `(text, name, params, seed)`, a deterministic perturbation always returns the same
string, so both chains above are reproducible. The library and the `POST /perturb` route produce
byte-for-byte identical output, so a chain gives the same result whichever interface runs it.

## Install

```bash
pip install .            # library + CLI + service (core perturbations)
pip install '.[llm]'     # also enable the model-backed perturbations (adds the openai client)
pip install '.[dev]'     # lint tooling
```

The native `synonym` perturbation uses WordNet and the perceptron POS tagger; both auto-download
on first use if they are not already present.

## Library

```python
from perturber import perturb, list_perturbations

perturb("The quick brown fox.", "realistic_typos", seed=0)
perturb('{"type":"object"}', "reorder_properties", category="native", seed=0)

# Discover what is available.
for entry in list_perturbations():
    print(entry["category"], entry["name"], entry["family"], entry["in_scope"])
```

`perturb` is a pure function of `(text, name, params, seed)` for the deterministic perturbations.
The model-backed perturbations are best-effort and non-deterministic.

## Service

```bash
perturber-serve                      # or: uvicorn perturber.api:app --port 8080
```

Every perturbation gets its own typed route, grouped by family in the OpenAPI docs
(`/docs`). An interactive playground is served at `/`.

- `POST /perturb/native/{name}`  apply a perturbation to a list of texts
- `GET  /perturbations[/native]` the catalog (family, params, `in_scope`, seed-sensitivity)
- `GET  /health`, `GET /version`

```bash
curl -s localhost:8080/perturb/native/realistic_typos \
  -H 'content-type: application/json' \
  -d '{"texts":["The quick brown fox."],"seed":0}'
```

## CLI

```bash
perturber realistic_typos --text "The quick brown fox." --seed 0
echo "The quick brown fox." | perturber random_case
perturber list
```

Output is written to stdout verbatim (no trailing newline) so round-trips are exact.

## Model-backed perturbations

Some perturbations (the `paraphrasing` family, `emotion_prompt`, and the `reasoning` probes)
call a language model through any OpenAI-compatible `/chat/completions` API. They carry no
secret in the service; credentials and the endpoint are supplied per request.

- **API key** (required): sent as the `Authorization: Bearer <key>` request header (the
  `X-LLM-API-Key` header is also accepted). The CLI and library fall back to the
  `OPENAI_API_KEY` environment variable. The key never enters the request body, the echoed
  `params`, or the response.
- **Base URL** (optional): sent as the `X-LLM-Base-URL` request header, used verbatim. When
  absent it falls back to `OPENAI_BASE_URL` (default `https://api.openai.com/v1`).
- **Model**: the `model` parameter selects the model per request. Its dropdown choices come
  from a static list, configurable via the `PERTURBER_LLM_MODELS` environment variable
  (comma-separated; the first entry is the default). Any model id the endpoint serves may be
  supplied.

```bash
curl -s localhost:8080/perturb/native/lexical_paraphrasing \
  -H 'content-type: application/json' \
  -H 'Authorization: Bearer YOUR_KEY' \
  -H 'X-LLM-Base-URL: https://api.openai.com/v1' \
  -d '{"texts":["The mitochondrion produces most of the cell's energy."],"model":"gpt-4o-mini"}'
```

## The `in_scope` flag

Each perturbation declares whether it is a natural, semantics-preserving surface transformation
(`in_scope=True`) or falls outside that definition (`in_scope=False`, for example the
prompt-injection probes and the content-adding reasoning probes). All perturbations stay
callable; the playground hides the out-of-scope ones from its dropdown, and a benchmark harness
can filter on the flag. The registry is the single source of truth; the flag is exposed in the
catalog.

## Docker

```bash
docker build -t perturber .
docker run -p 8080:8080 perturber
# model-backed perturbations: pass credentials per request, or bake defaults into the env:
docker run -p 8080:8080 -e OPENAI_API_KEY=... -e OPENAI_BASE_URL=... perturber
```

## Development

```bash
pip install -e '.[dev,llm]'
ruff check perturber
```

## License

Apache-2.0. See `LICENSE`. Contributions are governed by `CONTRIBUTING.md`.
