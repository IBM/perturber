# Contributing

Thank you for your interest in perturber. This project is maintained by a small team, and we
keep the contribution process deliberately light.

## Scope

We are not accepting unsolicited feature contributions at this time. The project has a focused
design (one primitive: apply one named perturbation to one string, under a seed), and we prefer
to keep its surface small. If you would like to propose a change, please open an issue first to
discuss it before writing any code, so we can tell you whether it fits the roadmap.

We do welcome:

- **Bug reports**, with a minimal reproduction (the input string, perturbation name, params, and
  seed, plus the expected and actual output).
- **Documentation fixes**.

## Reporting a bug

Open an issue describing the problem. For a perturbation that produces the wrong output, include
the exact `perturb(...)` call or `curl` request and its output, so we can reproduce it
deterministically.

## Proposing a change

Please raise an issue before sending a pull request, whether it is a new feature or a bug fix, so
the change can be discussed and tracked. We appreciate your effort and want to avoid a situation
where a contribution needs extensive rework, sits in the backlog, or cannot be accepted at all.

## Setup

```bash
pip install -e '.[dev,llm]'
```

The native `synonym` perturbation needs the WordNet corpus and the perceptron POS tagger; both
download automatically on first use.

## Testing and style

Run the linter before opening a pull request:

```bash
ruff check perturber
```

Match the style of the surrounding code: keep changes small and focused, and follow the existing
naming and comment conventions rather than introducing new ones.

Perturbations are expected to be deterministic for a fixed `(text, name, params, seed)`. If you
change one, confirm that the same inputs still produce the same output. The model-backed
perturbations are the exception and are not reproducible.

## Submitting a change

1. Fork the repository and create a branch from `main`.
2. Sign off your commits (see [Legal](#legal) below).
3. Run the linter.
4. Open the pull request against `main` and describe what changed and why.

### Merge approval

The maintainers use LGTM (Looks Good To Me) in review comments to indicate acceptance. A change
needs an LGTM from a maintainer of each affected component before it is merged.

## Legal

Each source file must include a license header for the Apache Software License 2.0. Using the
SPDX format is the simplest approach, e.g.

```
# Copyright <holder> All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
```

We use the [Developer's Certificate of Origin 1.1 (DCO)](https://developercertificate.org/) — the
same approach the Linux® Kernel
[community](https://elinux.org/Developer_Certificate_Of_Origin) uses — to manage code
contributions. By signing off on your commits you certify that you wrote the change, or otherwise
have the right to submit it under the project's license.

When submitting a patch for review, include a sign-off statement in the commit message:

```
Signed-off-by: John Doe <john.doe@example.com>
```

Git adds this trailer for you if you commit with `-s`:

```bash
git commit -s -m "your message"
```

Pull requests without a sign-off will fail the DCO check.

By contributing, you agree that your contributions are licensed under the project's
[Apache-2.0](LICENSE) license.
