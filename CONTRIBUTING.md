# Contributing

## Setup

```bash
uv sync --all-packages      # creates .venv with both projects and dev tools
uv run pre-commit install   # ruff + mypy + hygiene hooks on commit
```

## Workflow

* `make check` runs exactly what CI runs: `ruff check`, `ruff format --check`, `mypy` (strict)
  and `pytest` with an 85 % coverage gate.
* Keep changes scoped to one project where possible; shared tooling lives only in the root
  `pyproject.toml`.
* Any change that alters model outputs or extraction logic must regenerate the committed
  artefacts with `make artifacts` and include them in the same commit, so reviewers can diff
  the report, not just the code.
* Public functions and classes carry Google-style docstrings; formulas go in the docstring of
  the function that implements them and in the project's `docs/METHODOLOGY.md`.
* Tests live next to each project under `tests/`. Prefer small fixtures over large data files;
  anything above 1 MB is rejected by the pre-commit hook.

## Commit messages

Conventional style, present tense, scoped by project when relevant:

```
rag: reject rounded values in the grounding check
pd: add Jeffreys test to calibration section
ci: run tests on 3.13
```
