# finrisk-ai-lab

Two production-style projects at the intersection of **LLM engineering** and **credit / prudential
risk**, built to the standard a bank's model-risk or AI-engineering team would review against:
typed code, strict linting, 85 %+ test coverage, reproducible committed artefacts, CI on three
Python versions.

| Project | One line | Stack |
|---|---|---|
| [`projects/annual-report-risk-rag`](projects/annual-report-risk-rag) | Extract CET1, LCR, NPL ratio, ECL provisions, VaR ... from annual reports with hybrid retrieval + Claude structured outputs, and **refuse any number that cannot be traced to a verbatim sentence** | Python, Anthropic SDK, Pydantic, NumPy, BM25 + dense RRF, Typer |
| [`projects/ifrs9-pd-model`](projects/ifrs9-pd-model) | IFRS 9 / AASB 9 probability-of-default model (WoE scorecard, PIT/TTC, lifetime term structure, SICR staging, ECL) with an **independent validation suite that writes the model-validation report** | Python, pandas, scikit-learn, SciPy, matplotlib, Typer |

## Headline results (from the committed artefacts)

| | Metric | Value |
|---|---|---|
| RAG baseline | Micro-F1 vs gold labels, 2 synthetic reports x 12 metrics | 0.947 (precision 1.00) |
| RAG baseline | Grounding rate of accepted values | 100 % |
| IFRS 9 PD | Gini, development / out-of-time | 0.739 / 0.723 |
| IFRS 9 PD | KS, development / out-of-time | 0.593 / 0.577 |
| IFRS 9 PD | Score PSI (dev vs OOT) | 0.004 |
| IFRS 9 PD | Stage 1 / 2 / 3 share of exposure | 94.0 % / 5.6 % / 0.3 % of loans, ECL coverage 1.02 % |
| IFRS 9 PD | Out-of-time calibration (HL p-value, mean PD / observed DR) | 0.003 / 0.76, flagged **RED** by the validation suite |

Reports: [`eval_rules.md`](projects/annual-report-risk-rag/reports/eval_rules.md) and
[`validation_report.md`](projects/ifrs9-pd-model/reports/validation_report.md).

The committed IFRS 9 report deliberately rates the model **RED**: discrimination and stability
are green, but the out-of-time window shows under-prediction, and the suite turns that into a
high-severity finding with a recalibration recommendation. That is the point of an independent
validation layer; the project README explains the root cause and shows that other seeds rate
green with the same code.

## Repository layout

```
finrisk-ai-lab/
├── pyproject.toml            # uv workspace + shared ruff / mypy / pytest / coverage config
├── Makefile                  # make check | test | lint | typecheck | artifacts
├── .github/workflows/ci.yml  # lint + strict mypy, tests on 3.11/3.12/3.13, artefact reproduction
├── .pre-commit-config.yaml
├── projects/
│   ├── annual-report-risk-rag/
│   │   ├── src/report_rag/   # ingest, retrieval, extraction, llm, agent, evaluation, cli
│   │   ├── data/             # synthetic reports + gold labels
│   │   ├── reports/          # committed evaluation output
│   │   ├── docs/ tests/ README.md
│   └── ifrs9-pd-model/
│       ├── src/ifrs9_pd/     # data, features, model, validation, reporting, pipeline, cli
│       ├── reports/          # committed validation report, JSON metrics, figures, scorecard
│       ├── docs/ tests/ README.md
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## Quickstart

```bash
git clone https://github.com/RiverHe2000/finrisk-ai-lab && cd finrisk-ai-lab
uv sync --all-packages           # Python >= 3.11, uv >= 0.5
make check                       # ruff + mypy --strict + pytest (coverage gate 85 %)

make rag-demo                    # offline extraction over the bundled reports
make pd-pipeline                 # generate data -> train -> validate -> write report
```

The LLM-backed extractor and the agent need `ANTHROPIC_API_KEY`; everything else, including
CI, runs offline.

## Engineering standards

* **Typing**: `mypy --strict` over both packages, `pandas-stubs` / `scipy-stubs` installed.
* **Linting**: ruff with pycodestyle, pyflakes, isort, bugbear, pyupgrade, pep8-naming, pydocstyle
  (Google), annotations, bandit, pylint subsets. Line length 100.
* **Tests**: pytest, deterministic seeds, no network. Coverage gate enforced in CI.
* **Reproducibility**: every committed report is regenerated in CI and uploaded as an artefact;
  the IFRS 9 report is byte-identical across runs (fixed seed, no timestamps).
* **Boundaries**: LLM providers, embedders and extractors sit behind small protocols and are
  swapped in tests with stubs; no test touches the network.

## About

Built by Chuan He, PhD candidate in Computer Science (LLMs) at UNSW, with a B.Sc./M.Sc. in
Financial Engineering (risk management). The two projects are meant to be read together: the
first shows how I build LLM systems that a risk function can audit, the second shows that I
understand the models those functions own.
