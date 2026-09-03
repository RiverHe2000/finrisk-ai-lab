.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test test-fast cov check clean \
        rag-demo rag-eval pd-pipeline pd-report artifacts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install every workspace member with dev tools
	$(UV) sync --all-packages
	$(UV) run pre-commit install

lint: ## Ruff lint (no changes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format and auto-fix with ruff
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Strict mypy over both packages
	$(UV) run mypy

test: ## Full test suite with coverage gate
	$(UV) run pytest --cov --cov-report=term-missing

test-fast: ## Tests without slow/integration markers
	$(UV) run pytest -m "not slow and not integration" -q

cov: ## HTML coverage report
	$(UV) run pytest --cov --cov-report=html
	@echo "open htmlcov/index.html"

check: lint typecheck test ## Everything CI runs

rag-demo: ## Run the offline (rule-based) extraction demo on the bundled synthetic reports
	$(UV) run report-rag extract --extractor rules --report all

rag-eval: ## Evaluate rule-based extractor against gold labels and write the eval report
	$(UV) run report-rag evaluate --extractor rules --output projects/annual-report-risk-rag/reports/eval_rules.md

pd-pipeline: ## Generate data, train, stage, compute ECL and validate the IFRS 9 PD model
	$(UV) run ifrs9-pd run-all --output-dir projects/ifrs9-pd-model/reports

pd-report: pd-pipeline ## Alias: regenerate the committed validation report

artifacts: rag-eval pd-pipeline ## Regenerate every committed report artifact

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
