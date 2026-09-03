"""Shared fixtures: one small synthetic portfolio and one fitted model per session."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ifrs9_pd.config import DataConfig, PipelineConfig
from ifrs9_pd.data.synthetic import generate_portfolio, split_dev_oot
from ifrs9_pd.pipeline import FittedModel, fit_model, performing, run_pipeline
from ifrs9_pd.results import ValidationResults

SMALL_CFG = PipelineConfig(data=DataConfig(n_loans=3000, seed=1))


@pytest.fixture(scope="session")
def cfg() -> PipelineConfig:
    return SMALL_CFG


@pytest.fixture(scope="session")
def portfolio(cfg: PipelineConfig) -> pd.DataFrame:
    return generate_portfolio(cfg.data)


@pytest.fixture(scope="session")
def dev_oot(portfolio: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev, oot = split_dev_oot(portfolio, cfg.data.dev_cutoff)
    return performing(dev, cfg.staging.default_dpd), performing(oot, cfg.staging.default_dpd)


@pytest.fixture(scope="session")
def fitted(dev_oot: tuple[pd.DataFrame, pd.DataFrame], cfg: PipelineConfig) -> FittedModel:
    return fit_model(dev_oot[0], cfg.model)


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("run")


@pytest.fixture(scope="session")
def results(cfg: PipelineConfig, run_dir: Path) -> ValidationResults:
    return run_pipeline(cfg, run_dir)
