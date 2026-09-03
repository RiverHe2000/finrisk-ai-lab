from ifrs9_pd.pipeline import run_pipeline

# ``results`` triggers the first run into ``run_dir``; the second run must reproduce it exactly.


def test_same_seed_identical_outputs(cfg, results, run_dir, tmp_path):
    second = tmp_path / "second"
    run_pipeline(cfg, second)
    for name in ("validation_results.json", "validation_report.md", "scorecard.json"):
        assert (second / name).read_bytes() == (run_dir / name).read_bytes(), name


def test_results_reflect_config(results, cfg):
    assert results.seed == cfg.data.seed
    assert results.data_quality.n_total == cfg.data.n_loans
    assert results.overall_rating in {"green", "amber", "red"}
    assert results.samples["oot"].gini > 0.4
    assert set(results.traffic_lights) >= {"oot_gini", "score_psi", "gini_deterioration"}
    stages = {row["stage"] for row in results.ifrs9.stage_distribution}
    assert stages <= {1, 2, 3}
    assert 0 < results.ifrs9.coverage_ratio < 0.1
