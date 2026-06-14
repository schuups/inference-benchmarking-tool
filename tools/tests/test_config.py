"""M0 DoD: canonical example accepted; every violation class rejected."""

import copy

import pytest

from tools.common.config import (
    BenchmarkConfig,
    load_benchmark_config,
    validate_against_globals,
    validate_scenarios_registered,
)

# globals_cfg, canonical_dict, canonical_path fixtures come from conftest.py


def _validate(d, globals_cfg):
    cfg = BenchmarkConfig.model_validate(d)
    validate_scenarios_registered(cfg)
    validate_against_globals(cfg, globals_cfg)
    return cfg


def test_global_config_loads(globals_cfg):
    assert {"clariden", "bristen", "beverin", "breithorn"} <= set(globals_cfg.clusters)
    assert globals_cfg.slurm.account == "csstaff"
    assert all(c.gpus_per_node == 4 for c in globals_cfg.clusters.values())


def test_canonical_example_accepted(globals_cfg, canonical_path):
    cfg = load_benchmark_config(canonical_path, globals_cfg)
    assert cfg.dataset_config.output_length_mode == "forced"
    assert len(cfg.dataset_config.scenario_mix) == 2
    assert cfg.quality_eval.gate.on_fail == "abort"
    assert not cfg.quality_eval.skip_quality_gate


def test_weights_must_sum_to_one(canonical_dict, globals_cfg):
    canonical_dict["dataset_config"]["scenario_mix"][0]["weight"] = 0.7
    with pytest.raises(ValueError, match="must sum to 1.0"):
        _validate(canonical_dict, globals_cfg)


def test_unregistered_scenario_rejected(canonical_dict, globals_cfg):
    canonical_dict["dataset_config"]["scenario_mix"][0]["scenario"] = "no-such-scenario"
    canonical_dict["slos"] = [s for s in canonical_dict["slos"] if s["scenario"] != "agentic-coding"]
    with pytest.raises(ValueError, match="unregistered scenario"):
        _validate(canonical_dict, globals_cfg)


def test_all_registered_scenarios_accepted(canonical_dict, globals_cfg):
    for slug in ("long-context-followup", "smoke-synthetic"):
        d = copy.deepcopy(canonical_dict)
        d["dataset_config"]["scenario_mix"] = [{"scenario": slug, "weight": 1.0}]
        d["slos"] = [s for s in d["slos"] if s["scenario"] == "all"]
        _validate(d, globals_cfg)


def test_bad_slo_metric_rejected(canonical_dict, globals_cfg):
    canonical_dict["slos"][0]["metric"] = "ttft_seconds"
    with pytest.raises(ValueError, match="unknown SLO metric"):
        _validate(canonical_dict, globals_cfg)


def test_latency_slo_requires_percentile(canonical_dict, globals_cfg):
    del canonical_dict["slos"][0]["percentile"]
    with pytest.raises(ValueError, match="needs percentile"):
        _validate(canonical_dict, globals_cfg)


def test_error_rate_slo_takes_no_percentile(canonical_dict, globals_cfg):
    canonical_dict["slos"][3]["percentile"] = "p95"
    with pytest.raises(ValueError, match="no percentile"):
        _validate(canonical_dict, globals_cfg)


def test_slo_scenario_must_be_in_mix(canonical_dict, globals_cfg):
    canonical_dict["slos"][0]["scenario"] = "long-context-followup"
    with pytest.raises(ValueError, match="not in scenario_mix"):
        _validate(canonical_dict, globals_cfg)


def test_tp_above_gpus_per_node_rejected(canonical_dict, globals_cfg):
    canonical_dict["deployments"][0]["backend_config"]["tensor_parallel_size"] = 8
    with pytest.raises(ValueError, match="exceeds\\s+gpus_per_node"):
        _validate(canonical_dict, globals_cfg)


def test_unknown_target_rejected(canonical_dict, globals_cfg):
    canonical_dict["deployments"][0]["target"] = "santis"
    with pytest.raises(ValueError, match="unknown target"):
        _validate(canonical_dict, globals_cfg)


def test_mmpp_requires_burst_params(canonical_dict, globals_cfg):
    canonical_dict["arrival_process"] = {"kind": "burst_mmpp"}
    with pytest.raises(ValueError, match="burst_mmpp needs"):
        _validate(canonical_dict, globals_cfg)


def test_poisson_rejects_burst_params(canonical_dict, globals_cfg):
    canonical_dict["arrival_process"] = {"kind": "poisson", "burst_factor": 5.0}
    with pytest.raises(ValueError, match="no burst parameters"):
        _validate(canonical_dict, globals_cfg)


def test_natural_output_mode_accepted(canonical_dict, globals_cfg):
    canonical_dict["dataset_config"]["output_length_mode"] = "natural"
    cfg = _validate(canonical_dict, globals_cfg)
    assert cfg.dataset_config.output_length_mode == "natural"


def test_negative_rate_level_rejected(canonical_dict, globals_cfg):
    canonical_dict["rate_levels"] = [0.1, -0.5]
    with pytest.raises(ValueError, match="must be positive"):
        _validate(canonical_dict, globals_cfg)


def test_unknown_top_level_key_rejected(canonical_dict, globals_cfg):
    canonical_dict["scenario"] = "agentic-coding"  # pre-mix legacy key
    with pytest.raises(ValueError, match="extra|scenario"):
        _validate(canonical_dict, globals_cfg)
