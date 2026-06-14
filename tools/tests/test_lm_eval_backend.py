"""Coverage for the production LmEvalBackend result parsers (§12.5).

The backend's `evaluate` needs a GPU endpoint + the lm-eval-harness, but
`_pick_metric` / `_graded_count` are pure parsers over the harness results dict —
exactly the brittle adapter code that breaks silently when the harness output
shape drifts. These exercise them with synthetic results (no network, no GPU).
"""

import pytest

from tools.benchmarker.quality_eval.base import QualityEvalError
from tools.benchmarker.quality_eval.lm_eval_backend import _graded_count, _pick_metric


def test_pick_metric_prefers_exact_match_and_skips_stderr_alias():
    results = {"results": {"gsm8k": {
        "exact_match,strict-match": 0.82,
        "exact_match_stderr,strict-match": 0.01,
        "alias": "gsm8k",
    }}}
    metric, score = _pick_metric(results, "gsm8k")
    assert metric == "exact_match,strict-match"
    assert score == pytest.approx(0.82)


def test_pick_metric_prefers_acc_norm_over_acc():
    results = {"results": {"hellaswag": {"acc,none": 0.50, "acc_norm,none": 0.62}}}
    metric, score = _pick_metric(results, "hellaswag")
    assert metric == "acc_norm,none"
    assert score == pytest.approx(0.62)


def test_pick_metric_falls_back_to_first_scalar():
    results = {"results": {"custom": {"f1,none": 0.7, "f1_stderr,none": 0.03}}}
    metric, score = _pick_metric(results, "custom")
    assert metric == "f1,none"
    assert score == pytest.approx(0.7)


def test_pick_metric_no_results_raises():
    with pytest.raises(QualityEvalError, match="no results"):
        _pick_metric({"results": {}}, "gsm8k")


def test_pick_metric_no_scalar_metric_raises():
    with pytest.raises(QualityEvalError, match="no scalar metric"):
        _pick_metric({"results": {"gsm8k": {"alias": "gsm8k"}}}, "gsm8k")


def test_graded_count_prefers_effective():
    results = {"n-samples": {"gsm8k": {"original": 1319, "effective": 200}}}
    assert _graded_count(results, "gsm8k", 200) == 200


def test_graded_count_falls_back_to_sample_size():
    assert _graded_count({}, "gsm8k", 50) == 50
    assert _graded_count({}, "gsm8k", None) == 0
