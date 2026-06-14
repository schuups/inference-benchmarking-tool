"""Synthetic results DB with analytically-known answers, for M9's report tests
and as a runnable demo input for the template notebook.

Two deployment configs (a baseline and an fp8 KV-cache variant) over one chat
class, crafted so the derived quantities are exact:

  run A (baseline): ttft p95 = {λ1:300, λ2:450, λ3:600} vs SLO 500  → λ* = 2
  run B (fp8):      ttft p95 = {λ1:300, λ2:350, λ3:450} vs SLO 500  → λ* = 3

  warmup-phase requests carry ttft=9999 and MUST be excluded from the
  measurement-phase percentiles (a measurement-window-filtering check).

At λ*: 20 single-turn sessions (run A) / 30 (run B) in a 10s measurement window
→ throughput 2.0 / 3.0 sessions·s⁻¹; with 3600 sessions·user⁻¹·h⁻¹ (=1·s⁻¹)
→ 2.0 / 3.0 supportable users; e2e 1000ms → 2.0 / 3.0 concurrent (Little's law).
gsm8k Stage-B: 0.80 (A) vs 0.76 (B) → 1.5× users at −0.04 pts.
"""

from __future__ import annotations

from pathlib import Path

from tools.common.results_db import ResultsDB

SCENARIO = "chat-short-turns"
RATE_LEVELS = [1.0, 2.0, 3.0]
WARMUP_S = 10
MEASUREMENT_S = 10
SLOS = [
    {"scenario": SCENARIO, "metric": "ttft_ms", "percentile": "p95", "threshold": 500.0},
    {"scenario": "all", "metric": "error_rate_pct", "percentile": None, "threshold": 5.0},
]
_MANIFEST = {
    "mix": [{"scenario": SCENARIO, "weight": 1.0, "expected_request_share": 1.0}],
    "classes": [{
        "name": SCENARIO, "summary": "Short-turn conversational chat.",
        "maturity": "established",
        "modelled": ["multi-turn prefix-cache hits", "small per-turn token counts"],
        "not_modelled": ["no image inputs", "no tool-call interleaving"],
        "assumptions": ["input length: lognormal mean=512", "max output tokens = 256"],
    }],
    "run_assumptions": [
        "arrival process: poisson — λ counts session starts (§11.3)",
        "routing strategy: random", "output_length_mode: forced",
        "master seed: 1", "tokenizer: word",
    ],
}
_QUALITY_EVAL = {
    "gate": {"suite": "gsm8k", "sample_size": 100, "floor": 0.5, "on_fail": "abort"},
    "compare": {"suites": ["gsm8k"], "eval_concurrency": [1, 32]},
    "skip_quality_gate": False, "skip_quality_compare": False,
}


def _request_rows(run_id, ttft_by_lambda, n_by_lambda, sidx_base):
    rows = []
    sidx = sidx_base
    for lam in RATE_LEVELS:
        # warmup-phase rows (issued < warmup) with bad ttft — must be excluded.
        for w in range(2):
            rows.append(_req(run_id, lam, w, sidx, issued_ms=5000.0, ttft=9999.0, e2e=20000.0))
            sidx += 1
        n = n_by_lambda[lam]
        for i in range(n):  # measurement-phase rows, evenly spread within [10s, 20s)
            issued = WARMUP_S * 1000 + (i + 0.5) * (MEASUREMENT_S * 1000 / n)
            rows.append(_req(run_id, lam, i, sidx, issued_ms=issued, ttft=ttft_by_lambda[lam], e2e=1000.0))
            sidx += 1
    return rows, sidx


def _req(run_id, lam, request_id, session_idx, *, issued_ms, ttft, e2e):
    return {
        "run_id": run_id, "rate_lambda": lam, "request_id": request_id,
        "session_idx": session_idx, "instance_id": "i0", "scenario": SCENARIO,
        "turn_idx": 0, "issued_at_ms": issued_ms, "final_turn": 1,
        "ttft_ms": ttft, "tpot_ms": 10.0, "e2e_ms": e2e,
        "input_tokens": 100, "output_tokens": 50, "success": 1, "error": None,
    }


def _quality_rows(run_id, compare_score):
    rows = [{
        "run_id": run_id, "instance_id": "i0", "stage": "gate", "suite": "gsm8k",
        "eval_concurrency": 1, "sample_size": 100, "metric": "exact_match",
        "score": 0.85, "floor": 0.5, "status": "pass",
        "sampling_params": {"temperature": 0.0}, "harness_version": "fixture",
        "ts": "2026-06-13T00:00:00+00:00",
    }]
    for conc in (1, 32):
        rows.append({
            "run_id": run_id, "instance_id": "i0", "stage": "compare", "suite": "gsm8k",
            "eval_concurrency": conc, "sample_size": 200, "metric": "exact_match",
            "score": compare_score, "floor": None, "status": None,
            "sampling_params": {"temperature": 0.0}, "harness_version": "fixture",
            "ts": "2026-06-13T00:00:00+00:00",
        })
    return rows


def _hardware_rows(run_id):
    sm_by_lambda = {1.0: 35.0, 2.0: 60.0, 3.0: 85.0}
    rows = []
    for lam in RATE_LEVELS:
        for tick in range(2):
            rows.append({
                "run_id": run_id, "instance_id": "i0", "rate_lambda": lam,
                "ts": f"2026-06-13T00:0{int(lam)}:0{tick}+00:00", "gpu_index": 0,
                "gpu_sm_active_pct": sm_by_lambda[lam], "gpu_tensor_active_pct": sm_by_lambda[lam] - 5,
                "gpu_power_w": 300.0 + lam * 50, "gpu_mem_pct": 70.0,
            })
    return rows


def _add_run(db: ResultsDB, run_id, backend_config, ttft_by_lambda, n_by_lambda, compare_score, sidx_base):
    db.insert("experiments", {
        "run_id": run_id, "model": "fixture/model", "backend": "vllm",
        "backend_config": backend_config, "dataset_config": {"seed": 1},
        "scenario_mix": [{"scenario": SCENARIO, "weight": 1.0}],
        "scenario_manifest": _MANIFEST, "slos": SLOS, "quality_eval": _QUALITY_EVAL,
        "rate_levels": RATE_LEVELS, "warmup_s": WARMUP_S, "measurement_s": MEASUREMENT_S,
        "created_at": f"2026-06-13T00:00:0{sidx_base % 10}+00:00",
    })
    db.insert("instances", {
        "run_id": run_id, "instance_id": "i0", "endpoint": "http://nid000001:8000",
        "node": "nid000001", "model_load_total_s": 120.0, "model_load_weights_s": 25.0,
        "model_load_engine_init_s": 60.0, "model_load_cuda_graph_capture_s": 23.0,
        "model_load_inductor_compile_s": 12.0,
    })
    db.insert_many("system_prechecks", [
        {"run_id": run_id, "instance_id": "i0", "metric": "nccl_all_reduce_128_mib",
         "measured": 118.0, "expected": 122.0, "tolerance_pct": -10.0, "status": "pass",
         "ts": "2026-06-13T00:00:00+00:00"},
        {"run_id": run_id, "instance_id": "i0", "metric": "nvshmem_alltoall_latency_128_kib",
         "measured": 9.0, "expected": 7.0, "tolerance_pct": 20.0, "status": "warn",
         "ts": "2026-06-13T00:00:00+00:00"},
    ])
    rows, sidx = _request_rows(run_id, ttft_by_lambda, n_by_lambda, sidx_base)
    db.insert_many("requests", rows)
    db.insert_many("quality_evals", _quality_rows(run_id, compare_score))
    db.insert_many("hardware_stats", _hardware_rows(run_id))
    db.insert_many("server_stats", [
        {"run_id": run_id, "instance_id": "i0", "rate_lambda": lam,
         "ts": f"2026-06-13T00:0{int(lam)}:00+00:00", "requests_running": int(lam * 4),
         "requests_waiting": 0, "gpu_cache_pct": lam * 20.0, "spec_accept_rate": None}
        for lam in RATE_LEVELS
    ])
    return sidx


def build_fixture_db(path: Path | str) -> dict:
    """Build the two-run fixture DB; return the analytically-known expected values."""
    db = ResultsDB(path)
    try:
        sidx = _add_run(db, "fixtureA", {}, {1.0: 300.0, 2.0: 450.0, 3.0: 600.0},
                        {1.0: 20, 2.0: 20, 3.0: 20}, compare_score=0.80, sidx_base=0)
        _add_run(db, "fixtureB", {"kv_cache_dtype": "fp8"}, {1.0: 300.0, 2.0: 350.0, 3.0: 450.0},
                 {1.0: 20, 2.0: 20, 3.0: 30}, compare_score=0.76, sidx_base=sidx)
    finally:
        db.close()
    return {
        "run_a": "fixtureA", "run_b": "fixtureB", "scenario": SCENARIO,
        "lambda_star_a": 2.0, "lambda_star_b": 3.0,
        "users_a": 2.0, "users_b": 3.0, "concurrent_a": 2.0,
        "quality_a": 0.80, "quality_b": 0.76,
        "sessions_per_user_per_hour": {SCENARIO: 3600.0},
    }
