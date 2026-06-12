"""Shared fixture DB with analytically known λ* and users count (M9 DoD)."""

from __future__ import annotations

from tools.benchmarker.db import ResultsDB

RUN_ID = "fixture-run"
WARMUP_S, MEASUREMENT_S = 10, 100
WINDOW_LO_MS = WARMUP_S * 1000

# chat TTFT p95 by λ — crafted so the 800ms SLO passes at 0.1/0.2, fails at 0.4
CHAT_P95 = {0.1: 500.0, 0.2: 700.0, 0.4: 900.0}
SLOS = [
    {"scenario": "chat-short-turns", "metric": "ttft_ms", "percentile": "p95", "threshold": 800},
    {"scenario": "all", "metric": "error_rate_pct", "threshold": 1.0},
]
# at λ* = 0.2: 20 chat sessions in the 100s window -> 0.2/s; 80 agentic -> 0.8/s
SESSIONS_IN_WINDOW = {"chat-short-turns": 20, "agentic-coding": 80}
USER_RATES = {"chat-short-turns": 2.0, "agentic-coding": 4.0}
EXPECTED_POPULATION = {"chat-short-turns": 0.2 * 3600 / 2.0, "agentic-coding": 0.8 * 3600 / 4.0}
EXPECTED_LAMBDA_STAR = 0.2
AGENTIC_WALL_MS = 30_000.0  # two turns, fixed spacing


def build_fixture_db(path) -> ResultsDB:
    db = ResultsDB(path)
    db.insert(
        "experiments",
        {
            "run_id": RUN_ID, "model": "org/m", "backend": "vllm",
            "scenario_mix": [
                {"scenario": "agentic-coding", "weight": 0.8},
                {"scenario": "chat-short-turns", "weight": 0.2},
            ],
            "scenario_manifest": {
                "mix": [
                    {"scenario": "agentic-coding", "weight": 0.8, "expected_request_share": 0.89},
                    {"scenario": "chat-short-turns", "weight": 0.2, "expected_request_share": 0.11},
                ],
                "classes": [
                    {"name": "agentic-coding", "weight": 0.8, "summary": "fixture", "maturity": "established",
                     "modelled": ["two-turn sessions"], "not_modelled": ["everything else"],
                     "assumptions": ["fixture data"]},
                    {"name": "chat-short-turns", "weight": 0.2, "summary": "fixture", "maturity": "established",
                     "modelled": ["single turns"], "not_modelled": ["everything else"],
                     "assumptions": ["fixture data"]},
                ],
                "run_assumptions": ["arrival process: poisson — λ counts session starts (§11.3)",
                                    "output_length_mode: forced", "master seed: 1234", "tokenizer: word"],
            },
            "slos": SLOS,
            "rate_levels": [0.1, 0.2, 0.4],
            "warmup_s": WARMUP_S, "measurement_s": MEASUREMENT_S,
            "created_at": "2026-06-12T00:00:00Z",
        },
    )
    db.insert(
        "instances",
        {"run_id": RUN_ID, "instance_id": "i0", "endpoint": "http://node:8000",
         "node": "nid000001", "model_load_total_s": 120.5, "model_load_weights_s": 25.3,
         "model_load_engine_init_s": 61.2, "model_load_cuda_graph_capture_s": 23.0,
         "model_load_inductor_compile_s": None},
    )
    db.insert(
        "system_prechecks",
        {"run_id": RUN_ID, "instance_id": "i0", "metric": "nccl_all_reduce_128_mib",
         "measured": 128.1, "expected": None, "tolerance_pct": None, "status": "pass",
         "ts": "2026-06-12T00:00:00Z"},
    )
    db.insert(
        "quality_evals",
        {"run_id": RUN_ID, "instance_id": "i0", "stage": "gate", "suite": "gsm8k",
         "eval_concurrency": 1, "sample_size": 100, "metric": "exact_match",
         "score": 0.82, "floor": 0.5, "status": "pass",
         "sampling_params": '{"temperature": 0.0}', "harness_version": "lm-eval-0.4",
         "ts": "2026-06-12T00:00:00Z"},
    )
    rows, request_id, session_idx = [], 0, 0
    for rate in (0.1, 0.2, 0.4):
        for i in range(SESSIONS_IN_WINDOW["chat-short-turns"]):
            rows.append(
                {
                    "run_id": RUN_ID, "rate_lambda": rate, "request_id": request_id,
                    "session_idx": session_idx, "instance_id": "i0",
                    "scenario": "chat-short-turns", "turn_idx": 0, "final_turn": 1,
                    "issued_at_ms": WINDOW_LO_MS + i * 4000.0,
                    "ttft_ms": CHAT_P95[rate], "tpot_ms": 20.0, "e2e_ms": 2000.0,
                    "input_tokens": 200, "output_tokens": 100, "success": 1, "error": None,
                }
            )
            request_id += 1
            session_idx += 1
        for i in range(SESSIONS_IN_WINDOW["agentic-coding"]):
            start = WINDOW_LO_MS + i * 1000.0
            for turn_idx, (offset, final) in enumerate([(0.0, 0), (25_000.0, 1)]):
                rows.append(
                    {
                        "run_id": RUN_ID, "rate_lambda": rate, "request_id": request_id,
                        "session_idx": session_idx, "instance_id": "i0",
                        "scenario": "agentic-coding", "turn_idx": turn_idx, "final_turn": final,
                        "issued_at_ms": start + offset,
                        "ttft_ms": 300.0, "tpot_ms": 30.0,
                        "e2e_ms": 5000.0 if not final else AGENTIC_WALL_MS - 25_000.0,
                        "input_tokens": 15000, "output_tokens": 800, "success": 1, "error": None,
                    }
                )
                request_id += 1
            session_idx += 1
    db.insert_many("requests", rows)
    return db
