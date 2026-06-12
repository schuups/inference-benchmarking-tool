"""M1 DoD tests: §10.8 byte-identity, §10.6 mechanics, §13.7 manifest, §10.1 aborts."""

import json
import statistics
from pathlib import Path

import pytest
import yaml

from tools.common.config import SCENARIOS_DIR, BenchmarkConfig
from tools.benchmarker.dataset_gen import sources
from tools.benchmarker.dataset_gen.generator import (
    MANIFEST_FILENAME,
    POOL_FILENAME,
    generate,
)
from tools.benchmarker.dataset_gen.registry import load_scenario
from tools.benchmarker.dataset_gen.sampling import sub_seed
from tools.benchmarker.dataset_gen.sources import DatasetSourceError
from tools.benchmarker.dataset_gen.tokenizers import WordTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ fixtures


def _synthetic_scenario(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "summary": f"test scenario {name}",
        "maturity": "exploratory",
        "source": {"kind": "synthetic"},
        "input_length": {"distribution": "lognormal", "params": {"mean": 120, "sigma": 0.4, "min": 20, "max": 600}},
        "output_length": {"distribution": "lognormal", "params": {"mean": 80, "sigma": 0.4, "min": 8, "max": 400}},
        "thinking": False,
        "session": {
            "mode": "sequential",
            "turns_per_session": {"distribution": "lognormal", "params": {"mean": 4, "sigma": 0.4, "min": 2, "max": 10}},
            "prefix_strategy": "append_delta",
            "think_time_ms": {"distribution": "lognormal", "params": {"mean": 1000, "sigma": 0.5}},
        },
        "manifest": {"modelled": ["synthetic multi-turn test"], "not_modelled": ["everything else"]},
    }
    base.update(overrides)
    return base


@pytest.fixture()
def registry_dir(tmp_path):
    reg = tmp_path / "scenarios"
    reg.mkdir()
    entries = {
        "synth-multi": _synthetic_scenario("synth-multi"),
        "synth-single": _synthetic_scenario(
            "synth-single",
            session={
                "mode": "open_loop",
                "turns_per_session": {"distribution": "fixed", "params": {"value": 1}},
                "prefix_strategy": "append_delta",
            },
        ),
        "synth-thinking": _synthetic_scenario("synth-thinking", thinking=True),
    }
    for slug, entry in entries.items():
        (reg / f"{slug}.yaml").write_text(yaml.safe_dump(entry))
    return reg


def _config(mix, num_prompts=400, seed=1234) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "name": "test",
            "deployments": [
                {"target": "clariden", "backend": "vllm", "backend_version": "x", "model": "org/test-model"}
            ],
            "dataset_config": {"scenario_mix": mix, "num_prompts": num_prompts, "seed": seed},
            "rate_levels": [0.1],
            "phases": {"warmup_s": 60, "measurement_s": 120},
        }
    )


MIX_80_20 = [
    {"scenario": "synth-multi", "weight": 0.8},
    {"scenario": "synth-single", "weight": 0.2},
]


def _records(out_dir: Path) -> list[dict]:
    with open(out_dir / POOL_FILENAME) as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------- §10.8/§10.6


def test_byte_identical_regeneration(tmp_path, registry_dir):
    cfg = _config(MIX_80_20)
    a, b = tmp_path / "a", tmp_path / "b"
    generate(cfg, WordTokenizer(), a, registry_dir)
    generate(cfg, WordTokenizer(), b, registry_dir)
    assert (a / POOL_FILENAME).read_bytes() == (b / POOL_FILENAME).read_bytes()
    assert (a / MANIFEST_FILENAME).read_bytes() == (b / MANIFEST_FILENAME).read_bytes()


def test_different_seed_different_pool(tmp_path, registry_dir):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(_config(MIX_80_20, seed=1), WordTokenizer(), a, registry_dir)
    generate(_config(MIX_80_20, seed=2), WordTokenizer(), b, registry_dir)
    assert (a / POOL_FILENAME).read_bytes() != (b / POOL_FILENAME).read_bytes()


def test_sub_seeds_differ_by_class_and_axis():
    assert sub_seed(1, "a", "turns") != sub_seed(1, "b", "turns")
    assert sub_seed(1, "a", "turns") != sub_seed(1, "a", "thinktime")
    assert sub_seed(1, "a", "turns") != sub_seed(2, "a", "turns")


def test_headers_unique_pool_wide(tmp_path, registry_dir):
    generate(_config(MIX_80_20), WordTokenizer(), tmp_path / "o", registry_dir)
    records = _records(tmp_path / "o")
    headers = [r["prompt_text"].split()[0] for r in records if r["turn_idx"] == 0]
    assert len(headers) == len(set(headers))
    assert all(h.startswith("[session-") or h.startswith("[prompt-") for h in headers)
    # follow-up turns do not repeat the header (append_delta carries the prefix)
    followups = [r for r in records if r["turn_idx"] > 0]
    assert followups and all(not r["prompt_text"].startswith("[session-") for r in followups)


def test_session_structure_and_think_time(tmp_path, registry_dir):
    generate(_config(MIX_80_20), WordTokenizer(), tmp_path / "o", registry_dir)
    records = _records(tmp_path / "o")
    by_session: dict[int, list[dict]] = {}
    for r in records:
        by_session.setdefault(r["session_idx"], []).append(r)
    for turns in by_session.values():
        assert [t["turn_idx"] for t in turns] == list(range(len(turns)))
        assert turns[0]["think_time_ms"] is None
        for t in turns[1:]:
            assert t["think_time_ms"] > 0
    single = [r for r in records if r["scenario"] == "synth-single"]
    assert single and all(r["turn_idx"] == 0 for r in single)


def test_mix_split_matches_expected_request_share(tmp_path, registry_dir):
    cfg = _config(MIX_80_20, num_prompts=2000)
    manifest = generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    records = _records(tmp_path / "o")
    share = {m["scenario"]: m["expected_request_share"] for m in manifest["mix"]}
    counts = {s: sum(1 for r in records if r["scenario"] == s) for s in share}
    total = len(records)
    assert abs(total - 2000) / 2000 < 0.15  # num_prompts is the approximate pool total
    for slug in share:
        assert counts[slug] / total == pytest.approx(share[slug], rel=0.15)
    assert sum(share.values()) == pytest.approx(1.0, abs=0.01)


def test_input_length_distribution_tolerance(tmp_path, registry_dir):
    cfg = _config([{"scenario": "synth-multi", "weight": 1.0}], num_prompts=3000)
    generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    lengths = [r["text_tokens"] for r in _records(tmp_path / "o")]
    # clamps truncate the lognormal tail, so the realized mean sits slightly
    # below the declared 120; ±15% tolerance per the M1 DoD
    assert statistics.mean(lengths) == pytest.approx(120, rel=0.15)


def test_thinking_widens_output(tmp_path, registry_dir):
    base = _config([{"scenario": "synth-multi", "weight": 1.0}], num_prompts=2000)
    think = _config([{"scenario": "synth-thinking", "weight": 1.0}], num_prompts=2000)
    generate(base, WordTokenizer(), tmp_path / "a", registry_dir)
    generate(think, WordTokenizer(), tmp_path / "b", registry_dir)
    mean_base = statistics.mean(r["max_tokens"] for r in _records(tmp_path / "a"))
    mean_think = statistics.mean(r["max_tokens"] for r in _records(tmp_path / "b"))
    # ×2.5 on the mean, but the preserved max=400 clamp compresses the realized ratio
    assert 1.5 < mean_think / mean_base < 2.6


def test_followup_input_length(tmp_path, registry_dir):
    entry = _synthetic_scenario("synth-followup")
    entry["input_length"] = {"distribution": "fixed", "params": {"value": 200}}
    entry["session"]["followup_input_length"] = {"distribution": "fixed", "params": {"value": 30}}
    (registry_dir / "synth-followup.yaml").write_text(yaml.safe_dump(entry))
    cfg = _config([{"scenario": "synth-followup", "weight": 1.0}], num_prompts=300)
    manifest = generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    records = _records(tmp_path / "o")
    assert {r["text_tokens"] for r in records if r["turn_idx"] == 0} == {200}
    assert {r["text_tokens"] for r in records if r["turn_idx"] > 0} == {30}
    assert any("follow-up input length" in a for a in manifest["classes"][0]["assumptions"])


def test_followup_defaults_to_input_length(tmp_path, registry_dir):
    cfg = _config([{"scenario": "synth-multi", "weight": 1.0}], num_prompts=300)
    manifest = generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    assert not any("follow-up input length" in a for a in manifest["classes"][0]["assumptions"])


def test_per_class_override_applies(tmp_path, registry_dir):
    mix = [
        {
            "scenario": "synth-multi",
            "weight": 1.0,
            "input_length": {"distribution": "fixed", "params": {"value": 50}},
        }
    ]
    cfg = _config(mix, num_prompts=200)
    manifest = generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    lengths = {r["text_tokens"] for r in _records(tmp_path / "o")}
    assert lengths == {50}
    assert any("fixed" in a for a in manifest["classes"][0]["assumptions"])


# ----------------------------------------------------------------------- §13.7


def test_manifest_schema(tmp_path, registry_dir):
    manifest = generate(_config(MIX_80_20), WordTokenizer(), tmp_path / "o", registry_dir)
    assert set(manifest) == {"mix", "classes", "run_assumptions"}
    for cls in manifest["classes"]:
        assert set(cls) == {"name", "weight", "summary", "maturity", "modelled", "not_modelled", "assumptions"}
        assert cls["modelled"] and cls["not_modelled"] and cls["assumptions"]
    joined = " ".join(manifest["run_assumptions"])
    for needle in ("session starts", "routing strategy", "output_length_mode", "master seed", "tokenizer"):
        assert needle in joined


# ------------------------------------------------------------------ §10.1/§10.5


def test_unimplemented_source_aborts(tmp_path):
    cfg = _config([{"scenario": "chat-short-turns", "weight": 1.0}])
    with pytest.raises(DatasetSourceError, match="not implemented"):
        generate(cfg, WordTokenizer(), tmp_path / "o", SCENARIOS_DIR)


def test_longbench_missing_datasets_pkg_aborts(tmp_path, registry_dir, monkeypatch):
    entry = _synthetic_scenario("synth-lb", source={"kind": "longbench", "config": {"tasks": ["lcc"]}})
    (registry_dir / "synth-lb.yaml").write_text(yaml.safe_dump(entry))

    def boom(tasks):
        raise DatasetSourceError("longbench: failed to load task 'lcc': offline")

    monkeypatch.setattr(sources, "_load_longbench_items", boom)
    cfg = _config([{"scenario": "synth-lb", "weight": 1.0}])
    with pytest.raises(DatasetSourceError, match="failed to load"):
        generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)


def test_longbench_with_fake_corpus(tmp_path, registry_dir, monkeypatch):
    entry = _synthetic_scenario("synth-lb", source={"kind": "longbench", "config": {"tasks": ["lcc"]}})
    (registry_dir / "synth-lb.yaml").write_text(yaml.safe_dump(entry))
    corpus = [f"def fn_{i}(x):\n    return x + {i}\n" * 30 for i in range(20)]
    monkeypatch.setattr(sources, "_load_longbench_items", lambda tasks: corpus)
    cfg = _config([{"scenario": "synth-lb", "weight": 1.0}], num_prompts=300)
    generate(cfg, WordTokenizer(), tmp_path / "o", registry_dir)
    records = _records(tmp_path / "o")
    assert records and all("def fn_" in r["prompt_text"] for r in records)


def test_modality_rejection(tmp_path, registry_dir):
    entry = _synthetic_scenario("synth-img", modalities=["text", "image"])
    (registry_dir / "synth-img.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(ValueError, match="text-only"):
        load_scenario(registry_dir, "synth-img")


def test_multi_turn_requires_think_time(registry_dir):
    entry = _synthetic_scenario("synth-nothink")
    del entry["session"]["think_time_ms"]
    (registry_dir / "synth-nothink.yaml").write_text(yaml.safe_dump(entry))
    with pytest.raises(ValueError, match="think_time_ms"):
        load_scenario(registry_dir, "synth-nothink")


def test_real_registry_entries_load():
    for slug in ("agentic-coding", "chat-short-turns", "long-context-followup", "smoke-synthetic"):
        scenario = load_scenario(SCENARIOS_DIR, slug)
        assert scenario.name == slug
