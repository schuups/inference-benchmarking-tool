"""Benchmark-YAML and global-config schema (IMPLEMENTATION_PLAN.md M0).

Spec references: §2.3 (global config), §10.4 (dataset_config / scenario_mix),
§10.6 (output_length_mode), §11.3 (arrival process), §11.4 (routing), §12.4 (SLOs),
§12.5 (quality_eval), §15.2 (BackendConfig), §7.2-7.5 (pre-check surface),
§5.1 (TP <= gpus_per_node).

CLI: `python -m tools.common.config <benchmark.yaml>` validates and exits 0/1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = REPO_ROOT / "tools" / "scenarios"
GLOBAL_CONFIG_PATH = Path(__file__).resolve().parent / "global.yaml"

LATENCY_SLO_METRICS = {"ttft_ms", "tpot_ms", "e2e_ms", "session_e2e_ms"}
SLO_METRICS = LATENCY_SLO_METRICS | {"error_rate_pct"}
SLO_PERCENTILES = {"p50", "p90", "p95", "p99"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- global.yaml


class ClusterInfo(StrictModel):
    type: Literal["slurm", "k8s"]
    gpus_per_node: int = Field(gt=0)
    gpu_label: str
    platform: Literal["mlp", "hpc"] | None = None
    partition: str | None = None
    namespace: str | None = None
    node_type: str | None = None

    @model_validator(mode="after")
    def _per_type_fields(self) -> "ClusterInfo":
        if self.type == "slurm" and (self.platform is None or self.partition is None):
            raise ValueError("slurm cluster needs 'platform' and 'partition'")
        if self.type == "k8s" and self.namespace is None:
            raise ValueError("k8s cluster needs 'namespace'")
        return self


class SlurmGlobals(StrictModel):
    account: str


class RegistryGlobals(StrictModel):
    jfrog_base: str


class GlobalConfig(StrictModel):
    clusters: dict[str, ClusterInfo]
    slurm: SlurmGlobals
    scratch_base: str
    collective_tests_cache_dir: str
    registry: RegistryGlobals


def load_global_config(path: Path = GLOBAL_CONFIG_PATH) -> GlobalConfig:
    with open(path) as f:
        return GlobalConfig.model_validate(yaml.safe_load(f))


# ------------------------------------------------------------- benchmark YAML


class SpeculativeDecoding(StrictModel):
    draft_model: str
    num_speculative_tokens: int = Field(gt=0)
    draft_tensor_parallel_size: int = Field(default=1, gt=0)


class BackendConfig(StrictModel):
    """Sweepable engine knobs (§15.2). All optional; engine defaults apply."""

    tensor_parallel_size: int = Field(default=1, gt=0)
    pipeline_parallel_size: int = Field(default=1, gt=0)
    data_parallel_size: int = Field(default=1, gt=0)
    expert_parallel_size: int = Field(default=1, gt=0)
    max_model_len: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    kv_cache_dtype: str | None = None
    enable_prefix_caching: bool = True
    safetensors_load_strategy: str | None = None
    kv_offloading_size: int | None = Field(default=None, gt=0)
    kv_offloading_backend: str | None = None
    speculative_decoding: SpeculativeDecoding | None = None


class Deployment(StrictModel):
    """One engine launch == one run_id (§15 deployment sweep, explicit list)."""

    target: str
    backend: Literal["vllm", "sglang", "dynamo"]
    backend_version: str
    model: str
    backend_config: BackendConfig = BackendConfig()
    image: str | None = None  # canonical JFrog tag (§8.1); derived from global.yaml when absent


class MixEntry(StrictModel):
    scenario: str
    weight: float = Field(gt=0, le=1)
    # Per-class overrides; full shape validation happens in the M1 registry loader.
    input_length: dict | None = None
    output_length: dict | None = None
    session: dict | None = None
    source_overrides: dict | None = None


class DatasetConfig(StrictModel):
    scenario_mix: list[MixEntry] = Field(min_length=1)
    num_prompts: int = Field(gt=0)
    seed: int
    tokenizer_id: str | None = None
    output_length_mode: Literal["forced", "natural"] = "forced"

    @model_validator(mode="after")
    def _weights_sum(self) -> "DatasetConfig":
        total = sum(e.weight for e in self.scenario_mix)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scenario_mix weights must sum to 1.0, got {total:.6f}")
        return self


class SLO(StrictModel):
    scenario: str  # class slug or "all"
    metric: str
    percentile: str | None = None
    threshold: float = Field(gt=0)

    @field_validator("metric")
    @classmethod
    def _metric(cls, v: str) -> str:
        if v not in SLO_METRICS:
            raise ValueError(f"unknown SLO metric '{v}' (allowed: {sorted(SLO_METRICS)})")
        return v

    @model_validator(mode="after")
    def _percentile_rules(self) -> "SLO":
        if self.metric in LATENCY_SLO_METRICS:
            if self.percentile not in SLO_PERCENTILES:
                raise ValueError(
                    f"SLO on '{self.metric}' needs percentile in {sorted(SLO_PERCENTILES)}"
                )
        elif self.percentile is not None:
            raise ValueError("'error_rate_pct' SLO takes no percentile")
        return self


class QualityGate(StrictModel):
    suite: str = "gsm8k"
    sample_size: int = Field(default=100, gt=0)
    floor: float = Field(default=0.5, ge=0, le=1)
    on_fail: Literal["abort", "continue"] = "abort"


class QualityCompare(StrictModel):
    suites: list[str] = ["gsm8k", "gpqa_diamond"]
    eval_concurrency: list[int] = [1, 32]

    @field_validator("eval_concurrency")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or any(c <= 0 for c in v):
            raise ValueError("eval_concurrency must be non-empty positive integers")
        return v


class QualityEval(StrictModel):
    gate: QualityGate = QualityGate()
    compare: QualityCompare = QualityCompare()
    skip_quality_gate: bool = False
    skip_quality_compare: bool = False


class ArrivalProcess(StrictModel):
    kind: Literal["poisson", "burst_mmpp"] = "poisson"
    burst_factor: float | None = Field(default=None, gt=1)
    mean_burst_s: float | None = Field(default=None, gt=0)
    mean_idle_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _mmpp_params(self) -> "ArrivalProcess":
        mmpp_fields = (self.burst_factor, self.mean_burst_s, self.mean_idle_s)
        if self.kind == "burst_mmpp" and any(p is None for p in mmpp_fields):
            raise ValueError(
                "burst_mmpp needs burst_factor, mean_burst_s, and mean_idle_s"
            )
        if self.kind == "poisson" and any(p is not None for p in mmpp_fields):
            raise ValueError("poisson takes no burst parameters")
        return self


class Phases(StrictModel):
    warmup_s: int = Field(gt=0)
    measurement_s: int = Field(gt=0)
    drain_timeout_s: int = Field(default=300, gt=0)
    request_timeout_s: int = Field(default=300, gt=0)
    server_ready_timeout_s: int = Field(default=3600, gt=0)


class SystemPrechecks(StrictModel):
    skip_system_prechecks: bool = False
    system_prechecks_on_warn: Literal["abort", "continue"] = "abort"
    system_prechecks_on_fail: Literal["abort", "continue"] = "abort"
    system_prechecks_timeout_s: int = Field(default=120, gt=0)
    collective_tests_version: str | None = None
    collective_tests_cache_dir: str | None = None  # default from global.yaml
    shmem_required: bool = False


class BenchmarkConfig(StrictModel):
    name: str
    description: str | None = None
    deployments: list[Deployment] = Field(min_length=1)
    dataset_config: DatasetConfig
    rate_levels: list[float] = Field(min_length=1)
    arrival_process: ArrivalProcess = ArrivalProcess()
    routing_strategy: Literal["random", "session_affinity"] = "random"
    phases: Phases
    slos: list[SLO] | None = None
    quality_eval: QualityEval = QualityEval()
    system_prechecks: SystemPrechecks = SystemPrechecks()
    hardware_sampling_interval_s: float = Field(default=1.0, gt=0)
    server_time_limit: str | None = None  # HH:MM:SS; shared by all jobs (§5.1)

    @field_validator("rate_levels")
    @classmethod
    def _rates(cls, v: list[float]) -> list[float]:
        if any(r <= 0 for r in v):
            raise ValueError("rate_levels must be positive (session starts/s, §11.3)")
        return v

    @model_validator(mode="after")
    def _slo_scenarios_in_mix(self) -> "BenchmarkConfig":
        classes = {e.scenario for e in self.dataset_config.scenario_mix}
        for slo in self.slos or []:
            if slo.scenario != "all" and slo.scenario not in classes:
                raise ValueError(
                    f"SLO references scenario '{slo.scenario}' not in scenario_mix {sorted(classes)}"
                )
        return self


def validate_scenarios_registered(
    cfg: BenchmarkConfig, registry_dir: Path = SCENARIOS_DIR
) -> None:
    """Every mix entry must name a registered scenario (§10.3/§10.4)."""
    known = sorted(p.stem for p in registry_dir.glob("*.yaml"))
    for entry in cfg.dataset_config.scenario_mix:
        if entry.scenario not in known:
            raise ValueError(
                f"unregistered scenario '{entry.scenario}' (registered: {known})"
            )


def validate_against_globals(cfg: BenchmarkConfig, glob: GlobalConfig) -> None:
    """Cross-checks needing the cluster catalogue (§5.1)."""
    for i, dep in enumerate(cfg.deployments):
        cluster = glob.clusters.get(dep.target)
        if cluster is None:
            raise ValueError(
                f"deployments[{i}]: unknown target '{dep.target}' "
                f"(known: {sorted(glob.clusters)})"
            )
        tp = dep.backend_config.tensor_parallel_size
        if tp > cluster.gpus_per_node:
            raise ValueError(
                f"deployments[{i}]: tensor_parallel_size={tp} exceeds "
                f"gpus_per_node={cluster.gpus_per_node} on '{dep.target}' (§5.1 — "
                f"cross-node TP is impractical on Alps; scale out via PP or DP)"
            )


def load_benchmark_config(
    path: Path,
    global_config: GlobalConfig | None = None,
    registry_dir: Path = SCENARIOS_DIR,
) -> BenchmarkConfig:
    with open(path) as f:
        cfg = BenchmarkConfig.model_validate(yaml.safe_load(f))
    validate_scenarios_registered(cfg, registry_dir)
    validate_against_globals(cfg, global_config or load_global_config())
    return cfg


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a benchmark YAML (M0)")
    parser.add_argument("yaml_path", type=Path)
    args = parser.parse_args()
    from pydantic import ValidationError

    try:
        cfg = load_benchmark_config(args.yaml_path)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        msg = first["msg"].removeprefix("Value error, ")
        print(f"INVALID: {args.yaml_path}: {loc}: {msg}")
        return 1
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"INVALID: {args.yaml_path}: {exc}")
        return 1
    n_classes = len(cfg.dataset_config.scenario_mix)
    print(
        f"OK: {cfg.name} — {len(cfg.deployments)} deployment(s), "
        f"{n_classes} workload class(es), {len(cfg.rate_levels)} rate level(s), "
        f"{len(cfg.slos or [])} SLO(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
