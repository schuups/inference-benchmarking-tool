"""Scenario-registry loader (SPECIFICATIONS.md §10.3).

The registry is data, not code: one YAML per scenario under tools/scenarios/.
Loading validates the schema, rejects non-text modalities (§10.5), and enforces
that multi-turn scenarios carry a think_time_ms distribution (§10.3/§10.7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_KINDS = {"synthetic", "longbench", "reasoning_trace_replay", "wildchat"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Distribution(StrictModel):
    distribution: Literal["lognormal", "normal", "fixed"]
    params: dict[str, float]

    @model_validator(mode="after")
    def _required_params(self) -> "Distribution":
        p = self.params
        if self.distribution == "fixed":
            if "value" not in p:
                raise ValueError("fixed distribution needs params.value")
        elif self.distribution == "lognormal":
            if "mean" not in p or "sigma" not in p:
                raise ValueError("lognormal needs params.mean and params.sigma")
        elif "mean" not in p or ("stdev" not in p and "sigma" not in p):
            raise ValueError("normal needs params.mean and params.stdev (or sigma)")
        lo, hi = p.get("min"), p.get("max")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("params.min must be <= params.max")
        return self


class Source(StrictModel):
    kind: str
    config: dict = {}

    @model_validator(mode="after")
    def _known_kind(self) -> "Source":
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source.kind '{self.kind}' (allowed: {sorted(SOURCE_KINDS)})")
        return self


class Session(StrictModel):
    mode: Literal["open_loop", "sequential"]
    turns_per_session: Distribution
    prefix_strategy: Literal["append_delta"]
    think_time_ms: Distribution | None = None
    followup_input_length: Distribution | None = None  # defaults to input_length (§10.3)


class ManifestLists(StrictModel):
    modelled: list[str] = Field(min_length=1)
    not_modelled: list[str] = Field(min_length=1)


class Scenario(StrictModel):
    name: str
    summary: str
    maturity: Literal["established", "emerging", "exploratory"]
    source: Source
    input_length: Distribution
    output_length: Distribution
    thinking: bool = False
    session: Session
    manifest: ManifestLists
    modalities: list[str] = ["text"]

    @model_validator(mode="after")
    def _rules(self) -> "Scenario":
        if self.modalities != ["text"]:
            raise ValueError(
                f"scenario '{self.name}': modalities {self.modalities} not supported — "
                "v1 is text-only (§10.5); see TODOs.md *Multimodality*"
            )
        if self._can_be_multi_turn() and self.session.think_time_ms is None:
            raise ValueError(
                f"scenario '{self.name}': multi-turn scenarios require "
                "session.think_time_ms (§10.3)"
            )
        return self

    def _can_be_multi_turn(self) -> bool:
        d = self.session.turns_per_session
        if d.distribution == "fixed":
            return d.params["value"] > 1
        return d.params.get("max", float("inf")) > 1


def load_scenario(registry_dir: Path, slug: str) -> Scenario:
    path = registry_dir / f"{slug}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in registry_dir.glob("*.yaml"))
        raise FileNotFoundError(f"unregistered scenario '{slug}' (registered: {known})")
    with open(path) as f:
        scenario = Scenario.model_validate(yaml.safe_load(f))
    if scenario.name != slug:
        raise ValueError(f"{path}: 'name: {scenario.name}' does not match filename slug '{slug}'")
    return scenario
