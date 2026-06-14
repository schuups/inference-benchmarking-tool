"""Dataset generator core (SPECIFICATIONS.md §11).

Produces a deterministic prompt pool (prompts.jsonl, one record per turn) and
the scenario manifest (manifest.json, §14.7) from a validated BenchmarkConfig.
The §11.8 contract: same dataset_config + same registry revision + same
tokenizer -> byte-identical pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tools.common.config import BenchmarkConfig, MixEntry

from .registry import Distribution, Scenario, Session, load_scenario
from .sampling import class_rng, expected_mean, sample, sample_int, widen_for_thinking
from .sources import ConversationSource, TraceSource, make_source, trim_to_tokens
from .tokenizers import Tokenizer

POOL_FILENAME = "prompts.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class _ClassPlan:
    entry: MixEntry
    scenario: Scenario
    input_length: Distribution
    followup_input_length: Distribution
    output_length: Distribution
    session: Session
    expected_turns: float
    num_sessions: int = 0


def _effective_class(entry: MixEntry, scenario: Scenario) -> _ClassPlan:
    """Apply §11.4 per-class overrides on top of the registry entry."""
    input_length = (
        Distribution.model_validate(entry.input_length) if entry.input_length else scenario.input_length
    )
    output_length = (
        Distribution.model_validate(entry.output_length) if entry.output_length else scenario.output_length
    )
    session = (
        Session.model_validate({**scenario.session.model_dump(exclude_none=True), **entry.session})
        if entry.session
        else scenario.session
    )
    if scenario.thinking:
        output_length = widen_for_thinking(output_length)
    return _ClassPlan(
        entry=entry,
        scenario=scenario,
        input_length=input_length,
        followup_input_length=session.followup_input_length or input_length,
        output_length=output_length,
        session=session,
        expected_turns=expected_mean(session.turns_per_session),
    )


def _plan_classes(cfg: BenchmarkConfig, registry_dir: Path) -> list[_ClassPlan]:
    dc = cfg.dataset_config
    plans = [_effective_class(e, load_scenario(registry_dir, e.scenario)) for e in dc.scenario_mix]
    # §11.4: split num_prompts (total turn records) ∝ weight × E[turns_per_session],
    # i.e. sessions_c ∝ weight_c — so no class exhausts its sub-pool early.
    denominator = sum(p.entry.weight * p.expected_turns for p in plans)
    for p in plans:
        share = p.entry.weight * p.expected_turns / denominator
        p.num_sessions = max(1, round(dc.num_prompts * share / p.expected_turns))
    return plans


def _header(session: Session, session_idx: int) -> str:
    multi_turn = (
        session.turns_per_session.distribution != "fixed"
        or session.turns_per_session.params["value"] > 1
    )
    tag = "session" if multi_turn else "prompt"
    return f"[{tag}-{session_idx:06d}]"


def generate(
    cfg: BenchmarkConfig,
    tokenizer: Tokenizer,
    out_dir: Path,
    registry_dir: Path,
) -> dict:
    """Materialize the pool and manifest; returns the manifest dict."""
    dc = cfg.dataset_config
    plans = _plan_classes(cfg, registry_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    next_session_idx = 0
    for plan in plans:
        slug = plan.scenario.name
        source = make_source(plan.scenario.source, tokenizer)  # aborts per §11.1 on failure
        rng_in = class_rng(dc.seed, slug, "length_input")
        rng_out = class_rng(dc.seed, slug, "length_output")
        rng_sel = class_rng(dc.seed, slug, "selection")
        rng_turns = class_rng(dc.seed, slug, "turns")
        rng_think = class_rng(dc.seed, slug, "thinktime")

        conversational = isinstance(source, ConversationSource)
        trace_based = isinstance(source, TraceSource)
        max_turns = plan.session.turns_per_session.params.get("max")

        for _ in range(plan.num_sessions):
            session_idx = next_session_idx
            next_session_idx += 1
            if conversational:
                # §11.5: corpus turn boundaries drive the structure, clamped to
                # the declared turns ceiling.
                user_turns = source.conversation(rng_sel)
                if max_turns is not None:
                    user_turns = user_turns[: int(max_turns)]
                n_turns = len(user_turns)
            else:
                n_turns = sample_int(plan.session.turns_per_session, rng_turns)
            header = _header(plan.session, session_idx)
            for turn_idx in range(n_turns):
                length_dist = plan.input_length if turn_idx == 0 else plan.followup_input_length
                recorded_output: int | None = None
                if conversational:
                    # real content, clamped to the distribution's max bound (§11.5)
                    bound = length_dist.params.get("max")
                    body = user_turns[turn_idx]
                    if bound is not None:
                        budget = int(bound) - (tokenizer.count(header) if turn_idx == 0 else 0)
                        body = trim_to_tokens(body, max(1, budget), tokenizer)
                elif trace_based:
                    # recorded trace: question is the prompt; the answer's token
                    # count overrides output_length sampling (§11.5)
                    question, answer = source.trace(rng_sel)
                    bound = length_dist.params.get("max")
                    body = question
                    if bound is not None:
                        budget = int(bound) - (tokenizer.count(header) if turn_idx == 0 else 0)
                        body = trim_to_tokens(body, max(1, budget), tokenizer)
                    recorded_output = max(1, tokenizer.count(answer))
                else:
                    target = sample_int(length_dist, rng_in)
                    body = source.body(
                        max(1, target - tokenizer.count(header) if turn_idx == 0 else target),
                        rng_sel,
                    )
                text = f"{header} {body}" if turn_idx == 0 else body
                think_time_ms = (
                    round(sample(plan.session.think_time_ms, rng_think), 1)
                    if turn_idx > 0 and plan.session.think_time_ms is not None
                    else None
                )
                records.append(
                    {
                        "scenario": slug,
                        "session_idx": session_idx,
                        "session_mode": plan.session.mode,
                        "turn_idx": turn_idx,
                        "prompt_text": text,
                        "text_tokens": tokenizer.count(text),
                        "max_tokens": (
                            recorded_output
                            if recorded_output is not None
                            else sample_int(plan.output_length, rng_out)
                        ),
                        "think_time_ms": think_time_ms,
                    }
                )

    manifest = _build_manifest(cfg, plans, tokenizer)

    with open(out_dir / POOL_FILENAME, "w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    with open(out_dir / MANIFEST_FILENAME, "w") as f:
        json.dump(manifest, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


def _class_assumptions(plan: _ClassPlan) -> list[str]:
    s = plan.scenario
    out: list[str] = [
        f"input length distribution: {plan.input_length.distribution} {plan.input_length.params}",
        f"output length distribution: {plan.output_length.distribution} {plan.output_length.params}"
        + (" (widened for thinking per §11.6)" if s.thinking else ""),
    ]
    if plan.followup_input_length != plan.input_length:
        out.insert(
            1,
            "follow-up input length distribution: "
            f"{plan.followup_input_length.distribution} {plan.followup_input_length.params}",
        )
    out += [
        f"turns per session: {plan.session.turns_per_session.distribution} "
        f"{plan.session.turns_per_session.params}",
        f"session mode: {plan.session.mode}",
        f"prefix strategy: {plan.session.prefix_strategy}",
        f"source: {s.source.kind} {s.source.config or ''}".rstrip(),
    ]
    if plan.session.think_time_ms is not None:
        out.append(
            f"think time ms: {plan.session.think_time_ms.distribution} {plan.session.think_time_ms.params}"
        )
    if s.source.kind == "wildchat":
        out.append(
            "session structure and per-turn lengths driven by real conversation "
            "content; lengths clamped to the declared distribution max bounds (§11.5)"
        )
    if s.source.kind == "reasoning_trace_replay":
        out.append(
            "output lengths replayed from recorded reasoning traces — overrides "
            "the declared output_length distribution (§11.5)"
        )
    return out


def _build_manifest(cfg: BenchmarkConfig, plans: list[_ClassPlan], tokenizer: Tokenizer) -> dict:
    """§14.7: mix (+ expected request share), per-class disclosure, run assumptions."""
    weighted = [(p, p.entry.weight * p.expected_turns) for p in plans]
    total = sum(w for _, w in weighted)
    mix = [
        {
            "scenario": p.scenario.name,
            "weight": p.entry.weight,
            "expected_request_share": round(w / total, 4),
        }
        for p, w in weighted
    ]
    classes = [
        {
            "name": p.scenario.name,
            "weight": p.entry.weight,
            "summary": p.scenario.summary,
            "maturity": p.scenario.maturity,
            "modelled": p.scenario.manifest.modelled,
            "not_modelled": p.scenario.manifest.not_modelled,
            "assumptions": _class_assumptions(p),
        }
        for p in plans
    ]
    ap = cfg.arrival_process
    arrival = f"arrival process: {ap.kind}"
    if ap.kind == "burst_mmpp":
        arrival += (
            f" (burst_factor={ap.burst_factor}, mean_burst_s={ap.mean_burst_s}, "
            f"mean_idle_s={ap.mean_idle_s})"
        )
    run_assumptions = [
        arrival + " — λ counts session starts (§12.3)",
        f"routing strategy: {cfg.routing_strategy}",
        f"output_length_mode: {cfg.dataset_config.output_length_mode}",
        f"master seed: {cfg.dataset_config.seed}",
        f"tokenizer: {tokenizer.tokenizer_id}",
    ]
    return {"mix": mix, "classes": classes, "run_assumptions": run_assumptions}
