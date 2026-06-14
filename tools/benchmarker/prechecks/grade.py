"""Parse + grade system pre-check measurements (SPECIFICATIONS.md §8.3-§8.4).

Consumes the raw benchmark outputs captured by run_system_prechecks.sh, parses
per-metric values, grades them against tools/system_prechecks_reference.yaml,
and emits JSON rows shaped for the `system_prechecks` table (§14.6) plus an
overall gate outcome.

Grading (§8.3/§8.4):
- tolerance_pct < 0 (higher is better): warn when measured < (1+tol/100)·expected;
  fail when measured < 0.5·expected.
- tolerance_pct > 0 (lower is better): warn when measured > (1+tol/100)·expected;
  fail when measured > 2·expected.
- expected TBD / no matching reference: informational — recorded with
  expected=None and status "pass" (the gate is unenforceable, §8.3).
- benchmark errored / output unparseable: status "fail" with measured=None.

Exit codes (consumed by run_system_prechecks.sh in the `&& exec <engine>` chain):
0 = proceed; 3 = warn + on_warn=abort; 4 = fail + on_fail=abort.

NVSHMEM perftest output shapes vary across SDK versions; the parsers here are
deliberately tolerant and the fixtures must be re-captured from the real engine
image at E1 (same discipline as the vLLM log parser).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "system_prechecks_reference.yaml"

_SIZE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
_BW_TO_GBS = {"GB/s": 1.0, "MB/s": 1e-3, "KB/s": 1e-6}


def size_label_to_bytes(label: str) -> int | None:
    m = re.match(r"\s*([\d.]+)\s*(B|KiB|MiB|GiB)\b", label)
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2)]) if m else None


# ------------------------------------------------------------------- parsers


def parse_nccl_output(text: str, target_bytes: int) -> float | None:
    """Out-of-place busbw (GB/s) at the target message size from nccl-tests output."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit() and int(parts[0]) == target_bytes:
            try:
                return float(parts[7])  # size count type redop root time algbw busbw ...
            except ValueError:
                return None
    return None


def parse_nvshmem_output(text: str, target_bytes: int, want: str = "latency") -> float | None:
    """Value at the target message-size row of an NVSHMEM perftest table.

    The perftest prints one or more tables under a header like:
        size(B)  count  type  scope  latency(us)  algbw(GB/s)  busbw(GB/s)
    `want` selects the column: "latency" -> latency(us); "bw" -> busbw/bandwidth.
    The column index is taken from the header (robust to spacing/layout), and the
    last matching table wins (the final / most-representative scope).
    """
    col = None
    found = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("size("):  # header row -> (re)locate the wanted column
            col = next(
                (i for i, name in enumerate(parts)
                 if (want == "latency" and name.startswith("latency"))
                 or (want == "bw" and ("busbw" in name or name.startswith("bw")))),
                None,
            )
            continue
        if col is None:
            continue
        try:
            size = int(float(parts[0]))
            value = float(parts[col])
        except (ValueError, IndexError):
            continue
        if size == target_bytes:
            found = value
    return found


def parse_dd_output(text: str) -> float | None:
    """Read bandwidth (GB/s) from dd's summary line."""
    m = re.search(r"copied,\s*[\d.]+\s*s,\s*([\d.]+)\s*(GB/s|MB/s|KB/s)", text)
    if not m:
        return None
    return float(m.group(1)) * _BW_TO_GBS[m.group(2)]


def nvshmem_max_busbw(text: str) -> float:
    """Largest busbw(GB/s) across every NVSHMEM perftest row.

    busbw is derived from algbw and the PE count, so it is identically 0 when the
    run wired up a single PE (the 2(n-1)/n factor is 0 at n=1). A whole-table max
    of 0 is therefore a reliable "degenerate single-PE run" signal — the latency
    column is then a local copy, not a real collective (see parse_nvshmem_output
    for the table layout)."""
    col = None
    best = 0.0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("size("):
            col = next((i for i, n in enumerate(parts) if "busbw" in n), None)
            continue
        if col is None:
            continue
        try:
            best = max(best, float(parts[col]))
        except (ValueError, IndexError):
            continue
    return best


# ------------------------------------------------------------------- grading


def load_reference(cluster: str, path: Path = REFERENCE_PATH) -> list[dict]:
    with open(path) as f:
        rows = yaml.safe_load(f)
    return [r for r in rows if r["cluster"] == cluster]


def _expected_value(raw) -> float | None:
    m = re.match(r"\s*([\d.]+)", str(raw))
    return float(m.group(1)) if m else None  # "TBD ..." -> None (informational)


def find_reference(refs: list[dict], benchmark: str, scope: str) -> dict | None:
    for r in refs:
        if r["benchmark"] == benchmark and r["scope"] == scope:
            return r
    return None


def grade(measured: float | None, expected: float | None, tolerance_pct: float | None) -> str:
    if measured is None:
        return "fail"  # §8.4: the benchmark itself errored
    if expected is None or tolerance_pct is None:
        return "pass"  # informational — gate unenforceable until characterised (§8.3)
    if tolerance_pct < 0:  # higher is better
        if measured < 0.5 * expected:
            return "fail"
        if measured < (1 + tolerance_pct / 100) * expected:
            return "warn"
    else:  # lower is better
        if measured > 2 * expected:
            return "fail"
        if measured > (1 + tolerance_pct / 100) * expected:
            return "warn"
    return "pass"


def metric_id(benchmark: str, size_label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{benchmark} {size_label}".lower()).strip("_")
    return slug


def build_rows(measurements: list[dict], refs: list[dict]) -> list[dict]:
    """measurements: [{benchmark, scope, size, measured}, ...] -> §14.6 rows."""
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for m in measurements:
        ref = find_reference(refs, m["benchmark"], m["scope"])
        expected = _expected_value(ref["expected"]) if ref else None
        tolerance = ref.get("tolerance_pct") if ref else None
        rows.append(
            {
                "metric": metric_id(m["benchmark"], m["size"]),
                "measured": m["measured"],
                "expected": expected,
                "tolerance_pct": tolerance if expected is not None else None,
                "status": grade(m["measured"], expected, tolerance if expected is not None else None),
                "ts": ts,
            }
        )
    return rows


def outcome_exit_code(rows: list[dict], on_warn: str, on_fail: str) -> int:
    statuses = {r["status"] for r in rows}
    if "fail" in statuses and on_fail == "abort":
        return 4
    if "warn" in statuses and on_warn == "abort":
        return 3
    return 0


# ----------------------------------------------------------------------- CLI


def collect_measurements(out_dir: Path, cluster: str, scope: str, storage_scope: str, refs: list[dict]) -> list[dict]:
    """Map captured output files onto reference benchmark names."""
    is_amd = any(r["benchmark"].startswith("RCCL") for r in refs)
    coll_prefix = "RCCL" if is_amd else "NCCL"
    shmem_prefix = "ROCm SHMEM" if is_amd else "NVSHMEM"
    measurements = []

    for path in sorted(out_dir.glob("collective_*.out")):
        name = path.stem.removeprefix("collective_")
        benchmark = f"{coll_prefix} {name}"
        ref = find_reference(refs, benchmark, scope)
        size_label = ref["size"] if ref else "128 MiB"
        target = size_label_to_bytes(size_label) or 128 * 1024**2
        measurements.append(
            {"benchmark": benchmark, "scope": scope, "size": size_label,
             "measured": parse_nccl_output(path.read_text(), target)}
        )
    for path, suffix, want, default_size in [
        (out_dir / "nvshmem_alltoall_latency.out", "alltoall_latency", "latency", "128 KiB"),
        (out_dir / "nvshmem_put_bw.out", "shmem_put_bw", "bw", "128 MiB"),
    ]:
        if not path.exists():
            continue  # skipped-with-warning path (§8.1) — no row, orchestrator logs it
        text = path.read_text()
        if "perftest not found" in text or not text.strip():
            # §8.1 skip: the runner tees the warning into the capture file, so
            # absence-of-file is not the only skip signal — content is.
            continue
        if "requires exactly" in text or nvshmem_max_busbw(text) == 0.0:
            # Degenerate single-PE run: the perftest wired up only 1 PE (PMIx
            # bootstrap failed). A collective then reports busbw ≡ 0 and a
            # pt-to-pt test aborts with "requires exactly two processes" — either
            # way the value is not a real multi-PE measurement, so skip the row
            # rather than record a misleading number (§8.1). run-nvshmem.sh logs
            # the cause into the captured .out.
            continue
        benchmark = f"{shmem_prefix} {suffix}"
        ref = find_reference(refs, benchmark, scope)
        size_label = ref["size"] if ref else default_size
        target = size_label_to_bytes(size_label) or size_label_to_bytes(default_size)
        measurements.append(
            {"benchmark": benchmark, "scope": scope, "size": size_label,
             "measured": parse_nvshmem_output(text, target, want)}
        )
    storage = out_dir / "storage_read.out"
    if storage.exists():
        measurements.append(
            {"benchmark": "Sequential read", "scope": storage_scope, "size": "1 MiB blocks",
             "measured": parse_dd_output(storage.read_text())}
        )
    return measurements


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade §8 pre-check outputs")
    parser.add_argument("--out-dir", type=Path, required=True, help="dir with captured *.out files")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--scope", required=True, help='reference scope, e.g. "4× GH200, 1 node"')
    parser.add_argument("--storage-scope", default="", help="reference scope of the weights mount")
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--on-warn", choices=["abort", "continue"], default="abort")
    parser.add_argument("--on-fail", choices=["abort", "continue"], default="abort")
    parser.add_argument("--smoke", action="store_true", help="smoke-test mode flag (§8.2 cache miss)")
    parser.add_argument("--results", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    refs = load_reference(args.cluster, args.reference)
    measurements = collect_measurements(args.out_dir, args.cluster, args.scope, args.storage_scope, refs)
    rows = build_rows(measurements, refs)
    code = outcome_exit_code(rows, args.on_warn, args.on_fail)
    payload = {
        "cluster": args.cluster,
        "scope": args.scope,
        "smoke_test_mode": args.smoke,
        "gate_exit_code": code,
        "rows": rows,
    }
    args.results.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(f"[prechecks] {row['status']:4s} {row['metric']}: measured={row['measured']} expected={row['expected']}")
    if args.smoke:
        print("[prechecks] SMOKE-TEST MODE: collective-tests cache was cold — results will NOT be persisted (§8.2)")
    return code


if __name__ == "__main__":
    sys.exit(main())
