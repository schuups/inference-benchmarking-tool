#!/usr/bin/env python3
"""Hardware telemetry sampler (SPECIFICATIONS.md §13.3) — single-file, stdlib-only.

Runs ON the inference-server nodes, backgrounded inside the engine container
session before `exec <engine>` (plan M3/M6; review finding H3). Therefore:
NO third-party imports, NO repo imports — it must execute in any engine image
with a bare python3. Signals a platform cannot expose are emitted as null and
stored as NULL (§13.3).

Output: NDJSON, one node-scoped row (gpu_index=null) plus one row per GPU per
tick. Ingested into `hardware_stats` by the Benchmarker at finalisation
(tools/common/results_db.py ingest_hardware_ndjson).

Usage:
    python3 hw_sampler.py --out /scratch/<run>/hw-$(hostname).ndjson \
        [--interval 1.0] [--duration 3600]

DCGM profiling counters (sm/tensor activity, DRAM/NVLink/PCIe bandwidth) are
emitted as null in this version: wiring `dcgmi dmon` blind — with no GH200 to
validate the parse against — would violate the parser-fixture discipline. They
get wired and fixture-tested at E1 on a real node (plan M3 DoD). `rocm-smi`
support is best-effort until beverin runs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import socket
import subprocess
import sys
import time

GPU_FIELDS = (
    "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_pct", "gpu_power_w", "gpu_temp_c",
    "gpu_sm_active_pct", "gpu_tensor_active_pct", "gpu_dram_bw_gbs",
    "nvlink_rx_gbs", "nvlink_tx_gbs", "pcie_rx_gbs", "pcie_tx_gbs",
)
NODE_FIELDS = (
    "cpu_util_pct", "cpu_iowait_pct", "ram_used_gb", "ram_pct", "ram_bw_gbs",
    "storage_read_gbs", "storage_read_iops", "net_rx_gbs", "net_tx_gbs",
)

_PARTITION_RE = re.compile(r"^(sd[a-z]+\d+|nvme\d+n\d+p\d+|mmcblk\d+p\d+)$")
_SKIP_DEV_RE = re.compile(r"^(loop|ram|zram|dm-|md)")


# ----------------------------------------------------------- /proc parsers
# Pure functions over file contents -> testable on any platform.


def parse_proc_stat(text: str) -> tuple[int, int, int] | None:
    """Returns (busy_ticks, iowait_ticks, total_ticks) from the aggregate cpu line."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = [int(x) for x in line.split()[1:]]
            if len(parts) < 5:
                return None  # malformed aggregate cpu line
            idle, iowait = parts[3], parts[4]
            total = sum(parts)
            return total - idle - iowait, iowait, total
    return None


def cpu_pcts(prev: tuple | None, cur: tuple | None) -> tuple[float | None, float | None]:
    if not prev or not cur or cur[2] <= prev[2]:
        return None, None
    dtotal = cur[2] - prev[2]
    return 100.0 * (cur[0] - prev[0]) / dtotal, 100.0 * (cur[1] - prev[1]) / dtotal


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    """(ram_used_gb, ram_pct) from MemTotal/MemAvailable."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            values[parts[0].rstrip(":")] = int(parts[1])  # kB
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None, None
    used_kb = values["MemTotal"] - values["MemAvailable"]
    return used_kb / 1024**2, 100.0 * used_kb / values["MemTotal"]


def parse_diskstats(text: str) -> tuple[int, int]:
    """(reads_completed, sectors_read) summed over physical, non-partition devices."""
    reads = sectors = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        name = parts[2]
        if _SKIP_DEV_RE.match(name) or _PARTITION_RE.match(name):
            continue
        reads += int(parts[3])
        sectors += int(parts[5])
    return reads, sectors


def parse_net_dev(text: str) -> tuple[int, int]:
    """(rx_bytes, tx_bytes) summed over non-loopback interfaces."""
    rx = tx = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        if name.strip() == "lo":
            continue
        parts = rest.split()
        rx += int(parts[0])
        tx += int(parts[8])
    return rx, tx


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


# ------------------------------------------------------------ GPU queries


_NVSMI_QUERY = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Rows from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        util, mem_used, mem_total, power, temp = (_maybe_float(p) for p in parts)
        rows.append(
            {
                "gpu_util_pct": util,
                "gpu_mem_used_gb": mem_used / 1024 if mem_used is not None else None,
                "gpu_mem_pct": (
                    100.0 * mem_used / mem_total
                    if mem_used is not None and mem_total
                    else None
                ),
                "gpu_power_w": power,
                "gpu_temp_c": temp,
            }
        )
    return rows


def _maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:  # nvidia-smi prints "[N/A]" for unsupported fields
        return None


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def query_gpus() -> list[dict]:
    out = _run(["nvidia-smi", f"--query-gpu={_NVSMI_QUERY}", "--format=csv,noheader,nounits"])
    if out is not None:
        return parse_nvidia_smi_csv(out)
    # AMD path (best-effort until validated on beverin): rocm-smi JSON
    out = _run(["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--showtemp", "--json"])
    if out is not None:
        return _parse_rocm_smi_json(out)
    return []


def _parse_rocm_smi_json(text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = []
    for key in sorted(k for k in data if k.startswith("card")):
        card = data[key]

        def grab(*names):
            for n in names:
                for k, v in card.items():
                    if n.lower() in k.lower():
                        try:
                            return float(str(v).rstrip("%W° c"))
                        except ValueError:
                            pass
            return None

        rows.append(
            {
                "gpu_util_pct": grab("GPU use"),
                "gpu_mem_used_gb": None,
                "gpu_mem_pct": grab("GPU Memory Allocated", "GPU memory use"),
                "gpu_power_w": grab("Average Graphics Package Power", "Current Socket Graphics Package Power"),
                "gpu_temp_c": grab("Temperature (Sensor junction)", "Temperature (Sensor edge)"),
            }
        )
    return rows


# --------------------------------------------------------------- sampler


class Sampler:
    def __init__(self) -> None:
        self.node = socket.gethostname()
        self._prev_cpu: tuple | None = None
        self._prev_disk: tuple[int, int] | None = None
        self._prev_net: tuple[int, int] | None = None
        self._prev_t: float | None = None

    def tick(self) -> list[dict]:
        now = time.monotonic()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dt = now - self._prev_t if self._prev_t is not None else None

        node_row: dict = {"ts": ts, "node": self.node, "gpu_index": None}
        node_row.update({f: None for f in NODE_FIELDS})
        node_row.update({f: None for f in GPU_FIELDS})

        stat = _read("/proc/stat")
        cur_cpu = parse_proc_stat(stat) if stat else None
        util, iowait = cpu_pcts(self._prev_cpu, cur_cpu)
        node_row["cpu_util_pct"], node_row["cpu_iowait_pct"] = util, iowait
        self._prev_cpu = cur_cpu

        meminfo = _read("/proc/meminfo")
        if meminfo:
            node_row["ram_used_gb"], node_row["ram_pct"] = parse_meminfo(meminfo)

        disk = _read("/proc/diskstats")
        if disk:
            cur_disk = parse_diskstats(disk)
            if self._prev_disk and dt and dt > 0:
                node_row["storage_read_iops"] = (cur_disk[0] - self._prev_disk[0]) / dt
                node_row["storage_read_gbs"] = (cur_disk[1] - self._prev_disk[1]) * 512 / dt / 1e9
            self._prev_disk = cur_disk

        net = _read("/proc/net/dev")
        if net:
            cur_net = parse_net_dev(net)
            if self._prev_net and dt and dt > 0:
                node_row["net_rx_gbs"] = (cur_net[0] - self._prev_net[0]) / dt / 1e9
                node_row["net_tx_gbs"] = (cur_net[1] - self._prev_net[1]) / dt / 1e9
            self._prev_net = cur_net

        self._prev_t = now
        rows = [node_row]
        for index, gpu in enumerate(query_gpus()):
            gpu_row = {"ts": ts, "node": self.node, "gpu_index": index}
            gpu_row.update({f: None for f in GPU_FIELDS})
            gpu_row.update({f: None for f in NODE_FIELDS})
            gpu_row.update(gpu)
            rows.append(gpu_row)
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="§13.3 hardware sampler (stdlib-only)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None, help="seconds; default: until killed")
    args = parser.parse_args()

    sampler = Sampler()
    start = time.monotonic()
    with open(args.out, "a", buffering=1) as out:
        while args.duration is None or time.monotonic() - start < args.duration:
            tick_started = time.monotonic()
            for row in sampler.tick():
                out.write(json.dumps(row, sort_keys=True) + "\n")
            sleep_s = args.interval - (time.monotonic() - tick_started)
            if sleep_s > 0:
                time.sleep(sleep_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
