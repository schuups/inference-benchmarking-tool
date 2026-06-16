"""M4 DoD (local half): parsers against fixtures, §8.3/§8.4 grading, gate policy.

Container execution on clariden is the cluster half, validated at E1.
NVSHMEM fixtures are provisional shapes — re-capture from the engine image at E1.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.benchmarker.prechecks.grade import (
    REFERENCE_PATH,
    build_rows,
    collect_measurements,
    grade,
    load_reference,
    metric_id,
    nvshmem_max_busbw,
    outcome_exit_code,
    parse_dd_output,
    parse_nccl_output,
    parse_nvshmem_output,
    parse_storage_parallel,
    size_label_to_bytes,
)

STORAGE_PARALLEL_FIXTURE = (
    "PARALLEL_READ streams=8 bytes=17179869184 t0=1000.0 t1=1003.4567\n"
    "seconds=3.4567 gbps=4.9700\n"
    "--- per-stream dd ---\n"
    "2147483648 bytes (2.1 GB, 2.0 GiB) copied, 3.4 s, 0.63 GB/s\n"
)

STORAGE_BUFFERED_FIXTURE = (
    "BUFFERED_READ streams=8 bytes=17179869184 t0=2000.0 t1=2006.8700\n"
    "seconds=6.8700 gbps=2.5000\n"
    "--- per-stream dd ---\n"
)

NCCL_FIXTURE = """\
# nThread 1 nGpus 1 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    67108864      16777216     float     sum      -1    825.1   81.33  121.99      0    824.0   81.44  122.16      0
   134217728      33554432     float     sum      -1   1571.2   85.42  128.13      0   1569.8   85.50  128.25      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 95.21
"""

# Real NVSHMEM perftest layout, captured on the Alps gh200 image at E2a (2026-06-14):
# a preamble then one or more tables headed "size(B) count type scope latency(us) ...".
# A genuine multi-PE run: busbw is non-zero (it is 0 only when a single PE wired up).
NVSHMEM_LATENCY_FIXTURE = """\
Runtime options after parsing command line arguments
NVSHMEM v3.6.5
mype: 0 mype_node: 0 device name: NVIDIA GH200 120GB bus id: 1
#alltoall_device
size(B)     count     type      scope     latency(us)       algbw(GB/s)   busbw(GB/s)
1024        256       32-bit    thread    8.120000          0.119         0.089
131072      32768     32-bit    thread    19.460000         12.760        19.140
"""

NVSHMEM_PUT_BW_FIXTURE = """\
#shmem_put_bw
size(B)     count     type      scope     latency(us)       algbw(GB/s)   busbw(GB/s)
1048576     1         32-bit    -         45.000000         12.300        12.300
134217728   1         32-bit    -         120.000000        23.450        23.450
"""

# Degenerate single-PE alltoall (PMIx wire-up failed): busbw ≡ 0 across the table.
# Captured at E2a run 2, before the run-nvshmem.sh PMIx-reset fix. The latency
# column is a local copy, not a collective — grading must SKIP it, not record it.
NVSHMEM_DEGENERATE_FIXTURE = """\
NVSHMEM v3.6.5
mype: 0 mype_node: 0 device name: NVIDIA GH200 120GB bus id: 1
#alltoall_device
size(B)     count     type      scope     latency(us)       algbw(GB/s)   busbw(GB/s)
131072      16384     64-bit    block     10.256000         12.780        0.000
4194304     524288    64-bit    block     197.980797        21.185        0.000
"""

# pt-to-pt put_bw launched with a single PE: the binary aborts before any table.
NVSHMEM_PUTBW_1PE_FIXTURE = """\
[nvshmem] using /opt/nvshmem/bin/perftest
This test requires exactly two processes
[/tmp/nvshmem-src/perftest/common/utils.cu:614] cuda failed with invalid argument
"""

DD_FIXTURE = "4096+0 records in\n4096+0 records out\n4294967296 bytes (4.3 GB, 4.0 GiB) copied, 2.951 s, 1.5 GB/s\n"


def test_size_label_to_bytes():
    assert size_label_to_bytes("128 MiB") == 134217728
    assert size_label_to_bytes("128 KiB") == 131072
    assert size_label_to_bytes("1 MiB blocks") == 1048576
    assert size_label_to_bytes("nonsense") is None


def test_parse_nccl_output_busbw_at_target():
    assert parse_nccl_output(NCCL_FIXTURE, 134217728) == pytest.approx(128.13)
    assert parse_nccl_output(NCCL_FIXTURE, 67108864) == pytest.approx(121.99)
    assert parse_nccl_output(NCCL_FIXTURE, 999) is None
    assert parse_nccl_output("garbage\n", 134217728) is None


def test_parse_nvshmem_output():
    # latency column at the target size (128 KiB)
    assert parse_nvshmem_output(NVSHMEM_LATENCY_FIXTURE, 131072) == pytest.approx(19.46)
    assert parse_nvshmem_output(NVSHMEM_LATENCY_FIXTURE, 1024) == pytest.approx(8.12)
    assert parse_nvshmem_output(NVSHMEM_LATENCY_FIXTURE, 7) is None
    # bandwidth column (busbw) at the target size (128 MiB)
    assert parse_nvshmem_output(NVSHMEM_PUT_BW_FIXTURE, 134217728, "bw") == pytest.approx(23.45)


def test_parse_dd_output_units():
    assert parse_dd_output(DD_FIXTURE) == pytest.approx(1.5)
    assert parse_dd_output(DD_FIXTURE.replace("1.5 GB/s", "750 MB/s")) == pytest.approx(0.75)
    assert parse_dd_output("dd: error reading") is None


def test_grading_rules():
    # higher is better (tolerance -10): pass / warn / fail bands (§8.3-§8.4)
    assert grade(100.0, 100.0, -10) == "pass"
    assert grade(91.0, 100.0, -10) == "pass"
    assert grade(85.0, 100.0, -10) == "warn"
    assert grade(49.0, 100.0, -10) == "fail"      # < 50% of expected
    # lower is better (tolerance +20)
    assert grade(100.0, 100.0, +20) == "pass"
    assert grade(125.0, 100.0, +20) == "warn"
    assert grade(201.0, 100.0, +20) == "fail"     # > 2x expected
    # informational: no reference yet (§8.3 TBD)
    assert grade(100.0, None, None) == "pass"
    # benchmark errored
    assert grade(None, 100.0, -10) == "fail"


def test_reference_loads_real_yaml():
    refs = load_reference("clariden", REFERENCE_PATH)
    assert len(refs) == 15  # 1-node (3 NCCL + 2 NVSHMEM) + 2-node (3 NCCL + sendrecv + 2 NVSHMEM) + 2 mounts × (seq + parallel)
    assert all(r["cluster"] == "clariden" for r in refs)
    # clariden 4× GH200 1-node NCCL is characterised at E2a -> enforceable grading
    rows = build_rows(
        [{"benchmark": "NCCL all_reduce", "scope": "4× GH200, 1 node", "size": "128 MiB", "measured": 100.0}],
        refs,
    )
    assert rows[0]["expected"] == pytest.approx(317.7)
    assert rows[0]["status"] == "fail"  # 100 < 0.5 × 317.7
    # the 2-node ladder is now characterised at E2b -> enforceable too
    rows_2n = build_rows(
        [{"benchmark": "NCCL all_reduce", "scope": "8× GH200, 2 nodes", "size": "128 MiB", "measured": 50.0}],
        refs,
    )
    assert rows_2n[0]["expected"] == pytest.approx(131.1)
    assert rows_2n[0]["status"] == "fail"  # 50 < 0.5 × 131.1
    # still-TBD entries (e.g. NVSHMEM shmem_put_bw) remain informational
    rows_tbd = build_rows(
        [{"benchmark": "NVSHMEM shmem_put_bw", "scope": "8× GH200, 2 nodes", "size": "128 MiB", "measured": 5.0}],
        refs,
    )
    assert rows_tbd[0]["expected"] is None and rows_tbd[0]["status"] == "pass"


def test_collect_measurements_sendrecv_2node(tmp_path):
    """The PP-link `sendrecv` collective maps to 'NCCL sendrecv' at the 2-node scope
    with no grade.py change (collective_*.out → 'NCCL <name>')."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "collective_sendrecv.out").write_text(NCCL_FIXTURE)
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(out_dir, "clariden", "8× GH200, 2 nodes", "", refs)
    by = {m["benchmark"]: m for m in measurements}
    assert by["NCCL sendrecv"]["measured"] == pytest.approx(128.13)
    assert by["NCCL sendrecv"]["scope"] == "8× GH200, 2 nodes"


def test_outcome_exit_codes():
    rows_ok = [{"status": "pass"}]
    rows_warn = [{"status": "pass"}, {"status": "warn"}]
    rows_fail = [{"status": "warn"}, {"status": "fail"}]
    assert outcome_exit_code(rows_ok, "abort", "abort") == 0
    assert outcome_exit_code(rows_warn, "abort", "abort") == 3
    assert outcome_exit_code(rows_warn, "continue", "abort") == 0
    assert outcome_exit_code(rows_fail, "abort", "abort") == 4
    assert outcome_exit_code(rows_fail, "abort", "continue") == 3  # warn still aborts
    assert outcome_exit_code(rows_fail, "continue", "continue") == 0


def test_metric_id_slug():
    assert metric_id("NCCL all_reduce", "128 MiB") == "nccl_all_reduce_128_mib"
    assert metric_id("ROCm SHMEM alltoall_latency", "128 KiB") == "rocm_shmem_alltoall_latency_128_kib"


def _write_fixture_outputs(out_dir: Path):
    out_dir.mkdir()
    for c in ("all_reduce", "all_gather", "alltoall"):
        (out_dir / f"collective_{c}.out").write_text(NCCL_FIXTURE)
    (out_dir / "nvshmem_alltoall_latency.out").write_text(NVSHMEM_LATENCY_FIXTURE)
    (out_dir / "storage_read.out").write_text(DD_FIXTURE)
    (out_dir / "storage_parallel.out").write_text(STORAGE_PARALLEL_FIXTURE)
    (out_dir / "storage_buffered.out").write_text(STORAGE_BUFFERED_FIXTURE)


def test_parse_storage_parallel():
    assert parse_storage_parallel(STORAGE_PARALLEL_FIXTURE) == pytest.approx(4.97)
    assert parse_storage_parallel("no summary here\n") is None


def test_collect_measurements_maps_files_to_reference_rows(tmp_path):
    out_dir = tmp_path / "out"
    _write_fixture_outputs(out_dir)
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(
        out_dir, "clariden", "4× GH200, 1 node", "capstor weights mount (Lustre, HDD)", refs
    )
    by_benchmark = {m["benchmark"]: m for m in measurements}
    assert by_benchmark["NCCL all_reduce"]["measured"] == pytest.approx(128.13)
    assert by_benchmark["NVSHMEM alltoall_latency"]["measured"] == pytest.approx(19.46)
    assert by_benchmark["Sequential read"]["measured"] == pytest.approx(1.5)
    # parallel aggregate read recorded at the (storage) scope, comparable to vLLM load
    assert by_benchmark["Parallel read"]["measured"] == pytest.approx(4.97)
    assert by_benchmark["Parallel read"]["scope"] == "capstor weights mount (Lustre, HDD)"
    # buffered (readahead) aggregate — informational, no reference row
    assert by_benchmark["Buffered read"]["measured"] == pytest.approx(2.5)
    assert by_benchmark["Buffered read"]["scope"] == "capstor weights mount (Lustre, HDD)"
    # nvshmem_put_bw.out absent -> skipped-with-warning, no row (§8.1)
    assert "NVSHMEM shmem_put_bw" not in by_benchmark


def test_nvshmem_max_busbw():
    assert nvshmem_max_busbw(NVSHMEM_LATENCY_FIXTURE) == pytest.approx(19.14)
    assert nvshmem_max_busbw(NVSHMEM_DEGENERATE_FIXTURE) == 0.0
    assert nvshmem_max_busbw("garbage with no table\n") == 0.0


def test_nvshmem_degenerate_single_pe_skipped(tmp_path):
    """busbw ≡ 0 (collective) or "requires exactly two processes" (pt-to-pt) means
    the perftest wired up a single PE — the value is not a real measurement and
    must be skipped, not recorded (§8.1)."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "nvshmem_alltoall_latency.out").write_text(NVSHMEM_DEGENERATE_FIXTURE)
    (out_dir / "nvshmem_put_bw.out").write_text(NVSHMEM_PUTBW_1PE_FIXTURE)
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(out_dir, "clariden", "4× GH200, 1 node", "", refs)
    assert not any("NVSHMEM" in m["benchmark"] for m in measurements)


def test_nvshmem_put_bw_collected_when_multi_pe(tmp_path):
    """A genuine 2-PE put_bw run (non-zero busbw) is parsed and recorded."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "nvshmem_put_bw.out").write_text(NVSHMEM_PUT_BW_FIXTURE)
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(out_dir, "clariden", "4× GH200, 1 node", "", refs)
    by_benchmark = {m["benchmark"]: m for m in measurements}
    assert by_benchmark["NVSHMEM shmem_put_bw"]["measured"] == pytest.approx(23.45)


def test_nvshmem_skip_detected_by_content(tmp_path):
    """E1 attempt #4: the runner tees the §8.1 skip warning INTO the capture
    file, so a 'skipped' NVSHMEM must not grade as fail."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "nvshmem_alltoall_latency.out").write_text(
        "[nvshmem] NVSHMEM perftest not found in this engine image\n"
        "[nvshmem]   tried: /opt/nvshmem/bin/perftest ...\n"
    )
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(out_dir, "clariden", "4× GH200, 1 node", "", refs)
    assert not any("NVSHMEM" in m["benchmark"] for m in measurements)


def test_grade_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    _write_fixture_outputs(out_dir)
    results = tmp_path / "results.json"
    proc = subprocess.run(
        [
            sys.executable, "tools/benchmarker/prechecks/grade.py",
            "--out-dir", str(out_dir), "--cluster", "bristen",
            "--scope", "4× A100, 1 node",
            "--storage-scope", "capstor weights mount (Lustre, HDD)",
            "--results", str(results), "--smoke",
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    # bristen is still fully TBD -> informational pass (clariden 1-node is now enforceable)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(results.read_text())
    assert payload["smoke_test_mode"] is True
    assert len(payload["rows"]) == 7  # 3 NCCL + NVSHMEM alltoall + Sequential + Parallel + Buffered read
    assert "SMOKE-TEST MODE" in proc.stdout


def test_runner_script_syntax():
    for script in Path("tools/benchmarker/prechecks").glob("*.sh"):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script}: {proc.stderr}"


# --- K8s torch.distributed collective probe (decision 9): the probe emits nccl-tests-shaped
#     rows so grade.py is reused unchanged. Verify that contract + the busbw factors here;
#     the GPU measurement itself is validated on-cluster (no torch off-GPU). ---

def test_probe_line_roundtrips_through_grader():
    from tools.benchmarker.prechecks.collective_probe import format_nccl_line

    size = 128 * 1024**2
    line = format_nccl_line(size, time_us=12345.6, algbw_gbs=200.0, busbw_gbs=317.70)
    # grade.parse_nccl_output keys on the size column and returns the busbw (field 7).
    assert parse_nccl_output(line, size) == pytest.approx(317.70)
    assert parse_nccl_output(line, size * 2) is None  # wrong target size → no match

    # End-to-end: a probe row grades against the populated clariden 1-node reference.
    refs = load_reference("clariden", REFERENCE_PATH)
    hdr = "# size count type redop root time algbw busbw wrong"
    text = hdr + "\n" + format_nccl_line(size, 1000.0, 200.0, 300.0) + "\n"
    measurements = [{"benchmark": "NCCL all_reduce", "scope": "4× GH200, 1 node",
                     "size": "128 MiB", "measured": parse_nccl_output(text, size)}]
    rows = build_rows(measurements, refs)
    assert rows[0]["measured"] == pytest.approx(300.0)
    assert rows[0]["status"] in {"pass", "warn", "fail"}  # graded against the real reference


def test_probe_busbw_factors_match_nccl_tests():
    from tools.benchmarker.prechecks.collective_probe import busbw_factor

    assert busbw_factor("all_reduce", 4) == pytest.approx(2 * 3 / 4)  # 2(n-1)/n
    assert busbw_factor("all_gather", 4) == pytest.approx(3 / 4)      # (n-1)/n
    assert busbw_factor("alltoall", 4) == pytest.approx(3 / 4)
    assert busbw_factor("all_reduce", 1) == 0.0  # single rank: no inter-GPU transfer
