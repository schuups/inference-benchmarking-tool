"""torch.distributed collective-bandwidth probe — the K8s §8 foundation check
(SPECIFICATIONS.md §8.2, IMPLEMENTATION_PLAN.md decision 9).

K8s has no `srun`/PMIx, so instead of the MPI nccl-tests SLURM uses, we measure the SAME
all_reduce / all_gather / alltoall bus bandwidth on vLLM's own torch+NCCL stack, launched the
way the engine is on K8s — one rank per GPU via `torchrun` (single-node / 1 pod) or Ray
placement (multi-node, future). The probe body is launcher-agnostic: ranks + rendezvous come
from the env (RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT), which torchrun and
Ray both set.

Output mimics one nccl-tests out-of-place row per collective (`collective_<name>.out`), so
grade.py / the reference table consume it unchanged (grade.parse_nccl_output keys on the size
column and reads busbw). The launch differs from SLURM's PMIx nccl-tests, but busbw is
comparable for the §16.1 platform overlay (decision 9).

Run (single-node, one pod):
    torchrun --standalone --nproc-per-node=<gpus> collective_probe.py --out-dir <dir>

torch is imported lazily inside _run() so the pure helpers below import without it (tested off
GPU); the measurement itself is validated on-cluster in the engine image (torch 2.x present).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# nccl-tests busbw correction factors (github.com/NVIDIA/nccl-tests "Performance"): the wire
# traffic per rank as a fraction of the buffer, so torch numbers line up with the SLURM rows.
_BUSBW_FACTORS = {
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
}
SUPPORTED = tuple(_BUSBW_FACTORS)


def busbw_factor(collective: str, world_size: int) -> float:
    """nccl-tests bus-bandwidth factor for `collective` at `world_size` ranks (0 at n<=1:
    a single rank does no inter-GPU transfer, so busbw is undefined/degenerate)."""
    if world_size <= 1:
        return 0.0
    return _BUSBW_FACTORS[collective](world_size)


def format_nccl_line(size_bytes: int, time_us: float, algbw_gbs: float, busbw_gbs: float) -> str:
    """One nccl-tests-shaped out-of-place row that grade.parse_nccl_output reads:
    `size count type redop root time(us) algbw(GB/s) busbw(GB/s) #wrong` (busbw at field 7)."""
    count = size_bytes // 4  # float32 elements
    return (
        f"{size_bytes:>13d} {count:>13d}   float     sum      -1 "
        f"{time_us:9.1f} {algbw_gbs:8.2f} {busbw_gbs:8.2f}       0"
    )


def _run(args) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = args.bytes
    n = size // 4  # float32 elements in the (total) message

    def timed(op) -> float:
        for _ in range(args.warmup):
            op()
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            op()
        end.record()
        torch.cuda.synchronize()
        secs = (start.elapsed_time(end) / 1000.0) / args.iters  # per-iter wall seconds
        t = torch.tensor([secs], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank sets the collective time
        return float(t.item())

    header = ("#       size(B)         count    type   redop    root   time(us)"
              "    algbw    busbw  #wrong")
    for c in args.collectives:
        if c not in SUPPORTED:
            if rank == 0:
                print(f"[probe] unsupported collective '{c}' — skipping")
            continue
        try:
            if c == "all_reduce":
                buf = torch.ones(n, dtype=torch.float32, device=device)
                op = lambda buf=buf: dist.all_reduce(buf)
            elif c == "all_gather":
                inp = torch.ones(n // world, dtype=torch.float32, device=device)
                out = torch.empty(n, dtype=torch.float32, device=device)
                op = lambda out=out, inp=inp: dist.all_gather_into_tensor(out, inp)
            else:  # alltoall
                inp = torch.ones(n, dtype=torch.float32, device=device)
                out = torch.empty(n, dtype=torch.float32, device=device)
                op = lambda out=out, inp=inp: dist.all_to_all_single(out, inp)
            secs = timed(op)
            algbw = size / secs / 1e9  # GB/s
            busbw = algbw * busbw_factor(c, world)
            if rank == 0:
                line = format_nccl_line(size, secs * 1e6, algbw, busbw)
                (out_dir / f"collective_{c}.out").write_text(header + "\n" + line + "\n")
                print(f"[probe] {c}: algbw={algbw:.2f} busbw={busbw:.2f} GB/s "
                      f"@ {size} B over {world} ranks")
        except Exception as exc:  # a failed collective leaves no file → grade.py marks it fail
            if rank == 0:
                print(f"[probe] {c} FAILED: {exc}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="torch.distributed collective-bandwidth probe (§8, K8s)")
    p.add_argument("--collectives", default="all_reduce all_gather alltoall",
                   help="space-separated subset of: " + " ".join(SUPPORTED))
    p.add_argument("--bytes", type=int, default=128 * 1024**2, help="message size (default 128 MiB)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    args.collectives = args.collectives.split()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
