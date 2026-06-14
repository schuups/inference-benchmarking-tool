"""Shared async subprocess helper.

A single `create_subprocess_exec` wrapper so the K8s cluster backend
(`coordinator.backend`) and the Cleaner (`tools.cleaner`) don't each carry their
own copy of the same (returncode, stdout, stderr) plumbing.
"""

from __future__ import annotations

import asyncio


async def run_cmd(*args: str) -> tuple[int, str, str]:
    """Run `args`, returning (returncode, stdout, stderr) with text decoded
    (replace on errors). Never raises on a non-zero exit — callers decide."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def kubectl(*args: str) -> tuple[int, str, str]:
    """`run_cmd("kubectl", *args)`."""
    return await run_cmd("kubectl", *args)
