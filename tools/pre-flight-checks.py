#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Two FirecREST MCP servers, one per platform (see firecrest-mcp/README.md):
#   firecrest-mlp -> clariden, bristen   ·   firecrest-hpc -> beverin
FIRECREST_SERVERS = ("firecrest-mlp", "firecrest-hpc")
_FIRECREST_TOOLS = (
    "get_userinfo", "stat_path", "get_job_status", "tail_file",
    "download_file", "get_systems", "view_file", "list_files",
)
REQUIRED_CLAUDE_PERMISSIONS = {
    f"mcp__{server}__{tool}"
    for server in FIRECREST_SERVERS
    for tool in _FIRECREST_TOOLS
} | {
    "WebFetch(domain:docs.cscs.ch)",
    "Bash(kubectl get *)",
    "Bash(kubectl logs *)",
    "Bash(kubectl describe *)",
}


def run(cmd, timeout=30):
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)


def die(msg):
    sys.exit(f"ERROR: {msg}")


def check_binary(name):
    if not shutil.which(name):
        die(f"Missing binary: {name}")


def check_claude_permissions(path):
    path = Path(path).expanduser()

    if not path.exists():
        die(f"Claude settings file not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        die(f"Invalid JSON in {path}: {e}")

    allowed = set(data.get("permissions", {}).get("allow", []))
    missing = sorted(REQUIRED_CLAUDE_PERMISSIONS - allowed)

    if missing:
        die(
            f"Missing Claude permissions in {path}:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    print(f"✓ Claude permissions OK: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-host", default="clariden", help="a host on the ML Platform (clariden/bristen)")
    p.add_argument("--hpc-host", default="beverin", help="a host on the HPC Platform (beverin)")
    p.add_argument("--firecrest-user", default="stefschu")
    p.add_argument("--kube-context", default="breithorn")
    p.add_argument("--kube-namespace", default="ml")
    p.add_argument("--storage-conf", default="~/.config/containers/storage.conf")
    p.add_argument("--claude-settings", default=".claude/settings.local.json")
    args = p.parse_args()

    check_binary("claude")
    check_binary("kubectl")
    check_claude_permissions(args.claude_settings)

    print("\n==> kubectl check")
    try:
        ctx = run(["kubectl", "config", "current-context"], timeout=5).stdout.strip()
    except subprocess.TimeoutExpired:
        die("kubectl config current-context timed out")

    if ctx != args.kube_context:
        die(f"Wrong context: {ctx}")

    for cmd in [
        [
            "kubectl",
            "--context",
            args.kube_context,
            "-n",
            args.kube_namespace,
            "auth",
            "can-i",
            "get",
            "pods",
        ],
        [
            "kubectl",
            "--context",
            args.kube_context,
            "-n",
            args.kube_namespace,
            "get",
            "pods",
        ],
    ]:
        r = run(cmd)
        if r.returncode != 0:
            die(r.stderr.strip() or " ".join(cmd))

    print("\n==> FirecREST MCP check (both platform servers)")
    for server, host in zip(FIRECREST_SERVERS, (args.mlp_host, args.hpc_host)):
        prompt = (
            f"Using the {server} MCP server, connect to {host}, verify the remote user "
            f"is {args.firecrest_user}, and check that {args.storage_conf} exists there. "
            "Return only SUCCESS or FAILURE."
        )
        r = run(["claude", "-p", prompt])
        print(f"[{server} -> {host}] {r.stdout.strip()}")
        if r.returncode != 0 or "SUCCESS" not in r.stdout:
            die(f"FirecREST MCP check failed for {server} ({host})")

    print("\nSUCCESS")


if __name__ == "__main__":
    main()
