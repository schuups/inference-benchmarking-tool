#!/usr/bin/env bash
# Wire the Alps runtime env into BASH_ENV so it applies in NON-login shells.
#
# /etc/profile.d/99-alps-env.sh sources alps-runtime.env for LOGIN shells only.
# The engine, however, runs via `bash -c` — on SLURM (`srun ... bash -c
# "run_system_prechecks && exec <engine>"`, §9.0) and in Kubernetes pods
# (`command: ["bash","-c", ...]`, §7.2) — which source BASH_ENV, not profile.d.
# Without this, NCCL_NET / FI_PROVIDER / FI_CXI_* / OMPI_MCA_* / PMIX_MCA_psec /
# NVSHMEM_* are unset at runtime and the Slingshot path is not used. Baking it
# into the image (rather than the launch command) is essential for the K8s
# target, where there is nowhere to inject such adjustments.
#
# Run at BUILD time AFTER install-alps-runtime-warnings.sh, so this hook is
# prepended ABOVE the warning hook — the env (FI_PROVIDER, …) is set before the
# rendezvous-get warning checks it.
set -euo pipefail

marker="# ALPS_RUNTIME_ENV_HOOK"
hook="$(cat <<'EOF'
# ALPS_RUNTIME_ENV_HOOK
if [ -f /opt/alps/env/alps-versions.env ]; then . /opt/alps/env/alps-versions.env; fi
if [ -f /opt/alps/env/alps-runtime.env ]; then . /opt/alps/env/alps-runtime.env; fi
EOF
)"

be="$(bash -c 'printf "%s" "${BASH_ENV-}"')"
if [ -z "$be" ] || [ ! -f "$be" ]; then
    be=/etc/bash.bashrc
    touch "$be"
fi

if grep -Fq "$marker" "$be"; then
    echo "INFO(alps-runtime-env): hook already present in $be"
else
    tmp="$(mktemp)"
    printf '%s\n' "$hook" > "$tmp"
    cat "$be" >> "$tmp"
    cat "$tmp" > "$be"
    rm -f "$tmp"
    echo "INFO(alps-runtime-env): prepended runtime-env hook to $be"
fi
