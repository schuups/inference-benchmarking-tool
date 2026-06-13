#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

# REMOVE_HPCX_DIRS can be space-separated or newline-separated
if [[ -n "${REMOVE_HPCX_DIRS:-}" ]]; then
    while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        echo "Removing HPCX plugin dir: $d"
        rm -rf "$d" || true
    done < <(printf "%s\n" ${REMOVE_HPCX_DIRS})
    ldconfig
fi
