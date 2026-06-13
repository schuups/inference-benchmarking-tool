#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

rm -rf /opt/amazon/efa || true
grep -R "/opt/amazon/efa" -n /etc/ld.so.conf.d || true
for f in /etc/ld.so.conf.d/*; do
    [[ -f "$f" ]] || continue
    if grep -q "/opt/amazon/efa" "$f"; then rm -f "$f"; fi
done
ldconfig
