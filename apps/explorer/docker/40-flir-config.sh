#!/bin/sh
set -eu
case "${FLIR_DATA_MODE:-demo}" in demo|real) ;; *) echo 'Invalid FLIR_DATA_MODE' >&2; exit 1;; esac
snapshot_url=${FLIR_LEAKAGE_SNAPSHOT_URL:-/runtime/leakage-snapshot.json}
# Allowlist before JS interpolation. Never expose arbitrary env values or secrets.
case "$snapshot_url" in /*.json) ;; *) echo 'Invalid snapshot URL' >&2; exit 1;; esac
case "$snapshot_url" in *[!a-zA-Z0-9_./-]*|//*|*/../*) echo 'Invalid snapshot URL' >&2; exit 1;; esac
printf 'window.__FLIR_CONFIG__={"dataMode":"%s","snapshotUrl":"%s"};\n' "${FLIR_DATA_MODE:-demo}" "$snapshot_url" > /usr/share/nginx/html/runtime-config.js
