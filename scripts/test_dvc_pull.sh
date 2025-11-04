#!/usr/bin/env bash

# Simple helper to verify Azure-backed DVC remote access locally.
# Expects AZURE_STORAGE_CONNECTION_STRING to be set in the environment.

set -euo pipefail

if [[ -z "${AZURE_STORAGE_CONNECTION_STRING:-}" ]]; then
  echo "[ERROR] AZURE_STORAGE_CONNECTION_STRING is not set." >&2
  exit 1
fi

echo "[INFO] Using Azure storage connection string from environment."

echo "[INFO] Configuring DVC remote 'azuremodelstore' (local override)..."
dvc remote modify --local azuremodelstore connection_string "$AZURE_STORAGE_CONNECTION_STRING"

echo "[INFO] Running 'dvc pull' to validate access..."
dvc pull --verbose

echo "[INFO] DVC pull completed successfully. Remote credentials are working."
