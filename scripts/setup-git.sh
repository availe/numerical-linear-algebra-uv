#!/usr/bin/env bash
# Optional, repository-local notebook filters. Run uv sync --locked first.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$(git rev-parse --show-toplevel)" != "$ROOT" ]]; then
    echo 'Run this script from a checkout of this repository.' >&2
    exit 1
fi
uv run --no-sync nbstripout --version >/dev/null

# --no-sync makes Git independent of network access and lockfile updates.
FILTER='uv run --no-sync nbstripout --keep-output --keep-id --keep-metadata-keys="cell.metadata.collapsed cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.scrolled"'
git config --local filter.nbstripout.clean "$FILTER"
git config --local filter.nbstripout.smudge cat
git config --local filter.nbstripout.required true
git config --local diff.ipynb.textconv "$FILTER -t"
printf '%s\n' 'Git notebook filtering configured for this checkout only.'
