#!/usr/bin/env bash
# Optional, repository-local notebook filters. Run uv sync --locked first.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git rev-parse --is-inside-work-tree >/dev/null
if [[ -n "$(git rev-parse --show-prefix)" ]]; then
    echo 'Run this script from a checkout of this repository.' >&2
    exit 1
fi
uv run --no-sync nbstripout --version >/dev/null

# --no-sync makes Git independent of network access and lockfile updates.
FILTER='uv run --no-sync nbstripout --keep-id --extra-keys="metadata.language_info.version metadata.signature metadata.widgets cell.metadata.trusted"'
git config --local filter.nbstripout.clean "$FILTER"
git config --local filter.nbstripout.smudge cat
git config --local filter.nbstripout.required true
git config --local diff.ipynb.textconv "$FILTER -t"
printf '%s\n' 'Git notebook filtering configured. Outputs stay local; Git stores source and lesson metadata.'
