#!/usr/bin/env bash
set -euo pipefail

FILTER='uv run nbstripout --keep-output --keep-id --keep-metadata-keys="cell.metadata.collapsed cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.scrolled"'

git config --local filter.nbstripout.clean "$FILTER"
git config --local filter.nbstripout.smudge cat
git config --local filter.nbstripout.required true
git config --local diff.ipynb.textconv "$FILTER -t"

echo "Git notebook filtering configured."
