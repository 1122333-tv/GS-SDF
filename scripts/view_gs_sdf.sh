#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
local_deps="$repo/build/gaussian-viewer/python"
if [[ -d "$local_deps" ]]; then
    export PYTHONPATH="$local_deps${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "${GS_SDF_VIEWER_PYTHON:-python3}" "$repo/scripts/gaussian_viewer/viewer.py" "$@"
