#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_python="${ISAAC_PYTHON:-/home/yuchen/miniforge3/envs/isaaclab/bin/python}"
if [[ ! -x "$isaac_python" ]]; then
    echo "Isaac Python not found: $isaac_python. Set ISAAC_PYTHON to your Isaac Sim 6 Python." >&2
    exit 1
fi
# The installed Sim 6 asset importer needs libxml2.so.2, supplied by the existing
# tacex50 environment. Preload only this library, not the other environment's PATH.
xml_library="${ISAAC_XML_LIBRARY:-/home/yuchen/miniforge3/envs/tacex50/lib/libxml2.so.2}"
if [[ -f "$xml_library" ]]; then
    export LD_PRELOAD="$xml_library${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$isaac_python" "$script_dir/isaac_franka_teleop.py" "$@"
