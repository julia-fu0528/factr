#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_PRELOAD="/home/yuchen/miniforge3/envs/tacex50/lib/libxml2.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
exec /home/yuchen/miniforge3/envs/isaaclab/bin/python replay_cube_policy.py "$@"
