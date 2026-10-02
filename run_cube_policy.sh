#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/.training_deps${PYTHONPATH:+:$PYTHONPATH}"
exec /home/yuchen/miniforge3/envs/tacex50/bin/python train_cube_policy.py --headless "$@"
