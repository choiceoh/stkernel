#!/usr/bin/env bash
# The engine's GLM-5.3 check on the SERVED kernels, inside the judge image:
# run_mk_probe.sh's container (composed overlay sources on their target
# paths, /cache for the JIT builds) plus the rank files and the megakernel
# MLA lane armed. Same composition as the reference run, kernels swapped --
# the lane adapters are what is judged here.
#
#   bash probes/run_engine_check.sh [--layers 0-4] [--tokens 512] [--chunk 256]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-/home/choiceoh/models}
export PROBE_CACHE=1
export MK_PROBE_DOCKER_ARGS="--mount type=bind,src=$MODELS,dst=$MODELS,readonly ${MK_PROBE_DOCKER_ARGS:-}"
export VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MLA=1      # the megakernel master flag AND the MLA segment: both, or the lane stays inert
exec bash "$REPO/probes/run_mk_probe.sh" engine/profiles/glm53/check.py --lanes served "$@"
