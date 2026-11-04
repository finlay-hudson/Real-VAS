#!/usr/bin/env bash
# Real-VAS two-clip demo.
# Produces $OUT_DIR/{frames,gt_masks,visible_masks,debugs,composite.mp4}.
#
# Default example is the air-hockey TTE pair from the Real-VAS dataset:
# a cardboard box standing on the table (fg occluder, examples/fg_box.mp4)
# composited over another shot of the same table where a hand slides a
# puck across (bg occludee, examples/bg_puck.mp4). Both clips come from
# the same static camera, so centroid alignment is disabled with
# --no-align. The shipped PNGs under examples/masks/ are the exact
# first-frame per-object masks — SAM2 propagates them through the clip.
# See examples/README.md for details.
#
# Required:
#   SAM_CKPT   path to a SAM2 checkpoint, e.g. sam2_hiera_large.pt
#
# Optional:
#   CACHE_ROOT off-$HOME directory used for matplotlib / HF / torch / CUDA
#              caches. Useful when the home partition is small or full.
#              Without these redirects, matplotlib can't write its font cache
#              AND CUDA can't write its compiled-kernel cache, which surfaces
#              downstream as a bogus `CUDA driver error: invalid argument`
#              inside GroundingDINO's cumsum call.
#   OUT_DIR    output directory (default: out/demo).
#
# Example:
#   SAM_CKPT=/models/sam2_hiera_large.pt scripts/run_demo.sh

set -euo pipefail

: "${SAM_CKPT:?Set SAM_CKPT to the path of your SAM2 checkpoint}"

OUT_DIR="${OUT_DIR:-out/demo}"

if [[ -n "${CACHE_ROOT:-}" ]]; then
    mkdir -p "$CACHE_ROOT"/{mpl,hf,torch,cuda,torch_ext,triton}
    export MPLCONFIGDIR="$CACHE_ROOT/mpl"
    export HF_HOME="$CACHE_ROOT/hf"
    export TORCH_HOME="$CACHE_ROOT/torch"
    export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
    export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_ext"
    export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
fi

python -m real_vas.demo \
    --fg examples/fg_box.mp4  --fg-mask-png examples/masks/fg_box_frame0.png \
    --bg examples/bg_puck.mp4 --bg-mask-png examples/masks/bg_puck_frame0.png \
    --sam-ckpt "$SAM_CKPT" \
    --longest-edge 1024 \
    --no-align \
    --out "$OUT_DIR"
