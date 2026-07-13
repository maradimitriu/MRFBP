#!/usr/bin/env bash
# Reproduce the headline result at the ORIGINAL PAPER'S RESOLUTION (N = N_d = 1024).
#
# The paper reconstructs on a 1024x1024 grid with 1024 detectors, from a 4096x4096
# phantom. Our default (N=256) is a 4x-cheaper stand-in. This script checks that the
# reproduction of Fig. 7 -- MR-FBP below every fixed-filter FBP, above SIRT --
# survives at their resolution, so nobody can attribute our findings to a small grid.
#
# SIRT-1000 is dropped: at 1024^2 it costs ~2000 projections per reconstruction and
# adds nothing over SIRT-200 here. Angle counts are thinned for the same reason.
#
# Expect ~40-60 min on a Colab T4. Needs ~4 GB of GPU memory (the phantom is
# generated and projected at 4096x4096).
set -e
cd "$(dirname "$0")"

python experiments/exp1_projections.py \
    --n 1024 \
    --angles 32 64 128 256 \
    --phantoms ellipses blocks \
    --methods fbp-ram-lak fbp-shepp-logan fbp-hann sirt-200 mrfbp \
    --seeds 0

echo
echo "Compare against the N=256 run: the ORDERING of the methods should be identical."
