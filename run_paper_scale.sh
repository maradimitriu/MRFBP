#!/usr/bin/env bash
# reruns the reproduction at the paper's resolution (N = N_d = 1024) to check the
# ordering of the methods still holds. sirt-1000 is dropped and angles thinned so
# it stays feasible.
set -e
cd "$(dirname "$0")"

python experiments/exp1_projections.py \
    --n 1024 \
    --angles 32 64 128 256 \
    --phantoms ellipses blocks \
    --methods fbp-ram-lak fbp-shepp-logan fbp-hann sirt-200 mrfbp \
    --seeds 0
