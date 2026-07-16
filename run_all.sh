#!/usr/bin/env bash
# Reproduce every figure in the paper. Results land in results/.
# Defaults are the fast settings (N=256); pass --n 1024 for paper scale.
set -e
cd "$(dirname "$0")"

python tests/test_core.py                 # correctness of the core algorithm (no ASTRA)
python scripts/smoke_test.py              # ASTRA sanity check
python scripts/make_phantom_figure.py    # phantom-families figure for the paper

python experiments/exp1_projections.py    # quality vs number of projections
python experiments/exp2_noise.py          # Poisson noise robustness
python experiments/exp3_timing.py         # reconstruction time
python experiments/exp4_binning.py        # exponential binning / n_l sweep
python experiments/exp5_filters.py        # the computed filters, in Fourier space
python experiments/exp6_gradmin.py        # MR-FBP_GM (gradient prior)
python experiments/exp7_bases.py          # OWN: alternative filter bases
python experiments/exp8_transfer.py       # OWN: filter transferability
python experiments/exp9_regime.py         # OWN: regime map -- where MR-FBP wins and loses
