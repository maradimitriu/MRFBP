#!/usr/bin/env bash
# reproduces every figure in the paper (results/ gets the pngs). the seeded
# experiments use the same seeds as the paper, so this takes a while (~1.5-2h).
set -e
cd "$(dirname "$0")"

python tests/test_core.py
python scripts/smoke_test.py
python scripts/make_phantom_figure.py

python experiments/exp1_projections.py --seeds 0 1 2 3 4
python experiments/exp2_noise.py
python experiments/exp3_timing.py
python experiments/exp4_binning.py
python experiments/exp5_filters.py
python experiments/exp6_gradmin.py
python experiments/exp7_bases.py --seeds 0 1 2 3 4
python experiments/exp8_transfer.py --seeds 0 1 2
python experiments/exp9_regime.py --seeds 0 1 2 3 4
