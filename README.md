# MR-FBP

Own implementation of Pelt & Batenburg, *"Improving filtered backprojection
reconstruction by data-dependent filtering"* (IEEE TIP, 2014), plus a few
experiments that go beyond the original paper.

FBP is linear in its filter, so instead of picking Ram-Lak by hand we solve for the
filter that minimises the projection residual (`A_p h = p`, `A_p = W W^T C_p`).
Exponential binning keeps the system small, so the solve is direct and fast.

## Layout

    src/            geometry (astra wrapper), filters, fbp, sirt, mrfbp, bases,
                    phantoms, noise, metrics, pipeline
    tests/          test_core.py -- verifies the core, pure numpy, no astra
    scripts/        smoke_test.py, make_phantom_figure.py
    experiments/    exp1..exp9, one script per experiment
    run_all.sh      reproduces every figure

## Install and run

Core check (works anywhere, no astra needed):

    pip install numpy scipy scikit-image matplotlib
    python tests/test_core.py

Full pipeline (needs astra; linux or windows, gpu recommended):

    pip install -r requirements.txt
    python scripts/smoke_test.py
    ./run_all.sh

Every experiment takes `--help` and prints its parameters at startup. Defaults use
`N = 256`; `run_paper_scale.sh` reruns the reproduction at the paper's `N = 1024`.

> astra has no macos build. on a mac run the astra parts on google colab
> (`colab_start.ipynb`) or a linux machine. `tests/test_core.py` runs natively.

## Experiments

    exp1  quality vs number of projections (3 phantoms, 6 methods)
    exp2  robustness to poisson noise
    exp3  reconstruction time
    exp4  exponential binning / n_l sweep
    exp5  the computed filters, in fourier space
    exp6  mr-fbp_gm gradient prior + lambda sweep
    exp7  own: does the filter basis matter?
    exp8  own: how data-dependent is the filter?
    exp9  own: regime map -- where mr-fbp wins and loses

## Reproducibility

Everything is seeded (phantom generation and noise). Our phantoms are random, so a
single seed is a single sample; `exp1`, `exp7`, `exp8`, `exp9` take `--seeds` and
average over draws. The results in the paper are fully reproducable with `run_all.sh`
(the seeded runs make it take a while). Phantoms are generated at a higher resolution
and the sinogram is rebinned, so we never reconstruct with the operator that made the
data.
