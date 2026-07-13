# MR-FBP — Minimum Residual Filtered Backprojection

Own implementation of Pelt & Batenburg, *"Improving filtered backprojection
reconstruction by data-dependent filtering"*, IEEE Trans. Image Processing 23(11), 2014.

## The idea

FBP is **linear in its filter**. Convolution commutes, so

    FBP_h(p) = W^T C_h p = W^T C_p h

i.e. the reconstruction is a linear map of the filter `h`. Rather than picking
Ram-Lak by hand, we *solve* for the filter that minimises the projection residual:

    h* = argmin_h || p - W W^T C_p h ||^2      =>      A_p h = p,   A_p = W W^T C_p

`A_p` has `N_theta * N_d` rows but only one column per filter basis function.
Exponential binning keeps that at `O(log N_d)` (~12 for `N_d = 1024`), so the
least-squares solve is **direct, iteration-free and parameter-free**. The
reconstruction is then a plain FBP using the computed filter — near-FBP speed,
near-algebraic quality.

## Layout

    src/filters.py    static filters (Ram-Lak, Shepp-Logan, Hann) and the filtering step C_h p
    src/bases.py      bases for the filter: exponential bins, equidistant, Gaussian RBF, DCT
    src/geometry.py   ASTRA wrapper -- geom.fp(x) = W x, geom.bp(p) = W^T p
    src/fbp.py        FBP_h(p) = W^T C_h p
    src/mrfbp.py      *** the algorithm: build_Ap, mrfbp, mrfbp_gm ***
    src/sirt.py       SIRT, the algebraic baseline
    src/phantoms.py   seeded random phantom generator (3 families)
    src/noise.py      Poisson photon-count noise
    src/metrics.py    MAE (Eq. 22), SSIM, mean absolute residual (Eq. 23)
    src/pipeline.py   phantom -> high-res projection -> rebinned sinogram (avoids inverse crime)

    tests/test_core.py     verification of the core -- pure NumPy, NO ASTRA needed
    scripts/smoke_test.py  ASTRA sanity check
    experiments/exp*.py    one script per experiment in the paper
    run_all.sh             reproduces every figure

## Install and run

### Verification only — works anywhere, including macOS (no ASTRA)

```bash
pip install numpy scipy scikit-image matplotlib
python tests/test_core.py
```

Builds the projection matrix `W` densely for a 32x32 problem (so `W^T` is the
*exact* transpose) and checks the three claims the paper rests on:

1. FBP is linear in the filter: `C_h p == C_p h`
2. `build_Ap` really computes `W W^T C_p`
3. the least-squares filter beats any fixed filter on projection residual

### Full pipeline — needs ASTRA (Linux or Windows; GPU strongly recommended)

```bash
pip install -r requirements.txt
python scripts/smoke_test.py     # fp/bp round-trip, our FBP vs ASTRA's FBP, first MR-FBP
./run_all.sh                     # every experiment, results/ gets figures + .npz
```

Each experiment takes `--help` and prints all its parameters at startup, so every
number in the paper can be traced to a command line. Defaults are fast (`N = 256`);
pass `--n 1024` for the resolution used in the paper.

> **ASTRA has no macOS build.** PyPI ships only `manylinux_x86_64` wheels and conda
> only `linux-64` / `win-64`. On a Mac, run the ASTRA parts on Google Colab
> (`colab_start.ipynb`) or a Linux machine. `tests/test_core.py` needs no ASTRA and
> runs natively.

## Experiments

| script | what it shows | paper |
|---|---|---|
| `exp1_projections.py` | quality vs number of projections, all methods, 3 phantoms | Figs. 7, 8 |
| `exp2_noise.py` | robustness to Poisson noise, sweeping photon count `I0` | Figs. 10, 11 |
| `exp3_timing.py` | reconstruction time vs `N_theta` and vs `N_d` | Fig. 9 |
| `exp4_binning.py` | exponential binning: cost vs quality, sweeping `n_l` | Sec. VII-D |
| `exp5_filters.py` | the computed filters themselves, in Fourier space | Fig. 16 |
| `exp6_gradmin.py` | MR-FBP_GM: a gradient prior on a sparse-gradient phantom | Figs. 14, 15 |
| `exp7_bases.py` | **own**: does the filter basis matter? exp/equidistant/Gaussian/DCT | — |
| `exp8_transfer.py` | **own**: how data-dependent is the filter? transfer penalty | — |

Experiments 7 and 8 are our own contributions. Exp. 7 takes up the paper's own
stated future work ("other bases ... can be used"); exp. 8 quantifies a claim the
paper asserts but never measures ("there is no single filter that is ideal for
every problem").

## Averaging over seeds

Our phantoms are *random* (the original paper uses three fixed images), so a single
seed is a single sample. `exp1`, `exp7`, `exp8` and `exp9` take `--seeds`, and average
over independent draws of **both** the phantom and the noise. Plots show the mean with
a +/- 1 std band; `exp9` reports its headline statistic as mean +/- std across seeds.

```bash
python experiments/exp9_regime.py --seeds 0 1 2 3 4      # headline, with error bars
python experiments/exp1_projections.py --seeds 0 1 2
```

Default is `--seeds 0` so the quick runs stay quick.

## Paper-scale run

Everything defaults to `N = N_d = 256`. The original paper uses 1024. To check the
reproduction holds at their resolution:

```bash
./run_paper_scale.sh          # ~40-60 min on a Colab T4
```

## Reproducibility

All randomness is seeded: phantom generation and Poisson noise. Phantoms are
generated at `oversample` times the reconstruction resolution and the sinogram is
rebinned down, so we never reconstruct with the exact operator used to simulate.
