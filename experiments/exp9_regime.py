"""Experiment 9 (OWN CONTRIBUTION) -- where does MR-FBP actually win?

MOTIVATION. The paper evaluates noise robustness at ONE projection count
(N_theta = 64, their Fig. 10) and concludes that MR-FBP matches SIRT and beats
static-filter FBP "for every noise level". Our exp2/exp5/exp8 suggest that
conclusion is regime-specific:

  * exp5: the computed filter drifts TOWARD Ram-Lak as N_theta grows
          (40% -> 57% of Ram-Lak's Nyquist gain from 64 to 128 projections).
  * exp2: under noise, MR-FBP therefore gets WORSE with more projections
          (MAE 0.0415 -> 0.0499 -> 0.0641 at I0=1024, N_theta = 32/64/128)
          while FBP-RL gets BETTER (0.1716 -> 0.1195 -> 0.0835).
  * exp8: a filter learned from FEWER projections beats the native filter on
          noisy data, by up to 2.3x.

Mechanism: MR-FBP's noise robustness is not a property of the method, it is an
artefact of DATA SCARCITY. With few angles it cannot explain high frequencies, so
the least-squares suppresses them -- which happens to be the right thing to do
under noise. With many angles it becomes confident enough to amplify them, which
is exactly wrong.

THIS EXPERIMENT maps the (N_theta, I0) plane and shows where each method wins.
The output is a regime map with the MR-FBP / FBP-Hann crossover contour: the
boundary beyond which the data-dependent filter is a LIABILITY.

    python experiments/exp9_regime.py                 # ~15-25 min on a Colab T4
    python experiments/exp9_regime.py --no-sirt       # ~3x faster
"""
import argparse

from _common import RESULTS, banner, np, plt, save

from src.methods import LABELS, reconstruct
from src.metrics import mae
from src.noise import add_poisson_noise
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--angles", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128, 192])
P.add_argument("--i0", type=float, nargs="+", default=[2.0 ** k for k in (6, 8, 10, 12, 14, 16, 20)])
P.add_argument("--phantom", default="ellipses")
P.add_argument("--no-sirt", action="store_true", help="drop SIRT (it dominates the runtime)")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp9: regime map -- where does MR-FBP win?", cfg)

methods = ["fbp-ram-lak", "fbp-hann", "mrfbp"] + ([] if cfg.no_sirt else ["sirt-200"])

# E[method][i, j] = MAE at (N_theta = angles[i], I0 = i0[j])
E = {m: np.full((len(cfg.angles), len(cfg.i0)), np.nan) for m in methods}

for i, na in enumerate(cfg.angles):
    geom, p_clean, gt = simulate(cfg.phantom, cfg.n, cfg.n, na, seed=cfg.seed,
                                 oversample=cfg.oversample, use_gpu=not cfg.cpu)
    for j, i0 in enumerate(cfg.i0):
        p = add_poisson_noise(p_clean, i0, seed=cfg.seed)
        for m in methods:
            E[m][i, j] = mae(reconstruct(m, geom, p), gt)
    print(f"  N_theta={na:4d}  " + "  ".join(
        f"{LABELS[m]}:{E[m][i, 0]:.3f}..{E[m][i, -1]:.3f}" for m in methods))

np.savez(RESULTS / "exp9_regime.npz", angles=cfg.angles, i0=cfg.i0,
         methods=methods, **{m: E[m] for m in methods})

X, Y = np.meshgrid(cfg.i0, cfg.angles)          # x = photon count, y = projections

# --- (a) the regime map: which method has the lowest MAE at each grid point ----
stack = np.stack([E[m] for m in methods])
winner = np.argmin(stack, axis=0)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(methods)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.pcolormesh(X, Y, winner, cmap=plt.matplotlib.colors.ListedColormap(colors),
              vmin=-0.5, vmax=len(methods) - 0.5, shading="nearest")

# The crossover contour: MR-FBP beats FBP-Hann where this ratio is < 1.
ratio = E["mrfbp"] / E["fbp-hann"]
cs = a1.contour(X, Y, ratio, levels=[1.0], colors="k", linewidths=2.2)
a1.clabel(cs, fmt={1.0: "MR-FBP = FBP-Hann"}, fontsize=8)

a1.set_xscale("log"); a1.set_xlabel(r"photon count $I_0$   (lower = noisier)")
a1.set_ylabel(r"number of projections $N_\theta$")
a1.set_title("(a) which method has the lowest MAE")
a1.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=c, label=LABELS[m])
                   for c, m in zip(colors, methods)], fontsize=8, loc="upper left")

# --- (b) MR-FBP relative to FBP-Hann: >1 means the static filter WINS ---------
lim = np.nanmax(np.abs(np.log2(ratio)))
im = a2.pcolormesh(X, Y, np.log2(ratio), cmap="RdBu_r", vmin=-lim, vmax=lim,
                   shading="nearest")
a2.contour(X, Y, ratio, levels=[1.0], colors="k", linewidths=2.2)
a2.set_xscale("log"); a2.set_xlabel(r"photon count $I_0$   (lower = noisier)")
a2.set_ylabel(r"number of projections $N_\theta$")
a2.set_title("(b) $\\log_2$( MAE$_{\\rm MR\\text{-}FBP}$ / MAE$_{\\rm FBP\\text{-}HN}$ )")
fig.colorbar(im, ax=a2, label="red = MR-FBP worse,  blue = MR-FBP better")

fig.suptitle(rf"Regime map, {cfg.phantom} phantom "
             rf"($N$ = $N_d$ = {cfg.n}, seed {cfg.seed})")
save(fig, "exp9_regime")

# --- (c) the mechanism, as slices: error vs N_theta at fixed noise ------------
fig, ax = plt.subplots(1, len(cfg.i0), figsize=(3.1 * len(cfg.i0), 3.2), sharey=True)
for j, (a, i0) in enumerate(zip(np.atleast_1d(ax), cfg.i0)):
    for m in methods:
        a.plot(cfg.angles, E[m][:, j], "o-", ms=3, label=LABELS[m])
    a.set_xlabel(r"$N_\theta$"); a.set_yscale("log")
    a.set_title(rf"$I_0$={i0:.0f}", fontsize=9); a.grid(alpha=.3)
np.atleast_1d(ax)[0].set_ylabel("mean absolute error")
np.atleast_1d(ax)[0].legend(fontsize=7)
fig.suptitle("Error vs number of projections, at each noise level "
             "(MR-FBP turns UPWARD when the data is noisy)")
save(fig, "exp9_slices")

# --- summary --------------------------------------------------------------
print("\nSUMMARY")
loses = ratio > 1.0
print(f"  MR-FBP is beaten by FBP-Hann in {loses.sum()}/{loses.size} of the "
      f"(N_theta, I0) grid.")
if loses.any():
    i, j = np.unravel_index(np.nanargmax(ratio), ratio.shape)
    print(f"  Worst case: N_theta={cfg.angles[i]}, I0={cfg.i0[j]:.0f} -> "
          f"MR-FBP {E['mrfbp'][i, j]:.4f} vs FBP-Hann {E['fbp-hann'][i, j]:.4f} "
          f"({ratio[i, j]:.2f}x worse)")
for j, i0 in enumerate(cfg.i0):
    col = E["mrfbp"][:, j]
    trend = "RISES with more projections" if col[-1] > col[0] else "falls with more projections"
    print(f"  I0={i0:>9.0f}: MR-FBP MAE {col[0]:.4f} -> {col[-1]:.4f}   ({trend})")
