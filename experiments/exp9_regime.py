# exp9: regime map over (N_theta, I0) -- where mr-fbp wins and where it loses
import argparse

from _common import RESULTS, banner, mean_std, np, plt, save

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
P.add_argument("--seeds", type=int, nargs="+", default=[0],
               help="independent draws of phantom AND noise; results are averaged")
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp9: regime map -- where does MR-FBP win?", cfg)

methods = ["fbp-ram-lak", "fbp-hann", "mrfbp"] + ([] if cfg.no_sirt else ["sirt-200"])

# runs[method] = list over seeds of the (N_theta x I0) MAE grid
runs = {m: [] for m in methods}
loss_frac = []          # fraction of the grid where FBP-Hann beats MR-FBP, per seed

for seed in cfg.seeds:
    G = {m: np.full((len(cfg.angles), len(cfg.i0)), np.nan) for m in methods}
    for i, na in enumerate(cfg.angles):
        geom, p_clean, gt = simulate(cfg.phantom, cfg.n, cfg.n, na, seed=seed,
                                     oversample=cfg.oversample, use_gpu=not cfg.cpu)
        for j, i0 in enumerate(cfg.i0):
            p = add_poisson_noise(p_clean, i0, seed=seed)
            for m in methods:
                G[m][i, j] = mae(reconstruct(m, geom, p), gt)
        print(f"  seed={seed} N_theta={na:4d}  " + "  ".join(
            f"{LABELS[m]}:{G[m][i, 0]:.3f}..{G[m][i, -1]:.3f}" for m in methods))
    for m in methods:
        runs[m].append(G[m])
    loss_frac.append(float((G["mrfbp"] > G["fbp-hann"]).mean()))
    print(f"  seed={seed}: MR-FBP beaten by FBP-Hann on {loss_frac[-1]*100:.0f}% of the grid\n")

# average the error grids over seeds
E = {m: mean_std(runs[m])[0] for m in methods}
E_std = {m: mean_std(runs[m])[1] for m in methods}

np.savez(RESULTS / "exp9_regime.npz", angles=cfg.angles, i0=cfg.i0, seeds=cfg.seeds,
         methods=methods, loss_frac=loss_frac,
         **{m: E[m] for m in methods}, **{f"{m}|std": E_std[m] for m in methods})

X, Y = np.meshgrid(cfg.i0, cfg.angles)          # x = photon count, y = projections

# (a) regime map: which method has the lowest mae at each grid point
stack = np.stack([E[m] for m in methods])
winner = np.argmin(stack, axis=0)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(methods)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.pcolormesh(X, Y, winner, cmap=plt.matplotlib.colors.ListedColormap(colors),
              vmin=-0.5, vmax=len(methods) - 0.5, shading="nearest")

# crossover contour: mr-fbp beats fbp-hann where this ratio < 1
ratio = E["mrfbp"] / E["fbp-hann"]
cs = a1.contour(X, Y, ratio, levels=[1.0], colors="k", linewidths=2.2)
a1.clabel(cs, fmt={1.0: "MR-FBP = FBP-Hann"}, fontsize=8)

a1.set_xscale("log"); a1.set_xlabel(r"photon count $I_0$   (lower = noisier)")
a1.set_ylabel(r"number of projections $N_\theta$")
a1.set_title("(a) which method has the lowest MAE")
a1.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=c, label=LABELS[m])
                   for c, m in zip(colors, methods)], fontsize=8, loc="upper left")

# (b) mr-fbp relative to fbp-hann: >1 means the static filter wins
lim = np.nanmax(np.abs(np.log2(ratio)))
im = a2.pcolormesh(X, Y, np.log2(ratio), cmap="RdBu_r", vmin=-lim, vmax=lim,
                   shading="nearest")
a2.contour(X, Y, ratio, levels=[1.0], colors="k", linewidths=2.2)
a2.set_xscale("log"); a2.set_xlabel(r"photon count $I_0$   (lower = noisier)")
a2.set_ylabel(r"number of projections $N_\theta$")
a2.set_title("(b) $\\log_2$( MAE$_{\\rm MR\\text{-}FBP}$ / MAE$_{\\rm FBP\\text{-}HN}$ )")
fig.colorbar(im, ax=a2, label="red = MR-FBP worse,  blue = MR-FBP better")

fig.suptitle(rf"Regime map, {cfg.phantom} phantom "
             rf"($N$ = $N_d$ = {cfg.n}, mean of {len(cfg.seeds)} seed(s))")
save(fig, "exp9_regime")

# (c) slices: error vs N_theta at fixed noise
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

# summary
print("\nSUMMARY")
loses = ratio > 1.0
lf = np.array(loss_frac) * 100
print(f"  MR-FBP is beaten by FBP-Hann on {lf.mean():.0f}% +/- {lf.std():.0f}% of the "
      f"(N_theta, I0) grid  (per-seed: {', '.join(f'{v:.0f}%' for v in lf)})")
if loses.any():
    i, j = np.unravel_index(np.nanargmax(ratio), ratio.shape)
    print(f"  Worst case: N_theta={cfg.angles[i]}, I0={cfg.i0[j]:.0f} -> "
          f"MR-FBP {E['mrfbp'][i, j]:.4f} vs FBP-Hann {E['fbp-hann'][i, j]:.4f} "
          f"({ratio[i, j]:.2f}x worse)")
for j, i0 in enumerate(cfg.i0):
    col = E["mrfbp"][:, j]
    trend = "RISES with more projections" if col[-1] > col[0] else "falls with more projections"
    print(f"  I0={i0:>9.0f}: MR-FBP MAE {col[0]:.4f} -> {col[-1]:.4f}   ({trend})")
