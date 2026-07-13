"""Experiment 8 (OWN CONTRIBUTION) -- how data-dependent is the filter, really?

MR-FBP's selling point is a filter tailored to the problem at hand. The paper
asserts (Fig. 16) that "there is no single filter that is ideal for every
problem" but never quantifies the cost of using the wrong one.

Here we compute h* on a SOURCE problem, apply it as a fixed filter to a TARGET
problem, and report the resulting error. The diagonal is true MR-FBP; the
off-diagonal is the penalty for transferring a filter. Ram-Lak is included as
the "no adaptation at all" baseline.

    python experiments/exp8_transfer.py
"""
import argparse

from _common import RESULTS, banner, mean_std, np, plt, save

from src.fbp import fbp
from src.filters import make_filter
from src.metrics import mae
from src.mrfbp import mrfbp
from src.noise import add_poisson_noise
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--phantom", default="ellipses")
P.add_argument("--angles", type=int, nargs="+", default=[32, 64, 128])
P.add_argument("--i0", type=float, nargs="+", default=[np.inf, 2 ** 10])
P.add_argument("--seeds", type=int, nargs="+", default=[0])
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp8: filter transferability", cfg)

# One "problem" = (phantom family, N_theta, noise level).
problems = [(ph, na, i0)
            for ph in ("ellipses", "blocks")
            for na in cfg.angles
            for i0 in cfg.i0]
labels = [f"{ph[:4]}/{na}/" + ("clean" if np.isinf(i0) else f"I0={i0:.0f}")
          for ph, na, i0 in problems]


def data(ph, na, i0, seed):
    geom, p, gt = simulate(ph, cfg.n, cfg.n, na, seed=seed,
                           oversample=cfg.oversample, use_gpu=not cfg.cpu)
    if not np.isinf(i0):
        p = add_poisson_noise(p, i0, seed=seed)
    return geom, p, gt


Ms, rls = [], []
for seed in cfg.seeds:
    # Filters learned on each source problem.
    filters = []
    for ph, na, i0 in problems:
        geom, p, _ = data(ph, na, i0, seed)
        filters.append(mrfbp(geom, p)[1])

    # Apply every filter to every target problem. The reconstruction is rescaled
    # to best match the ground truth, so we compare filter SHAPE, not brightness.
    M = np.zeros((len(problems), len(problems)))
    rl = np.zeros(len(problems))
    for j, (ph, na, i0) in enumerate(problems):
        geom, p, gt = data(ph, na, i0, seed)
        for i, h in enumerate(filters):
            x = fbp(geom, p, h)
            s = (x * gt).sum() / max((x * x).sum(), 1e-12)
            M[i, j] = mae(s * x, gt)
        x = fbp(geom, p, make_filter(geom.n_det, "ram-lak"))
        s = (x * gt).sum() / max((x * x).sum(), 1e-12)
        rl[j] = mae(s * x, gt)
        print(f"  seed={seed} target {labels[j]:22s} own={M[j, j]:.4f}  ram-lak={rl[j]:.4f}")
    Ms.append(M); rls.append(rl)

M, M_std = mean_std(Ms)
rl, _ = mean_std(rls)
np.savez(RESULTS / "exp8_transfer.npz", matrix=M, matrix_std=M_std,
         ram_lak=rl, labels=labels, seeds=cfg.seeds)

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(M, cmap="viridis")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("target problem (filter applied to)")
ax.set_ylabel("source problem (filter computed on)")
ax.set_title("Mean absolute error when transferring an MR-FBP filter\n"
             f"(diagonal = true MR-FBP; off-diagonal = transfer penalty; "
             f"mean of {len(cfg.seeds)} seed(s))")
fig.colorbar(im, label="mean absolute error")
save(fig, "exp8_transfer")
