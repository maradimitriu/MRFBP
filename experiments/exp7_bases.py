# exp7: does the filter basis matter (exponential / equidistant / gaussian / dct)
import argparse
import time

from _common import RESULTS, band, banner, mean_std, np, plt, save

from src.bases import dct_basis, equidistant_basis, exponential_basis, gaussian_basis
from src.metrics import mae, residual, ssim
from src.mrfbp import mrfbp
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--angles", type=int, default=64)
P.add_argument("--phantom", default="ellipses")
P.add_argument("--n-b", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 24, 32],
               help="numbers of filter unknowns to compare across bases")
P.add_argument("--seeds", type=int, nargs="+", default=[0])
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp7: alternative filter bases", cfg)

geom, p, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=cfg.seeds[0],
                       oversample=cfg.oversample, use_gpu=not cfg.cpu)
nd = geom.n_det


def basis(name, nb):
    if name == "equidistant":
        return equidistant_basis(nd, nb)
    if name == "dct":
        return dct_basis(nd, nb)
    # exponential/gaussian column count is set by n_l, so search for the n_l
    # that gives nb columns
    for n_l in range(1, nd):
        Q = exponential_basis(nd, n_l) if name == "exponential" else gaussian_basis(nd, n_l)
        if Q.shape[1] == nb:
            return Q
        if Q.shape[1] > nb:
            return None
    return None


names = ["exponential", "equidistant", "gaussian", "dct"]
# n_b values each basis can actually realise (exponential/gaussian are
# quantised by n_l, so they cannot hit every requested N_b).
nbs = {n: [Q.shape[1] for nb in cfg.n_b if (Q := basis(n, nb)) is not None] for n in names}
runs = {n: {"mae": [], "ssim": [], "time": []} for n in names}

for seed in cfg.seeds:
    geom, p, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=seed,
                           oversample=cfg.oversample, use_gpu=not cfg.cpu)
    for name in names:
        curve = {"mae": [], "ssim": [], "time": []}
        for nb in cfg.n_b:
            Q = basis(name, nb)
            if Q is None:
                continue
            t0 = time.perf_counter()
            x, _ = mrfbp(geom, p, Q=Q)
            dt = time.perf_counter() - t0
            curve["mae"].append(mae(x, gt)); curve["ssim"].append(ssim(x, gt))
            curve["time"].append(dt)
            print(f"  seed={seed} {name:12s} N_b={Q.shape[1]:3d}  "
                  f"MAE={curve['mae'][-1]:.4f}  SSIM={curve['ssim'][-1]:.4f}  t={dt:.3f}s")
        for k in runs[name]:
            runs[name][k].append(np.array(curve[k]))

stats = {n: {k: mean_std(v) for k, v in runs[n].items()} for n in names}
np.savez(RESULTS / "exp7_bases.npz", seeds=cfg.seeds,
         **{f"{n}|nb": np.array(nbs[n]) for n in names},
         **{f"{n}|{k}|{s}": val for n in names for k in stats[n]
            for s, val in zip(("mean", "std"), stats[n][k])})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for name in names:
    band(a1, nbs[name], *stats[name]["mae"], name)
    band(a2, nbs[name], *stats[name]["ssim"], name)
a1.set_xlabel(r"number of filter unknowns $N_b$"); a1.set_ylabel("mean absolute error")
a2.set_xlabel(r"number of filter unknowns $N_b$"); a2.set_ylabel("SSIM index")
for a in (a1, a2):
    a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(rf"Filter basis comparison at matched cost ({cfg.phantom}, "
             rf"$N_\theta$={cfg.angles}, $N_d$={nd}, mean of {len(cfg.seeds)} seed(s); "
             rf"shading = $\pm$1 std)")
save(fig, "exp7_bases")
