# exp3: reconstruction time vs number of projections and vs number of detectors
import argparse
import time

from _common import RESULTS, banner, np, plt, save

from src.methods import LABELS, reconstruct
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256, help="grid size for sweep (a)")
P.add_argument("--angles", type=int, nargs="+", default=[16, 32, 64, 128, 256])
P.add_argument("--dets", type=int, nargs="+", default=[64, 128, 256, 512])
P.add_argument("--fixed-angles", type=int, default=64, help="N_theta for sweep (b)")
P.add_argument("--repeats", type=int, default=5)
P.add_argument("--phantom", default="ellipses")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=2)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp3: reconstruction time", cfg)

methods = ["fbp-ram-lak", "sirt-200", "sirt-1000", "mrfbp"]


def timed(m, geom, p):
    reconstruct(m, geom, p)                       # per-config warm-up
    ts = []
    for _ in range(cfg.repeats):
        t0 = time.perf_counter()
        reconstruct(m, geom, p)
        ts.append(time.perf_counter() - t0)
    # min not median: on a shared gpu each sample is the true cost plus some
    # positive contention noise, so the minimum is the cleanest estimate
    return float(np.min(ts))


# global warm-up: the first astra call pays for cuda context + kernel build,
# which would otherwise inflate the first point of the sweep
_g, _p, _ = simulate(cfg.phantom, 128, 128, 32, seed=cfg.seed,
                     oversample=1, use_gpu=not cfg.cpu)
for _m in methods:
    reconstruct(_m, _g, _p)
print("  (global GPU warm-up done)\n")


ta = {m: [] for m in methods}
for na in cfg.angles:
    geom, p, _ = simulate(cfg.phantom, cfg.n, cfg.n, na, seed=cfg.seed,
                          oversample=cfg.oversample, use_gpu=not cfg.cpu)
    for m in methods:
        ta[m].append(timed(m, geom, p))
    print(f"  N_theta={na:4d}  " + "  ".join(f"{LABELS[m]}:{ta[m][-1]*1e3:8.1f}ms" for m in methods))

tb = {m: [] for m in methods}
for nd in cfg.dets:
    geom, p, _ = simulate(cfg.phantom, nd, nd, cfg.fixed_angles, seed=cfg.seed,
                          oversample=cfg.oversample, use_gpu=not cfg.cpu)
    for m in methods:
        tb[m].append(timed(m, geom, p))
    print(f"  N=N_d={nd:5d}  " + "  ".join(f"{LABELS[m]}:{tb[m][-1]*1e3:8.1f}ms" for m in methods))

np.savez(RESULTS / "exp3_timing.npz", angles=cfg.angles, dets=cfg.dets,
         **{f"a|{m}": np.array(v) for m, v in ta.items()},
         **{f"b|{m}": np.array(v) for m, v in tb.items()})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for m in methods:
    a1.plot(cfg.angles, ta[m], "o-", ms=4, label=LABELS[m])
    a2.plot(cfg.dets, tb[m], "o-", ms=4, label=LABELS[m])
a1.set_xlabel(r"number of projection angles $N_\theta$")
a1.set_ylabel("reconstruction time (s)"); a1.set_yscale("log")
a1.set_title(f"(a) N = $N_d$ = {cfg.n}")
a2.set_xlabel(r"number of detectors $N_d$ (= grid size $N$)")
a2.set_ylabel("reconstruction time (s)"); a2.set_yscale("log"); a2.set_xscale("log")
a2.set_title(rf"(b) $N_\theta$ = {cfg.fixed_angles}")
for a in (a1, a2):
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
save(fig, "exp3_timing")
