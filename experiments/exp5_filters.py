# exp5: the mr-fbp filters plotted in fourier space
import argparse

from _common import RESULTS, banner, np, plt, save

from src.filters import make_filter
from src.mrfbp import mrfbp
from src.noise import add_poisson_noise
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--i0", type=float, default=2 ** 8, help="photon count for the noisy case")
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp5: computed filters", cfg)

# (phantom, N_theta, noisy?) settings whose filters we compare
settings = [("ellipses", 64, False), ("ellipses", 128, False),
            ("ellipses", 64, True), ("blocks", 64, False)]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
store = {}
for ph, na, noisy in settings:
    geom, p, _ = simulate(ph, cfg.n, cfg.n, na, seed=cfg.seed,
                          oversample=cfg.oversample, use_gpu=not cfg.cpu)
    if noisy:
        p = add_poisson_noise(p, cfg.i0, seed=cfg.seed)
    _, h = mrfbp(geom, p)
    u = np.fft.rfftfreq(h.size)
    H = np.abs(np.fft.rfft(np.fft.ifftshift(h)))
    lab = f"{ph}, $N_\\theta$={na}" + (", noisy" if noisy else "")
    store[lab] = H
    ax[0].plot(u, H, lw=1.4, label=lab)               # raw gain
    ax[1].plot(u, H / H.max(), lw=1.4, label=lab)     # shape only

# the filter fbp actually applies is (pi/N_theta)*h_RL, so that is the fair
# reference to compare the computed filters against
h_rl = make_filter(cfg.n, "ram-lak")
u_rl = np.fft.rfftfreq(h_rl.size)
H_rl = np.abs(np.fft.rfft(np.fft.ifftshift(h_rl)))
for na, ls in [(64, "--"), (128, ":")]:
    ax[0].plot(u_rl, np.pi / na * H_rl, "k" + ls, lw=1.3,
               label=rf"Ram-Lak $\times\,\pi/N_\theta$, $N_\theta$={na}")
ax[1].plot(u_rl, H_rl / H_rl.max(), "k--", lw=1.3, label="Ram-Lak")

ax[0].set_ylabel(r"$|\hat{h}(u)|$   (raw: the filter actually applied)")
ax[0].set_title("(a) absolute filter gain")
ax[1].set_ylabel(r"$|\hat{h}(u)|\;/\;\max$")
ax[1].set_title("(b) filter shape (each normalised to its own peak)")
for a in ax:
    a.set_xlabel("normalised spatial frequency"); a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle("Filters computed by MR-FBP for different reconstruction problems")
save(fig, "exp5_filters")
np.savez(RESULTS / "exp5_filters.npz", ram_lak=H_rl, freq=u_rl, **store)
