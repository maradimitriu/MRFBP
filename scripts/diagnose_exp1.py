"""Why does MR-FBP get WORSE with more projections on `ellipses`?

Observation from exp1: MR-FBP's MAE falls to N_theta=32, then RISES, ending ~2x
worse than plain FBP. The image grid shows the reconstruction is OVER-SMOOTHED,
not noisy -- so the computed filter is suppressing high frequencies.

HYPOTHESIS (model mismatch). To avoid the inverse crime we generate the phantom
at `oversample` x the reconstruction resolution and rebin the sinogram. So `p`
contains detail that NO 256x256 image can reproduce: part of the residual is
irreducible, and it is concentrated at high spatial frequencies. MR-FBP
minimises ||p - W FBP_h(p)||, so raising the high-frequency gain of h adds image
detail that cannot match the unrepresentable part of p -- it only ADDS residual.
The least-squares therefore damps high frequencies. More angles sample the
mismatch more thoroughly => stronger damping => blurrier image => higher MAE.

PREDICTION. With oversample=1 (i.e. committing the inverse crime, so W explains p
exactly) the degradation should vanish and MR-FBP should beat FBP at every
N_theta, as in the paper. With oversample=4 it should be present.

If the prediction holds, this is a genuine limitation of MR-FBP worth a paragraph
in the Discussion -- not a bug.

    python scripts/diagnose_exp1.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fbp import fbp                                    # noqa: E402
from src.filters import make_filter                        # noqa: E402
from src.geometry import Geometry                          # noqa: E402
from src.metrics import mae, residual                      # noqa: E402
from src.mrfbp import mrfbp                                # noqa: E402
from src.phantoms import make_phantom, rebin_sinogram      # noqa: E402

N = 256
ANGLES = [32, 64, 128, 256]
OVERSAMPLE = [1, 2, 4]          # 1 = inverse crime (W explains p exactly)
PHANTOM = "ellipses"

print(f"phantom={PHANTOM}  N={N}  (MAE lower = better; residual is what MR-FBP minimises)\n")
print(f"{'oversample':>10} {'N_theta':>8} {'MAE FBP-RL':>11} {'MAE MR-FBP':>11} "
      f"{'resid FBP-RL':>13} {'resid MR-FBP':>13}")

filters = {}
for os_ in OVERSAMPLE:
    for na in ANGLES:
        hi, gt = make_phantom(PHANTOM, N, seed=0, oversample=os_)
        p = rebin_sinogram(Geometry(N * os_, N * os_, na).fp(hi), os_)
        geom = Geometry(N, N, na)

        x_rl = fbp(geom, p, make_filter(N, "ram-lak"))
        x_mr, h = mrfbp(geom, p)
        filters[(os_, na)] = h
        print(f"{os_:>10} {na:>8} {mae(x_rl, gt):>11.4f} {mae(x_mr, gt):>11.4f} "
              f"{residual(geom, x_rl, p):>13.4f} {residual(geom, x_mr, p):>13.4f}")
    print()

# The computed filters, in Fourier space, for each oversample factor.
fig, ax = plt.subplots(1, len(OVERSAMPLE), figsize=(5 * len(OVERSAMPLE), 3.8), sharey=True)
for a, os_ in zip(np.atleast_1d(ax), OVERSAMPLE):
    for na in ANGLES:
        h = filters[(os_, na)]
        u = np.fft.rfftfreq(h.size)
        H = np.abs(np.fft.rfft(np.fft.ifftshift(h)))
        a.plot(u, H / H.max(), lw=1.3, label=rf"$N_\theta$={na}")
    h_rl = make_filter(N, "ram-lak")
    H = np.abs(np.fft.rfft(np.fft.ifftshift(h_rl)))
    a.plot(np.fft.rfftfreq(h_rl.size), H / H.max(), "k--", lw=1.2, label="Ram-Lak")
    a.set_title(f"oversample = {os_}" + ("  (inverse crime)" if os_ == 1 else ""))
    a.set_xlabel("normalised spatial frequency"); a.grid(alpha=.3); a.legend(fontsize=8)
np.atleast_1d(ax)[0].set_ylabel(r"$|\hat h|$ (normalised)")
fig.suptitle("Computed MR-FBP filters vs forward-model accuracy")
fig.savefig(ROOT / "results" / "diagnose_filters.png", dpi=140, bbox_inches="tight")
print("wrote results/diagnose_filters.png\n")

print("HOW TO READ THIS")
print("  If at oversample=1 MR-FBP beats FBP-RL at every N_theta and its MAE falls")
print("  monotonically, but at oversample=4 it rises -> hypothesis CONFIRMED: MR-FBP")
print("  is damping high frequencies to avoid chasing residual it cannot explain.")
print("  The filter plot should show the high-frequency roll-off getting stronger as")
print("  oversample (and N_theta) increase.")
