import sys
from pathlib import Path

import astra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "results").mkdir(exist_ok=True)      # a fresh clone has no results/ folder
from src.geometry import Geometry          # noqa: E402
from src.fbp import fbp                    # noqa: E402
from src.filters import make_filter        # noqa: E402
from src.mrfbp import mrfbp                # noqa: E402

N, N_DET, N_ANGLES = 256, 256, 32
USE_GPU = True


def disc_phantom(n):
    # a plain white disc -- offsets and artifacts are easy to spot on it
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    return (r < n / 4).astype(np.float64)


def astra_fbp(geom, p):
    # astra's own fbp, as an independent reference for ours
    pid = astra.data2d.create("-sino", geom.proj_geom, p)
    rid = astra.data2d.create("-vol", geom.vol_geom, 0)
    cfg = astra.astra_dict("FBP_CUDA" if USE_GPU else "FBP")
    cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = pid, rid
    if not USE_GPU:
        cfg["ProjectorId"] = geom.proj_id
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid)
    out = astra.data2d.get(rid)
    astra.algorithm.delete(aid); astra.data2d.delete([pid, rid])
    return out


def main():
    geom = Geometry(N, N_DET, N_ANGLES, use_gpu=USE_GPU)
    x = disc_phantom(N)

    # fp/bp round-trip: a blurry disc means the geometry is sane
    p = geom.fp(x)
    bp = geom.bp(p)

    # our fbp vs astra's fbp: should agree up to a scale factor
    ours = fbp(geom, p, make_filter(N_DET, "ram-lak"))
    ref = astra_fbp(geom, p)
    scale = (ours * ref).sum() / (ours * ours).sum()
    rel = np.abs(scale * ours - ref).max() / np.abs(ref).max()
    print(f"our FBP vs ASTRA FBP: best-fit scale = {scale:.4f}, rel. max diff = {rel:.3e}")
    ok = abs(scale - 1.0) < 0.1
    print(f"  scale within 10% of 1.0? {'YES' if ok else 'NO -- normalisation is off'}")

    # mr-fbp runs end to end
    rec, h = mrfbp(geom, p)
    print(f"MR-FBP done. filter length {h.size}, reconstruction range "
          f"[{rec.min():.3f}, {rec.max():.3f}]")

    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
    for a, im, t in zip(ax, [x, p, bp, ours, rec],
                        ["phantom", "sinogram", "backprojection",
                         "FBP (Ram-Lak)", "MR-FBP"]):
        # only the sinogram is non-square, the images keep their aspect ratio
        a.imshow(im, cmap="gray", aspect="auto" if t == "sinogram" else "equal")
        a.set_title(t); a.axis("off")
    out = Path(__file__).resolve().parents[1] / "results" / "smoke_test.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
