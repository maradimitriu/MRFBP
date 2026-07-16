# makes results/phantom_preview.png -- two draws of each phantom family
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "results").mkdir(exist_ok=True)

from src.phantoms import PHANTOMS, make_phantom   # noqa: E402

N, SEEDS = 256, [0, 1]
fig, ax = plt.subplots(len(SEEDS), len(PHANTOMS), figsize=(9, 6))
for j, name in enumerate(PHANTOMS):
    for i, seed in enumerate(SEEDS):
        _, gt = make_phantom(name, N, seed=seed, oversample=2)
        ax[i, j].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[i, j].axis("off")
        if i == 0:
            ax[i, j].set_title(name, fontsize=12)
out = ROOT / "results" / "phantom_preview.png"
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
