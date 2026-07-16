# verifies the mr-fbp core without astra: build W densely (so W^T is the exact
# transpose) and check the three claims the paper rests on.
import sys
from pathlib import Path

import numpy as np
from skimage.transform import radon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.filters import apply_filter, make_filter, filter_length   # noqa: E402
from src.bases import exponential_basis                            # noqa: E402
from src.mrfbp import build_Ap                                     # noqa: E402

N, N_ANGLES = 32, 16          # small enough to build W densely
RNG = np.random.default_rng(0)


class DenseGeometry:
    # same fp/bp interface as src.geometry, but W is an explicit matrix so bp is
    # the exact transpose of fp
    def __init__(self, n_pixels, n_angles):
        self.n_pixels = n_pixels
        self.n_angles = n_angles
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        cols = []
        for j in range(n_pixels * n_pixels):
            e = np.zeros((n_pixels, n_pixels))
            e.flat[j] = 1.0
            cols.append(radon(e, theta=theta, circle=True).T.ravel())
        self.W = np.column_stack(cols)
        self.n_det = n_pixels
        self.shape = (n_angles, self.n_det)

    def fp(self, x):
        return (self.W @ np.asarray(x).ravel()).reshape(self.shape)

    def bp(self, p):
        return (self.W.T @ np.asarray(p).ravel()).reshape(self.n_pixels, self.n_pixels)


def main():
    geom = DenseGeometry(N, N_ANGLES)
    n_det, M = geom.n_det, filter_length(geom.n_det)

    phantom = np.zeros((N, N))
    yy, xx = np.mgrid[:N, :N]
    phantom[(yy - N // 2) ** 2 + (xx - N // 2) ** 2 < (N // 4) ** 2] = 1.0
    p = geom.fp(phantom)

    # 1. convolution commutes: C_h p == C_p h
    C_p = np.column_stack([
        apply_filter(p, np.eye(M)[j]).ravel() for j in range(M)
    ])
    h_rand = RNG.standard_normal(M)
    err = np.abs(C_p @ h_rand - apply_filter(p, h_rand).ravel()).max()
    print(f"[1] max |C_p h - C_h p|            = {err:.3e}")
    assert err < 1e-8, "FBP is not linear in the filter -- apply_filter is wrong"

    # 2. A_p == W W^T C_p
    Q = exponential_basis(n_det, n_l=2)
    Ap_dense = geom.W @ (geom.W.T @ (C_p @ Q))          # textbook definition
    Ap_cols = build_Ap(geom, p, Q)                      # our column-by-column code
    err = np.abs(Ap_dense - Ap_cols).max() / np.abs(Ap_dense).max()
    print(f"[2] rel. max |A_p(dense) - A_p()|  = {err:.3e}   ({Q.shape[1]} bins)")
    assert err < 1e-6, "build_Ap does not equal W W^T C_p"

    # 3. h* has a smaller residual than ram-lak fit in the same basis
    coeffs, *_ = np.linalg.lstsq(Ap_cols, p.ravel(), rcond=None)
    res_star = np.linalg.norm(Ap_cols @ coeffs - p.ravel())
    h_rl = make_filter(n_det, "ram-lak")
    c_rl, *_ = np.linalg.lstsq(Q, h_rl, rcond=None)
    res_rl = np.linalg.norm(Ap_cols @ c_rl - p.ravel())
    print(f"[3] residual  MR-FBP = {res_star:.4e}   Ram-Lak = {res_rl:.4e}")
    assert res_star <= res_rl + 1e-9, "least-squares filter is not optimal -- bug"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
