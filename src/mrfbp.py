import numpy as np
from scipy.ndimage import convolve

from .filters import apply_filter, filter_length
from .bases import exponential_basis

# sobel kernels (paper eq. 19), used by mr-fbp_gm
GX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


def build_Ap(geom, p, Q, with_gradients=False):
    # A_p = W W^T C_p Q, one column at a time. column i = forward project the fbp
    # reconstruction obtained with basis filter q_i. one bp + one fp each.
    cols, gx, gy = [], [], []
    for i in range(Q.shape[1]):
        rec = geom.bp(apply_filter(p, Q[:, i]))     # W^T C_p q_i  (fbp with filter q_i)
        cols.append(geom.fp(rec).ravel())           # W (...)
        if with_gradients:
            # the sobel gradients reuse this reconstruction, so no extra projections
            gx.append(convolve(rec, GX).ravel())
            gy.append(convolve(rec, GY).ravel())
    Ap = np.column_stack(cols)
    if with_gradients:
        return Ap, np.column_stack(gx), np.column_stack(gy)
    return Ap


def mrfbp(geom, p, n_l=2, Q=None, lam_ridge=0.0):
    # returns (reconstruction, computed filter h*)
    if Q is None:
        Q = exponential_basis(geom.n_det, n_l=n_l)
    Ap = build_Ap(geom, p, Q)                          # 1) build the system
    A, b = Ap, p.ravel()
    if lam_ridge > 0:                                  # optional tikhonov damping
        A = np.vstack([Ap, lam_ridge * np.eye(Q.shape[1])])
        b = np.concatenate([b, np.zeros(Q.shape[1])])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)          # 2) solve for the coefficients
    h = Q @ c                                          # coefficients -> filter
    return geom.bp(apply_filter(p, h)), h              # 3) plain fbp with h*


def mrfbp_gm(geom, p, lam, n_l=2, Q=None):
    # mr-fbp with gradient minimisation (paper sec. v): stack lam*Dx and lam*Dy
    # blocks onto the system so the reconstruction gradient is penalised too.
    if Q is None:
        Q = exponential_basis(geom.n_det, n_l=n_l)
    Ap, Gx, Gy = build_Ap(geom, p, Q, with_gradients=True)
    A = np.vstack([Ap, lam * Gx, lam * Gy])
    b = np.concatenate([p.ravel(), np.zeros(Gx.shape[0] + Gy.shape[0])])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    h = Q @ c
    return geom.bp(apply_filter(p, h)), h
