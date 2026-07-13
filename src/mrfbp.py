"""Minimum Residual Filtered Backprojection (Pelt & Batenburg, IEEE TIP 2014).

KEY IDEA: FBP is LINEAR IN ITS FILTER. Because convolution commutes,

    FBP_h(p) = W^T C_h p = W^T C_p h

so the reconstruction is a linear map of the filter h. We can therefore SOLVE
for the filter that minimises the projection residual instead of picking
Ram-Lak by hand:

    h* = argmin_h || p - W W^T C_p h ||^2      =>      A_p h = p,   A_p = W W^T C_p

Column i of A_p is  W W^T C_p q_i = W( FBP_{q_i}(p) ): filter the sinogram with
basis vector q_i, backproject, forward project. One BP + one FP per column.

A_p has N_theta*N_d rows but only as many columns as there are BASIS FUNCTIONS
for the filter. Exponential binning keeps that at O(log N_d) (~12 for N_d=1024),
so the least-squares solve is direct, iteration-free and parameter-free.

The basis is a free choice -- see `bases.py`. `build_Ap` takes any basis matrix,
which is what makes the alternative-basis experiment (exp7) essentially free.
"""
import numpy as np
from scipy.ndimage import convolve

from .filters import apply_filter, filter_length
from .bases import exponential_basis

# Sobel kernels of paper Eq. 19, used by MR-FBP_GM.
GX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


def build_Ap(geom, p, Q, with_gradients=False):
    """A_p = W W^T C_p Q, built one column at a time (paper Eq. 16).

    Q : (2*n_det-1, n_basis) matrix whose columns are the filter basis vectors.

    Returns (n_angles*n_det, n_basis). If `with_gradients`, also returns the
    horizontal and vertical Sobel gradients of each basis reconstruction --
    these cost NO extra projections, since the reconstruction is already in hand.
    """
    cols, gx, gy = [], [], []
    for i in range(Q.shape[1]):
        rec = geom.bp(apply_filter(p, Q[:, i]))     # W^T C_p q_i = FBP with filter q_i
        cols.append(geom.fp(rec).ravel())           # W ( ... )
        if with_gradients:
            gx.append(convolve(rec, GX).ravel())
            gy.append(convolve(rec, GY).ravel())
    Ap = np.column_stack(cols)
    if with_gradients:
        return Ap, np.column_stack(gx), np.column_stack(gy)
    return Ap


def mrfbp(geom, p, n_l=2, Q=None, lam_ridge=0.0):
    """Algorithm 1. Returns (reconstruction, computed filter h*).

    n_l       : exponential-binning parameter (bins of width 1 for |i| < n_l).
    Q         : optional custom filter basis; defaults to exponential binning.
    lam_ridge : optional Tikhonov damping on the basis coefficients (0 = the
                original, parameter-free method).
    """
    if Q is None:
        Q = exponential_basis(geom.n_det, n_l=n_l)
    Ap = build_Ap(geom, p, Q)                              # 1) A_p = W W^T C_p Q
    A, b = Ap, p.ravel()
    if lam_ridge > 0:                                      #    (optional damping)
        A = np.vstack([Ap, lam_ridge * np.eye(Q.shape[1])])
        b = np.concatenate([b, np.zeros(Q.shape[1])])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)              # 2) direct least squares
    h = Q @ c                                              #    coefficients -> filter
    return geom.bp(apply_filter(p, h)), h                  # 3) plain FBP with h*


def mrfbp_gm(geom, p, lam, n_l=2, Q=None):
    """MR-FBP with gradient minimisation (paper Sec. V, Eq. 20-21).

    Stacks two extra blocks onto the linear system, penalising the Sobel
    gradients of the reconstruction:

        [   W W^T C_p Q  ]        [ p ]
        [ lam Dx W^T C_p Q ] c =  [ 0 ]
        [ lam Dy W^T C_p Q ]      [ 0 ]

    Exploits prior knowledge that the object has a sparse/small gradient
    (our `blocks` phantom). Costs no extra projections over plain MR-FBP.
    """
    if Q is None:
        Q = exponential_basis(geom.n_det, n_l=n_l)
    Ap, Gx, Gy = build_Ap(geom, p, Q, with_gradients=True)
    A = np.vstack([Ap, lam * Gx, lam * Gy])
    b = np.concatenate([p.ravel(), np.zeros(Gx.shape[0] + Gy.shape[0])])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    h = Q @ c
    return geom.bp(apply_filter(p, h)), h
