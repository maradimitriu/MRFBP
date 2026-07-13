"""Bases for the filter h.

The MR-FBP linear system has one unknown per basis function, so the basis
controls both the cost and the expressiveness of the method. The paper uses
exponential binning; its conclusion notes that "other bases ... can be used
however, which might enable us to reduce computation time even further, or
improve reconstruction quality" -- which is what exp7 investigates.

Every function returns Q of shape (2*n_det-1, n_basis): columns are filter
vectors, centred so that index n_det-1 is offset 0. All bases are symmetric,
since the optimal filter is.
"""
import numpy as np

from .filters import filter_length


def _widths_exponential(n_det, n_l):
    """Half-axis bin widths: d_i = 1 for i < n_l, else 2^(i - n_l)."""
    widths, start, i = [], 0, 0
    while start <= n_det - 1:
        w = 1 if i < n_l else 2 ** (i - n_l)
        widths.append(min(w, n_det - start))
        start += w
        i += 1
    return widths


def _from_half_widths(n_det, widths):
    """Piecewise-constant symmetric indicator basis from half-axis bin widths."""
    M, centre = filter_length(n_det), n_det - 1
    Q, start = [], 0
    for w in widths:
        offs = np.arange(start, min(start + w, n_det))
        q = np.zeros(M)
        q[centre + offs] = 1.0
        q[centre - offs] = 1.0                 # symmetrise (offset 0 hit twice, still 1)
        Q.append(q)
        start += w
    return np.column_stack(Q)


def exponential_basis(n_det, n_l=2):
    """Paper Sec. IV-B. ~2+log2(n_det) bins: fine near offset 0, coarse far out."""
    return _from_half_widths(n_det, _widths_exponential(n_det, n_l))


def equidistant_basis(n_det, n_basis):
    """Control: bins of equal width. Same number of unknowns, no exponential prior."""
    edges = np.linspace(0, n_det, n_basis + 1).astype(int)
    return _from_half_widths(n_det, np.diff(edges))


def gaussian_basis(n_det, n_l=2):
    """Gaussian RBFs at the exponential bin centres, width = bin width.

    Smooth counterpart of exponential binning -- no piecewise-constant staircase.
    """
    M, centre = filter_length(n_det), n_det - 1
    n = np.arange(M) - centre
    widths = _widths_exponential(n_det, n_l)
    Q, start = [], 0
    for w in widths:
        mu, sig = start + w / 2, max(w, 1) / 2
        Q.append(np.exp(-((np.abs(n) - mu) ** 2) / (2 * sig ** 2)))
        start += w
    return np.column_stack(Q)


def dct_basis(n_det, n_basis):
    """Low-frequency cosine modes: h(n) = cos(pi k |n| / n_det), k = 0..n_basis-1.

    A band-limited (rather than spatially-localised) basis for the filter.
    """
    M, centre = filter_length(n_det), n_det - 1
    n = np.abs(np.arange(M) - centre)
    return np.column_stack([np.cos(np.pi * k * n / n_det) for k in range(n_basis)])


BASES = {
    "exponential": lambda n_det, nb=None: exponential_basis(n_det),
    "equidistant": lambda n_det, nb: equidistant_basis(n_det, nb),
    "gaussian": lambda n_det, nb=None: gaussian_basis(n_det),
    "dct": lambda n_det, nb: dct_basis(n_det, nb),
}
