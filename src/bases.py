import numpy as np

from .filters import filter_length

# bases for the filter h. the mr-fbp system has one unknown per basis vector, so
# the basis sets both the cost and the expressiveness. every function returns Q of
# shape (2*n_det-1, n_basis) with symmetric, centred columns.


def _widths_exponential(n_det, n_l):
    # half-axis bin widths: 1 for i < n_l, then 2^(i-n_l)
    widths, start, i = [], 0, 0
    while start <= n_det - 1:
        w = 1 if i < n_l else 2 ** (i - n_l)
        widths.append(min(w, n_det - start))
        start += w
        i += 1
    return widths


def _from_half_widths(n_det, widths):
    # piecewise-constant symmetric indicator basis from a list of bin widths
    M, centre = filter_length(n_det), n_det - 1
    Q, start = [], 0
    for w in widths:
        offs = np.arange(start, min(start + w, n_det))
        q = np.zeros(M)
        q[centre + offs] = 1.0
        q[centre - offs] = 1.0
        Q.append(q)
        start += w
    return np.column_stack(Q)


def exponential_basis(n_det, n_l=2):
    # fine near offset 0, coarse in the tails
    return _from_half_widths(n_det, _widths_exponential(n_det, n_l))


def equidistant_basis(n_det, n_basis):
    # bins of equal width -- same number of unknowns, no exponential prior
    edges = np.linspace(0, n_det, n_basis + 1).astype(int)
    return _from_half_widths(n_det, np.diff(edges))


def gaussian_basis(n_det, n_l=2):
    # gaussian bumps at the exponential bin centres -- smooth, seperate from the
    # piecewise-constant staircase but with the same placement
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
    # low-frequency cosine modes: cos(pi k |n| / n_det), band-limited not localised
    M, centre = filter_length(n_det), n_det - 1
    n = np.abs(np.arange(M) - centre)
    cols = []
    for k in range(n_basis):
        cols.append(np.cos(np.pi * k * n / n_det))
    return np.column_stack(cols)


BASES = {
    "exponential": lambda n_det, nb=None: exponential_basis(n_det),
    "equidistant": lambda n_det, nb: equidistant_basis(n_det, nb),
    "gaussian": lambda n_det, nb=None: gaussian_basis(n_det),
    "dct": lambda n_det, nb: dct_basis(n_det, nb),
}
