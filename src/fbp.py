"""Filtered backprojection:  FBP_h(p) = W^T C_h p."""
import numpy as np
from .filters import apply_filter


def fbp(geom, p, h):
    """Reconstruct by filtering the sinogram with h, then backprojecting.

    The pi/N_theta factor is the standard parallel-beam FBP normalisation.
    (MR-FBP does not need it -- the least-squares solve absorbs any constant --
    but the fixed-filter baselines do.)
    """
    return (np.pi / geom.n_angles) * geom.bp(apply_filter(p, h))
