import numpy as np
from .filters import apply_filter


def fbp(geom, p, h):
    # standard FBP: filter the sinogram, backproject. the pi/N_theta factor is the
    # usual parallel-beam scaling (mr-fbp doesn't need it, the fixed filters do).
    return (np.pi / geom.n_angles) * geom.bp(apply_filter(p, h))
