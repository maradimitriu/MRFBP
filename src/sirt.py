import numpy as np


def sirt(geom, p, n_iter=200, nonneg=False):
    # algebraic baseline. x <- x + C W^T R (p - W x), with R = 1/row sums and
    # C = 1/col sums (floored so we never divide by zero).
    R = 1.0 / np.maximum(geom.fp(np.ones((geom.n_pixels, geom.n_pixels))), 1e-6)
    C = 1.0 / np.maximum(geom.bp(np.ones((geom.n_angles, geom.n_det))), 1e-6)
    x = np.zeros((geom.n_pixels, geom.n_pixels))
    for _ in range(n_iter):
        x = x + C * geom.bp(R * (p - geom.fp(x)))
        if nonneg:
            x = np.maximum(x, 0.0)
    return x
