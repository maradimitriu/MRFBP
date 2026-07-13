"""Shared experiment setup: phantom -> high-res projection -> rebinned sinogram.

Every experiment script calls `simulate(...)`, so the data-generation path
(and the inverse-crime avoidance) is defined in exactly one place.
"""
from .geometry import Geometry
from .phantoms import make_phantom, rebin_sinogram


def simulate(name, n, n_det, n_angles, seed=0, oversample=4, use_gpu=True):
    """Returns (geom, sinogram p, ground truth gt).

    The phantom is generated and projected at `oversample` times the
    reconstruction resolution; the sinogram is then rebinned down to `n_det`
    detectors. Reconstruction happens on an n x n grid. This avoids the
    "inverse crime" of reconstructing with the very operator used to simulate.
    """
    hi, gt = make_phantom(name, n, seed=seed, oversample=oversample)
    geom_hi = Geometry(n * oversample, n_det * oversample, n_angles, use_gpu=use_gpu)
    p = rebin_sinogram(geom_hi.fp(hi), oversample)
    geom = Geometry(n, n_det, n_angles, use_gpu=use_gpu)
    return geom, p, gt
