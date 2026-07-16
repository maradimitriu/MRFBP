from .geometry import Geometry
from .phantoms import make_phantom, rebin_sinogram


def simulate(name, n, n_det, n_angles, seed=0, oversample=4, use_gpu=True):
    # generate + project the phantom at oversample x resolution, then rebin the
    # sinogram down. this way we never recontruct with the same operator that made
    # the data (the inverse crime). returns (geom, sinogram, ground truth).
    hi, gt = make_phantom(name, n, seed=seed, oversample=oversample)
    geom_hi = Geometry(n * oversample, n_det * oversample, n_angles, use_gpu=use_gpu)
    p = rebin_sinogram(geom_hi.fp(hi), oversample)
    geom = Geometry(n, n_det, n_angles, use_gpu=use_gpu)
    return geom, p, gt
