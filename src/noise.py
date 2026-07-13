"""Poisson (photon-count) noise, as in the paper Sec. VII-A."""
import numpy as np


def add_poisson_noise(p, I0, seed=0, mu_max=3.0):
    """I0 = photon count at zero attenuation, i.e. the LARGEST count over all
    detectors. Lower I0 = more noise.

    The sinogram is scaled so its maximum line integral equals `mu_max`, turned
    into virtual photon counts I0*exp(-mu), Poisson-sampled, transformed back,
    and rescaled to the original units.
    """
    rng = np.random.default_rng(seed)
    s = p.max()
    mu = p / s * mu_max
    counts = rng.poisson(I0 * np.exp(-mu))
    counts = np.maximum(counts, 1)              # guard against log(0)
    mu_noisy = -np.log(counts / I0)
    return mu_noisy / mu_max * s
