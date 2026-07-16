import numpy as np


def add_poisson_noise(p, I0, seed=0, mu_max=3.0):
    # I0 is the photon count at zero attenuation (the largest count), so lower I0
    # means more noise. scale to photon counts, sample, transform back.
    rng = np.random.default_rng(seed)
    s = p.max()
    mu = p / s * mu_max
    counts = rng.poisson(I0 * np.exp(-mu))
    counts = np.maximum(counts, 1)      # floor at 1 to avoid log(0)
    mu_noisy = -np.log(counts / I0)
    return mu_noisy / mu_max * s
