"""Quality metrics (paper Sec. VI)."""
import numpy as np
from skimage.metrics import structural_similarity


def _disc_mask(n):
    """Eq. 22 averages only over the central disc of radius N/2."""
    yy, xx = np.mgrid[:n, :n]
    return (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (n / 2) ** 2


def mae(x, gt):
    """Eq. 22: mean absolute error over the central disc, normalised by the
    dynamic range of the ground truth."""
    m = _disc_mask(gt.shape[0])
    return np.abs(x[m] - gt[m]).mean() / (gt.max() - gt.min())


def ssim(x, gt):
    return structural_similarity(gt, x, data_range=gt.max() - gt.min())


def residual(geom, x, p):
    """Eq. 23: mean absolute projection residual, (N_theta N_d)^-1 ||Wx - p||_1."""
    return np.abs(geom.fp(x) - p).mean()
