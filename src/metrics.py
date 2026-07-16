import numpy as np
from skimage.metrics import structural_similarity


def _disc_mask(n):
    # we only score inside the central disc of radius N/2 (the corners carry no data)
    yy, xx = np.mgrid[:n, :n]
    return (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (n / 2) ** 2


def mae(x, gt):
    # mean absolute error over the disc, normalised by the ground-truth range
    m = _disc_mask(gt.shape[0])
    return np.abs(x[m] - gt[m]).mean() / (gt.max() - gt.min())


def ssim(x, gt):
    return structural_similarity(gt, x, data_range=gt.max() - gt.min())


def residual(geom, x, p):
    # mean absolute projection residual, what mr-fbp actually minimises
    return np.abs(geom.fp(x) - p).mean()
