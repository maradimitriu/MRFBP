import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.draw import disk, ellipse, polygon


# seeded random phantom generator, three families mirroring the paper's phantoms:
#   ellipses -> smooth continuous grey values
#   blocks   -> piecewise constant, sparse gradient (what mr-fbp_gm needs)
#   mixed    -> thin high-contrast lines on a smooth background
# keep the object inside 85% of the fov radius. if it grazes the detector edge the
# data is effectively truncated, which mr-fbp handles badly (see paper sec. vii-g).
SUPPORT_FRAC = 0.85


def _circle_mask(n, frac=SUPPORT_FRAC):
    yy, xx = np.mgrid[:n, :n]
    return (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= (frac * n / 2) ** 2


def ellipses(n, seed=0, n_ell=12):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    r = SUPPORT_FRAC * n / 2
    img[disk((n / 2, n / 2), r, shape=img.shape)] = 0.25           # outer disc
    img[disk((n / 2, n / 2), 0.92 * r, shape=img.shape)] = 0.15
    for _ in range(n_ell):
        cy, cx = rng.uniform(0.3, 0.7, 2) * n
        ry, rx = rng.uniform(0.03, 0.18, 2) * n
        rr, cc = ellipse(cy, cx, ry, rx, shape=img.shape,
                         rotation=rng.uniform(0, np.pi))
        img[rr, cc] += rng.uniform(-0.15, 0.35)
    img = np.clip(img, 0, 1) * _circle_mask(n)
    return img


def blocks(n, seed=0, n_rect=9, n_holes=14):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    img[disk((n / 2, n / 2), SUPPORT_FRAC * n / 2, shape=img.shape)] = 0.35
    for _ in range(n_rect):
        cy, cx = rng.uniform(0.25, 0.75, 2) * n
        hy, hx = rng.uniform(0.05, 0.20, 2) * n
        r = polygon(*_rect_corners(cy, cx, hy, hx, rng.uniform(0, np.pi)),
                    shape=img.shape)
        img[r] = rng.choice([0.65, 1.0])
    for _ in range(n_holes):                       # holes give strong sparse edges
        cy, cx = rng.uniform(0.25, 0.75, 2) * n
        img[disk((cy, cx), rng.uniform(0.02, 0.06) * n, shape=img.shape)] = 0.0
    return img * _circle_mask(n)


def _rect_corners(cy, cx, hy, hx, ang):
    dy = np.array([-hy, -hy, hy, hy])
    dx = np.array([-hx, hx, hx, -hx])
    c, s = np.cos(ang), np.sin(ang)
    return cy + c * dy - s * dx, cx + s * dy + c * dx


def mixed(n, seed=0, n_spokes=7):
    rng = np.random.default_rng(seed)
    bg = gaussian_filter(rng.random((n, n)), sigma=n / 12)
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-12) * 0.45

    bars = np.zeros((n, n))
    for _ in range(n_spokes):
        w = max(1, int(rng.uniform(0.004, 0.012) * n))
        L = rng.uniform(0.22, 0.36) * n
        strip = np.zeros((n, n))
        y0 = int(n / 2 - w / 2)
        x0, x1 = int(n / 2), int(n / 2 + L)
        strip[y0:y0 + w, x0:x1] = 1.0
        bars = np.maximum(bars, rotate(strip, rng.uniform(0, 360), reshape=False,
                                       order=0, mode="constant"))
    return np.clip(bg + bars, 0, 1) * _circle_mask(n)


PHANTOMS = {"ellipses": ellipses, "blocks": blocks, "mixed": mixed}


def make_phantom(name, n, seed=0, oversample=4):
    # high-res object (for projecting) + block-averaged ground truth (for scoring)
    hi = PHANTOMS[name](n * oversample, seed=seed)
    gt = hi.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
    return hi, gt


def rebin_sinogram(p_hi, oversample):
    # average oversample adjacent detectors; the /oversample converts line integrals
    # from fine-pixel units back to coarse-pixel units
    n_ang, n_det_hi = p_hi.shape
    n_det = n_det_hi // oversample
    return p_hi[:, :n_det * oversample].reshape(n_ang, n_det, oversample).mean(2) / oversample
