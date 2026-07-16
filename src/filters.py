import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


def filter_length(n_det):
    # filters live on integer detector offsets -(n_det-1) .. (n_det-1)
    return 2 * n_det - 1


def make_filter(n_det, name="ram-lak"):
    M = filter_length(n_det)
    u = np.fft.fftfreq(M)
    ramp = np.abs(u)              # |u|, the ideal ramp (pi/N_theta factor is in fbp())
    if name == "ram-lak":
        H = ramp
    elif name == "shepp-logan":
        H = ramp * np.sinc(u)
    elif name == "hann":
        H = ramp * (0.5 + 0.5 * np.cos(2 * np.pi * u))
    elif name == "cosine":
        H = ramp * np.cos(np.pi * u)
    else:
        raise ValueError(f"unknown filter: {name}")
    h = np.real(np.fft.ifft(H))
    return np.fft.fftshift(h)     # centre the kernel


def apply_filter(p, h):
    # convolve every sinogram row with h. zero-pad the fft so the convolution is
    # linaer and not circular, otherwise the filter tails wrap around the detector.
    n_det = p.shape[1]
    M = h.shape[0]
    nfft = next_fast_len(n_det + M - 1)
    q = irfft(rfft(p, nfft, axis=1) * rfft(h, nfft), nfft, axis=1)
    lo = (M - 1) // 2
    return q[:, lo:lo + n_det]
