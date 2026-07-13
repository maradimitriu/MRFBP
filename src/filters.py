"""FBP filters and the filtering step  q = C_h p  (convolution along the detector axis)."""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


def filter_length(n_det):
    """Filters live on integer detector offsets n = -(n_det-1) ... (n_det-1)."""
    return 2 * n_det - 1


def make_filter(n_det, name="ram-lak"):
    """Standard static FBP filters, returned as a *centred* real-space vector.

    Index n_det-1 corresponds to offset 0. Built from the frequency response
    |u| times a window, so all filters share one representation.
    """
    M = filter_length(n_det)
    u = np.fft.fftfreq(M)                 # normalised frequency, [-0.5, 0.5)
    ramp = np.abs(u)                      # |u|: the ideal ramp. The pi/N_theta
    #                                       normalisation lives in fbp(); do not
    #                                       also put a factor 2 here.
    if name == "ram-lak":
        H = ramp
    elif name == "shepp-logan":
        H = ramp * np.sinc(u)             # np.sinc(u) = sin(pi u)/(pi u)
    elif name == "hann":
        H = ramp * (0.5 + 0.5 * np.cos(2 * np.pi * u))
    elif name == "cosine":
        H = ramp * np.cos(np.pi * u)
    else:
        raise ValueError(f"unknown filter: {name}")
    h = np.real(np.fft.ifft(H))
    return np.fft.fftshift(h)             # centre the kernel


def apply_filter(p, h):
    """q = C_h p : convolve every sinogram row with the centred filter h.

    p : (n_angles, n_det)   sinogram
    h : (2*n_det-1,)        centred filter
    Returns (n_angles, n_det).

    The FFT is zero-padded to at least len(p_row)+len(h)-1, so this is a true
    linear convolution -- no circular wrap-around. Getting this wrong is the
    single most common source of subtly incorrect FBP reconstructions.
    """
    n_det = p.shape[1]
    M = h.shape[0]
    L = n_det + M - 1                     # full linear-convolution length
    nfft = next_fast_len(L)
    q = irfft(rfft(p, nfft, axis=1) * rfft(h, nfft), nfft, axis=1)
    lo = (M - 1) // 2                     # crop back to 'same' support
    return q[:, lo:lo + n_det]
