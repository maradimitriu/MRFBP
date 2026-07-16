from .fbp import fbp
from .filters import make_filter
from .mrfbp import mrfbp
from .sirt import sirt

# method names used across the experiments and the paper
METHODS = ["fbp-ram-lak", "fbp-shepp-logan", "fbp-hann", "sirt-200", "sirt-1000", "mrfbp"]

LABELS = {"fbp-ram-lak": "FBP-RL", "fbp-shepp-logan": "FBP-SL", "fbp-hann": "FBP-HN",
          "sirt-200": "SIRT-200", "sirt-1000": "SIRT-1000", "mrfbp": "MR-FBP",
          "mrfbp-gm": "MR-FBP$_{GM}$"}


def reconstruct(method, geom, p, **kw):
    if method.startswith("fbp-"):
        return fbp(geom, p, make_filter(geom.n_det, method[4:]))
    if method.startswith("sirt-"):
        return sirt(geom, p, n_iter=int(method[5:]))
    if method == "mrfbp":
        return mrfbp(geom, p, **kw)[0]
    raise ValueError(f"unknown method: {method}")
