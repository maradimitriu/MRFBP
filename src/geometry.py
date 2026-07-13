"""Thin ASTRA wrapper exposing the two operators the whole project needs:

    geom.fp(x) -> p     the projection matrix   W
    geom.bp(p) -> x     its (approximate) transpose  W^T

NOTE: ASTRA's backprojection is not the exact transpose of its forward
projection. That is fine -- but the SAME operator pair must be used
everywhere (building A_p, reconstructing, SIRT). Consistency, not exactness,
is what makes the computed filter correct.
"""
import numpy as np
import astra


class Geometry:
    def __init__(self, n_pixels, n_det, n_angles, use_gpu=True):
        self.n_pixels = n_pixels
        self.n_det = n_det
        self.n_angles = n_angles
        # Parallel beam, angles equidistant over [0, pi), detector width 1, pixel width 1.
        self.angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, self.angles)
        self.proj_id = astra.create_projector(
            "cuda" if use_gpu else "linear", self.proj_geom, self.vol_geom
        )

    def fp(self, x):
        """Forward projection W x -> sinogram of shape (n_angles, n_det)."""
        sid, p = astra.create_sino(np.ascontiguousarray(x, dtype=np.float32), self.proj_id)
        astra.data2d.delete(sid)          # ASTRA leaks memory if ids are not freed
        return p

    def bp(self, p):
        """Backprojection W^T p -> image of shape (n_pixels, n_pixels)."""
        vid, x = astra.create_backprojection(
            np.ascontiguousarray(p, dtype=np.float32), self.proj_id
        )
        astra.data2d.delete(vid)
        return x
