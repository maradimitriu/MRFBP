import numpy as np
import astra


# thin astra wrapper. astra's backprojection is not the exact transpose of its
# forward projection, so we use the same operator pair everywhere (building A_p,
# reconstructing, sirt) to keep everything consistant.
class Geometry:
    def __init__(self, n_pixels, n_det, n_angles, use_gpu=True):
        self.n_pixels = n_pixels
        self.n_det = n_det
        self.n_angles = n_angles
        self.angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, self.angles)
        self.proj_id = astra.create_projector(
            "cuda" if use_gpu else "linear", self.proj_geom, self.vol_geom
        )

    def fp(self, x):
        # forward projection W x -> sinogram
        sid, p = astra.create_sino(np.ascontiguousarray(x, dtype=np.float32), self.proj_id)
        astra.data2d.delete(sid)      # free the id or astra leaks memory
        return p

    def bp(self, p):
        # backprojection W^T p -> image
        vid, x = astra.create_backprojection(
            np.ascontiguousarray(p, dtype=np.float32), self.proj_id
        )
        astra.data2d.delete(vid)
        return x
