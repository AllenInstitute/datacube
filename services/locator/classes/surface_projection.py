import os
import h5py
import numpy as np

class SurfacePoint ():
    def __init__(self, config):
        self.config = config
        self.spacing_um = [10, 10, 10]


    def get (self, seedPoint):

        if len(seedPoint) !=  3:
            raise ValueError("seedPoint must have three dimensions")

        path = self.config.get_property("p56_file_path")

        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise IOError("could not find surface_coords_10.h5")
        
        f = h5py.File(path)

        seedPoint = np.array(seedPoint, dtype=np.int) / self.spacing_um
        one_d_index = f['lut'][seedPoint[0], seedPoint[1], seedPoint[2]]
        ind = np.unravel_index(one_d_index, f['lut'].shape)

        result = np.array(ind) * self.spacing_um
        
        return result.tolist()

    def micron2vox(self, micron):
        return int(micron * 0.1)

    def vox2micron(self, vox):
        return vox * 10
