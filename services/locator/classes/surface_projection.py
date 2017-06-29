import os
import h5py
import numpy

class SurfacePoint ():
    def __init__(self, config):
        self.config = config


    def get (self, seedPoint):

        if len(seedPoint) !=  3:
            raise ValueError("seedPoint must have three dimensions")

        path = self.config.get_property("p56_file_path")

        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise IOError("could not find surface_coords_10.h5")
        
        f = h5py.File(path)

        # Watch for an argument out of range err?
        seedPoint = [self.micron2vox(i) for i in seedPoint]
        one_d_index = f['lut'][seedPoint[0], seedPoint[1], seedPoint[2]]
        ind = numpy.unravel_index(one_d_index, f['lut'].shape)

        result = list(ind)
        result = [self.vox2micron(i) for i in result]
        
        return result

    def micron2vox(self, micron):
        return int(micron * 0.1)

    def vox2micron(self, vox):
        return vox * 10
