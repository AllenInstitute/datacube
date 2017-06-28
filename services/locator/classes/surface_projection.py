import os
import binascii
import h5py
import numpy

class SurfacePoint ():
    def __init__(self):
        global filePath
        global file

        filePath = "/allen/programs/celltypes/production/0378/informatics/model/P56/corticalCoordinates/"
        file = "surface_coords_10.h5"


    def get (self, seedPoint):

        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise FileNotFoundError("could not find surface_coords_10.h5")
        
        f = h5py.File(filePath + file)

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
