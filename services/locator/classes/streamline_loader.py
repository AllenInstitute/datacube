import nrrd
import json
import numpy as np
import struct

class StreamlineLoader():
    def __init__(self, base_path):
        self.base_path = base_path
        self.injections_file = "grid/injection_fraction_100.nrrd"
        self.streamlines_file = "grid/streamlines.sl"

    def load_lines(self):
        path = self.base_path + self.streamlines_file
        lines = []

        with open(path, 'rb') as f:
            magic = struct.unpack('I', f.read(4))[0]
            version = struct.unpack('H', f.read(2))[0]
            num_lines = struct.unpack('I', f.read(4))[0]

            for i in xrange(num_lines):
                num_points = struct.unpack('H', f.read(2))[0]
                points = []

                for j in xrange(num_points):
                    x,y,z,density,intensity, = struct.unpack('fffff', f.read(20))
                    points.append({
                        'x': int(x * 100),
                        'y': int(y * 100),
                        'z': int(z * 100),
                        'density': density,
                        'intensity': intensity
                    })

                lines.append(points)

        return lines

    def load_injections(self):
        path = self.base_path + self.injections_file
        img, info = nrrd.read(path)

        dims = info['sizes']
        inds = np.where(img > .5)
        data = [ { 'x': int(d[0] * 100), 'y': int(d[1] * 100), 'z': int(d[2] * 100) } for d in zip(inds[0], inds[1], inds[2])]

        return data
