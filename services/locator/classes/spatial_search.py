import os
import numpy as np
import struct
import collections

class SpatialSearch():
    def __init__(self, config):
        self.config = config
        self.header_size = 20
        self.streamline_point_size = 20
        self.uint_size = 4

    def get(self, voxel = None, map_dir = None, density_threshold = 0.01):
        results = dict()
        results.setdefault('success', False)

        if len(voxel) < 3 or map_dir == None:
            results.setdefault("message", "invalid/missing voxel coordinate or map_dir")

        map_file = self.get_projection_map_file(voxel, map_dir)
        if not os.path.isfile(map_file):
            return {'success': True, 'results': []}

        pmaps = self.read_projection_map_header(map_file)
        num_experiments = len(pmaps)

        pmaps = [ pmap for pmap in pmaps if pmap['density'] >= density_threshold]

        results.setdefault("results", [])
        with open(map_file, 'rb') as f:
            f.seek(self.uint_size + self.header_size * num_experiments)
            pmap_data = np.fromfile(f, dtype='float32')

        for pmap in pmaps:
            coords = self.read_projection_map_streamlines(pmap_data, pmap['num_points'], pmap['file_offset'])
            pmap['coords'] = coords
            results['results'].append(pmap)

        results["success"] = True
        return results


    def get_projection_map_file (self, voxel, map_dir):
        voxel = np.round(voxel, decimals = -2)
        map_file = os.path.join(map_dir, "%d" % voxel[0], "%d_%d_%d" % (voxel[0], voxel[1], voxel[2]))

        return map_file

    def read_projection_map_header(self, map_file):
        with open(map_file, 'rb') as f:
            experiment_count, = struct.unpack('<I', f.read(self.uint_size))

            result = []
            for i in range(experiment_count):
                data_id, density, transgenic_line, file_offset, num_points = struct.unpack('<IfIII', f.read(self.header_size))

                result.append({
                    'data_set_id': data_id,
                    'density': density,
                    'transgenic_line_id': transgenic_line, 
                    'file_offset': file_offset, 
                    'num_points': num_points
                })

            return result

    def read_projection_map_streamlines(self, pmap_data, num_points, offset):
        offset = offset // np.dtype('<f').itemsize
        data = pmap_data[offset:(offset+num_points*5)]

        x,y,z,d,i = data[0::5], data[1::5], data[2::5], data[3::5], data[4::5]
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()
        d = d.tolist()
        i = i.tolist()
        coords = [
            {'coord': (xi,yi,zi), 'density': di, 'intensity': ii}
            for xi,yi,zi,di,ii in zip(x,y,z,d,i)
        ]
        #coords = {'coords': (x,y,z), 'density': d, 'intensity': i}

        return coords
