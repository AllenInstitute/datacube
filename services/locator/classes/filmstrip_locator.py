import os
import h5py
import numpy as np
import json
import math
import struct

class FilmStripLocator ():
    def __init__(self, config):
        self.horizontal_index = 2
        self.annot_dimensions = config.get_property("ccf_annotation_dimensions")
        self.annot_spacing = config.get_property("ccf_annotation_spacing")
        self.frames = 36
        self.rotation = 360.0
        self.config = config

    def get(self, pixel, distance_map_path, direction, results):
        try:
            self.prepare_files(pixel, distance_map_path, direction)
            self.get_volume_coordinate(pixel, distance_map_path, direction)
            self.phys_coord = [i * self.annot_spacing for i in self.vol_coord]
            
            results.setdefault("volumeCoordinate", list(self.vol_coord))
            results.setdefault("physicalCoordinate", self.phys_coord)
            results["success"] = True
            
            return results

        except (IOError, ValueError) as e:
            return results.setdefault('message', str(e))


    def prepare_files(self, pixel, distance_map_path, direction):
        if len(pixel) != 2: 
            raise ValueError("Pixel requires two dimensions")

        if not os.path.exists(distance_map_path) or not os.path.getsize(distance_map_path) > 0:
            raise IOError("Distance map does not exist or was not provided")

        # new up our structures, just in case one ends up being empty when we return errything
        self.vol_coord = np.array([-1, -1, -1])
        self.phys_coord = np.array([-1, -1, -1])

        self.read_mhd_header(distance_map_path)

        self.frame_width = self.dist_map_dimensions[0] / self.frames
        self.vol_padding = np.array([0.0, 0.0, 0.0])

        self.dist_map_spacing.append(self.dist_map_spacing[0])
        if direction == "vertical":
            x_len = self.annot_dimensions[0]
            y_len = self.annot_dimensions[1]
            z_len = self.annot_dimensions[2]

            max_hor_size = np.linalg.norm([x_len, z_len])
            max_ver_size = np.linalg.norm([y_len, z_len])

            self.vol_padding[0] = (max_hor_size - x_len) * .5
            self.vol_padding[1] = (max_ver_size - y_len) * .5


        self.center_of_rotation = [0, 0, 0]
        self.depth_scale = [1.0, 1.0, 1.0]

        self.center_of_rotation = [i * .5 for i in self.annot_dimensions]
        self.depth_scale = [float(i) / float(self.annot_spacing) for i in self.dist_map_spacing]


    def get_volume_coordinate(self, pixel, distance_map_path, direction):
        depth = self.sample_distance_map(pixel)

        if depth <= 0.0:
            raise IOError("pixel not in range of distance map")
        
        frame = pixel[0] // self.frame_width

        self.vol_coord = np.array([pixel[0] - (frame * self.frame_width), pixel[1], depth])
        self.vol_coord = (self.vol_coord - self.vol_padding) * self.depth_scale
        self.vol_coord = self.rotate_volume_coordinate(self.vol_coord, frame, direction)

    def get_rotation_matrix(self, theta, rotation_direction):
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        if rotation_direction == "horizontal":
            return [[ ctheta, 0.0,   -stheta, 0.0],
                    [ 0.0,    1.0,   0.0,     0.0],
                    [ stheta, 0.0,   ctheta,  0.0],
                    [ 0.0,    0.0,   0.0,     1.0]]

        if rotation_direction == "vertical":
            return [[ 1.0,   0.0,     0.0,       0.0 ],
                    [ 0.0,   ctheta,  -stheta,   0.0 ],
                    [ 0.0,   stheta,  ctheta,    0.0 ],
                    [ 0.0,   0.0,     0.0,       1.0 ]]

    def sample_distance_map(self, pixel_coord):
        with open(self.dist_map_raw, 'rb') as raw_file:
            index = int(pixel_coord[1] * self.frame_width * self.frames + pixel_coord[0])
            unsigned_int_size = 4
        
            raw_file.seek(index * unsigned_int_size)
            d = struct.unpack('<i', raw_file.read(unsigned_int_size))[0]
        return d

    def rotate_volume_coordinate(self, vol_coords, angle, direction):
        deg = -angle * self.rotation / self.frames
        theta = deg * math.pi / 180.0

        tmp_coord = np.ones(4)
        tmp_coord[:3] = vol_coords - self.center_of_rotation

        transform = np.array(self.get_rotation_matrix(theta, direction))

        tmp_coord = np.dot(transform, tmp_coord)

        return tmp_coord[:3] + self.center_of_rotation

    def read_mhd_header(self, meta_file):
        with open(meta_file, 'r') as f:
            meta = f.readlines()
        
        for line in meta:
            if "ElementSpacing" in line:
                line = line.split('=')
                line = line[1].strip().split(' ')
                self.dist_map_spacing = [int(line[0]), int(line[1])]

            if "DimSize" in line:
                line = line.split('=')
                line = line[1].strip().split(' ')
                self.dist_map_dimensions = [int(line[0]), int(line[1])] 

            if "ElementDataFile" in line:
                line = line.split('=')
                line = line[1].strip()
                self.dist_map_raw = line

        self.dist_map_raw = os.path.join(os.path.dirname(meta_file), self.dist_map_raw)




