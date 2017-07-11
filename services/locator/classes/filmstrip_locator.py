import os
import h5py
import numpy as np
import json
import math
import struct
from ontology_service import OntologyService

#  #MATH #LOL #CPLUSPLUS

class FilmStripLocator ():
    def __init__(self, config):
        self.horizontal_index = 2
        self.annotation_dimensions = []
        self.frames = 36
        self.rotation = 360.0
        self.config = config

    def get(self, pixel, distance_map_path, direction, results):

        try:
            self.prepare_files(pixel, distance_map_path, direction)
            self.get_volume_coordinate(pixel, distance_map_path, direction)
    
            # get_physical_coordinate
            self.phys_coord = [int(i * self.annot_spacing) for i in self.vol_coord]
            
            self.get_hemisphere()
            self.get_volume_annotation()
            results.setdefault("volumeCoordinate", list(self.vol_coord))
            results.setdefault("physicalCoordinate", self.phys_coord)
            results.setdefault("hemisphere", self.hem)
            results.setdefault("structure", self.structure)
            results["success"] = True
            
            return results

        except (IOError, ValueError) as e:
            print "?!?!?!??"
            return results.setdefault('message', e.message)


    def prepare_files(self, pixel, distance_map_path, direction):
        if len(pixel) != 2: 
            raise ValueError("Pixel requires two dimensions")

        if not os.path.exists(distance_map_path) or not os.path.getsize(distance_map_path) > 0:
            raise IOError("Distance map does not exist or was not provided")

        # new up our structures, just in case one ends up being empty when we return errything
        self.vol_coord = np.array([-1, -1, -1])
        self.phys_coord = np.array([-1, -1, -1])
        self.hem = ""
        self.structure = dict()

        self.get_annotation_files()
        self.read_annotation_header()
        self.read_mhd_header(distance_map_path)

        self.frame_width = self.dist_map_dimensions[0] / self.frames
        self.vol_padding = np.array([0, 0, 0])

        self.dist_map_spacing.append(self.dist_map_spacing[0])
        if direction == "vertical":
            x_len = self.annot_dimensions[0]
            y_len = self.annot_dimensions[0]
            z_len = self.annot_dimensions[0]

            max_hor_size = np.linalg.norm([x_len, z_len])
            max_ver_size = np.linalg.norm([y_len, z_len])

            self.vol_padding[0] = (max_hor_size - x_len) * .5
            self.vol_padding[1] = (max_ver_size - y_len) * .5


        self.center_of_rotation = [0, 0, 0]
        self.depth_scale = [1, 1, 1]

        self.center_of_rotation = [i * .5 for i in self.annot_dimensions]
        self.depth_scale = [i / self.annot_spacing for i in self.dist_map_spacing]


    def get_volume_coordinate(self, pixel, distance_map_path, direction):
        depth = self.sample_distance_map(pixel)

        if depth <= 0.0:
            raise IOError("pixel not in range of distance map")

        angle = pixel[0] / self.frame_width

        self.vol_coord = np.array([pixel[0] - (angle * self.frame_width), pixel[1], depth])

        self.vol_coord = (self.vol_coord - self.vol_padding) * self.depth_scale
    
        self.rotate_volume_coordinate(self.vol_coord, angle, direction)

    def get_annotation_files(self):
        meta_file = self.config.get_property("p56_annotation_meta_file")

        with open (meta_file) as js_dat:
            self.annotation_meta = json.load(js_dat)

    def get_volume_annotation(self):
        annotation_file = self.config.get_property("p56_annotation_file")
        shape = tuple(self.annot_dimensions)
        
        annotation_volume = np.load(annotation_file)
        
        id = annotation_volume[self.vol_coord[0], self.vol_coord[1], self.vol_coord[2]]
        
        ontology = OntologyService(self.config)
        
        s = ontology.get_structure_by_id(id)

        self.structure.setdefault('id', int(id))
        self.structure.setdefault('name', s['safe_name'] )
        self.structure.setdefault('abbreviation', s['acronym'])
        self.structure.setdefault('color', s['color_hex_triplet'])
        
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
                    [ 0.0,   0.0,     0.0,       1.0 ] ]

    def get_hemisphere(self):
        if self.vol_coord[self.horizontal_index] > self.annot_dimensions[self.horizontal_index] / 2:
            self.hem = u'right'
        else:
            self.hem = u'left'

    def sample_distance_map(self, pixel_coord):
        raw_file = file(self.dist_map_raw)
        
        index = pixel_coord[1] * self.frame_width * self.frames + pixel_coord[0]
        unsigned_int_size = 4
        
        raw_file.seek(index * unsigned_int_size)
        
        return struct.unpack('<i', raw_file.read(unsigned_int_size))[0]

    def rotate_volume_coordinate(self, vol_coords, angle, direction):
        deg = -angle * self.rotation / self.frames
        theta = deg * math.pi / 180

        tmp_coord = vol_coords
        
        tmp_coord = [tmp_coord[i] - self.center_of_rotation[i] for i in range(0,3)]
        tmp_coord.append(1)

        transform = np.matrix(self.get_rotation_matrix(theta, direction))

        tmp_coord = np.dot(transform, tmp_coord)

        vol_coords = [int(vol_coords[i] + self.center_of_rotation[i]) for i in range(0, 3)]

    def read_mhd_header(self, meta_file):
        meta = open(meta_file).readlines()
        
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


    def read_annotation_header(self):
        self.annot_dimensions = self.annotation_meta['sizes']
        self.annot_spacing = int(self.annotation_meta['space directions'][0][0])



