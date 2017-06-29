import os
import h5py

class FilmStripLocator ():
    def __init__(self, config):
        self.horizontal_index = 2
        self.annotation_dimensions = []
        self.config = config

    def get(self, pixel, distanceMapPath, direction):

        if len(pixel) != 2: 
            raise ValueError("pixel requires two dimensions")

        if not os.path.exists(distanceMapPath) or not os.path.getsize(distanceMapPath) > 0:
            raise IOError("file does not exist")

        # new up our structures, just in case one ends up being empty when we return errything
        vol_coord = []
        phys_coord = []
        hem = ""
        structure = dict()

        annotation_image = h5py.File(self.config.get_property("p56_file_path"))

        print annotation_image['lut']



        return vol_coord, phys_coord, hem, structure

    def get_volume_coordinate(self):
        pass

    def get_volume_annotation(self, phys_coord):
        pass

    def sample_distance_map(self, file, pixel_coord):
        pass

    def get_rotation_matrix(self, theta, rotation_direction, matrix):
        pass

    def rotate_volume_coordinate(self, volume_coords, angle):
        pass

    def read_meta_header(self, header, dimensions, spacing):
        # determine if header exists
        pass

    def get_physical_coordinate(self, vol_coord, phys_coord):
        pass


    def get_hemisphere(self, vol_coords):
        if vol_coords[horizontal_index] > self.annotation_dimensions / 2:
            return "right"
        else:
            return "left"

