import os
import numpy as np
import struct
import binascii

class ProjectionPoint ():
    def __init__(self, config):
        self.config = config
        self.components = 3
        self.projection_width = 1140
        self.projection_height = 1320
        self.size_of_int = 4

    def get(self, path, pixel):

        if len(pixel) != 2:
            raise ValueError("pixel requires two dimensions")

        if pixel[0] < 0 or pixel[0] >= self.projection_width or pixel[1] < 0 or pixel[1] >= self.projection_height:
            raise ValueError("pixel coordinates out of range(" + self.projection_width + "," + self.projection_height + "), given: [" + pixel[0] + ", " + pixel[1] + "]")

        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise IOError("file does not exist")

        f = file(path, 'r')
        
        pos = (self.projection_height * pixel[0] + pixel[1]) * self.components * self.size_of_int

        self.skip_header(f)

        f.seek(pos, 1)

        return struct.unpack('<iii', f.read(self.size_of_int*3))        

    def skip_header (self, f):
        in_header = True

        while in_header:
            b = f.readline()

            if b == '\n':
                in_header = False
        
