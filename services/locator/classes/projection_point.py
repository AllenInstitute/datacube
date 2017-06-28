import os
import numpy as np
import struct
import binascii

class ProjectionPoint ():
    def __init__(self):
        global components
        global projection_width
        global projection_height
        global size_of_int

        components = 3
        projection_width = 1140
        projection_height = 1320
        size_of_int = 4

    def get(self, path, pixel):
        if pixel[0] < 0 or pixel[0] >= projection_width or pixel[1] < 0 or pixel[1] >= projection_height:
            raise ValueError("pixel coordinates out of range(" + projection_width + "," + projection_height + "), given: [" + pixel[0] + ", " + pixel[1] + "]")

        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise IOError("file does not exist")

        f = file(path, 'r')
        
        pos = (projection_height * pixel[0] + pixel[1]) * components * size_of_int

        self.skip_header(f)

        f.seek(pos, 1)

        x = struct.unpack('<i', f.read(size_of_int))[0]
        y = struct.unpack('<i', f.read(size_of_int))[0]
        z = struct.unpack('<i', f.read(size_of_int))[0]

        return [x, y, z]
        

    def skip_header (self, f):
        in_header = True

        while in_header:
            b = f.readline()

            if b == '\n':
                in_header = False
        
