import os

class ProjectionPoint ():
    def __init__(self):
        global components
        global projection_width
        global projection_height

        components = 3
        projection_width = 1140
        projection_height = 1320

    def get(self, path, pixel):
        if pixel[0] < 0 or pixel[0] >= projection_width or pixel[1] < 0 or pixel[1] >= projection_height:
            raise ValueError("pixel coordinates out of range(" + projection_width + "," + projection_height + "), given: [" + pixel[0] + ", " + pixel[1] + "]")


        if not os.path.exists(path) or not os.path.getsize(path) > 0:
            raise IOError("file does not exist")

        
        
        f = file(path)

        self.skip_nrrd_header(f)

        pos = (projection_height * pixel[0] + pixel[1]) * components * 4 # byte size of an int


    def skip_nrrd_header (self, file):
        in_header = True

        while in_header:
            b = file.read(1)

            print b
            
            if b is '\n':
                in_header = False

        file.read(1)

