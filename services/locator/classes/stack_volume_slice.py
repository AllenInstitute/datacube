import os
from .metaimage import MetaImage, slice3d
from scipy.ndimage import zoom
import numpy as np
from PIL import Image
from io import StringIO, BytesIO
import base64
from cachetools import LRUCache
import threading

STACK_VOLUME_CACHE_SIZE = 10
STACK_VOLUME_CACHE = None

def load_images(k):
    storage_dir, spacing = k
    red_path = os.path.join(storage_dir,
                            "grid",
                            "red_%s.mhd" % spacing)

    green_path = os.path.join(storage_dir,
                              "grid",
                              "green_%s.mhd" % spacing)

    blue_path = os.path.join(storage_dir,
                             "grid",
                             "blue_%s.mhd" % spacing)
    
    images = ( MetaImage(red_path).memmap_image(),
               MetaImage(green_path).memmap_image(),
               MetaImage(blue_path).memmap_image() )

    return images

def resample_image(im, width, height, order, max_dimension):          
    if width is not None or height is not None:
        if height is not None:
            height_factor = float(height) / float(im.shape[0])
            if width is None:
                width_factor = height_factor

        if width is not None:
            width_factor = float(width) / float(im.shape[1])
            if height is None:
                height_factor = width_factor

        if (width is not None and width * width_factor > max_dimension) or \
           (height is not None and height * height_factor > max_dimension):
            raise IOError("Requested dimensions exceed maximum dimension size (%d)" % max_dimension)    

        im = zoom(im, [ height_factor, width_factor, 1.0 ], order=order)

    return im

def image_16b_to_8b(im, value_range):
    value_range = np.array(value_range) / 255.0
    
    shift = [ - value_range[0] / (value_range[1] - value_range[0]),
              - value_range[2] / (value_range[3] - value_range[2]),
              - value_range[4] / (value_range[5] - value_range[4]) ]

    scale = [ 1.0 / (value_range[1] - value_range[0]),
              1.0 / (value_range[3] - value_range[2]),
              1.0 / (value_range[5] - value_range[4]) ]
    
    return np.clip(im*scale + shift, 0, 255).astype(np.uint8)

def image_jpeg_response(im, quality):
    im = Image.fromarray(im)
    buf = BytesIO()
    im.save(buf, format='JPEG', quality=quality)
    return { 'data': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() }

STACK_VOLUME_CACHE = LRUCache(maxsize=STACK_VOLUME_CACHE_SIZE, missing=load_images)

class StackVolumeSlice():
    def __init__(self, config):
        self.config = config

    def get(self, storage_dir, plane, index, 
            width, height, 
            value_range, 
            quality=40):

        spacing = self.config.get_property("stack_volume_spacing")
        r, g, b = STACK_VOLUME_CACHE[(storage_dir, spacing)]

        # begin mutex
        im = np.dstack((slice3d(r, plane, index),
                        slice3d(g, plane, index),
                        slice3d(b, plane, index))).astype(float)
        # end mutex

        im = resample_image(im, width, height, 
                            self.config.get_property('stack_volume_interpolation_order'),
                            self.config.get_property('stack_volume_max_dimension'))

        im = image_16b_to_8b(im, value_range)
        
        return image_jpeg_response(im, quality)

if __name__ == "__main__":
    sdir = "/external/ctyconn/prod17/image_series_609475867/"
    svs = StackVolumeSlice({})
    import time
    for i in range(100):
        for axis, index in [(0,100), (1,100), (2,100)]:
            t1 = time.time()
            buf = svs.get(sdir, 0, 300, None, 30, [0,761,0,1286,0,81])
            t2 = time.time()
            print(t2-t1)


        

        
