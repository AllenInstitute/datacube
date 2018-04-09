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
LOCK = threading.RLock()

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
    
    images = ( MetaImage(red_path),
               MetaImage(green_path),
               MetaImage(blue_path) )

    return images

def load_memmap_images(k):
    images = load_images(k)
    return tuple(i.memmap_image() for i in images)

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

def plane_width_height_index(plane, width, height, index, dims, spacing, max_dimension):
    if plane == 'sagittal':
        d0, d1, d2 = 2, 1, 0
    elif plane == 'coronal':
        d0, d1, d2 = 1, 0, 2
    else:
        d0, d1, d2 = 0, 2, 1

    index = int(float(index) / float(spacing[d2]))

    width_height_ratio = (float(dims[d0]) * spacing[d0]) / (float(dims[d1]) * spacing[d1])
    real_width = dims[d0]
    real_height = dims[d1]

    if height and height > max_dimension:
        height = None

    if width and width > max_dimension:
        width = None

    if height is None:
        if width:
            height = int(float(width) / width_height_ratio)
        else:
            height = real_height
            width = int(float(height * width_height_ratio))
    
    if width is None:
        if height:
            width = int(float(height) * width_height_ratio)
        else:
            width = real_width
            height = int(float(width / width_height_ratio))

    return width, height, index, (d0, d1, d2)
        
        

STACK_VOLUME_CACHE = LRUCache(maxsize=STACK_VOLUME_CACHE_SIZE, missing=load_memmap_images)

class StackVolumeSlice():
    def __init__(self, config):
        self.config = config

    def image_metadata(self, storage_dir, spacing):
        r,_,_ = load_images((storage_dir, spacing))
        return r.metadata

    def get(self, storage_dir, plane, index, 
            width, height, 
            value_range, 
            quality=40):
        
        spacing = self.config.get_property("stack_volume_spacing")
        r, g, b = STACK_VOLUME_CACHE[(storage_dir, spacing)]
        md = self.image_metadata(storage_dir, spacing)

        width, height, index, dims = plane_width_height_index(plane, width, height, index, 
                                                              md['DimSize'], md['ElementSpacing'],
                                                              self.config.get_property('stack_volume_max_dimension'))

        with LOCK:
            im = np.dstack((slice3d(r, dims[2], index),
                            slice3d(g, dims[2], index),
                            slice3d(b, dims[2], index))).astype(float)                    

        if plane == 'sagittal':
            im = np.fliplr(im)
        elif plane == 'coronal':
            im = np.transpose(im, (1,0,2))
        elif plane == 'horizontal':
            im = np.flipud(np.fliplr(im))

        width_factor = float(width) / im.shape[1]
        height_factor = float(height) / im.shape[0]
        
        # resample image
        im = zoom(im, [ height_factor, width_factor, 1.0 ], 
                  order=self.config.get_property('stack_volume_interpolation_order'))

        im = image_16b_to_8b(im, value_range)

        return image_jpeg_response(im, quality)

class StackVolumeImageInfo():
    def __init__(self, config):
        self.config = config

    def get(self, storage_dir):
        spacing = self.config.get_property("stack_volume_spacing")
        r, _, _ = load_images((storage_dir, spacing))
        md = r.metadata
        
        return {'data': { 'size': md['DimSize'], 
                          'spacing': md['ElementSpacing'] }}

if __name__ == "__main__":
    sdir = "/external/conn/prod626/image_series_268321927/"
    plane='horizontal'
    index=7100
    width=179
    height=134
    value_range=[0,1116,0,2652,0,4095]

    from configuration_manager import ConfigurationManager
    svs = StackVolumeSlice(ConfigurationManager())
    buf = svs.get(sdir, plane, index, width, height, value_range)
    print(buf)


        

        
