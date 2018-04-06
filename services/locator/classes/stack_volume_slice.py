import os
from metaimage import MetaImage
from scipy.ndimage import zoom
import numpy as np
from PIL import Image
from io import StringIO, BytesIO
import base64

class StackVolumeCache():
    def __init__(self):
        self.images = {}

    def get_images(self, key):
        return self.images.get(key, None)

    def add_images(self, key, images):
        self.images[key] = images

class StackVolumeSlice():
    def __init__(self, config):
        self.config = config
        self.cache = StackVolumeCache()

    def get_images(self, storage_dir, spacing):
        images_key = (storage_dir, spacing)
        images = self.cache.get_images(images_key)

        if images is not None:
            return images

        red_path = os.path.join(storage_dir,
                                "grid",
                                "red_%s.mhd" % spacing)

        green_path = os.path.join(storage_dir,
                                  "grid",
                                  "green_%s.mhd" % spacing)

        blue_path = os.path.join(storage_dir,
                                 "grid",
                                 "blue_%s.mhd" % spacing)

        images = MetaImage(red_path), MetaImage(green_path), MetaImage(blue_path)

        self.cache.add_images(images_key, images)

        return images

    def get(self, storage_dir, plane, index, 
            width, height, 
            value_range, 
            spacing="44.8", quality=40):
        r, g, b = self.get_images(storage_dir, spacing)

        im = np.dstack((r.slice3d(plane, index),
                        g.slice3d(plane, index),
                        b.slice3d(plane, index))).astype(float)

        if width is not None or height is not None:
            if height is not None:
                height_factor = float(height) / float(im.shape[0])
                if width is None:
                    width_factor = height_factor

            if width is not None:
                width_factor = float(width) / float(im.shape[1])
                if height is None:
                    height_factor = width_factor

            im = zoom(im, [ height_factor, width_factor, 1.0 ], order=0)

        value_range = np.array(value_range) / 255.0

        shift = [ - value_range[0] / (value_range[1] - value_range[0]),
                  - value_range[2] / (value_range[3] - value_range[2]),
                  - value_range[4] / (value_range[5] - value_range[4]) ]
        scale = [ 1.0 / (value_range[1] - value_range[0]),
                  1.0 / (value_range[3] - value_range[2]),
                  1.0 / (value_range[5] - value_range[4]) ]
        
        im = np.clip(im*scale + shift, 0, 255).astype(np.uint8)
        
        im = Image.fromarray(im)
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=quality)
        return { 'data': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() }
        
        


if __name__ == "__main__":
    sdir = "/external/ctyconn/prod17/image_series_609475867/"
    svs = StackVolumeSlice({})
    buf = svs.get(sdir, 0, 300, None, 30, [0,761,0,1286,0,81])
    print(buf)
        

        
