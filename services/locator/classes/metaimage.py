import os
import numpy as np

class MetaImage():
    def __init__(self, mhd_file):
        self.mhd_file = mhd_file
        self._metadata = None
        self._image = None
        self.type_map = {
            'MET_SHORT': np.int16,
            'MET_USHORT': np.uint16,
            'MET_INT': np.int32,
            'MET_UINT': np.uint32,
            'MET_FLOAT': np.float16,
            'MET_DOUBLE': np.float32
        }
            

    @property
    def metadata(self):
        if self._metadata is not None:
            return self._metadata

        with open(self.mhd_file) as f:
            lines = f.readlines()

        params = {}

        for line in lines:
            for k in [ "ElementSpacing" ]:
                if k in line:
                    line = line.split('=')
                    line = line[1].strip().split(' ')
                    params[k] = [ float(v) for v in line ]

            for k in [ "DimSize" ]:
                if k in line:
                    line = line.split('=')
                    line = line[1].strip().split(' ')
                    params[k] = tuple(int(v) for v in line)

            for k in [ "ElementDataFile" ]:
                if k in line:
                    line = line.split('=')
                    line = line[1].strip()
                    params[k] = line

            for k in [ "ElementType" ]:
                if k in line:
                    line = line.split('=')
                    line = line[1].strip()
                    params[k] = self.type_map.get(line)

        if "ElementDataFile" in params:
            dirname = os.path.dirname(self.mhd_file)
            params["ElementDataFile"] = os.path.join(dirname, params["ElementDataFile"])

        self._metadata = params
        return self._metadata

    def memmap_image(self):
        return np.memmap(self.metadata['ElementDataFile'],
                         dtype=np.dtype(self.metadata["ElementType"]).str,
                         shape=self.metadata["DimSize"],
                         mode='r',
                         order='F')

    @property
    def image(self):
        if self._image is not None:
            return self._image 

        with open(self.metadata['ElementDataFile'], 'r') as f:    
            v = np.fromfile(f, dtype=self.metadata["ElementType"])
            
        self._image = np.reshape(v, self.metadata["DimSize"])

        return self._image

    def sample(self, coord):
        index = np.ravel_multi_index(coord, self.metadata["DimSize"])
        type_size = np.dtype(self.metadata["ElementType"]).itemsize

        with open(self.metadata['ElementDataFile'], 'rb') as f:
            f.seek(type_size*index) 
            v = np.fromfile(f, 
                            dtype=self.metadata["ElementType"], 
                            count=1)
            
        return v

    def slice3d(self, axis, index):
        return slice3d(self.image, axis, index)

def slice3d(image, axis, index):
    if axis == 0:
        return image[index,:,:]
    if axis == 1:
        return image[:,index,:]
    if axis == 2:
        return image[:,:,index]


if __name__ == "__main__":
    fname = "/external/ctyconn/prod17/image_series_609475867/grid/red_22.4.mhd"
    m = MetaImage(fname)
    print(m.metadata)
    print(m.sample([300,200,100]))
