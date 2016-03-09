import json
import struct
import numpy as np
import scipy as sp
from scipy import stats

from datacube import Datacube

# This handles the dispatching of function calls on a datacube.  The functions are stored in the functions table.
# The return protocol is stored in the isBinary table.  TODO: do we need this?  It would be nice to have client specify, but autobahn.js does not support this. Maybe everything should be binary.
class Dispatch:
    def __init__(self, datacube):
        assert(isinstance(datacube, Datacube))
        self.datacube = datacube

        # This is a dispatch table of functions that respond to the call request
        self.functions = {  'cube': self.cube,
                            'corr': self.corr
        }

        self.zscore = sp.stats.mstats.zscore(self.datacube.data, axis=1)

    # Return values from a section of the datacube (binary).
    def cube(self, request):
        subscripts = [slice(None,None,None)]*self.datacube.data.ndim
        if 'select' in request:
            assert(isinstance(request['select'], list))
            assert(len(request['select']) == self.datacube.data.ndim)
            for axis, selector in enumerate(request['select']):
                if isinstance(selector, list):
                    if isinstance(selector[0], int):
                        subscripts[axis] = np.array(selector, dtype=np.int)
                    elif isintance(selector[0], bool):
                        subscripts[axis] = np.array(selector, dtype=np.bool)
                elif isinstance(selector, dict):
                    if 'step' in selector:
                        subscripts[axis] = slice(selector['start'], selector['stop'], selector['step'])
                    else:
                        subscripts[axis] = slice(selector['start'], selector['stop'], None)

        shape = self.datacube.data[subscripts].shape
        return struct.pack('>I', shape[1],) + struct.pack('>I', shape[0]) + self.zscore[subscripts].tobytes()

    # Return the correlation calculation based on a seed row (JSON)
    def corr(self, request):
        r=self.datacube.get_correlated(request['seed'], 0)
        return json.dumps(np.argsort(-r).tolist())
