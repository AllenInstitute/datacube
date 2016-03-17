import json
import struct
import numpy as np
import scipy as sp
from scipy import stats

from datacube import Datacube
from database import Database

# This handles the dispatching of function calls on a datacube.  The functions are stored in the functions table.
class Dispatch:
    def __init__(self, datacube, database):
        assert(isinstance(datacube, Datacube))
        assert(isinstance(database, Database))
        self.datacube = datacube
        self.database = database

        # This is a dispatch table of functions that respond to the call request
        self.functions = {  'cube': self.cube,
                            'corr': self.corr,
                            'meta': self.meta
        }

        self.zscore = sp.stats.mstats.zscore(self.datacube.data, axis=1)

    def _get_subscripts_from_request(self, request):
        subscripts = [slice(None,None,None)]*self.datacube.ndim
        if 'select' in request:
            assert(isinstance(request['select'], list))
            assert(len(request['select']) == self.datacube.ndim)
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
        return subscripts

    # Return values from a section of the datacube (binary).
    def cube(self, request):
        subscripts = self._get_subscripts_from_request(request)
        shape = self.datacube.data[subscripts].shape
        return struct.pack('>I', shape[1],) + struct.pack('>I', shape[0]) + self.zscore[subscripts].tobytes()

    # Return the correlation calculation based on a seed row (JSON)
    def corr(self, request):
        query = self._get_subscripts_from_request(request)
        for axis, subs in enumerate(query):
            if subs == slice(None,None,None):
                query[axis] = np.ones(1)
            elif isinstance(subs, slice):
                query[axis] = np.array(range(*subs.indices(self.datacube.shape[axis])), dtype=np.int)

        r=self.datacube.get_correlated(request['seed'], 0, query)
        return json.dumps(np.argsort(-r).tolist())

    # Return metadata (JSON)
    def meta(self, request):
        r=self.database.get_meta(self.datacube)
        return r
