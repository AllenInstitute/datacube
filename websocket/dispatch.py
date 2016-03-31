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
                            'shape': self.shape,
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
                    if isinstance(selector[0], bool):
                        subscripts[axis] = np.array(selector, dtype=np.bool)
                    elif isinstance(selector[0], int):
                        subscripts[axis] = np.array(selector, dtype=np.int)
                elif isinstance(selector, dict):
                    subscripts[axis] = slice(selector.get('start'), selector.get('stop'), selector.get('step'))
        return subscripts

    # Get the shape of the datacube.
    def shape(self, request):
        if request['binary']:
            return struct.pack('!%sI' % self.datacube.ndim, *self.datacube.shape)
        else:
            return json.dumps(list(self.datacube.shape))

    # Return values from a section of the datacube.
    def cube(self, request):
        subscripts = self._get_subscripts_from_request(request)
        for axis, subs in enumerate(subscripts):
            if isinstance(subs, np.ndarray) and subs.dtype == np.bool:
                subscripts[axis] = subs.nonzero()[0]
            elif isinstance(subs, slice):
                subscripts[axis] = np.array(range(*subs.indices(self.datacube.shape[axis])), dtype=np.int)
        subscripts = np.ix_(*subscripts)
        shape = self.datacube.data[subscripts].shape
        if request['binary']:
            return struct.pack('!I', shape[0]) + struct.pack('!I', shape[1]) + self.zscore[subscripts].tobytes()
        else:
            return json.dumps({'shape': shape, 'data': [None if np.isnan(x) else float(x) for x in self.zscore[subscripts].flat]})

    # Return the correlation calculation based on a seed row (JSON)
    def corr(self, request):
        query = self._get_subscripts_from_request(request)
        for axis, subs in enumerate(query):
            if subs == slice(None,None,None):
                query[axis] = np.ones(1)
            elif isinstance(subs, slice):
                query[axis] = np.array(range(*subs.indices(self.datacube.shape[axis])), dtype=np.int)

        r=self.datacube.get_correlated(request['seed'], 0, query)
        sort_idxs = np.argsort(-r)
        return json.dumps({'indexes': sort_idxs.tolist(), 'correlations': [None if np.isnan(x) else x for x in r[sort_idxs]]})

    # Return metadata (JSON)
    def meta(self, request):
        r=self.database.get_meta(self.datacube)
        return r
