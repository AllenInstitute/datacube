import json
import struct
import numpy as np
import scipy as sp
from scipy import stats
import jsonschema
import copy

from datacube import Datacube
from database import Database

FUNCTIONS = ['raw', 'log2_1p', 'standard', 'info', 'corr', 'meta']

class FunctionNameError(RuntimeError):
    pass

class RequestValidator:
    def __init__(self, file, shape):
        self.schema = json.loads(open(file).read())
        # dynamically modify schema a bit to take into account datacube ndim and shape
        self.schema['definitions']['select']['items'] = \
            [copy.deepcopy(self.schema['definitions']['selector']) for axis in range(0, len(shape))]
        for axis in range(0, len(shape)):
            # NOTE: this assumes boolean array selector is at index 2 in selector schema in request_schema.json
            self.schema['definitions']['select']['items'][axis]['oneOf'][2]['minItems'] = shape[axis]
            self.schema['definitions']['select']['items'][axis]['oneOf'][2]['maxItems'] = shape[axis]
        self.schema['definitions']['select']['minItems'] = len(shape)
        self.schema['definitions']['select']['maxItems'] = len(shape)
        self.schema['definitions']['axis']['maximum'] = len(shape)

    # validate request against JSON schema, and find the most sensible error to respond with
    def validate(self, request):
        # make sure function exists
        if 'call' not in request or request['call'] not in FUNCTIONS:
            raise FunctionNameError('The specified function \'%s\' does not exist or is spelled incorrectly.' % request['call'])

        def filter_call_pattern(error):
            validator = error.validator
            if list(error.path) == ['call']:
                if error.instance in FUNCTIONS:
                    call_precedence = 1
                else:
                    call_precedence = -1
            else:
                call_precedence = 0
            return call_precedence, validator not in ['minItems', 'maxItems'], jsonschema.exceptions.relevance(error)

        err = jsonschema.exceptions.best_match( \
            jsonschema.Draft4Validator(self.schema).iter_errors(request), key=filter_call_pattern)
        if err: raise err

# This handles the dispatching of function calls on a datacube.
class Dispatch:
    def __init__(self, datacube, database):
        assert(isinstance(datacube, Datacube))
        assert(isinstance(database, Database))
        self.datacube = datacube
        self.database = database
        self.request_validator = RequestValidator('request_schema.json', self.datacube.shape)

    def call(self, request):
        self.request_validator.validate(request)
        # dispatch to function
        assert(request['call'] in FUNCTIONS)
        return getattr(self, request['call'])(request)

    def _parse_select_from_request(self, request):
        select = [slice(None,None,None)]*self.datacube.ndim
        if 'select' in request:
            assert(isinstance(request['select'], list))
            assert(len(request['select']) == self.datacube.ndim)
            for axis, selector in enumerate(request['select']):
                if isinstance(selector, list):
                    if len(selector) == 0:
                        select[axis] = np.array([], dtype=np.int)
                    elif isinstance(selector[0], bool):
                        select[axis] = np.array(selector, dtype=np.bool)
                    elif isinstance(selector[0], int):
                        select[axis] = np.array(selector, dtype=np.int)
                elif isinstance(selector, dict):
                    select[axis] = slice(selector.get('start'), selector.get('stop'), selector.get('step'))
        return select

    def _get_subscripts_from_request(self, request):
        subscripts = self._parse_select_from_request(request)
        for axis, subs in enumerate(subscripts):
            if isinstance(subs, np.ndarray) and subs.dtype == np.bool:
                subscripts[axis] = subs.nonzero()[0]
            elif isinstance(subs, slice):
                subscripts[axis] = np.array(range(*subs.indices(self.datacube.shape[axis])), dtype=np.int)
        subscripts = np.ix_(*subscripts)
        return subscripts

    # Get the shape of the datacube.
    def info(self, request):
        return {
            'ndim': self.datacube.ndim,
            'shape': list(self.datacube.shape),
            'dtype': self.datacube.dtype.name
        }

    def raw(self, request):
        subscripts = self._get_subscripts_from_request(request)
        data = self.datacube.get_data(subscripts)
        return self._format_data_response(data, request['binary'])

    def log2_1p(self, request):
        subscripts = self._get_subscripts_from_request(request)
        data = self.datacube.get_log2_1p(subscripts)
        return self._format_data_response(data, request['binary'])

    def standard(self, request):
        subscripts = self._get_subscripts_from_request(request)
        data = self.datacube.get_standard(subscripts, request['axis'])
        return self._format_data_response(data, request['binary'])

    # Return values from a section of the datacube.
    def _format_data_response(self, data, binary):
        shape = data.shape
        if binary:
            big_endian = data.byteswap() # copy array into network-order (big-endian)
            return struct.pack('!I', shape[0]) + struct.pack('!I', shape[1]) + big_endian.tobytes()
        else:
            return {'shape': shape, 'data': [None if np.isnan(x) else float(x) for x in data.flat]}

    # Return the correlation calculation based on a seed row (JSON)
    def corr(self, request):
        query = self._parse_select_from_request(request)
        for axis, subs in enumerate(query):
            if subs == slice(None,None,None):
                query[axis] = np.ones(1)
            elif isinstance(subs, slice):
                query[axis] = np.array(range(*subs.indices(self.datacube.shape[axis])), dtype=np.int)

        r=self.datacube.get_correlated(request['seed'], request['axis'], query)
        sort_idxs = np.argsort(-r)
        return {'indexes': sort_idxs.tolist(), 'correlations': [None if np.isnan(x) else x for x in r[sort_idxs]]}

    # Return metadata (JSON)
    def meta(self, request):
        r=self.database.get_meta(self.datacube)
        return r
