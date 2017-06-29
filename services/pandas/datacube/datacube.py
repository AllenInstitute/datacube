import numpy as np
from scipy.stats import rankdata
import json
import pickle
import redis
#import txredisapi
#from twisted.internet import reactor
import xarray as xr
import dask
from PIL import Image
from io import BytesIO
import base64

from six import iteritems
from builtins import int


class Datacube:


    def __init__(self, nc_file=None, chunks=None):
        if nc_file: self.load(nc_file, chunks)
        #todo: would be nice to find a way to swap these out,
        # and also to be able to run without redis (numpy-only)
        #if reactor.running:
        #    self.redis_client = txredisapi.Connection('localhost', 6379)
        #else:
        self.redis_client = redis.StrictRedis('localhost', 6379)


    def load(self, nc_file, chunks=None):
        #todo: rename df
        #todo: argsorts need to be cached to a file
        self.df = xr.open_dataset(nc_file, chunks=chunks)
        self.argsorts = {}
        for field in self.df.keys():
            self.argsorts[field] = np.argsort(self.df[field].values)


    def _validate_select(self, select):
        assert(all(f in list(self.df.keys()) for f in select.keys()))

        for axis, selector in iteritems(select):
            if isinstance(selector, list):
                if any(type(x) != type(selector[0]) for x in selector):
                    raise RuntimeError('All elements of selector for axis {0} do not have the same type.'.format(axis))
                if not isinstance(selector[0], int) and not isinstance(selector[0], bool):
                    raise RuntimeError('Elements of list selector for axis {0} must be of type int or bool.'.format(axis))
                if isinstance(selector[0], bool) and len(selector) != self.df.dims[axis]:
                    raise RuntimeError('Boolean list selector for axis {0} must have length {1} to match the size of the datacube.'.format(axis, self.df.dims[axis]))
            elif isinstance(selector, dict):
                keys = ['start', 'stop', 'step']
                for key in keys:
                    if key in selector and selector[key] is not None and not isinstance(selector[key], int):
                        raise RuntimeError('Slice selector for axis {0} must have ''{1}'' of type int.'.format(axis, key))


    # convert dict selectors into slice objects,
    # index and bool ndarrays
    def _parse_select(self, select_element):
        select = {axis: slice(None,None,None) for axis in self.df.dims.keys()}
        for axis, selector in iteritems(select_element):
            if isinstance(selector, list):
                if len(selector) == 0:
                    select[axis] = np.array([], dtype=np.int)
                elif all(type(x) == bool for x in selector):
                    select[axis] = np.array(selector, dtype=np.bool)
                elif all(isinstance(x, int) for x in selector):
                    select[axis] = np.array(selector, dtype=np.int)
                else:
                    raise RuntimeError('All elements of selector for axis {0} do not have the same type.'.format(axis))
            elif isinstance(selector, dict):
                select[axis] = slice(selector.get('start'), selector.get('stop'), selector.get('step'))
        return select


    # parse and convert all request selectors into index arrays
    def _get_subscripts_from_select(self, select_element):
        subscripts = self._parse_select(select_element)
        for axis, subs in iteritems(subscripts):
            if isinstance(subs, np.ndarray) and subs.dtype == np.bool:
                subscripts[axis] = subs.nonzero()[0]
            elif isinstance(subs, slice):
                subscripts[axis] = np.array(range(*subs.indices(self.df.dims[axis])), dtype=np.int)
        #subscripts = np.ix_(*subscripts)
        return subscripts


    def get_data(self, subscripts, fields):
        return self.df[subscripts][fields]


    def raw(self, select, fields):
        assert(all(f in list(self.df.keys()) for f in fields))
        self._validate_select(select)
        subscripts = self._get_subscripts_from_select(select)
        return self.get_data(subscripts, fields)


    #todo: add dims argument so e.g. ['y', 'x'] would give transposed image of ['x', 'y']
    def image(self, select, field, image_format='jpeg'):
        data = self.raw(select, [field])
        dims = list(data.dims.keys())
        if 'RGBA' in dims:
            dims.remove('RGBA')
            dims.append('RGBA')
            data = data.transpose(*dims)
        data = data[field].values
        data = np.squeeze(data)

        if data.ndim != 2 and not (data.ndim == 3 and data.shape[2] == 4):
            raise RuntimeError('Non 2-d region selected when requesting image')

        if data.ndim == 2 and data.dtype != np.uint8:
            #todo: precompute min, max for each numeric field on the datacube
            data = ((data / np.max(data)) * 255.0).astype(np.uint8)

        image = Image.fromarray(data)
        buf = BytesIO()
        if image_format.lower() == 'jpeg':
            image.save(buf, format='JPEG', quality=40)
        elif image_format.lower() == 'png':
            image.save(buf, format='PNG')
        else:
            raise RuntimeError('Invalid or unsupported image format')
        #return {'data': bytes(buf.getvalue())}
        return {'data': 'data:image/' + image_format.lower() + ';base64,' + base64.b64encode(buf.getvalue()).decode()}


    def select(self,
               dim=None,
               filters=None,
               sort=None,
               ascending=None,
               start=0,
               stop=None,
               indexes=None,
               fields=None,
               options={}):
        r = self.df
        if dim is None:
            dim = list(r.dims)[0]
        redis_client = self.redis_client
        if not filters and not fields == "indexes_only":
            if indexes:
                num_results = np.array(indexes)[start:stop].size
            else:
                num_results = r[dim][start:stop].size
            if options.get('max_records') and num_results > options.get('max_records'):
                raise ValueError('Requested would return ' + str(num_results) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
        inds = np.array(range(r.dims[dim]), dtype=np.int)
        if filters is not None:
            inds = self._query(dim, filters)
        if indexes is not None:
            indexes = [x for x in indexes if x != None]
            if inds is not None:
                inds = inds[np.in1d(inds, indexes)]
            else:
                inds = np.array(indexes, dtype=np.int)
        if sort and r[dim].size > 0:
            sorted_inds = self._sort(dim, sort, ascending)
            if inds is not None:
                inds = sorted_inds[np.in1d(sorted_inds, inds)]
            else:
                inds = sorted_inds
        inds = inds[start:stop]
        if fields == "indexes_only":
            return inds
        else:
            if options.get('max_records') and inds.size > options.get('max_records'):
                raise ValueError('Requested would return ' + str(inds.size) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
            if fields and type(fields) is list:
                for field in fields:
                    if field not in r.keys():
                        raise ValueError('Requested field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.keys()))
                res = r[{dim: inds}][fields]
            else:
                res = r[{dim: inds}]
            return res


    def _sort(self, dim, sort, ascending):
        df = self.df
        redis_client = self.redis_client
        sort_cache_key = json.dumps(['sort', dim, sort, ascending])
        cached = redis_client.get(sort_cache_key)
        if not cached:
            if not ascending:
                ascending = [True] * len(sort)
            for field in sort:
                if field not in df.keys():
                    raise ValueError('Requested sort field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(df.keys()))
            s = df[sort]
            ranks = np.zeros((s.dims[dim], len(s.keys())), dtype=np.int)
            for idx, (field, asc) in enumerate(zip(sort[::-1], ascending[::-1])):
                if s[field].dtype.name.startswith('string') or s[field].dtype.name.startswith('unicode'):
                    blank_count = np.count_nonzero(s[field] == '')
                else:
                    blank_count = np.count_nonzero(np.isnan(s[field]))
                ranks[:,idx] = rankdata(s[field], method='dense')
                maxrank = np.max(ranks[:,idx])
                if not asc:
                    ranks[:,idx] = maxrank - (ranks[:,idx] - 1) - blank_count
                    ranks[:,idx][ranks[:,idx]<=0] = maxrank - blank_count + 1
                else:
                    ranks[:,idx] = np.clip(ranks[:,idx], 0, maxrank - blank_count + 1)
            res = np.lexsort(tuple(ranks[:,idx] for idx in range(ranks.shape[1])))
            redis_client.setnx(sort_cache_key, pickle.dumps(res))
            return res
        else:
            return pickle.loads(cached)


    def _query(self, dim, filters):
        df = self.df
        redis_client = self.redis_client
        column_argsort = self.argsorts
        if not filters:
            return np.array(range(df.dims[dim]), dtype=np.int)
        else:
            # todo: add name to datacube object and include in cache keys

            def _do_search(f):
                op = f['op']
                field = f['field']
                value = f['value']

                res = None
                if isinstance(df[field].data, dask.array.core.Array):
                    if op == '=' or op == 'is':
                        res = (df[field] == value)
                    elif op == '<':
                        res = (df[field] < value)
                    elif op == '>':
                        res = (df[field] > value)
                    elif op == '<=':
                        res = (df[field] <= value)
                    elif op == '>=':
                        res = (df[field] >= value)
                else:
                    start = 0
                    stop = df[field].size
                    res = np.zeros(df[field].shape, dtype=np.bool)
                    if op == '=' or op == 'is':
                        start = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                        stop = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                    elif op == '<':
                        stop = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                    elif op == '>':
                        start = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                    elif op == '<=':
                        stop = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                    elif op == '>=':
                        start = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                    res[column_argsort[field][start:stop]] = True
                return res

            expanded_filters = []
            for f in filters:
                op, field, value = (f['op'], f['field'], f['value'])
                if op == 'between':
                    expanded_filters.append({'op': '>=', 'field': field, 'value': value[0]})
                    expanded_filters.append({'op': '<=', 'field': field, 'value': value[1]})
                elif op == 'in':
                    or_clause = []
                    for v in value:
                        or_clause.append({'op': '=', 'field': field, 'value': v})
                    expanded_filters.append(or_clause)
                else:
                    expanded_filters.append(f)

            def _filter_cache_key(f):
                return json.dumps(['filter', dim, f['op'], f['field'], f['value']])

            def _pack(bool_array):
                return str(np.packbits(bool_array).tobytes())

            def _unpack(bitstring, sz):
                res = np.fromstring(bitstring, dtype=np.uint8)
                res = res[:sz]
                return res

            def _cache_filter(f):
                filter_cache_key = _filter_cache_key(f)
                cached = redis_client.exists(filter_cache_key)
                if not cached:
                    filtered = _do_search(f)
                    redis_client.setnx(filter_cache_key, _pack(filtered))
                return filter_cache_key

            and_keys = []
            for f in expanded_filters:
                if isinstance(f, list):
                    or_keys = []
                    for o in f:
                        key = _cache_filter(o)
                        or_keys.append(key)
                    or_cache_key = json.dumps(['filter_or', dim, or_keys])
                    lua = '''
                        local exists=redis.call('EXISTS', KEYS[1])
                        if 0 == exists then
                            redis.call('BITOP', 'OR', unpack(KEYS))
                        end
                        return redis.call('GET', KEYS[1])
                    '''
                    eval_keys = (or_cache_key,)+tuple(or_keys)
                    redis_client.eval(lua, len(eval_keys), *eval_keys)
                    and_keys.append(or_cache_key)
                else:
                    key = _cache_filter(f)
                    and_keys.append(key)

            and_cache_key = json.dumps(['filter_and', dim, and_keys])
            lua = '''
                local exists=redis.call('EXISTS', KEYS[1])
                if 0 == exists then
                    redis.call('BITOP', 'AND', unpack(KEYS))
                end
                return redis.call('GET', KEYS[1])
            '''
            eval_keys = (and_cache_key,)+tuple(and_keys)
            res = redis_client.eval(lua, len(eval_keys), *eval_keys)
            res = _unpack(res, df.dims[dim])
            res = np.flatnonzero(res)
            return res
