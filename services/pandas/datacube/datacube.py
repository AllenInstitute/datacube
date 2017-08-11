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
from functools import reduce

from six import iteritems
from builtins import int


# https://github.com/dask/dask/issues/732
import dask.array as da

def da_einsum(*args, **kwargs):
    args = list(args)
    out_inds = args.pop()
    in_inds = args[1::2]
    arrays = args[0::2]
    dtype = kwargs.get('dtype')
    einsum_dtype = dtype
    if dtype is None:
        dtype = np.result_type(*[a.dtype for a in arrays])
    casting = kwargs.get('casting')
    if casting is None:
        casting = 'safe'
    
    full_inds = list(set().union(*in_inds))
    contract_inds = tuple(set(full_inds)-set(out_inds))
    
    def kernel(*operands, **kwargs):
        kernel_dtype = kwargs.get('kernel_dtype')
        casting = kwargs.get('casting')
        chunk = np.einsum(dtype=kernel_dtype, casting=casting, *([arg for arg_pair in zip(operands, in_inds) for arg in arg_pair]+[out_inds]))
        chunk = np.array(chunk)
        chunk.shape = tuple([1 if ind not in out_inds else chunk.shape[out_inds.index(ind)] for ind in full_inds])
        return chunk
    
    adjust_chunks = {ind: 1 for ind in contract_inds}
    result = da.atop(kernel, full_inds, *args, dtype=dtype, kernel_dtype=einsum_dtype, casting=casting, adjust_chunks=adjust_chunks)
    if contract_inds:
        result = da.sum(result, axis=contract_inds)
    return result

# monkey-patch the method into the module for now
setattr(da, 'einsum', da_einsum)


class Datacube:


    def test(self):
        return 1


    def __init__(self, name, nc_file, chunks=None, max_cacheable_bytes=10*1024*1024):
        self.name = name
        self.max_cacheable_bytes = max_cacheable_bytes
        if nc_file: self.load(nc_file, chunks)
        #todo: would be nice to find a way to swap these out,
        # and also to be able to run without redis (numpy-only)
        #if reactor.running:
        #    self.redis_client = txredisapi.Connection('localhost', 6379)
        #else:
        self.redis_client = redis.StrictRedis('localhost', 6379)


    def load(self, nc_file, chunks=None):
        #todo: rename df
        #todo: argsorts need to be cached to a file (?)
        self.df = xr.open_dataset(nc_file, chunks=chunks)
        self.argsorts = {}
        for field in self.df.keys():
            if self.df[field].size/8. < self.max_cacheable_bytes:
                self.argsorts[field] = np.argsort(self.df[field].values, axis=None)
        #todo: would be nice to cache these instead of computing on every startup
        self.mins = {}
        self.maxes = {}
        self.means = {}
        self.stds = {}
        for field in self.df.keys():
            self.mins[field] = self.df[field].min().values
            self.maxes[field] = self.df[field].max().values
            if not self.df[field].dtype.kind in 'OSU':
                self.means[field] = self.df[field].mean().values
                self.stds[field] = self.df[field].std().values


    def _validate_select(self, select):
        assert(all(f in list(self.df.dims.keys()) for f in select.keys()))

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
            elif isinstance(selector, int):
                select[axis] = selector
            elif selector is None:
                pass
            else:
                raise ValueError('Unexpected selector of type ' + str(type(selector)))
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


    def get_data(self, subscripts, fields, dim_order=None):
        res = self.df[subscripts][fields]
        if dim_order:
            res = res.transpose(*dim_order)
        return res


    def info(self):
        dims = {name: size for name, size in self.df.dims.items()}
        variables = {name: {'type': da.dtype.name, 'dims': da.dims} for name, da in self.df.variables.items()}
        for name, da in self.df.variables.items():
            variables[name]['attrs'] = dict(da.attrs.items())
        attrs = dict(self.df.attrs.items())
        return {'dims': dims, 'vars': variables, 'attrs': attrs}


    def raw(self, select, fields, dim_order=None):
        assert(all(f in list(self.df.keys()) for f in fields))
        self._validate_select(select)
        subscripts = self._get_subscripts_from_select(select)
        return self.get_data(subscripts, fields, dim_order)


    def image(self, select, field, dim_order=None, image_format='jpeg'):
        if dim_order:
            if 'RGBA' in self.df[field].dims:
                if 'RGBA' in dim_order:
                    dim_order.remove('RGBA')
                dim_order.append('RGBA')
        data = self.raw(select, [field], dim_order)
        dims = list(data[field].dims)
        if 'RGBA' in dims:
            dims.remove('RGBA')
            dims.append('RGBA')
            data = data.transpose(*dims)
        data = data[field].values
        data = np.squeeze(data)

        if data.ndim != 2 and not (data.ndim == 3 and data.shape[2] == 4):
            raise RuntimeError('Non 2-d region selected when requesting image')

        #todo: probably ought to make normalization configurable in the request in some manner
        if data.ndim == 2 and data.dtype != np.uint8:
            data = ((data / self.maxes[field]) * 255.0).astype(np.uint8)

        image = Image.fromarray(data)
        buf = BytesIO()
        if image_format.lower() == 'jpeg':
            image = image.convert('RGB')
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
        filtered_total = inds.size
        inds = inds[start:stop]
        if fields == "indexes_only":
            return inds, filtered_total
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
        sort_cache_key = json.dumps([self.name, 'sort', dim, sort, ascending])
        cached = self.redis_client.get(sort_cache_key)
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
            self.redis_client.setnx(sort_cache_key, pickle.dumps(res))
            return res
        else:
            return pickle.loads(cached)


    def _query(self, filters):
        df = self.df
        if not filters:
            return (self.df, [])
        else:
            df = self.df
            res = xr.Dataset()

            def _filter(f):
                op = f['op']
                field = f['field']
                value = f['value']

                def _filter_key(f):
                    return json.dumps([self.name, 'filter', f['op'], f['field'], f['value']])

                res = {'inds': {}, 'masks': []}
                if field not in self.argsorts:
                    if op == '=' or op == 'is':
                        mask = (df[field] == value)
                    elif op == '<':
                        mask = (df[field] < value)
                    elif op == '>':
                        mask = (df[field] > value)
                    elif op == '<=':
                        mask = (df[field] <= value)
                    elif op == '>=':
                        mask = (df[field] >= value)
                    res['masks'].append(mask)
                else:
                    filter_key = _filter_key(f)
                    cached = self.redis_client.get(filter_key)
                    if not cached:
                        start = 0
                        stop = df[field].size
                        if op == '=' or op == 'is':
                            start = np.searchsorted(df[field], value, side='left', sorter=self.argsorts[field])
                            stop = np.searchsorted(df[field], value, side='right', sorter=self.argsorts[field])
                        elif op == '<':
                            stop = np.searchsorted(df[field], value, side='left', sorter=self.argsorts[field])
                        elif op == '>':
                            start = np.searchsorted(df[field], value, side='right', sorter=self.argsorts[field])
                        elif op == '<=':
                            stop = np.searchsorted(df[field], value, side='right', sorter=self.argsorts[field])
                        elif op == '>=':
                            start = np.searchsorted(df[field], value, side='left', sorter=self.argsorts[field])
                        inds = self.argsorts[field][start:stop]
                        self.redis_client.setnx(filter_key, pickle.dumps(inds))
                    else:
                        inds = pickle.loads(cached)
                    
                    if 1==self.df[field].ndim:
                        res['inds'][dim] = xr.DataArray(inds, dims=df[field].dims)
                    else:
                        unravel_inds = np.unravel_index(inds, df[field].shape)
                        for i, dim in enumerate(df[field].dims):
                            res['inds'][dim] = xr.DataArray(np.unique(unravel_inds[i]), dims=dim)
                        mask = xr.zeros_like(df[field], dtype=np.bool)
                        mask.values.flat[inds] = True
                        res['masks'].append(mask)

            def _and(m1, m2):
                res = {}
                for mask in m1['masks']+m2['masks']:
                    d = {}
                    d.setdefault(mask.dims, []).append(mask)
                    for k, v in d:
                        res.setdefault('masks', []).append(reduce(xr.ufuncs.logical_and, v))
                res['inds'] = m1['inds']
                for dim, inds in m2['inds']:
                    if dim in res['inds']:
                        res['inds'][dim] = np.intersect1d(res['inds'][dim], m2['inds'][dim], assume_unique=True)
                return res

            def _or(m1, m2):
                res['masks'] = [xr.ufuncs.logical_or(reduce(xr.ufuncs.logical_and, m1['masks']), reduce(xr.ufuncs.logical_and, m2['masks']))]
                res['inds'] = m1['inds']
                for dim, inds in m2['inds']:
                    if dim in res['inds']:
                        res['inds'][dim] = np.union1d(res['inds'][dim], m2['inds'][dim])
                return res

            # --- todo ---

            #todo: support {'and': [...]} and {'or': [...]}

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


            and_keys = []
            for f in expanded_filters:
                if isinstance(f, list):
                    or_keys = []
                    for o in f:
                        key = _cache_filter(o)
                        or_keys.append(key)
                    or_cache_key = json.dumps([self.name, 'filter_or', or_keys])
                    lua = '''
                        local exists=redis.call('EXISTS', KEYS[1])
                        if 0 == exists then
                            redis.call('BITOP', 'OR', unpack(KEYS))
                        end
                        return redis.call('GET', KEYS[1])
                    '''
                    eval_keys = (or_cache_key,)+tuple(or_keys)
                    self.redis_client.eval(lua, len(eval_keys), *eval_keys)
                    and_keys.append(or_cache_key)
                else:
                    key = _cache_filter(f)
                    and_keys.append(key)

            and_cache_key = json.dumps([self.name, 'filter_and', and_keys])
            lua = '''
                local exists=redis.call('EXISTS', KEYS[1])
                if 0 == exists then
                    redis.call('BITOP', 'AND', unpack(KEYS))
                end
                return redis.call('GET', KEYS[1])
            '''
            eval_keys = (and_cache_key,)+tuple(and_keys)
            res = self.redis_client.eval(lua, len(eval_keys), *eval_keys)
            res = _unpack(res, df.dims[dim])
            res = np.flatnonzero(res)
            return res
